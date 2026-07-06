import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import joblib

class TradeLearner:
    def __init__(self, history_file="trade_history.csv", model_file="ai/trade_learner_model.joblib"):
        self.history_file = history_file
        self.model_file = model_file
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_file):
            # FIX 4 (partial): Catch and log the specific exception instead of
            # silently swallowing all errors.
            try:
                self.model = joblib.load(self.model_file)
            except Exception as e:
                print(f"[LEARNER] Warning: could not load model from {self.model_file}: {e}")
                self.model = None

    def train(self):
        """
        Reads trade_history.csv, labels outcomes by checking price movement,
        and trains a simple model.
        """
        if not os.path.exists(self.history_file):
            return "No history file found."

        df = pd.read_csv(self.history_file)

        # trade_history.csv also contains "NO TRADE" rejection rows written by
        # log_evaluation() — those are gate decisions, not trades, and labeling
        # them by subsequent price movement would teach the model nonsense.
        if 'Signal' in df.columns:
            df = df[~df['Signal'].astype(str).str.strip().str.upper().eq('NO TRADE')]
            df = df.reset_index(drop=True)

        if len(df) < 5:
            return "Insufficient data for training."

        # 1. Prepare Features
        # FIX 1: Normalise conviction to millions at training time so that the
        # unit matches what `get_confidence_multiplier` receives from `insider_score`.
        # Previously, training used raw dollars (e.g. 1_500_000) while inference
        # multiplied by 1_000_000, creating a scale mismatch whenever insider_score
        # was not already in the millions range.
        df['conviction'] = df['Insider_Value'] / 1_000_000  # Now in $M, e.g. 1.5
        df['is_ceo'] = df['Reasons'].apply(lambda x: 1 if "CEO" in str(x) else 0)

        # 2. Prepare Targets (Actual Price Outcome)
        # This is SLOW because it calls yfinance for each row.
        print(f"[LEARNER] Fetching historical returns for {len(df)} past evaluations (this may take a minute)...")
        outcomes = []
        for index, row in df.iterrows():
            if index % 10 == 0:
                print(f"[LEARNER] Progress: {index}/{len(df)} processed...")
            ticker = row['Ticker']
            eval_date_str = row['Date'].split(' ')[0]  # YYYY-MM-DD
            eval_date = datetime.strptime(eval_date_str, "%Y-%m-%d")

            # FIX 4 (partial): Log per-row failures instead of swallowing them.
            try:
                # Fetch ~21 calendar days (≈ 15 trading days) to get a solid
                # 10-trading-day window with buffer for weekends/holidays.
                end_date = eval_date + timedelta(days=21)
                hist = yf.download(
                    ticker,
                    start=eval_date_str,
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                )

                if len(hist) >= 2:
                    start_price = hist['Close'].iloc[0]
                    # ~10 trading days ≈ 14 calendar days
                    look_forward_idx = min(len(hist) - 1, 10)
                    end_price = hist['Close'].iloc[look_forward_idx]

                    return_pct = (end_price / start_price) - 1
                    # Label as 1 if return > 2 %, else 0
                    outcomes.append(1 if return_pct > 0.02 else 0)
                else:
                    outcomes.append(np.nan)
            except Exception as e:
                print(f"[LEARNER] Warning: could not fetch data for {ticker} ({eval_date_str}): {e}")
                outcomes.append(np.nan)

        df['outcome'] = outcomes
        train_data = df.dropna(subset=['outcome'])

        if len(train_data) < 5:
            return "Insufficient labeled outcomes for training."

        X = train_data[['is_ceo', 'conviction']]
        # FIX 3: After dropna the column still has float dtype due to np.nan
        # contamination. Cast explicitly so RandomForestClassifier gets integer labels.
        y = train_data['outcome'].astype(int)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        # FIX 2: Create the output directory before trying to save the model.
        # Without this, joblib.dump raises FileNotFoundError when `ai/` doesn't exist.
        model_dir = os.path.dirname(self.model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.model, self.model_file)
        print(f"[LEARNER] AI Model trained and saved to {self.model_file}")
        return "Success"

    def get_confidence_multiplier(self, ticker: str, scoring_result: dict) -> float:
        """
        Predicts how likely this trade is to be a winner based on past history.
        Returns a multiplier for the Kelly Fraction (0.5 to 1.5).

        `scoring_result['insider_score']` must be in the same unit used at
        training time: **millions of dollars** (e.g. 1.5 means $1.5 M).
        """
        if self.model is None:
            return 1.0  # Neutral multiplier if no model

        try:
            is_ceo = (
                1
                if scoring_result.get('reasons')
                and any("CEO" in r for r in scoring_result['reasons'])
                else 0
            )

            # FIX 1: `conviction` is already in $M here (the caller supplies
            # `insider_score` in millions). Training now also uses millions, so
            # the scale is consistent — no multiplication needed.
            conviction = float(scoring_result.get('insider_score', 0))

            X = np.array([[is_ceo, conviction]])
            prob = self.model.predict_proba(X)[0][1]  # P(outcome == 1)

            # Map probability [0, 1] → multiplier [0.5, 1.5]
            multiplier = 0.5 + prob
            return round(multiplier, 2)
        except Exception as e:
            print(f"[LEARNER] Warning: get_confidence_multiplier failed for {ticker}: {e}")
            return 1.0

if __name__ == "__main__":
    learner = TradeLearner()
    print(learner.train())
    