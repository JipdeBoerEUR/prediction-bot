"""
Price Predictor v2 — Production-Grade PyTorch LSTM
===================================================
Predicts the cumulative 5-day forward return of a stock using a 60-day
lookback window of engineered features.

Architecture fixes over v1:
  1. StandardScaler (zero-centred) replaces MinMaxScaler — model CAN output negatives.
  2. Strict 60-day lookback, 5-day forward target.  No future data leakage.
  3. Model weights + scaler are persisted to disk; training only fires when
     no checkpoint exists or when the user explicitly requests retraining.
  4. Inverse-transforms the final prediction → raw pct return (e.g. 0.025 = +2.5%).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_DIR, "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "price_lstm.pt")
_SCALER_PATH = os.path.join(_MODEL_DIR, "price_scaler.pkl")

# ── Hyper-parameters ────────────────────────────────────────────────────────
LOOKBACK = 60          # days of history fed into the LSTM
HORIZON  = 5           # days of future return to predict
FEATURES = 5           # number of engineered input columns
HIDDEN   = 64
LAYERS   = 2
DROPOUT  = 0.2
LR       = 0.001
EPOCHS   = 80
BATCH    = 32


# ═════════════════════════════════════════════════════════════════════════════
#  LSTM Architecture
# ═════════════════════════════════════════════════════════════════════════════
class PriceLSTM(nn.Module):
    def __init__(self, input_size: int = FEATURES, hidden_size: int = HIDDEN,
                 num_layers: int = LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]     # (batch, hidden) — last time-step
        return self.head(last_hidden)         # (batch, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  Feature Engineering  (returns-based — all zero-centred by design)
# ═════════════════════════════════════════════════════════════════════════════
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 5 features from raw OHLCV; all are naturally mean-reverting."""
    feat = pd.DataFrame(index=df.index)

    # Flatten multi-index columns from yfinance if present
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()

    feat['daily_return']  = close.pct_change()                               # 1-day log-like return
    feat['range_pct']     = (high - low) / close                             # intraday range as % of close
    feat['volume_change'] = vol.pct_change()                                 # volume momentum
    feat['sma10_dev']     = (close - close.rolling(10).mean()) / close       # deviation from 10-day SMA
    feat['sma30_dev']     = (close - close.rolling(30).mean()) / close       # deviation from 30-day SMA

    return feat.dropna()


def _build_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Cumulative forward return over `horizon` trading days.
    target_t = (Close_{t+horizon} - Close_t) / Close_t
    Strictly uses FUTURE prices — no leakage into past features.
    """
    close = df['Close'].squeeze()
    fwd = close.shift(-horizon)
    return ((fwd - close) / close).dropna()


# ═════════════════════════════════════════════════════════════════════════════
#  Sequence Generator  (strictly no leakage)
# ═════════════════════════════════════════════════════════════════════════════
def _create_sequences(features: np.ndarray, targets: np.ndarray,
                      lookback: int = LOOKBACK):
    """
    For each index i:
      X[i] = features[i : i+lookback]   (60 days of history)
      y[i] = targets[i+lookback-1]       (the forward return STARTING at the
                                           last day of the window)

    This guarantees that the target is ALWAYS in the future relative to
    every feature timestamp in the window.
    """
    xs, ys = [], []
    for i in range(len(features) - lookback):
        target_idx = i + lookback - 1
        if target_idx >= len(targets):
            break
        xs.append(features[i : i + lookback])
        ys.append(targets[target_idx])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  Model Persistence
# ═════════════════════════════════════════════════════════════════════════════
def _save_model(model: PriceLSTM, scaler: StandardScaler):
    os.makedirs(_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), _MODEL_PATH)
    with open(_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[LSTM] Model and scaler saved to {_MODEL_DIR}")


def _load_model() -> tuple:
    """Returns (model, scaler) or (None, None) if no checkpoint exists."""
    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_SCALER_PATH):
        return None, None
    try:
        model = PriceLSTM()
        model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        with open(_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("[LSTM] Loaded saved model checkpoint.")
        return model, scaler
    except Exception as e:
        print(f"[LSTM] Failed to load checkpoint: {e}")
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═════════════════════════════════════════════════════════════════════════════
def _train_model(X: np.ndarray, y: np.ndarray):
    """Train the LSTM on (X, y) and return the trained model.
    The scaler is fit on X BEFORE sequence creation, so we receive pre-scaled data."""
    model = PriceLSTM(input_size=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Mini-batch training
    n_samples = len(X_tensor)
    model.train()
    for epoch in range(EPOCHS):
        # Shuffle indices each epoch
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, BATCH):
            idx = perm[start : start + BATCH]
            xb, yb = X_tensor[idx], y_tensor[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"[LSTM] Epoch {epoch+1}/{EPOCHS}  Loss: {epoch_loss/n_batches:.6f}")

    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════
def predict_price_movement(ticker: str) -> dict:
    """
    Predict the cumulative 5-day forward return for `ticker`.

    Returns
    -------
    dict with keys:
        is_bullish       : bool   — True IFF predicted_return > 0
        predicted_return : float  — raw pct, e.g. 0.025 means +2.5%
        confidence       : float  — [0, 1] heuristic strength
        reason           : str    — human-readable explanation
    """
    FAIL_OPEN = {
        "is_bullish": True,
        "predicted_return": 0.0,
        "confidence": 0.0,
        "reason": "Price predictor data unavailable — failing open.",
    }

    try:
        # ── 1. Download data ────────────────────────────────────────────
        # Need at least LOOKBACK + HORIZON + 30 (for rolling windows) days
        # Fetch ~2 years to have a solid training set
        print(f"[LSTM] Downloading historical data for {ticker}...")
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if df is None or len(df) < LOOKBACK + HORIZON + 50:
            print(f"[LSTM] Insufficient data for {ticker} ({len(df) if df is not None else 0} rows).")
            return FAIL_OPEN

        # ── 2. Feature engineering ──────────────────────────────────────
        features_df = _engineer_features(df)
        target_series = _build_target(df, HORIZON)

        # Align features and targets on the same index
        common_idx = features_df.index.intersection(target_series.index)
        features_df = features_df.loc[common_idx]
        target_series = target_series.loc[common_idx]

        if len(features_df) < LOOKBACK + 20:
            return FAIL_OPEN

        # ── 3. Scale features with StandardScaler (zero-centred) ────────
        model, scaler = _load_model()

        if model is None:
            # No checkpoint → fit scaler and train from scratch
            print(f"[LSTM] No checkpoint found. Training new model for {ticker}...")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df.values)

            X, y = _create_sequences(scaled_features, target_series.values, LOOKBACK)
            if len(X) < 20:
                return FAIL_OPEN

            model = _train_model(X, y)
            _save_model(model, scaler)
        else:
            # Checkpoint exists → just transform features with saved scaler
            scaled_features = scaler.transform(features_df.values)

        # ── 4. Predict using the last 60 days ───────────────────────────
        last_window = scaled_features[-LOOKBACK:]
        last_tensor = torch.FloatTensor(last_window).unsqueeze(0)  # (1, 60, 5)

        model.eval()
        with torch.no_grad():
            raw_prediction = model(last_tensor).item()

        # raw_prediction IS the predicted cumulative 5-day return (e.g. 0.025)
        # because we trained on raw returns directly (not scaled returns)
        predicted_return = float(raw_prediction)
        is_bullish = predicted_return > 0.0

        # Confidence: how far from zero, capped at 1.0
        # A 5% predicted move → conf ≈ 1.0; a 0.1% move → conf ≈ 0.02
        confidence = float(min(abs(predicted_return) / 0.05, 1.0))

        direction = "BULLISH" if is_bullish else "BEARISH"
        reason = (
            f"LSTM predicts {direction} movement: "
            f"{predicted_return:+.2%} cumulative return over {HORIZON} sessions "
            f"(confidence: {confidence:.1%})."
        )

        return {
            "is_bullish": is_bullish,
            "predicted_return": predicted_return,
            "confidence": confidence,
            "reason": reason,
        }

    except Exception as e:
        print(f"[LSTM] Error predicting price for {ticker}: {e}")
        return {
            "is_bullish": True,
            "predicted_return": 0.0,
            "confidence": 0.0,
            "reason": f"LSTM Error (failing open): {e}",
        }


if __name__ == "__main__":
    result = predict_price_movement("AAPL")
    print("\n=== LSTM PRICE PREDICTION ===")
    for k, v in result.items():
        print(f"  {k}: {v}")