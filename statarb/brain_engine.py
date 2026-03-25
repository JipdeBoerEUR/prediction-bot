"""
brain_engine.py

BrainEngine: A supervised ML "brain" that turns signals (residuals) + context
(volatility/regime/graph features) into:
  - P(win) for each candidate trade
  - A position size (qty) and a skip decision

Key design goals:
- Deterministic feature schema (same for backtest + live)
- Calibrated probabilities (so P(win) is meaningful)
- Conservative defaults and robust handling of missing optional features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute RSI (Wilder) for a price panel."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / float(period), min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / float(period), min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


@dataclass
class BrainConfig:
    # Entry gating (rule-based safety checks). ML handles probability + sizing.
    entry_abs_residual: float = 0.01   # 1% residual on daily-return scale
    min_health_score: float = 0.20     # regime safety switch

    # Probability gating
    min_prob: float = 0.55             # skip trades with P(win) below this

    # Sizing
    base_qty: int = 1
    max_qty: int = 20

    # Optional vol targeting: qty is scaled down when vol is high
    vol_target: float = 0.015          # target daily vol (1.5%)
    use_vol_targeting: bool = True


class BrainEngine:
    """
    Supervised trade selection + sizing.

    Canonical feature schema (one row per candidate trade):
      - date (datetime-like)          [used for splitting; not a feature]
      - ticker (str)                  [identifier; not a feature]
      - residual (float)
      - abs_residual (float)
      - signal_strength (float)       (e.g., |residual| or z-score magnitude)
      - market_vol (float)            (realized vol proxy at entry)
      - asset_vol (float)             (ticker realized vol at entry)
      - health_score (float)          (RegimeEngine score at entry)
      - avg_corr (float)              (mean off-diagonal correlation in window)
      - degree (float)                (optional graph feature; ok to set 0.0)
      - wdegree (float)               (optional graph feature; ok to set 0.0)
      - side (int): +1 long, -1 short
      - sector (str)                  (optional categorical)
      - label_win (int 0/1)           (target; only for training)

    Notes:
      - The class consumes feature tables; it does not fetch data itself.
      - Outputs are calibrated probabilities via CalibratedClassifierCV.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        random_state: int = 42,
        calibration: str = "sigmoid",   # "sigmoid" is more stable than isotonic on small data
        cv: int = 3,
        model_type: str = "hgb",
    ) -> None:
        self.random_state = int(random_state)
        self.calibration = str(calibration)
        self.cv = int(cv)
        self.model_type = str(model_type).lower().strip()

        self.feature_cols = feature_cols or [
            "residual",
            "abs_residual",
            "signal_strength",
            "residual_z",
            "rsi14",
            "spy_above_200dma",
            "vix_close",
            "market_vol",
            "asset_vol",
            "health_score",
            "avg_corr",
            "degree",
            "wdegree",
            "side",
            # High-alpha features
            "residual_velocity",  # residual_z[t] - residual_z[t-3]: captures compression vs divergence
            "degree_delta_5",     # degree[t] - degree[t-5]: graph centrality shift
            "vix_adj_residual",   # abs_residual / (1 + vix_close/20): regime-adjusted signal
        ]
        self.categorical_cols = categorical_cols or ["sector"]

        self.pipeline_: Optional[Pipeline] = None
        self.fitted_: bool = False

    def _build_pipeline(self) -> Pipeline:
        numeric_cols = [c for c in self.feature_cols if c not in self.categorical_cols]

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        if self.model_type in {"hgb", "hist", "histgb", "gradient_boosting"}:
            base = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=1000,           # Was 300 — now with early stopping
                n_iter_no_change=20,     # Early stopping: halt if no val improvement for 20 rounds
                validation_fraction=0.1, # Hold out 10% for early-stopping validation
                l2_regularization=1e-4,
                random_state=self.random_state,
            )
        else:
            base = LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )

        # sklearn renamed base_estimator -> estimator in newer versions.
        try:
            clf = CalibratedClassifierCV(estimator=base, method=self.calibration, cv=self.cv)
        except TypeError:
            clf = CalibratedClassifierCV(base_estimator=base, method=self.calibration, cv=self.cv)

        return Pipeline([("pre", pre), ("clf", clf)])

    def fit(self, df: pd.DataFrame, target_col: str = "label_win", date_col: str = "signal_dt") -> "BrainEngine":
        if df is None or df.empty:
            raise ValueError("Training DataFrame is empty.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in training data.")

        needed = set(self.feature_cols + self.categorical_cols + [target_col])
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Training data missing required columns: {missing}")

        # M3: walk-forward temporal split — sort by date so folds respect time order.
        # This prevents future data leaking into validation folds during CalibratedClassifierCV.
        if date_col in df.columns:
            df_sorted = df.sort_values(date_col).reset_index(drop=True)
        else:
            df_sorted = df.copy()

        X = df_sorted[self.feature_cols + self.categorical_cols].copy()
        y = df_sorted[target_col].astype(int).to_numpy()

        # Use TimeSeriesSplit for CalibratedClassifierCV so calibration folds are forward-only
        tscv = TimeSeriesSplit(n_splits=self.cv)

        # Rebuild pipeline using walk-forward CV
        numeric_cols = [c for c in self.feature_cols if c not in self.categorical_cols]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        if self.model_type in {"hgb", "hist", "histgb", "gradient_boosting"}:
            base = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=1000,
                n_iter_no_change=20,
                validation_fraction=0.1,
                l2_regularization=1e-4,
                random_state=self.random_state,
            )
        else:
            base = LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )

        try:
            clf = CalibratedClassifierCV(estimator=base, method=self.calibration, cv=tscv)
        except TypeError:
            clf = CalibratedClassifierCV(base_estimator=base, method=self.calibration, cv=tscv)

        self.pipeline_ = Pipeline([("pre", pre), ("clf", clf)])
        self.pipeline_.fit(X, y)
        self.fitted_ = True
        return self

    def evaluate(self, df: pd.DataFrame, target_col: str = "label_win") -> Dict[str, float]:
        if not self.fitted_ or self.pipeline_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if df is None or df.empty:
            raise ValueError("Evaluation DataFrame is empty.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in evaluation data.")

        X = df[self.feature_cols + self.categorical_cols].copy()
        y = df[target_col].astype(int).to_numpy()
        p = self.pipeline_.predict_proba(X)[:, 1]

        auc = float("nan")
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))

        return {
            "auc": auc,
            "brier": float(brier_score_loss(y, p)),
            "mean_prob": float(np.mean(p)),
            "win_rate": float(np.mean(y)),
            "n": float(len(df)),
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_ or self.pipeline_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load_model() first.")
        if df is None or df.empty:
            return np.array([], dtype=float)

        # M8: validate feature columns before passing to sklearn
        all_expected = self.feature_cols + self.categorical_cols
        missing = [c for c in all_expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"predict_proba: input DataFrame missing expected columns: {missing}. "
                "Did you call build_trade_features() before predict_proba()?"
            )

        X = df[all_expected].copy()
        return self.pipeline_.predict_proba(X)[:, 1].astype(float)

    def save_model(self, path: str) -> None:
        if not self.fitted_ or self.pipeline_ is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        dump(
            {
                "pipeline": self.pipeline_,
                "feature_cols": self.feature_cols,
                "categorical_cols": self.categorical_cols,
                "random_state": self.random_state,
                "calibration": self.calibration,
                "cv": self.cv,
                "model_type": self.model_type,
            },
            path,
        )

    def load_model(self, path: str) -> "BrainEngine":
        obj = load(path)
        self.pipeline_ = obj["pipeline"]
        self.feature_cols = obj["feature_cols"]
        self.categorical_cols = obj["categorical_cols"]
        self.random_state = obj.get("random_state", self.random_state)
        self.calibration = obj.get("calibration", self.calibration)
        self.cv = obj.get("cv", self.cv)
        self.model_type = obj.get("model_type", self.model_type)
        self.fitted_ = True
        return self

    def decide_and_size(self, features_df: pd.DataFrame, config: Optional[BrainConfig] = None) -> pd.DataFrame:
        if features_df is None or features_df.empty:
            return features_df.copy()

        cfg = config or BrainConfig()
        df = features_df.copy()

        # Ensure engineered columns exist
        if "abs_residual" not in df.columns and "residual" in df.columns:
            df["abs_residual"] = df["residual"].abs()

        p = self.predict_proba(df)
        df["P_win"] = p

        qty_out: List[int] = []
        reason_out: List[str] = []

        for _, row in df.iterrows():
            health = _safe_float(row.get("health_score", np.nan))
            abs_res = _safe_float(row.get("abs_residual", np.nan))
            prob = _safe_float(row.get("P_win", np.nan))
            avol = _safe_float(row.get("asset_vol", np.nan))

            reason = ""
            qty = 0

            if not np.isfinite(health) or health < cfg.min_health_score:
                reason = "REGIME_BLOCK"
            elif not np.isfinite(abs_res) or abs_res < cfg.entry_abs_residual:
                reason = "TOO_SMALL_SIGNAL"
            elif not np.isfinite(prob) or prob < cfg.min_prob:
                reason = "LOW_PROB"
            else:
                edge = max(0.0, prob - 0.5)
                confidence = min(1.0, edge / 0.10)  # 0.60 -> 1.0
                qty = int(round(cfg.base_qty + confidence * (cfg.max_qty - cfg.base_qty)))

                if cfg.use_vol_targeting and np.isfinite(avol) and avol > 0:
                    scale = min(1.0, cfg.vol_target / avol)
                    qty = int(max(cfg.base_qty, round(qty * scale)))

                qty = int(np.clip(qty, 0, cfg.max_qty))

            qty_out.append(qty)
            reason_out.append(reason)

        df["Qty"] = qty_out
        df["SkipReason"] = reason_out
        df["Skip"] = df["Qty"] <= 0
        return df


def build_trade_features(
    signals_df: pd.DataFrame,
    *,
    market_graph: Any,
    as_of: Union[str, pd.Timestamp],
    health_score: float,
    vol_window: int = 22,
) -> pd.DataFrame:
    """
    Deterministic feature builder for LIVE usage:
      - SignalEngine output (Residual, Signal_Strength)
      - market/asset volatility from MarketGraph.returns_
      - avg_corr from rolling correlation window
      - graph features (degree, weighted degree) from MarketGraph.graph_
      - regime feature (health_score)
    """
    if signals_df is None or signals_df.empty:
        return signals_df.copy()

    df = signals_df.copy()
    used_date = pd.Timestamp(as_of)

    # Standardize column names to lower-case expected by BrainEngine
    col_map = {"Ticker": "ticker", "Residual": "residual", "Signal_Strength": "signal_strength"}
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    df["side"] = np.where(df["residual"] < 0, 1, -1).astype(int)
    df["abs_residual"] = df["residual"].abs()
    # residual_z matches the z-scored residual used for signal_strength when enabled
    df["residual_z"] = np.sign(df["residual"].to_numpy(dtype=float)) * df["signal_strength"].to_numpy(dtype=float)

    if not hasattr(market_graph, "returns_") or market_graph.returns_.empty:
        raise ValueError("market_graph must have non-empty returns_. Call fetch_data() first.")

    # M1: use strict less-than to exclude the prediction bar from feature computation
    prior_returns = market_graph.returns_[market_graph.returns_.index < used_date]
    rets = prior_returns.tail(int(vol_window))
    market_vol = float(rets.std().mean()) if len(rets) >= 2 else float("nan")
    df["market_vol"] = market_vol

    asset_vol = rets.std() if len(rets) >= 2 else pd.Series(index=rets.columns, dtype=float)
    df["asset_vol"] = df["ticker"].map(asset_vol).astype(float)

    # RSI (per-asset) — also strictly excludes the prediction bar
    if hasattr(market_graph, "prices_") and not market_graph.prices_.empty:
        prices = market_graph.prices_[market_graph.prices_.index < used_date]
        rsi = _compute_rsi(prices, period=14)
        if not rsi.empty:
            last_rsi = rsi.iloc[-1]
            df["rsi14"] = df["ticker"].map(last_rsi).astype(float)
        else:
            df["rsi14"] = float("nan")
    else:
        df["rsi14"] = float("nan")

    # avg_corr: mean off-diagonal correlation in the same rolling window (uses prior_returns, no lookahead)
    try:
        corr = rets.corr()
        vals = corr.to_numpy()
        mask = ~np.eye(vals.shape[0], dtype=bool)
        df["avg_corr"] = float(np.nanmean(vals[mask]))
    except Exception:
        df["avg_corr"] = float("nan")

    # Macro context: SPY trend + VIX
    spy_above = float("nan")
    vix_close = float("nan")
    try:
        import yfinance as yf

        end_dt = used_date.normalize()
        start_dt = end_dt - pd.Timedelta(days=800)
        spy = yf.download("SPY", start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True, progress=False)
        vix = yf.download("^VIX", start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)

        for dfm in (spy, vix):
            if isinstance(dfm.index, pd.DatetimeIndex) and dfm.index.tz is not None:
                dfm.index = dfm.index.tz_convert("UTC").tz_localize(None)

        if not spy.empty:
            if "Adj Close" in spy.columns:
                close = spy["Adj Close"].astype(float)
            elif "Close" in spy.columns:
                close = spy["Close"].astype(float)
            else:
                close = None
            if close is not None:
                sma200 = close.rolling(200, min_periods=200).mean()
                last = close.iloc[-1]
                sma = sma200.iloc[-1]
                if np.isfinite(last) and np.isfinite(sma):
                    spy_above = float(last > sma)

        if not vix.empty and "Close" in vix.columns:
            vix_close = float(vix["Close"].iloc[-1])
    except Exception:
        spy_above = float("nan")
        vix_close = float("nan")

    df["spy_above_200dma"] = float(spy_above) if np.isfinite(spy_above) else float("nan")
    df["vix_close"] = float(vix_close) if np.isfinite(vix_close) else float("nan")

    df["health_score"] = float(health_score)

    degree_map: Dict[str, float] = {}
    wdegree_map: Dict[str, float] = {}
    sector_map: Dict[str, str] = {}

    if hasattr(market_graph, "graph_") and market_graph.graph_ is not None:
        G = market_graph.graph_
        for n in G.nodes:
            degree_map[str(n)] = float(G.degree(n))
            wdegree_map[str(n)] = float(G.degree(n, weight="weight"))
            sector_map[str(n)] = str(G.nodes[n].get("sector", "Unknown"))

    df["degree"] = df["ticker"].map(degree_map).fillna(0.0).astype(float)
    df["wdegree"] = df["ticker"].map(wdegree_map).fillna(0.0).astype(float)
    df["sector"] = df["ticker"].map(sector_map).fillna("Unknown").astype(str)

    # High-alpha features
    # vix_adj_residual: computable live; others need history → default to NaN, imputed below
    vix_val = df["vix_close"].astype(float)
    abs_res_val = df["abs_residual"].astype(float)
    df["vix_adj_residual"] = abs_res_val / (1.0 + vix_val / 20.0)
    df["residual_velocity"] = float("nan")  # requires 3-bar history of residual_z; imputed below
    df["degree_delta_5"] = float("nan")     # requires 5-bar history of degree; imputed below

    # Fill any remaining NaNs in numeric features (live safety)
    # M9: use sensible domain defaults instead of 0 for health_score
    _FEATURE_DEFAULTS = {
        "health_score": 0.5,   # 0.5 = mid-range; 0.0 would falsely signal worst regime
        "spy_above_200dma": 0.5,
        "rsi14": 50.0,
        "avg_corr": 0.3,
    }
    numeric_cols = [
        "residual",
        "abs_residual",
        "signal_strength",
        "residual_z",
        "rsi14",
        "spy_above_200dma",
        "vix_close",
        "market_vol",
        "asset_vol",
        "health_score",
        "avg_corr",
        "degree",
        "wdegree",
        "side",
        "residual_velocity",
        "degree_delta_5",
        "vix_adj_residual",
    ]
    for c in numeric_cols:
        if c in df.columns:
            med = df[c].median()
            if not np.isfinite(med):
                med = float(_FEATURE_DEFAULTS.get(c, 0.0))
            df[c] = df[c].fillna(med)

    return df
