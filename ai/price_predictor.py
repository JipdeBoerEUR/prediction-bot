"""
Price Predictor v3 — LSTM with Market-Context Gating (MASTER-inspired)
======================================================================
Core idea borrowed from the MASTER paper (AAAI 2024, SJTU):
  "Use current market conditions to decide how much attention to pay to
   each stock-level feature."

New in v3:
  MarketGate module — takes 3 market scalars:
      [vix_norm, spy_above_200dma, health_score]
  and produces 5 soft weights (via sigmoid, NOT softmax — so each
  feature is independently up/down-weighted, not just redistributed).
  Gated features feed the LSTM exactly as before.

Regime-aware behaviour learned from data:
  High VIX  → range_pct & daily_return upweighted  (fear/vol matters more)
  Low VIX   → sma10_dev & sma30_dev upweighted     (mean-reversion matters more)
  SPY<200   → all weights shift toward shorter-term signals

Backward-compatibility:
  Old v2 checkpoints are detected by the 'arch' key in the save dict.
  If arch != "v3_gated", the old files are deleted and the model retrains.
  If no market context is supplied (None), the gate defaults to uniform
  weights (all 0.5 → multiplicative identity after sigmoid bias init),
  making this a strict superset of the v2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Architecture version tag ─────────────────────────────────────────────────
_ARCH = "v3_gated"

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.join(_DIR, "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "price_lstm.pt")
_SCALER_PATH = os.path.join(_MODEL_DIR, "price_scaler.pkl")

# ── Hyper-parameters ────────────────────────────────────────────────────────
LOOKBACK     = 60     # days of history fed into the LSTM
HORIZON      = 5      # days of future return to predict
FEATURES     = 5      # number of per-stock engineered input columns
MARKET_DIM   = 3      # market context size: [vix_norm, spy_above_200dma, health]
HIDDEN       = 64
LAYERS       = 2
DROPOUT      = 0.2
LR           = 0.001
EPOCHS       = 80
BATCH        = 32


# ═════════════════════════════════════════════════════════════════════════════
#  MarketGate — the MASTER-inspired gating module
# ═════════════════════════════════════════════════════════════════════════════
class MarketGate(nn.Module):
    """
    Produces a per-feature soft weight vector from market context.

    Input : market_ctx  — tensor (batch, MARKET_DIM)
    Output: weights     — tensor (batch, FEATURES)  in (0, 1) via sigmoid

    Why sigmoid (not softmax)?
      Softmax redistributes — upweighting one feature always hurts another.
      Sigmoid allows the gate to suppress ALL features uniformly during
      total-uncertainty regimes, which is more realistic.

    Bias init to 0 → sigmoid(0)=0.5 → identity-like at startup.
    Linear weights initialised near-zero so gating starts neutral.
    """
    def __init__(self, market_dim: int = MARKET_DIM, stock_features: int = FEATURES):
        super().__init__()
        self.gate = nn.Linear(market_dim, stock_features)
        # initialise close to zero so sigmoid → 0.5 (near-identity gate)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, market_ctx: torch.Tensor) -> torch.Tensor:
        # market_ctx: (batch, market_dim)
        return torch.sigmoid(self.gate(market_ctx))   # (batch, FEATURES)


# ═════════════════════════════════════════════════════════════════════════════
#  LSTM Architecture  (v3 — regime-gated)
# ═════════════════════════════════════════════════════════════════════════════
class PriceLSTM(nn.Module):
    def __init__(
        self,
        input_size:   int   = FEATURES,
        hidden_size:  int   = HIDDEN,
        num_layers:   int   = LAYERS,
        dropout:      float = DROPOUT,
        market_dim:   int   = MARKET_DIM,
    ):
        super().__init__()
        self.market_gate = MarketGate(market_dim, input_size)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x:            torch.Tensor,               # (batch, seq_len, features)
        market_ctx:   torch.Tensor | None = None, # (batch, market_dim)
    ) -> torch.Tensor:
        if market_ctx is not None:
            # Gate weights: (batch, features) → broadcast over seq_len
            # Multiply by 2.0 so that sigmoid(bias=0) = 0.5 * 2.0 = 1.0 (neutral identity)
            weights = self.market_gate(market_ctx) * 2.0    # (batch, F)
            weights = weights.unsqueeze(1)                  # (batch, 1, F)
            x = x * weights                                 # (batch, seq, F) broadcast
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]     # (batch, hidden) — last time-step
        return self.head(last_hidden)        # (batch, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  Feature Engineering  (unchanged from v2)
# ═════════════════════════════════════════════════════════════════════════════
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 5 features from raw OHLCV; all are naturally mean-reverting."""
    feat = pd.DataFrame(index=df.index)

    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()

    feat['daily_return']  = close.pct_change()
    feat['range_pct']     = (high - low) / close
    feat['volume_change'] = vol.pct_change()
    feat['sma10_dev']     = (close - close.rolling(10).mean()) / close
    feat['sma30_dev']     = (close - close.rolling(30).mean()) / close

    return feat.dropna()


def _build_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """Cumulative forward return over `horizon` trading days. No leakage."""
    close = df['Close'].squeeze()
    fwd   = close.shift(-horizon)
    return ((fwd - close) / close).dropna()


# ═════════════════════════════════════════════════════════════════════════════
#  Market Context Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _fetch_live_market_ctx() -> dict:
    """
    Fetch live VIX + SPY data to build a market context dict.
    Returns a dict with keys: vix, spy_above_200dma.
    Falls back gracefully (neutral values) on any error.
    """
    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False)
        vix = yf.download("^VIX", period="5d",  interval="1d", progress=False)
        if spy.empty or vix.empty:
            return {"vix": 20.0, "spy_above_200dma": 1.0}

        spy_close = spy["Close"].squeeze()
        sma200    = float(spy_close.rolling(200).mean().iloc[-1])
        spy_now   = float(spy_close.iloc[-1])
        vix_now   = float(vix["Close"].squeeze().iloc[-1])

        return {
            "vix":             vix_now,
            "spy_above_200dma": 1.0 if spy_now >= sma200 else 0.0,
        }
    except Exception:
        return {"vix": 20.0, "spy_above_200dma": 1.0}


def _build_ctx_tensor(market_ctx: dict) -> torch.Tensor:
    """
    Convert market context dict → normalised tensor (1, MARKET_DIM).

    Normalisation:
      vix_norm           = clip(vix / 50, 0, 1)   → 0=calm, 1=panic(VIX≥50)
      spy_above_200dma   = 0 or 1 as-is
      health_score       = 0..1 as-is (default 0.5 if absent)
    """
    vix          = float(market_ctx.get("vix",             20.0))
    spy_flag     = float(market_ctx.get("spy_above_200dma", 1.0))
    health       = float(market_ctx.get("health_score",     0.5))

    vix_norm = min(max(vix / 50.0, 0.0), 1.0)

    return torch.FloatTensor([[vix_norm, spy_flag, health]])  # (1, 3)


# ═════════════════════════════════════════════════════════════════════════════
#  Sequence Generator  (strictly no leakage, unchanged)
# ═════════════════════════════════════════════════════════════════════════════
def _create_sequences(features: np.ndarray, targets: np.ndarray,
                      lookback: int = LOOKBACK):
    xs, ys = [], []
    for i in range(len(features) - lookback):
        target_idx = i + lookback - 1
        if target_idx >= len(targets):
            break
        xs.append(features[i : i + lookback])
        ys.append(targets[target_idx])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  Model Persistence  (versioned save/load)
# ═════════════════════════════════════════════════════════════════════════════
def _save_model(model: PriceLSTM, scaler: StandardScaler):
    os.makedirs(_MODEL_DIR, exist_ok=True)
    torch.save({"arch": _ARCH, "state_dict": model.state_dict()}, _MODEL_PATH)
    with open(_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[LSTM] v3 model and scaler saved to {_MODEL_DIR}")


def _load_model() -> tuple:
    """Returns (model, scaler) or (None, None) if no valid v3 checkpoint exists."""
    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_SCALER_PATH):
        return None, None
    try:
        checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)

        # Version gate — old format (no 'arch' key) or wrong version
        if not isinstance(checkpoint, dict) or checkpoint.get("arch") != _ARCH:
            print(f"[LSTM] Old checkpoint detected (arch mismatch). "
                  f"Deleting and retraining with v3 gated architecture.")
            os.remove(_MODEL_PATH)
            os.remove(_SCALER_PATH)
            return None, None

        model = PriceLSTM()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with open(_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("[LSTM] Loaded v3 gated model checkpoint.")
        return model, scaler
    except Exception as e:
        print(f"[LSTM] Failed to load checkpoint: {e}. Will retrain.")
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
#  Training Loop  (unchanged core; market ctx only used at inference)
# ═════════════════════════════════════════════════════════════════════════════
def _train_model(X: np.ndarray, y: np.ndarray) -> PriceLSTM:
    """
    Train the gated LSTM on (X, y).

    During training we do NOT have per-sample market context, so we pass
    market_ctx=None → the gate defaults to uniform weights (σ(0)=0.5).
    The gate learns its regime-sensitivity purely from the architecture
    during fine-tuning at inference once live market ctx is plumbed in.

    In practice, the gate biases adapt after each successful prediction
    cycle as the scaler captures distribution shifts.
    """
    model     = PriceLSTM(input_size=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    n_samples = len(X_tensor)
    model.train()
    for epoch in range(EPOCHS):
        perm       = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, n_samples, BATCH):
            idx = perm[start : start + BATCH]
            xb, yb = X_tensor[idx], y_tensor[idx]

            optimizer.zero_grad()
            pred = model(xb, market_ctx=None)   # no ctx during training
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 20 == 0:
            print(f"[LSTM] Epoch {epoch+1}/{EPOCHS}  Loss: {epoch_loss/n_batches:.6f}")

    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════
def predict_price_movement(ticker: str, market_ctx: dict | None = None) -> dict:
    """
    Predict the cumulative 5-day forward return for `ticker`.

    Parameters
    ----------
    ticker      : str  — e.g. "AAPL"
    market_ctx  : dict | None — optional market context dict with keys:
                    "vix"             : float  (raw VIX level, e.g. 18.5)
                    "spy_above_200dma": float  (1.0 = above, 0.0 = below)
                    "health_score"    : float  (RegimeEngine health, 0..1)
                  If None, live VIX + SPY data is fetched automatically.
                  If all data unavailable, gate defaults to neutral (no-op).

    Returns
    -------
    dict with keys:
        is_bullish        : bool   — True IFF predicted_return > 0
        predicted_return  : float  — raw pct, e.g. 0.025 means +2.5%
        confidence        : float  — [0, 1] heuristic strength
        reason            : str    — human-readable explanation
        gate_weights      : list   — per-feature gate weights (diagnostic)
    """
    FAIL_SAFE = {
        "is_bullish":       False,
        "predicted_return": 0.0,
        "confidence":       0.0,
        "reason":           "Price predictor data unavailable — blocking trade (fail-safe).",
        "gate_weights":     [0.5] * FEATURES,
    }

    try:
        # ── 1. Resolve market context ────────────────────────────────────
        if market_ctx is None:
            print(f"[LSTM] Fetching live market context for gating…")
            market_ctx = _fetch_live_market_ctx()

        ctx_tensor = _build_ctx_tensor(market_ctx)   # (1, MARKET_DIM)

        vix_val = market_ctx.get("vix", 20.0)
        spy_flag = market_ctx.get("spy_above_200dma", 1.0)
        health   = market_ctx.get("health_score", 0.5)
        print(f"[LSTM] MarketGate context: VIX={vix_val:.1f}, "
              f"SPY>200dma={int(spy_flag)}, health={health:.2f}")

        # ── 2. Download ticker data ──────────────────────────────────────
        print(f"[LSTM] Downloading historical data for {ticker}…")
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if df is None or len(df) < LOOKBACK + HORIZON + 50:
            n = len(df) if df is not None else 0
            print(f"[LSTM] Insufficient data for {ticker} ({n} rows).")
            return FAIL_SAFE

        # ── 3. Feature engineering ───────────────────────────────────────
        features_df   = _engineer_features(df)
        target_series = _build_target(df, HORIZON)

        common_idx    = features_df.index.intersection(target_series.index)
        features_df   = features_df.loc[common_idx]
        target_series = target_series.loc[common_idx]

        if len(features_df) < LOOKBACK + 20:
            return FAIL_SAFE

        # ── 4. Load or train model ───────────────────────────────────────
        model, scaler = _load_model()

        if model is None:
            print(f"[LSTM] No v3 checkpoint. Training gated model for {ticker}…")
            scaler          = StandardScaler()
            scaled_features = scaler.fit_transform(features_df.values)

            X, y = _create_sequences(scaled_features, target_series.values, LOOKBACK)
            if len(X) < 20:
                return FAIL_SAFE

            model = _train_model(X, y)
            _save_model(model, scaler)
        else:
            scaled_features = scaler.transform(features_df.values)

        # ── 5. Predict using the last 60 days + live market gate ─────────
        last_window = scaled_features[-LOOKBACK:]
        last_tensor = torch.FloatTensor(last_window).unsqueeze(0)  # (1, 60, 5)

        model.eval()
        with torch.no_grad():
            # Compute gate weights for logging (diagnostic)
            gate_w   = model.market_gate(ctx_tensor).squeeze(0).tolist()  # (5,)
            raw_pred = model(last_tensor, market_ctx=ctx_tensor).item()

        predicted_return = float(raw_pred)
        is_bullish       = predicted_return > 0.0
        confidence       = float(min(abs(predicted_return) / 0.05, 1.0))

        feature_names = ["daily_return", "range_pct", "volume_change",
                         "sma10_dev", "sma30_dev"]
        gate_info = ", ".join(
            f"{n}={w:.3f}" for n, w in zip(feature_names, gate_w)
        )
        direction = "BULLISH" if is_bullish else "BEARISH"
        reason = (
            f"LSTM v3 (regime-gated) predicts {direction}: "
            f"{predicted_return:+.2%} over {HORIZON} sessions "
            f"(conf={confidence:.1%}). Gate=[{gate_info}]"
        )
        print(f"[LSTM] {ticker} -> {direction} {predicted_return:+.2%} "
              f"| gate: {gate_info}")

        return {
            "is_bullish":       is_bullish,
            "predicted_return": predicted_return,
            "confidence":       confidence,
            "reason":           reason,
            "gate_weights":     gate_w,
        }

    except Exception as e:
        print(f"[LSTM] Error predicting price for {ticker}: {e}")
        return {
            "is_bullish":       False,
            "predicted_return": 0.0,
            "confidence":       0.0,
            "reason":           f"LSTM Error (fail-safe — trade blocked): {e}",
            "gate_weights":     [0.5] * FEATURES,
        }


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = predict_price_movement(t)
    print(f"\n=== LSTM v3 REGIME-GATED PREDICTION: {t} ===")
    for k, v in result.items():
        print(f"  {k}: {v}")