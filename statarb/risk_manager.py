# risk_manager.py
"""Portfolio-level risk management and sizing.

Purpose
-------
Convert per-candidate trade scores (signals + ML probabilities) into
executable share/units quantities while enforcing explicit constraints:

- Gross leverage cap: sum(|notional|)/equity <= max_gross_leverage
- Annual risk cap: annualized volatility <= annual_risk_target

We assume the input features already include:
  - P_win (calibrated probability)
  - side (+1 for long, -1 for short)
  - signal_strength (ideally |Residual_Z|)
  - asset_vol (realized volatility on the trading bar timeframe)
  - avg_corr (mean correlation proxy for the window)
  - price (latest executable price)

The sizing model is intentionally conservative and simple:
  1) raw score_i = max(0, P_win - 0.5) * min(signal_strength, strength_cap)
  2) risk-adjust score: score_i / max(asset_vol, eps)
  3) normalize to gross leverage cap
  4) scale down further if annualized vol estimate exceeds annual_risk_target

This is not a full-blown optimizer; it is a robust "risk governor" that makes
sure you do not accidentally exceed leverage or your risk budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    # Hard limits
    max_gross_leverage: float = 2.0
    annual_risk_target: float = 0.08

    # Risk model parameters
    bars_per_year: int = 252  # daily default; set e.g. 252*78 for 5-min US equities
    corr_cap: float = 0.90    # cap correlation used in the proxy model

    # Position hygiene
    max_weight_per_asset: float = 0.25  # cap per-asset notional as fraction of equity
    min_qty: float = 0.001             # fractional shares supported (Alpaca); was int=1

    # Portfolio style
    dollar_neutral: bool = True  # when both longs and shorts are present, enforce sum(weights)=0

    # Scoring
    strength_cap: float = 5.0
    edge_power: float = 1.0
    price_slippage_bps: float = 0.0
    sizing_method: str = "volatility"  # "volatility" or "mean_variance"
    cov_regularization: float = 0.03   # Was 1e-6 — too weak for 80-asset matrices (cond # ~10⁶)
    eps: float = 1e-12


def _safe_float(x: object, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def estimate_portfolio_annual_vol(
    weights: np.ndarray,
    sigmas_bar: np.ndarray,
    *,
    avg_corr: float,
    bars_per_year: int,
    corr_cap: float,
) -> float:
    """Conservative constant-correlation vol estimate.

    weights: fraction of equity (notional/equity), signed
    sigmas_bar: per-bar vol of each asset

    We use |weights| in the correlation term to avoid overstating diversification.
    """
    if weights.size == 0:
        return 0.0

    rho = float(np.clip(abs(avg_corr), 0.0, float(corr_cap)))

    w = weights.astype(float)
    s = sigmas_bar.astype(float)

    # Independent component
    indep = (1.0 - rho) * np.sum((w ** 2) * (s ** 2))

    # Common-correlation component (conservative: ignore sign cancellation)
    common = rho * (np.sum(np.abs(w) * s) ** 2)

    var_bar = float(max(0.0, indep + common))
    vol_ann = float(np.sqrt(var_bar * float(bars_per_year)))
    return vol_ann


def allocate_quantities(
    candidates: pd.DataFrame,
    *,
    equity: float,
    config: RiskConfig,
    price_col: str = "price",
    prob_col: str = "P_win",
    strength_col: str = "signal_strength",
    vol_col: str = "asset_vol",
    corr_col: str = "avg_corr",
    side_col: str = "side",
    cov_matrix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a copy of candidates with target_notional, target_weight, Qty.

    Required columns:
      - ticker
      - price
      - side (+1/-1)
      - P_win
      - signal_strength
      - asset_vol
      - avg_corr

    Any row with missing/invalid fields is skipped (Qty=0, Skip=True).
    """
    df = candidates.copy()
    if df.empty:
        return df

    # Normalize / ensure required columns exist
    for col in ["ticker", price_col, side_col, prob_col, strength_col, vol_col]:
        if col not in df.columns:
            raise ValueError(f"allocate_quantities: missing required column '{col}'")

    eq = float(equity)
    if not np.isfinite(eq) or eq <= 0:
        raise ValueError(f"allocate_quantities: invalid equity={equity}")

    # Build raw scores
    p = df[prob_col].astype(float).to_numpy()
    strength = df[strength_col].astype(float).to_numpy()
    strength = np.clip(strength, 0.0, float(config.strength_cap))

    BASE_RATE = 0.30
    edge = np.maximum(0.0, p - BASE_RATE) 
    
    # We schalen dit op zodat 0.35 een mooie score geeft
    # (Bijv: 0.35 - 0.30 = 0.05. We vermenigvuldigen dit om gewicht te krijgen)
    edge = edge * 5.0 

    if float(config.edge_power) != 1.0:
        edge = np.power(edge, float(config.edge_power))

    raw = edge * strength

    # Side
    side = df[side_col].astype(float).to_numpy()
    side = np.where(side >= 0, 1.0, -1.0)

    # Risk-adjust by asset vol (per-bar)
    sig = df[vol_col].astype(float).to_numpy()
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)

    adj = raw / np.maximum(sig, float(config.eps))

    # Invalid rows => score 0
    valid = np.isfinite(adj) & np.isfinite(side)
    adj = np.where(valid, adj, 0.0)

    w = None
    method = str(config.sizing_method).lower().strip()
    if method == "mean_variance" and cov_matrix is not None and not cov_matrix.empty:
        tickers = df["ticker"].astype(str).tolist()
        cov = cov_matrix.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
        # Fallback diag from asset vol when missing
        diag = np.where(np.isfinite(sig), sig ** 2, float(config.eps))
        for i in range(len(diag)):
            if not np.isfinite(cov[i, i]) or cov[i, i] <= 0:
                cov[i, i] = diag[i]
        cov = np.where(np.isfinite(cov), cov, 0.0)
        cov = cov + np.eye(len(diag)) * float(config.cov_regularization)
        mu = side * raw
        try:
            w = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(cov) @ mu
        # Enforce direction: no sign flips
        w = np.where(np.sign(w) == np.sign(side), w, 0.0)
        # Drop low-signal names
        w = np.where(raw > 0, w, 0.0)
    if w is None:
        if float(np.sum(adj)) <= 0.0:
            df["target_weight"] = 0.0
            df["target_notional"] = 0.0
            df["Qty"] = 0.0
            df["Skip"] = True
            df["SkipReason"] = "NO_VALID_SCORES"
            return df
        w_mag = adj / float(np.sum(adj))
        w = side * w_mag

    # Optionally enforce dollar neutrality (only meaningful if both long and short present)
    if bool(config.dollar_neutral) and (np.any(w > 0) and np.any(w < 0)):
        w = w - float(np.mean(w))

    # Scale to gross leverage cap
    gross = float(np.sum(np.abs(w)))
    if gross > 0:
        w = w * (float(config.max_gross_leverage) / gross)

    # Per-asset weight cap
    w = np.clip(w, -float(config.max_weight_per_asset), float(config.max_weight_per_asset))

    # Recompute gross after per-asset cap; re-scale if needed
    gross = float(np.sum(np.abs(w)))
    if gross > float(config.max_gross_leverage) and gross > 0:
        w = w * (float(config.max_gross_leverage) / gross)
    # Re-clip after re-scaling to guarantee per-asset cap is never violated
    w = np.clip(w, -float(config.max_weight_per_asset), float(config.max_weight_per_asset))

    # Annual risk cap scaling
    avg_corr_val = _safe_float(df[corr_col].astype(float).mean(), default=0.0) if (corr_col in df.columns) else 0.0
    vol_ann = estimate_portfolio_annual_vol(
        w,
        np.where(np.isfinite(sig), sig, 0.0),
        avg_corr=avg_corr_val,
        bars_per_year=int(config.bars_per_year),
        corr_cap=float(config.corr_cap),
    )

    if np.isfinite(vol_ann) and vol_ann > float(config.annual_risk_target) and vol_ann > 0:
        scale = float(config.annual_risk_target) / vol_ann
        w = w * scale

    # Final notional + Qty
    price = df[price_col].astype(float).to_numpy()
    slip = max(0.0, float(config.price_slippage_bps)) / 10_000.0
    if slip > 0:
        price = price * (1.0 + slip)
    price_ok = np.isfinite(price) & (price > 0)

    notional = w * eq
    # Fractional shares: round to 3 decimal places instead of floor-to-int
    qty = np.where(price_ok, np.round(np.abs(notional) / price, 3), 0.0)

    # Enforce min qty (float comparison supports fractional minimums like 0.001)
    qty = np.where(qty >= float(config.min_qty), qty, 0.0)

    df["target_weight"] = w
    df["target_notional"] = notional
    df["Qty"] = qty  # float; Alpaca accepts fractional quantities

    # Mark skips
    df["Skip"] = df["Qty"] <= 0

    # Provide reason for skipped rows
    reasons = []
    for i in range(len(df)):
        if not valid[i]:
            reasons.append("INVALID_FEATURES")
        elif not price_ok[i]:
            reasons.append("BAD_PRICE")
        elif raw[i] <= 0:
            reasons.append("NO_EDGE")
        elif qty[i] <= 0:
            reasons.append("TOO_SMALL_NOTIONAL")
        else:
            reasons.append("")
    df["SkipReason"] = reasons

    return df
