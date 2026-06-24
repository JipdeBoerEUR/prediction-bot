"""dataset_builder_v2.py

Event-level dataset builder for the Topological Arbitrage "Brain" model.

Key properties:
  - Works for DAILY or INTRADAY bars (uses index steps, not calendar days).
  - Generates BOTH long and short examples (symmetric mean-reversion).
  - Uses signed z-scored residuals (timeframe / asset-class invariant).
  - Labels examples using a FIXED forward horizon (avoids path-dependent exits).

Output:
  - data/brain_events_<engine>.csv  (engine default: "default")

Label definition (default):
  - Enter at t+1 close (bar-lag = 1)
  - Exit at t+1+H close (H = HOLD_BARS)
  - Long if residual_z <= -ENTRY_Z
  - Short if residual_z >= +ENTRY_Z
  - label_win = 1 if net PnL > 0 after COST_BPS on both legs

This dataset is designed to be consumed by brain_engine.BrainEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from statarb import config as cfg
from statarb import sim_engine as sim
from statarb.graph_engine import MarketGraph
from statarb.regime_engine import RegimeEngine


OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --- yfinance intraday guardrails -------------------------------------------------
# Yahoo Finance intraday intervals (e.g., 5m/15m/30m) are typically limited to the last ~60 days.
# If the user requests an older range, silently returning empty data is common.
# We clamp the request to the maximum likely window to avoid "no data" failures.
_INTRADAY_LIMITED_INTERVALS = {"1m", "2m", "5m", "15m", "30m"}

def _clamp_intraday_range(start: str, end: str, interval: str) -> tuple[str, str]:
    if interval not in _INTRADAY_LIMITED_INTERVALS:
        return start, end
    try:
        today = pd.Timestamp.utcnow().normalize()
        min_start = today - pd.Timedelta(days=59)
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        # Clamp to [min_start, today]
        if s < min_start:
            s = min_start
        if e > today:
            e = today
        # If end before start, force end=today
        if e <= s:
            e = today
        return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")
    except Exception:
        return start, end


def _compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute RSI (Wilder) for a price panel."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / float(period), min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / float(period), min_periods=period, adjust=False).mean()
    # O7: guard against avg_loss=0 (no down-moves → RSI=100; 0/0 → neutral RSI=50)
    safe_loss = avg_loss.where(avg_loss > 1e-12, other=np.nan)
    rs = avg_gain / safe_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_gain=0 AND avg_loss=0 → rs=NaN → rsi=NaN → fill with 50 (neutral)
    # avg_gain>0 AND avg_loss=0 → rs=NaN → fill with 100 (no losses)
    rsi_no_loss = avg_gain.gt(0) & avg_loss.le(1e-12)
    rsi = rsi.where(~rsi_no_loss, other=100.0)
    rsi = rsi.fillna(50.0)
    return rsi


def _fetch_macro_series(start: str, end: Optional[str]) -> pd.DataFrame:
    """Fetch SPY trend + VIX level (daily) for macro context."""
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.utcnow().normalize()
    start_dt = pd.Timestamp(start) if start else (end_dt - pd.Timedelta(days=800))
    spy = yf.download("SPY", start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)

    for df in (spy, vix):
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

    out = pd.DataFrame(index=spy.index)
    spy_close = None
    if not spy.empty:
        if "Adj Close" in spy.columns:
            spy_close = spy["Adj Close"].astype(float)
        elif "Close" in spy.columns:
            spy_close = spy["Close"].astype(float)
    if spy_close is not None:
        sma200 = spy_close.rolling(200, min_periods=200).mean()
        out["spy_above_200dma"] = (spy_close > sma200).astype(float)
    else:
        out["spy_above_200dma"] = np.nan

    if not vix.empty and "Close" in vix.columns:
        out["vix_close"] = vix["Close"].astype(float)
    else:
        out["vix_close"] = np.nan

    return out



def _apply_cost(notional: float, cost_bps: float) -> float:
    return float(notional) * (float(cost_bps) / 10_000.0)


def _get_engine(engine: str) -> Dict:
    """Read engine config from cfg.ENGINES if present; else fallback to single-universe config."""
    engines = getattr(cfg, "ENGINES", None)
    if isinstance(engines, dict) and engine in engines:
        return dict(engines[engine])

    # Fallback: old config style
    return {
        "tickers": getattr(cfg, "TICKERS", []),
        "start": getattr(cfg, "DATASET_START", "2018-01-01"),
        "end": getattr(cfg, "DATASET_END", "2025-12-31"),
        "interval": getattr(cfg, "INTERVAL", "1d"),
    }


def _avg_corr(window_returns: pd.DataFrame) -> float:
    c = window_returns.corr().values
    n = c.shape[0]
    if n <= 1:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(c[mask]))


def _market_proxy_return(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1)


def _realized_vol(series: pd.Series, asof_i: int, lookback: int) -> float:
    # series indexed by datetime; use integer position for speed/stability
    if asof_i < 1:
        return float("nan")
    start_i = max(0, asof_i - int(lookback) + 1)
    x = series.iloc[start_i : asof_i + 1]
    if len(x) < 2:
        return float("nan")
    return float(x.std())


def _asset_vol(returns: pd.DataFrame, asof_i: int, lookback: int) -> pd.Series:
    if asof_i < 1:
        return pd.Series(index=returns.columns, dtype=float)
    start_i = max(0, asof_i - int(lookback) + 1)
    w = returns.iloc[start_i : asof_i + 1]
    if len(w) < 2:
        return pd.Series(index=returns.columns, dtype=float)
    return w.std()


def _compute_regime_health(window_returns: pd.DataFrame) -> float:
    reg = RegimeEngine()
    try:
        return float(reg.compute_regime(window_returns.corr()))
    except Exception:
        return float("nan")


def _degrees_from_W(W: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # degree = number of nonzero edges; weighted degree = sum of weights
    w = W.copy()
    np.fill_diagonal(w.values, 0.0)
    deg = (w > 0).sum(axis=1).astype(float)
    wdeg = w.sum(axis=1).astype(float)
    return deg, wdeg


def _compute_residual_row(
    returns: pd.DataFrame,
    i: int,
    lookback: int,
    sectors: Optional[pd.Series],
    sector_only: bool,
    graph_threshold: float,
    alpha: float,
    regularization: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Residual vector at index i and the adjacency W used."""
    window = returns.iloc[max(0, i - lookback + 1) : i + 1]
    W = sim.build_adjacency_from_returns(
        window,
        sectors=sectors,
        sector_only=bool(sector_only),
        threshold=float(graph_threshold),
    )

    x = returns.iloc[i].to_numpy(dtype=float)
    r = sim.solve_diffusion_residual(
        x,
        W.to_numpy(dtype=float),
        alpha=float(alpha),
        regularization=float(regularization),
    )
    return pd.Series(r, index=returns.columns), W


@dataclass
class BuildConfig:
    entry_z: float
    hold_bars: int
    exec_lag_bars: int
    lookback_bars: int
    cost_bps: float
    sector_only: bool
    graph_threshold: float
    alpha: float
    regularization: float
    use_regime_filter: bool
    regime_min_health: float
    vol_lookback: int
    max_events_per_bar: int


def _load_defaults_from_cfg() -> BuildConfig:
    # Prefer new names; fallback to legacy daily names.
    return BuildConfig(
        entry_z=float(getattr(cfg, "ENTRY_Z", 2.0)),
        hold_bars=int(getattr(cfg, "HOLD_BARS", getattr(cfg, "MAX_HOLD_DAYS", 10))),
        exec_lag_bars=int(getattr(cfg, "EXEC_LAG_BARS", 1)),
        lookback_bars=int(getattr(cfg, "LOOKBACK_BARS", getattr(cfg, "LOOKBACK_DAYS", 22))),
        cost_bps=float(getattr(cfg, "COST_BPS", 10.0)),
        sector_only=bool(getattr(cfg, "SECTOR_ONLY", False)),
        graph_threshold=float(getattr(cfg, "GRAPH_THRESHOLD", 0.30)),
        alpha=float(getattr(cfg, "ALPHA", 0.8)),
        regularization=float(getattr(cfg, "REGULARIZATION", 1e-6)),
        use_regime_filter=bool(getattr(cfg, "USE_REGIME_FILTER", True)),
        regime_min_health=float(getattr(cfg, "REGIME_MIN_HEALTH", 0.20)),
        vol_lookback=int(getattr(cfg, "VOL_LOOKBACK", 50)),
        max_events_per_bar=int(getattr(cfg, "MAX_EVENTS_PER_BAR", 20)),
    )

def _apply_engine_overrides(bcfg: BuildConfig, e: Dict) -> BuildConfig:
    if "entry_z" in e:
        bcfg.entry_z = float(e["entry_z"])
    if "hold_bars" in e:
        bcfg.hold_bars = int(e["hold_bars"])
    if "exec_lag_bars" in e:
        bcfg.exec_lag_bars = int(e["exec_lag_bars"])
    if "lookback_bars" in e:
        bcfg.lookback_bars = int(e["lookback_bars"])
    if "cost_bps" in e:
        bcfg.cost_bps = float(e["cost_bps"])
    if "sector_only" in e:
        bcfg.sector_only = bool(e["sector_only"])
    if "graph_threshold" in e:
        bcfg.graph_threshold = float(e["graph_threshold"])
    if "alpha" in e:
        bcfg.alpha = float(e["alpha"])
    if "regularization" in e:
        bcfg.regularization = float(e["regularization"])
    if "use_regime_filter" in e:
        bcfg.use_regime_filter = bool(e["use_regime_filter"])
    if "regime_min_health" in e:
        bcfg.regime_min_health = float(e["regime_min_health"])
    if "vol_lookback" in e:
        bcfg.vol_lookback = int(e["vol_lookback"])
    if "max_events_per_bar" in e:
        bcfg.max_events_per_bar = int(e["max_events_per_bar"])
    return bcfg


def build_events_for_engine(engine: str = "default") -> Path:
    e = _get_engine(engine)
    bcfg = _load_defaults_from_cfg()
    bcfg = _apply_engine_overrides(bcfg, e)

    tickers = list(e.get("tickers", []))
    if not tickers:
        raise ValueError("No tickers configured.")

    start = e.get("start", getattr(cfg, "DATASET_START", "2018-01-01"))
    end = e.get("end", getattr(cfg, "DATASET_END", "2025-12-31"))
    interval = e.get("interval", getattr(cfg, "INTERVAL", "1d"))
    provider = e.get("provider", getattr(cfg, "PROVIDER", "yfinance"))
    alpaca_feed = e.get("alpaca_feed", getattr(cfg, "ALPACA_FEED", None))
    # Clamp overly-old intraday requests to Yahoo's typical limits (prevents empty datasets).
    if str(provider).lower().strip() == "yfinance":
        start_clamped, end_clamped = _clamp_intraday_range(start, end, interval)
        if (start_clamped, end_clamped) != (start, end) and interval in _INTRADAY_LIMITED_INTERVALS:
            print(
                f"[dataset_builder_v2] NOTE: interval={interval} is typically limited on Yahoo; clamping range "
                f"{start}->{end} to {start_clamped}->{end_clamped}."
            )
        start, end = start_clamped, end_clamped
    out_path = OUT_DIR / f"brain_events_{engine}.csv"

    print(f"[dataset_builder_v2] engine={engine} | provider={provider} | tickers={len(tickers)} | interval={interval} | {start} -> {end}")
    # Provider-aware MarketGraph (provider and alpaca_feed are optional in older versions)
    try:
        mg = MarketGraph(tickers=tickers, start=start, end=end, provider=provider, alpaca_feed=alpaca_feed)
    except TypeError:
        mg = MarketGraph(tickers=tickers, start=start, end=end)
    # Newer MarketGraph may accept interval; older versions will ignore.
    try:
        mg.fetch_data(interval=interval, verbose=True)  # type: ignore[arg-type]
    except TypeError:
        mg.fetch_data(verbose=True)
    except ValueError as ex:
        msg = str(ex)
        p = str(provider).lower().strip()
        if p == "alpaca":
            extra = (
                "Data fetch failed for Alpaca. Verify credentials and permissions in your .env "
                "(ALPACA_API_KEY + ALPACA_SECRET_KEY or APCA_API_KEY_ID + APCA_API_SECRET_KEY), "
                "and confirm your account has access to the selected feed (iex/sip)."
            )
        elif p == "yfinance":
            extra = (
                f"Data fetch failed. If you are using Yahoo Finance intraday (interval={interval}), "
                "request a range within the provider's limit (often ~60 days for 5m/15m/30m), "
                "or switch to interval='60m' (often up to ~730 days). "
                "For production intraday FX, use a dedicated data feed (broker/OANDA/Dukascopy/Finnhub)."
            )
        else:
            extra = "Data fetch failed. Check provider credentials, symbol compatibility, and requested interval/date range."
        raise ValueError(f"{msg}\n{extra}") from ex


    # Sectors (equities only). FX will fall back to Unknown.
    sectors: Optional[pd.Series] = None
    try:
        mg.fetch_sectors(exclude_unknown=False, verbose=False)
        sectors = mg.sectors_
        # Some MarketGraph versions may return sectors_ as a dict; normalize to pd.Series
        if isinstance(sectors, dict):
            sectors = pd.Series(sectors)
    except Exception:
        sectors = None

    prices = mg.prices_.copy()
    returns = mg.returns_.copy()
    returns = returns.dropna(how="all")
    prices = prices.reindex(returns.index)

    # RSI (per-asset)
    rsi_period = int(getattr(cfg, "RSI_PERIOD", 14))
    rsi_df = _compute_rsi(prices, period=rsi_period).reindex(returns.index)

    # Macro context (SPY trend + VIX level)
    macro = _fetch_macro_series(start=start, end=end)
    if isinstance(macro.index, pd.DatetimeIndex) and macro.index.tz is not None:
        macro.index = macro.index.tz_convert("UTC").tz_localize(None)
    macro = macro.reindex(returns.index).ffill()  # O8: method="ffill" deprecated in newer pandas

    exit_z = float(getattr(cfg, "EXIT_Z", 0.35))
    trail_stop_pct = float(getattr(cfg, "TRAIL_STOP_PCT", 0.0))

    # Basic guardrail
    if len(returns) < (bcfg.lookback_bars + bcfg.exec_lag_bars + bcfg.hold_bars + 5):
        raise ValueError(
            f"Not enough bars for lookback/hold: bars={len(returns)} lookback={bcfg.lookback_bars} "
            f"exec_lag={bcfg.exec_lag_bars} hold={bcfg.hold_bars}."
        )

    # Precompute residuals (panel) for entry/exit logic
    trade_dates = returns.index[bcfg.lookback_bars - 1 :]
    resid_panel = sim.compute_residual_panel(
        returns,
        lookback_days=bcfg.lookback_bars,
        sectors=sectors,
        sector_only=bcfg.sector_only,
        graph_threshold=bcfg.graph_threshold,
        alpha=bcfg.alpha,
        regularization=bcfg.regularization,
        trade_dates=trade_dates,
        n_jobs=int(getattr(cfg, "RESIDUAL_N_JOBS", 1)),
        backend=str(getattr(cfg, "RESIDUAL_JOBLIB_BACKEND", "threading")),
    )
    resid_panel = resid_panel.reindex(returns.index)
    resid_std = resid_panel.std(axis=1)
    resid_z_panel = resid_panel.div(resid_std, axis=0)

    # --- High-alpha feature panels ---
    # Feature 1: residual_velocity = residual_z[t] - residual_z[t-3]
    resid_vel_panel = resid_z_panel.diff(3)

    # Feature 2: degree_delta_5 = degree[t] - degree[t-5]
    degree_panel = sim.compute_degree_panel(
        returns,
        lookback_days=bcfg.lookback_bars,
        sectors=sectors,
        sector_only=bcfg.sector_only,
        graph_threshold=bcfg.graph_threshold,
        trade_dates=trade_dates,
        n_jobs=int(getattr(cfg, "RESIDUAL_N_JOBS", 1)),
        backend=str(getattr(cfg, "RESIDUAL_JOBLIB_BACKEND", "threading")),
    )
    degree_panel = degree_panel.reindex(returns.index)
    degree_delta5_panel = degree_panel.diff(5)

    mkt_proxy = _market_proxy_return(returns)

    rows = []
    idx = returns.index
    abs_entry_z = abs(bcfg.entry_z)

    event_n_jobs = int(getattr(cfg, "EVENT_N_JOBS", 1))
    event_backend = str(getattr(cfg, "EVENT_JOBLIB_BACKEND", "threading"))

    # Iterate over signal bars; ensure we have t+exec_lag and t+exec_lag+hold
    last_i = len(idx) - 1 - (bcfg.exec_lag_bars + bcfg.hold_bars)

    def _rows_for_bar(i: int) -> list[Dict]:
        d = idx[i]

        # Residuals and z-scores at time i
        resid_z = resid_z_panel.iloc[i]
        if resid_z.isna().all():
            return []

        # Entry selection (symmetric). If nothing triggers, skip heavy work.
        long_cand = resid_z[resid_z <= -abs_entry_z].sort_values(ascending=True)
        short_cand = resid_z[resid_z >= abs_entry_z].sort_values(ascending=False)
        if long_cand.empty and short_cand.empty:
            return []

        # Regime (optional)
        health = float("nan")
        window = returns.iloc[i - bcfg.lookback_bars + 1 : i + 1]
        if bcfg.use_regime_filter:
            health = _compute_regime_health(window)
            if not np.isfinite(health) or health < bcfg.regime_min_health:
                return []
        else:
            health = 1.0

        # Market + asset vol
        market_vol = _realized_vol(mkt_proxy, i, bcfg.vol_lookback)
        av = _asset_vol(returns, i, bcfg.vol_lookback)

        # Graph features (adjacency from window)
        W = sim.build_adjacency_from_returns(
            window,
            sectors=sectors,
            sector_only=bool(bcfg.sector_only),
            threshold=float(bcfg.graph_threshold),
        )
        deg, wdeg = _degrees_from_W(W)
        avg_corr = _avg_corr(window)

        # Cap events per bar to control dataset size
        longs = list(long_cand.index[: bcfg.max_events_per_bar])
        shorts = list(short_cand.index[: bcfg.max_events_per_bar])

        # Entry/exit indices
        entry_i = i + bcfg.exec_lag_bars
        max_exit_i = entry_i + bcfg.hold_bars
        entry_dt = idx[entry_i]

        resid_vec = resid_panel.iloc[i]

        def _make_row(ticker: str, side: int) -> Optional[Dict]:
            if ticker not in prices.columns:
                return None
            ep = prices.iloc[entry_i][ticker]
            if pd.isna(ep) or ep <= 0:
                return None

            exit_i = max_exit_i
            exit_reason = "TIME"
            best_price = float(ep)
            for j in range(entry_i + 1, max_exit_i + 1):
                pj = prices.iloc[j][ticker]
                if pd.isna(pj) or pj <= 0:
                    continue

                if side == 1:
                    if pj > best_price:
                        best_price = float(pj)
                    if trail_stop_pct > 0 and pj <= best_price * (1.0 - trail_stop_pct):
                        exit_i = j
                        exit_reason = "TRAIL"
                        break
                else:
                    if pj < best_price:
                        best_price = float(pj)
                    if trail_stop_pct > 0 and pj >= best_price * (1.0 + trail_stop_pct):
                        exit_i = j
                        exit_reason = "TRAIL"
                        break

                rz_j = resid_z_panel.iloc[j].get(ticker, np.nan)
                if np.isfinite(rz_j) and abs(float(rz_j)) <= exit_z:
                    exit_i = j
                    exit_reason = "MEAN"
                    break

            xp = prices.iloc[exit_i][ticker]
            if pd.isna(xp) or xp <= 0:
                return None

            exit_dt = idx[exit_i]
            bars_held = int(exit_i - entry_i)

            qty = 1  # unit label; sizing is learned later
            gross = (float(xp) - float(ep)) * qty * side
            entry_notional = float(ep) * qty
            exit_notional = float(xp) * qty
            net = gross - _apply_cost(entry_notional, bcfg.cost_bps) - _apply_cost(exit_notional, bcfg.cost_bps)
            win = int(net > 0)

            sector = "Unknown"
            if sectors is not None and ticker in sectors.index:
                sector = str(sectors.loc[ticker])

            rz = float(resid_z.loc[ticker])
            abs_res = abs(float(resid_vec.loc[ticker]))
            rsi_val = float(rsi_df.loc[d, ticker]) if (ticker in rsi_df.columns and d in rsi_df.index) else float("nan")
            spy_above = float(macro.loc[d, "spy_above_200dma"]) if ("spy_above_200dma" in macro.columns and d in macro.index) else float("nan")
            vix_close_val = float(macro.loc[d, "vix_close"]) if ("vix_close" in macro.columns and d in macro.index) else float("nan")

            # High-alpha features
            res_vel = float(resid_vel_panel.iloc[i].get(ticker, np.nan)) if ticker in resid_vel_panel.columns else float("nan")
            deg_d5 = float(degree_delta5_panel.iloc[i].get(ticker, np.nan)) if ticker in degree_delta5_panel.columns else float("nan")
            vix_adj = abs_res / (1.0 + vix_close_val / 20.0) if np.isfinite(vix_close_val) and vix_close_val >= 0 else float("nan")

            return {
                "engine": engine,
                "signal_dt": d,
                "entry_dt": entry_dt,
                "exit_dt": exit_dt,
                "ticker": ticker,
                "side": int(side),
                "label_win": win,
                "pnl_net": float(net),
                "entry_price": float(ep),
                "exit_price": float(xp),
                "bars_held": bars_held,
                "exit_reason": exit_reason,

                # Core features (BrainEngine schema)
                "residual": float(resid_vec.loc[ticker]),
                "residual_z": rz,
                "abs_residual": abs_res,
                "signal_strength": abs(rz),
                "rsi14": rsi_val,
                "spy_above_200dma": spy_above,
                "vix_close": vix_close_val,
                "market_vol": float(market_vol),
                "asset_vol": float(av.get(ticker, np.nan)),
                "health_score": float(health),
                "avg_corr": float(avg_corr),
                "degree": float(deg.get(ticker, 0.0)),
                "wdegree": float(wdeg.get(ticker, 0.0)),
                "sector": str(sector),

                # High-alpha features
                "residual_velocity": res_vel,   # residual_z[t] - residual_z[t-3]
                "degree_delta_5": deg_d5,        # degree[t] - degree[t-5]
                "vix_adj_residual": vix_adj,     # abs_residual / (1 + vix_close/20)
            }

        out: list[Dict] = []
        for t in longs:
            r = _make_row(t, side=+1)
            if r is not None:
                out.append(r)
        for t in shorts:
            r = _make_row(t, side=-1)
            if r is not None:
                out.append(r)
        return out

    indices = list(range(bcfg.lookback_bars - 1, last_i))
    use_parallel = int(event_n_jobs) != 1 and len(indices) > 1
    if use_parallel:
        try:
            from joblib import Parallel, delayed  # type: ignore

            results = Parallel(n_jobs=int(event_n_jobs), prefer=str(event_backend))(
                delayed(_rows_for_bar)(i) for i in indices
            )
        except Exception:
            results = [_rows_for_bar(i) for i in indices]
    else:
        results = [_rows_for_bar(i) for i in indices]

    for chunk in results:
        if not chunk:
            continue
        rows.extend(chunk)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No events generated. Try lowering ENTRY_Z or increasing history.")

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["label_win", "residual", "signal_strength", "market_vol", "asset_vol"], inplace=True)
    df.sort_values(["entry_dt", "ticker"], inplace=True)

    df.to_csv(out_path, index=False)
    print(f"[dataset_builder_v2] wrote {len(df):,} rows -> {out_path}")
    return out_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="default", help="Engine name (cfg.ENGINES key).")
    args = p.parse_args()
    build_events_for_engine(engine=str(args.engine))


if __name__ == "__main__":
    main()
