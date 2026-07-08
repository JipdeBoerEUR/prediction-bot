# backtest_walkforward.py
"""
Walk-forward historical simulation of the stat-arb sleeve.
================================================================================

Simulates the ACTUAL production signal path day by day over years of history:

    trailing-window correlation graph  (mirrors graph_engine.build_graph)
      → Laplacian diffusion residuals  (statarb.sim_engine.solve_diffusion_residual
                                        — the SAME solver the live engine uses)
      → residual z-scores              (rolling history recomputed per graph
                                        rebuild, mirroring signal_engine's M4)
      → entries at next bar's OPEN     (no look-ahead), transaction costs
      → exits via trade_utils.check_position_exit — the SAME function the live
        position monitor calls — plus residual-reversal and max-hold exits.

Run `--exits fixed` vs `--exits vol` to A/B the legacy fixed -7%/+15% rules
against the volatility-scaled trailing-stop stack, on identical signals.

Outputs
-------
    BACKTEST_REPORT.md    metrics table: in-sample vs out-of-sample vs SPY
    backtest_report.png   equity curve + drawdown chart

Honest limitations (read before quoting numbers)
------------------------------------------------
* The TOPIC (news momentum) sleeve is NOT simulated — there is no historical
  archive of the RSS headline stream, and synthesizing one would be fiction.
  This harness evaluates the statarb sleeve and the exit machinery only.
* The live graph applies a same-sector edge mask (fetched from yfinance);
  historical sector membership isn't point-in-time available, so the backtest
  uses the unmasked correlation graph.
* Costs are a flat per-side bps haircut; no market impact or borrow fees.
* Daily bars: stops are checked against each bar's High/Low with pessimistic
  (stop-first) ordering and gap-open fills, but intrabar path is unknowable.

Usage
-----
    python backtest_walkforward.py                          # defaults, 2019→today
    python backtest_walkforward.py --exits fixed            # legacy exit A/B leg
    python backtest_walkforward.py --dollar-neutral         # add the short book
    python backtest_walkforward.py --start 2020-01-01 --oos-start 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statarb.sim_engine import solve_diffusion_residual          # noqa: E402
from trade_utils import check_position_exit, vol_scaled_stop_pct  # noqa: E402

# ── Universe ─────────────────────────────────────────────────────────────────
# statarb/config.py is git-ignored (holds API keys), so fall back to a liquid
# S&P subset when it is absent — the backtest must run out of the box.
_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "LRCX",
    "CRM", "ADBE", "NOW", "ORCL", "PANW",
    "V", "MA", "PYPL", "JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "SCHW", "C",
    "JNJ", "UNH", "PFE", "LLY", "MRK", "ABBV", "AMGN", "GILD", "TMO", "DHR", "CVS",
    "WMT", "COST", "TGT", "HD", "LOW", "MCD", "NKE", "SBUX",
    "XOM", "CVX", "COP", "SLB",
    "CAT", "DE", "BA", "GE", "HON", "UPS", "UNP",
]


def load_universe() -> List[str]:
    try:
        import statarb.config as cfg
        tickers = list(cfg.ENGINES["equities"]["tickers"])
        print(f"[UNIVERSE] {len(tickers)} tickers from statarb/config.py")
        return tickers
    except Exception:
        print(f"[UNIVERSE] statarb/config.py unavailable — using built-in "
              f"{len(_FALLBACK_UNIVERSE)}-name liquid universe")
        return list(_FALLBACK_UNIVERSE)


# ── Data layer (cached) ──────────────────────────────────────────────────────

def _fetch_one(ticker: str, start: str, end: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """Daily OHLC (auto-adjusted) with a parquet cache keyed by span."""
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    safe = ticker.replace("^", "_")
    path = os.path.join(cache_dir, f"{safe}_{start}_{end}.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Open", "High", "Low", "Close"]].astype(float)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.to_parquet(path)
        return df
    except Exception as e:
        print(f"[DATA] {ticker}: download failed ({e})")
        return None


def load_data(tickers: List[str], start: str, end: str, cache_dir: str):
    """Return (open_, high, low, close) ticker panels + SPY close + VIX close."""
    frames: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers):
        df = _fetch_one(t, start, end, cache_dir)
        if df is not None and len(df) > 0:
            frames[t] = df
        if (i + 1) % 20 == 0:
            print(f"[DATA] {i + 1}/{len(tickers)} tickers fetched…")

    if not frames:
        raise SystemExit("No price data could be fetched — aborting.")

    close = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()
    # Drop names with poor coverage — they distort the correlation graph.
    coverage = close.notna().mean()
    keep = coverage[coverage >= 0.70].index.tolist()
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        print(f"[DATA] Dropping {len(dropped)} low-coverage tickers: {dropped}")
    close = close[keep]
    open_ = pd.DataFrame({t: frames[t]["Open"] for t in keep}).reindex(close.index)
    high  = pd.DataFrame({t: frames[t]["High"] for t in keep}).reindex(close.index)
    low   = pd.DataFrame({t: frames[t]["Low"]  for t in keep}).reindex(close.index)

    spy = _fetch_one("SPY", start, end, cache_dir)
    vix = _fetch_one("^VIX", start, end, cache_dir)
    spy_close = spy["Close"].reindex(close.index).ffill() if spy is not None else pd.Series(index=close.index, dtype=float)
    vix_close = vix["Close"].reindex(close.index).ffill() if vix is not None else pd.Series(index=close.index, dtype=float)

    print(f"[DATA] Panel ready: {close.shape[1]} tickers × {close.shape[0]} days")
    return open_, high, low, close, spy_close, vix_close


# ── Signal layer (mirrors the live engines) ──────────────────────────────────

def build_adjacency(returns_window: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Mirror graph_engine.build_graph (minus the sector mask — see header)."""
    corr = returns_window.corr()
    # pandas 3 copy-on-write makes .values read-only — zero the diagonal on an
    # owned numpy copy instead of mutating the DataFrame's buffer.
    vals = corr.to_numpy(copy=True)
    np.fill_diagonal(vals, 0.0)
    corr = pd.DataFrame(vals, index=corr.index, columns=corr.columns)
    # Diagonal is 0 and 0 < threshold keeps it 0 through the filter.
    W = corr.where(corr >= threshold, 0.0)
    return (W + W.T) / 2.0


def daily_residuals(returns: pd.DataFrame, W: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Residual x − h for each row of `returns`, using the production solver.

    Per-day valid-ticker subsetting mirrors signal_engine.run_diffusion.
    """
    out = pd.DataFrame(index=returns.index, columns=W.index, dtype=float)
    for dt, row in returns.iterrows():
        valid = row.dropna().index.intersection(W.index)
        if len(valid) < 10:
            continue
        x = row.loc[valid].to_numpy(dtype=float)
        Wv = W.loc[valid, valid].to_numpy(dtype=float)
        try:
            res = solve_diffusion_residual(x, Wv, alpha=alpha)
        except Exception:
            continue
        out.loc[dt, valid] = res
    return out


# ── Portfolio bookkeeping ────────────────────────────────────────────────────

@dataclass
class Position:
    side: int              # +1 long, -1 short
    qty: float
    entry: float
    opened_idx: int
    stop_pct: float
    trail_pct: Optional[float]
    fixed_tp: Optional[float]
    extreme: float = field(default=0.0)

    def __post_init__(self):
        if self.extreme == 0.0:
            self.extreme = self.entry


def run_backtest(a) -> dict:
    universe = load_universe()
    # Warm-up margin before the simulation start: correlation lookback + SMA200.
    dl_start = (pd.Timestamp(a.start) - pd.Timedelta(days=int(max(a.lookback, 200) * 1.8))).strftime("%Y-%m-%d")
    open_, high, low, close, spy, vix = load_data(universe, dl_start, a.end, a.cache_dir)

    rets = close.pct_change()
    spy_sma200 = spy.rolling(200, min_periods=200).mean()
    dates = close.index
    sim_mask = dates >= pd.Timestamp(a.start)
    if not sim_mask.any():
        raise SystemExit(f"No trading days on or after {a.start}.")
    first_sim = int(np.argmax(sim_mask))
    if first_sim < a.lookback + 5:
        raise SystemExit("Not enough warm-up history before --start; widen the span.")

    equity = float(a.capital)
    cash = equity
    positions: Dict[str, Position] = {}
    pending_entries: List[dict] = []   # queued at close t → filled open t+1
    pending_exits: Dict[str, str] = {}  # ticker → reason

    W: Optional[pd.DataFrame] = None
    resid_hist = pd.DataFrame(dtype=float)
    z_now: Dict[str, float] = {}

    equity_curve = pd.Series(index=dates[first_sim:], dtype=float)
    exposure_curve = pd.Series(index=dates[first_sim:], dtype=float)
    trades: List[dict] = []
    fills_notional = 0.0
    cost_rate = a.cost_bps / 10_000.0

    def _fill_exit(t_idx: int, ticker: str, price: float, reason: str):
        nonlocal cash, fills_notional
        pos = positions.pop(ticker)
        notional = pos.qty * price
        cash += pos.side * notional          # long sale adds cash; short cover subtracts
        cash -= notional * cost_rate
        fills_notional += notional
        pnl = (price / pos.entry - 1.0) * pos.side
        trades.append({
            "ticker": ticker, "side": pos.side,
            "entry_date": dates[pos.opened_idx], "exit_date": dates[t_idx],
            "days_held": t_idx - pos.opened_idx, "reason": reason, "pnl_pct": pnl,
        })

    n_days = len(dates)
    for t in range(first_sim, n_days):
        day = dates[t]
        rel_day = t - first_sim

        # ── Graph rebuild (walk-forward: strictly prior data) ────────────────
        if W is None or rel_day % a.rebuild_every == 0:
            window = rets.iloc[max(0, t - a.lookback):t].dropna(axis=1, thresh=int(a.lookback * 0.6))
            if window.shape[1] >= 10:
                W = build_adjacency(window, a.graph_threshold)
                # Recompute trailing residual history under the new graph —
                # mirrors signal_engine's M4 rolling z-score denominator.
                hist_window = rets.iloc[max(0, t - 252):t]
                resid_hist = daily_residuals(hist_window, W, a.alpha)

        if W is None or resid_hist.empty:
            equity_curve.iloc[rel_day] = equity
            exposure_curve.iloc[rel_day] = 0.0
            continue

        open_t, high_t, low_t, close_t = open_.iloc[t], high.iloc[t], low.iloc[t], close.iloc[t]

        # ── 1. Fill queued exits at today's open ─────────────────────────────
        for ticker, reason in list(pending_exits.items()):
            if ticker in positions and np.isfinite(open_t.get(ticker, np.nan)):
                _fill_exit(t, ticker, float(open_t[ticker]), reason)
            pending_exits.pop(ticker, None)

        # ── 2. Fill queued entries at today's open ───────────────────────────
        for order in pending_entries:
            ticker = order["ticker"]
            px = open_t.get(ticker, np.nan)
            if ticker in positions or not np.isfinite(px) or px <= 0:
                continue
            if len(positions) >= a.max_positions:
                break
            notional = equity * a.max_weight
            qty = round(notional / px, 3)
            if qty <= 0:
                continue
            sigma = rets[ticker].iloc[max(0, t - 20):t].std()
            if a.exits == "vol":
                stop_pct = vol_scaled_stop_pct(float(sigma) if np.isfinite(sigma) else None)
                trail_pct: Optional[float] = stop_pct
                fixed_tp: Optional[float] = None
            else:
                stop_pct, trail_pct, fixed_tp = 0.07, None, 0.15
            positions[ticker] = Position(order["side"], qty, float(px), t,
                                         stop_pct, trail_pct, fixed_tp)
            cash -= order["side"] * qty * px
            cash -= qty * px * cost_rate
            fills_notional += qty * px
        pending_entries = []

        # ── 3. Intraday stop / trail / TP checks on today's bar ─────────────
        # Reuses the production exit function: probe with the adverse extreme
        # of the bar (Low for longs, High for shorts) against the PRIOR bar's
        # water mark, then advance the mark with today's favorable extreme.
        for ticker, pos in list(positions.items()):
            lo_v, hi_v = low_t.get(ticker, np.nan), high_t.get(ticker, np.nan)
            op_v = open_t.get(ticker, np.nan)
            if not (np.isfinite(lo_v) and np.isfinite(hi_v) and np.isfinite(op_v)):
                continue
            probe = lo_v if pos.side == 1 else hi_v
            reason, _ = check_position_exit(
                pos.side, pos.entry, float(probe), pos.extreme,
                pos.stop_pct, pos.trail_pct, pos.fixed_tp,
            )
            if reason in ("STOP_LOSS", "TRAILING_STOP"):
                if pos.side == 1:
                    level = pos.entry * (1 - pos.stop_pct)
                    if reason == "TRAILING_STOP":
                        level = pos.extreme * (1 - pos.trail_pct)
                    fill = min(float(op_v), level)     # gap-down opens fill worse
                else:
                    level = pos.entry * (1 + pos.stop_pct)
                    if reason == "TRAILING_STOP":
                        level = pos.extreme * (1 + pos.trail_pct)
                    fill = max(float(op_v), level)
                _fill_exit(t, ticker, fill, reason)
                continue
            if reason == "TAKE_PROFIT":
                level = pos.entry * (1 + pos.fixed_tp) if pos.side == 1 else pos.entry * (1 - pos.fixed_tp)
                fill = max(float(op_v), level) if pos.side == 1 else min(float(op_v), level)
                _fill_exit(t, ticker, fill, reason)
                continue
            # Advance the trailing water mark with today's favorable extreme.
            pos.extreme = max(pos.extreme, float(hi_v)) if pos.side == 1 else min(pos.extreme, float(lo_v))

        # ── 4. Compute today's residual z-scores (signal known at close t) ───
        resid_t = daily_residuals(rets.iloc[t:t + 1], W, a.alpha)
        resid_hist = pd.concat([resid_hist, resid_t]).tail(300)
        mu = resid_hist.rolling(a.z_window, min_periods=20).mean().iloc[-1]
        sd = resid_hist.rolling(a.z_window, min_periods=20).std().iloc[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            z_row = (resid_t.iloc[-1] - mu) / sd
        z_now = z_row.dropna().to_dict()

        # ── 5. Queue signal exits (residual reverted / max-hold) ─────────────
        for ticker, pos in positions.items():
            z = z_now.get(ticker)
            if z is not None:
                if pos.side == 1 and z >= -a.exit_z:
                    pending_exits[ticker] = "RESIDUAL_REVERTED"
                elif pos.side == -1 and z <= a.exit_z:
                    pending_exits[ticker] = "RESIDUAL_REVERTED"
            if t - pos.opened_idx >= a.max_hold and ticker not in pending_exits:
                pending_exits[ticker] = "MAX_HOLD"

        # ── 6. Queue entries (regime-gated, mirrors macro_filter) ────────────
        spy_ok = bool(spy.iloc[t] > spy_sma200.iloc[t]) if np.isfinite(spy_sma200.iloc[t]) else False
        vix_ok = bool(vix.iloc[t] < 30) if np.isfinite(vix.iloc[t]) else True
        if spy_ok and vix_ok:
            ranked = sorted(z_now.items(), key=lambda kv: kv[1])
            slots = a.max_positions - len(positions)
            for ticker, z in ranked:                      # most negative first → longs
                if slots <= 0:
                    break
                if z <= -a.entry_z and ticker not in positions and ticker not in pending_exits:
                    pending_entries.append({"ticker": ticker, "side": 1})
                    slots -= 1
            if a.dollar_neutral:
                for ticker, z in reversed(ranked):        # most positive first → shorts
                    if slots <= 0:
                        break
                    if z >= a.entry_z and ticker not in positions and ticker not in pending_exits:
                        pending_entries.append({"ticker": ticker, "side": -1})
                        slots -= 1

        # ── 7. Mark to market at close t ─────────────────────────────────────
        pos_value = 0.0
        gross = 0.0
        for ticker, pos in positions.items():
            px = close_t.get(ticker, np.nan)
            px = float(px) if np.isfinite(px) else pos.entry
            pos_value += pos.side * pos.qty * px
            gross += pos.qty * px
        equity = cash + pos_value
        equity_curve.iloc[rel_day] = equity
        exposure_curve.iloc[rel_day] = gross / equity if equity > 0 else 0.0

    equity_curve = equity_curve.dropna()
    spy_bench = spy.reindex(equity_curve.index).ffill()
    spy_bench = spy_bench / spy_bench.iloc[0] * a.capital

    return {
        "equity": equity_curve, "spy": spy_bench, "trades": trades,
        "exposure": exposure_curve.reindex(equity_curve.index),
        "fills_notional": fills_notional, "args": a,
    }


# ── Metrics & report ─────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series) -> dict:
    if len(equity) < 3:
        return {}
    r = equity.pct_change().dropna()
    years = len(r) / 252.0
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    downside = r[r < 0].std()
    sortino = (r.mean() / downside * np.sqrt(252)) if downside and downside > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    return {"Total Return": total, "CAGR": cagr, "Ann. Vol": vol,
            "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": dd.min()}


def _fmt(v: float, pct: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:+.1%}" if pct else f"{v:.2f}"


def write_report(result: dict, prefix: str) -> None:
    a = result["args"]
    eq, spy, trades = result["equity"], result["spy"], result["trades"]
    oos = pd.Timestamp(a.oos_start)

    segments = {
        "Full period": (eq, spy),
        f"In-sample (< {a.oos_start})": (eq[eq.index < oos], spy[spy.index < oos]),
        f"Out-of-sample (≥ {a.oos_start})": (eq[eq.index >= oos], spy[spy.index >= oos]),
    }

    rows = ["| Segment | Book | Total | CAGR | Vol | Sharpe | Sortino | MaxDD |",
            "|---|---|---|---|---|---|---|---|"]
    for name, (e_seg, s_seg) in segments.items():
        for label, series in (("strategy", e_seg), ("SPY", s_seg)):
            m = compute_metrics(series.dropna())
            if not m:
                continue
            rows.append(
                f"| {name} | {label} | {_fmt(m['Total Return'])} | {_fmt(m['CAGR'])} "
                f"| {_fmt(m['Ann. Vol'])} | {_fmt(m['Sharpe'], pct=False)} "
                f"| {_fmt(m['Sortino'], pct=False)} | {_fmt(m['Max Drawdown'])} |"
            )

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        wins = tdf[tdf.pnl_pct > 0]
        losses = tdf[tdf.pnl_pct <= 0]
        pf = wins.pnl_pct.sum() / abs(losses.pnl_pct.sum()) if len(losses) and losses.pnl_pct.sum() != 0 else np.nan
        trade_lines = [
            f"- Trades: **{len(tdf)}**  |  Win rate: **{len(wins) / len(tdf):.0%}**  "
            f"|  Avg win: {wins.pnl_pct.mean():+.2%}  |  Avg loss: "
            f"{losses.pnl_pct.mean():+.2%}  |  Profit factor: {pf:.2f}",
            f"- Avg holding period: {tdf.days_held.mean():.1f} trading days",
            "- Exits: " + ", ".join(f"{k} ×{v}" for k, v in tdf.reason.value_counts().items()),
        ]
    else:
        trade_lines = ["- No trades generated — thresholds too strict for this span."]

    avg_exp = result["exposure"].mean()
    years = max(len(eq) / 252.0, 1e-9)
    turnover = result["fills_notional"] / eq.mean() / years

    md = "\n".join([
        "# Walk-Forward Backtest — StatArb Sleeve",
        "",
        f"_Generated by `backtest_walkforward.py`; span {eq.index[0].date()} → "
        f"{eq.index[-1].date()}; exits mode: **{a.exits}**"
        f"{'; dollar-neutral' if a.dollar_neutral else '; long-only'}._",
        "",
        "## Performance",
        "",
        *rows,
        "",
        "## Trades",
        "",
        *trade_lines,
        f"- Avg gross exposure: {avg_exp:.0%} of equity  |  Annual turnover: {turnover:.1f}×",
        "",
        "## Configuration",
        "",
        f"```\nentry_z={a.entry_z} exit_z={a.exit_z} alpha={a.alpha} "
        f"graph_threshold={a.graph_threshold} lookback={a.lookback}d "
        f"rebuild_every={a.rebuild_every}d z_window={a.z_window} "
        f"max_positions={a.max_positions} max_weight={a.max_weight} "
        f"max_hold={a.max_hold}d cost_bps={a.cost_bps} capital={a.capital}\n```",
        "",
        "## Limitations",
        "",
        "- Topic (news momentum) sleeve **not** simulated — no historical headline archive exists.",
        "- No same-sector edge mask (live graph uses one); no borrow fees or market impact.",
        "- Signals fire at close t, fill at open t+1; stops are intraday vs High/Low with",
        "  pessimistic stop-first ordering and gap-open fills.",
        "",
    ])
    # Uppercase only the basename — the prefix may contain a directory path.
    md_path = os.path.join(
        os.path.dirname(prefix), f"{os.path.basename(prefix).upper()}_REPORT.md"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] {md_path} written.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        (eq / eq.iloc[0] * 100).plot(ax=ax1, label=f"StatArb ({a.exits} exits)", lw=1.4)
        (spy / spy.iloc[0] * 100).plot(ax=ax1, label="SPY buy & hold", lw=1.1, alpha=0.75)
        ax1.axvline(pd.Timestamp(a.oos_start), color="gray", ls="--", lw=0.9)
        ax1.text(pd.Timestamp(a.oos_start), ax1.get_ylim()[1], " OOS →",
                 va="top", fontsize=8, color="gray")
        ax1.set_ylabel("Growth of 100")
        ax1.legend(loc="upper left")
        ax1.set_title("Walk-forward backtest — statarb sleeve")

        dd = eq / eq.cummax() - 1.0
        dd.plot(ax=ax2, color="firebrick", lw=1.0)
        ax2.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.25)
        ax2.set_ylabel("Drawdown")
        fig.tight_layout()
        fig.savefig(f"{prefix}_report.png", dpi=140)
        print(f"[REPORT] {prefix}_report.png written.")
    except Exception as e:
        print(f"[REPORT] Chart skipped ({e}).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest of the statarb sleeve.")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--oos-start", default="2024-01-01",
                   help="Out-of-sample split date for reporting")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--entry-z", type=float, default=1.75, dest="entry_z")
    p.add_argument("--exit-z", type=float, default=0.30, dest="exit_z")
    p.add_argument("--alpha", type=float, default=0.80)
    p.add_argument("--graph-threshold", type=float, default=0.45, dest="graph_threshold")
    p.add_argument("--lookback", type=int, default=252, help="Correlation window (days)")
    p.add_argument("--rebuild-every", type=int, default=21, dest="rebuild_every")
    p.add_argument("--z-window", type=int, default=60, dest="z_window")
    p.add_argument("--max-positions", type=int, default=5, dest="max_positions")
    p.add_argument("--max-weight", type=float, default=0.20, dest="max_weight")
    p.add_argument("--max-hold", type=int, default=15, dest="max_hold",
                   help="Max holding period (trading days)")
    p.add_argument("--cost-bps", type=float, default=5.0, dest="cost_bps")
    p.add_argument("--exits", choices=["vol", "fixed"], default="vol",
                   help="vol = σ-scaled trailing stops; fixed = legacy -7%%/+15%%")
    p.add_argument("--dollar-neutral", action="store_true", dest="dollar_neutral")
    p.add_argument("--cache-dir", default="bt_cache", dest="cache_dir")
    p.add_argument("--report-prefix", default="backtest", dest="report_prefix")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n=== WALK-FORWARD BACKTEST | {args.start} → {args.end} "
          f"| exits={args.exits} ===\n")
    result = run_backtest(args)
    write_report(result, args.report_prefix)
    m = compute_metrics(result["equity"])
    if m:
        print("\n".join(f"  {k:<14} {_fmt(v, pct=(k not in ('Sharpe', 'Sortino')))}"
                        for k, v in m.items()))
    print("\nDone.")
