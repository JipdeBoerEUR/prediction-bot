"""Day-by-day trading simulator.

What happens on each trading day t, in this order:

  1. OPEN  — fill exits queued yesterday at today's open price.
  2. OPEN  — fill entries queued yesterday at today's open price.
             Size = `weight` × yesterday's closing equity.
             Stop distance = stop_k × the stock's daily volatility over the
             previous 20 days, clamped to 4%–15% (trade_utils.vol_scaled_stop_pct).
  3. DAY   — check stop-loss / trailing-stop against today's LOW (longs) or
             HIGH (shorts). The stop is tested BEFORE the trailing high-water
             mark is raised with today's high — the pessimistic assumption,
             since we don't know which came first. A gap through the stop
             fills at the (worse) open.
  4. CLOSE — read today's z-scores. Queue an exit if the residual has reverted
             (trade_utils.statarb_signal_exit — same rule as the live bot) or
             the position hit its max holding period.
  5. CLOSE — if the market regime is OK (SPY above its 200-day average AND
             VIX < 30), queue new entries: most-negative z first for longs
             (z <= -entry_z), most-positive first for shorts (z >= +entry_z),
             index members only, up to `max_long` / `max_short` positions.
  6. CLOSE — mark everything to market; charge the daily short-borrow fee.

Costs: every fill pays (commission_bps + slippage_bps) of traded notional;
shorts also pay borrow_bps per year, accrued daily. Cash earns 0% interest
(conservative) and short-sale proceeds earn no rebate.

Walk-forward: `segments` is a list of (start, end, params, z-panel). When the
calendar crosses into the next segment the NEW parameters take over, but open
positions are carried over rather than force-closed — like a real fund that
re-tunes its model once a year.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_utils import check_position_exit, statarb_signal_exit, vol_scaled_stop_pct  # noqa: E402

from .data import MarketData  # noqa: E402


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.0   # per side
    slippage_bps: float = 4.0     # per side: ~half the bid-ask spread + a little impact
    borrow_bps: float = 50.0      # per YEAR on short notional

    @property
    def per_side(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


BASE_COSTS = Costs(1.0, 4.0, 50.0)     # 5 bps/side, 0.5%/yr borrow
STRESS_COSTS = Costs(1.0, 14.0, 200.0)  # 15 bps/side, 2%/yr borrow


@dataclass(frozen=True)
class Params:
    # signal (graph) parameters
    lookback: int = 252
    threshold: float = 0.45
    alpha: float = 0.5
    z_window: int = 60
    # trading parameters
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_hold: int = 10
    stop_k: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    start: int               # first day position (inclusive)
    end: int                 # last day position (inclusive)
    params: Params
    z: np.ndarray            # (end-start+1, N) z-scores for this segment


@dataclass
class _Pos:
    side: int
    qty: float
    entry: float
    opened: int
    stop_pct: float
    extreme: float
    last_px: float
    cost_paid: float
    borrow_paid: float = 0.0
    nodata_days: int = 0


@dataclass
class SimResult:
    equity: pd.Series
    gross_exposure: pd.Series          # sum |position| / equity
    net_exposure: pd.Series            # (long − short) / equity
    traded_notional: pd.Series         # $ traded per day (both sides counted)
    trades: pd.DataFrame
    costs: dict = field(default_factory=dict)


# Structural choices — FIXED, not optimised (see REPORT.md "Design choices").
MAX_LONG = 5
MAX_SHORT = 5
WEIGHT = 0.10          # 10% of equity per position → gross exposure <= 100%
VIX_MAX = 30.0
NODATA_EXIT_DAYS = 5   # a held stock with no price for 5 days is closed at its last price


def simulate(md: MarketData, segments: List[Segment], costs: Costs,
             capital: float = 100_000.0,
             checkpoint: Optional[Callable[[int, pd.Series], None]] = None,
             checkpoint_every: int = 252) -> SimResult:
    first, last = segments[0].start, segments[-1].end
    n = last - first + 1
    tick = md.tickers
    cost_rate = costs.per_side
    borrow_daily = costs.borrow_bps / 1e4 / 252.0

    cash = capital
    equity = capital
    positions: dict[int, _Pos] = {}
    q_entries: list[tuple[int, int]] = []     # (col, side) queued at close → fill next open
    q_exits: dict[int, str] = {}              # col → reason
    eq = np.empty(n)
    gross_e = np.empty(n)
    net_e = np.empty(n)
    traded = np.zeros(n)
    trades: list[dict] = []
    paid = {"trading": 0.0, "borrow": 0.0}

    def close_pos(t: int, col: int, price: float, reason: str) -> None:
        nonlocal cash
        p = positions.pop(col)
        notional = p.qty * price
        c = notional * cost_rate
        cash += p.side * notional - c
        paid["trading"] += c
        traded[t - first] += notional
        gross_pnl = p.side * p.qty * (price - p.entry)
        trades.append({
            "ticker": tick[col], "side": p.side,
            "entry_date": md.dates[p.opened], "exit_date": md.dates[t],
            "days_held": t - p.opened, "reason": reason,
            "entry_px": p.entry, "exit_px": price,
            "ret_gross": p.side * (price / p.entry - 1.0),
            "pnl_net": gross_pnl - p.cost_paid - c - p.borrow_paid,
            "notional": p.qty * p.entry,
        })

    seg_i = 0
    for t in range(first, last + 1):
        while t > segments[seg_i].end:
            seg_i += 1
        seg = segments[seg_i]
        P = seg.params
        o, h, lo, c = md.open[t], md.high[t], md.low[t], md.close[t]

        # 1. queued exits at the open
        for col, reason in list(q_exits.items()):
            if col not in positions:
                q_exits.pop(col)
            elif np.isfinite(o[col]):
                close_pos(t, col, float(o[col]), reason)
                q_exits.pop(col)
            # no open price (halt / gap) → stays queued

        # 2. queued entries at the open
        n_long = sum(1 for p in positions.values() if p.side == 1)
        n_short = len(positions) - n_long
        for col, side in q_entries:
            px = o[col]
            if col in positions or not np.isfinite(px) or px <= 0:
                continue
            if (side == 1 and n_long >= MAX_LONG) or (side == -1 and n_short >= MAX_SHORT):
                continue
            notional = equity * WEIGHT
            qty = notional / px
            hist = md.rets[max(0, t - 20):t, col]
            sigma = float(np.nanstd(hist, ddof=1)) if np.isfinite(hist).sum() >= 10 else None
            stop = vol_scaled_stop_pct(sigma, k=P.stop_k)
            fee = notional * cost_rate
            cash -= side * notional + fee
            paid["trading"] += fee
            traded[t - first] += notional
            positions[col] = _Pos(side, qty, float(px), t, stop, float(px), float(px), fee)
            if side == 1:
                n_long += 1
            else:
                n_short += 1
        q_entries = []

        # 3. intraday stop / trailing stop
        for col, p in list(positions.items()):
            if not (np.isfinite(lo[col]) and np.isfinite(h[col]) and np.isfinite(o[col])):
                continue
            probe = lo[col] if p.side == 1 else h[col]
            reason, _ = check_position_exit(p.side, p.entry, float(probe), p.extreme,
                                            p.stop_pct, p.stop_pct, None)
            if reason in ("STOP_LOSS", "TRAILING_STOP"):
                base = p.extreme if reason == "TRAILING_STOP" else p.entry
                if p.side == 1:
                    fill = min(float(o[col]), base * (1 - p.stop_pct))
                else:
                    fill = max(float(o[col]), base * (1 + p.stop_pct))
                close_pos(t, col, fill, reason)
                q_exits.pop(col, None)
                continue
            p.extreme = max(p.extreme, float(h[col])) if p.side == 1 else min(p.extreme, float(lo[col]))

        # 4. signal exits / max hold / missing data (decided at the close)
        zrow = seg.z[t - seg.start]
        for col, p in list(positions.items()):
            if np.isfinite(c[col]):
                p.last_px, p.nodata_days = float(c[col]), 0
            else:
                p.nodata_days += 1
                if p.nodata_days >= NODATA_EXIT_DAYS:
                    close_pos(t, col, p.last_px, "NO_DATA")
                    q_exits.pop(col, None)
                    continue
            z = zrow[col]
            if statarb_signal_exit(p.side, float(z) if np.isfinite(z) else None, P.exit_z):
                q_exits[col] = "RESIDUAL_REVERTED"
            elif t - p.opened >= P.max_hold:
                q_exits.setdefault(col, "MAX_HOLD")

        # 5. new entries (regime-gated)
        regime_ok = (np.isfinite(md.spy_sma200[t]) and md.spy[t] > md.spy_sma200[t]
                     and not (np.isfinite(md.vix[t]) and md.vix[t] >= VIX_MAX))
        if regime_ok:
            ok = md.member[t] & np.isfinite(zrow) & np.isfinite(c)
            busy = set(positions) | set(q_exits)
            n_long = sum(1 for p in positions.values() if p.side == 1)
            n_short = len(positions) - n_long
            longs = [j for j in np.where(ok & (zrow <= -P.entry_z))[0] if j not in busy]
            shorts = [j for j in np.where(ok & (zrow >= P.entry_z))[0] if j not in busy]
            longs.sort(key=lambda j: zrow[j])
            shorts.sort(key=lambda j: -zrow[j])
            q_entries = ([(j, 1) for j in longs[:max(0, MAX_LONG - n_long)]]
                         + [(j, -1) for j in shorts[:max(0, MAX_SHORT - n_short)]])

        # 6. mark to market + borrow fee
        long_v = short_v = 0.0
        for p in positions.values():
            v = p.qty * p.last_px
            if p.side == 1:
                long_v += v
            else:
                short_v += v
                fee = v * borrow_daily
                cash -= fee
                p.borrow_paid += fee
                paid["borrow"] += fee
        equity = cash + long_v - short_v
        i = t - first
        eq[i] = equity
        gross_e[i] = (long_v + short_v) / equity if equity > 0 else np.nan
        net_e[i] = (long_v - short_v) / equity if equity > 0 else np.nan

        if checkpoint is not None and i > 0 and i % checkpoint_every == 0:
            checkpoint(i // checkpoint_every, pd.Series(eq[:i + 1], index=md.dates[first:t + 1]))

    idx = md.dates[first:last + 1]
    open_at_end = [{"ticker": tick[col], "side": p.side, "entry_date": md.dates[p.opened]}
                   for col, p in positions.items()]
    paid["open_positions_at_end"] = len(open_at_end)
    return SimResult(
        equity=pd.Series(eq, index=idx),
        gross_exposure=pd.Series(gross_e, index=idx),
        net_exposure=pd.Series(net_e, index=idx),
        traded_notional=pd.Series(traded, index=idx),
        trades=pd.DataFrame(trades),
        costs=paid,
    )
