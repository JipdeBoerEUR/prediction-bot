# trade_utils.py — small, dependency-free helpers used by the orchestrator.
#
# Kept free of heavy imports (no torch / pandas / yfinance) so the pure trading
# logic can be unit-tested quickly and in CI without the full ML stack.
#
# The exit logic here (vol_scaled_stop_pct / check_position_exit) is shared by
# BOTH the live position monitor and the walk-forward backtester, so backtest
# results exercise the exact code path production uses.

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional, Tuple


def is_sell_side(raw_side: object) -> bool:
    """Return True if an order's ``side`` represents a SELL.

    The ``side`` field can arrive in several shapes depending on the code path:
      * a float  (``-1.0`` / ``1.0``) — from the sizing DataFrame
      * an int   (``-1`` / ``1``)
      * a string (``"SELL"`` / ``"BUY"`` / ``"-1"`` / ``"1"``)

    Numeric values are sells when negative; strings are matched case-insensitively.
    This deliberately avoids the operator-precedence trap of
    ``x in {...} or int(x) == -1 if x.isdigit() else False`` which silently
    misclassified ``-1.0`` (a float) as a BUY.
    """
    try:
        return float(raw_side) < 0
    except (TypeError, ValueError):
        return str(raw_side).strip().upper() == "SELL"


def vol_scaled_stop_pct(
    sigma_daily: Optional[float],
    k: float = 3.0,
    floor: float = 0.04,
    cap: float = 0.15,
) -> float:
    """Stop distance as a fraction of entry price, scaled to the asset's volatility.

    A fixed -7% stop is ~3 sigma of daily moves for a staples stock but ~1 sigma
    for a high-beta chip name — so volatile names get stopped out on noise while
    quiet names give back far more than their signal justifies. Scaling the stop
    to k x sigma_daily (clamped to [floor, cap]) normalizes exit behavior across
    the universe.

    sigma_daily is the per-bar (daily) return standard deviation, e.g. 0.02 for
    a 2%-a-day mover. Invalid sigma (None / NaN / <= 0) falls back to `floor`.
    """
    if sigma_daily is None or not math.isfinite(sigma_daily) or sigma_daily <= 0:
        return floor
    return min(cap, max(floor, k * float(sigma_daily)))


def check_position_exit(
    side: int,
    entry_price: float,
    current_price: float,
    extreme_price: Optional[float],
    stop_pct: float,
    trail_pct: Optional[float] = None,
    fixed_tp_pct: Optional[float] = None,
) -> Tuple[Optional[str], float]:
    """Pure exit decision for one open position. Returns (reason | None, new_extreme).

    Parameters
    ----------
    side          : +1 long, -1 short.
    entry_price   : average fill price.
    current_price : latest mark.
    extreme_price : best price seen since entry (high-water mark for longs,
                    low-water mark for shorts). None on the first check.
    stop_pct      : hard stop distance from entry (fraction, e.g. 0.06).
    trail_pct     : trailing distance from the extreme. None disables trailing.
    fixed_tp_pct  : legacy fixed take-profit. Normally None when trailing is on —
                    a trailing stop lets winners run instead of capping them.

    Semantics
    ---------
    * The hard stop is anchored at entry; the trailing stop is anchored at the
      extreme and only activates once the position is in profit (extreme beyond
      entry), so trailing can never fire before the entry stop would.
    * When both levels are live the tighter (more protective) one wins, and the
      reason reported reflects which one actually triggered.
    """
    new_extreme = float(extreme_price) if extreme_price is not None else float(entry_price)

    if (
        side not in (1, -1)
        or not math.isfinite(entry_price) or entry_price <= 0
        or not math.isfinite(current_price) or current_price <= 0
        or not math.isfinite(stop_pct) or stop_pct <= 0
    ):
        return None, new_extreme

    if side == 1:
        new_extreme = max(new_extreme, current_price)
        stop_level = entry_price * (1.0 - stop_pct)
        effective, trailing_active = stop_level, False
        if trail_pct is not None and trail_pct > 0 and new_extreme > entry_price:
            trail_level = new_extreme * (1.0 - trail_pct)
            if trail_level > stop_level:
                effective, trailing_active = trail_level, True
        if current_price <= effective:
            return ("TRAILING_STOP" if trailing_active else "STOP_LOSS"), new_extreme
        if fixed_tp_pct is not None and current_price >= entry_price * (1.0 + fixed_tp_pct):
            return "TAKE_PROFIT", new_extreme
        return None, new_extreme

    # side == -1 (short): profit when price falls; loss (stop) when it rises.
    new_extreme = min(new_extreme, current_price)
    stop_level = entry_price * (1.0 + stop_pct)
    effective, trailing_active = stop_level, False
    if trail_pct is not None and trail_pct > 0 and new_extreme < entry_price:
        trail_level = new_extreme * (1.0 + trail_pct)
        if trail_level < stop_level:
            effective, trailing_active = trail_level, True
    if current_price >= effective:
        return ("TRAILING_STOP" if trailing_active else "STOP_LOSS"), new_extreme
    if fixed_tp_pct is not None and current_price <= entry_price * (1.0 - fixed_tp_pct):
        return "TAKE_PROFIT", new_extreme
    return None, new_extreme


def trading_days_since(opened_at_iso: str, today: Optional[date] = None) -> int:
    """Count Mon-Fri days strictly after `opened_at_iso` (YYYY-MM-DD) up to today.

    Same-day = 0. Weekends are excluded; market holidays are NOT (a one-day
    overestimate a few times a year is acceptable for a decay-based time stop).
    Unparseable input returns 0 (never force an exit on bad data).
    """
    try:
        opened = datetime.strptime(opened_at_iso.strip()[:10], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return 0
    end = today or date.today()
    if end <= opened:
        return 0
    days = 0
    cur = opened
    while cur < end:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            days += 1
    return days
