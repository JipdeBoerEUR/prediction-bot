"""Tests for the volatility-scaled / trailing exit logic in trade_utils.

These functions are shared by the live position monitor AND the walk-forward
backtester, so this file guards the exit behavior of both.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_utils import (  # noqa: E402
    check_position_exit,
    trading_days_since,
    vol_scaled_stop_pct,
)


# ── vol_scaled_stop_pct ──────────────────────────────────────────────────────

def test_stop_scales_with_sigma():
    # 2%-a-day name → 6% stop at k=3
    assert vol_scaled_stop_pct(0.02) == 0.06


def test_stop_floor_binds_for_quiet_names():
    # 1%-a-day name → 3% raw, floored at 4%
    assert vol_scaled_stop_pct(0.01) == 0.04


def test_stop_cap_binds_for_wild_names():
    # 6%-a-day name → 18% raw, capped at 15%
    assert vol_scaled_stop_pct(0.06) == 0.15


def test_invalid_sigma_falls_back_to_floor():
    assert vol_scaled_stop_pct(None) == 0.04
    assert vol_scaled_stop_pct(float("nan")) == 0.04
    assert vol_scaled_stop_pct(-0.02) == 0.04


# ── check_position_exit: longs ───────────────────────────────────────────────

def test_long_hard_stop_triggers():
    reason, _ = check_position_exit(1, 100.0, 93.9, None, stop_pct=0.06)
    assert reason == "STOP_LOSS"


def test_long_no_exit_above_stop():
    reason, extreme = check_position_exit(1, 100.0, 98.0, None, stop_pct=0.06)
    assert reason is None
    assert extreme == 100.0   # extreme never below entry for an unmoved long


def test_long_trailing_locks_in_profit():
    # Ran to 120, trailing 6% → exit at/below 112.8; hard stop (94) is far away.
    reason, _ = check_position_exit(
        1, 100.0, 112.0, extreme_price=120.0, stop_pct=0.06, trail_pct=0.06
    )
    assert reason == "TRAILING_STOP"


def test_long_trailing_inactive_before_profit():
    # Extreme never exceeded entry → trailing must not fire; price above stop.
    reason, _ = check_position_exit(
        1, 100.0, 95.0, extreme_price=100.0, stop_pct=0.06, trail_pct=0.06
    )
    assert reason is None


def test_long_extreme_ratchets_up():
    _, extreme = check_position_exit(
        1, 100.0, 111.0, extreme_price=108.0, stop_pct=0.06, trail_pct=0.06
    )
    assert extreme == 111.0


def test_long_fixed_tp_when_no_trailing():
    reason, _ = check_position_exit(
        1, 100.0, 115.5, None, stop_pct=0.07, trail_pct=None, fixed_tp_pct=0.15
    )
    assert reason == "TAKE_PROFIT"


def test_long_trailing_beats_fixed_tp_semantics():
    # With trailing enabled and no fixed TP, a big winner is NOT capped.
    reason, _ = check_position_exit(
        1, 100.0, 140.0, extreme_price=140.0, stop_pct=0.06, trail_pct=0.06
    )
    assert reason is None   # still riding


# ── check_position_exit: shorts (mirror) ─────────────────────────────────────

def test_short_hard_stop_on_rally():
    reason, _ = check_position_exit(-1, 100.0, 106.5, None, stop_pct=0.06)
    assert reason == "STOP_LOSS"


def test_short_trailing_locks_in_profit():
    # Fell to 80, trailing 6% → cover at/above 84.8.
    reason, _ = check_position_exit(
        -1, 100.0, 85.0, extreme_price=80.0, stop_pct=0.06, trail_pct=0.06
    )
    assert reason == "TRAILING_STOP"


def test_short_extreme_ratchets_down():
    _, extreme = check_position_exit(
        -1, 100.0, 88.0, extreme_price=92.0, stop_pct=0.06, trail_pct=0.06
    )
    assert extreme == 88.0


def test_short_fixed_tp():
    reason, _ = check_position_exit(
        -1, 100.0, 84.0, None, stop_pct=0.07, fixed_tp_pct=0.15
    )
    assert reason == "TAKE_PROFIT"


# ── check_position_exit: guard rails ─────────────────────────────────────────

def test_invalid_inputs_never_exit():
    assert check_position_exit(0, 100.0, 90.0, None, 0.06)[0] is None
    assert check_position_exit(1, 0.0, 90.0, None, 0.06)[0] is None
    assert check_position_exit(1, 100.0, -5.0, None, 0.06)[0] is None
    assert check_position_exit(1, 100.0, 90.0, None, 0.0)[0] is None


# ── trading_days_since ───────────────────────────────────────────────────────

def test_same_day_is_zero():
    assert trading_days_since("2026-07-06", today=date(2026, 7, 6)) == 0


def test_weekdays_counted():
    # Mon 2026-07-06 → Fri 2026-07-10 = 4 trading days after open day
    assert trading_days_since("2026-07-06", today=date(2026, 7, 10)) == 4


def test_weekend_skipped():
    # Fri 2026-07-10 → Mon 2026-07-13: Sat+Sun skipped → 1
    assert trading_days_since("2026-07-10", today=date(2026, 7, 13)) == 1


def test_two_full_weeks():
    assert trading_days_since("2026-06-22", today=date(2026, 7, 6)) == 10


def test_garbage_input_is_zero():
    assert trading_days_since("not-a-date") == 0
    assert trading_days_since("") == 0
