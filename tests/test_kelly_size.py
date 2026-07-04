"""Unit tests for kelly_size.RiskManager.calculate_position_size.

Pure math, no heavy dependencies — safe to run in CI without torch/pandas.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly_size import RiskManager  # noqa: E402


def size(**kw):
    base = dict(
        account_balance=100_000.0,
        win_rate=0.60,
        take_profit_pct=0.15,
        stop_loss_pct=0.05,
        current_price=100.0,
        kelly_fraction=0.25,
    )
    base.update(kw)
    return RiskManager.calculate_position_size(**base)


# ── Guard rails: invalid inputs must return a no-trade result ────────────────

def test_zero_price_returns_no_trade():
    r = size(current_price=0.0)
    assert r == {"shares_to_buy": 0, "capital_allocated": 0.0}


def test_negative_balance_returns_no_trade():
    r = size(account_balance=-5000.0)
    assert r["shares_to_buy"] == 0


def test_zero_stop_loss_does_not_divide_by_zero():
    # Must not raise ZeroDivisionError; returns no-trade.
    r = size(stop_loss_pct=0.0)
    assert r["shares_to_buy"] == 0


# ── Kelly edge logic ─────────────────────────────────────────────────────────

def test_negative_edge_returns_no_trade():
    # Low win rate with poor reward:risk => negative Kelly fraction => no trade.
    r = size(win_rate=0.30, take_profit_pct=0.05, stop_loss_pct=0.05)
    assert r["shares_to_buy"] == 0


def test_positive_edge_allocates_capital():
    r = size(win_rate=0.65, take_profit_pct=0.20, stop_loss_pct=0.05)
    assert r["shares_to_buy"] > 0
    assert r["capital_allocated"] > 0


# ── The 5% hard cap ──────────────────────────────────────────────────────────

def test_allocation_never_exceeds_five_percent_cap():
    # A very strong edge would ask for >5% of the account; the cap must bind.
    balance = 100_000.0
    price = 100.0
    r = size(
        account_balance=balance,
        win_rate=0.95,
        take_profit_pct=0.50,
        stop_loss_pct=0.02,
        current_price=price,
    )
    assert r["capital_allocated"] <= balance * 0.05 + price  # +1 share of rounding slack


def test_shares_are_rounded_down():
    # capital cap = 5% * 10_000 = 500; at price 130 => floor(500/130)=3 shares.
    r = size(
        account_balance=10_000.0,
        win_rate=0.99,
        take_profit_pct=1.0,
        stop_loss_pct=0.01,
        current_price=130.0,
    )
    assert r["shares_to_buy"] == math.floor(500.0 / 130.0)
    assert r["capital_allocated"] == r["shares_to_buy"] * 130.0
