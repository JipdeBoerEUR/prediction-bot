"""Tests for trade_utils.is_sell_side.

This locks in the fix for the buying-power guard bug where a float ``-1.0``
SELL order was silently misclassified as a BUY (operator-precedence trap),
causing statarb shorts to be wrongly gated by available buying power.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_utils import is_sell_side  # noqa: E402


# ── Sells (the previously-broken cases first) ────────────────────────────────

def test_float_negative_one_is_sell():
    # Regression: the sizing DataFrame stores side as float(-1.0).
    assert is_sell_side(-1.0) is True


def test_int_negative_one_is_sell():
    assert is_sell_side(-1) is True


def test_string_sell_is_sell():
    assert is_sell_side("SELL") is True
    assert is_sell_side("sell") is True
    assert is_sell_side(" Sell ") is True


def test_string_minus_one_is_sell():
    assert is_sell_side("-1") is True
    assert is_sell_side("-1.0") is True


# ── Buys ─────────────────────────────────────────────────────────────────────

def test_float_one_is_buy():
    assert is_sell_side(1.0) is False


def test_int_one_is_buy():
    assert is_sell_side(1) is False


def test_string_buy_is_buy():
    assert is_sell_side("BUY") is False
    assert is_sell_side("buy") is False


def test_string_one_is_buy():
    assert is_sell_side("1") is False


# ── Robustness ───────────────────────────────────────────────────────────────

def test_none_defaults_to_buy():
    # Unknown / missing side should not be treated as a cash-freeing sell.
    assert is_sell_side(None) is False


def test_garbage_string_defaults_to_buy():
    assert is_sell_side("banana") is False
