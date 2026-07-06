# trade_utils.py — small, dependency-free helpers used by the orchestrator.
#
# Kept free of heavy imports (no torch / pandas / yfinance) so the pure trading
# logic can be unit-tested quickly and in CI without the full ML stack.

from __future__ import annotations


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
