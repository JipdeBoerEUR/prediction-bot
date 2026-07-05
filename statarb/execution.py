# execution.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest


@dataclass
class ExecReport:
    submitted: int
    skipped: int
    errors: int
    # Per-row outcomes for logging/debugging
    details: List[dict] = field(default_factory=list)


class ExecutionEngine:
    """Thin Alpaca execution layer.

    IMPORTANT design rule:
    - This class does NOT decide BUY/SELL from residuals.
    - Upstream code (main/brain) must provide explicit 'side' and 'Qty'.

    Expected input DataFrame columns:
      - ticker: str
      - side: 'BUY' / 'SELL' (case-insensitive) OR +1/-1
      - Qty: positive int
      - (optional) Skip: bool

    This class will also accept aliases:
      - 'Ticker' -> 'ticker'
      - 'qty' -> 'Qty'
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account_equity(self) -> Optional[float]:
        """Return account equity if available; None on failure."""
        try:
            acct = self.client.get_account()
            eq = getattr(acct, "equity", None)
            if eq is None:
                eq = getattr(acct, "cash", None)
            if eq is None:
                return None
            return float(eq)
        except Exception:
            return None

    def get_buying_power(self) -> Optional[float]:
        """Return available buying power (cash + margin) from Alpaca; None on failure."""
        try:
            acct = self.client.get_account()
            bp = getattr(acct, "buying_power", None)
            if bp is None:
                bp = getattr(acct, "cash", None)
            return float(bp) if bp is not None else None
        except Exception:
            return None

    def get_positions(self) -> Dict[str, float]:
        """Public helper used by main.py to enforce MAX_POSITIONS and avoid duplicate buys.

        Quantities are floats — Alpaca supports fractional shares, and truncating
        to int made sub-share positions read as 0 (invisible, and therefore
        impossible to close through this engine).
        """
        out: Dict[str, float] = {}
        try:
            pos = self.client.get_all_positions()
        except Exception:
            return out
        for p in pos:
            try:
                out[str(p.symbol)] = float(p.qty)
            except Exception:
                continue
        return out

    def get_positions_detailed(self) -> List[dict]:
        """Return live positions with entry/current prices for SL/TP monitoring.

        Each dict: ticker, qty (float, negative = short), avg_entry_price,
        current_price, unrealized_plpc (fraction, signed for direction).
        """
        out: List[dict] = []
        try:
            pos = self.client.get_all_positions()
        except Exception:
            return out
        for p in pos:
            try:
                cur = getattr(p, "current_price", None)
                plpc = getattr(p, "unrealized_plpc", None)
                out.append({
                    "ticker": str(p.symbol),
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(cur) if cur is not None else None,
                    "unrealized_plpc": float(plpc) if plpc is not None else None,
                })
            except Exception:
                continue
        return out

    def _open_orders_symbols(self) -> set[str]:
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.client.get_orders(req)
        except Exception:
            return set()
        syms: set[str] = set()
        for o in orders:
            try:
                syms.add(str(o.symbol))
            except Exception:
                pass
        return syms

    def execute_trades(
        self,
        orders_df: pd.DataFrame,
        *,
        cancel_existing: bool = False,
        skip_if_open_order: bool = True,
        dry_run: bool = False,
        tif: TimeInForce = TimeInForce.DAY,
        allow_short: bool = True,
    ) -> ExecReport:
        if orders_df is None or len(orders_df) == 0:
            print("No orders to execute.")
            return ExecReport(submitted=0, skipped=0, errors=0)

        df = orders_df.copy()

        # Normalize column names
        if "ticker" not in df.columns and "Ticker" in df.columns:
            df["ticker"] = df["Ticker"]
        if "Qty" not in df.columns and "qty" in df.columns:
            df["Qty"] = df["qty"]
        if "Skip" not in df.columns:
            df["Skip"] = False

        required = {"ticker", "side", "Qty"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"orders_df missing required columns: {sorted(missing)}")

        # Optional: cancel all existing open orders (only use if you have a specific reason).
        if cancel_existing:
            try:
                self.client.cancel_orders()
            except Exception as e:
                print(f"Warning: Could not cancel old orders: {e}")

        positions = self.get_positions()
        open_order_symbols = self._open_orders_symbols() if skip_if_open_order else set()

        submitted = 0
        skipped = 0
        errors = 0
        details: List[dict] = []

        print("\n--- EXECUTING TRADES ---")

        for _, row in df.iterrows():
            rec = {"ticker": None, "side": None, "qty": None, "status": None, "reason": None}

            try:
                # Normalize/parse basics early so even skipped rows log side/qty
                ticker = str(row.get("ticker", "")).strip().upper()
                rec["ticker"] = ticker

                # Alpaca supports fractional shares — do NOT truncate to int.
                # The sizing pipeline emits quantities down to 0.001 shares;
                # int() silently dropped every sub-share order.
                qty = round(float(row.get("Qty", 0) or 0), 3)
                rec["qty"] = qty

                # Determine side early
                side_val = row.get("side", None)
                side: Optional[OrderSide] = None
                if isinstance(side_val, (int, float)) and int(side_val) in (1, -1):
                    side = OrderSide.BUY if int(side_val) == 1 else OrderSide.SELL
                else:
                    s = str(side_val).strip().upper()
                    if s in {"BUY", "LONG", "1"}:
                        side = OrderSide.BUY
                    elif s in {"SELL", "SHORT", "-1"}:
                        side = OrderSide.SELL

                rec["side"] = side.name if side is not None else None

                # --- Skip checks (now rec already has ticker/qty/side when possible) ---
                if bool(row.get("Skip", False)):
                    skipped += 1
                    rec["status"] = "skipped"
                    rec["reason"] = "Skip=True"
                    details.append(rec)
                    continue

                if not ticker:
                    skipped += 1
                    rec["status"] = "skipped"
                    rec["reason"] = "empty ticker"
                    details.append(rec)
                    continue

                if qty <= 0:
                    skipped += 1
                    rec["status"] = "skipped"
                    rec["reason"] = "qty<=0"
                    details.append(rec)
                    continue

                if side is None:
                    skipped += 1
                    rec["status"] = "skipped"
                    rec["reason"] = f"invalid side={side_val}"
                    details.append(rec)
                    continue

                if skip_if_open_order and (ticker in open_order_symbols):
                    skipped += 1
                    rec["status"] = "skipped"
                    rec["reason"] = "open order exists"
                    details.append(rec)
                    continue

                # Position sanity checks (fractional-aware, short-aware)
                held_qty = float(positions.get(ticker, 0.0))

                if side == OrderSide.SELL and held_qty > 0:
                    # Closing (part of) a long — never sell more than held,
                    # which would silently flip the position into a short.
                    if qty > held_qty:
                        qty = held_qty
                        rec["qty"] = qty

                elif side == OrderSide.SELL:
                    # No long position → this SELL is a short entry.
                    # (The old code skipped ALL of these as "no position to
                    # sell", so the allow_short flag below was dead code and
                    # short entries could never execute.)
                    if held_qty < 0:
                        skipped += 1
                        rec["status"] = "skipped"
                        rec["reason"] = "already short"
                        details.append(rec)
                        continue
                    if not allow_short:
                        skipped += 1
                        rec["status"] = "skipped"
                        rec["reason"] = "shorting disabled"
                        details.append(rec)
                        continue
                    # Alpaca does not support fractional short selling.
                    qty = float(int(qty))
                    rec["qty"] = qty
                    if qty < 1:
                        skipped += 1
                        rec["status"] = "skipped"
                        rec["reason"] = "fractional short not supported"
                        details.append(rec)
                        continue

                if side == OrderSide.BUY and held_qty > 0:
                    skipped += 1
                    rec["status"] = "skipped"
                    rec["reason"] = "already holding"
                    details.append(rec)
                    continue
                if side == OrderSide.BUY and held_qty < 0:
                    # Buy-to-cover an existing short. Alpaca rejects orders
                    # that would flip a position in one go — cap at short size.
                    if qty > abs(held_qty):
                        qty = abs(held_qty)
                        rec["qty"] = qty

                if dry_run:
                    print(f"DRY_RUN: {side.name} {qty} {ticker}")
                    rec["status"] = "dry_run"
                    details.append(rec)
                    submitted += 1
                    continue

                order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                )
                self.client.submit_order(order_data)
                print(f"SUCCESS: {side.name} {qty} {ticker}")
                submitted += 1
                rec["status"] = "submitted"
                details.append(rec)

                # Prevent duplicates within the same run if df accidentally repeats tickers
                if skip_if_open_order:
                    open_order_symbols.add(ticker)

            except Exception as e:
                errors += 1
                rec["status"] = "error"
                rec["reason"] = str(e)
                details.append(rec)
                print(f"ERROR: Could not execute row for {rec.get('ticker')}. Reason: {e}")

        return ExecReport(submitted=submitted, skipped=skipped, errors=errors, details=details)
