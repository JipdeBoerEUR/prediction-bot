import json
import pandas as pd
import yfinance as yf
from scoring_engine import ScoringEngine
from fundamental_engine import fetch_fundamentals_data
from position_ledger import load_ledger, remove_position, update_position
from trade_utils import check_position_exit, trading_days_since

# Legacy fixed thresholds — used for manual portfolio.json holdings and as a
# fallback for live positions recorded before vol-scaled exits existed.
STOP_LOSS_PCT = 0.07    # -7%
TAKE_PROFIT_PCT = 0.15  # +15%

# Narrative momentum decays in days: a topic trade that has gone nowhere by
# day 10 is a dead thesis occupying one of only a few position slots.
TOPIC_MAX_HOLD_DAYS = 10


def monitor_live_positions(
    exec_engine,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    dry_run: bool = False,
) -> None:
    """Enforce stop-loss / take-profit on the LIVE Alpaca positions the bot opened.

    evaluate_portfolio() below only watches the static portfolio.json (manual
    Trading 212 holdings) — it never sees the positions the bot itself opens
    through ExecutionEngine. Without this function, bot-opened positions had
    no automated downside protection at all.

    Handles shorts too: a short's loss is a rising price, so the exit order is
    a BUY-to-cover. Uses Alpaca's own unrealized_plpc (already signed for
    direction) when available.
    """
    if exec_engine is None:
        print("[LIVE-MONITOR] No execution engine — skipped.")
        return

    try:
        positions = exec_engine.get_positions_detailed()
    except Exception as e:
        print(f"[LIVE-MONITOR] Could not fetch live positions: {e}")
        return

    if not positions:
        print("[LIVE-MONITOR] No live positions to monitor.")
        return

    ledger = load_ledger()

    close_rows = []
    for pos in positions:
        ticker = pos.get("ticker", "")
        qty = float(pos.get("qty") or 0.0)
        if not ticker or qty == 0:
            continue

        side_int  = 1 if qty > 0 else -1
        direction = "LONG" if qty > 0 else "SHORT"
        meta      = ledger.get(ticker) if isinstance(ledger.get(ticker), dict) else {}

        # ── Time exit: narrative-momentum positions decay, don't linger ──────
        opened_at = meta.get("opened_at")
        if meta.get("source") == "topic" and opened_at:
            held = trading_days_since(str(opened_at))
            if held >= TOPIC_MAX_HOLD_DAYS:
                side = "SELL" if qty > 0 else "BUY"
                print(f"[LIVE-MONITOR] {ticker} TIME-EXIT — topic position held "
                      f"{held} trading days (max {TOPIC_MAX_HOLD_DAYS}) → {side} {abs(qty)}.")
                close_rows.append({"ticker": ticker, "side": side, "Qty": abs(qty)})
                continue

        entry   = float(pos.get("avg_entry_price") or meta.get("entry_price") or 0.0)
        current = pos.get("current_price")

        if current is not None and entry > 0:
            # ── Vol-scaled stop + trailing stop (parameters set at entry) ────
            # Positions recorded before vol-scaled exits (or with unknown
            # sigma) fall back to the legacy fixed stop and fixed take-profit.
            stop_pct  = meta.get("stop_pct") or stop_loss_pct
            trail_pct = meta.get("trail_pct")
            fixed_tp  = None if trail_pct else take_profit_pct

            reason, new_extreme = check_position_exit(
                side_int, entry, float(current), meta.get("extreme_price"),
                stop_pct, trail_pct, fixed_tp,
            )

            # Persist the high/low-water mark so the trailing stop survives
            # restarts (Alpaca doesn't track it for us).
            if trail_pct and new_extreme != meta.get("extreme_price"):
                try:
                    update_position(ticker, extreme_price=new_extreme)
                except Exception:
                    pass

            pl = (float(current) - entry) / entry * side_int
            trail_txt = f", trail {trail_pct:.1%} from {new_extreme:.2f}" if trail_pct else ""
            print(f"[LIVE-MONITOR] {ticker} {direction} {abs(qty)} | P/L {pl:+.2%} "
                  f"(stop {stop_pct:.1%}{trail_txt})")

            if reason is None:
                continue
            side = "SELL" if qty > 0 else "BUY"   # BUY = cover the short
            print(f"[LIVE-MONITOR] {ticker} {reason} TRIGGERED at {pl:+.2%} "
                  f"→ {side} {abs(qty)} at market.")
            close_rows.append({"ticker": ticker, "side": side, "Qty": abs(qty)})
            continue

        # ── Fallback: no usable price — use Alpaca's own P/L% + fixed rules ──
        plpc = pos.get("unrealized_plpc")
        if plpc is None:
            print(f"[LIVE-MONITOR] {ticker}: no P/L data available — skipped.")
            continue
        plpc = float(plpc)
        print(f"[LIVE-MONITOR] {ticker} {direction} {abs(qty)} | P/L {plpc:+.2%} (legacy rules)")

        if plpc <= -stop_loss_pct:
            trigger = "STOP-LOSS"
        elif plpc >= take_profit_pct:
            trigger = "TAKE-PROFIT"
        else:
            continue

        side = "SELL" if qty > 0 else "BUY"
        print(f"[LIVE-MONITOR] {ticker} {trigger} TRIGGERED at {plpc:+.2%} "
              f"→ {side} {abs(qty)} at market.")
        close_rows.append({"ticker": ticker, "side": side, "Qty": abs(qty)})

    if not close_rows:
        return

    try:
        report = exec_engine.execute_trades(pd.DataFrame(close_rows), dry_run=dry_run)
        print(f"[LIVE-MONITOR] Exit orders: submitted={report.submitted} | "
              f"skipped={report.skipped} | errors={report.errors}")
        for rec in report.details:
            if rec.get("status") in {"submitted", "dry_run"}:
                try:
                    remove_position(rec.get("ticker", ""))
                except Exception:
                    pass
    except Exception as e:
        print(f"[LIVE-MONITOR] Execution failed: {e}")


def evaluate_portfolio(exec_engine=None, dry_run=False) -> None:
    print("\n=== EVALUATING CURRENT TRADING 212 PORTFOLIO HOLDINGS ===")

    try:
        with open("portfolio.json", "r") as f:
            portfolio = json.load(f)
    except FileNotFoundError:
        print("portfolio.json not found! Please create it to use the monitor.")
        return

    holdings = portfolio.get("holdings", [])

    if not holdings:
        print("No active holdings found in portfolio.json.")
        return

    for position in holdings:
        ticker = position['ticker']
        avg_cost = position['average_cost']
        shares = position['shares']

        print(f"\n--- Analyzing Holding: {ticker} ---")

        try:
            stock = yf.Ticker(ticker)
            current_price = float(stock.fast_info['lastPrice'])
        except Exception as e:
            print(f"Failed to fetch real-time price for {ticker}: {e}")
            continue

        profit_loss = (current_price - avg_cost) / avg_cost
        print(
            f"Position: {shares} shares | Avg Cost: ${avg_cost:.2f} | "
            f"Current Price: ${current_price:.2f} | Return: {profit_loss:.2%}"
        )

        # --- Stop-loss / Take-profit enforcement ---
        triggered = False

        if profit_loss <= -STOP_LOSS_PCT:
            print(
                f"[{ticker}] STOP-LOSS TRIGGERED: return {profit_loss:.2%} breached "
                f"-{STOP_LOSS_PCT:.0%} threshold. Submitting SELL."
            )
            triggered = True

        elif profit_loss >= TAKE_PROFIT_PCT:
            print(
                f"[{ticker}] TAKE-PROFIT TRIGGERED: return {profit_loss:.2%} reached "
                f"+{TAKE_PROFIT_PCT:.0%} threshold. Submitting SELL."
            )
            triggered = True

        if triggered:
            if exec_engine is not None:
                order_df = pd.DataFrame([{
                    "ticker": ticker,
                    "side": "SELL",
                    "Qty": shares,
                }])
                exec_engine.execute_trades(order_df, dry_run=dry_run)
            else:
                print(
                    f"  [DRY-RUN / NO ENGINE] Would sell {shares} shares of {ticker} at market."
                )
            # Position is being closed — skip fundamentals check
            continue

        # --- Fundamentals check (soft signal — warn only, never auto-sell) ---
        fund_data = fetch_fundamentals_data(ticker)
        if not fund_data:
            print(f"[{ticker}] Could not fetch fundamentals. Maintaining hold.")
            continue

        fundamentals, valuation = fund_data

        scoring_result = ScoringEngine.evaluate(ticker, [], fundamentals, valuation)

        fundamental_failures = [
            r for r in scoring_result['reasons']
            if "Failed:" in r and "insider conviction" not in r
        ]

        if fundamental_failures:
            print(f"[WARN] {ticker} fundamentals have deteriorated below strict rules!")
            for r in fundamental_failures:
                print(f"  - {r}")
            print("  => Consider closing the position manually. No auto-sell on fundamentals alone.")
        else:
            print(f"[{ticker}] Fundamentals remain completely stable across all 8 metrics. HOLD position.")


if __name__ == "__main__":
    evaluate_portfolio()
