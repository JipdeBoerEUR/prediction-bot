import json
import pandas as pd
import yfinance as yf
from scoring_engine import ScoringEngine
from fundamental_engine import fetch_fundamentals_data
from position_ledger import remove_position

STOP_LOSS_PCT = 0.07    # -7%
TAKE_PROFIT_PCT = 0.15  # +15%


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

    close_rows = []
    for pos in positions:
        ticker = pos.get("ticker", "")
        qty = float(pos.get("qty") or 0.0)
        if not ticker or qty == 0:
            continue

        plpc = pos.get("unrealized_plpc")
        if plpc is None:
            avg = float(pos.get("avg_entry_price") or 0.0)
            cur = pos.get("current_price")
            if not avg or cur is None:
                print(f"[LIVE-MONITOR] {ticker}: no P/L data available — skipped.")
                continue
            raw = (float(cur) - avg) / avg
            plpc = raw if qty > 0 else -raw   # a short profits when price falls
        plpc = float(plpc)

        direction = "LONG" if qty > 0 else "SHORT"
        print(f"[LIVE-MONITOR] {ticker} {direction} {abs(qty)} | P/L {plpc:+.2%}")

        if plpc <= -stop_loss_pct:
            trigger = "STOP-LOSS"
        elif plpc >= take_profit_pct:
            trigger = "TAKE-PROFIT"
        else:
            continue

        side = "SELL" if qty > 0 else "BUY"   # BUY = cover the short
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
