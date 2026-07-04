import json
import pandas as pd
import yfinance as yf
from scoring_engine import ScoringEngine
from fundamental_engine import fetch_fundamentals_data

STOP_LOSS_PCT = 0.07    # -7%
TAKE_PROFIT_PCT = 0.15  # +15%


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
