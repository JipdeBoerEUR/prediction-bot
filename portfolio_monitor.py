import json
import yfinance as yf
from scoring_engine import ScoringEngine
from fundamental_engine import fetch_fundamentals_data
from notifier import send_trade_alert

def evaluate_portfolio():
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
        print(f"Position: {shares} shares | Avg Cost: ${avg_cost:.2f} | Current Price: ${current_price:.2f} | Return: {profit_loss:.2%}")

        fund_data = fetch_fundamentals_data(ticker)
        if not fund_data:
            print(f"[{ticker}] Could not fetch fundamentals. Maintaining hold.")
            continue
            
        fundamentals, valuation = fund_data
        
        # We pass empty insider data to re-evaluate the pure fundamentals
        scoring_result = ScoringEngine.evaluate(ticker, [], fundamentals, valuation)
        
        # It will gracefully fail the insider check (because we pass []), 
        # so we specifically look to see if it failed any fundamental checks.
        fundamental_failures = [r for r in scoring_result['reasons'] if "Failed:" in r and "insider conviction" not in r]
        
        if fundamental_failures:
            print(f"SELL WARNING: {ticker} fundamentals have deteriorated below our strict rules!")
            for r in fundamental_failures:
                print(f"  - {r}")
                
            # Alert user logic:
            # You could add a send_portfolio_alert() here dynamically
            print(f"=> Action: Consider closing the position on Trading 212 at market open.")
            
        else:
            print(f"[{ticker}] Fundamentals remain completely stable across all 8 metrics. HOLD position.")

if __name__ == "__main__":
    evaluate_portfolio()
