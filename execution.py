"""
The "Hands" - Executes Paper Trades.
Goal: Connect the final logic to a broker API (like Alpaca) strictly for paper-trading.
"""
import os

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

# Ensure these match your Alpaca PAPER TRADING keys
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "your_paper_api_key")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "your_paper_secret_key")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

def execute_trade(ticker: str, shares: int):
    print("--- Paper Trading Execution ---")
    print(f"Attempting to buy {shares} shares of {ticker}...")
    
    if not tradeapi:
        print("Warning: alpaca_trade_api is not installed. Run `pip install alpaca-trade-api`")
        return
        
    if ALPACA_API_KEY == "your_paper_api_key":
        print("Notice: Please securely set your Alpaca Paper Trading API Keys in the environment to execute real API trades.")
        return
        
    try:
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
        
        # Submit a market order
        order = api.submit_order(
            symbol=ticker,
            qty=shares,
            side='buy',
            type='market',
            time_in_force='gtc' # Good till cancel
        )
        
        print(f"Success! Order placed: {order.id}")
        print(f"Order status: {order.status}")
        
    except Exception as e:
        print(f"Failed to execute trade for {ticker}: {e}")

if __name__ == "__main__":
    execute_trade("AAPL", 10)
