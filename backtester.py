"""
The "Detective" - Historical Backtesting Engine.
Evaluates the historical performance of buy signals against the S&P 500 benchmark.
"""
import yfinance as yf
import pandas as pd

def run_backtest(ticker: str, buy_dates: list, hold_period_days: int = 90):
    print(f"\n=== BACKTESTING {ticker} FOR {len(buy_dates)} HISTORICAL SIGNALS ===")
    
    try:
        # Fetch up to 5 years of daily data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        
        if hist.empty:
            print(f"Failed to fetch historical data for {ticker}.")
            return
            
        benchmark = yf.Ticker("^GSPC") # S&P 500 tracking
        bench_hist = benchmark.history(period="5y")
        
        results = []
        
        for date_str in buy_dates:
            try:
                # Synchronize timezone to avoid issues
                trade_date = pd.to_datetime(date_str)
                if hist.index.tz is not None:
                    trade_date = trade_date.tz_localize(hist.index.tz)
                    
                valid_dates = hist[hist.index >= trade_date]
                if valid_dates.empty:
                    continue
                    
                # Snap to the exact entry price at the next available market open
                entry_date = valid_dates.index[0]
                entry_price = valid_dates.iloc[0]['Open']
                
                # Advance N days to simulate the hold period
                exit_target = entry_date + pd.Timedelta(days=hold_period_days)
                exit_dates = hist[hist.index >= exit_target]
                
                if exit_dates.empty:
                    # Not enough days have passed by today's date, close the position now
                    exit_date = hist.index[-1]
                    exit_price = hist.iloc[-1]['Close']
                else:
                    exit_date = exit_dates.index[0]
                    exit_price = exit_dates.iloc[0]['Close']
                    
                # Calculate Absolute Return
                strategy_return = (exit_price - entry_price) / entry_price
                
                # Calculate S&P 500 return for the EXACT same timeframe (The Alpha test)
                bench_entry = bench_hist[bench_hist.index >= entry_date].iloc[0]['Open']
                bench_exit = bench_hist[bench_hist.index >= exit_date].iloc[0]['Close']
                bench_return = (bench_exit - bench_entry) / bench_entry
                
                results.append({
                    'buy_date': entry_date.strftime("%Y-%m-%d"),
                    'exit_date': exit_date.strftime("%Y-%m-%d"),
                    'strategy_return': strategy_return,
                    'sp500_return': bench_return,
                    'alpha': strategy_return - bench_return
                })
                
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                
        # --- Summarize and Print Deep Analytics ---
        if not results:
            print("No valid trades executed in backtest horizon.")
            return
            
        df = pd.DataFrame(results)
        
        win_rate = (df['strategy_return'] > 0).mean()
        avg_return = df['strategy_return'].mean()
        avg_bench_return = df['sp500_return'].mean()
        avg_alpha = df['alpha'].mean()
        
        print(f"Total Evaluated Trades: {len(df)}")
        print(f"Hold Period: {hold_period_days} Days")
        print(f"Historical Win Rate: {win_rate:.1%}")
        print(f"Average Trade Return: {avg_return:.2%} (vs S&P 500: {avg_bench_return:.2%})")
        print(f"Average Alpha Generated: {avg_alpha:.2%}")
        
        if avg_alpha > 0:
            print("\n=> CONCLUSION: Strategy BEATS the market benchmark on these historical signals.")
        else:
            print("\n=> CONCLUSION: Strategy UNDERPERFORMS the market. Consider tightening the fundamental rules.")
            
    except Exception as e:
        print(f"Backtest failed structurally: {e}")

if __name__ == "__main__":
    # Mocking known highly profitable historical dip-buy dates for Apple as a demo
    mock_insider_buy_dates = ["2020-03-23", "2022-10-15", "2023-11-01"]
    
    # Run the simulation assuming we held exactly 180 days after each signal
    run_backtest("AAPL", mock_insider_buy_dates, hold_period_days=180)
