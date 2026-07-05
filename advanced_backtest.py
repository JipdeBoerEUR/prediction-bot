"""
Advanced Backtesting Engine - Phase 5
Simulates the core logic of the Prediction Bot (Value Drops + Trend Recovery + Stop Loss/Take Profit)
across 5 years of historical data for multiple tickers, and benchmarks it against the S&P 500.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# BOT SIMULATION CONFIG
# ---------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM", "BAC", "TSLA", "META", "NFLX"]
BENCHMARK_TICKER = "^GSPC" # S&P 500
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Kelly Engine equivalents (Risk Limits)
TAKE_PROFIT_PCT = 0.20
STOP_LOSS_PCT = 0.08
HOLD_PERIOD_DAYS = 60 # Max hold time if neither TP/SL hit

def run_deep_backtest():
    print("=========================================================")
    print("=== DEEP BACKTEST: UNIFIED PREDICTION BOT (5-YEAR) ===")
    print(f"=== Period: {START_DATE} to {END_DATE} ===")
    print("=========================================================\n")
    
    # 1. Fetch Benchmark Data
    print(f"[FETCH] Downloading 5-year benchmark data ({BENCHMARK_TICKER})...")
    benchmark_df = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    if benchmark_df.empty:
        print("Failed to download benchmark data. Exiting.")
        return
        
    benchmark_df = benchmark_df['Close']
    
    all_trade_results = []
    
    # 2. Iterate through Tickers
    for ticker in TICKERS:
        print(f"[BACKTEST] Simulating {ticker}...")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df.empty or len(df) < 200:
                 continue
                 
            # Flatten columns if multi-index from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # Compute Technical Mocks (Simulating the preconditions for the bot)
            # The bot requires the stock NOT to be plunging below SMA 200 (unless it's a deep reversion)
            # To simulate Insider Entries, we look for local oversold conditions (e.g., 14-day RSI < 35)
            # combined with fundamental baseline support (price > 100-day SMA).
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            
            # --- SIGNAL GENERATION ---
            # Simulate a strong buy signal: Oversold (RSI < 35) but in a larger uptrend (Close > SMA 100)
            df['Buy_Signal'] = (df['RSI'] < 35) & (df['Close'] > df['SMA_100'])
            
            in_position = False
            entry_price = 0.0
            entry_date = None
            
            for i in range(100, len(df)):
                date = df.index[i]
                current_price = df['Close'].iloc[i]
                
                # Active Position Management
                if in_position:
                    days_held = (date - entry_date).days
                    bar_open  = float(df['Open'].iloc[i])
                    bar_high  = float(df['High'].iloc[i])
                    bar_low   = float(df['Low'].iloc[i])

                    stop_price   = entry_price * (1.0 - STOP_LOSS_PCT)
                    target_price = entry_price * (1.0 + TAKE_PROFIT_PCT)

                    # Intraday-aware exits (checking only closes ignored every
                    # intraday breach and overstated results). Conservative
                    # ordering: if both levels are touched in the same bar,
                    # assume the stop hit first. Gap opens fill at the open,
                    # not at the level.
                    exit_price = None
                    exit_reason = None
                    if bar_low <= stop_price:
                        exit_price = min(bar_open, stop_price)
                        exit_reason = "STOP_LOSS"
                    elif bar_high >= target_price:
                        exit_price = max(bar_open, target_price)  # limit fills at target or better
                        exit_reason = "TAKE_PROFIT"
                    elif days_held >= HOLD_PERIOD_DAYS:
                        exit_price = current_price
                        exit_reason = "TIME_LIMIT"

                    if exit_price is not None:
                        strategy_return = (exit_price - entry_price) / entry_price

                        # Calculate Alpha against S&P 500 in the exact same window
                        bench_entry = benchmark_df[benchmark_df.index <= entry_date].iloc[-1]
                        bench_exit = benchmark_df[benchmark_df.index <= date].iloc[-1]

                        # Ensure we get scalar floats
                        if isinstance(bench_entry, pd.Series): bench_entry = float(bench_entry.iloc[0])
                        if isinstance(bench_exit, pd.Series): bench_exit = float(bench_exit.iloc[0])

                        sp500_return = (bench_exit - bench_entry) / bench_entry
                        alpha = strategy_return - sp500_return

                        all_trade_results.append({
                            'Ticker': ticker,
                            'Entry_Date': entry_date,
                            'Exit_Date': date,
                            'Days_Held': days_held,
                            'Exit_Reason': exit_reason,
                            'Strategy_Return': strategy_return,
                            'SP500_Return': float(sp500_return),
                            'Alpha': float(alpha)
                        })

                        in_position = False
                        entry_price = 0.0
                        entry_date = None

                # Entry Logic — the signal is only known after bar i closes,
                # so enter at the NEXT bar's open. Entering at the very close
                # that generated the signal was look-ahead bias.
                elif df['Buy_Signal'].iloc[i] and i + 1 < len(df):
                    in_position = True
                    entry_price = float(df['Open'].iloc[i + 1])
                    entry_date = df.index[i + 1]
                    
        except Exception as e:
            print(f"[ERROR] Failed to simulate {ticker}: {e}")
            continue

    # 3. Aggregate and Report Results
    if not all_trade_results:
        print("\n[RESULT] No trades generated by the strategy across 5 years.")
        return
        
    results_df = pd.DataFrame(all_trade_results)
    
    total_trades = len(results_df)
    win_rate = (results_df['Strategy_Return'] > 0).mean()
    avg_return = results_df['Strategy_Return'].mean()
    avg_bench = results_df['SP500_Return'].mean()
    avg_alpha = results_df['Alpha'].mean()
    avg_hold = results_df['Days_Held'].mean()
    
    tp_hits = len(results_df[results_df['Exit_Reason'] == 'TAKE_PROFIT'])
    sl_hits = len(results_df[results_df['Exit_Reason'] == 'STOP_LOSS'])
    time_hits = len(results_df[results_df['Exit_Reason'] == 'TIME_LIMIT'])
    
    print("\n=========================================================")
    print("=== BACKTEST REPORT ===")
    print("=========================================================")
    print(f"Total Trades Simulated: {total_trades}")
    print(f"Win Rate:               {win_rate:.2%}")
    print(f"Average Return / Trade: {avg_return:.2%}")
    print(f"S&P 500 Return / Trade: {avg_bench:.2%}")
    print(f"Alpha Generated:        {avg_alpha:.2%}")
    print(f"Average Hold Time:      {avg_hold:.1f} days")
    print("\n--- Exit Rationale Distribution ---")
    print(f"Take Profits Hit: {tp_hits} ({tp_hits/total_trades:.1%})")
    print(f"Stop Losses Hit:  {sl_hits} ({sl_hits/total_trades:.1%})")
    print(f"Time Limits Hit:  {time_hits} ({time_hits/total_trades:.1%})")
    print("=========================================================")
    
    if avg_alpha > 0:
        print("\n=> STRATEGY VALIDATED: Significant positive alpha generated against the market milestone.")
    else:
        print("\n=> STRATEGY RE-EVALUATION NEEDED: Underperforms broad market index holding.")

if __name__ == "__main__":
    run_deep_backtest()
