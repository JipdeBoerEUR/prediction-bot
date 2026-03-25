"""
Macro Kill-Switch (Market Regime Filter)
========================================
Protects the portfolio from "catching falling knives" during market panics.
Checks two primary indicators:
1. S&P 500 (^GSPC) vs its 200-day Simple Moving Average.
2. CBOE Volatility Index (^VIX) absolute level.
"""

import yfinance as yf
import pandas as pd
from typing import Tuple

def is_market_crashing() -> Tuple[bool, str]:
    """
    Evaluates the overall market regime.
    
    Returns:
        (bool, str): (True if the market is crashing/bearish, Reason string)
    """
    try:
        # Fetch S&P 500 (SPY proxy or ^GSPC index) and VIX
        print("[MACRO] Checking overall market regime (S&P 500 & VIX)...")
        # Get 1 year of data for S&P 500 to calculate the 200 SMA safely
        sp500_history = yf.download('SPY', period='1y', interval='1d', progress=False)
        vix_history = yf.download('^VIX', period='5d', interval='1d', progress=False)
        
        if sp500_history.empty or vix_history.empty:
            print("[MACRO] Warning: Could not fetch macro data. Defaulting to safe (no crash).")
            return False, "Data unavailable"

        # Calculate S&P 500 metrics
        # Flatten yfinance dataframe if it's MultiIndex
        sp500_close = sp500_history['Close'].squeeze()
        current_sp500 = float(sp500_close.iloc[-1])
        sma_200 = float(sp500_close.rolling(window=200).mean().iloc[-1])
        
        # Calculate VIX metrics
        vix_close = vix_history['Close'].squeeze()
        current_vix = float(vix_close.iloc[-1])
        
        # --- RULE 1: Bear Market Trend (Price < 200 SMA) ---
        if current_sp500 < sma_200:
            reason = f"BEAR MARKET: S&P 500 ({current_sp500:.2f}) is below its 200-day moving average ({sma_200:.2f})."
            return True, reason
            
        # --- RULE 2: Panic Benchmark (VIX > 30) ---
        if current_vix > 30.0:
            reason = f"PANIC SELLING: Volatility Index (VIX) is spiking at {current_vix:.2f} (Threshold: 30.0)."
            return True, reason
            
        # Market is healthy
        return False, f"Market Healthy: SPY > 200SMA ({current_sp500:.2f} > {sma_200:.2f}) and VIX is low ({current_vix:.2f})."
        
    except Exception as e:
        print(f"[MACRO] Error analyzing market regime: {e}")
        # Fail open: if Yahoo Finance is down, don't permanently halt the bot
        return False, f"Error: {e}"

if __name__ == "__main__":
    is_crash, reason = is_market_crashing()
    if is_crash:
        print(f"KILL-SWITCH ACTIVATED: {reason}")
    else:
        print(f"CLEAR TO TRADE: {reason}")
