"""
The "Brain" - Analyzes the fundamentals.
Goal: Connect to a financial data API to pull balance sheet and income statement metrics.
Output: True/False if the company passes the screen.
"""

import yfinance as yf
from typing import Tuple, Dict, Any, Optional

def fetch_fundamentals_data(ticker: str) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Fetches raw fundamental and valuation metrics from Yahoo Finance.
    Returns:
        Tuple[fundamentals_dict, valuation_dict] or None if fetch fails.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Financial Health Metrics
        fundamentals = {
            'revenue_growth_yoy': info.get('revenueGrowth', 0.0) or 0.0,
            'net_income_ttm': info.get('netIncomeToCommon', 0.0) or 0.0,
            'debt_to_equity': (info.get('debtToEquity', 999.0) or 999.0) / 100.0,
            'current_ratio': info.get('currentRatio', 0.0) or 0.0,
            'free_cash_flow': info.get('freeCashflow', 0.0) or 0.0
        }
        
        # Valuation Metrics
        valuation = {
            'pe_ratio': info.get('trailingPE', 999.0) or 999.0,
            'pb_ratio': info.get('priceToBook', 999.0) or 999.0,
            'eps_growth_qoq': info.get('earningsQuarterlyGrowth', -1.0) or -1.0
        }

        return fundamentals, valuation
            
    except Exception as e:
        print(f"[{ticker}] Failed to fetch fundamental data: {e}")
        return None

if __name__ == "__main__":
    res = fetch_fundamentals_data("AAPL")
    if res:
        f, v = res
        print("Fundamentals:", f)
        print("Valuation:", v)
