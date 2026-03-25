import yfinance as yf
import pandas as pd
from typing import List, Dict, Any

class APIClient:
    """
    Client for fetching real Data using Yahoo Finance (yfinance).
    This is completely free to use and requires no API key.
    Note: Real insider data can sometimes be sparse on free endpoints.
    """

    @staticmethod
    def get_insider_trades(ticker: str) -> List[Dict[str, Any]]:
        """
        Fetches recent Insider Trades for a given ticker.
        Maps the Yahoo Finance DataFrame into the ScoringEngine dict format.
        """
        try:
            stock = yf.Ticker(ticker)
            transactions = stock.insider_transactions
            
            if transactions is None or transactions.empty:
                return []
                
            formatted_trades = []
            
            # yfinance returns a DataFrame for insider transactions.
            for index, row in transactions.iterrows():
                # We only want to look at positive share additions (buys)
                shares = row.get('Shares', 0)
                if pd.isna(shares) or shares <= 0:
                    continue
                    
                position = str(row.get('Position', 'Unknown'))
                
                # Normalize roles for our ScoringEngine weights
                role = 'Unknown'
                if 'ceo' in position.lower() or 'chief executive' in position.lower():
                    role = 'CEO'
                elif 'cfo' in position.lower() or 'chief financial' in position.lower():
                    role = 'CFO'
                elif 'director' in position.lower() or 'board' in position.lower():
                    role = 'Director'
                elif '10%' in position.lower():
                    role = '10% Owner'
                
                value = row.get('Value', 0.0)
                if pd.isna(value) or value <= 0:
                    continue
                    
                # Format to exactly what ScoringEngine module expects
                formatted_trades.append({
                    'role': role,
                    'value': float(value),
                    'type': 'P', # Assuming 'Purchase' since shares > 0
                    'date': str(row.get('Start Date', index))
                })
                
            return formatted_trades
            
        except Exception as e:
            print(f"Error fetching insider trades for {ticker}: {e}")
            return []

    @staticmethod
    def get_fundamentals(ticker: str) -> Dict[str, float]:
        """
        Fetches fundamentals using yfinance's extensive 'info' dictionary.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'revenue_growth_yoy': info.get('revenueGrowth', 0.0) or 0.0,
                'net_income_ttm': info.get('netIncomeToCommon', 0.0) or 0.0,
                # yfinance returns debtToEquity as a whole number percentage (e.g. 65.4 for 0.654)
                'debt_to_equity': (info.get('debtToEquity', 999.0) or 999.0) / 100.0, 
                'current_ratio': info.get('currentRatio', 0.0) or 0.0,
                'free_cash_flow': info.get('freeCashflow', 0.0) or 0.0
            }
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return {}

    @staticmethod
    def get_valuation(ticker: str) -> Dict[str, float]:
        """
        Fetches valuation multiples using yfinance's 'info' dictionary.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'pe_ratio': info.get('trailingPE', 999.0) or 999.0,
                'pb_ratio': info.get('priceToBook', 999.0) or 999.0,
                'eps_growth_qoq': info.get('earningsQuarterlyGrowth', -1.0) or -1.0
            }
        except Exception as e:
            print(f"Error fetching valuation for {ticker}: {e}")
            return {}

if __name__ == "__main__":
    client = APIClient()
    test_ticker = "AAPL"
    
    print(f"--- Fetching Live Data for {test_ticker} using Yahoo Finance ---")
    
    print("\nInsider Trades (Top 3 open market buys):")
    trades = client.get_insider_trades(test_ticker)
    for t in trades[:3]:
        print(f"  Role: {t['role']}, Value: ${t['value']:,.2f}")
    if not trades:
        print("  No recent insider buys found on free API list.")
        
    print("\nFundamentals:")
    funds = client.get_fundamentals(test_ticker)
    for k, v in funds.items():
        print(f"  {k}: {v}")
        
    print("\nValuation:")
    vals = client.get_valuation(test_ticker)
    for k, v in vals.items():
        print(f"  {k}: {v}")
