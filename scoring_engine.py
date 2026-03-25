from typing import List, Dict, Union, Any

class ScoringEngine:
    """
    Evaluates a stock based on insider buying, financial health, and valuation metrics.
    Outputs a Trade / No Trade signal along with a breakdown of its reasoning.
    """

    # --- Configuration & Thresholds ---
    
    # Insider Weights
    ROLE_WEIGHTS = {
        'CEO': 3.0,
        'CFO': 3.0,
        'COB': 2.5,  # Chairman of the Board
        'Director': 1.0,
        '10% Owner': 0.5
    }
    MIN_INSIDER_BUY_VALUE = 50000.0  # Minimum $ value to consider a buy significant
    MIN_INSIDER_SCORE = 3.0          # Minimum aggregate insider score to pass (e.g., 1 CEO buy, or 3 Director buys)

    # Fundamental Health
    MIN_REVENUE_GROWTH = 0.05        # Require at least 5% YoY Revenue Growth
    MAX_DEBT_TO_EQUITY = 1.0         # Max Debt allowed relative to Equity
    MIN_CURRENT_RATIO = 1.5          # Can cover short term liabilities 1.5x over

    # Valuation Multiples
    MAX_PE_RATIO = 20.0              # Don't overpay for earnings
    MAX_PB_RATIO = 3.0               # Don't overpay for book value
    MIN_EPS_GROWTH = 0.0             # Require positive QoQ EPS growth

    @classmethod
    def evaluate(
        cls,
        ticker: str,
        insider_data: List[Dict[str, Any]],
        fundamentals: Dict[str, float],
        valuation: Dict[str, float]
    ) -> Dict[str, Union[str, float, List[str]]]:
        """
        Runs the full assessment pipeline for a given stock.
        
        Args:
            ticker: The stock ticker symbol.
            insider_data: A list of dicts with keys ['role', 'value', 'type'] for recent transactions.
            fundamentals: A dict of financial health metrics.
            valuation: A dict of valuation multiples.
            
        Returns:
            Dict containing the final decision ('TRADE' | 'NO TRADE'), total score, and a list of reasons.
        """
        reasons = []
        is_tradeable = True

        # --- 1. Insider Buying Assessment ---
        insider_score = 0.0
        valid_buys_count = 0
        total_buy_value = 0.0

        for trade in insider_data:
            # We only look at Opportunistic Open Market Purchases (Code 'P')
            if trade.get('type') != 'P':
                continue
                
            value = trade.get('value', 0.0)
            if value >= cls.MIN_INSIDER_BUY_VALUE:
                role = trade.get('role', 'Unknown')
                weight = cls.ROLE_WEIGHTS.get(role, 0.5) # Default to minor weight for unknown roles
                
                insider_score += weight
                valid_buys_count += 1
                total_buy_value += value

        if insider_score < cls.MIN_INSIDER_SCORE:
            is_tradeable = False
            reasons.append(f"Failed: Insufficient insider conviction. Score: {insider_score} (Needs {cls.MIN_INSIDER_SCORE}).")
        else:
            reasons.append(f"Passed: Strong insider conviction. {valid_buys_count} significant buys totaling ${total_buy_value:,.2f}.")

        # --- 2. Financial Health Assessment ---
        if fundamentals.get('revenue_growth_yoy', 0) < cls.MIN_REVENUE_GROWTH:
            is_tradeable = False
            reasons.append(f"Failed: Revenue growth YoY ({fundamentals.get('revenue_growth_yoy', 0):.1%}) is below 5%.")
            
        if fundamentals.get('net_income_ttm', 0) <= 0:
            is_tradeable = False
            reasons.append("Failed: Negative Net Income (TTM).")
            
        if fundamentals.get('debt_to_equity', 99) > cls.MAX_DEBT_TO_EQUITY:
            is_tradeable = False
            reasons.append(f"Failed: Debt-to-Equity ({fundamentals.get('debt_to_equity')}) is too high (> {cls.MAX_DEBT_TO_EQUITY}).")
            
        if fundamentals.get('current_ratio', 0) < cls.MIN_CURRENT_RATIO:
            is_tradeable = False
            reasons.append(f"Failed: Current ratio ({fundamentals.get('current_ratio')}) is too low (< {cls.MIN_CURRENT_RATIO}).")
            
        if fundamentals.get('free_cash_flow', 0) <= 0:
            is_tradeable = False
            reasons.append("Failed: Negative Free Cash Flow.")

        # --- 3. Valuation Assessment ---
        pe_ratio = valuation.get('pe_ratio', 999)
        if pe_ratio > cls.MAX_PE_RATIO or pe_ratio <= 0:
            is_tradeable = False
            reasons.append(f"Failed: P/E ratio ({pe_ratio}) is over maximum ({cls.MAX_PE_RATIO}) or negative.")
            
        pb_ratio = valuation.get('pb_ratio', 999)
        if pb_ratio > cls.MAX_PB_RATIO or pb_ratio <= 0:
            is_tradeable = False
            reasons.append(f"Failed: P/B ratio ({pb_ratio}) is over maximum ({cls.MAX_PB_RATIO}) or negative.")
            
        if valuation.get('eps_growth_qoq', -1) <= cls.MIN_EPS_GROWTH:
            is_tradeable = False
            reasons.append("Failed: EPS is not growing Quarter-over-Quarter.")

        # --- 4. Final Decision ---
        signal = "TRADE" if is_tradeable else "NO TRADE"
        if is_tradeable:
            reasons.insert(0, "Passed all fundamental, valuation, and insider screens.")

        return {
            'ticker': ticker,
            'signal': signal,
            'insider_score': insider_score,
            'reasons': reasons
        }

if __name__ == "__main__":
    # --- Mock Data Example ---
    mock_ticker = "AAPL"
    
    # Example: A strong cluster buy by the CEO and a Director.
    # The CFO buy is ignored because it's under the $50k threshold.
    mock_insiders = [
        {'role': 'CEO', 'value': 250000.0, 'type': 'P'},
        {'role': 'Director', 'value': 75000.0, 'type': 'P'},
        {'role': 'CFO', 'value': 10000.0, 'type': 'P'} 
    ]
    
    # Healthy Fundamentals
    mock_fundamentals = {
        'revenue_growth_yoy': 0.08, # 8%
        'net_income_ttm': 95000000.0,
        'debt_to_equity': 0.8,
        'current_ratio': 1.8,
        'free_cash_flow': 45000000.0
    }
    
    # Fair Valuation
    mock_valuation = {
        'pe_ratio': 18.5,
        'pb_ratio': 2.5,
        'eps_growth_qoq': 0.02 # 2%
    }

    print(f"--- Running Scoring Engine for {mock_ticker} ---")
    result = ScoringEngine.evaluate(mock_ticker, mock_insiders, mock_fundamentals, mock_valuation)
    
    print(f"Signal: {result['signal']}")
    print(f"Insider Score: {result['insider_score']}")
    print("Reasons Breakdown:")
    for r in result['reasons']:
        print(f" - {r}")
