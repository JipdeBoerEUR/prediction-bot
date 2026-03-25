import math
from typing import Dict, Union

class RiskManager:
    """
    Core risk management module for position sizing.
    Calculates optimal allocations utilizing a Fractional Kelly Criterion.
    """

    @staticmethod
    def calculate_position_size(
        account_balance: float,
        win_rate: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        current_price: float,
        kelly_fraction: float = 0.25
    ) -> Dict[str, Union[int, float]]:
        """
        Calculates the optimal number of shares and total capital to allocate for a trade.

        Args:
            account_balance: Total base capital available in the account.
            win_rate: The probability of a winning trade (W), expressed as a float (e.g., 0.55 for 55%).
            take_profit_pct: Expected percentage gain if the trade hits its target.
            stop_loss_pct: Expected percentage risk if the trade hits its stop loss.
            current_price: The current price per share of the asset.
            kelly_fraction: The fraction of the theoretical Kelly allocation to use (default: 0.25).

        Returns:
            A dictionary containing:
            - 'shares_to_buy' (int): The number of complete shares to purchase.
            - 'capital_allocated' (float): The actual capital value required for those shares.
        """
        # Defensive constraints: ensure valid inputs that are strictly positive
        if current_price <= 0 or account_balance <= 0:
            return {'shares_to_buy': 0, 'capital_allocated': 0.0}

        # Gracefully handle zero-division errors if stop_loss_pct is passed as 0
        if stop_loss_pct <= 0:
            return {'shares_to_buy': 0, 'capital_allocated': 0.0}

        # Calculate Risk/Reward ratio (R)
        reward_risk_ratio = take_profit_pct / stop_loss_pct

        if reward_risk_ratio <= 0:
            return {'shares_to_buy': 0, 'capital_allocated': 0.0}

        # Apply the Kelly formula: f = W - ((1 - W) / R)
        w = win_rate
        f = w - ((1.0 - w) / reward_risk_ratio)

        # If edge is negative or 0, we don't take the trade
        if f <= 0:
            return {'shares_to_buy': 0, 'capital_allocated': 0.0}

        # Multiply by kelly fraction to get final ideal allocation percentage
        allocation_pct = f * kelly_fraction

        # CRITICAL Constraint: Never allow allocation to exceed 5% of account balance
        max_risk_allocation = 0.05
        allocation_pct = min(allocation_pct, max_risk_allocation)

        # Determine raw capital to allocate given the constraints
        capital_to_allocate = account_balance * allocation_pct

        # Calculate integer shares rounded DOWN
        shares_to_buy = math.floor(capital_to_allocate / current_price)

        # Compute the exact capital practically allocated 
        capital_allocated = float(shares_to_buy * current_price)

        return {
            'shares_to_buy': shares_to_buy,
            'capital_allocated': capital_allocated
        }
