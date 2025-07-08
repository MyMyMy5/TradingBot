# --- SharpeHyperOptLoss.py ---
# Located at: ~/user_data/hyperopts/SharpeHyperOptLoss.py

from freqtrade.optimize.hyperopt import IHyperOptLoss
from pandas import DataFrame

class SharpeHyperOptLoss(IHyperOptLoss):
    """
    Defines a custom loss function for hyperopt.
    
    This loss function optimizes for the Sharpe Ratio, aiming to maximize
    risk-adjusted returns. A higher Sharpe Ratio is better.
    
    The Freqtrade hyperopt engine attempts to minimize the loss function,
    so we return the negative Sharpe Ratio.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, **kwargs) -> float:
        """
        Objective function, returns the loss.
        """
        total_profit = results['profit_abs'].sum()
        sharpe_ratio = results.get('sharpe_ratio', 0.0)

        # If the strategy made a profit, we use the sharpe ratio as the metric.
        # Otherwise, we use the total profit (which will be negative), scaled
        # to ensure it's penalized more heavily than a poor sharpe ratio.
        if total_profit > 0:
            return -sharpe_ratio
        else:
            # Scale the loss to be worse than any potential sharpe ratio
            return abs(total_profit) + 10
