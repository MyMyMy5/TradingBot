# --- MyHyperOptStrategy.py ---
# Located at: ~/user_data/strategies/MyHyperOptStrategy.py

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from pandas import DataFrame
import talib.abstract as ta

class MyHyperOptStrategy(IStrategy):
    """
    This strategy is designed specifically for hyperparameter optimization.
    It uses optimizable parameters for RSI, ROI, and Stoploss to find the
    most profitable combinations over historical data.
    """

    # --- Hyperparameter Optimization Space ---
    
    # Define the search space for the parameters to be optimized.
    # 'buy' space parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")

    # 'sell' space parameters
    sell_rsi = IntParameter(60, 90, default=70, space="sell")
    
    # 'roi' space parameters
    # Freqtrade will optimize the number of ROI entries, their time, and their value.
    # Example: 1 to 4 ROI entries.
    minimal_roi = {
        "0": 0.99  # Disable ROI by default for optimization space
    }

    # 'stoploss' space parameters
    stoploss = DecimalParameter(-0.35, -0.05, default=-0.10, space="stoploss")

    # --- Strategy Configuration ---
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add technical indicators to the dataframe.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the entry (buy) signal logic based on hyperoptable parameters.
        """
        dataframe.loc[
            (dataframe['rsi'] < self.buy_rsi.value),
            'enter_long'] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the exit (sell) signal logic based on hyperoptable parameters.
        """
        dataframe.loc[
            (dataframe['rsi'] > self.sell_rsi.value),
            'exit_long'] = 1
            
        return dataframe
