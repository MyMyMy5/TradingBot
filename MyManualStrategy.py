from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime
from typing import Optional, Union
from freqtrade.persistence import Trade

class MyManualStrategy(IStrategy):
    """
    Manual strategy that provides technical indicators for the AI agent.
    The AI (via external agent using force entries/exits) makes the trading decisions.
    """

    # Strategy configuration
    timeframe = "5m"                   # Analyzing 5-minute candles
    process_only_new_candles = True    # Process only new candle close data
    startup_candle_count = 50          # Number of candles the strategy needs before producing signals

    # ROI and stoploss for safety (these act as a fallback in case AI does not exit)
    minimal_roi = {
        "0": 0.05,    # 5% ROI target from start
        "240": 0.02,  # 2% ROI after 4 hours
        "1440": 0.01, # 1% after 1 day
        "2880": 0     # 0% (no minimum) after 2 days
    }
    stoploss = -0.03   # Stoploss at -3%

    # Allow up to 2 concurrent trades (one per pair in our whitelist)
    max_open_trades = 2

    # Use market orders for instant execution on force entries/exits
    order_types = {
        "buy": "market",
        "sell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add various technical indicators to the dataframe for analysis.
        (These indicators can be used for logging, debugging, or extended AI prompts.)
        """
        # Shortcut references
        close = dataframe["close"]
        high = dataframe["high"]
        low = dataframe["low"]
        volume = dataframe["volume"]

        # RSI (14-period)
        dataframe["rsi"] = ta.RSI(close, timeperiod=14)

        # Simple Moving Averages
        dataframe["sma_20"] = ta.SMA(close, timeperiod=20)
        dataframe["sma_50"] = ta.SMA(close, timeperiod=50)

        # Exponential Moving Averages
        dataframe["ema_12"] = ta.EMA(close, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(close, timeperiod=26)

        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd
        dataframe["macdsignal"] = macd_signal
        dataframe["macdhist"] = macd_hist

        # Bollinger Bands (20-period SMA with 2 std dev)
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_lower
        # %B (position of price within Bollinger Bands)
        dataframe["bb_percent"] = (close - bb_lower) / (bb_upper - bb_lower)

        # Bollinger Band Width (as a measure of volatility)
        dataframe["bb_width"] = (bb_upper - bb_lower) / bb_middle

        # Volume indicators
        dataframe["volume_sma"] = ta.SMA(volume, timeperiod=20)
        dataframe["volume_ratio"] = volume / dataframe["volume_sma"]

        # Average True Range (volatility)
        dataframe["atr"] = ta.ATR(high, low, close, timeperiod=14)

        # Stochastic Oscillator
        stoch_k, stoch_d = ta.STOCH(high, low, close)
        dataframe["stoch_k"] = stoch_k
        dataframe["stoch_d"] = stoch_d

        # Custom simple trend strength indicator (count of bullish conditions met)
        trend = np.zeros(len(dataframe))
        # Condition 1: Price above short-term MA
        trend += (close > dataframe["sma_20"]).astype(int)
        # Condition 2: Short-term MA above longer-term MA
        trend += (dataframe["sma_20"] > dataframe["sma_50"]).astype(int)
        # Condition 3: RSI above 50 (bullish momentum)
        trend += (dataframe["rsi"] > 50).astype(int)
        # Condition 4: MACD above signal (bullish crossover)
        trend += (dataframe["macd"] > dataframe["macdsignal"]).astype(int)
        dataframe["trend_strength"] = trend  # 0 to 4

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry (buy) signals. Always 0 because AI will decide when to buy.
        """
        dataframe["enter_long"] = 0  # no automatic entries
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit (sell) signals. Always 0 because AI will decide when to sell.
        """
        dataframe["exit_long"] = 0  # no automatic exits
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss to override or complement the base stoploss.
        Here, we simply return the base stoploss value to use -3% consistently.
        """
        return self.stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit logic for safety:
        - Take profit if profit > 5%.
        - Take profit if in profit and RSI > 70.
        - Take profit if in profit and MACD < signal.
        - Cut loss if loss < -2%.
        Returns exit reason if conditions met, otherwise None.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is not None and not df.empty:
            print(f"Dataframe for {pair} has {len(df)} rows")  # Debugging statement
            if current_profit > 0.05:
                return "take_profit_5pct"
            elif current_profit > 0 and df['rsi'].iloc[-1] > 70:
                return "take_profit_rsi_overbought"
            elif current_profit > 0 and df['macd'].iloc[-1] < df['macdsignal'].iloc[-1]:
                return "take_profit_macd_bearish"
            elif current_profit < -0.02:
                return "cut_loss_2pct"
        else:
            # Fallback to percentage-based exits if dataframe is not available
            if current_profit > 0.05:
                return "take_profit_5pct"
            elif current_profit < -0.02:
                return "cut_loss_2pct"
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Confirm trade entries initiated via force entry.
        Always return True to not block AI decisions.
        """
        return True

    def bot_start(self, **kwargs) -> None:
        """
        Logging on startup (optional).
        """
        print("✅ MyManualStrategy loaded. Indicators initialized, awaiting AI decisions...")
