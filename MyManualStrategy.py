from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime
from typing import Optional, Union
from freqtrade.persistence import Trade
import logging  # Add this line
logger = logging.getLogger(__name__)
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
        "0": 0.04,    # 4% ROI target from start
        "20": 0.02,   # 2% after 20 minutes
        "30": 0.01,   # 1% after 30 minutes
        "40": 0       # 0% after 40 minutes
    }
    stoploss = -0.10   # Stoploss at -10%

    # Allow up to 3 concurrent trades (adjust based on config)
    max_open_trades = 3

    # Use market orders for instant execution on force entries/exits
    order_types = {
        "buy": "market",
        "sell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add various technical indicators to the dataframe for analysis.
        These indicators are used by the AI agent for decision-making.
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

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd
        dataframe["macd_signal"] = macd_signal
        dataframe["macd_hist"] = macd_hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_lower

        # Stochastic Oscillator
        stoch_k, stoch_d = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["stoch_k"] = stoch_k
        dataframe["stoch_d"] = stoch_d

        # Ichimoku Cloud
        tenkan_sen = ta.HT_TRENDLINE(close)
        kijun_sen = ta.HT_TRENDLINE(close)  # Simplified
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = ta.HT_TRENDLINE(close)  # Simplified
        dataframe["ichimoku_tenkan"] = tenkan_sen
        dataframe["ichimoku_kijun"] = kijun_sen
        dataframe["ichimoku_senkou_a"] = senkou_span_a
        dataframe["ichimoku_senkou_b"] = senkou_span_b

        # On-Balance Volume (OBV)
        dataframe["obv"] = ta.OBV(close, volume)

        # Accumulation/Distribution Line (ADL)
        dataframe["adl"] = ta.AD(high, low, close, volume)

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
        Custom stoploss to implement dynamic adjustments.
        """
        if current_profit < 0.02:
            return self.stoploss  # Initial stoploss
        elif current_profit >= 0.02 and current_profit < 0.05:
            return 0.01  # Move stoploss to 1% above open price
        elif current_profit >= 0.05 and current_profit < 0.10:
            return 0.03  # Move stoploss to 3% above open price
        elif current_profit >= 0.10:
            return 0.07  # Move stoploss to 7% above open price
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
            if current_profit > 0.05:
                return "take_profit_5pct"
            elif current_profit > 0 and df['rsi'].iloc[-1] > 70:
                return "take_profit_rsi_overbought"
            elif current_profit > 0 and df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
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
        logger.info("✅ MyManualStrategy loaded. Indicators initialized, awaiting AI decisions...")

