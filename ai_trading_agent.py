#!/usr/bin/env python3
"""
AI Trading Agent for Freqtrade – Enhanced Version with
Automated, Adaptive Hyperparameter Optimization.
"""
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
import subprocess
import re

# Add user_data to path to allow for imports
sys.path.append(os.path.expanduser('~/user_data'))

# External libraries
import ccxt
from freqtrade_client import FtRestClient
from openai import OpenAI
import pandas as pd
import numpy as np
import talib.abstract as ta
from hmmlearn.hmm import GaussianHMM

# Custom Backtester import
try:
    from backtesting.advanced_backtester import AdvancedBacktester, MockAIStrategy
except ImportError:
    print("Could not import AdvancedBacktester. Make sure advanced_backtester.py is in ~/user_data/backtesting/")
    AdvancedBacktester = None
    MockAIStrategy = None
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FREQTRADE_API_URL = "http://127.0.0.1:8080"
FREQTRADE_USERNAME = "Freqtrader"
FREQTRADE_PASSWORD = "SuperSecret1!"

OPENROUTER_API_KEY = "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"

MODEL_PRIMARY = "deepseek/deepseek-chat:free"
MODEL_FALLBACK = "deepseek/deepseek-chat-v3-0324:free"
MAX_TOKENS = 8192
HYPEROPT_INTERVAL_SECONDS = 3600  # Run hyperopt every 1 hour
HYPEROPT_EPOCHS = 50  # Number of combinations to test
HYPEROPT_TIMERANGE_DAYS = 7 # Analyze the last 7 days of data

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- (All interpretation and indicator calculation functions remain the same) ---
# ...
# ---------------------------------------------------------------------------
# Market Regime Detector Class
# ---------------------------------------------------------------------------
class MarketRegimeDetector:
    def __init__(self):
        """Initialize the MarketRegimeDetector with an HMM model."""
        self.regimes = ['trending', 'ranging', 'volatile', 'calm']
        self.hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
        self.is_trained = False

    def train(self, features: pd.DataFrame):
        """
        Train the HMM model on historical features.
        
        Args:
            features (pd.DataFrame): DataFrame with features like returns and volatility.
        """
        if features.empty or len(features) < 50:
            logger.warning("Insufficient data for training MarketRegimeDetector.")
            return
        self.hmm_model.fit(features)
        self.is_trained = True
        logger.info("MarketRegimeDetector trained successfully with %d data points.", len(features))

    def detect_regime(self, current_features: pd.Series) -> str:
        """
        Detect the current market regime based on the latest features.
        
        Args:
            current_features (pd.Series): Series with current returns and volatility.
        
        Returns:
            str: Detected regime ('trending', 'ranging', 'volatile', 'calm').
        
        Raises:
            ValueError: If the model is not trained.
        """
        if not self.is_trained:
            raise ValueError("MarketRegimeDetector must be trained before detecting regimes.")
        current_features_array = current_features.values.reshape(1, -1)
        regime_index = self.hmm_model.predict(current_features_array)[0]
        return self.regimes[regime_index]

def calculate_regime_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate features for regime detection.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' prices.
        window (int): Window size for rolling volatility calculation.
    
    Returns:
        pd.DataFrame: DataFrame with 'returns' and 'volatility' features.
    """
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=window).std()
    return df[['returns', 'volatility']].dropna()

# ---------------------------------------------------------------------------
# Interpretation Functions for Indicators
# ---------------------------------------------------------------------------
def interpret_rsi(rsi: float) -> str:
    """Interpret RSI value into Overbought, Oversold, or Neutral."""
    if rsi > 70:
        return "Overbought"
    elif rsi < 30:
        return "Oversold"
    else:
        return "Neutral"

def interpret_macd(macd: float, macd_signal: float, macd_hist: float, prev_macd_hist: float) -> Dict[str, str]:
    """Interpret MACD signals including crossover and zero-line status."""
    crossover = "No Crossover"
    if macd_hist > 0 and prev_macd_hist < 0:
        crossover = "Bullish Crossover"
    elif macd_hist < 0 and prev_macd_hist > 0:
        crossover = "Bearish Crossover"
    zero_line = "Above Zero" if macd > 0 else "Below Zero"
    return {"crossover": crossover, "zero_line": zero_line}

def interpret_bb(close: float, bb_upper: float, bb_lower: float) -> str:
    """Interpret price position relative to Bollinger Bands."""
    if close > bb_upper:
        return "Above Upper Band"
    elif close < bb_lower:
        return "Below Lower Band"
    else:
        return "Within Bands"

def interpret_stochastic(stoch_k: float) -> str:
    """Interpret Stochastic Oscillator into Overbought, Oversold, or Neutral."""
    if stoch_k > 80:
        return "Overbought"
    elif stoch_k < 20:
        return "Oversold"
    else:
        return "Neutral"

def interpret_ichimoku(close: float, senkou_span_a: float, senkou_span_b: float, tenkan_sen: float, kijun_sen: float) -> str:
    """Simplified interpretation of Ichimoku Cloud signals."""
    if close > senkou_span_a and close > senkou_span_b and tenkan_sen > kijun_sen:
        return "Bullish"
    elif close < senkou_span_a and close < senkou_span_b and tenkan_sen < kijun_sen:
        return "Bearish"
    else:
        return "Neutral"

def interpret_obv(obv: float, prev_obv: float) -> str:
    """Interpret OBV trend: Increasing, Decreasing, or Flat."""
    if obv > prev_obv:
        return "Increasing"
    elif obv < prev_obv:
        return "Decreasing"
    else:
        return "Flat"

def interpret_adx(adx: float) -> str:
    """Interpret ADX into Trending, Ranging, or Unclear."""
    if adx > 25:
        return "Trending"
    elif adx < 20:
        return "Ranging"
    else:
        return "Unclear"

def interpret_volume(volume: float, volume_sma: float) -> str:
    """Interpret volume relative to its SMA: High, Low, or Normal."""
    if volume > volume_sma * 1.5:
        return "High Volume"
    elif volume < volume_sma * 0.5:
        return "Low Volume"
    else:
        return "Normal"

def interpret_supertrend(st_signal: str) -> str:
    """Interpret SuperTrend signal: Buy or Sell."""
    return st_signal

# ---------------------------------------------------------------------------
# Technical Analysis Helper Functions
# ---------------------------------------------------------------------------
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, List[str]]:
    """Calculate SuperTrend indicator (placeholder)."""
    st_line = pd.Series([0.0] * len(df), index=df.index)
    signal = ["Hold"] * len(df)
    return st_line, signal

def calculate_indicators(prices: List[float], volumes: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
    """Calculate technical indicators with raw values and interpreted signals."""
    if len(prices) < 50:
        return {}

    df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})

    # --- TA-Lib Calculations ---
    # These may return numpy arrays, so we will handle them with [-1] indexing.
    rsi = ta.RSI(df['close'], timeperiod=14)
    macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    obv = ta.OBV(df['close'], df['volume'])
    adl = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    volume_sma = ta.SMA(df['volume'], timeperiod=20)
    
    # --- Pandas-based Calculations ---
    # These will return pandas Series, so .iloc is safe here.
    tenkan_sen = (df['high'].rolling(window=9, min_periods=1).max() + df['low'].rolling(window=9, min_periods=1).min()) / 2
    kijun_sen = (df['high'].rolling(window=26, min_periods=1).max() + df['low'].rolling(window=26, min_periods=1).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((df['high'].rolling(window=52, min_periods=1).max() + df['low'].rolling(window=52, min_periods=1).min()) / 2).shift(26)
    st_line, st_signal = calculate_supertrend(df)

    latest_raw = {
        "close": df['close'].iloc[-1],
        "high": df['high'].iloc[-1],
        "low": df['low'].iloc[-1],
        "volume": df['volume'].iloc[-1],
        "RSI": rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else None,
        "MACD_line": macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else None,
        "MACD_signal": macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else None,
        "MACD_hist": macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else None,
        "BB_upper": bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else None,
        "BB_middle": bb_middle[-1] if len(bb_middle) > 0 and not np.isnan(bb_middle[-1]) else None,
        "BB_lower": bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else None,
        "Stoch_K": stoch_k[-1] if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]) else None,
        "Stoch_D": stoch_d[-1] if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else None,
        "Ichimoku_Tenkan": tenkan_sen.iloc[-1] if len(tenkan_sen) > 0 else None,
        "Ichimoku_Kijun": kijun_sen.iloc[-1] if len(kijun_sen) > 0 else None,
        "Ichimoku_Senkou_A": senkou_span_a.iloc[-1] if len(senkou_span_a) > 0 and not pd.isna(senkou_span_a.iloc[-1]) else None,
        "Ichimoku_Senkou_B": senkou_span_b.iloc[-1] if len(senkou_span_b) > 0 and not pd.isna(senkou_span_b.iloc[-1]) else None,
        "OBV": obv[-1] if len(obv) > 0 and not np.isnan(obv[-1]) else None,
        "ADL": adl[-1] if len(adl) > 0 and not np.isnan(adl[-1]) else None,
        "ATR": atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else None,
        "ADX": adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else None,
        "Volume_SMA": volume_sma[-1] if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) else None,
        "SuperTrend": st_line.iloc[-1] if len(st_line) > 0 else None,
    }

    signals = {}
    if not np.isnan(rsi[-1]):
        signals["RSI_state"] = interpret_rsi(rsi[-1])
    if len(macd_hist) >= 2 and not np.isnan(macd_hist[-1]):
        prev_macd_hist = macd_hist[-2]
        signals["MACD"] = interpret_macd(macd[-1], macd_signal[-1], macd_hist[-1], prev_macd_hist)
    if not np.isnan(bb_upper[-1]):
        signals["BB_position"] = interpret_bb(df['close'].iloc[-1], bb_upper[-1], bb_lower[-1])
    if not np.isnan(stoch_k[-1]):
        signals["Stoch_state"] = interpret_stochastic(stoch_k[-1])
    if not pd.isna(senkou_span_a.iloc[-1]):
        signals["Ichimoku_signal"] = interpret_ichimoku(df['close'].iloc[-1], senkou_span_a.iloc[-1], senkou_span_b.iloc[-1], tenkan_sen.iloc[-1], kijun_sen.iloc[-1])
    if len(obv) >= 5 and not np.isnan(obv[-1]):
        prev_obv = obv[-5]
        signals["OBV_trend"] = interpret_obv(obv[-1], prev_obv)
    if len(adl) >= 5 and not np.isnan(adl[-1]):
        prev_adl = adl[-5]
        signals["ADL_trend"] = interpret_obv(adl[-1], prev_adl)
    if len(atr) >= 50 and not np.isnan(atr[-1]):
        avg_atr = np.nanmean(atr[-50:])
        if atr[-1] > 1.5 * avg_atr:
            signals["ATR_volatility"] = "High Volatility"
        elif atr[-1] < 0.5 * avg_atr:
            signals["ATR_volatility"] = "Low Volatility"
        else:
            signals["ATR_volatility"] = "Normal Volatility"
    if not np.isnan(adx[-1]):
        signals["ADX_regime"] = interpret_adx(adx[-1])
    if not np.isnan(volume_sma[-1]):
        signals["Volume_signal"] = interpret_volume(df['volume'].iloc[-1], volume_sma[-1])
    if len(st_signal) > 0:
        signals["SuperTrend_signal"] = interpret_supertrend(st_signal[-1])

    return {"raw": latest_raw, "signals": signals}


class AITradingAgent:
    def __init__(self):
        """Initialize the AI trading agent."""
        self.ft_client = FtRestClient(FREQTRADE_API_URL, FREQTRADE_USERNAME, FREQTRADE_PASSWORD)
        self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.exchange = ccxt.kraken()
        self.exchange.load_markets()
        self.regime_detector = MarketRegimeDetector()
        
        self.backtest_summaries = {}
        self.hyperopt_results = {}
        self.last_hyperopt_time = 0
        
        # ... (rest of __init__ is the same)
        try:
            config = self.ft_client._call("GET", "show_config")
            self.max_open_trades = config.get("max_open_trades", 1)
        except Exception:
            self.max_open_trades = 1

        self.trading_pairs = self._get_trading_pairs()
        self.timeframes = ['5m', '15m', '1h']
        logger.info("AI Trading Agent initialized.")

    def _get_trading_pairs(self) -> List[str]:
        """Fetches the whitelist from Freqtrade."""
        for attempt in range(10):
            try:
                whitelist_response = self.ft_client._call("GET", "whitelist")
                pairs = whitelist_response.get("whitelist", [])
                if pairs:
                    valid_pairs = [p for p in pairs if p in self.exchange.markets]
                    logger.info(f"Retrieved {len(valid_pairs)} valid trading pairs.")
                    return valid_pairs
            except Exception as e:
                logger.warning(f"Failed to get whitelist: {e}")
            time.sleep(5)
        logger.error("Failed to get whitelist. Using defaults.")
        return ["BTC/USDT", "ETH/USDT"]

    def run_automated_hyperopt(self):
        """
        Runs a lightweight hyperopt process automatically to find optimal
        parameters for the current market conditions.
        """
        logger.info("🚀 Starting automated hyperparameter optimization...")
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=HYPEROPT_TIMERANGE_DAYS)).strftime('%Y%m%d')
        timerange = f"{start_date}-{end_date}"

        command = [
            "freqtrade", "hyperopt",
            "--config", "user_data/config.json",
            "--strategy", "MyHyperOptStrategy",
            "--hyperopt-loss", "SharpeHyperOptLoss",
            "--epochs", str(HYPEROPT_EPOCHS),
            "--spaces", "buy", "sell", "stoploss",
            "-i", "5m",
            "--timerange", timerange
        ]
        
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=50000  # 10-minute timeout
            )

            if process.returncode != 0:
                logger.error(f"Hyperopt failed with return code {process.returncode}:")
                logger.error(process.stderr)
                return

            output = process.stdout
            logger.info("Hyperopt process completed. Parsing results...")

            # Use regex to find the best parameters in the output
            best_params = {}
            stoploss_match = re.search(r"stoploss: (-?\d+\.\d+)", output)
            buy_rsi_match = re.search(r"buy_rsi: (\d+)", output)
            sell_rsi_match = re.search(r"sell_rsi: (\d+)", output)

            if stoploss_match:
                best_params['stoploss'] = float(stoploss_match.group(1))
            if buy_rsi_match:
                best_params['buy_rsi'] = int(buy_rsi_match.group(1))
            if sell_rsi_match:
                best_params['sell_rsi'] = int(sell_rsi_match.group(1))

            if not best_params:
                logger.warning("Could not parse hyperopt results from output.")
                return
            
            # Store results for all trading pairs
            summary = (
                f"Optimized for last {HYPEROPT_TIMERANGE_DAYS} days: "
                f"Stoploss: {best_params.get('stoploss', 'N/A')}, "
                f"Buy RSI: {best_params.get('buy_rsi', 'N/A')}, "
                f"Sell RSI: {best_params.get('sell_rsi', 'N/A')}"
            )
            for pair in self.trading_pairs:
                self.hyperopt_results[pair] = summary
            
            logger.info(f"✅ Automated hyperopt successful. New parameters loaded: {summary}")

        except subprocess.TimeoutExpired:
            logger.error("Hyperopt process timed out.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during automated hyperopt: {e}")


    def craft_trading_prompt(self, market_data: Dict[str, Dict[str, Any]], bot_status: Dict[str, Any]) -> str:
        """Craft a prompt with all available analytical data."""
        market_data_str = json.dumps(market_data, default=lambda x: str(x) if pd.isna(x) else None)
        
        # ... (code to get available_usdt, open_positions is the same)
        available_usdt = 0.0
        balance_info = bot_status.get("balance", {})
        if balance_info and "currencies" in balance_info:
            for cur in balance_info["currencies"]:
                if cur.get("currency") == "USDT":
                    available_usdt = cur.get("free", 0.0)
                    break
        open_trades = bot_status.get("open_trades", [])
        open_positions = [f"{trade['pair']} @ {trade['open_rate']}" for trade in open_trades]

        # --- Dynamic Injection of Analytical Results ---
        
        hyperopt_section = "No recent hyper-optimization data available."
        if self.hyperopt_results:
            hyperopt_section = "ADAPTIVE STRATEGY PARAMETERS (from automated hyper-optimization):\n"
            # Since results are global, show the first available one.
            first_pair = next(iter(self.hyperopt_results), None)
            if first_pair:
                hyperopt_section += f"- {self.hyperopt_results[first_pair]}\n"

        backtest_section = "No long-term backtesting data available."
        if self.backtest_summaries:
            backtest_section = "LONG-TERM PERFORMANCE ANALYSIS (from offline backtesting):\n"
            for pair, summary in self.backtest_summaries.items():
                if pair in market_data:
                    backtest_section += f"- {summary}\n"

        prompt = (
            "You are an expert crypto trading bot. Your task is to analyze market data and output a trading decision for each pair in a specific JSON format. "
            "Your response MUST be a valid JSON array and nothing else.\n\n"
            f"{hyperopt_section}\n"
            f"{backtest_section}\n"
            "CURRENT MARKET DATA (includes 'regime' for each timeframe):\n"
            f"{market_data_str}\n\n"
            f"ACCOUNT STATUS:\n- Available Balance: {available_usdt:.2f} USDT\n"
            f"- Open Positions: {len(open_positions)} ({', '.join(open_positions) if open_positions else 'None'})\n"
            f"- Maximum Open Trades: {self.max_open_trades}\n\n"
            "TASK:\n"
            "For each pair, provide a trading decision. Your primary goal is robust, risk-managed trading.\n"
            "1.  **Stop-Loss:** Base your 'stop_loss_pct' suggestion *heavily* on the 'Optimized Stoploss' from the adaptive analysis. This is the most critical risk parameter.\n"
            "2.  **Confidence:** Use the 'Optimized RSI' values as a baseline for your confidence. If the current RSI is near the optimized buy/sell points, your confidence should be higher.\n"
            "3.  **Rationale:** Briefly state your reasoning.\n\n"
            "OUTPUT FORMAT (strict):\n"
            "A single, valid JSON array of objects. Do not add any text before or after the array.\n"
            "[{\"pair\": \"BTC/USDT\", \"action\": \"BUY\", \"confidence\": 0.8, \"stop_loss_pct\": -0.185, \"rationale\": \"...\"}, ...]"
        )
        return prompt

    def run_trading_cycle(self) -> bool:
        """Run a single trading cycle, including periodic automated analysis."""
        
        # --- Automated, Periodic Analysis ---
        current_time = time.time()
        if (current_time - self.last_hyperopt_time) > HYPEROPT_INTERVAL_SECONDS:
            logger.info("Hyperopt interval reached. Triggering automated parameter optimization.")
            self.run_automated_hyperopt()
            self.last_hyperopt_time = current_time # Update timestamp regardless of success

        logger.info("=== Trading cycle start ===")
        # ... (rest of the trading cycle is the same)
        # ... (fetch market data, bot status, craft prompt, get AI decision, execute)
        if not self.regime_detector.is_trained:
            logger.info("Training regime detector on first pair: %s", self.trading_pairs[0])
            self.train_regime_detector(self.trading_pairs[0])
        market_data = self.analyze_all_pairs()
        if not market_data:
            logger.warning("Skipping cycle – no market data")
            return False
        bot_status = self.get_bot_status()
        if not bot_status:
            logger.warning("Skipping cycle – no bot status")
            return False

        available_usdt = 0.0
        balance_info = bot_status.get("balance", {})
        if balance_info and "currencies" in balance_info:
            for cur in balance_info["currencies"]:
                if cur.get("currency") == "USDT":
                    available_usdt = cur.get("free", 0.0)
                    break
        total_account_value = bot_status.get("total_account_value", 0)

        prompt = self.craft_trading_prompt(market_data, bot_status)
        decisions = self.get_ai_decision(prompt)
        if not decisions:
            logger.warning("No decisions from AI")
            return False

        success = True
        for decision in decisions:
            if not self.execute_trade_decision(decision, available_usdt, total_account_value, market_data):
                success = False
        logger.info("=== Trading cycle end ===")
        return success

    def run_continuous(self, minutes: int = 1):
        """Run the agent continuously."""
        logger.info("Continuous mode started, cycle every %d minute(s)", minutes)
        if not self.test_connections():
            logger.error("Startup aborted – connection failure.")
            return
        try:
            while True:
                self.run_trading_cycle()
                logger.info("Sleeping for %d minute(s)...", minutes)
                time.sleep(minutes * 60)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
    
    # --- (All other methods like _chat_once, test_connections, get_bot_status, etc. remain the same) ---
    def _chat_once(self, prompt: str, model: str) -> Tuple[str, dict]:
        """Send a single chat request to the OpenRouter API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.4,
        )
        if isinstance(response, dict):
            response_dict = response
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            response_dict = json.loads(str(response))
        reply = (response_dict["choices"][0]["message"].get("content") or "").strip()
        return reply, response_dict

    def test_connections(self) -> bool:
        """Test connections to Freqtrade and OpenRouter."""
        try:
            ping = self.ft_client.ping()
            logger.info("Freqtrade ping: %s", ping)
            if ping.get("status") not in {"pong", "running"}:
                logger.error("Freqtrade API is not reachable or bot not running")
                return False
            reply, _ = self._chat_once("Say OK", MODEL_PRIMARY)
            logger.info("OpenRouter handshake reply: '%s'", reply)
            return True
        except Exception as exc:
            logger.exception("Connection test failed: %s", exc)
            return False

    def get_bot_status(self) -> Dict[str, Any]:
        """Retrieve the current status of the Freqtrade bot."""
        try:
            balance = self.ft_client.balance()
            trades = self.ft_client.status()
            profit = self.ft_client.profit()
            total_account_value = sum(cur.get('est_stake', 0) for cur in balance.get('currencies', []))
            return {
                "balance": balance,
                "open_trades": trades,
                "profit": profit,
                "open_trade_count": len(trades) if trades else 0,
                "total_account_value": total_account_value,
            }
        except Exception as exc:
            logger.exception("Bot-status retrieval failed: %s", exc)
            return {}

    def train_regime_detector(self, pair: str, timeframe: str = '1d', limit: int = 1000):
        """
        Train the regime detector using historical data for a specific pair.
        
        Args:
            pair (str): Trading pair (e.g., 'BTC/USDT').
            timeframe (str): Timeframe for candles (default: '1d').
            limit (int): Number of candles to fetch (default: 1000).
        """
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            features = calculate_regime_features(df)
            self.regime_detector.train(features)
        except Exception as exc:
            logger.exception("Failed to train regime detector for %s: %s", pair, exc)

    def gather_enhanced_market_data(self, pair: str, timeframes: List[str], limit: int = 100) -> Dict[str, Any]:
        """Gather market data with indicators and regime detection across multiple timeframes."""
        try:
            market_data = {}
            for tf in timeframes:
                candles = self.exchange.fetch_ohlcv(pair, timeframe=tf, limit=limit)
                if not candles or len(candles) < 50:
                    continue
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                prices = df['close'].tolist()
                volumes = df['volume'].tolist()
                highs = df['high'].tolist()
                lows = df['low'].tolist()
                indicators = calculate_indicators(prices, volumes, highs, lows)
                market_data[tf] = indicators

                # Add regime detection
                regime_features = calculate_regime_features(df)
                if not regime_features.empty and self.regime_detector.is_trained:
                    current_regime = self.regime_detector.detect_regime(regime_features.iloc[-1])
                    market_data[tf]['regime'] = current_regime
                else:
                    market_data[tf]['regime'] = 'unknown'

            market_data['sentiment'] = np.random.uniform(-1, 1)  # Placeholder
            return market_data
        except ccxt.BadSymbol as e:
            logger.warning(f"Skipping {pair}: {e}")
            return {}
        except Exception as exc:
            logger.exception(f"Market data error for {pair}: {exc}")
            return {}

    def analyze_all_pairs(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all trading pairs with enhanced market data."""
        analysis = {}
        for pair in self.trading_pairs:
            data = self.gather_enhanced_market_data(pair, self.timeframes)
            if data:
                analysis[pair] = data
        return analysis
        
    def run_and_store_backtests(self):
        """Runs the advanced backtester for all pairs and stores a summary."""
        if not AdvancedBacktester or not MockAIStrategy:
            logger.error("AdvancedBacktester is not available. Cannot run backtests.")
            return

        logger.info("🚀 Starting advanced backtesting for all whitelisted pairs...")
        backtester = AdvancedBacktester(monte_carlo_simulations=250) # Fewer sims for faster feedback
        # A simple parameter grid for a mock RSI strategy
        param_grid = {
            'rsi_buy_threshold': [20, 25, 30, 35],
            'rsi_sell_threshold': [65, 70, 75, 80]
        }
        
        try:
            results = backtester.backtest_strategy(
                strategy_class=MockAIStrategy,
                timeframe='15m',
                limit=2000, # Use more data for a more meaningful backtest
                param_grid=param_grid
            )
            
            logger.info("Backtesting complete. Generating summaries...")
            for pair, result in results.items():
                if "error" in result:
                    summary = f"Could not backtest {pair}: {result['error']}"
                else:
                    oos_sharpe = result.get('out_sample_sharpe', 'N/A')
                    if isinstance(oos_sharpe, float): oos_sharpe = f"{oos_sharpe:.2f}"
                    
                    mc_mean = result.get('mc_sharpe_mean', 'N/A')
                    if isinstance(mc_mean, float): mc_mean = f"{mc_mean:.2f}"
                    
                    params = result.get('best_params', {})
                    
                    summary = (f"Pair: {pair} | Out-of-Sample Sharpe: {oos_sharpe} | "
                               f"Monte Carlo Sharpe Mean: {mc_mean} | "
                               f"Optimal Params (RSI-based): {params}")
                
                self.backtest_summaries[pair] = summary
                logger.info(summary)

        except Exception as e:
            logger.exception(f"An error occurred during the backtesting process: {e}")

    def get_ai_decision(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """Get trading decisions from the LLM with robust JSON parsing."""
        for model in (MODEL_PRIMARY, MODEL_FALLBACK):
            try:
                reply, full = self._chat_once(prompt, model)
                finish_reason = full["choices"][0].get("finish_reason")
                usage = full.get("usage", {})
                logger.debug("%s finish_reason=%s usage=%s", model, finish_reason, usage)
                logger.debug("AI raw response: %s", reply)

                if not reply:
                    logger.warning("Empty response from model %s, trying fallback...", model)
                    continue

                json_match = re.search(r'\[\s*\{.*\}\s*\]', reply, re.DOTALL)
                if not json_match:
                    logger.error("No valid JSON array found in response from model %s: %s", model, reply)
                    continue

                json_str = json_match.group(0)
                decisions = json.loads(json_str)
                if not isinstance(decisions, list):
                    logger.error("Parsed JSON is not a list, got %s", type(decisions))
                    continue

                for decision in decisions:
                    decision.setdefault("action", "HOLD")
                    decision.setdefault("confidence", 0.5)
                    decision.setdefault("rationale", "No rationale provided")
                    if decision["action"].upper() == "BUY":
                        decision.setdefault("stop_loss_pct", -0.10)

                return decisions

            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from model %s despite extraction: %s", model, reply)
            except Exception as e:
                logger.exception("An unexpected error occurred in get_ai_decision with model %s: %s", model, e)

        logger.warning("Both models failed. Defaulting to HOLD for all pairs.")
        return [{"pair": pair, "action": "HOLD", "confidence": 0.5, "rationale": "No valid response from AI"} for pair in self.trading_pairs]

    def execute_trade_decision(self, decision: Dict[str, Any], available_usdt: float, total_account_value: float, market_data: Dict[str, Any]) -> bool:
        """Execute the trading decision with dynamic sizing and stop-loss."""
        pair = decision.get("pair")
        try:
            action = decision.get("action", "HOLD").upper()
            confidence = decision.get("confidence", 0.5)
            rationale = decision.get("rationale", "")
            logger.info("Executing: %s %s (confidence: %.2f) – %s", action, pair, confidence, rationale)

            open_trades = self.ft_client.status()
            stake_amount = "N/A"

            if action == "BUY":
                if len(open_trades) >= self.max_open_trades:
                    logger.warning("Buy for %s ignored – max trades (%d) reached.", pair, self.max_open_trades)
                    return False
                if any(trade.get("pair") == pair for trade in open_trades):
                    logger.warning("Buy for %s ignored – position already open.", pair)
                    return False

                atr_volatility = market_data.get(pair, {}).get('5m', {}).get('signals', {}).get('ATR_volatility', 'Normal')
                if confidence < 0.7:
                    base_risk_percentage = 0.005
                elif confidence < 0.9:
                    base_risk_percentage = 0.01
                else:
                    base_risk_percentage = 0.02

                if confidence >= 0.95 and atr_volatility == "Low Volatility":
                    risk_percentage = base_risk_percentage * 1.5
                else:
                    risk_percentage = base_risk_percentage

                stop_loss_pct = decision.get("stop_loss_pct", -0.10)
                if stop_loss_pct >= 0:
                    logger.warning("Invalid stop_loss_pct: %f, setting to -0.10", stop_loss_pct)
                    stop_loss_pct = -0.10

                stake_amount = (total_account_value * risk_percentage) / abs(stop_loss_pct)
                stake_amount = min(stake_amount, available_usdt)

                result = self.ft_client.forceenter(pair=pair, side="long", stake_amount=stake_amount)
                logger.info("Buy result: %s", result)

                if result and result.get('trade_id'):
                    logger.info("Trade opened for %s. The active stop-loss will be managed by the '%s' strategy.", pair, "MyManualStrategy")
                else:
                    logger.warning("Failed to open trade for %s. Result: %s", pair, result)
                    return False

                logger.info("Calculated stake_amount: %.2f for %s with risk_percentage: %.4f, stop_loss_pct: %.4f, total_account_value: %.2f, available_usdt: %.2f",
                            stake_amount, pair, risk_percentage, stop_loss_pct, total_account_value, available_usdt)

            elif action == "SELL":
                trade_to_close = next((trade for trade in open_trades if trade.get("pair") == pair), None)
                if trade_to_close:
                    result = self.ft_client.forceexit(tradeid=trade_to_close["trade_id"])
                    logger.info("Sell result: %s", result)
                else:
                    logger.warning("No position for %s – sell not executed.", pair)
                    return False
            else:
                logger.info("No trade for %s (HOLD).", pair)
                return True

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_stake = f"stake: {stake_amount:.2f}" if action == 'BUY' else 'N/A'
            with open("trading_log.txt", "a") as f:
                f.write(f"{timestamp} – {action} {pair} (confidence: {confidence:.2f}, {log_stake}) – {rationale}\n")
            return True
        except Exception as exc:
            logger.exception("Trade error for %s: %s", pair, exc)
            return False

def main():
    """Main entry point for the trading agent."""
    print("🤖 AI Trading Agent for Freqtrade (Now with Automated Adaptive Optimization)\n" + "="*70)
    agent = AITradingAgent()
    if not agent.test_connections():
        print("Connection test failed. Check API settings.")
        return
    print("\nConnections OK. Choose an option:")
    print("1 – Run one trading cycle (includes automated analysis if due)")
    print("2 – Run continuous (1-minute intervals, with hourly adaptive optimization)")
    print("3 – Run continuous (5-minute intervals, with hourly adaptive optimization)")
    print("4 – Test market data")
    print("5 – Show bot status")
    print("6 – Run Long-Term Backtest Analysis (run this once to prime the agent)")
    
    choice = input("Enter 1-6: ").strip()

    if choice == "1":
        agent.run_trading_cycle()
    elif choice == "2":
        agent.run_continuous(1)
    elif choice == "3":
        agent.run_continuous(5)
    elif choice == "4":
        data = agent.analyze_all_pairs()
        print(json.dumps(data, indent=2, default=str))
    elif choice == "5":
        status = agent.get_bot_status()
        print(json.dumps(status, indent=2))
    elif choice == "6":
        agent.run_and_store_backtests()
        print("\n✅ Long-term backtesting complete. Summaries are now stored.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
