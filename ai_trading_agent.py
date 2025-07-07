#!/usr/bin/env python3
"""
AI Trading Agent for Freqtrade – Enhanced Version
Integrates Freqtrade, OpenRouter AI, and automated trading decisions with advanced features.

Advanced Features:
- Dynamic Position Sizing: Adjusts stake amount based on LLM confidence and market volatility (ATR).
- Multi-Timeframe Analysis: Analyzes 5m, 15m, and 1h timeframes for richer context.
- Sentiment Integration: Includes a placeholder sentiment score (to be replaced with real data).
- Adaptive Risk Management: Allows LLM to set dynamic stop-loss percentages based on volatility.
- Comprehensive logging with decision rationales and sizing details.
- Enhanced Technical Indicator Input: Provides both raw values and interpreted signals for LLM.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# External libraries
import ccxt
from freqtrade_client import FtRestClient
from openai import OpenAI
import pandas as pd
import numpy as np
import talib.abstract as ta
import re # Add this import at the top of your file with the others

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FREQTRADE_API_URL = "http://127.0.0.1:8080"
FREQTRADE_USERNAME = "Freqtrader"
FREQTRADE_PASSWORD = "SuperSecret1!"

OPENROUTER_API_KEY = "CENCORED"

MODEL_PRIMARY = "deepseek/deepseek-chat:free"   # Primary model (fast & follows instructions)
MODEL_FALLBACK = "deepseek/deepseek-chat-v3-0324:free"    # Fallback model
MAX_TOKENS = 8192 

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    return st_signal  # Assuming st_signal is "Buy" or "Sell"

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
        return {}  # Insufficient data

    df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})

    # Compute indicators (all return NumPy arrays except st_line)
    rsi = ta.RSI(df['close'], timeperiod=14)
    macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    tenkan_sen = ta.HT_TRENDLINE(df['close'])  # Simplified Ichimoku
    kijun_sen = ta.HT_TRENDLINE(df['close'])
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = ta.HT_TRENDLINE(df['close'])
    obv = ta.OBV(df['close'], df['volume'])
    adl = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    volume_sma = ta.SMA(df['volume'], timeperiod=20)
    st_line, st_signal = calculate_supertrend(df)  # st_line is a Pandas Series, st_signal is a list

    # Latest raw values
    latest_raw = {
        "close": df['close'].iloc[-1],
        "high": df['high'].iloc[-1],
        "low": df['low'].iloc[-1],
        "volume": df['volume'].iloc[-1],
        "RSI": rsi[-1] if len(rsi) > 0 else None,
        "MACD_line": macd[-1] if len(macd) > 0 else None,
        "MACD_signal": macd_signal[-1] if len(macd_signal) > 0 else None,
        "MACD_hist": macd_hist[-1] if len(macd_hist) > 0 else None,
        "BB_upper": bb_upper[-1] if len(bb_upper) > 0 else None,
        "BB_middle": bb_middle[-1] if len(bb_middle) > 0 else None,
        "BB_lower": bb_lower[-1] if len(bb_lower) > 0 else None,
        "Stoch_K": stoch_k[-1] if len(stoch_k) > 0 else None,
        "Stoch_D": stoch_d[-1] if len(stoch_d) > 0 else None,
        "Ichimoku_Tenkan": tenkan_sen[-1] if len(tenkan_sen) > 0 else None,
        "Ichimoku_Kijun": kijun_sen[-1] if len(kijun_sen) > 0 else None,
        "Ichimoku_Senkou_A": senkou_span_a[-1] if len(senkou_span_a) > 0 else None,
        "Ichimoku_Senkou_B": senkou_span_b[-1] if len(senkou_span_b) > 0 else None,
        "OBV": obv[-1] if len(obv) > 0 else None,
        "ADL": adl[-1] if len(adl) > 0 else None,
        "ATR": atr[-1] if len(atr) > 0 else None,
        "ADX": adx[-1] if len(adx) > 0 else None,
        "Volume_SMA": volume_sma[-1] if len(volume_sma) > 0 else None,
        "SuperTrend": st_line.iloc[-1] if len(st_line) > 0 else None,
    }

    # Interpreted signals
    signals = {}
    if len(rsi) > 0 and not pd.isna(rsi[-1]):
        signals["RSI_state"] = interpret_rsi(rsi[-1])
    if len(macd_hist) >= 2 and not pd.isna(macd_hist[-1]):
        prev_macd_hist = macd_hist[-2]
        signals["MACD"] = interpret_macd(macd[-1], macd_signal[-1], macd_hist[-1], prev_macd_hist)
    if len(bb_upper) > 0 and not pd.isna(bb_upper[-1]):
        signals["BB_position"] = interpret_bb(df['close'].iloc[-1], bb_upper[-1], bb_lower[-1])
    if len(stoch_k) > 0 and not pd.isna(stoch_k[-1]):
        signals["Stoch_state"] = interpret_stochastic(stoch_k[-1])
    if len(tenkan_sen) > 0 and not pd.isna(tenkan_sen[-1]):
        signals["Ichimoku_signal"] = interpret_ichimoku(df['close'].iloc[-1], senkou_span_a[-1], senkou_span_b[-1], tenkan_sen[-1], kijun_sen[-1])
    if len(obv) >= 5 and not pd.isna(obv[-1]):
        prev_obv = obv[-5]
        signals["OBV_trend"] = interpret_obv(obv[-1], prev_obv)
    if len(adl) >= 5 and not pd.isna(adl[-1]):
        prev_adl = adl[-5]
        signals["ADL_trend"] = interpret_obv(adl[-1], prev_adl)
    if len(atr) >= 50 and not pd.isna(atr[-1]):
        avg_atr = atr[-50:].mean()
        if atr[-1] > 1.5 * avg_atr:
            signals["ATR_volatility"] = "High Volatility"
        elif atr[-1] < 0.5 * avg_atr:
            signals["ATR_volatility"] = "Low Volatility"
        else:
            signals["ATR_volatility"] = "Normal Volatility"
    if len(adx) > 0 and not pd.isna(adx[-1]):
        signals["ADX_regime"] = interpret_adx(adx[-1])
    if len(volume_sma) > 0 and not pd.isna(volume_sma[-1]):
        signals["Volume_signal"] = interpret_volume(df['volume'].iloc[-1], volume_sma[-1])
    if len(st_signal) > 0:
        signals["SuperTrend_signal"] = interpret_supertrend(st_signal[-1])

    return {"raw": latest_raw, "signals": signals}

# ---------------------------------------------------------------------------
# AI Trading Agent Class
# ---------------------------------------------------------------------------
class AITradingAgent:
    def __init__(self):
        """Initialize the AI trading agent with Freqtrade and exchange connections."""
        self.ft_client = FtRestClient(FREQTRADE_API_URL, FREQTRADE_USERNAME, FREQTRADE_PASSWORD)
        self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.exchange = ccxt.kraken()
        self.exchange.load_markets()
        logger.info("Available markets on Kraken: %s", len(list(self.exchange.markets.keys())))

        # Get max_open_trades from config
        try:
            config = self.ft_client._call("GET", "show_config")
            logger.info("Config response: %s", json.dumps(config))
            self.max_open_trades = config.get("max_open_trades", 1)
            logger.info("Retrieved max_open_trades: %d", self.max_open_trades)
        except Exception as exc:
            logger.warning("Failed to get config, using default max_open_trades=1: %s", exc)
            self.max_open_trades = 1

        # Wait for Freqtrade to initialize and populate the whitelist
        self.trading_pairs = []
        for attempt in range(10):
            try:
                whitelist_response = self.ft_client._call("GET", "whitelist")
                logger.info("Whitelist response: %s", json.dumps(whitelist_response))
                self.trading_pairs = whitelist_response.get("whitelist", [])
                if self.trading_pairs:
                    break
            except Exception as exc:
                logger.warning("Failed to get whitelist: %s", exc)
            logger.info("Waiting for Freqtrade to initialize (attempt %d/10)...", attempt + 1)
            time.sleep(5)
        else:
            logger.error("Failed to get non-empty whitelist after 10 attempts. Using defaults.")
            self.trading_pairs = ["BTC/USDT", "ETH/USDT"]

        # Filter trading pairs to those available on the exchange
        self.trading_pairs = [pair for pair in self.trading_pairs if pair in self.exchange.markets]
        if not self.trading_pairs:
            logger.error("No valid trading pairs found in whitelist.")
            raise ValueError("No valid trading pairs")
        logger.info("Retrieved %d trading pairs: %s", len(self.trading_pairs), self.trading_pairs)

        logger.info("AI Trading Agent initialized with max_open_trades=%d and %d trading pairs",
                    self.max_open_trades, len(self.trading_pairs))
        self.timeframes = ['5m', '15m', '1h']

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

    def gather_enhanced_market_data(self, pair: str, timeframes: List[str], limit: int = 100) -> Dict[str, Any]:
        """Gather market data with indicators across multiple timeframes."""
        try:
            market_data = {}
            for tf in timeframes:
                candles = self.exchange.fetch_ohlcv(pair, timeframe=tf, limit=limit)
                if not candles or len(candles) < 50:
                    continue
                prices = [candle[4] for candle in candles]
                volumes = [candle[5] for candle in candles]
                highs = [candle[2] for candle in candles]
                lows = [candle[3] for candle in candles]
                indicators = calculate_indicators(prices, volumes, highs, lows)
                market_data[tf] = indicators
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

    def craft_trading_prompt(self, market_data: Dict[str, Dict[str, Any]], bot_status: Dict[str, Any]) -> str:
        """Craft a prompt with structured market data for the LLM."""
        # ... (the first part of the function is unchanged) ...
        market_data_str = json.dumps(market_data)
        available_usdt = 0.0
        balance_info = bot_status.get("balance", {})
        if balance_info and "currencies" in balance_info:
            for cur in balance_info["currencies"]:
                if cur.get("currency") == "USDT":
                    available_usdt = cur.get("free", 0.0)
                    break
        open_trades = bot_status.get("open_trades", [])
        open_positions = [f"{trade['pair']} @ {trade['open_rate']}" for trade in open_trades]

        # MODIFIED PROMPT
        prompt = (
            "You are an expert crypto trading bot. Your task is to analyze market data and output a trading decision for each pair in a specific JSON format. "
            "Your response MUST be a valid JSON array and nothing else. Do not include any introductory text, explanations, or ```json``` formatting.\n\n"
            "MARKET DATA:\n"
            f"{market_data_str}\n\n"
            f"ACCOUNT STATUS:\n- Available Balance: {available_usdt:.2f} USDT\n"
            f"- Open Positions: {len(open_positions)} ({', '.join(open_positions) if open_positions else 'None'})\n"
            f"- Maximum Open Trades: {self.max_open_trades}\n\n"
            "TASK:\n"
            "For each pair, provide a trading decision with a confidence score (0.0 to 1.0), a suggested stop_loss_pct for BUY decisions (negative decimal, e.g., -0.05), and a brief rationale.\n"
            "Consider all indicators across all timeframes, market trends, and account status.\n"
            "Use 'ATR_volatility' to help set stop_loss_pct (e.g., -0.02 for Low Volatility, -0.05 for Normal, -0.10 for High Volatility).\n\n"
            "OUTPUT FORMAT (strict):\n"
            "A single, valid JSON array of objects. Do not add any text before or after the array.\n"
            "[{\"pair\": \"BTC/USDT\", \"action\": \"BUY\", \"confidence\": 0.8, \"stop_loss_pct\": -0.05, \"rationale\": \"...\"}, ...]"
        )
        return prompt
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

                # Robustly find and extract the JSON array
                json_match = re.search(r'\[\s*\{.*\}\s*\]', reply, re.DOTALL)
                if not json_match:
                    logger.error("No valid JSON array found in response from model %s: %s", model, reply)
                    continue
                
                json_str = json_match.group(0)
                
                decisions = json.loads(json_str)
                if not isinstance(decisions, list):
                    logger.error("Parsed JSON is not a list, got %s", type(decisions))
                    continue

                # Validate and normalize decisions
                for decision in decisions:
                    decision.setdefault("action", "HOLD")
                    decision.setdefault("confidence", 0.5)
                    decision.setdefault("rationale", "No rationale provided")
                    if decision["action"].upper() == "BUY":
                        decision.setdefault("stop_loss_pct", -0.10)
                
                return decisions # Success

            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from model %s despite extraction: %s", model, reply)
            except Exception as e:
                logger.exception("An unexpected error occurred in get_ai_decision with model %s: %s", model, e)
                
        logger.warning("Both models failed. Defaulting to HOLD for all pairs.")
        # Use self.trading_pairs which is available in the class instance
        return [{"pair": pair, "action": "HOLD", "confidence": 0.5, "rationale": "No valid response from AI"} for pair in self.trading_pairs]
    def execute_trade_decision(self, decision: Dict[str, Any], available_usdt: float, total_account_value: float, market_data: Dict[str, Any]) -> bool:
        """Execute the trading decision with dynamic sizing and stop-loss."""
        pair = decision.get("pair") # Define pair early for exception logging
        try:
            action = decision.get("action", "HOLD").upper()
            confidence = decision.get("confidence", 0.5)
            rationale = decision.get("rationale", "")
            logger.info("Executing: %s %s (confidence: %.2f) – %s", action, pair, confidence, rationale)

            open_trades = self.ft_client.status()
            stake_amount = "N/A" # Initialize for logging

            if action == "BUY":
                if len(open_trades) >= self.max_open_trades:
                    logger.warning("Buy for %s ignored – max trades (%d) reached.", pair, self.max_open_trades)
                    return False
                if any(trade.get("pair") == pair for trade in open_trades):
                    logger.warning("Buy for %s ignored – position already open.", pair)
                    return False

                # Dynamic position sizing
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

                # --- THIS IS THE CORRECTED LINE ---
                result = self.ft_client.forceenter(pair=pair, side="long", stake_amount=stake_amount)
                # ------------------------------------
                
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
            else: # HOLD
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
    
    def run_trading_cycle(self) -> bool:
        """Run a single trading cycle."""
        logger.info("=== Trading cycle start ===")
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

def main():
    """Main entry point for the trading agent."""
    print("🤖 AI Trading Agent for Freqtrade (Enhanced)\n" + "="*50)
    agent = AITradingAgent()
    if not agent.test_connections():
        print("Connection test failed. Check API settings.")
        return
    print("\nConnections OK. Choose an option:")
    print("1 – Run one trading cycle")
    print("2 – Run continuous (1-minute intervals)")
    print("3 – Run continuous (5-minute intervals)")
    print("4 – Test market data")
    print("5 – Show bot status")
    choice = input("Enter 1-5: ").strip()
    if choice == "1":
        agent.run_trading_cycle()
    elif choice == "2":
        agent.run_continuous(1)
    elif choice == "3":
        agent.run_continuous(5)
    elif choice == "4":
        data = agent.analyze_all_pairs()
        print(json.dumps(data, indent=2))
    elif choice == "5":
        status = agent.get_bot_status()
        print(json.dumps(status, indent=2))
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
