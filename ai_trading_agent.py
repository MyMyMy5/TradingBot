#!/usr/bin/env python3
"""
AI Trading Agent for Freqtrade – Enhanced Version with Advanced Indicators and Adaptive Features.
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
import numpy as np
import pandas as pd
import talib.abstract as ta
from hmmlearn.hmm import GaussianHMM
import ccxt
from freqtrade_client import FtRestClient
from openai import OpenAI

# Add user_data to path
sys.path.append(os.path.expanduser('~/user_data'))

try:
    from backtesting.advanced_backtester import AdvancedBacktester, MockAIStrategy
except ImportError:
    print("Could not import AdvancedBacktester.")
    AdvancedBacktester = None
    MockAIStrategy = None

# Configuration
FREQTRADE_API_URL = "http://127.0.0.1:8080"
FREQTRADE_USERNAME = "Freqtrader"
FREQTRADE_PASSWORD = "SuperSecret1!"
OPENROUTER_API_KEY = "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"
MODEL_PRIMARY = "deepseek/deepseek-chat:free"
MODEL_FALLBACK = "deepseek/deepseek-chat-v3-0324:free"
MAX_TOKENS = 8192
HYPEROPT_EPOCHS = 50
HYPEROPT_TIMERANGE_DAYS = 7

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Market Regime Detector
class MarketRegimeDetector:
    def __init__(self):
        self.regimes = ['trending', 'ranging', 'volatile', 'calm']
        self.hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
        self.is_trained = False

    def train(self, features: pd.DataFrame):
        if features.empty or len(features) < 50:
            logger.warning("Insufficient data for training MarketRegimeDetector.")
            return
        self.hmm_model.fit(features)
        self.is_trained = True
        logger.info("MarketRegimeDetector trained successfully with %d data points.", len(features))

    def detect_regime(self, current_features: pd.Series) -> str:
        if not self.is_trained:
            raise ValueError("MarketRegimeDetector must be trained before detecting regimes.")
        current_features_array = current_features.values.reshape(1, -1)
        regime_index = self.hmm_model.predict(current_features_array)[0]
        return self.regimes[regime_index]

def calculate_regime_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=window).std()
    return df[['returns', 'volatility']].dropna()

# Technical Analysis Functions
def interpret_rsi(rsi: float) -> str:
    if rsi > 70: return "Overbought"
    elif rsi < 30: return "Oversold"
    else: return "Neutral"

def interpret_macd(macd: float, macd_signal: float, macd_hist: float, prev_macd_hist: float) -> Dict[str, str]:
    crossover = "No Crossover"
    if macd_hist > 0 and prev_macd_hist < 0: crossover = "Bullish Crossover"
    elif macd_hist < 0 and prev_macd_hist > 0: crossover = "Bearish Crossover"
    zero_line = "Above Zero" if macd > 0 else "Below Zero"
    return {"crossover": crossover, "zero_line": zero_line}

def interpret_bb(close: float, bb_upper: float, bb_lower: float) -> str:
    if close > bb_upper: return "Above Upper Band"
    elif close < bb_lower: return "Below Lower Band"
    else: return "Within Bands"

def interpret_stochastic(stoch_k: float) -> str:
    if stoch_k > 80: return "Overbought"
    elif stoch_k < 20: return "Oversold"
    else: return "Neutral"

def interpret_ichimoku(close: float, senkou_span_a: float, senkou_span_b: float, tenkan_sen: float, kijun_sen: float) -> str:
    if close > senkou_span_a and close > senkou_span_b and tenkan_sen > kijun_sen: return "Bullish"
    elif close < senkou_span_a and close < senkou_span_b and tenkan_sen < kijun_sen: return "Bearish"
    else: return "Neutral"

def interpret_adx(adx: float) -> str:
    if adx > 25: return "Trending"
    elif adx < 20: return "Ranging"
    else: return "Unclear"

def calculate_indicators(prices: List[float], volumes: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
    if len(prices) < 50: return {}
    df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})
    rsi = ta.RSI(df['close'], timeperiod=14)
    macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20)
    stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    tenkan_sen = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    kijun_sen = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    latest_raw = {
        "RSI": rsi[-1] if len(rsi) > 0 else None,
        "MACD_line": macd[-1] if len(macd) > 0 else None,
        "MACD_hist": macd_hist[-1] if len(macd_hist) > 0 else None,
        "BB_upper": bb_upper[-1] if len(bb_upper) > 0 else None,
        "BB_lower": bb_lower[-1] if len(bb_lower) > 0 else None,
        "Stoch_K": stoch_k[-1] if len(stoch_k) > 0 else None,
        "Stoch_D": stoch_d[-1] if len(stoch_d) > 0 else None,
        "ATR": atr[-1] if len(atr) > 0 else None,
        "ADX": adx[-1] if len(adx) > 0 else None,
        "Ichimoku_Tenkan": tenkan_sen.iloc[-1] if len(tenkan_sen) > 0 else None,
        "Ichimoku_Kijun": kijun_sen.iloc[-1] if len(kijun_sen) > 0 else None,
        "Ichimoku_Senkou_A": senkou_span_a.iloc[-1] if len(senkou_span_a) > 0 else None,
        "Ichimoku_Senkou_B": senkou_span_b.iloc[-1] if len(senkou_span_b) > 0 else None,
        "VWAP": vwap.iloc[-1] if len(vwap) > 0 else None,
    }
    signals = {}
    if latest_raw["RSI"] is not None:
        signals["RSI_state"] = interpret_rsi(latest_raw["RSI"])
    if latest_raw["MACD_hist"] is not None and len(macd_hist) >= 2:
        signals["MACD"] = interpret_macd(latest_raw["MACD_line"], macd_signal[-1], latest_raw["MACD_hist"], macd_hist[-2])
    if latest_raw["BB_upper"] is not None:
        signals["BB_position"] = interpret_bb(df['close'].iloc[-1], latest_raw["BB_upper"], latest_raw["BB_lower"])
    if latest_raw["Stoch_K"] is not None:
        signals["Stoch_state"] = interpret_stochastic(latest_raw["Stoch_K"])
    if all(val is not None for val in [latest_raw["Ichimoku_Senkou_A"], latest_raw["Ichimoku_Senkou_B"], latest_raw["Ichimoku_Tenkan"], latest_raw["Ichimoku_Kijun"]]):
        signals["Ichimoku_signal"] = interpret_ichimoku(df['close'].iloc[-1], latest_raw["Ichimoku_Senkou_A"], latest_raw["Ichimoku_Senkou_B"], latest_raw["Ichimoku_Tenkan"], latest_raw["Ichimoku_Kijun"])
    if latest_raw["ATR"] is not None:
        signals["ATR_volatility"] = "High" if latest_raw["ATR"] > 1.5 * np.mean(atr[-50:]) else "Low" if latest_raw["ATR"] < 0.5 * np.mean(atr[-50:]) else "Normal"
    if latest_raw["ADX"] is not None:
        signals["ADX_regime"] = interpret_adx(latest_raw["ADX"])
    if meat_raw["VWAP"] is not None:
        signals["VWAP_position"] = "Above VWAP" if df['close'].iloc[-1] > latest_raw["VWAP"] else "Below VWAP"
    return {"raw": latest_raw, "signals": signals}

class AITradingAgent:
    def __init__(self):
        self.ft_client = FtRestClient(FREQTRADE_API_URL, FREQTRADE_USERNAME, FREQTRADE_PASSWORD)
        self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.exchange = ccxt.kraken()
        self.exchange.load_markets()
        self.regime_detector = MarketRegimeDetector()
        self.backtest_summaries = {}
        self.hyperopt_results = {}
        self.last_hyperopt_time = 0
        self.data_cache = {}
        self.trading_pairs = self._get_trading_pairs()
        self.timeframes = ['5m', '15m', '1h']
        try:
            config = self.ft_client._call("GET", "show_config")
            self.max_open_trades = config.get("max_open_trades", 1)
        except Exception:
            self.max_open_trades = 1
        logger.info("AI Trading Agent initialized.")

    def _get_trading_pairs(self) -> List[str]:
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

    def run_automated_hyperopt(self, volatility: float):
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
            process = subprocess.run(command, capture_output=True, text=True, timeout=50000)
            if process.returncode != 0:
                logger.error(f"Hyperopt failed: {process.stderr}")
                return
            output = process.stdout
            best_params = {}
            for param in ['stoploss', 'buy_rsi', 'sell_rsi']:
                match = re.search(rf"{param}: (-?\d+\.\d+|\d+)", output)
                if match:
                    best_params[param] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
            if best_params:
                summary = f"Optimized: Stoploss: {best_params.get('stoploss', 'N/A')}, Buy RSI: {best_params.get('buy_rsi', 'N/A')}, Sell RSI: {best_params.get('sell_rsi', 'N/A')}"
                for pair in self.trading_pairs:
                    self.hyperopt_results[pair] = summary
                logger.info(f"✅ Hyperopt successful: {summary}")
        except Exception as e:
            logger.exception(f"Hyperopt error: {e}")

    def craft_trading_prompt(self, market_data: Dict[str, Dict[str, Any]], bot_status: Dict[str, Any]) -> str:
        market_data_str = json.dumps(market_data, default=str)
        available_usdt = next((cur["free"] for cur in bot_status.get("balance", {}).get("currencies", []) if cur["currency"] == "USDT"), 0.0)
        open_trades = bot_status.get("open_trades", [])
        open_positions = [f"{trade['pair']} @ {trade['open_rate']}" for trade in open_trades]
        hyperopt_section = "No recent hyper-optimization data." if not self.hyperopt_results else f"ADAPTIVE STRATEGY PARAMETERS:\n- {self.hyperopt_results.get(self.trading_pairs[0], '')}"
        backtest_section = "No backtesting data." if not self.backtest_summaries else f"LONG-TERM PERFORMANCE:\n" + "\n".join([f"- {summary}" for pair, summary in self.backtest_summaries.items() if pair in market_data])
        prompt = (
            "You are an expert crypto trading bot. Output a JSON array of trading decisions.\n\n"
            f"{hyperopt_section}\n{backtest_section}\n"
            f"CURRENT MARKET DATA:\n{market_data_str}\n\n"
            f"ACCOUNT STATUS:\n- Available Balance: {available_usdt:.2f} USDT\n"
            f"- Open Positions: {len(open_positions)} ({', '.join(open_positions) or 'None'})\n"
            f"- Max Open Trades: {self.max_open_trades}\n\n"
            "TASK:\nFor each pair, decide based on robust, risk-managed trading.\n"
            "OUTPUT FORMAT:\n[{\"pair\": \"BTC/USDT\", \"action\": \"BUY\", \"confidence\": 0.8, \"stop_loss_pct\": -0.185, \"rationale\": \"...\"}, ...]"
        )
        return prompt

    def run_trading_cycle(self) -> bool:
        current_time = time.time()
        market_data = self.analyze_all_pairs()
        if not market_data:
            logger.warning("Skipping cycle – no market data")
            return False
        bot_status = self.get_bot_status()
        if not bot_status:
            logger.warning("Skipping cycle – no bot status")
            return False
        atr_value = market_data.get('BTC/USDT', {}).get('5m', {}).get('raw', {}).get('ATR', 0.0)
        hyperopt_interval = 1800 if atr_value > 0.01 else 3600
        if (current_time - self.last_hyperopt_time) > hyperopt_interval:
            self.run_automated_hyperopt(atr_value)
            self.last_hyperopt_time = current_time
        if not self.regime_detector.is_trained:
            self.train_regime_detector(self.trading_pairs[0])
        prompt = self.craft_trading_prompt(market_data, bot_status)
        decisions = self.get_ai_decision(prompt)
        if not decisions:
            logger.warning("No AI decisions")
            return False
        available_usdt = next((cur["free"] for cur in bot_status.get("balance", {}).get("currencies", []) if cur["currency"] == "USDT"), 0.0)
        total_account_value = bot_status.get("total_account_value", 0)
        success = True
        for decision in decisions:
            if not self.execute_trade_decision(decision, available_usdt, total_account_value, market_data):
                success = False
        return success

    def gather_enhanced_market_data(self, pair: str, timeframes: List[str], limit: int = 100) -> Dict[str, Any]:
        cache_key = f"{pair}_{','.join(timeframes)}_{limit}"
        if cache_key in self.data_cache and (time.time() - self.data_cache[cache_key]['timestamp']) < 300:
            return self.data_cache[cache_key]['data']
        try:
            market_data = {}
            for tf in timeframes:
                candles = self.exchange.fetch_ohlcv(pair, timeframe=tf, limit=limit)
                if not candles or len(candles) < 50: continue
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                prices = df['close'].tolist()
                volumes = df['volume'].tolist()
                highs = df['high'].tolist()
                lows = df['low'].tolist()
                indicators = calculate_indicators(prices, volumes, highs, lows)
                market_data[tf] = indicators
                regime_features = calculate_regime_features(df)
                if not regime_features.empty and self.regime_detector.is_trained:
                    market_data[tf]['regime'] = self.regime_detector.detect_regime(regime_features.iloc[-1])
                else:
                    market_data[tf]['regime'] = 'unknown'
            market_data['sentiment'] = self.get_sentiment(pair.split('/')[0])
            self.data_cache[cache_key] = {'data': market_data, 'timestamp': time.time()}
            return market_data
        except Exception as e:
            logger.exception(f"Market data error for {pair}: {e}")
            return {}

    def get_sentiment(self, asset: str) -> float:
        # Placeholder: replace with actual sentiment analysis
        return np.random.uniform(-1, 1)

    def run_continuous(self, minutes: int = 1):
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

    def _chat_once(self, prompt: str, model: str) -> Tuple[str, dict]:
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
        try:
            ping = self.ft_client.ping()
            logger.info("Freqtrade ping: %s", ping)
            if ping.get("status") not in {"pong", "running"}:
                logger.error("Freqtrade API not reachable or bot not running")
                return False
            reply, _ = self._chat_once("Say OK", MODEL_PRIMARY)
            logger.info("OpenRouter handshake reply: '%s'", reply)
            return True
        except Exception as exc:
            logger.exception("Connection test failed: %s", exc)
            return False

    def get_bot_status(self) -> Dict[str, Any]:
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
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            features = calculate_regime_features(df)
            self.regime_detector.train(features)
        except Exception as exc:
            logger.exception("Failed to train regime detector for %s: %s", pair, exc)

    def analyze_all_pairs(self) -> Dict[str, Dict[str, Any]]:
        analysis = {}
        for pair in self.trading_pairs:
            data = self.gather_enhanced_market_data(pair, self.timeframes)
            if data:
                analysis[pair] = data
        return analysis

    def run_and_store_backtests(self):
        if not AdvancedBacktester or not MockAIStrategy:
            logger.error("AdvancedBacktester not available. Cannot run backtests.")
            return
        logger.info("🚀 Starting advanced backtesting for all whitelisted pairs...")
        backtester = AdvancedBacktester(monte_carlo_simulations=250)
        param_grid = {
            'rsi_buy_threshold': [20, 25, 30, 35],
            'rsi_sell_threshold': [65, 70, 75, 80]
        }
        try:
            results = backtester.backtest_strategy(
                strategy_class=MockAIStrategy,
                timeframe='15m',
                limit=2000,
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
                    summary = f"Pair: {pair} | Out-of-Sample Sharpe: {oos_sharpe} | Monte Carlo Sharpe Mean: {mc_mean} | Optimal Params: {params}"
                self.backtest_summaries[pair] = summary
                logger.info(summary)
        except Exception as e:
            logger.exception(f"Backtesting error: {e}")

    def get_ai_decision(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        for model in (MODEL_PRIMARY, MODEL_FALLBACK):
            try:
                reply, full = self._chat_once(prompt, model)
                finish_reason = full["choices"][0].get("finish_reason")
                usage = full.get("usage", {})
                logger.debug("%s finish_reason=%s usage=%s", model, finish_reason, usage)
                logger.debug("AI raw response: %s", reply)
                if not reply:
                    logger.warning("Empty response from %s, trying fallback...", model)
                    continue
                json_match = re.search(r'\[\s*\{.*\}\s*\]', reply, re.DOTALL)
                if not json_match:
                    logger.error("No JSON array in response from %s: %s", model, reply)
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
                logger.error("JSON parse failed from %s: %s", model, reply)
            except Exception as e:
                logger.exception("AI decision error with %s: %s", model, e)
        logger.warning("Both models failed. Defaulting to HOLD.")
        return [{"pair": pair, "action": "HOLD", "confidence": 0.5, "rationale": "No valid AI response"} for pair in self.trading_pairs]

    def execute_trade_decision(self, decision: Dict[str, Any], available_usdt: float, total_account_value: float, market_data: Dict[str, Any]) -> bool:
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
                base_risk_percentage = 0.005 if confidence < 0.7 else 0.01 if confidence < 0.9 else 0.02
                risk_percentage = base_risk_percentage * 1.5 if confidence >= 0.95 and atr_volatility == "Low" else base_risk_percentage
                stop_loss_pct = decision.get("stop_loss_pct", -0.10)
                if stop_loss_pct >= 0:
                    logger.warning("Invalid stop_loss_pct: %f, setting to -0.10", stop_loss_pct)
                    stop_loss_pct = -0.10
                stake_amount = (total_account_value * risk_percentage) / abs(stop_loss_pct)
                stake_amount = min(stake_amount, available_usdt)
                result = self.ft_client.forceenter(pair=pair, side="long", stake_amount=stake_amount)
                logger.info("Buy result: %s", result)
                if result and result.get('trade_id'):
                    logger.info("Trade opened for %s.", pair)
                else:
                    logger.warning("Failed to open trade for %s. Result: %s", pair, result)
                    return False
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
    print("🤖 AI Trading Agent for Freqtrade (Enhanced)\n" + "="*70)
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
    print("6 – Run Long-Term Backtest Analysis")
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
        print("\n✅ Backtesting complete.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
