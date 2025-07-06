#!/usr/bin/env python3
"""
AI Trading Agent for Freqtrade – Enhanced Version
Integrates Freqtrade, OpenRouter AI, and automated trading decisions.

Key improvements:
- Multi-pair support with automatic filtering of available pairs on the exchange.
- Enhanced market analysis with RSI, MACD, Bollinger Bands, Volume indicators, and interpreted signals.
- Independent evaluation of each trading pair for BUY, SELL, or HOLD decisions.
- Improved logging with detailed decision reasons.
- Respects max_open_trades from Freqtrade config for better risk management.
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

# Technical analysis helpers
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FREQTRADE_API_URL = "http://127.0.0.1:8080"
FREQTRADE_USERNAME = "Freqtrader"
FREQTRADE_PASSWORD = "SuperSecret1!"

OPENROUTER_API_KEY = "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"

MODEL_PRIMARY = "deepseek/deepseek-chat:free"   # primary model (fast & follows instructions)
MODEL_FALLBACK = "openchat/openchat-7b:free"    # fallback model
MAX_TOKENS = 512

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
# Technical Analysis Helper Functions
# ---------------------------------------------------------------------------
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index) from a list of prices."""
    if len(prices) < period + 1:
        return 50.0  # neutral if not enough data
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(change if change > 0 else 0)
        losses.append(-change if change < 0 else 0)
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0  # all gains, no losses -> RSI 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: List[float]) -> Tuple[float, float, float]:
    """Calculate MACD line, Signal line, and Histogram from prices (simplified EMA)."""
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    ema12 = sum(prices[-12:]) / 12
    ema26 = sum(prices[-26:]) / 26
    macd_line = ema12 - ema26
    signal_line = macd_line  # Simplified, in reality, it's EMA of MACD
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands (middle, upper, lower) for given prices."""
    if len(prices) < period:
        avg = sum(prices) / len(prices)
        return avg, avg * 1.02, avg * 0.98  # fallback if not enough data
    window = prices[-period:]
    sma = sum(window) / period
    variance = sum((p - sma) ** 2 for p in window) / period
    std_dev = variance ** 0.5
    upper_band = sma + 2 * std_dev
    lower_band = sma - 2 * std_dev
    return sma, upper_band, lower_band

# ---------------------------------------------------------------------------
# AI Trading Agent Class
# ---------------------------------------------------------------------------
class AITradingAgent:
    def __init__(self):
        self.ft_client = FtRestClient(FREQTRADE_API_URL, FREQTRADE_USERNAME, FREQTRADE_PASSWORD)
        self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.exchange = ccxt.kraken()
        self.exchange.load_markets()
        try:
            config = self.ft_client._call("GET", "/config")
            self.max_open_trades = config.get("max_open_trades", 1)
            self.trading_pairs = config.get("exchange", {}).get("pair_whitelist", [])
            if not self.trading_pairs:
                logger.error("No trading pairs found in Freqtrade config. Using defaults.")
                self.trading_pairs = ["BTC/USDT", "ETH/USDT"]
            # Filter pairs to ensure they are available on the exchange
            self.trading_pairs = [pair for pair in self.trading_pairs if pair in self.exchange.markets]
            if not self.trading_pairs:
                logger.error("No valid trading pairs found in pair_whitelist.")
                raise ValueError("No valid trading pairs")
        except Exception as exc:
            logger.exception("Failed to get config, using defaults")
            self.max_open_trades = 1
            self.trading_pairs = ["BTC/USDT", "ETH/USDT"]
        logger.info("AI Trading Agent initialized with max_open_trades=%d and %d trading pairs", self.max_open_trades, len(self.trading_pairs))

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
                logger.error("Freqtrade API is not reachable or bot not running")
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
            return {
                "balance": balance,
                "open_trades": trades,
                "profit": profit,
                "open_trade_count": len(trades) if trades else 0,
            }
        except Exception as exc:
            logger.exception("Bot-status retrieval failed: %s", exc)
            return {}

    def gather_enhanced_market_data(self, pair: str, timeframe: str = "5m", limit: int = 50) -> Dict[str, Any]:
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            if not candles or len(candles) < 2:
                logger.warning("No candle data for %s", pair)
                return {}
            prices = [candle[4] for candle in candles]
            volumes = [candle[5] for candle in candles]
            latest_price = prices[-1]
            prev_price = prices[-2]
            price_change_pct = (latest_price - prev_price) / prev_price * 100 if prev_price != 0 else 0.0
            sma_20 = sum(prices[-20:]) / min(20, len(prices))
            sma_50 = sum(prices[-50:]) / min(50, len(prices))
            rsi = calculate_rsi(prices)
            macd_line, signal_line, hist = calculate_macd(prices)
            bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(prices)
            avg_vol = sum(volumes[-10:]) / min(10, len(volumes))
            volume_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
            trend_strength = sum([latest_price > sma_20, sma_20 > sma_50, rsi > 50, macd_line > signal_line])
            rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            macd_signal = "bullish" if macd_line > signal_line else "bearish"
            bb_signal = ("above upper band" if latest_price > bb_upper else "below lower band" if latest_price < bb_lower else "within bands")
            volume_signal = "high" if volume_ratio > 1.5 else "normal"
            return {
                "pair": pair,
                "latest_price": latest_price,
                "price_change_percent": price_change_pct,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": rsi,
                "rsi_signal": rsi_signal,
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": hist,
                "macd_signal": macd_signal,
                "bb_upper": bb_upper,
                "bb_middle": bb_mid,
                "bb_lower": bb_lower,
                "bb_signal": bb_signal,
                "volume_ratio": volume_ratio,
                "volume_signal": volume_signal,
                "trend_strength": trend_strength,
            }
        except ccxt.base.errors.BadSymbol as e:
            logger.warning(f"Skipping {pair}: {e}")
            return {}
        except Exception as exc:
            logger.exception(f"Market data error for {pair}: {exc}")
            return {}

    def analyze_all_pairs(self) -> Dict[str, Dict[str, Any]]:
        analysis = {}
        for pair in self.trading_pairs:
            data = self.gather_enhanced_market_data(pair)
            if data:
                analysis[pair] = data
        return analysis

    def craft_trading_prompt(self, market_data: Dict[str, Dict[str, Any]], bot_status: Dict[str, Any]) -> str:
        market_summary = ""
        for pair, data in market_data.items():
            market_summary += (
                f"{pair}: Price={data['latest_price']:.2f} ({data['price_change_percent']:.2f}%), "
                f"RSI={data['rsi']:.2f} ({data['rsi_signal']}), MACD={data['macd_signal']}, "
                f"BB={data['bb_signal']}, Volume={data['volume_signal']}, Trend={data['trend_strength']}/4\n"
            )
        available_usdt = 0.0
        balance_info = bot_status.get("balance", {})
        if balance_info and "currencies" in balance_info:
            for cur in balance_info["currencies"]:
                if cur.get("currency") == "USDT":
                    available_usdt = cur.get("free", 0.0)
                    break
        open_trades = bot_status.get("open_trades", [])
        open_positions = [f"{trade['pair']} @ {trade['open_rate']}" for trade in open_trades]
        prompt = (
            "You are an AI crypto trading agent acting like a human trader. Analyze the market data and decide whether to BUY, SELL, or HOLD for each pair.\n\n"
            f"MARKET DATA:\n{market_summary}\n\n"
            f"ACCOUNT STATUS:\n- Available Balance: {available_usdt:.2f} USDT\n"
            f"- Open Positions: {len(open_positions)} ({', '.join(open_positions) if open_positions else 'None'})\n"
            f"- Maximum Open Trades: {self.max_open_trades}\n\n"
            "RULES:\n"
            f"1. Up to {self.max_open_trades} open positions allowed to manage risk.\n"
            "2. Trade amount is fixed at $10 per trade (dry-run).\n"
            "3. For each pair, decide independently whether to BUY, SELL, or HOLD based on its market data.\n"
            "4. Consider the overall account status and avoid overexposure.\n"
            "5. BUY if there's a strong bullish signal (e.g., RSI oversold < 30, trend_strength 3-4/4, high volume >1.5x, MACD bullish, price near lower Bollinger Band) and there is room for more open positions.\n"
            "6. SELL if in profit or to cut loss (e.g., RSI overbought > 70, trend reverses, MACD bearish, price near upper Bollinger Band) for pairs with open positions.\n"
            "7. HOLD if no clear signal or if maximum open trades are reached for new buys.\n"
            "8. Consider Bitcoin's price trend, as it often influences the broader market.\n\n"
            "Provide a list of decisions, one for each pair, in JSON format: "
            "[{\"pair\": \"BTC/USDT\", \"action\": \"BUY/SELL/HOLD\", \"reason\": \"...\"}, {\"pair\": \"ETH/USDT\", \"action\": \"BUY/SELL/HOLD\", \"reason\": \"...\"}]"
        )
        return prompt

    def get_ai_decision(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        for model in (MODEL_PRIMARY, MODEL_FALLBACK):
            reply, full = self._chat_once(prompt, model)
            finish_reason = full["choices"][0].get("finish_reason")
            usage = full.get("usage", {})
            logger.debug("%s finish_reason=%s usage=%s", model, finish_reason, usage)
            logger.debug("AI raw response: %s", reply)
            if not reply:
                logger.warning("Empty response from model %s, trying fallback...", model)
                continue
            if reply.startswith("```"):
                reply = reply.strip("` \n")
                if reply.lower().startswith("json"):
                    reply = reply[len("json"):].strip()
            try:
                decisions = json.loads(reply)
                if not isinstance(decisions, list):
                    logger.error("Expected list of decisions, got %s", type(decisions))
                    continue
                for decision in decisions:
                    if "action" in decision:
                        decision["action"] = decision["action"].upper()
                    else:
                        decision["action"] = "HOLD"
                    if "pair" not in decision:
                        decision["pair"] = self.trading_pairs[0]
                    if "reason" not in decision:
                        decision["reason"] = "No reason provided"
                    decision["amount"] = 10  # Ensure amount is set
                return decisions
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from model %s: %s", model, reply)
                text = reply.upper()
                action = "HOLD"
                if "BUY" in text:
                    action = "BUY"
                elif "SELL" in text:
                    action = "SELL"
                pair = self.trading_pairs[0]
                for p in self.trading_pairs:
                    if p in text or p.replace("/", "") in text:
                        pair = p
                        break
                reason = reply[:100]
                return [{"pair": pair, "action": action, "amount": 10, "reason": reason}]
        logger.warning("Both models returned no decision. Defaulting to HOLD for all pairs.")
        return [{"pair": pair, "action": "HOLD", "amount": 10, "reason": "No response from AI"} for pair in self.trading_pairs]

    def execute_trade_decision(self, decision: Dict[str, Any]) -> bool:
        try:
            action = decision.get("action", "HOLD").upper()
            pair = decision.get("pair", self.trading_pairs[0])
            amount = decision.get("amount", 10)
            reason = decision.get("reason", "")
            logger.info("Executing decision: %s %s (amount: %s) – %s", action, pair, amount, reason)
            open_trades = self.ft_client.status()
            if action == "BUY":
                if len(open_trades) >= self.max_open_trades:
                    logger.warning("Buy signal for %s ignored – maximum open trades (%d) reached.", pair, self.max_open_trades)
                    return False
                if any(trade.get("pair") == pair for trade in open_trades):
                    logger.warning("Buy signal for %s ignored – already have an open trade for this pair.", pair)
                    return False
                result = self.ft_client.forceenter(pair=pair, side="long")
                logger.info("Buy order result: %s", result)
            elif action == "SELL":
                if not open_trades:
                    logger.warning("Sell signal for %s ignored – no open trades to sell.", pair)
                    return False
                trade_to_close = next((trade for trade in open_trades if trade.get("pair") == pair), None)
                if trade_to_close:
                    result = self.ft_client.forceexit(tradeid=trade_to_close["trade_id"])
                    logger.info("Sell order result: %s", result)
                else:
                    logger.warning("No open position found for %s – sell not executed.", pair)
                    return False
            else:
                logger.info("No trade executed for %s (HOLD decision). Waiting for better opportunity.", pair)
                return True
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("trading_log.txt", "a") as f:
                f.write(f"{timestamp} – {action} {pair} – {reason}\n")
            return True
        except Exception as exc:
            logger.exception("Trade execution error for %s: %s", pair, exc)
            return False

    def run_trading_cycle(self) -> bool:
        logger.info("=== Trading cycle start ===")
        market_data = self.analyze_all_pairs()
        if not market_data:
            logger.warning("Skipping cycle – no market data gathered")
            return False
        bot_status = self.get_bot_status()
        if not bot_status:
            logger.warning("Skipping cycle – unable to get bot status")
            return False
        prompt = self.craft_trading_prompt(market_data, bot_status)
        decisions = self.get_ai_decision(prompt)
        if not decisions:
            logger.warning("No decisions from AI (cycle skipped)")
            return False
        success = True
        for decision in decisions:
            if not self.execute_trade_decision(decision):
                success = False
        logger.info("=== Trading cycle end ===")
        return success

    def run_continuous(self, minutes: int = 1):
        logger.info("Continuous mode started, cycle every %d minute(s)", minutes)
        if not self.test_connections():
            logger.error("Startup aborted – unable to connect to Freqtrade or OpenRouter.")
            return
        try:
            while True:
                self.run_trading_cycle()
                logger.info("Sleeping for %d minute(s)...", minutes)
                time.sleep(minutes * 60)
        except KeyboardInterrupt:
            logger.info("Stopped by user")

def main():
    print("🤖 AI Trading Agent for Freqtrade (Enhanced)\n" + "="*50)
    agent = AITradingAgent()
    if not agent.test_connections():
        print("Connection test failed. Check API settings and try again.")
        return
    print("\nConnections OK. Choose an option:")
    print("1 – Run one trading cycle")
    print("2 – Run continuous trading (1-minute intervals)")
    print("3 – Run continuous trading (5-minute intervals)")
    print("4 – Test market data gathering")
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
