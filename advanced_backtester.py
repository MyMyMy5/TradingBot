#will be located at: "~/user_data/backtesting/advanced_backtester.py"

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import json
import os
import ccxt
# At the top of advanced_backtester.py
from freqtrade.strategy import IStrategy
from strategies.MyManualStrategy import MyManualStrategy

class Strategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

class MockAIStrategy(Strategy):
    def __init__(self, rsi_buy_threshold=30, rsi_sell_threshold=70):
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, dtype='object')
        signals[data['rsi'] < self.rsi_buy_threshold] = 'buy'
        signals[data['rsi'] > self.rsi_sell_threshold] = 'sell'
        signals = signals.fillna('hold')
        return signals

def simulate_trades(data: pd.DataFrame, signals: pd.Series, initial_capital: float, stake_amount: float, slippage: float, fee: float) -> pd.DataFrame:
    cash = initial_capital
    position = 0  # number of units held
    portfolio_value = []
    for t in data.index:
        signal = signals.get(t, 'hold')
        close_price = data['close'][t]
        if signal == 'buy' and cash >= stake_amount:
            buy_price = close_price * (1 + slippage)
            units_to_buy = stake_amount / buy_price
            fee_paid = units_to_buy * buy_price * fee
            position += units_to_buy
            cash -= stake_amount + fee_paid
        elif signal == 'sell' and position > 0:
            sell_price = close_price * (1 - slippage)
            cash_received = position * sell_price
            fee_paid = cash_received * fee
            cash += cash_received - fee_paid
            position = 0
        current_value = cash + position * close_price
        portfolio_value.append(current_value)
    portfolio_df = pd.DataFrame({'portfolio_value': portfolio_value}, index=data.index)
    return portfolio_df

def calculate_sharpe_ratio(portfolio_value: pd.Series, risk_free_rate: float = 0.0) -> float:
    daily_returns = portfolio_value.pct_change().dropna()
    if len(daily_returns) < 2:
        return np.nan
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe_ratio = np.sqrt(252) * (mean_return - risk_free_rate) / std_return if std_return != 0 else np.nan
    return sharpe_ratio

def generate_simulated_path(data: pd.DataFrame, block_size: int = 5) -> pd.DataFrame:
    n = len(data)
    # Ensure there's enough data to form at least one block
    if n <= block_size:
        return data.copy().reset_index(drop=True)
        
    simulated_data_list = []
    current_len = 0
    while current_len < n:
        start_idx = np.random.randint(0, n - block_size)
        block = data.iloc[start_idx:start_idx + block_size]
        simulated_data_list.append(block)
        current_len += block_size
        
    simulated_data = pd.concat(simulated_data_list)
    return simulated_data.iloc[:n].reset_index(drop=True)

class AdvancedBacktester:
    def __init__(self, walk_forward_analysis=True, monte_carlo_simulations=1000, realistic_execution=True):
        """Initialize the AdvancedBacktester with configuration from config.json.

        Args:
            walk_forward_analysis (bool): Whether to perform walk-forward optimization.
            monte_carlo_simulations (int): Number of Monte Carlo simulations to run.
            realistic_execution (bool): Whether to include slippage and fees in execution.
        """
        self.walk_forward_analysis = walk_forward_analysis
        self.monte_carlo_simulations = monte_carlo_simulations
        self.realistic_execution = realistic_execution
        self.exchange = ccxt.kraken()
        
        # Load config.json from ~/user_data
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.pairs = config['exchange']['pair_whitelist']
            self.stake_currency = config['stake_currency']
        except FileNotFoundError:
            print("Error: config.json not found at ~/user_data/config.json")
            self.pairs = ["BTC/USDT", "ETH/USDT"]  # Fallback
            self.stake_currency = "USDT"
        except KeyError as e:
            print(f"Error: Missing key {e} in config.json")
            self.pairs = ["BTC/USDT", "ETH/USDT"]  # Fallback
            self.stake_currency = "USDT"

    def fetch_data(self, pair: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch historical OHLCV data for a given pair and timeframe."""
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            return pd.DataFrame()

    def backtest_strategy(self, strategy_class, timeframe: str = '5m', limit: int = 1000, 
                         initial_capital: float = 1000.0, stake_amount: float = 100.0, 
                         slippage: float = 0.001, fee: float = 0.001, param_grid: dict = None) -> dict:
        """Backtest a trading strategy across all pairs with comprehensive features.

        Args:
            strategy_class: The strategy class (e.g., MockAIStrategy) that implements generate_signals.
            timeframe (str): Timeframe for historical data (e.g., '5m').
            limit (int): Number of candles to fetch.
            initial_capital (float): Starting capital for the simulation.
            stake_amount (float): Fixed stake amount per trade.
            slippage (float): Slippage factor per trade (e.g., 0.001 for 0.1%).
            fee (float): Fee per trade as a fraction (e.g., 0.001 for 0.1%).
            param_grid (dict): Dictionary of parameter combinations for optimization.

        Returns:
            dict: Dictionary containing performance metrics and best parameters for each pair.
        """
        if param_grid is None:
            param_grid = {
                'rsi_buy_threshold': [20, 25, 30, 35, 40],
                'rsi_sell_threshold': [60, 65, 70, 75, 80]
            }

        results = {}
        for pair in self.pairs:
            print(f"Backtesting {pair}...")
            data = self.fetch_data(pair, timeframe, limit)
            if data.empty:
                results[pair] = {'error': f"No data fetched for {pair}"}
                continue

            # Use MyManualStrategy to compute indicators
            strategy_instance = MyManualStrategy(config={})
            data = strategy_instance.populate_indicators(data.copy(), {'pair': pair})

            # Split into in-sample (80%) and out-of-sample (20%)
            in_sample_size = int(0.8 * len(data))
            in_sample_data = data.iloc[:in_sample_size]
            out_sample_data = data.iloc[in_sample_size:]

            if self.walk_forward_analysis:
                # Walk-forward analysis with 5 folds
                tscv = TimeSeriesSplit(n_splits=5)
                testing_sharpe_ratios = []
                best_params_list = []

                for train_idx, test_idx in tscv.split(in_sample_data):
                    train_data = in_sample_data.iloc[train_idx]
                    test_data = in_sample_data.iloc[test_idx]
                    
                    if train_data.empty or test_data.empty:
                        continue

                    # Optimize parameters on training data
                    best_sharpe = -np.inf
                    best_params = None
                    for params in ParameterGrid(param_grid):
                        strategy = strategy_class(**params)
                        signals = strategy.generate_signals(train_data)
                        portfolio = simulate_trades(train_data, signals, initial_capital, stake_amount, slippage, fee)
                        sharpe = calculate_sharpe_ratio(portfolio['portfolio_value'])
                        if not pd.isna(sharpe) and sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params

                    # Test best parameters on testing data
                    if best_params:
                        strategy = strategy_class(**best_params)
                        signals = strategy.generate_signals(test_data)
                        portfolio = simulate_trades(test_data, signals, initial_capital, stake_amount, slippage, fee)
                        test_sharpe = calculate_sharpe_ratio(portfolio['portfolio_value'])
                        if not pd.isna(test_sharpe):
                            testing_sharpe_ratios.append(test_sharpe)
                        best_params_list.append(best_params)

                avg_testing_sharpe = np.nanmean(testing_sharpe_ratios) if testing_sharpe_ratios else np.nan

                # Monte Carlo simulations
                mc_sharpe_ratios = []
                if self.monte_carlo_simulations > 0 and best_params_list:
                    # Use the best parameters from the last fold for simulation
                    last_best_params = best_params_list[-1]
                    mc_strategy = strategy_class(**last_best_params)
                    for _ in range(self.monte_carlo_simulations):
                        simulated_path = generate_simulated_path(in_sample_data)
                        # Recalculate indicators on the new synthetic data path
                        simulated_data_with_indicators = strategy_instance.populate_indicators(simulated_path.copy(), {'pair': pair})
                        signals = mc_strategy.generate_signals(simulated_data_with_indicators)
                        portfolio = simulate_trades(simulated_data_with_indicators, signals, initial_capital, stake_amount, slippage, fee)
                        sharpe = calculate_sharpe_ratio(portfolio['portfolio_value'])
                        if not np.isnan(sharpe):
                            mc_sharpe_ratios.append(sharpe)


                # Out-of-sample testing
                out_sample_sharpe = np.nan
                if best_params_list:
                    strategy = strategy_class(**best_params_list[-1])
                    signals = strategy.generate_signals(out_sample_data)
                    portfolio = simulate_trades(out_sample_data, signals, initial_capital, stake_amount, slippage, fee)
                    out_sample_sharpe = calculate_sharpe_ratio(portfolio['portfolio_value'])

                # Compile results for the pair
                results[pair] = {
                    'avg_testing_sharpe': avg_testing_sharpe,
                    'mc_sharpe_mean': np.nanmean(mc_sharpe_ratios) if mc_sharpe_ratios else np.nan,
                    'mc_sharpe_std': np.nanstd(mc_sharpe_ratios) if mc_sharpe_ratios else np.nan,
                    'out_sample_sharpe': out_sample_sharpe,
                    'best_params': best_params_list[-1] if best_params_list else None,
                }
            else:
                # Basic backtest without walk-forward
                strategy = strategy_class(**list(ParameterGrid(param_grid))[0])
                signals = strategy.generate_signals(data)
                portfolio = simulate_trades(data, signals, initial_capital, stake_amount, slippage, fee)
                sharpe = calculate_sharpe_ratio(portfolio['portfolio_value'])
                results[pair] = {'sharpe_ratio': sharpe}

        return results

# Example usage
if __name__ == "__main__":
    backtester = AdvancedBacktester()
    param_grid = {
        'rsi_buy_threshold': [20, 25, 30, 35, 40],
        'rsi_sell_threshold': [60, 65, 70, 75, 80]
    }
    results = backtester.backtest_strategy(
        strategy_class=MockAIStrategy,
        timeframe='5m',
        limit=2000, # Increased limit for more robust testing
        initial_capital=1000.0,
        stake_amount=100.0,
        slippage=0.001,
        fee=0.001,
        param_grid=param_grid
    )
    print(json.dumps(results, indent=2, default=lambda x: str(x) if pd.isna(x) else x))
