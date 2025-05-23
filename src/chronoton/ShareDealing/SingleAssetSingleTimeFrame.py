# ShareDealing/SingleAssetSingleTimeFrame.py
from typing import Callable, Any
import numpy as np
import pandas as pd
import polars as pl

class ShareSAST:
    def __init__(self, buy_logic=None, sell_logic=None, short_logic=None, cover_logic=None, df=None):
        """
        Initialize the strategy with trading logic and data.
        
        Parameters:
        -----------
        buy_logic : callable, optional
            Function that accepts data and returns True/False for buy signals
        sell_logic : callable, optional
            Function that accepts data and returns True/False for sell signals
        short_logic : callable, optional
            Function that accepts data and returns True/False for short signals
        cover_logic : callable, optional
            Function that accepts data and returns True/False for cover signals
        df : numpy.ndarray, pandas.DataFrame, or polars.DataFrame, optional
            Market data to run the backtest against
        """
        # Validate and set logic callables
        if buy_logic is not None and not callable(buy_logic):
            raise TypeError("Buy logic must be callable")
        self.buy_logic = buy_logic
        
        if sell_logic is not None and not callable(sell_logic):
            raise TypeError("Sell logic must be callable")
        self.sell_logic = sell_logic
        
        if short_logic is not None and not callable(short_logic):
            raise TypeError("Short logic must be callable")
        self.short_logic = short_logic
        
        if cover_logic is not None and not callable(cover_logic):
            raise TypeError("Cover logic must be callable")
        self.cover_logic = cover_logic
        
        # Store the data
        if df is not None:
            try:
            has_polars = True
            except ImportError:
            has_polars = False
            
            is_numpy = isinstance(df, np.ndarray)
            is_pandas = isinstance(df, pd.DataFrame)
            is_polars = has_polars and isinstance(df, pl.DataFrame)
            
            if not (is_numpy or is_pandas or is_polars):
            raise TypeError("Data must be a numpy array, pandas DataFrame, or polars DataFrame")
        self.df = df


    def run_backtest(self, data = self.df, commission=0.0, commission_pct=0.0):
        """
        Execute a single backtest with current strategy parameters.
        
        Parameters:
        -----------
        data : numpy.ndarray, pandas.DataFrame, or polars.DataFrame
            Market data to run the backtest against
        """
        # Validate strategy and data
        self._validate_strategy()
        self._validate_data(data)
        
        # Prepare signals by calling the callables with data - NEED TO CONTINUE FROM HERE*******
        buy_signals, sell_signals, short_signals, cover_signals = self._prepare_signals(data)

        #Make data numpy array if not already
        data = self._prepare_data(data)
        
        # Execute backtest
        equity_curve, trades = self._execute_backtest_loop_ShareDealing(data, signals, commission, commission_pct)
        
        # Store results as attributes rather than returning them
        self.equity_curve = equity_curve
        self.trades = trades
        self.summary = self._calculate_basic_metrics(equity_curve, trades)
        
        return self
    
    def run_optimization(self, data = self.df, params, optimizer="grid", objective="sharpe", max_evals=None):
        """
        Optimize strategy parameters using the specified method.
        
        Parameters:
        -----------
        data : numpy.ndarray, pandas.DataFrame, or polars.DataFrame
            Market data to run the backtest against
        params : dict
            Parameter space to explore, e.g. {"ma_period": (5, 50, 5)}
        optimizer : str or callable
            Optimization method to use ("grid", "random", "bayesian", etc.)
        objective : str or callable
            Metric to optimize ("sharpe", "profit_factor", "return", etc.)
        """
        # Optimization logic
        return self
        
    def analyze(self, detailed=True, monte_carlo=False, plots=True):
        """
        Generate detailed analysis of backtest results.
        
        Parameters:
        -----------
        detailed : bool
            Calculate advanced metrics (drawdown profiles, etc.)
        monte_carlo : bool
            Run Monte Carlo simulations for robustness testing
        plots : bool
            Generate visualization plots
        """
        if not hasattr(self, 'summary'):
            raise RuntimeError("Must run backtest before analysis")
            
        analysis = {}
        
        # Add detailed statistics
        if detailed:
            analysis['advanced_metrics'] = self._calculate_advanced_metrics()
            
        # Add Monte Carlo results
        if monte_carlo:
            analysis['monte_carlo'] = self._run_monte_carlo()
            
        # Generate plots
        if plots:
            analysis['plots'] = self._generate_plots()
            
        self.analysis = analysis
        return self
        
    def clear_backtest_results(self):
        """
        Clear all backtest results (equity curve, trades, summary metrics).
        Useful when running multiple backtests with the same strategy.
        """
        if hasattr(self, 'equity_curve'):
            del self.equity_curve
        
        if hasattr(self, 'trades'):
            del self.trades
        
        if hasattr(self, 'summary'):
            del self.summary
        
        return self

    def clear_analysis(self):
        """
        Clear detailed analysis results (advanced metrics, Monte Carlo, plots).
        Retains basic backtest results.
        """
        if hasattr(self, 'analysis'):
            del self.analysis
        
        return self

    def clear_all(self):
        """
        Clear all results and analysis data.
        """
        self.clear_backtest_results()
        self.clear_analysis()
        return self
        
    def _validate_strategy(self):
        """Check if strategy is properly configured with required logic"""
        # Check for at least one pair of opposing signals
        has_long = self.buy_logic is not None and self.sell_logic is not None
        has_short = self.short_logic is not None and self.cover_logic is not None
        
        if not (has_long or has_short):
            raise ValueError("Strategy must have at least one pair of opposing signals (buy/sell or short/cover)")
    
    def _validate_data(self, data = self.df):
        """Validate data format and check for NAs or infinite values"""
        # Check data type
        try:
            has_polars = True
            import polars as pl
        except ImportError:
            has_polars = False
        
        is_numpy = isinstance(data, np.ndarray)
        is_pandas = isinstance(data, pd.DataFrame)
        is_polars = has_polars and isinstance(data, pl.DataFrame)
        
        if not (is_numpy or is_pandas or is_polars):
            raise TypeError("Data must be a numpy array, pandas DataFrame, or polars DataFrame")
        
        # Check for NAs or infinite values
        if is_numpy:
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Data contains NaN or infinite values")
        elif is_pandas:
            if data.isna().any().any() or np.isinf(data).any().any():
                raise ValueError("Data contains NaN or infinite values")
        elif is_polars:
            # Check for null values
            if data.null_count() > 0:
                raise ValueError("Data contains null values")
            # For infinity checks in polars we need to check numeric columns
            numeric_cols = data.select(pl.col(col) for col in data.columns 
                                      if data[col].dtype in [pl.Float32, pl.Float64])
            if numeric_cols.shape[1] > 0:  # Only if there are numeric columns
                if numeric_cols.select(pl.all().is_infinite()).any():
                    raise ValueError("Data contains infinite values")
        
    def _prepare_data(self, data = self.df):
        """Convert data to numpy array format for processing"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, pl.DataFrame):
            return data.to_numpy()
        else:
            raise TypeError("Data must be a numpy array, pandas DataFrame, or polars DataFrame")
    
    def _prepare_signals(self, data, unit_type = None, stop_loss = None, take_profit = None, stop_type = None, trailing_step = None):
        """
        Apply callables to data to generate signal arrays.

        Parameters:
        -----------
        data : numpy.ndarray, pandas.DataFrame, or polars.DataFrame
            Market data to run the backtest against
        unit_type : str or None. Either "currency" or "percent"
            Type of unit for stop loss and take profit calculations
        stop_loss : float or None
            Stop loss value for the strategy
        take_profit : float or None
            Take profit value for the strategy
        stop_type : str or None. Either "fixed" or "trailing". Only used if stop_loss is not None
            Type of stop loss or take profit calculation
        trailing_step : float or None. Only used if trailing_type is "trailing"
            Trailing step value for the strategy


        Returns:
        --------
        tuple
            Tuple of (buy, sell, short, cover) numpy arrays
        """
        # Determine the length of the data
        if isinstance(data, np.ndarray):
            data_length = data.shape[0]
        elif isinstance(data, pd.DataFrame):
            data_length = len(data)
        elif isinstance(data, pl.DataFrame):
            data_length = len(data)
        else:
            raise TypeError("Data must be a numpy array, pandas DataFrame, or polars DataFrame")
        
        # Create initial arrays of zeros
        buy_signals = np.zeros(data_length, dtype=bool)
        sell_signals = np.zeros(data_length, dtype=bool)
        short_signals = np.zeros(data_length, dtype=bool)
        cover_signals = np.zeros(data_length, dtype=bool)
        
        # Fill arrays with signals from callables when they exist
        if self.buy_logic is not None:
            buy_signals = self.buy_logic(data)
        
        if self.sell_logic is not None:
            sell_signals = self.sell_logic(data)
        
        if self.short_logic is not None:
            short_signals = self.short_logic(data)
        
        if self.cover_logic is not None:
            cover_signals = self.cover_logic(data)
        
        return buy_signals, sell_signals, short_signals, cover_signals

        # Update the signals based on stop loss and take profit logic
        if stop_loss is not None:
            if unit_type == "currency":
                # Apply stop loss logic based on currency.  - NEED TO CONTINUE FROM HERE. Make this a cython function*******
                pass

    def _execute_backtest_loop_ShareDealing(self, data, signals, commission=0.0, commission_pct=0.0):
        """
        Execute the backtest loop using the provided signals.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Market data to run the backtest against
        signals : tuple
            Tuple of (buy, sell, short, cover) numpy arrays
        commission : float
            Fixed commission per trade
        commission_pct : float
            Percentage commission per trade
        
        Returns:
        --------
        tuple
            Tuple of (equity_curve, trades)
        """
        # Placeholder for actual backtest logic
        equity_curve = np.zeros(data.shape[0])
        trades = []

        
        return equity_curve, trades