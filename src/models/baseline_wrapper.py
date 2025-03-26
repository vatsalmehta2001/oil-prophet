"""
Wrapper for baseline forecasting models.

This module provides a unified interface for the different baseline forecasting models.
"""

import numpy as np
from typing import Dict
from src.models.baseline import NaiveForecaster, ARIMAForecaster, ExponentialSmoothingForecaster

class BaselineForecaster:
    """
    Wrapper class for different baseline forecasting models.
    """
    
    def __init__(self, method: str = 'naive', window: int = 7):
        """
        Initialize the baseline forecaster.
        
        Args:
            method: Forecasting method ('naive', 'drift', 'sma', 'ema', 'arima')
            window: Window size for moving average methods
        """
        self.method = method
        self.window = window
        
        if method in ['naive', 'drift', 'mean']:
            self.model = NaiveForecaster(method=method if method != 'naive' else 'last_value')
        elif method == 'sma':
            # Simple Moving Average - implemented as a wrapper around NaiveForecaster
            self.model = NaiveForecaster(method='mean')
            self.window_size = window
        elif method == 'ema':
            # Exponential Moving Average
            self.model = ExponentialSmoothingForecaster(trend=None, seasonal=None)
        elif method == 'arima':
            self.model = ARIMAForecaster(order=(1, 1, 1))
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.history = None
        
    def fit(self, train_data: np.ndarray) -> None:
        """
        Fit the forecaster to training data.
        
        Args:
            train_data: Historical time series data
        """
        self.history = train_data
        
        if self.method == 'sma':
            # For SMA, we only use the last window_size values for the mean
            if len(train_data) >= self.window_size:
                self.model.fit(train_data[-self.window_size:])
            else:
                self.model.fit(train_data)
        else:
            self.model.fit(train_data)
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        return self.model.predict(steps)
    
    def evaluate(self, test_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the forecaster on test data.
        
        Args:
            test_data: Actual values to compare with forecasts
            
        Returns:
            Dictionary with evaluation metrics
        """
        return self.model.evaluate(test_data)