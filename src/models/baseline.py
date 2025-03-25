"""
Baseline forecasting models for oil price prediction.

This module implements simple baseline forecasting models for comparison with 
more advanced approaches like the CEEMDAN-LSTM hybrid.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NaiveForecaster:
    """
    Implements simple baseline forecasting methods.
    """
    
    def __init__(self, method: str = 'last_value'):
        """
        Initialize the naive forecaster.
        
        Args:
            method: Forecasting method to use:
                - 'last_value': Use the last observed value for all future predictions
                - 'mean': Use the mean of historical values
                - 'drift': Extrapolate based on the trend between first and last observations
        """
        valid_methods = ['last_value', 'mean', 'drift']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        self.method = method
        self.history = None
    
    def fit(self, train_data: np.ndarray) -> None:
        """
        Fit the forecaster to training data.
        
        Args:
            train_data: Historical time series data
        """
        self.history = train_data
        logger.info(f"Fitted naive forecaster with method '{self.method}' on {len(train_data)} samples")
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if self.history is None:
            raise ValueError("Forecaster not fitted. Call fit() first.")
        
        if self.method == 'last_value':
            # Use the last value for all future steps
            return np.full(steps, self.history[-1])
        
        elif self.method == 'mean':
            # Use the mean of historical values
            return np.full(steps, np.mean(self.history))
        
        elif self.method == 'drift':
            # Calculate the average change per period
            first, last = self.history[0], self.history[-1]
            change_per_period = (last - first) / (len(self.history) - 1)
            
            # Extrapolate future values
            return np.array([last + (i + 1) * change_per_period for i in range(steps)])
    
    def evaluate(self, test_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the forecaster on test data.
        
        Args:
            test_data: Actual values to compare with forecasts
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate forecasts
        forecasts = self.predict(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecasts)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, forecasts)
        mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
        
        logger.info(f"Naive forecaster evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return metrics


class ARIMAForecaster:
    """
    ARIMA model for time series forecasting.
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize the ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, S) or None for non-seasonal model
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray) -> None:
        """
        Fit the ARIMA model to training data.
        
        Args:
            train_data: Historical time series data
        """
        try:
            # Create the ARIMA model
            if self.seasonal_order:
                self.model = ARIMA(
                    train_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order
                )
            else:
                self.model = ARIMA(train_data, order=self.order)
            
            # Fit the model
            self.fitted_model = self.model.fit()
            
            logger.info(f"Fitted ARIMA{self.order} model on {len(train_data)} samples")
            
        except Exception as e:
            logger.error(f"ARIMA model fitting failed: {str(e)}")
            raise
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast
    
    def evaluate(self, test_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ARIMA model on test data.
        
        Args:
            test_data: Actual values to compare with forecasts
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate forecasts
        forecasts = self.predict(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecasts)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, forecasts)
        mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
        
        logger.info(f"ARIMA model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the ARIMA model.
        
        Args:
            filepath: Path to save the model
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model parameters
        model_data = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'model_str': str(self.fitted_model.summary())
        }
        
        # Save the fitted model
        self.fitted_model.save(filepath)
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=4)
        
        logger.info(f"ARIMA model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ARIMAForecaster':
        """
        Load a saved ARIMA model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            ARIMAForecaster instance
        """
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        # Create instance with the same parameters
        instance = cls(
            order=model_data['order'],
            seasonal_order=model_data['seasonal_order']
        )
        
        # Load the fitted model
        instance.fitted_model = ARIMA.load(filepath)
        
        logger.info(f"ARIMA model loaded from {filepath}")
        return instance


class ExponentialSmoothingForecaster:
    """
    Exponential Smoothing model for time series forecasting.
    """
    
    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False
    ):
        """
        Initialize the Exponential Smoothing forecaster.
        
        Args:
            trend: Type of trend component ('add', 'mul', or None)
            seasonal: Type of seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a seasonal cycle
            damped_trend: Whether to use damped trend
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray) -> None:
        """
        Fit the Exponential Smoothing model to training data.
        
        Args:
            train_data: Historical time series data
        """
        try:
            # Ensure seasonal_periods is set if seasonal component is specified
            if self.seasonal and not self.seasonal_periods:
                raise ValueError("seasonal_periods must be specified when seasonal is not None")
            
            # Create the model
            self.model = ExponentialSmoothing(
                train_data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend
            )
            
            # Fit the model
            self.fitted_model = self.model.fit()
            
            logger.info(f"Fitted Exponential Smoothing model on {len(train_data)} samples")
            
        except Exception as e:
            logger.error(f"Exponential Smoothing model fitting failed: {str(e)}")
            raise
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast
    
    def evaluate(self, test_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the Exponential Smoothing model on test data.
        
        Args:
            test_data: Actual values to compare with forecasts
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate forecasts
        forecasts = self.predict(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecasts)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, forecasts)
        mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
        
        logger.info(f"Exponential Smoothing model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the Exponential Smoothing model.
        
        Args:
            filepath: Path to save the model
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model parameters and fitted model
        model_data = {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'fitted_model': self.fitted_model
        }
        
        # Save using joblib
        joblib.dump(model_data, filepath)
        
        logger.info(f"Exponential Smoothing model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ExponentialSmoothingForecaster':
        """
        Load a saved Exponential Smoothing model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            ExponentialSmoothingForecaster instance
        """
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create instance with the same parameters
        instance = cls(
            trend=model_data['trend'],
            seasonal=model_data['seasonal'],
            seasonal_periods=model_data['seasonal_periods'],
            damped_trend=model_data['damped_trend']
        )
        
        # Set the fitted model
        instance.fitted_model = model_data['fitted_model']
        
        logger.info(f"Exponential Smoothing model loaded from {filepath}")
        return instance


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import OilDataProcessor
    
    # Load and prepare data
    processor = OilDataProcessor()
    
    try:
        # Load data
        data = processor.load_data(oil_type="brent", freq="daily")
        
        # Use the last 30 days for testing
        train_data = data['Price'].values[:-30]
        test_data = data['Price'].values[-30:]
        
        print(f"Train data: {len(train_data)} samples")
        print(f"Test data: {len(test_data)} samples")
        
        # Naive forecaster
        print("\nNaive Forecaster (last value):")
        naive = NaiveForecaster(method='last_value')
        naive.fit(train_data)
        naive_metrics = naive.evaluate(test_data)
        print(f"  RMSE: {naive_metrics['rmse']:.2f}")
        
        # Naive forecaster with drift
        print("\nNaive Forecaster (drift):")
        naive_drift = NaiveForecaster(method='drift')
        naive_drift.fit(train_data)
        naive_drift_metrics = naive_drift.evaluate(test_data)
        print(f"  RMSE: {naive_drift_metrics['rmse']:.2f}")
        
        # ARIMA model
        print("\nARIMA Model:")
        try:
            arima = ARIMAForecaster(order=(1, 1, 1))
            arima.fit(train_data)
            arima_metrics = arima.evaluate(test_data)
            print(f"  RMSE: {arima_metrics['rmse']:.2f}")
        except Exception as e:
            print(f"  Error: {str(e)}")
        
        # Exponential Smoothing model
        print("\nExponential Smoothing Model:")
        try:
            es = ExponentialSmoothingForecaster(trend='add')
            es.fit(train_data)
            es_metrics = es.evaluate(test_data)
            print(f"  RMSE: {es_metrics['rmse']:.2f}")
        except Exception as e:
            print(f"  Error: {str(e)}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")