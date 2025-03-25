"""
Data preprocessing module for Oil Prophet.

This module provides utilities for loading, cleaning, and preparing
oil price data for model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the project root directory
# This helps with referencing files from the project root
try:
    # If invoked directly
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
except NameError:
    # If invoked from different file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))

class OilDataProcessor:
    """Processor for oil price time series data."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the raw data files
        """
        if data_dir is None:
            self.data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        else:
            self.data_dir = data_dir
            
        logger.info(f"Data directory set to: {self.data_dir}")
        self.scalers = {}
        self.datasets = {}

    def load_data(self, oil_type: str = "brent", freq: str = "daily") -> pd.DataFrame:
        """
        Load oil price data from CSV files.
        
        Args:
            oil_type: Type of oil price data ('brent' or 'wti')
            freq: Frequency of data ('daily', 'weekly', 'monthly', 'year')
            
        Returns:
            DataFrame with parsed dates and prices
        """
        # Construct filename with correct formatting
        filename = f"{oil_type}-{freq}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        logger.info(f"Attempting to load data from: {filepath}")
        
        # Debug: List all files in the directory
        if os.path.exists(self.data_dir):
            files_in_dir = os.listdir(self.data_dir)
            logger.info(f"Files in directory: {files_in_dir}")
        else:
            logger.error(f"Data directory does not exist: {self.data_dir}")
            # Try to find data directory by searching upward
            current_dir = os.getcwd()
            possible_data_dir = None
            for _ in range(5):  # Look up to 5 levels up
                data_dir_candidate = os.path.join(current_dir, "data", "raw")
                if os.path.exists(data_dir_candidate):
                    possible_data_dir = data_dir_candidate
                    break
                current_dir = os.path.dirname(current_dir)
            
            if possible_data_dir:
                logger.info(f"Found alternate data directory: {possible_data_dir}")
                self.data_dir = possible_data_dir
                filepath = os.path.join(self.data_dir, filename)
            else:
                raise FileNotFoundError(f"Cannot find data directory")

        try:
            # Check if file exists
            if not os.path.exists(filepath):
                # Try alternate formatting (without hyphen)
                alternate_filename = f"{oil_type}{freq}.csv"
                alternate_filepath = os.path.join(self.data_dir, alternate_filename)
                
                if os.path.exists(alternate_filepath):
                    logger.info(f"Using alternate file format: {alternate_filename}")
                    filepath = alternate_filepath
                else:
                    available_files = os.listdir(self.data_dir) if os.path.exists(self.data_dir) else []
                    logger.error(f"File not found: {filepath} or {alternate_filepath}")
                    logger.info(f"Available files in directory: {available_files}")
                    raise FileNotFoundError(f"File not found: {filepath} or {alternate_filepath}")
            
            df = pd.read_csv(filepath)
            
            # Ensure column names are standardized
            if 'Date' not in df.columns or 'Price' not in df.columns:
                if len(df.columns) == 2:
                    df.columns = ['Date', 'Price']
                else:
                    raise ValueError(f"Unexpected columns in {filename}: {df.columns}")
            
            # Parse dates and ensure price is float
            df['Date'] = pd.to_datetime(df['Date'])
            df['Price'] = df['Price'].astype(float)
            
            # Sort by date and set it as index
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def prepare_dataset(
        self, 
        oil_type: str = "brent", 
        freq: str = "daily", 
        window_size: int = 30,
        forecast_horizon: int = 7,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Prepare time series data for model training.
        
        Args:
            oil_type: Type of oil price data ('brent' or 'wti')
            freq: Frequency of data ('daily', 'weekly', 'monthly', 'year')
            window_size: Number of time steps to use as input features
            forecast_horizon: Number of time steps to predict
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test arrays
        """
        key = f"{oil_type}_{freq}"
        
        # Load and preprocess data
        df = self.load_data(oil_type, freq)
        
        # Create sliding windows for time series forecasting
        X, y = self._create_windows(df, window_size, forecast_horizon)
        
        # Scale the data
        X_scaled, y_scaled, scaler = self._scale_data(X, y)
        self.scalers[key] = scaler
        
        # Split into train, validation, and test sets
        n_samples = len(X_scaled)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        X_train, y_train = X_scaled[:val_idx], y_scaled[:val_idx]
        X_val, y_val = X_scaled[val_idx:test_idx], y_scaled[val_idx:test_idx]
        X_test, y_test = X_scaled[test_idx:], y_scaled[test_idx:]
        
        # Store the dataset split information
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'dates_test': df.index[test_idx + window_size:],
            'original_data': df
        }
        
        self.datasets[key] = dataset
        
        logger.info(f"Prepared dataset with shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Validation set: X_val: {X_val.shape}, y_val: {y_val.shape}")
        logger.info(f"Test set: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return dataset
    
    def _create_windows(
        self, 
        df: pd.DataFrame, 
        window_size: int, 
        forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for time series forecasting.
        
        Args:
            df: DataFrame with time series data
            window_size: Number of time steps to use as input features
            forecast_horizon: Number of time steps to predict
            
        Returns:
            X and y arrays for the windowed time series
        """
        data = df['Price'].values
        X, y = [], []
        
        for i in range(len(data) - window_size - forecast_horizon + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + forecast_horizon])
        
        return np.array(X).reshape(-1, window_size, 1), np.array(y)
    
    def _scale_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Scale the data using MinMaxScaler.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Scaled X and y arrays, and the fitted scaler
        """
        # Reshape for scaling
        X_reshape = X.reshape(-1, 1)
        y_reshape = y.reshape(-1, 1)
        
        # Fit scaler on both X and y to ensure same scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.vstack((X_reshape, y_reshape)))
        
        # Transform the data
        X_scaled = scaler.transform(X_reshape).reshape(X.shape)
        y_scaled = scaler.transform(y_reshape).reshape(y.shape)
        
        return X_scaled, y_scaled, scaler
    
    def inverse_transform(self, data: np.ndarray, oil_type: str, freq: str) -> np.ndarray:
        """
        Inverse transform scaled predictions to original scale.
        
        Args:
            data: Scaled data to transform back
            oil_type: Type of oil price data ('brent' or 'wti')
            freq: Frequency of data ('daily', 'weekly', 'monthly', 'year')
            
        Returns:
            Data in original scale
        """
        key = f"{oil_type}_{freq}"
        scaler = self.scalers.get(key)
        
        if scaler is None:
            raise ValueError(f"No scaler found for {key}. Did you prepare the dataset first?")
        
        # Reshape for inverse transformation
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
        
        # Inverse transform
        data_original = scaler.inverse_transform(data_reshaped)
        
        # Reshape back to original shape
        return data_original.reshape(original_shape)


if __name__ == "__main__":
    # Example usage
    processor = OilDataProcessor()
    
    # Print available files in data directory
    print(f"Project root: {PROJECT_ROOT}")
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    print(f"Data directory: {data_dir}")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"Available files: {files}")
    else:
        print(f"Data directory does not exist: {data_dir}")
    
    try:
        dataset = processor.prepare_dataset(oil_type="brent", freq="daily", window_size=30, forecast_horizon=7)
        print(f"Dataset prepared with {len(dataset['X_train'])} training samples")
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")