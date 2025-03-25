"""
Ensemble model for Oil Prophet.

This module implements an ensemble forecasting approach that combines predictions from
multiple models, including CEEMDAN-LSTM hybrids and traditional forecasting methods.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import logging
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

from src.models.lstm_attention import LSTMWithAttention
from src.models.ceemdan import CEEMDANDecomposer
from src.data.preprocessing import OilDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HybridCEEMDANLSTM:
    """
    Hybrid model combining CEEMDAN decomposition with LSTM-Attention forecasting.
    
    This model decomposes the time series into IMFs using CEEMDAN, trains separate
    LSTM-Attention models for each IMF, and then combines the predictions.
    """
    
    def __init__(
        self,
        window_size: int = 30,
        forecast_horizon: int = 7,
        ceemdan_params: Optional[Dict[str, Any]] = None,
        lstm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the hybrid model.
        
        Args:
            window_size: Number of time steps to use as input features
            forecast_horizon: Number of time steps to predict
            ceemdan_params: Parameters for CEEMDAN decomposition
            lstm_params: Parameters for LSTM models
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        # Default CEEMDAN parameters
        self.ceemdan_params = {
            'ensemble_size': 100,
            'noise_scale': 0.05,
            'max_imfs': None,
            'random_state': 42
        }
        
        # Override with provided parameters
        if ceemdan_params:
            self.ceemdan_params.update(ceemdan_params)
        
        # Default LSTM parameters
        self.lstm_params = {
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'bidirectional': True
        }
        
        # Override with provided parameters
        if lstm_params:
            self.lstm_params.update(lstm_params)
        
        # Initialize decomposer and models
        self.decomposer = CEEMDANDecomposer(**self.ceemdan_params)
        self.models = []
        self.imfs = None
        self.scalers = {}
        
    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        Decompose the signal into IMFs using CEEMDAN.
        
        Args:
            signal: The time series signal to decompose
            
        Returns:
            Array of IMFs
        """
        self.imfs = self.decomposer.decompose(signal)
        return self.imfs
    
    def prepare_imf_data(
        self, 
        imf: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Prepare a single IMF for LSTM training.
        
        Args:
            imf: Intrinsic Mode Function
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Create sliding windows
        X, y = [], []
        for i in range(len(imf) - self.window_size - self.forecast_horizon + 1):
            X.append(imf[i:i + self.window_size])
            y.append(imf[i + self.window_size:i + self.window_size + self.forecast_horizon])
        
        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)
        
        # Split into train, validation, and test sets
        n_samples = len(X)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def train(
        self,
        signal: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        model_dir: str = 'models/hybrid',
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the hybrid model.
        
        Args:
            signal: Full time series signal
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            batch_size: Batch size for LSTM training
            epochs: Maximum number of epochs for LSTM training
            patience: Patience for early stopping
            model_dir: Directory to save the models
            verbose: Verbosity level for LSTM training
            
        Returns:
            Dictionary containing training metrics
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Decompose the signal if not already done
        if self.imfs is None:
            self.imfs = self.decompose(signal)
        
        # Initialize models list
        self.models = []
        
        # Training metrics
        all_metrics = {}
        
        # Train a separate LSTM model for each IMF
        for i, imf in enumerate(self.imfs):
            logger.info(f"Training model for IMF {i+1}/{len(self.imfs)}")
            
            # Prepare data for this IMF
            imf_data = self.prepare_imf_data(imf, test_size, validation_size)
            
            # Create and train LSTM model
            model = LSTMWithAttention(
                input_shape=(self.window_size, 1),
                output_dim=self.forecast_horizon,
                lstm_units=self.lstm_params['lstm_units'],
                dropout_rate=self.lstm_params['dropout_rate'],
                learning_rate=self.lstm_params['learning_rate'],
                bidirectional=self.lstm_params['bidirectional']
            )
            
            # Train the model
            history = model.fit(
                imf_data['X_train'],
                imf_data['y_train'],
                imf_data['X_val'],
                imf_data['y_val'],
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                model_path=f"{model_dir}/lstm_imf_{i}.h5",
                verbose=verbose
            )
            
            # Evaluate the model
            metrics = model.evaluate(imf_data['X_test'], imf_data['y_test'])
            
            # Add to models list
            self.models.append(model)
            
            # Save metrics
            all_metrics[f"imf_{i}"] = {
                'history': history,
                'metrics': metrics
            }
            
            # Log progress
            logger.info(f"Model for IMF {i+1} trained. Test RMSE: {metrics['rmse']:.4f}")
        
        # Save the decomposer
        self.decomposer.save(f"{model_dir}/ceemdan.pkl")
        
        # Save hybrid model metadata
        metadata = {
            'window_size': self.window_size,
            'forecast_horizon': self.forecast_horizon,
            'ceemdan_params': self.ceemdan_params,
            'lstm_params': self.lstm_params,
            'num_imfs': len(self.imfs),
            'metrics': all_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{model_dir}/hybrid_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4, default=lambda o: str(o))
        
        logger.info(f"Hybrid model training completed and saved to {model_dir}")
        
        return all_metrics
    
    def predict(
        self, 
        signal: np.ndarray, 
        steps_ahead: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate predictions using the hybrid model.
        
        Args:
            signal: Time series signal to predict from
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Predicted values
        """
        if steps_ahead is None:
            steps_ahead = self.forecast_horizon
        
        if len(self.models) == 0:
            raise ValueError("No trained models available. Train the model first.")
        
        # Ensure signal has enough data for the window
        if len(signal) < self.window_size:
            raise ValueError(f"Signal length ({len(signal)}) is less than window_size ({self.window_size})")
        
        # Use the last window_size values for prediction
        input_signal = signal[-self.window_size:]
        
        # Decompose the input signal
        decomposed = self.decomposer.decompose(signal)
        
        # Make predictions for each IMF
        imf_predictions = []
        
        for i, (imf, model) in enumerate(zip(decomposed, self.models)):
            # Prepare input for this IMF
            imf_input = imf[-self.window_size:].reshape(1, self.window_size, 1)
            
            # Generate prediction
            imf_pred = model.predict(imf_input)[0]
            
            # Append to predictions list
            imf_predictions.append(imf_pred[:steps_ahead])
        
        # Combine predictions from all IMFs
        combined_predictions = np.sum(imf_predictions, axis=0)
        
        return combined_predictions
    
    def save(self, model_dir: str) -> None:
        """
        Save the hybrid model.
        
        Args:
            model_dir: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save CEEMDAN decomposer
        self.decomposer.save(f"{model_dir}/ceemdan.pkl")
        
        # Save each LSTM model
        for i, model in enumerate(self.models):
            model.save(
                f"{model_dir}/lstm_imf_{i}.h5",
                f"{model_dir}/lstm_imf_{i}_metadata.json"
            )
        
        # Save hybrid model metadata
        metadata = {
            'window_size': self.window_size,
            'forecast_horizon': self.forecast_horizon,
            'ceemdan_params': self.ceemdan_params,
            'lstm_params': self.lstm_params,
            'num_imfs': len(self.imfs) if self.imfs is not None else 0,
            'num_models': len(self.models),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{model_dir}/hybrid_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4, default=lambda o: str(o))
        
        logger.info(f"Hybrid model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'HybridCEEMDANLSTM':
        """
        Load a hybrid model from directory.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            HybridCEEMDANLSTM instance
        """
        # Load metadata
        with open(f"{model_dir}/hybrid_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance with the same parameters
        instance = cls(
            window_size=metadata['window_size'],
            forecast_horizon=metadata['forecast_horizon'],
            ceemdan_params=metadata['ceemdan_params'],
            lstm_params=metadata['lstm_params']
        )
        
        # Load CEEMDAN decomposer
        instance.decomposer = CEEMDANDecomposer.load(f"{model_dir}/ceemdan.pkl")
        instance.imfs = instance.decomposer.imfs
        
        # Load LSTM models
        instance.models = []
        for i in range(metadata['num_models']):
            model = LSTMWithAttention.load(
                f"{model_dir}/lstm_imf_{i}.h5",
                f"{model_dir}/lstm_imf_{i}_metadata.json"
            )
            instance.models.append(model)
        
        logger.info(f"Hybrid model loaded from {model_dir} with {len(instance.models)} component models")
        
        return instance


class EnsembleForecaster:
    """
    Ensemble forecasting model that combines predictions from multiple models.
    """
    
    def __init__(
        self,
        models: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize the ensemble forecaster.
        
        Args:
            models: List of forecasting models
            weights: List of weights for each model (must sum to 1)
        """
        self.models = models or []
        self.weights = weights
        
        # Validate weights if provided
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
            
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
        
        # Metrics for each model
        self.metrics = {}
        
    def add_model(self, model: Any, weight: Optional[float] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Forecasting model
            weight: Weight for this model (if None, weights will be recalculated)
        """
        self.models.append(model)
        
        # Recalculate weights if not provided
        if weight is None:
            n_models = len(self.models)
            self.weights = [1.0 / n_models] * n_models
        else:
            # Add new weight and renormalize
            if self.weights is None:
                self.weights = [1.0]
            else:
                weight_sum = sum(self.weights) + weight
                self.weights = [w / weight_sum for w in self.weights] + [weight / weight_sum]
        
        logger.info(f"Added model to ensemble. Total models: {len(self.models)}")
    
    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Generate predictions using the ensemble.
        
        Args:
            X: Input features for prediction (can be different for each model)
            
        Returns:
            Ensemble prediction
        """
        if len(self.models) == 0:
            raise ValueError("No models in the ensemble. Add models first.")
        
        # Initialize predictions
        all_predictions = []
        
        # Generate predictions from each model
        for i, model in enumerate(self.models):
            # Get input for this model
            if isinstance(X, list):
                x_input = X[i]
            else:
                x_input = X
            
            # Generate prediction
            prediction = model.predict(x_input)
            
            # Append to predictions list
            all_predictions.append(prediction)
        
        # Combine predictions using weights
        if self.weights is None:
            # Equal weights if not specified
            ensemble_prediction = np.mean(all_predictions, axis=0)
        else:
            # Weighted average
            ensemble_prediction = np.zeros_like(all_predictions[0])
            for i, pred in enumerate(all_predictions):
                ensemble_prediction += self.weights[i] * pred
        
        return ensemble_prediction
    
    def evaluate(
        self, 
        X: Union[np.ndarray, List[np.ndarray]], 
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble and individual models.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Evaluate individual models
        individual_metrics = []
        individual_predictions = []
        
        for i, model in enumerate(self.models):
            # Get input for this model
            if isinstance(X, list):
                x_input = X[i]
            else:
                x_input = X
            
            # Generate prediction
            prediction = model.predict(x_input)
            individual_predictions.append(prediction)
            
            # Calculate metrics
            mse = mean_squared_error(y, prediction)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, prediction)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            }
            
            individual_metrics.append(metrics)
            self.metrics[f"model_{i}"] = metrics
        
        # Generate ensemble prediction
        ensemble_prediction = self.predict(X)
        
        # Calculate ensemble metrics
        ensemble_mse = mean_squared_error(y, ensemble_prediction)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_mae = mean_absolute_error(y, ensemble_prediction)
        
        ensemble_metrics = {
            'mse': float(ensemble_mse),
            'rmse': float(ensemble_rmse),
            'mae': float(ensemble_mae)
        }
        
        self.metrics['ensemble'] = ensemble_metrics
        
        # Log evaluation results
        for i, metrics in enumerate(individual_metrics):
            logger.info(f"Model {i} metrics: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        
        logger.info(f"Ensemble metrics: MSE={ensemble_mse:.4f}, RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f}")
        
        return self.metrics
    
    def plot_predictions(
        self, 
        X: Union[np.ndarray, List[np.ndarray]], 
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot predictions from individual models and the ensemble.
        
        Args:
            X: Input features
            y: Target values
            dates: Optional dates for x-axis
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Generate predictions
        individual_predictions = []
        for i, model in enumerate(self.models):
            # Get input for this model
            if isinstance(X, list):
                x_input = X[i]
            else:
                x_input = X
            
            # Generate prediction
            prediction = model.predict(x_input)
            individual_predictions.append(prediction)
        
        # Generate ensemble prediction
        ensemble_prediction = self.predict(X)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Use dates if provided, otherwise use indices
        x_values = dates if dates is not None else np.arange(len(y))
        
        # Top plot: Individual model predictions
        for i, pred in enumerate(individual_predictions):
            label = f"Model {i+1}"
            ax1.plot(x_values, pred, '--', alpha=0.7, label=label)
        
        # Add actual values
        ax1.plot(x_values, y, 'k-', label='Actual')
        
        ax1.set_title('Individual Model Predictions vs Actual Values')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Bottom plot: Ensemble prediction
        ax2.plot(x_values, ensemble_prediction, 'r-', linewidth=2, label='Ensemble')
        ax2.plot(x_values, y, 'k-', label='Actual')
        
        ax2.set_title('Ensemble Prediction vs Actual Values')
        ax2.set_xlabel('Date' if dates is not None else 'Time Step')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        # Format dates if provided
        if dates is not None:
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")
        
        return fig
    
    def save(self, model_dir: str) -> None:
        """
        Save the ensemble model.
        
        Args:
            model_dir: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble metadata
        metadata = {
            'num_models': len(self.models),
            'weights': self.weights,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{model_dir}/ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4, default=lambda o: str(o))
        
        # Save weights
        if self.weights is not None:
            np.save(f"{model_dir}/ensemble_weights.npy", np.array(self.weights))
        
        logger.info(f"Ensemble model metadata saved to {model_dir}")
        logger.info(f"Note: Individual models need to be saved separately")
    
    @classmethod
    def load(cls, model_dir: str, models: Optional[List[Any]] = None) -> 'EnsembleForecaster':
        """
        Load an ensemble model from directory.
        
        Args:
            model_dir: Directory containing the saved model
            models: Optional list of pre-loaded models
            
        Returns:
            EnsembleForecaster instance
        """
        # Load metadata
        with open(f"{model_dir}/ensemble_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load weights if available
        try:
            weights = np.load(f"{model_dir}/ensemble_weights.npy").tolist()
        except FileNotFoundError:
            weights = metadata.get('weights')
        
        # Create instance
        instance = cls(models=models, weights=weights)
        
        # Load metrics if available
        if 'metrics' in metadata:
            instance.metrics = metadata['metrics']
        
        logger.info(f"Ensemble model loaded from {model_dir}")
        
        return instance


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import OilDataProcessor
    
    # Load and prepare data
    processor = OilDataProcessor()
    dataset = processor.prepare_dataset(
        oil_type="brent", 
        freq="daily", 
        window_size=30, 
        forecast_horizon=7
    )
    
    # Create a hybrid CEEMDAN-LSTM model
    hybrid_model = HybridCEEMDANLSTM(
        window_size=30,
        forecast_horizon=7
    )
    
    # Train the hybrid model
    metrics = hybrid_model.train(
        dataset['original_data']['Price'].values,
        model_dir='models/hybrid_brent_daily'
    )
    
    # Create an ensemble with the hybrid model
    ensemble = EnsembleForecaster()
    ensemble.add_model(hybrid_model)
    
    # Evaluate the ensemble
    ensemble_metrics = ensemble.evaluate(
        dataset['X_test'],
        dataset['y_test']
    )
    
    # Save the ensemble
    ensemble.save('models/ensemble_brent_daily')