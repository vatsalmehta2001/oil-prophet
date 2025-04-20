"""
LSTM with Attention mechanism for oil price forecasting.

This module implements an LSTM network with an attention layer to improve forecasting
accuracy by focusing on the most relevant parts of the input sequence.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Concatenate
from tensorflow.keras.layers import Layer, Lambda, Attention, TimeDistributed, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AttentionLayer(Layer):
    """
    Attention layer for sequence-to-sequence models.
    
    This layer computes attention weights for each time step in the input sequence,
    allowing the model to focus on the most relevant parts of the sequence.
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create trainable weight variables for this layer
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Calculate attention scores - using tf.matmul for better shape compatibility
        e = tf.nn.tanh(tf.matmul(
            tf.reshape(x, (-1, x.shape[-1])),  # Reshape to 2D: (batch*time, features)
            self.W  # Shape: (features, 1)
        ) + tf.reshape(self.b, (1, -1)))  # Reshape bias for broadcasting
        
        # Reshape back to original batch and time dimensions
        e = tf.reshape(e, (-1, x.shape[1]))  # Shape: (batch, time)
        
        # Calculate attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights to input (with proper broadcasting)
        a_expanded = tf.expand_dims(a, axis=2)  # Shape: (batch, time, 1)
        weighted_input = x * a_expanded  # Shape: (batch, time, features)
        
        # Sum over the time dimension
        context_vector = tf.reduce_sum(weighted_input, axis=1)  # Shape: (batch, features)
        
        return context_vector
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


class LSTMWithAttention:
    """
    LSTM model with attention mechanism for time series forecasting.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_dim: int,
        lstm_units: List[int] = [128, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = True
    ):
        """
        Initialize the LSTM model with attention.
        
        Args:
            input_shape: Shape of input features (time steps, features)
            output_dim: Dimension of the output (forecast horizon)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            bidirectional: Whether to use bidirectional LSTM layers
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> Model:
        """
        Build the LSTM model with attention.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # LSTM layers
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or True  # Always True for attention
            
            if self.bidirectional:
                x = Bidirectional(
                    LSTM(units, return_sequences=return_sequences)
                )(x)
            else:
                x = LSTM(units, return_sequences=return_sequences)(x)
            
            # Add dropout after each LSTM layer
            x = Dropout(self.dropout_rate)(x)
        
        # Use TensorFlow's built-in attention mechanism
        # Create a query vector to attend to the sequence
        query = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(x)
        
        # Apply attention
        attention = Attention()([query, x])
        attention = Lambda(lambda x: tf.squeeze(x, axis=1))(attention)
        
        # Output layer
        outputs = Dense(self.output_dim, activation='linear')(attention)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built LSTM-Attention model with {len(self.lstm_units)} LSTM layers")
        logger.info(f"Model architecture: {[layer.__class__.__name__ for layer in model.layers]}")
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        model_path: str = 'models/lstm_attention.h5',
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            model_path: Path to save the best model
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Dictionary containing training history
        """
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info(f"Starting model training with {X_train.shape[0]} samples")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        
        # Log training results
        logger.info(f"Model training completed after {len(history.epoch)} epochs")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate additional metrics
        y_pred = self.predict(X_test)
        mse = np.mean(np.square(y_test - y_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'loss': float(loss),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape)
        }
        
        logger.info(f"Model evaluation results: {metrics}")
        
        return metrics
    
    def save(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save the model and its metadata.
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save model metadata
        """
        # Save the model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata if path is provided
        if metadata_path:
            metadata = {
                'input_shape': self.input_shape,
                'output_dim': self.output_dim,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'bidirectional': self.bidirectional,
                'history': self.history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None) -> 'LSTMWithAttention':
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            metadata_path: Path to the model metadata
            
        Returns:
            LSTMWithAttention instance
        """
        # Load metadata if provided
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create instance with the same parameters
            instance = cls(
                input_shape=tuple(metadata['input_shape']),
                output_dim=metadata['output_dim'],
                lstm_units=metadata['lstm_units'],
                dropout_rate=metadata['dropout_rate'],
                learning_rate=metadata['learning_rate'],
                bidirectional=metadata['bidirectional']
            )
            
            # Load training history
            instance.history = metadata.get('history')
        else:
            # If no metadata, create an instance with default parameters
            # and update input_shape and output_dim after loading the model
            instance = cls(
                input_shape=(None, None),
                output_dim=1
            )
        
        try:
            # First try loading with custom objects for the AttentionLayer
            custom_objects = {'AttentionLayer': AttentionLayer}
            loaded_model = load_model(model_path, custom_objects=custom_objects)
        except:
            # If that fails, try loading with standard objects (for the built-in attention mechanism)
            logger.info("Failed to load with custom AttentionLayer, trying standard loading...")
            loaded_model = load_model(model_path)
        
        # Replace the model in the instance
        instance.model = loaded_model
        
        # Update input_shape and output_dim based on the loaded model
        instance.input_shape = loaded_model.input_shape[1:]
        instance.output_dim = loaded_model.output_shape[1]
        
        logger.info(f"Model loaded from {model_path}")
        return instance
    
    def plot_history(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation loss.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(self.history['loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history['mae'], label='Training MAE')
        ax2.plot(self.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        return fig
    
    def plot_predictions(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        dates: Optional[pd.DatetimeIndex] = None,
        n_samples: int = 100,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot test predictions against actual values.
        
        Args:
            X_test: Test features
            y_test: Test targets
            dates: Optional dates for x-axis
            n_samples: Number of samples to plot
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Generate predictions
        y_pred = self.predict(X_test[:n_samples])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use dates if provided, otherwise use indices
        x_values = dates[:n_samples] if dates is not None else np.arange(n_samples)
        
        # Forecast horizon
        horizon = y_test.shape[1]
        
        # Plot actual values
        for i in range(n_samples):
            if i == 0:
                ax.plot(x_values[i:i+horizon], y_test[i], 'b-', alpha=0.5, label='Actual')
            else:
                ax.plot(x_values[i:i+horizon], y_test[i], 'b-', alpha=0.5)
        
        # Plot predictions
        for i in range(n_samples):
            if i == 0:
                ax.plot(x_values[i:i+horizon], y_pred[i], 'r--', alpha=0.7, label='Predicted')
            else:
                ax.plot(x_values[i:i+horizon], y_pred[i], 'r--', alpha=0.7)
        
        ax.set_title(f'LSTM-Attention Predictions vs Actual Values (first {n_samples} samples)')
        ax.set_xlabel('Date' if dates is not None else 'Time Step')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        # Format dates if provided
        if dates is not None:
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import OilDataProcessor
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and prepare data
    processor = OilDataProcessor()
    dataset = processor.prepare_dataset(
        oil_type="brent", 
        freq="daily", 
        window_size=30, 
        forecast_horizon=7
    )
    
    # Define model parameters
    input_shape = (dataset['X_train'].shape[1], dataset['X_train'].shape[2])
    output_dim = dataset['y_train'].shape[1]
    
    # Create and train the model
    model = LSTMWithAttention(
        input_shape=input_shape,
        output_dim=output_dim,
        lstm_units=[128, 64],
        dropout_rate=0.2,
        bidirectional=True
    )
    
    # Train the model
    history = model.fit(
        dataset['X_train'],
        dataset['y_train'],
        dataset['X_val'],
        dataset['y_val'],
        batch_size=32,
        epochs=50,
        patience=10,
        model_path='models/lstm_attention_brent_daily.h5'
    )
    
    # Evaluate the model
    metrics = model.evaluate(dataset['X_test'], dataset['y_test'])
    print(f"Test metrics: {metrics}")
    
    # Plot training history
    model.plot_history(save_path='notebooks/lstm_attention_history.png')
    
    # Plot predictions
    model.plot_predictions(
        dataset['X_test'],
        dataset['y_test'],
        dates=dataset['dates_test'],
        save_path='notebooks/lstm_attention_predictions.png'
    )
    
    # Save model and metadata
    model.save(
        'models/lstm_attention_brent_daily.h5',
        'models/lstm_attention_brent_daily_metadata.json'
    )