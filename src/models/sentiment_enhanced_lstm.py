"""
Sentiment-Enhanced LSTM Model for Oil Price Forecasting

This module integrates sentiment analysis features with the LSTM with attention model
to create a combined forecasting system that leverages both technical price patterns
and market sentiment indicators for improved prediction accuracy.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Concatenate, GlobalAveragePooling1D, Lambda, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import json
from typing import Tuple, List, Dict, Any, Optional, Union

# Import existing modules
from src.models.lstm_attention import LSTMWithAttention, AttentionLayer
from src.nlp.finbert_sentiment import OilFinBERT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SentimentEnhancedLSTM:
    """
    LSTM model enhanced with sentiment data for time series forecasting.
    """
    
    def __init__(
        self,
        price_input_shape: Tuple[int, int],
        sentiment_input_shape: Tuple[int, int],
        output_dim: int,
        lstm_units: List[int] = [128, 64],
        sentiment_dense_units: List[int] = [32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
        price_weight: float = 0.7,
        sentiment_weight: float = 0.3
    ):
        """
        Initialize the sentiment-enhanced LSTM model.
        
        Args:
            price_input_shape: Shape of price input features (time steps, features)
            sentiment_input_shape: Shape of sentiment input features (time steps, features)
            output_dim: Dimension of the output (forecast horizon)
            lstm_units: List of units for each LSTM layer
            sentiment_dense_units: List of units for sentiment dense layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            bidirectional: Whether to use bidirectional LSTM layers
            price_weight: Weight for price features
            sentiment_weight: Weight for sentiment features
        """
        self.price_input_shape = price_input_shape
        self.sentiment_input_shape = sentiment_input_shape
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.sentiment_dense_units = sentiment_dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.price_weight = price_weight
        self.sentiment_weight = sentiment_weight
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self) -> Model:
        """
        Build the combined model with price and sentiment branches.
        
        Returns:
            Compiled Keras model
        """
        # Price input branch
        price_inputs = Input(shape=self.price_input_shape, name='price_input')
        
        # LSTM layers for price data
        x_price = price_inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            
            if self.bidirectional:
                x_price = Bidirectional(
                    LSTM(units, return_sequences=return_sequences)
                )(x_price)
            else:
                x_price = LSTM(units, return_sequences=return_sequences)(x_price)
            
            # Add dropout after each LSTM layer
            x_price = Dropout(self.dropout_rate)(x_price)
        
        # Sentiment input branch
        sentiment_inputs = Input(shape=self.sentiment_input_shape, name='sentiment_input')
        
        # Process sentiment with TimeDistributed Dense layers
        x_sentiment = sentiment_inputs
        for units in self.sentiment_dense_units:
            x_sentiment = TimeDistributed(Dense(units, activation='relu'))(x_sentiment)
            x_sentiment = TimeDistributed(Dropout(self.dropout_rate))(x_sentiment)
        
        # Use LSTM to process sentiment sequences
        if self.bidirectional:
            x_sentiment = Bidirectional(LSTM(32, return_sequences=False))(x_sentiment)
        else:
            x_sentiment = LSTM(32, return_sequences=False)(x_sentiment)
        
        x_sentiment = Dropout(self.dropout_rate)(x_sentiment)
        
        # Apply weighting to each branch
        price_weighted = Lambda(lambda x: x * self.price_weight)(x_price)
        sentiment_weighted = Lambda(lambda x: x * self.sentiment_weight)(x_sentiment)
        
        # Concatenate the weighted features
        combined = Concatenate()([price_weighted, sentiment_weighted])
        
        # Final dense layers
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(self.dropout_rate)(combined)
        
        # Output layer
        outputs = Dense(self.output_dim, activation='linear')(combined)
        
        # Create and compile model
        model = Model(
            inputs=[price_inputs, sentiment_inputs],
            outputs=outputs
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built Sentiment-Enhanced LSTM model with {len(self.lstm_units)} LSTM layers")
        logger.info(f"Price weight: {self.price_weight}, Sentiment weight: {self.sentiment_weight}")
        
        return model
    
    def fit(
        self,
        X_price_train: np.ndarray,
        X_sentiment_train: np.ndarray,
        y_train: np.ndarray,
        X_price_val: np.ndarray,
        X_sentiment_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        model_path: str = 'models/sentiment_enhanced_lstm.h5',
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_price_train: Training price features
            X_sentiment_train: Training sentiment features
            y_train: Training targets
            X_price_val: Validation price features
            X_sentiment_val: Validation sentiment features
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
        logger.info(f"Starting model training with {X_price_train.shape[0]} samples")
        history = self.model.fit(
            [X_price_train, X_sentiment_train], y_train,
            validation_data=([X_price_val, X_sentiment_val], y_val),
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
    
    def predict(
        self, 
        X_price: np.ndarray, 
        X_sentiment: np.ndarray
    ) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X_price: Price input features
            X_sentiment: Sentiment input features
            
        Returns:
            Predicted values
        """
        return self.model.predict([X_price, X_sentiment])
    
    def evaluate(
        self, 
        X_price_test: np.ndarray, 
        X_sentiment_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_price_test: Price test features
            X_sentiment_test: Sentiment test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        loss, mae = self.model.evaluate(
            [X_price_test, X_sentiment_test], 
            y_test, 
            verbose=0
        )
        
        # Calculate additional metrics
        y_pred = self.predict(X_price_test, X_sentiment_test)
        mse = np.mean(np.square(y_test - y_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-7))) * 100
        
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
                'price_input_shape': self.price_input_shape,
                'sentiment_input_shape': self.sentiment_input_shape,
                'output_dim': self.output_dim,
                'lstm_units': self.lstm_units,
                'sentiment_dense_units': self.sentiment_dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'bidirectional': self.bidirectional,
                'price_weight': self.price_weight,
                'sentiment_weight': self.sentiment_weight,
                'history': self.history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None) -> 'SentimentEnhancedLSTM':
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            metadata_path: Path to the model metadata
            
        Returns:
            SentimentEnhancedLSTM instance
        """
        # Load metadata if provided
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create instance with the same parameters
            instance = cls(
                price_input_shape=tuple(metadata['price_input_shape']),
                sentiment_input_shape=tuple(metadata['sentiment_input_shape']),
                output_dim=metadata['output_dim'],
                lstm_units=metadata['lstm_units'],
                sentiment_dense_units=metadata.get('sentiment_dense_units', [32]),
                dropout_rate=metadata['dropout_rate'],
                learning_rate=metadata['learning_rate'],
                bidirectional=metadata['bidirectional'],
                price_weight=metadata.get('price_weight', 0.7),
                sentiment_weight=metadata.get('sentiment_weight', 0.3)
            )
            
            # Load training history
            instance.history = metadata.get('history')
        else:
            # If no metadata, create an instance with default parameters
            # and update input_shape and output_dim after loading the model
            instance = cls(
                price_input_shape=(None, None),
                sentiment_input_shape=(None, None),
                output_dim=1
            )
        
        # Load the model with custom objects for the attention layer
        custom_objects = {'AttentionLayer': AttentionLayer}
        loaded_model = load_model(model_path, custom_objects=custom_objects)
        
        # Replace the model in the instance
        instance.model = loaded_model
        
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
        X_price_test: np.ndarray, 
        X_sentiment_test: np.ndarray, 
        y_test: np.ndarray, 
        dates: Optional[pd.DatetimeIndex] = None,
        n_samples: int = 100,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot test predictions against actual values.
        
        Args:
            X_price_test: Price test features
            X_sentiment_test: Sentiment test features
            y_test: Test targets
            dates: Optional dates for x-axis
            n_samples: Number of samples to plot
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Generate predictions
        y_pred = self.predict(X_price_test[:n_samples], X_sentiment_test[:n_samples])
        
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
        
        ax.set_title(f'Sentiment-Enhanced LSTM Predictions vs Actual Values (first {n_samples} samples)')
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
    
    def compare_with_price_only_model(
        self,
        price_only_model,
        X_price_test: np.ndarray,
        X_sentiment_test: np.ndarray,
        y_test: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare predictions with a price-only model.
        
        Args:
            price_only_model: Model trained only on price data
            X_price_test: Price test features
            X_sentiment_test: Sentiment test features
            y_test: Test targets
            dates: Optional dates for x-axis
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Generate predictions from both models
        combined_pred = self.predict(X_price_test, X_sentiment_test)
        price_only_pred = price_only_model.predict(X_price_test)
        
        # Sample size for visualization
        n_samples = 10
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
        
        # Use dates if provided, otherwise use indices
        if dates is not None:
            time_index = dates
        else:
            time_index = np.arange(y_test.shape[1])
        
        # Plot each sample
        for i in range(n_samples):
            ax = axes[i]
            
            # Plot actual values
            ax.plot(time_index[:y_test.shape[1]], y_test[i], 'k-', linewidth=2, label='Actual')
            
            # Plot price-only model predictions
            ax.plot(time_index[:price_only_pred.shape[1]], price_only_pred[i], 'b--', linewidth=1.5, label='Price-Only')
            
            # Plot combined model predictions
            ax.plot(time_index[:combined_pred.shape[1]], combined_pred[i], 'r--', linewidth=1.5, label='Price+Sentiment')
            
            # Add legend for first subplot only
            if i == 0:
                ax.legend()
            
            # Add y-label showing the sample index
            ax.set_ylabel(f'Sample {i+1}')
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        # Add common labels
        fig.text(0.5, 0.02, 'Time', ha='center', va='center', fontsize=12)
        fig.text(0.02, 0.5, 'Price', ha='center', va='center', rotation='vertical', fontsize=12)
        
        # Add title
        fig.suptitle('Model Comparison: Price-Only vs Price+Sentiment', fontsize=16)
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {save_path}")
        
        return fig


def prepare_sentiment_features(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    date_col_price: str = 'Date',
    date_col_sentiment: str = 'created_date',
    price_cols: List[str] = ['Price'],
    sentiment_cols: List[str] = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative'],
    window_size: int = 30,
    forecast_horizon: int = 7,
    smoothing_window: int = 3
) -> Dict[str, np.ndarray]:
    """
    Prepare combined price and sentiment features for the sentiment-enhanced model.
    
    Args:
        price_df: DataFrame with price data
        sentiment_df: DataFrame with sentiment data
        date_col_price: Date column in price DataFrame
        date_col_sentiment: Date column in sentiment DataFrame
        price_cols: Columns to use as price features
        sentiment_cols: Columns to use as sentiment features
        window_size: Size of lookback window
        forecast_horizon: Number of steps to forecast
        smoothing_window: Window size for smoothing sentiment
        
    Returns:
        Dictionary with prepared datasets
    """
    # Initialize FinBERT for sentiment analysis
    sentiment_analyzer = OilFinBERT()
    
    # Convert dates to datetime
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()
    
    # Check if price_df already has date as index
    if isinstance(price_df.index, pd.DatetimeIndex):
        # Already indexed by date, no need to set index
        pass
    elif date_col_price in price_df.columns:
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(price_df[date_col_price]):
            price_df[date_col_price] = pd.to_datetime(price_df[date_col_price])
        # Set date as index
        price_df = price_df.set_index(date_col_price)
    else:
        # No date column and not DatetimeIndex, raise error
        raise ValueError(f"No date column '{date_col_price}' found in price_df and index is not DatetimeIndex")
    
    if date_col_sentiment in sentiment_df.columns and not pd.api.types.is_datetime64_dtype(sentiment_df[date_col_sentiment]):
        sentiment_df[date_col_sentiment] = pd.to_datetime(sentiment_df[date_col_sentiment])
    elif 'created_utc' in sentiment_df.columns:
        sentiment_df[date_col_sentiment] = pd.to_datetime(sentiment_df['created_utc'], unit='s')
    
    # Aggregate sentiment by date
    sentiment_agg = sentiment_analyzer.aggregate_sentiment_by_time(
        sentiment_df,
        date_col=date_col_sentiment,
        time_freq='D',
        min_count=1
    )
    
    # Join price and sentiment data
    combined = price_df[price_cols].join(
        sentiment_agg[sentiment_cols],
        how='left'
    )
    
    # Fill missing sentiment values (days with no posts/comments)
    # Use forward fill first, then backward fill for any remaining NaNs
    combined[sentiment_cols] = combined[sentiment_cols].ffill().bfill()
    
    # If there are still NaNs, fill with zeros (neutral sentiment)
    combined = combined.fillna(0)
    
    # Apply smoothing to sentiment
    for col in sentiment_cols:
        combined[f'{col}_smooth'] = combined[col].rolling(window=smoothing_window, min_periods=1).mean()
    
    # Get smoothed sentiment columns
    smoothed_cols = [f'{col}_smooth' for col in sentiment_cols]
    
    # Create sequences for price and sentiment
    X_price, X_sentiment, y, dates = [], [], [], []
    
    # Iterate through the data with a sliding window
    for i in range(len(combined) - window_size - forecast_horizon + 1):
        # Extract window for price data
        price_window = combined.iloc[i:i+window_size][price_cols].values
        
        # Extract window for sentiment data
        sentiment_window = combined.iloc[i:i+window_size][smoothed_cols].values
        
        # Extract target values (future prices)
        target = combined.iloc[i+window_size:i+window_size+forecast_horizon][price_cols[0]].values
        
        # Extract dates
        window_dates = combined.index[i:i+window_size+forecast_horizon]
        
        # Add to lists
        X_price.append(price_window)
        X_sentiment.append(sentiment_window)
        y.append(target)
        dates.append(window_dates)
    
    # Convert to numpy arrays
    X_price = np.array(X_price)
    X_sentiment = np.array(X_sentiment)
    y = np.array(y)
    
    # Split into train, validation, test (70%, 15%, 15%)
    n_samples = len(X_price)
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    
    X_price_train = X_price[:train_size]
    X_sentiment_train = X_sentiment[:train_size]
    y_train = y[:train_size]
    
    X_price_val = X_price[train_size:train_size+val_size]
    X_sentiment_val = X_sentiment[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_price_test = X_price[train_size+val_size:]
    X_sentiment_test = X_sentiment[train_size+val_size:]
    y_test = y[train_size+val_size:]
    dates_test = dates[train_size+val_size:]
    
    logger.info(f"Prepared dataset with {n_samples} samples")
    logger.info(f"Train: {len(X_price_train)}, Validation: {len(X_price_val)}, Test: {len(X_price_test)}")
    
    return {
        'X_price_train': X_price_train,
        'X_sentiment_train': X_sentiment_train,
        'y_train': y_train,
        'X_price_val': X_price_val,
        'X_sentiment_val': X_sentiment_val,
        'y_val': y_val,
        'X_price_test': X_price_test,
        'X_sentiment_test': X_sentiment_test,
        'y_test': y_test,
        'dates_test': dates_test
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from src.data.preprocessing import OilDataProcessor
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and prepare price data
    processor = OilDataProcessor()
    price_data = processor.load_data(oil_type="brent", freq="daily")
    
    # Load sentiment data
    try:
        sentiment_data = pd.read_csv("data/processed/reddit_test_small/comprehensive_sentiment_dataset.csv")
        
        # Check if sentiment data needs analysis
        if 'sentiment_compound' not in sentiment_data.columns:
            logger.info("Sentiment data needs analysis, running FinBERT...")
            
            # Initialize FinBERT analyzer
            analyzer = OilFinBERT(models=["finbert"])
            
            # Analyze sentiment
            sentiment_data = analyzer.analyze_reddit_data(
                sentiment_data,
                text_col='text',
                title_col='title',
                batch_size=32
            )
            
            # Save analyzed data
            sentiment_data.to_csv("data/processed/reddit_test_small/comprehensive_sentiment_dataset_analyzed.csv", index=False)
            logger.info("Sentiment analysis complete and saved")
        
        # Prepare combined dataset
        logger.info("Preparing combined dataset...")
        dataset = prepare_sentiment_features(
            price_df=price_data,
            sentiment_df=sentiment_data,
            window_size=30,
            forecast_horizon=7
        )
        
        # Create and train price-only LSTM model for comparison
        logger.info("Training price-only LSTM model for comparison...")
        price_only_model = LSTMWithAttention(
            input_shape=(dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2]),
            output_dim=dataset['y_train'].shape[1],
            lstm_units=[128, 64],
            dropout_rate=0.2,
            bidirectional=True
        )
        
        price_only_model.fit(
            dataset['X_price_train'],
            dataset['y_train'],
            dataset['X_price_val'],
            dataset['y_val'],
            batch_size=32,
            epochs=50,
            patience=10,
            model_path='models/lstm_price_only.h5'
        )
        
        # Create and train sentiment-enhanced model
        logger.info("Training sentiment-enhanced LSTM model...")
        sentiment_model = SentimentEnhancedLSTM(
            price_input_shape=(dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2]),
            sentiment_input_shape=(dataset['X_sentiment_train'].shape[1], dataset['X_sentiment_train'].shape[2]),
            output_dim=dataset['y_train'].shape[1],
            lstm_units=[128, 64],
            dropout_rate=0.2,
            bidirectional=True
        )
        
        sentiment_model.fit(
            dataset['X_price_train'],
            dataset['X_sentiment_train'],
            dataset['y_train'],
            dataset['X_price_val'],
            dataset['X_sentiment_val'],
            dataset['y_val'],
            batch_size=32,
            epochs=50,
            patience=10,
            model_path='models/sentiment_enhanced_lstm.h5'
        )
        
        # Evaluate both models
        logger.info("Evaluating models...")
        price_only_metrics = price_only_model.evaluate(dataset['X_price_test'], dataset['y_test'])
        sentiment_metrics = sentiment_model.evaluate(
            dataset['X_price_test'], 
            dataset['X_sentiment_test'], 
            dataset['y_test']
        )
        
        # Compare metrics
        logger.info("Price-only model metrics:")
        for key, value in price_only_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
        logger.info("Sentiment-enhanced model metrics:")
        for key, value in sentiment_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Plot comparison of predictions
        logger.info("Generating comparison plots...")
        sentiment_model.compare_with_price_only_model(
            price_only_model,
            dataset['X_price_test'],
            dataset['X_sentiment_test'],
            dataset['y_test'],
            dates=[dates[-1] for dates in dataset['dates_test']],
            save_path='notebooks/plots/model_comparison_demo.png'
        )
        
        logger.info("Demo complete. Sentiment-enhanced model training and evaluation successful.")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())