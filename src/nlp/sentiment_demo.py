"""
Demonstration script for using sentiment analysis in oil price forecasting.

This script shows how to combine sentiment data with your LSTM model to improve forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sentiment_data(filepath: str = 'data/processed/oil_market_sentiment_indicators.csv') -> pd.DataFrame:
    """
    Load sentiment data from CSV file.
    
    Args:
        filepath: Path to sentiment data file
        
    Returns:
        DataFrame with sentiment data
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Sentiment data file not found: {filepath}")
            logger.info("Sentiment data needs to be generated first")
            return pd.DataFrame()
        
        # Load the sentiment data
        sentiment_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        logger.info(f"Loaded sentiment data with {len(sentiment_df)} records")
        return sentiment_df
    
    except Exception as e:
        logger.error(f"Error loading sentiment data: {str(e)}")
        return pd.DataFrame()


def align_sentiment_with_price_data(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align sentiment data with price data based on dates.
    
    Args:
        price_df: DataFrame with price data
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        DataFrame with aligned price and sentiment data
    """
    if price_df.empty or sentiment_df.empty:
        logger.warning("Empty dataframes provided for alignment")
        return pd.DataFrame()
    
    # Ensure DatetimeIndex
    if not isinstance(price_df.index, pd.DatetimeIndex):
        if 'Date' in price_df.columns:
            price_df = price_df.set_index('Date')
        else:
            logger.error("Price dataframe has no DatetimeIndex or 'Date' column")
            return pd.DataFrame()
    
    if not isinstance(sentiment_df.index, pd.DatetimeIndex):
        if 'date' in sentiment_df.columns:
            sentiment_df = sentiment_df.set_index('date')
        else:
            logger.error("Sentiment dataframe has no DatetimeIndex or 'date' column")
            return pd.DataFrame()
    
    # Rename sentiment columns to avoid conflicts
    sentiment_df = sentiment_df.add_prefix('sentiment_')
    
    # Merge the dataframes on date index
    merged_df = price_df.join(sentiment_df, how='left')
    
    # Handle NaN values in sentiment columns
    # Forward fill for recent dates where we have some sentiment data
    # For older dates where we have no sentiment data, use neutral values
    sentiment_cols = [col for col in merged_df.columns if col.startswith('sentiment_')]
    
    # Fill missing values with forward fill
    merged_df[sentiment_cols] = merged_df[sentiment_cols].ffill()
    
    # Then backward fill any remaining gaps
    merged_df[sentiment_cols] = merged_df[sentiment_cols].bfill()
    
    # For any remaining NaNs, fill with neutral sentiment
    merged_df['sentiment_sentiment_compound'] = merged_df['sentiment_sentiment_compound'].fillna(0)
    merged_df['sentiment_sentiment_pos'] = merged_df['sentiment_sentiment_pos'].fillna(0.33)
    merged_df['sentiment_sentiment_neg'] = merged_df['sentiment_sentiment_neg'].fillna(0.33)
    merged_df['sentiment_sentiment_neu'] = merged_df['sentiment_sentiment_neu'].fillna(0.34)
    
    # Fill remaining sentiment indicators with appropriate values
    if 'sentiment_sentiment_ma7' in merged_df.columns:
        merged_df['sentiment_sentiment_ma7'] = merged_df['sentiment_sentiment_ma7'].fillna(0)
    
    if 'sentiment_bullish_bias' in merged_df.columns:
        merged_df['sentiment_bullish_bias'] = merged_df['sentiment_bullish_bias'].fillna(0.5)
    
    if 'sentiment_bearish_bias' in merged_df.columns:
        merged_df['sentiment_bearish_bias'] = merged_df['sentiment_bearish_bias'].fillna(0.5)
    
    if 'sentiment_sentiment_signal' in merged_df.columns:
        merged_df['sentiment_sentiment_signal'] = merged_df['sentiment_sentiment_signal'].fillna(0)
    
    logger.info(f"Aligned sentiment data with price data: {len(merged_df)} records")
    
    return merged_df


def create_enhanced_features(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    window_size: int = 30,
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create enhanced features by combining price and sentiment data.
    
    Args:
        price_df: DataFrame with price data
        sentiment_df: DataFrame with sentiment data
        window_size: Size of the sliding window
        forecast_horizon: Number of steps to predict
        
    Returns:
        Tuple of X (features) and y (target) arrays
    """
    # Align data
    merged_df = align_sentiment_with_price_data(price_df, sentiment_df)
    
    if merged_df.empty:
        logger.error("Failed to create enhanced features due to empty data")
        return np.array([]), np.array([])
    
    # Select important sentiment features
    sentiment_features = [
        'sentiment_sentiment_compound',
        'sentiment_sentiment_pos',
        'sentiment_sentiment_neg'
    ]
    
    if 'sentiment_sentiment_ma7' in merged_df.columns:
        sentiment_features.append('sentiment_sentiment_ma7')
    
    if 'sentiment_bullish_bias' in merged_df.columns:
        sentiment_features.append('sentiment_bullish_bias')
    
    # Create sliding windows with both price and sentiment features
    X, y = [], []
    
    for i in range(len(merged_df) - window_size - forecast_horizon + 1):
        # Get window of data
        window_df = merged_df.iloc[i:i + window_size]
        
        # Create feature vector with price and sentiment
        feature_vector = []
        
        # Add price features
        price_features = window_df['Price'].values.reshape(-1, 1)
        feature_vector.append(price_features)
        
        # Add sentiment features
        for feature in sentiment_features:
            if feature in window_df.columns:
                sentiment_feature = window_df[feature].values.reshape(-1, 1)
                feature_vector.append(sentiment_feature)
        
        # Combine all features
        combined_features = np.hstack(feature_vector)
        X.append(combined_features)
        
        # Get target values (future prices)
        target = merged_df.iloc[i + window_size:i + window_size + forecast_horizon]['Price'].values
        y.append(target)
    
    return np.array(X), np.array(y)


def plot_price_with_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot oil price with sentiment overlay.
    
    Args:
        price_df: DataFrame with price data
        sentiment_df: DataFrame with sentiment data
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Align data
    merged_df = align_sentiment_with_price_data(price_df, sentiment_df)
    
    if merged_df.empty:
        logger.error("Failed to create plot due to empty data")
        return None
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot price on primary y-axis
    ax1.plot(merged_df.index, merged_df['Price'], 'b-', linewidth=2, label='Oil Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create secondary y-axis for sentiment
    ax2 = ax1.twinx()
    ax2.plot(
        merged_df.index,
        merged_df['sentiment_sentiment_compound'],
        'r-',
        linewidth=1.5,
        alpha=0.7,
        label='Sentiment'
    )
    ax2.set_ylabel('Sentiment Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Plot sentiment moving average if available
    if 'sentiment_sentiment_ma7' in merged_df.columns:
        ax2.plot(
            merged_df.index,
            merged_df['sentiment_sentiment_ma7'],
            'g-',
            linewidth=1.5,
            alpha=0.7,
            label='7-day MA'
        )
    
    # Add horizontal lines for neutral sentiment
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Set title
    plt.title('Oil Price with Market Sentiment')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved price-sentiment plot to {save_path}")
    
    return fig


def run_demonstration():
    """Run a demonstration of using sentiment data for oil price forecasting."""
    from src.data.preprocessing import OilDataProcessor
    
    # Create output directory for plots
    os.makedirs('notebooks/plots', exist_ok=True)
    
    try:
        # Load price data
        processor = OilDataProcessor()
        price_data = processor.load_data(oil_type="brent", freq="daily")
        
        # Load sentiment data
        sentiment_data = load_sentiment_data()
        
        if sentiment_data.empty:
            logger.warning("No sentiment data available. Running with simulated sentiment.")
            
            # Create simulated sentiment data for demonstration
            # FIX: Ensure consistent date ranges and array shapes
            
            # Get the last year of data, ensuring we're working with trading days
            start_date = price_data.index[-365]
            end_date = price_data.index[-1]
            
            # Extract price data for this date range
            price_slice = price_data.loc[start_date:end_date]
            
            # Calculate price changes, filling NaN values
            price_changes = price_slice['Price'].pct_change().fillna(0)
            
            # Generate random component with the same length
            random_component = np.random.normal(0, 0.1, len(price_changes))
            
            # Create sentiment with correlation to price changes
            sentiment_values = 0.3 * price_changes.values + 0.7 * random_component
            
            # Create DataFrame with the same index as price_slice
            sentiment_data = pd.DataFrame({
                'sentiment_compound': sentiment_values,
                'sentiment_pos': 0.5 + 0.5 * np.clip(sentiment_values, -1, 1),
                'sentiment_neg': 0.5 - 0.5 * np.clip(sentiment_values, -1, 1),
                'sentiment_neu': 0.5 * np.ones_like(sentiment_values),
                'count': 10 * np.ones_like(sentiment_values)
            }, index=price_slice.index)
            
            # Calculate moving averages for simulated sentiment
            sentiment_data['sentiment_ma7'] = sentiment_data['sentiment_compound'].rolling(window=7).mean().fillna(0)
            sentiment_data['sentiment_ma14'] = sentiment_data['sentiment_compound'].rolling(window=14).mean().fillna(0)
            
            logger.info(f"Created simulated sentiment data for demonstration with shape: {sentiment_data.shape}")
            logger.info(f"Price data shape: {price_slice.shape}")
        
        # Plot price with sentiment overlay
        plot_price_with_sentiment(
            price_data.loc['2020-01-01':],  # Focus on recent data
            sentiment_data,
            save_path='notebooks/plots/price_with_sentiment.png'
        )
        
        logger.info("Demonstration: Integrating sentiment with price data")
        
        # Create enhanced features for model training
        window_size = 30
        forecast_horizon = 7
        
        X_enhanced, y_enhanced = create_enhanced_features(
            price_data,
            sentiment_data,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        
        if len(X_enhanced) > 0:
            # Show the shape of enhanced features
            num_sentiment_features = X_enhanced.shape[2] - 1  # Subtract price feature
            
            logger.info(f"Enhanced feature shape: {X_enhanced.shape}")
            logger.info(f"Target shape: {y_enhanced.shape}")
            logger.info(f"Number of sentiment features: {num_sentiment_features}")
            
            logger.info("\nTo train a model with these enhanced features:")
            logger.info("1. Use the created X_enhanced and y_enhanced arrays")
            logger.info("2. Split into train, validation, and test sets")
            logger.info("3. Train the LSTM-Attention model with these features")
            
            logger.info("\nSample code:")
            logger.info("from src.models.lstm_attention import LSTMWithAttention")
            logger.info("model = LSTMWithAttention(")
            logger.info(f"    input_shape=({window_size}, {X_enhanced.shape[2]}),")
            logger.info(f"    output_dim={forecast_horizon}")
            logger.info(")")
            logger.info("model.fit(X_train, y_train, X_val, y_val)")
        else:
            logger.error("Failed to create enhanced features")
    
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running sentiment analysis demonstration...")
    run_demonstration()
    print("\nCheck the notebooks/plots directory for output visualization.")