"""
Visualization module for Oil Prophet.

This module provides functions for visualizing oil price forecasts,
model performance, and other key components of the forecasting system.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_forecasts(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    model_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Oil Price Forecast",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot historical data and forecasts from multiple models.
    
    Args:
        historical_data: DataFrame with historical price data
        forecast_data: DataFrame with forecast data for each model
        model_names: List of model names to include
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    ax.plot(
        historical_data.index,
        historical_data['Price'],
        'k-',
        linewidth=2,
        label='Historical'
    )
    
    # Plot forecasts for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for i, model in enumerate(model_names):
        if model in forecast_data.columns:
            ax.plot(
                forecast_data.index,
                forecast_data[model],
                '--',
                color=colors[i],
                linewidth=2,
                label=f"{model} Forecast"
            )
    
    # Format plot
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved forecast plot to {save_path}")
    
    return fig


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['rmse', 'mae', 'mape'],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models based on evaluation metrics.
    
    Args:
        metrics: Dictionary with metrics for each model
        metric_names: List of metrics to plot
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metric_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Convert to list if only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Get model names and colors
    model_names = list(metrics.keys())
    n_models = len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        # Get metric values for each model
        values = [metrics[model][metric] for model in model_names]
        
        # Create bar chart
        bars = axes[i].bar(
            np.arange(n_models),
            values,
            color=colors,
            alpha=0.7
        )
        
        # Add metric name as title
        axes[i].set_title(metric.upper())
        
        # Add model names as x-tick labels
        axes[i].set_xticks(np.arange(n_models))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add grid
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                rotation=0
            )
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    
    return fig


def plot_decomposition(
    original_signal: np.ndarray,
    components: List[np.ndarray],
    component_names: List[str],
    time_index: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 15),
    title: str = "Signal Decomposition",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot original signal and its decomposed components.
    
    Args:
        original_signal: The original time series signal
        components: List of decomposed signal components
        component_names: Names for each component
        time_index: Optional time indices for x-axis
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    n_components = len(components)
    
    # Create x-axis values if not provided
    if time_index is None:
        time_index = np.arange(len(original_signal))
    
    # Create figure
    fig, axes = plt.subplots(n_components + 1, 1, figsize=figsize, sharex=True)
    
    # Plot original signal
    axes[0].plot(time_index, original_signal, 'k-')
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot components
    for i, component in enumerate(components):
        axes[i+1].plot(time_index, component)
        axes[i+1].set_title(component_names[i])
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].grid(True, alpha=0.3)
    
    # Set x-label for the last subplot
    axes[-1].set_xlabel('Time')
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    # Format x-axis if time_index is DatetimeIndex
    if isinstance(time_index, pd.DatetimeIndex):
        fig.autofmt_xdate()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved decomposition plot to {save_path}")
    
    return fig


def plot_price_with_sentiment(
    price_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    sentiment_columns: List[str] = ['sentiment_compound', 'sentiment_ma7'],
    column_labels: List[str] = ['Sentiment', '7-day MA'],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Oil Price with Market Sentiment",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot oil price with sentiment overlay.
    
    Args:
        price_data: DataFrame with price data
        sentiment_data: DataFrame with sentiment data
        sentiment_columns: Columns to plot from sentiment data
        column_labels: Labels for the sentiment columns
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Ensure same index for both DataFrames
    common_index = price_data.index.intersection(sentiment_data.index)
    price_slice = price_data.loc[common_index]
    sentiment_slice = sentiment_data.loc[common_index]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot price on primary y-axis
    ax1.plot(price_slice.index, price_slice['Price'], 'b-', linewidth=2, label='Oil Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create secondary y-axis for sentiment
    ax2 = ax1.twinx()
    
    # Plot sentiment columns
    colors = ['r', 'g', 'c', 'm']
    for i, (col, label) in enumerate(zip(sentiment_columns, column_labels)):
        if col in sentiment_slice.columns:
            ax2.plot(
                sentiment_slice.index,
                sentiment_slice[col],
                f'{colors[i]}-',
                linewidth=1.5,
                alpha=0.7,
                label=label
            )
    
    ax2.set_ylabel('Sentiment Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add horizontal lines for neutral sentiment
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Set title
    plt.title(title)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved price-sentiment plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    
    # Price data
    price = 60 + np.cumsum(np.random.normal(0, 0.5, 365))
    price_df = pd.DataFrame({'Price': price}, index=dates)
    
    # Sentiment data
    sentiment = 0.2 * np.sin(np.linspace(0, 8*np.pi, 365)) + np.random.normal(0, 0.1, 365)
    sentiment_ma7 = pd.Series(sentiment).rolling(window=7).mean().fillna(0).values
    sentiment_df = pd.DataFrame({
        'sentiment_compound': sentiment,
        'sentiment_ma7': sentiment_ma7
    }, index=dates)
    
    # Forecast data
    forecast_dates = pd.date_range(start='2021-01-01', periods=30, freq='D')
    naive_forecast = price[-1] * np.ones(30)
    ema_forecast = price[-1] * (1 + 0.01 * np.arange(30))
    lstm_forecast = price[-1] * (1 + 0.015 * np.arange(30) + 0.005 * np.sin(np.linspace(0, 3*np.pi, 30)))
    
    forecast_df = pd.DataFrame({
        'Naive': naive_forecast,
        'EMA': ema_forecast,
        'LSTM': lstm_forecast
    }, index=forecast_dates)
    
    # Test plotting functions
    print("Testing plot_price_with_sentiment...")
    fig1 = plot_price_with_sentiment(
        price_df,
        sentiment_df,
        save_path='price_with_sentiment.png'
    )
    
    print("Testing plot_forecasts...")
    fig2 = plot_forecasts(
        price_df.iloc[-60:],
        forecast_df,
        ['Naive', 'EMA', 'LSTM'],
        save_path='forecasts.png'
    )
    
    print("Testing plot_model_comparison...")
    metrics = {
        'Naive': {'rmse': 3.2, 'mae': 2.7, 'mape': 5.2},
        'EMA': {'rmse': 2.8, 'mae': 2.3, 'mape': 4.5},
        'LSTM': {'rmse': 2.2, 'mae': 1.8, 'mape': 3.6}
    }
    
    fig3 = plot_model_comparison(
        metrics,
        save_path='model_comparison.png'
    )
    
    print("Testing plot_decomposition...")
    # Generate components
    trend = savgol_filter(price, 101, 3)
    cycle1 = 5 * np.sin(np.linspace(0, 8*np.pi, 365))
    cycle2 = 3 * np.sin(np.linspace(0, 20*np.pi, 365))
    residual = price - trend - cycle1 - cycle2
    
    fig4 = plot_decomposition(
        price,
        [trend, cycle1, cycle2, residual],
        ['Trend', 'Cycle 1', 'Cycle 2', 'Residual'],
        time_index=dates,
        save_path='decomposition.png'
    )
    
    print("All tests completed successfully!")