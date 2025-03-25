"""
Visualization module for Oil Prophet.

This module provides plotting functions for data exploration, model evaluation,
and forecast visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from typing import List, Dict, Tuple, Optional, Union, Any
import os
from datetime import datetime, timedelta
import seaborn as sns

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


def format_price(x, pos):
    """Format price values with dollar sign."""
    return f'${x:.2f}'


def plot_time_series(
    data: pd.DataFrame,
    title: str = 'Oil Price Time Series',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series data.
    
    Args:
        data: DataFrame with DatetimeIndex and 'Price' column
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure data has DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data = data.set_index('Date')
        else:
            raise ValueError("Data must have a DatetimeIndex or 'Date' column")
    
    # Plot the time series
    ax.plot(data.index, data['Price'], linewidth=2)
    
    # Format the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    
    # Format y-axis to show prices
    ax.yaxis.set_major_formatter(FuncFormatter(format_price))
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_decomposition(
    components: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 15),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot signal decomposition components.
    
    Args:
        components: Array of components (first row is reconstructed signal)
        dates: Optional DatetimeIndex for x-axis
        titles: Optional list of titles for each component
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    n_components = len(components)
    
    # Create default component titles if not provided
    if titles is None:
        titles = ['Original Signal']
        for i in range(1, n_components):
            if i == 1:
                titles.append('Trend')
            elif i == n_components - 1:
                titles.append('Residual (Noise)')
            else:
                titles.append(f'Cyclical Component {i-1}')
    
    # Create x-axis values if not provided
    if dates is None:
        dates = np.arange(len(components[0]))
    
    # Create figure
    fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)
    
    # Set colors for different components
    colors = ['black', 'red', 'blue', 'teal', 'orange', 'purple', 'green']
    
    # Plot each component
    for i, component in enumerate(components):
        color_idx = min(i, len(colors) - 1)
        axes[i].plot(dates, component, color=colors[color_idx], linewidth=1.5)
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].set_ylabel('Amplitude', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # Add x-label to bottom subplot
    axes[-1].set_xlabel('Time', fontsize=12)
    
    # Format x-axis to show dates nicely if provided
    if isinstance(dates, pd.DatetimeIndex):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved decomposition plot to {save_path}")
    
    return fig


def plot_forecast(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    forecast: np.ndarray,
    window_size: int = 30,
    title: str = 'Oil Price Forecast',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot forecasted values against actual values.
    
    Args:
        train_data: DataFrame with training data
        test_data: DataFrame with test data
        forecast: Forecasted values
        window_size: Size of the input window used for forecasting
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure data has DatetimeIndex
    if not isinstance(train_data.index, pd.DatetimeIndex):
        if 'Date' in train_data.columns:
            train_data = train_data.set_index('Date')
        else:
            raise ValueError("Data must have a DatetimeIndex or 'Date' column")
    
    if not isinstance(test_data.index, pd.DatetimeIndex):
        if 'Date' in test_data.columns:
            test_data = test_data.set_index('Date')
        else:
            raise ValueError("Data must have a DatetimeIndex or 'Date' column")
    
    # Get the last window_size points from training data for context
    context_data = train_data.iloc[-window_size:]
    
    # Plot training context
    ax.plot(
        context_data.index, 
        context_data['Price'], 
        color='blue',
        label='Training Data',
        linewidth=2,
        alpha=0.7
    )
    
    # Plot test data (ground truth)
    ax.plot(
        test_data.index,
        test_data['Price'],
        color='green',
        label='Actual Values',
        linewidth=2
    )
    
    # Plot forecast
    forecast_dates = test_data.index[:len(forecast)]
    ax.plot(
        forecast_dates,
        forecast,
        color='red',
        label='Forecast',
        linestyle='--',
        linewidth=2
    )
    
    # Add shaded area for forecast uncertainty
    ax.fill_between(
        forecast_dates,
        forecast * 0.95,  # 5% below forecast
        forecast * 1.05,  # 5% above forecast
        color='red',
        alpha=0.2,
        label='Uncertainty'
    )
    
    # Format the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    
    # Format y-axis to show prices
    ax.yaxis.set_major_formatter(FuncFormatter(format_price))
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Add vertical line to separate training and test data
    ax.axvline(
        x=train_data.index[-1],
        color='gray',
        linestyle=':',
        alpha=0.7,
        label='Train/Test Split'
    )
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved forecast plot to {save_path}")
    
    return fig


def plot_model_comparison(
    actual_values: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    metrics_dict: Optional[Dict[str, Dict[str, float]]] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predictions from multiple models for comparison.
    
    Args:
        actual_values: Actual values (ground truth)
        predictions_dict: Dictionary of model predictions {model_name: predictions}
        dates: Optional DatetimeIndex for x-axis
        metrics_dict: Optional dictionary of metrics {model_name: {metric: value}}
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Create x-axis values if not provided
    if dates is None:
        dates = np.arange(len(actual_values))
    
    # Plot actual values
    ax1.plot(
        dates,
        actual_values,
        color='black',
        linewidth=2,
        label='Actual Values'
    )
    
    # Plot model predictions
    colors = sns.color_palette('colorblind', n_colors=len(predictions_dict))
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        # Ensure predictions match the length of actual values
        pred_len = min(len(predictions), len(actual_values))
        
        # Get error metrics if available
        metric_label = ""
        if metrics_dict and model_name in metrics_dict:
            metrics = metrics_dict[model_name]
            if 'rmse' in metrics:
                metric_label = f" (RMSE: {metrics['rmse']:.2f})"
            elif 'mae' in metrics:
                metric_label = f" (MAE: {metrics['mae']:.2f})"
        
        # Plot predictions
        ax1.plot(
            dates[:pred_len],
            predictions[:pred_len],
            color=colors[i],
            linewidth=1.5,
            linestyle='--',
            label=f'{model_name}{metric_label}'
        )
    
    # Calculate and plot prediction errors
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        # Ensure predictions match the length of actual values
        pred_len = min(len(predictions), len(actual_values))
        
        # Calculate prediction error
        error = actual_values[:pred_len] - predictions[:pred_len]
        
        # Plot error
        ax2.plot(
            dates[:pred_len],
            error,
            color=colors[i],
            linewidth=1.5,
            alpha=0.7,
            label=f'{model_name} Error'
        )
    
    # Format the plots
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_price))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely if provided
    if isinstance(dates, pd.DatetimeIndex):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = 'Model Training History',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary of training history {metric: values}
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    if 'loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot metrics (MAE or others)
    metrics = [k for k in history.keys() if k not in ['loss', 'val_loss'] and not k.startswith('lr')]
    if metrics:
        # Use the first metric found
        metric = metrics[0]
        val_metric = f'val_{metric}' if f'val_{metric}' in history else None
        
        epochs = range(1, len(history[metric]) + 1)
        
        ax2.plot(epochs, history[metric], 'b-', label=f'Training {metric.upper()}')
        if val_metric:
            ax2.plot(epochs, history[val_metric], 'r--', label=f'Validation {metric.upper()}')
        
        ax2.set_title(metric.upper())
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric.upper())
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Adjust for the overall title
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    input_sequence: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = 'Attention Weights',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights for time series prediction.
    
    Args:
        attention_weights: Attention weights (shape: batch_size, input_length)
        input_sequence: Input sequence data
        dates: Optional DatetimeIndex for x-axis
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Use the first sample if attention_weights has batch dimension
    if len(attention_weights.shape) > 1:
        attention_weights = attention_weights[0]
    
    # Reshape input if needed
    input_sequence = input_sequence.reshape(-1)
    
    # Create x-axis values if not provided
    if dates is None:
        dates = np.arange(len(input_sequence))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot input sequence
    ax1.plot(dates, input_sequence, 'b-', linewidth=2)
    ax1.set_title('Input Sequence', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_price))
    ax1.grid(True, alpha=0.3)
    
    # Plot attention weights as bars
    ax2.bar(
        dates, 
        attention_weights, 
        color='orange', 
        alpha=0.7, 
        width=0.8 * (dates[1] - dates[0]) if len(dates) > 1 else 0.8
    )
    ax2.set_title('Attention Weights', fontsize=14)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Highlight the time points with highest attention
    threshold = np.percentile(attention_weights, 75)
    high_attention_indices = attention_weights >= threshold
    
    # Highlight in the input sequence plot
    ax1.scatter(
        dates[high_attention_indices],
        input_sequence[high_attention_indices],
        color='red',
        s=50,
        zorder=5,
        label='High Attention'
    )
    ax1.legend()
    
    # Format x-axis to show dates nicely if provided
    if isinstance(dates, pd.DatetimeIndex):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Adjust for the overall title
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention weights plot to {save_path}")
    
    return fig


def create_interactive_dashboard(data_path: str = None):
    """
    Create an interactive dashboard for exploring oil price data.
    
    This is a placeholder function that would typically use a library like 
    Dash, Streamlit, or Panel to create an interactive web-based dashboard.
    
    Args:
        data_path: Path to the data directory
    """
    # This would typically be implemented using Dash, Streamlit, or Panel
    print("Interactive dashboard functionality requires additional dependencies.")
    print("Consider installing Dash, Streamlit, or Panel to implement this feature.")


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import OilDataProcessor
    import matplotlib as mpl
    
    # Set up larger font size for better readability
    mpl.rcParams.update({'font.size': 12})
    
    # Create output directory for plots
    os.makedirs('notebooks/plots', exist_ok=True)
    
    # Load data
    processor = OilDataProcessor()
    try:
        # Load Brent daily data
        brent_daily = processor.load_data(oil_type="brent", freq="daily")
        
        # Plot time series
        plot_time_series(
            brent_daily,
            title='Brent Crude Oil Price (Daily)',
            save_path='notebooks/plots/brent_daily_timeseries.png'
        )
        
        # Prepare dataset for model training
        dataset = processor.prepare_dataset(oil_type="brent", freq="daily")
        
        # Simple prediction example (for demonstration)
        test_len = len(dataset['y_test'])
        dummy_prediction = dataset['original_data']['Price'].iloc[-test_len:].values
        dummy_prediction = dummy_prediction + np.random.normal(0, 2, size=test_len)  # Add some noise
        
        # Plot model comparison
        actual_values = dataset['original_data']['Price'].iloc[-test_len:].values
        predictions_dict = {
            'Naive': actual_values,  # Just for demonstration
            'LSTM': dummy_prediction  # Simulated prediction
        }
        metrics_dict = {
            'Naive': {'rmse': 0.0},
            'LSTM': {'rmse': 2.0}
        }
        
        plot_model_comparison(
            actual_values,
            predictions_dict,
            dates=dataset['original_data'].index[-test_len:],
            metrics_dict=metrics_dict,
            title='Model Comparison (Demonstration)',
            save_path='notebooks/plots/model_comparison_demo.png'
        )
        
        print("Generated example plots in notebooks/plots/")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")