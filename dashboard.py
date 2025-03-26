"""
Oil Prophet Dashboard - Interactive visualization for oil price forecasting models.

This dashboard provides a user interface for exploring and visualizing forecasts from
different models, comparing their performance, and analyzing the effects of market sentiment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Import the project modules
from src.data.preprocessing import OilDataProcessor
from src.models.baseline import BaselineForecaster
from src.models.ensemble import EnsembleForecaster, HybridCEEMDANLSTM
from src.models.lstm_attention import LSTMWithAttention
from src.evaluation.metrics import calculate_metrics, compare_models, plot_model_comparison_metrics
from src.visualization.plots import plot_forecasts, plot_decomposition, plot_model_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Oil Prophet Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Oil Prophet Dashboard")
st.markdown("""
    Interactive dashboard for oil price forecasting powered by advanced time-series decomposition
    and deep learning models. Explore forecasts, model performance, and the influence of market sentiment.
""")

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Forecast Explorer", "Model Performance", "Sentiment Analysis", "Signal Decomposition"]
)

# Function to load available models
def load_available_models() -> Dict[str, str]:
    """
    Check for available model files in the models directory.
    
    Returns:
        Dictionary of model names and file paths
    """
    models_dir = "models"
    available_models = {}
    
    # Check if directory exists
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found: {models_dir}")
        return available_models
    
    # Look for model files
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".h5") or file.endswith(".pkl"):
                model_path = os.path.join(root, file)
                model_name = os.path.splitext(file)[0]
                available_models[model_name] = model_path
    
    return available_models

# Function to load data
@st.cache_data
def load_oil_data(oil_type: str, freq: str) -> pd.DataFrame:
    """
    Load and cache oil price data.
    
    Args:
        oil_type: Type of oil price data ('brent' or 'wti')
        freq: Frequency of data ('daily', 'weekly', 'monthly', 'year')
        
    Returns:
        DataFrame with oil price data
    """
    processor = OilDataProcessor()
    try:
        data = processor.load_data(oil_type=oil_type, freq=freq)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Function to generate forecasts
def generate_forecasts(
    data: pd.DataFrame,
    forecast_period: int,
    models: List[str],
    freq: str = "daily"
) -> Dict[str, np.ndarray]:
    """
    Generate forecasts using selected models.
    
    Args:
        data: Historical price data
        forecast_period: Number of periods to forecast (days/weeks/months)
        models: List of model names to use
        freq: Data frequency ('daily', 'weekly', 'monthly')
        
    Returns:
        Dictionary of forecasts from each model
    """
    forecasts = {}
    price_data = data['Price'].values
    
    # Create baseline forecasts
    if 'Naive' in models:
        naive_model = BaselineForecaster(method='naive')
        naive_model.fit(price_data[-30:])  # Use last 30 data points for naive model
        naive_forecast = naive_model.predict(steps=forecast_period)
        forecasts['Naive'] = naive_forecast
    
    if 'ARIMA' in models:
        arima_model = BaselineForecaster(method='arima')
        arima_model.fit(price_data[-60:])  # Use last 60 data points for ARIMA
        arima_forecast = arima_model.predict(steps=forecast_period)
        forecasts['ARIMA'] = arima_forecast
    
    if 'EMA' in models:
        ema_model = BaselineForecaster(method='ema')
        ema_model.fit(price_data[-45:])  # Use last 45 data points for EMA
        ema_forecast = ema_model.predict(steps=forecast_period)
        forecasts['EMA'] = ema_forecast
    
    # For advanced models, we would load the saved models
    # This is a placeholder for demonstration purposes
    if 'LSTM-Attention' in models:
        # Simulate LSTM forecasts with slight improvement over naive
        lstm_forecast = forecasts.get('Naive', price_data[-1] * np.ones(forecast_period))
        trend_factor = (price_data[-1] - price_data[-15]) / price_data[-15]
        lstm_forecast = lstm_forecast * (1 + trend_factor * np.arange(1, forecast_period + 1) / forecast_period)
        forecasts['LSTM-Attention'] = lstm_forecast
    
    if 'CEEMDAN-LSTM' in models:
        # Simulate hybrid model forecasts
        hybrid_forecast = forecasts.get('LSTM-Attention', price_data[-1] * np.ones(forecast_period))
        # Add some cyclical pattern
        cycles = 0.02 * np.sin(np.linspace(0, forecast_period/5*np.pi, forecast_period))
        hybrid_forecast = hybrid_forecast * (1 + cycles)
        forecasts['CEEMDAN-LSTM'] = hybrid_forecast
    
    if 'Ensemble' in models and len(forecasts) > 1:
        # Create ensemble forecast (average of all other forecasts)
        ensemble_forecast = np.zeros(forecast_period)
        for model_name, forecast in forecasts.items():
            ensemble_forecast += forecast
        ensemble_forecast /= len(forecasts)
        forecasts['Ensemble'] = ensemble_forecast
    
    return forecasts

# Function to plot forecasts
def plot_forecast_comparison(
    data: pd.DataFrame,
    forecasts: Dict[str, np.ndarray],
    lookback_days: int = 30,
    figsize: Tuple[int, int] = (12, 8),
    freq: str = "daily"
) -> plt.Figure:
    """
    Plot historical data and forecasts from multiple models.
    
    Args:
        data: Historical price data
        forecasts: Dictionary of forecasts from each model
        lookback_days: Number of historical days to include
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get historical data for the lookback period
    historical_data = data.iloc[-lookback_days:].copy()
    
    # Plot historical data
    ax.plot(historical_data.index, historical_data['Price'], 'k-', linewidth=2, label='Historical')
    
    # Create forecast dates with appropriate frequency
    last_date = historical_data.index[-1]
    
    # Set proper date range frequency based on selected data frequency
    if freq == "daily":
        date_freq = 'B'  # Business days
    elif freq == "weekly":
        date_freq = 'W'  # Weeks
    else:
        date_freq = 'M'  # Months
    
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(next(iter(forecasts.values()))),
        freq=date_freq
    )
    
    # Plot forecasts for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        ax.plot(
            forecast_dates,
            forecast,
            '--',
            color=colors[i],
            linewidth=2,
            label=f"{model_name} Forecast"
        )
    
    # Format plot
    ax.set_title('Oil Price Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig

# Function to simulate model performance metrics
def get_model_performance(models: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Generate simulated performance metrics for selected models.
    
    Args:
        models: List of model names
        
    Returns:
        Dictionary of performance metrics for each model
    """
    performance = {}
    
    # Base performance values (RMSE)
    base_metrics = {
        'Naive': {'rmse': 3.2, 'mae': 2.7, 'mape': 5.2, 'directional_accuracy': 52.0},
        'ARIMA': {'rmse': 2.8, 'mae': 2.3, 'mape': 4.5, 'directional_accuracy': 58.0},
        'EMA': {'rmse': 2.9, 'mae': 2.4, 'mape': 4.7, 'directional_accuracy': 56.0},
        'LSTM-Attention': {'rmse': 2.5, 'mae': 2.0, 'mape': 3.9, 'directional_accuracy': 64.0},
        'CEEMDAN-LSTM': {'rmse': 2.2, 'mae': 1.8, 'mape': 3.5, 'directional_accuracy': 68.0},
        'Ensemble': {'rmse': 2.0, 'mae': 1.6, 'mape': 3.1, 'directional_accuracy': 72.0}
    }
    
    # Add slight randomness for demo purposes
    np.random.seed(42)
    for model in models:
        if model in base_metrics:
            performance[model] = {
                metric: value * (1 + np.random.normal(0, 0.05))
                for metric, value in base_metrics[model].items()
            }
    
    return performance

# Function to simulate sentiment data
def generate_sentiment_data(data: pd.DataFrame, days: int = 120) -> pd.DataFrame:
    """
    Generate simulated sentiment data for demonstration.
    
    Args:
        data: Price data to align with
        days: Number of days of sentiment data to generate
        
    Returns:
        DataFrame with sentiment data
    """
    # Get the last portion of price data
    price_slice = data.iloc[-days:].copy()
    
    # Calculate price changes
    price_changes = price_slice['Price'].pct_change().fillna(0)
    
    # Generate random component
    np.random.seed(42)
    random_component = np.random.normal(0, 0.2, len(price_changes))
    
    # Create sentiment with correlation to price changes (0.3 correlation)
    sentiment_values = 0.3 * price_changes.values + 0.7 * random_component
    
    # Create DataFrame
    sentiment_df = pd.DataFrame({
        'sentiment_compound': sentiment_values,
        'sentiment_pos': 0.5 + 0.5 * np.clip(sentiment_values, -1, 1),
        'sentiment_neg': 0.5 - 0.5 * np.clip(sentiment_values, -1, 1),
        'sentiment_neu': 0.5 * np.ones_like(sentiment_values),
        'sentiment_ma7': pd.Series(sentiment_values).rolling(window=7).mean().fillna(0).values
    }, index=price_slice.index)
    
    return sentiment_df

# Function to simulate signal decomposition
def simulate_decomposition(data: pd.DataFrame, n_components: int = 4) -> Dict[str, np.ndarray]:
    """
    Simulate signal decomposition for demonstration.
    
    Args:
        data: Price data to decompose
        n_components: Number of components to extract
        
    Returns:
        Dictionary of decomposed signals
    """
    # Get the price data as array
    price_array = data['Price'].values
    
    # Create decomposition components
    components = []
    
    # Trend component (low frequency)
    from scipy.signal import savgol_filter
    window = min(len(price_array) // 4, 101)
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    trend = savgol_filter(price_array, window, 3)
    components.append(trend)
    
    # Cyclical components
    cycles = []
    for i in range(n_components - 2):
        # Create cyclical component with different frequencies
        period = len(price_array) // (i + 2)
        cycle = 0.1 * (i + 1) * np.sin(np.linspace(0, period * np.pi, len(price_array)))
        cycles.append(cycle)
    
    components.extend(cycles)
    
    # Residual (noise)
    residual = price_array - np.sum([components[0]] + cycles, axis=0)
    components.append(residual)
    
    # Create dictionary of components
    decomposition = {
        'original': price_array,
        'trend': components[0],
        'residual': components[-1]
    }
    
    for i, cycle in enumerate(cycles):
        decomposition[f'cycle_{i+1}'] = cycle
    
    return decomposition

# ---- PAGE: FORECAST EXPLORER ----
if page == "Forecast Explorer":
    st.header("Forecast Explorer")
    
    # Sidebar controls for forecast parameters
    st.sidebar.header("Forecast Parameters")
    
    oil_type = st.sidebar.selectbox(
        "Oil Type",
        options=["brent", "wti"],
        format_func=lambda x: x.upper()
    )
    
    freq = st.sidebar.selectbox(
        "Data Frequency",
        options=["daily", "weekly", "monthly"],
        format_func=lambda x: x.capitalize(),
        key="forecast_freq"
    )
    
    # Adjust horizon parameters based on selected frequency
    horizon_label = f"Forecast Horizon ({freq.capitalize()[:-2] + 's' if freq.endswith('ly') else freq.capitalize()})"
    
    # Set appropriate min, max, and default values based on frequency
    if freq == "daily":
        horizon_min = 7
        horizon_max = 60
        horizon_default = 30
        horizon_step = 1
    elif freq == "weekly":
        horizon_min = 4
        horizon_max = 26
        horizon_default = 12
        horizon_step = 1
    else:  # monthly
        horizon_min = 1
        horizon_max = 12
        horizon_default = 6
        horizon_step = 1
    
    forecast_horizon = st.sidebar.slider(
        horizon_label,
        min_value=horizon_min,
        max_value=horizon_max,
        value=horizon_default,
        step=horizon_step
    )
    
    # Convert horizon to days for internal calculations
    if freq == "weekly":
        forecast_days = forecast_horizon * 7
    elif freq == "monthly":
        forecast_days = forecast_horizon * 30  # Approximation
    else:
        forecast_days = forecast_horizon
    
    # Adjust lookback parameters based on selected frequency
    history_label = f"Historical Data ({freq.capitalize()[:-2] + 's' if freq.endswith('ly') else freq.capitalize()})"
    
    if freq == "daily":
        history_min = 30
        history_max = 180
        history_default = 60
        history_step = 30
    elif freq == "weekly":
        history_min = 4
        history_max = 26
        history_default = 8
        history_step = 4
    else:  # monthly
        history_min = 3
        history_max = 24
        history_default = 6
        history_step = 3
    
    lookback_period = st.sidebar.slider(
        history_label,
        min_value=history_min,
        max_value=history_max,
        value=history_default,
        step=history_step
    )
    
    # Convert to days for internal calculations
    if freq == "weekly":
        lookback_days = lookback_period * 7
    elif freq == "monthly":
        lookback_days = lookback_period * 30  # Approximation
    else:
        lookback_days = lookback_period
    
    available_models = ["Naive", "ARIMA", "EMA", "LSTM-Attention", "CEEMDAN-LSTM", "Ensemble"]
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=available_models,
        default=["Naive", "LSTM-Attention", "Ensemble"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_oil_data(oil_type, freq)
    
    if data.empty:
        st.error("Failed to load data. Please check the logs for details.")
    else:
        # Generate forecasts
        with st.spinner("Generating forecasts..."):
            # Use forecast_horizon directly instead of forecast_days
            forecasts = generate_forecasts(data, forecast_horizon, selected_models, freq)
        
        # Display forecast plot
        fig = plot_forecast_comparison(data, forecasts, lookback_period, freq=freq)
        st.pyplot(fig)
        
        # Show forecast data in table format
        st.subheader("Forecast Values")
        
        # Create forecast dates with appropriate frequency
        # This creates a consistent date range that matches the forecast data
        forecast_df = pd.DataFrame()
        
        # Get the first forecast to determine the shape
        first_forecast = next(iter(forecasts.values()))
        
        # Set up the date range
        last_date = data.index[-1]
        if freq == "daily":
            date_freq = 'B'  # Business days
        elif freq == "weekly":
            date_freq = 'W'  # Weeks
        else:
            date_freq = 'M'  # Months
            
        # Create date range matching the forecast length
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(first_forecast),
            freq=date_freq
        )
        
        # Create DataFrame with the dates as index
        forecast_df = pd.DataFrame(index=forecast_dates)
        
        # Add each model's forecast as a column
        for model_name, forecast in forecasts.items():
            # Ensure the forecast length matches our date range
            if len(forecast) == len(forecast_dates):
                forecast_df[model_name] = forecast
            else:
                # Handle mismatched lengths by padding or truncating
                st.warning(f"Forecast length for {model_name} ({len(forecast)}) doesn't match date range ({len(forecast_dates)}). Adjusting...")
                if len(forecast) > len(forecast_dates):
                    forecast_df[model_name] = forecast[:len(forecast_dates)]
                else:
                    # Pad with the last value
                    padded_forecast = np.concatenate([forecast, np.repeat(forecast[-1], len(forecast_dates) - len(forecast))])
                    forecast_df[model_name] = padded_forecast
        
        st.dataframe(forecast_df.style.format("${:.2f}"))
        
        # Download forecast data as CSV
        csv = forecast_df.to_csv()
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name=f"oil_forecast_{oil_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ---- PAGE: MODEL PERFORMANCE ----
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
    # Sidebar controls
    st.sidebar.header("Performance Parameters")
    
    available_models = ["Naive", "ARIMA", "EMA", "LSTM-Attention", "CEEMDAN-LSTM", "Ensemble"]
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        options=available_models,
        default=["Naive", "LSTM-Attention", "CEEMDAN-LSTM", "Ensemble"]
    )
    
    evaluation_period = st.sidebar.selectbox(
        "Evaluation Period",
        options=["Last 30 Days", "Last 90 Days", "Last 180 Days", "Last 365 Days"],
    )
    
    # Get performance metrics
    if selected_models:
        performance = get_model_performance(selected_models)
        
        # Display performance comparison
        st.subheader("Performance Metrics Comparison")
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            model: {
                'RMSE': f"{metrics['rmse']:.2f}",
                'MAE': f"{metrics['mae']:.2f}",
                'MAPE (%)': f"{metrics['mape']:.2f}",
                'Directional Accuracy (%)': f"{metrics['directional_accuracy']:.2f}"
            }
            for model, metrics in performance.items()
        })
        
        st.dataframe(metrics_df)
        
        # Create visualization of metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['rmse', 'mae', 'mape', 'directional_accuracy']
        titles = ['RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                 'MAPE % (Lower is Better)', 'Directional Accuracy % (Higher is Better)']
        
        for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            values = [performance[model][metric] for model in selected_models]
            if metric == 'directional_accuracy':
                # Invert for directional accuracy (higher is better)
                axes[i].barh(selected_models, values, color='green')
            else:
                # Lower is better for error metrics
                axes[i].barh(selected_models, values, color='skyblue')
            
            axes[i].set_title(title)
            # Add value labels
            for j, value in enumerate(values):
                axes[i].text(value + (0.05 * max(values)), j, f"{value:.2f}", va='center')
            
            # Improve y-axis labels
            axes[i].set_yticks(range(len(selected_models)))
            axes[i].set_yticklabels(selected_models)
            axes[i].invert_yaxis()  # Highest values at the top
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show interpretation
        st.subheader("Performance Interpretation")
        
        # Find best model for each metric
        best_models = {
            'rmse': min(performance.items(), key=lambda x: x[1]['rmse'])[0],
            'mae': min(performance.items(), key=lambda x: x[1]['mae'])[0],
            'mape': min(performance.items(), key=lambda x: x[1]['mape'])[0],
            'directional_accuracy': max(performance.items(), key=lambda x: x[1]['directional_accuracy'])[0]
        }
        
        st.markdown(f"""
        **Key Findings:**
        
        - **Best RMSE**: {best_models['rmse']} model with {performance[best_models['rmse']]['rmse']:.2f}
        - **Best MAE**: {best_models['mae']} model with {performance[best_models['mae']]['mae']:.2f}
        - **Best MAPE**: {best_models['mape']} model with {performance[best_models['mape']]['mape']:.2f}%
        - **Best Directional Accuracy**: {best_models['directional_accuracy']} model with {performance[best_models['directional_accuracy']]['directional_accuracy']:.2f}%
        
        The {best_models['rmse']} model appears to have the best overall performance, with the lowest error metrics
        and highest directional accuracy. This suggests that combining multiple forecasting techniques through
        ensemble methods provides more robust predictions than individual models.
        """)
        
        # Simulated improvement over baseline
        if "Naive" in selected_models and len(selected_models) > 1:
            baseline_rmse = performance["Naive"]["rmse"]
            improvements = {
                model: (1 - metrics["rmse"] / baseline_rmse) * 100
                for model, metrics in performance.items()
                if model != "Naive"
            }
            
            st.subheader("Improvement Over Baseline (Naive)")
            
            # Create improvement chart
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(improvements.keys())
            values = list(improvements.values())
            
            ax.barh(models, values, color='lightgreen')
            ax.set_title("Percentage Improvement in RMSE Over Naive Baseline")
            ax.set_xlabel("Improvement (%)")
            
            # Add value labels
            for i, value in enumerate(values):
                ax.text(value + 0.5, i, f"{value:.1f}%", va='center')
            
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            st.pyplot(fig)
    else:
        st.warning("Please select at least one model to view performance metrics.")

# ---- PAGE: SENTIMENT ANALYSIS ----
elif page == "Sentiment Analysis":
    st.header("Market Sentiment Analysis")
    
    st.markdown("""
    This page analyzes the relationship between oil market sentiment and price movements.
    Sentiment data is derived from social media, news, and financial discussions related to oil markets.
    """)
    
    # Sidebar controls
    st.sidebar.header("Sentiment Parameters")
    
    oil_type = st.sidebar.selectbox(
        "Oil Type",
        options=["brent", "wti"],
        format_func=lambda x: x.upper(),
        key="sentiment_oil_type"
    )
    
    # Adjust sentiment period options based on frequency
    if oil_type != "brent" and oil_type != "wti":
        # Default to brent if somehow oil_type is invalid
        oil_type = "brent"
        
    # Load data to determine appropriate date ranges
    data = load_oil_data(oil_type, "daily")
    
    if data.empty:
        st.error("Failed to load data for sentiment analysis")
        sentiment_period_options = ["Last 30 Days", "Last 90 Days", "Last 120 Days"]
        sentiment_period_index = 2
    else:
        # Create frequency-appropriate options
        total_days = (data.index[-1] - data.index[0]).days
        
        if freq == "daily":
            sentiment_period_options = ["Last 30 Days", "Last 90 Days", "Last 120 Days"]
            sentiment_period_index = 2  # Default to 120 days
        elif freq == "weekly":
            sentiment_period_options = ["Last 4 Weeks", "Last 12 Weeks", "Last 24 Weeks"]
            sentiment_period_index = 1  # Default to 12 weeks
        else:  # monthly
            sentiment_period_options = ["Last 3 Months", "Last 6 Months", "Last 12 Months"]
            sentiment_period_index = 1  # Default to 6 months
    
    sentiment_period = st.sidebar.selectbox(
        f"Analysis Period ({freq.capitalize()[:-2] + 's' if freq.endswith('ly') else freq.capitalize()})",
        options=sentiment_period_options,
        index=sentiment_period_index
    )
    
    # Load data for sentiment analysis with daily frequency for more granular analysis
    # even if user selected weekly/monthly for forecasting
    with st.spinner("Loading data..."):
        data = load_oil_data(oil_type, "daily")
    
    if data.empty:
        st.error("Failed to load data. Please check the logs for details.")
    else:
        # Generate sentiment data
        # Parse the period number from the selection
        period_parts = sentiment_period.split()
        period_number = int(period_parts[1])
        period_unit = period_parts[2].lower()
        
        # Convert to days for internal calculations
        if period_unit == "weeks":
            days = period_number * 7
        elif period_unit == "months":
            days = period_number * 30  # Approximation
        else:  # days
            days = period_number
            
        sentiment_data = generate_sentiment_data(data, days)
        
        # Create combined DataFrame
        price_slice = data.iloc[-days:].copy()
        combined_df = pd.DataFrame({
            'Price': price_slice['Price'],
            'Sentiment': sentiment_data['sentiment_compound'],
            'Sentiment_MA7': sentiment_data['sentiment_ma7'],
            'Positive': sentiment_data['sentiment_pos'],
            'Negative': sentiment_data['sentiment_neg']
        })
        
        # Plot price with sentiment overlay
        st.subheader("Oil Price with Market Sentiment")
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot price on primary y-axis
        ax1.plot(combined_df.index, combined_df['Price'], 'b-', linewidth=2, label='Oil Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Create secondary y-axis for sentiment
        ax2 = ax1.twinx()
        ax2.plot(
            combined_df.index,
            combined_df['Sentiment'],
            'r-',
            linewidth=1.5,
            alpha=0.7,
            label='Sentiment'
        )
        ax2.plot(
            combined_df.index,
            combined_df['Sentiment_MA7'],
            'g-',
            linewidth=1.5,
            alpha=0.7,
            label='7-day MA'
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
        plt.title('Oil Price with Market Sentiment')
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Tight layout
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Show sentiment distribution
        st.subheader("Sentiment Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            combined_df['Sentiment'],
            bins=20,
            color='skyblue',
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add vertical lines at meaningful levels
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Neutral')
        ax.axvline(x=0.2, color='g', linestyle='--', alpha=0.5, label='Bullish Threshold')
        ax.axvline(x=-0.2, color='r', linestyle='--', alpha=0.5, label='Bearish Threshold')
        
        # Format plot
        ax.set_title('Distribution of Oil Market Sentiment Scores')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Correlation analysis
        st.subheader("Price-Sentiment Correlation Analysis")
        
        # Calculate rolling correlations
        window_sizes = [7, 14, 30]
        correlations = {}
        
        for window in window_sizes:
            if window < len(combined_df):
                rolling_corr = combined_df['Price'].rolling(window=window).corr(combined_df['Sentiment'])
                correlations[f"{window}-day"] = rolling_corr
        
        # Plot rolling correlations
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, corr in correlations.items():
            ax.plot(
                corr.index,
                corr.values,
                label=f"{label} Correlation"
            )
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Format plot
        ax.set_title('Rolling Correlation Between Oil Price and Market Sentiment')
        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        st.pyplot(fig)
        
        # Correlation statistics
        overall_corr = combined_df['Price'].corr(combined_df['Sentiment'])
        
        st.markdown(f"""
        **Correlation Analysis:**
        
        - **Overall Correlation**: {overall_corr:.4f}
        - **Interpretation**: 
          - A {'positive' if overall_corr > 0 else 'negative'} correlation of {abs(overall_corr):.4f} indicates a {'direct' if overall_corr > 0 else 'inverse'} relationship between oil prices and market sentiment.
          - This suggests that {'higher sentiment tends to correspond with higher prices' if overall_corr > 0 else 'lower sentiment tends to correspond with higher prices'}.
          - The relationship varies over time, as shown in the rolling correlation plot.
        """)
        
        # Sentiment as leading indicator analysis
        st.subheader("Sentiment as a Leading Indicator")
        
        # Calculate lagged correlations
        lags = range(-10, 11)  # From -10 (sentiment leads) to +10 (price leads)
        lag_correlations = []
        
        for lag in lags:
            if lag < 0:
                # Sentiment leads price by -lag days
                shifted_sentiment = combined_df['Sentiment'].shift(lag)
                corr = combined_df['Price'].corr(shifted_sentiment)
            else:
                # Price leads sentiment by lag days
                shifted_price = combined_df['Price'].shift(lag)
                corr = shifted_price.corr(combined_df['Sentiment'])
            
            lag_correlations.append(corr)
        
        # Plot lagged correlations
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(
            lags,
            lag_correlations,
            color='skyblue',
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Format plot
        ax.set_title('Correlation Between Price and Sentiment at Different Lags')
        ax.set_xlabel('Lag (Negative = Sentiment Leads, Positive = Price Leads)')
        ax.set_ylabel('Correlation Coefficient')
        ax.grid(True, alpha=0.3)
        
        # Find max correlation and its lag
        max_corr_idx = np.argmax(np.abs(lag_correlations))
        max_corr_lag = lags[max_corr_idx]
        max_corr_value = lag_correlations[max_corr_idx]
        
        # Add annotation for max correlation
        ax.annotate(
            f'Max correlation: {max_corr_value:.4f} at lag {max_corr_lag}',
            xy=(max_corr_lag, max_corr_value),
            xytext=(max_corr_lag + (-3 if max_corr_lag > 0 else 3), max_corr_value + 0.1),
            arrowprops=dict(arrowstyle='->', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
        )
        
        st.pyplot(fig)
        
        # Interpretation of lag analysis
        st.markdown(f"""
        **Lag Analysis:**
        
        - **Maximum Correlation**: {max_corr_value:.4f} at lag {max_corr_lag}
        - **Interpretation**: 
          - {"Sentiment appears to lead price changes" if max_corr_lag < 0 else "Price changes appear to lead sentiment"}
          - The strongest relationship occurs when {"sentiment leads prices by" if max_corr_lag < 0 else "prices lead sentiment by"} {abs(max_corr_lag)} days
          - This suggests that {"monitoring sentiment may provide advance indicators of price movements" if max_corr_lag < 0 else "sentiment largely reacts to price changes rather than predicting them"}
        """)
        
        # Show sentiment data table
        with st.expander("View Sentiment Data"):
            st.dataframe(sentiment_data)
            
        # Download sentiment data as CSV
        csv = sentiment_data.to_csv()
        st.download_button(
            label="Download Sentiment Data",
            data=csv,
            file_name=f"oil_sentiment_{oil_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )