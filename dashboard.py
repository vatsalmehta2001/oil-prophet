"""
Oil Prophet Dashboard - Interactive visualization for oil price forecasting with sentiment analysis.

This dashboard provides a user interface for exploring and visualizing forecasts from
different models, comparing their performance, and analyzing the effects of market sentiment
on oil price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Import project modules
from src.data.preprocessing import OilDataProcessor
from src.models.baseline import BaselineForecaster
from src.models.lstm_attention import LSTMWithAttention, AttentionLayer
from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
from src.models.ensemble import EnsembleForecaster
from src.models.ceemdan import SimplifiedDecomposer
from src.nlp.finbert_sentiment import OilFinBERT
from src.evaluation.metrics import calculate_metrics

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
    Interactive dashboard for oil price forecasting powered by advanced time-series decomposition,
    deep learning models, and market sentiment analysis. Explore forecasts, model performance,
    and the influence of market sentiment on oil prices.
""")

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Forecast Explorer", "Model Performance", "Sentiment Analysis", "Signal Decomposition"]
)

# Function to load available models
@st.cache_resource
def load_models() -> Dict[str, object]:
    """
    Load the trained models.
    
    Returns:
        Dictionary of model objects
    """
    models = {}
    
    # Check if model files exist
    model_files = {
        "LSTM-Attention": "models/lstm_price_only.h5" if os.path.exists("models/lstm_price_only.h5") else None,
        "Sentiment-Enhanced LSTM": "models/sentiment_enhanced_lstm.h5" if os.path.exists("models/sentiment_enhanced_lstm.h5") else None
    }
    
    for model_name, model_path in model_files.items():
        if model_path is not None:
            try:
                if model_name == "LSTM-Attention":
                    # Load LSTM with Attention model
                    custom_objects = {'AttentionLayer': AttentionLayer}
                    models[model_name] = LSTMWithAttention.load(model_path)
                    logger.info(f"Loaded {model_name} model from {model_path}")
                elif model_name == "Sentiment-Enhanced LSTM":
                    # Load Sentiment-Enhanced LSTM model
                    models[model_name] = SentimentEnhancedLSTM.load(model_path)
                    logger.info(f"Loaded {model_name} model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")
                
    
    # Add baseline models
    try:
        models["Naive"] = BaselineForecaster(method='naive')
        models["ARIMA"] = BaselineForecaster(method='arima')
        models["EMA"] = BaselineForecaster(method='ema')
        logger.info("Created baseline models")
    except Exception as e:
        logger.error(f"Error creating baseline models: {str(e)}")
    
    return models

# Function to load data
@st.cache_data
def load_oil_data(oil_type: str, freq: str) -> pd.DataFrame:
    """
    Load and cache oil price data.
    
    Args:
        oil_type: Type of oil price data ('brent' or 'wti')
        freq: Frequency of data ('daily', 'weekly', 'monthly')
        
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

# Function to load sentiment data
@st.cache_data
def load_sentiment_data(data_dir: str) -> pd.DataFrame:
    """
    Load and cache sentiment data.
    
    Args:
        data_dir: Directory containing sentiment data
        
    Returns:
        DataFrame with sentiment data
    """
    # Look for the sentiment dataset in the specified directory
    sentiment_file = os.path.join(data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    if os.path.exists(sentiment_file):
        try:
            sentiment_data = pd.read_csv(sentiment_file)
            
            # Ensure created_date is a datetime
            if 'created_date' in sentiment_data.columns:
                sentiment_data['created_date'] = pd.to_datetime(sentiment_data['created_date'])
            if not forecasts and 'Naive' in models:
                # Generate at least a naive forecast
                try:
                    if 'X_price_test' in dataset and len(dataset['X_price_test']) > 0:
                        price_data = dataset['X_price_test'][-1, :, 0]
                        naive_model = models['Naive']
                        naive_model.fit(price_data)
                        forecasts['Naive'] = naive_model.predict(steps=forecast_horizon)
                except Exception as e:
                    logger.error(f"Error generating fallback forecast: {str(e)}")
                
            # Or try created_utc
            elif 'created_utc' in sentiment_data.columns:
                sentiment_data['created_date'] = pd.to_datetime(sentiment_data['created_utc'], unit='s')
            
            return sentiment_data
        except Exception as e:
            st.error(f"Error loading sentiment data: {str(e)}")
    else:
        st.warning(f"No sentiment data found at {sentiment_file}")
    
    return pd.DataFrame()

# Function to prepare dataset for prediction
def prepare_prediction_dataset(
    price_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    window_size: int = 30,
    forecast_horizon: int = 7
) -> Dict[str, np.ndarray]:
    """
    Prepare dataset for prediction.
    
    Args:
        price_data: DataFrame with price data
        sentiment_data: DataFrame with sentiment data
        window_size: Size of lookback window
        forecast_horizon: Number of steps to forecast
        
    Returns:
        Dictionary with prepared dataset
    """
    try:
        dataset = prepare_sentiment_features(
            price_df=price_data,
            sentiment_df=sentiment_data,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        return dataset
    except Exception as e:
        st.error(f"Error preparing dataset: {str(e)}")
        return {}

# Function to generate forecasts
def generate_forecasts(
    models: Dict[str, object],
    dataset: Dict[str, np.ndarray],
    forecast_horizon: int
) -> Dict[str, np.ndarray]:
    """
    Generate forecasts using selected models.
    
    Args:
        models: Dictionary of model objects
        dataset: Prepared dataset
        forecast_horizon: Number of periods to forecast
        
    Returns:
        Dictionary of forecasts from each model
    """
    forecasts = {}
    
    # Get the latest data for prediction
    if 'X_price_test' in dataset and len(dataset['X_price_test']) > 0:
        X_price_recent = dataset['X_price_test'][-1:]
        
        # For baseline models, use the raw price data
        price_data = dataset['X_price_test'][-1, :, 0]  # Extract price values
        
        # Generate forecasts for each model
        for model_name, model in models.items():
            try:
                if model_name == "LSTM-Attention":
                    forecast = model.predict(X_price_recent)[0]
                    forecasts[model_name] = forecast
                
                elif model_name == "Sentiment-Enhanced LSTM" and 'X_sentiment_test' in dataset:
                    X_sentiment_recent = dataset['X_sentiment_test'][-1:]
                    forecast = model.predict(X_price_recent, X_sentiment_recent)[0]
                    forecasts[model_name] = forecast
                
                elif model_name in ["Naive", "ARIMA", "EMA"]:
                    model.fit(price_data)
                    forecast = model.predict(steps=forecast_horizon)
                    forecasts[model_name] = forecast
            
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
    
    # If we have at least two models, create an ensemble
    if len(forecasts) >= 2:
        try:
            ensemble_forecast = np.zeros(forecast_horizon)
            for forecast in forecasts.values():
                ensemble_forecast += forecast
            ensemble_forecast /= len(forecasts)
            forecasts["Ensemble"] = ensemble_forecast
        except Exception as e:
            logger.error(f"Error creating ensemble forecast: {str(e)}")
    
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
        freq: Data frequency
        
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

# Function to decompose price signal
@st.cache_data
def decompose_price_signal(price_data: pd.DataFrame, n_components: int = 5) -> Dict[str, np.ndarray]:
    """
    Decompose price signal into components.
    
    Args:
        price_data: DataFrame with price data
        n_components: Number of components to extract
        
    Returns:
        Dictionary of decomposed components
    """
    try:
        # Extract price data
        price_array = price_data['Price'].values
        
        # Create decomposer
        decomposer = SimplifiedDecomposer(n_components=n_components)
        
        # Decompose signal
        components = decomposer.decompose(price_array)
        
        # Create result dictionary
        decomposition = {
            'original': price_array,
            'trend': components[0]
        }
        
        # Add cyclical components
        for i in range(1, n_components-1):
            decomposition[f'cycle_{i}'] = components[i]
        
        # Add residual
        decomposition['residual'] = components[-1]
        
        return decomposition
    except Exception as e:
        logger.error(f"Error decomposing price signal: {str(e)}")
        return {}

# Function to aggregate sentiment by time
def aggregate_sentiment(
    sentiment_data: pd.DataFrame,
    freq: str = 'D',
    date_col: str = 'created_date'
) -> pd.DataFrame:
    """
    Aggregate sentiment data by time period.
    
    Args:
        sentiment_data: DataFrame with sentiment data
        freq: Aggregation frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        date_col: Column containing dates
        
    Returns:
        DataFrame with aggregated sentiment
    """
    if sentiment_data.empty or date_col not in sentiment_data.columns:
        return pd.DataFrame()
    
    # Initialize FinBERT for aggregation
    analyzer = OilFinBERT()
    
    # Use the analyzer to aggregate sentiment
    try:
        agg_sentiment = analyzer.aggregate_sentiment_by_time(
            sentiment_data,
            date_col=date_col,
            time_freq=freq[0],  # Take first letter (D, W, M)
            min_count=1
        )
        return agg_sentiment
    except Exception as e:
        logger.error(f"Error aggregating sentiment: {str(e)}")
        return pd.DataFrame()

# Function to calculate model performance metrics
def calculate_model_performance(
    test_data: np.ndarray,
    forecasts: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics for each model.
    
    Args:
        test_data: Actual values
        forecasts: Dictionary of forecasts from each model
        
    Returns:
        Dictionary of performance metrics for each model
    """
    performance = {}
    
    for model_name, forecast in forecasts.items():
        try:
            # Ensure forecast and test data have the same length
            min_len = min(len(forecast), len(test_data))
            
            # Calculate metrics
            metrics = calculate_metrics(test_data[:min_len], forecast[:min_len])
            
            # Store in performance dictionary
            performance[model_name] = metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
    
    return performance

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
    
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="data/processed/reddit_test_small",
        help="Directory containing Reddit data with sentiment analysis"
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
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        
        # If models were loaded, create a selection box
        if models:
            available_models = list(models.keys())
            if "Ensemble" not in available_models:
                available_models.append("Ensemble")
        else:
            # If no models were loaded, use placeholders
            available_models = ["Naive", "ARIMA", "EMA", "LSTM-Attention", "Sentiment-Enhanced LSTM", "Ensemble"]
    
    selected_models = st.sidebar.multiselect(
    "Select Models",
    options=available_models,
    default=["Naive"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        # Load price data
        price_data = load_oil_data(oil_type, freq)
        
        # Load sentiment data
        sentiment_data = load_sentiment_data(data_dir)
    
    if price_data.empty:
        st.error("Failed to load price data. Please check the logs for details.")
    else:
        # Progress info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Loaded {len(price_data)} price data points")
        with col2:
            if not sentiment_data.empty:
                st.success(f"Loaded {len(sentiment_data)} sentiment data points")
            else:
                st.warning("No sentiment data loaded. Sentiment-enhanced models may not work correctly.")
        
        # Prepare dataset for prediction if sentiment data is available
        if not sentiment_data.empty:
            with st.spinner("Preparing dataset..."):
                dataset = prepare_prediction_dataset(
                    price_data=price_data,
                    sentiment_data=sentiment_data,
                    window_size=30,  # Default window size
                    forecast_horizon=forecast_horizon
                )
        else:
            dataset = {
                'X_price_test': np.expand_dims(price_data['Price'].values[-30:], axis=(0, 2)),
                'y_test': np.zeros((1, forecast_horizon))  # Placeholder
            }
        
        # Generate forecasts
        with st.spinner("Generating forecasts..."):
            # Filter to only use selected models
            selected_model_objects = {name: model for name, model in models.items() if name in selected_models}
            
            # Generate forecasts
            forecasts = generate_forecasts(
                selected_model_objects,
                dataset,
                forecast_horizon
            )
        
        # Check if forecasts were generated
        if forecasts:
            # Display forecast plot
            fig = plot_forecast_comparison(price_data, forecasts, lookback_period, freq=freq)
            st.pyplot(fig)
            
            # Show forecast data in table format
            st.subheader("Forecast Values")
            
            # Get the first forecast to determine the shape
            first_forecast = next(iter(forecasts.values()))
            
            # Set up the date range
            last_date = price_data.index[-1]
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
        else:
            st.error("Failed to generate forecasts. Please check the logs for details.")

# ---- PAGE: MODEL PERFORMANCE ----
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
    # Sidebar controls
    st.sidebar.header("Performance Parameters")
    
    oil_type = st.sidebar.selectbox(
        "Oil Type",
        options=["brent", "wti"],
        format_func=lambda x: x.upper(),
        key="perf_oil_type"
    )
    
    freq = st.sidebar.selectbox(
        "Data Frequency",
        options=["daily", "weekly", "monthly"],
        format_func=lambda x: x.capitalize(),
        key="perf_freq"
    )
    
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="data/processed/reddit_test_small",
        help="Directory containing Reddit data with sentiment analysis",
        key="perf_data_dir"
    )
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        
    # Get available model names
    if models:
        available_models = list(models.keys())
        if "Ensemble" not in available_models:
            available_models.append("Ensemble")
    else:
        available_models = ["Naive", "ARIMA", "EMA", "LSTM-Attention", "Sentiment-Enhanced LSTM", "Ensemble"]
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        options=available_models,
        default=["Naive", "LSTM-Attention", "Sentiment-Enhanced LSTM", "Ensemble"]
    )
    
    test_period_options = {
        "daily": ["Last 30 Days", "Last 90 Days", "Last 180 Days"],
        "weekly": ["Last 4 Weeks", "Last 12 Weeks", "Last 24 Weeks"],
        "monthly": ["Last 3 Months", "Last 6 Months", "Last 12 Months"]
    }
    
    test_period = st.sidebar.selectbox(
        "Test Period",
        options=test_period_options[freq],
        index=1  # Default to middle option
    )
    
    # Load data
    with st.spinner("Loading data..."):
        # Load price data
        price_data = load_oil_data(oil_type, freq)
        
        # Load sentiment data
        sentiment_data = load_sentiment_data(data_dir)
    
    if price_data.empty:
        st.error("Failed to load price data. Please check the logs for details.")
    elif not selected_models:
        st.warning("Please select at least one model to view performance metrics.")
    else:
        # Progress info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Loaded {len(price_data)} price data points")
        with col2:
            if not sentiment_data.empty:
                st.success(f"Loaded {len(sentiment_data)} sentiment data points")
            else:
                st.warning("No sentiment data loaded. Sentiment-enhanced models may not work correctly.")
        
        # Prepare test dataset
        with st.spinner("Preparing test dataset..."):
            # Determine test period length
            period_parts = test_period.split()
            period_number = int(period_parts[1])
            period_unit = period_parts[2].lower()
            
            # Convert to appropriate number of data points
            if period_unit == "days":
                test_length = period_number
            elif period_unit == "weeks":
                test_length = period_number if freq == "weekly" else period_number * 7
            else:  # months
                test_length = period_number if freq == "monthly" else period_number * 30
            
            # Ensure test_length is not larger than available data
            test_length = min(test_length, len(price_data))
            
            # Extract test data
            test_data = price_data.iloc[-test_length:]
            
            if not sentiment_data.empty:
                # Prepare dataset with sentiment
                dataset = prepare_prediction_dataset(
                    price_data=price_data,
                    sentiment_data=sentiment_data,
                    window_size=30,  # Default window size
                    forecast_horizon=7  # Default forecast horizon for evaluation
                )
            else:
                # Basic dataset without sentiment
                dataset = {
                    'X_price_test': np.expand_dims(price_data['Price'].values[-30:], axis=(0, 2)),
                    'y_test': np.zeros((1, 7))  # Placeholder
                }
        
        # Check if test data is available
        if 'y_test' in dataset:
            # Filter to only use selected models
            selected_model_objects = {name: model for name, model in models.items() if name in selected_models}
            
            # Use a shorter forecast horizon for evaluation
            forecast_horizon = 7  # Default to 7 days/weeks/months for evaluation
            
            # Generate forecasts for test period
            with st.spinner("Evaluating models..."):
                forecasts = generate_forecasts(
                    selected_model_objects,
                    dataset,
                    forecast_horizon
                )
                
                # Calculate performance metrics
                performance = calculate_model_performance(dataset['y_test'][0], forecasts)
        
            # Display performance comparison
            if performance:
                st.subheader("Performance Metrics Comparison")
                
                # Create metrics DataFrame
                metrics_df = pd.DataFrame({
                    model: {
                        'RMSE': f"{metrics.get('rmse', 0):.2f}",
                        'MAE': f"{metrics.get('mae', 0):.2f}",
                        'MAPE (%)': f"{metrics.get('mape', 0):.2f}",
                        'Directional Accuracy (%)': f"{metrics.get('directional_accuracy', 0):.2f}"
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
                    values = [performance.get(model, {}).get(metric, 0) for model in performance.keys()]
                    if metric == 'directional_accuracy':
                        # Invert for directional accuracy (higher is better)
                        axes[i].barh(list(performance.keys()), values, color='green')
                    else:
                        # Lower is better for error metrics
                        axes[i].barh(list(performance.keys()), values, color='skyblue')
                    
                    axes[i].set_title(title)
                    # Add value labels
                    for j, value in enumerate(values):
                        axes[i].text(value + (0.05 * max(values)) if values else 0, j, f"{value:.2f}", va='center')
                    
                    # Improve y-axis labels
                    axes[i].set_yticks(range(len(performance)))
                    axes[i].set_yticklabels(list(performance.keys()))
                    axes[i].invert_yaxis()  # Highest values at the top
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show interpretation
                st.subheader("Performance Interpretation")
                
                # Find best model for each metric
                best_models = {}
                for metric in metrics_to_plot:
                    if metric == 'directional_accuracy':
                        # Higher is better
                        best_model = max(
                            performance.items(),
                            key=lambda x: x[1].get(metric, 0),
                            default=(None, {})
                        )[0]
                    else:
                        # Lower is better
                        best_model = min(
                            performance.items(),
                            key=lambda x: x[1].get(metric, float('inf')),
                            default=(None, {})
                        )[0]
                    
                    best_models[metric] = best_model
                
                # Count which model is best most often
                model_counts = {}
                for model in best_models.values():
                    if model:
                        model_counts[model] = model_counts.get(model, 0) + 1
                
                overall_best = max(model_counts.items(), key=lambda x: x[1], default=(None, 0))[0]
                
                st.markdown(f"""
                **Key Findings:**
                
                - **Best RMSE**: {best_models.get('rmse', 'N/A')} model with {performance.get(best_models.get('rmse', ''), {}).get('rmse', 0):.2f}
                - **Best MAE**: {best_models.get('mae', 'N/A')} model with {performance.get(best_models.get('mae', ''), {}).get('mae', 0):.2f}
                - **Best MAPE**: {best_models.get('mape', 'N/A')} model with {performance.get(best_models.get('mape', ''), {}).get('mape', 0):.2f}%
                - **Best Directional Accuracy**: {best_models.get('directional_accuracy', 'N/A')} model with {performance.get(best_models.get('directional_accuracy', ''), {}).get('directional_accuracy', 0):.2f}%
                
                The **{overall_best or 'N/A'}** model appears to have the best overall performance across multiple metrics.
                This suggests that {"combining market sentiment with technical price patterns provides more accurate forecasts" if overall_best == "Sentiment-Enhanced LSTM" else "ensemble methods provide more robust predictions than individual models" if overall_best == "Ensemble" else "this model balances accuracy and reliability well for oil price forecasting"}.
                """)
                
                # Simulated improvement over baseline
                if "Naive" in performance and len(performance) > 1:
                    baseline_rmse = performance["Naive"].get("rmse", 0)
                    if baseline_rmse > 0:
                        improvements = {
                            model: (1 - metrics.get("rmse", 0) / baseline_rmse) * 100
                            for model, metrics in performance.items()
                            if model != "Naive"
                        }
                        
                        st.subheader("Improvement Over Naive Baseline")
                        
                        # Create improvement chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        models = list(improvements.keys())
                        values = list(improvements.values())
                        
                        ax.barh(models, values, color=['lightgreen' if v > 0 else 'lightcoral' for v in values])
                        ax.set_title("Percentage Improvement in RMSE Over Naive Baseline")
                        ax.set_xlabel("Improvement (%)")
                        
                        # Add value labels
                        for i, value in enumerate(values):
                            ax.text(value + 0.5, i, f"{value:.1f}%", va='center')
                        
                        ax.grid(True, alpha=0.3)
                        ax.set_axisbelow(True)
                        
                        st.pyplot(fig)
            else:
                st.warning("No performance metrics were calculated. This may be due to insufficient test data.")
        else:
            st.warning("No test data available for model evaluation.")

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
    
    freq = st.sidebar.selectbox(
        "Data Frequency",
        options=["daily", "weekly", "monthly"],
        format_func=lambda x: x.capitalize(),
        key="sentiment_freq"
    )
    
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="data/processed/reddit_test_small",
        help="Directory containing Reddit data with sentiment analysis",
        key="sentiment_data_dir"
    )
    
    # Load data
    with st.spinner("Loading data..."):
        # Load price data
        price_data = load_oil_data(oil_type, freq)
        
        # Load sentiment data
        sentiment_data = load_sentiment_data(data_dir)
    
    if price_data.empty:
        st.error("Failed to load price data. Please check the logs for details.")
    elif sentiment_data.empty:
        st.error("Failed to load sentiment data. Please check the data directory.")
    else:
        # Aggregate sentiment by time
        agg_sentiment = aggregate_sentiment(
            sentiment_data,
            freq=freq[0],  # 'D', 'W', or 'M'
            date_col='created_date'
        )
        
        if agg_sentiment.empty:
            st.error("Failed to aggregate sentiment data. Check the log for details.")
        else:
            # Align price and sentiment data
            # Convert price data index to datetime if needed
            if not isinstance(price_data.index, pd.DatetimeIndex):
                st.warning("Price data index is not a DatetimeIndex. This may affect time-based analysis.")
            
            # Merge price and sentiment data
            price_df = price_data.reset_index()
            price_df.columns = ['Date', 'Price']  # Match original column names
            price_df = price_df.rename(columns={'Date': 'date', 'Price': 'price'})  # Then standardize
            
            # Rename sentiment index to 'date' for merging
            sentiment_df = agg_sentiment.reset_index()
            sentiment_df.columns = ['date'] + list(sentiment_df.columns[1:])
            
            # Merge on closest date
            merged = pd.merge_asof(
                price_df.sort_values('date'),
                sentiment_df.sort_values('date'),
                on='date',
                direction='nearest'
            )
            
            # Fill any missing sentiment values with zeros (neutral)
            for col in ['sentiment_compound', 'sentiment_positive', 'sentiment_negative']:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)
            
            # Create 7-day moving average for sentiment
            if 'sentiment_compound' in merged.columns and len(merged) > 7:
                merged['sentiment_ma7'] = merged['sentiment_compound'].rolling(window=7).mean().fillna(0)
            
            # Plot price with sentiment overlay
            st.subheader("Oil Price with Market Sentiment")
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot price on primary y-axis
            ax1.plot(merged['date'], merged['price'], 'b-', linewidth=2, label='Oil Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price ($)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Create secondary y-axis for sentiment
            ax2 = ax1.twinx()
            if 'sentiment_compound' in merged.columns:
                ax2.plot(
                    merged['date'],
                    merged['sentiment_compound'],
                    'r-',
                    linewidth=1.5,
                    alpha=0.7,
                    label='Sentiment'
                )
                
                if 'sentiment_ma7' in merged.columns:
                    ax2.plot(
                        merged['date'],
                        merged['sentiment_ma7'],
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
            
            # Sentiment distribution
            if 'sentiment_compound' in merged.columns:
                st.subheader("Sentiment Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(
                    merged['sentiment_compound'].dropna(),
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
                    if window < len(merged):
                        rolling_corr = merged['price'].rolling(window=window).corr(merged['sentiment_compound'])
                        correlations[f"{window}-day"] = rolling_corr
                
                # Plot rolling correlations
                if correlations:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for label, corr in correlations.items():
                        ax.plot(
                            merged['date'].iloc[window_sizes[0]-1:],  # Start at largest window size
                            corr.iloc[window_sizes[0]-1:],
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
                    overall_corr = merged['price'].corr(merged['sentiment_compound'])
                    
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
                        shifted_sentiment = merged['sentiment_compound'].shift(lag)
                        corr = merged['price'].corr(shifted_sentiment)
                    else:
                        # Price leads sentiment by lag days
                        shifted_price = merged['price'].shift(lag)
                        corr = shifted_price.corr(merged['sentiment_compound'])
                    
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
                with st.expander("View Aggregated Sentiment Data"):
                    st.dataframe(agg_sentiment)
                
                # Show raw sentiment data
                with st.expander("View Raw Sentiment Data"):
                    st.dataframe(sentiment_data)
                    
                # Download sentiment data as CSV
                csv = agg_sentiment.to_csv()
                st.download_button(
                    label="Download Aggregated Sentiment Data",
                    data=csv,
                    file_name=f"oil_sentiment_{oil_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ---- PAGE: SIGNAL DECOMPOSITION ----
elif page == "Signal Decomposition":
    st.header("Signal Decomposition Analysis")
    
    st.markdown("""
    This page analyzes the decomposition of oil price signals into trend, cyclical, and residual components.
    Decomposition helps understand the underlying patterns in price movements and can improve forecasting accuracy.
    """)
    
    # Sidebar controls
    st.sidebar.header("Decomposition Parameters")
    
    oil_type = st.sidebar.selectbox(
        "Oil Type",
        options=["brent", "wti"],
        format_func=lambda x: x.upper(),
        key="decomp_oil_type"
    )
    
    freq = st.sidebar.selectbox(
        "Data Frequency",
        options=["daily", "weekly", "monthly"],
        format_func=lambda x: x.capitalize(),
        key="decomp_freq"
    )
    
    # Add slider for number of components
    n_components = st.sidebar.slider(
        "Number of Components",
        min_value=3,
        max_value=7,
        value=5,
        step=1,
        help="Number of components to extract from the price signal"
    )
    
    # Add slider for time period
    period_options = {
        "daily": [90, 180, 365, 730],
        "weekly": [26, 52, 104, 156],
        "monthly": [12, 24, 36, 60]
    }
    
    period_option_labels = {
        "daily": ["3 Months", "6 Months", "1 Year", "2 Years"],
        "weekly": ["6 Months", "1 Year", "2 Years", "3 Years"],
        "monthly": ["1 Year", "2 Years", "3 Years", "5 Years"]
    }
    
    period_idx = st.sidebar.selectbox(
        "Analysis Period",
        options=range(len(period_options[freq])),
        format_func=lambda x: period_option_labels[freq][x],
        key="decomp_period"
    )
    
    period_length = period_options[freq][period_idx]
    
    # Load data
    with st.spinner("Loading data..."):
        # Load price data
        price_data = load_oil_data(oil_type, freq)
    
    if price_data.empty:
        st.error("Failed to load price data. Please check the logs for details.")
    else:
        # Limit to selected period
        price_data_period = price_data.iloc[-period_length:]
        
        st.info(f"Loaded {len(price_data_period)} price data points from {price_data_period.index[0].date()} to {price_data_period.index[-1].date()}")
        
        # Decompose price signal
        with st.spinner("Decomposing price signal..."):
            decomposition = decompose_price_signal(price_data_period, n_components=n_components)
        
        if not decomposition:
            st.error("Failed to decompose price signal. Check the log for details.")
        else:
            # Show original price series
            st.subheader("Original Price Series")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(price_data_period.index, decomposition['original'], 'b-', linewidth=2)
            ax.set_title(f"{oil_type.upper()} Price ({freq.capitalize()})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig)
            
            # Show decomposed components
            st.subheader("Decomposed Components")
            
            # Create subplot grid for all components
            n_plots = len(decomposition)
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
            
            # Plot each component
            for i, (name, component) in enumerate(decomposition.items()):
                ax = axes[i] if n_plots > 1 else axes
                ax.plot(price_data_period.index, component, linewidth=2)
                ax.set_title(f"Component: {name.capitalize()}")
                ax.grid(True, alpha=0.3)
                
                # Add y-label only for the original and trend components
                if name in ['original', 'trend']:
                    ax.set_ylabel("Price (USD)")
                else:
                    ax.set_ylabel("Amplitude")
            
            # Add x-label to the bottom plot
            if n_plots > 1:
                axes[-1].set_xlabel("Date")
            else:
                axes.set_xlabel("Date")
            
            # Format x-axis dates
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interactive component analysis
            st.subheader("Interactive Component Analysis")
            
            # Select components to compare
            components_to_compare = st.multiselect(
                "Select Components to Compare",
                options=list(decomposition.keys()),
                default=["original", "trend"],
                key="components_to_compare"
            )
            
            if components_to_compare:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for name in components_to_compare:
                    if name in decomposition:
                        ax.plot(
                            price_data_period.index,
                            decomposition[name],
                            linewidth=2,
                            label=name.capitalize()
                        )
                
                ax.set_title("Component Comparison")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                st.pyplot(fig)
                
                # Provide interpretation
                st.subheader("Decomposition Interpretation")
                
                # Calculate variance explained by each component
                variance_original = np.var(decomposition['original'])
                variance_explained = {}
                
                for name, component in decomposition.items():
                    if name != 'original':
                        variance_explained[name] = np.var(component) / variance_original * 100
                
                # Sort components by variance explained
                sorted_variance = sorted(variance_explained.items(), key=lambda x: x[1], reverse=True)
                
                # Create variance explained chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                component_names = [item[0].capitalize() for item in sorted_variance]
                variances = [item[1] for item in sorted_variance]
                
                ax.bar(component_names, variances, color='skyblue', alpha=0.7)
                ax.set_title("Variance Explained by Each Component")
                ax.set_xlabel("Component")
                ax.set_ylabel("Variance Explained (%)")
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, value in enumerate(variances):
                    ax.text(i, value + 1, f"{value:.1f}%", ha='center')
                
                st.pyplot(fig)
                
                # Textual interpretation
                st.markdown(f"""
                **Key Insights from Decomposition:**
                
                - **Trend Component:** Explains {variance_explained.get('trend', 0):.1f}% of the total variance, representing the long-term price direction.
                - **Cyclical Components:** {"The cyclical components capture periodic patterns of different frequencies in the oil price movements." if any(k.startswith('cycle') for k in decomposition.keys()) else "No distinct cyclical components were identified in this time period."}
                - **Residual Component:** Represents {variance_explained.get('residual', 0):.1f}% of the variance, showing the random noise or unexplained movements.
                
                Decomposition can help improve forecasting by separately modeling each component and combining predictions.
                The Sentiment-Enhanced LSTM model leverages this approach by focusing more attention on important patterns.
                """)
                
                # Download decomposition data
                decomp_df = pd.DataFrame({
                    name: component
                    for name, component in decomposition.items()
                }, index=price_data_period.index)
                
                csv = decomp_df.to_csv()
                st.download_button(
                    label="Download Decomposition Data",
                    data=csv,
                    file_name=f"oil_decomposition_{oil_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    # Optional: Add custom CSS for better styling
    st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
div[data-testid="stVerticalBlock"] > div {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
div.stDataFrame > div {
    width: 100%;
}
div[data-testid="column"] {
    padding: 0.5rem;
}
.css-1r6slb0 {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
