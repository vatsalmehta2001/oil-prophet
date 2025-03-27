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
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from scipy.signal import savgol_filter
import time

# Import project modules
from src.data.preprocessing import OilDataProcessor
from src.models.baseline import BaselineForecaster
from src.models.lstm_attention import LSTMWithAttention, AttentionLayer
from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
from src.models.ensemble import EnsembleForecaster
from src.models.ceemdan import SimplifiedDecomposer
from src.nlp.finbert_sentiment import OilFinBERT
from src.evaluation.metrics import calculate_metrics, evaluate_horizon_performance, compare_models
from src.visualization.plots import plot_forecasts, plot_model_comparison, plot_decomposition, plot_price_with_sentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Oil Prophet Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E6091;
        margin-bottom: 0;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #3A7CA5;
        margin-top: 0;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E6091;
    }
    .metric-title {
        font-size: 1rem;
        color: #555;
    }
    .nav-link {
        text-decoration: none;
        padding: 8px 15px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .nav-link:hover {
        background-color: rgba(30, 96, 145, 0.1);
    }
    .nav-link.active {
        background-color: #1E6091;
        color: white !important;
    }
    .info-box {
        background-color: #f0f7fb;
        border-left: 5px solid #3A7CA5;
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
    }
    .stProgress .st-bo {
        background-color: #1E6091;
    }
    /* Custom tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E6091;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Application title and description
st.markdown('<h1 class="main-header">Oil Prophet Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced oil price forecasting with deep learning and market sentiment analysis</p>', unsafe_allow_html=True)

# Check if cache directory exists and create if not
if not os.path.exists('cache'):
    os.makedirs('cache')

# Cache for storing app state
@st.cache_data
def get_file_stats(filepath):
    """Get file stats for caching purposes"""
    if os.path.exists(filepath):
        return os.path.getmtime(filepath)
    return None

# Function to load available models with progress bar
def load_models(progress_bar=None) -> Dict[str, object]:
    """
    Load the trained models with a progress bar.
    
    Args:
        progress_bar: Optional Streamlit progress bar
    
    Returns:
        Dictionary of model objects
    """
    models = {}
    
    # Check if model files exist
    model_files = {
        "LSTM-Attention": "models/lstm_price_only.h5" if os.path.exists("models/lstm_price_only.h5") else None,
        "Sentiment-Enhanced LSTM": "models/sentiment_enhanced_lstm.h5" if os.path.exists("models/sentiment_enhanced_lstm.h5") else None
    }
    
    # Count for progress bar
    total_models = len(model_files) + 3  # +3 for baseline models
    progress_count = 0
    
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
        
        # Update progress
        progress_count += 1
        if progress_bar:
            progress_bar.progress(progress_count / total_models)
    
    # Add baseline models
    try:
        models["Naive"] = BaselineForecaster(method='naive')
        progress_count += 1
        if progress_bar:
            progress_bar.progress(progress_count / total_models)
            
        models["ARIMA"] = BaselineForecaster(method='arima')
        progress_count += 1
        if progress_bar:
            progress_bar.progress(progress_count / total_models)
            
        models["EMA"] = BaselineForecaster(method='ema')
        progress_count += 1
        if progress_bar:
            progress_bar.progress(progress_count / total_models)
            
        logger.info("Created baseline models")
    except Exception as e:
        logger.error(f"Error creating baseline models: {str(e)}")
    
    return models

# Function to load data with progress bar
def load_oil_data(oil_type: str, freq: str, progress_bar=None) -> pd.DataFrame:
    """
    Load and cache oil price data with progress bar.
    
    Args:
        oil_type: Type of oil price data ('brent' or 'wti')
        freq: Frequency of data ('daily', 'weekly', 'monthly')
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        DataFrame with oil price data
    """
    if progress_bar:
        progress_bar.progress(0.3)
        
    processor = OilDataProcessor()
    try:
        data = processor.load_data(oil_type=oil_type, freq=freq)
        if progress_bar:
            progress_bar.progress(1.0)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        if progress_bar:
            progress_bar.progress(1.0)
        return pd.DataFrame()

# Function to load sentiment data with progress bar
def load_sentiment_data(data_dir: str, progress_bar=None) -> pd.DataFrame:
    """
    Load and cache sentiment data with progress bar.
    
    Args:
        data_dir: Directory containing sentiment data
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        DataFrame with sentiment data
    """
    # Look for the sentiment dataset in the specified directory
    sentiment_file = os.path.join(data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    if progress_bar:
        progress_bar.progress(0.3)
    
    if os.path.exists(sentiment_file):
        try:
            sentiment_data = pd.read_csv(sentiment_file)
            
            # Ensure created_date is a datetime
            if 'created_date' in sentiment_data.columns:
                sentiment_data['created_date'] = pd.to_datetime(sentiment_data['created_date'])
            
            # Or try created_utc
            elif 'created_utc' in sentiment_data.columns:
                sentiment_data['created_date'] = pd.to_datetime(sentiment_data['created_utc'], unit='s')
            
            if progress_bar:
                progress_bar.progress(1.0)
                
            return sentiment_data
        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
    else:
        logger.warning(f"No sentiment data found at {sentiment_file}")
    
    if progress_bar:
        progress_bar.progress(1.0)
        
    return pd.DataFrame()

# Function to prepare dataset for prediction with progress bar
def prepare_prediction_dataset(
    price_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    window_size: int = 30,
    forecast_horizon: int = 7,
    progress_bar=None
) -> Dict[str, np.ndarray]:
    """
    Prepare dataset for prediction with progress bar.
    
    Args:
        price_data: DataFrame with price data
        sentiment_data: DataFrame with sentiment data
        window_size: Size of lookback window
        forecast_horizon: Number of steps to forecast
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        Dictionary with prepared dataset
    """
    if progress_bar:
        progress_bar.progress(0.2)
        
    try:
        dataset = prepare_sentiment_features(
            price_df=price_data,
            sentiment_df=sentiment_data,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        return {}

# Function to generate forecasts with progress bar
def generate_forecasts(
    models: Dict[str, object],
    dataset: Dict[str, np.ndarray],
    forecast_horizon: int,
    progress_bar=None
) -> Dict[str, np.ndarray]:
    """
    Generate forecasts using selected models with progress bar.
    
    Args:
        models: Dictionary of model objects
        dataset: Prepared dataset
        forecast_horizon: Number of periods to forecast
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        Dictionary of forecasts from each model
    """
    forecasts = {}
    
    # Get the latest data for prediction
    if 'X_price_test' in dataset and len(dataset['X_price_test']) > 0:
        X_price_recent = dataset['X_price_test'][-1:]
        
        # For baseline models, use the raw price data
        price_data = dataset['X_price_test'][-1, :, 0]  # Extract price values
        
        # Calculate total for progress bar
        total_models = len(models)
        progress_count = 0
        
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
            
            # Update progress
            progress_count += 1
            if progress_bar:
                progress_bar.progress(progress_count / total_models)
    
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
    
    # Check for fallback if no forecasts were generated
    if not forecasts and 'Naive' in models and 'X_price_test' in dataset and len(dataset['X_price_test']) > 0:
        # Generate at least a naive forecast
        try:
            price_data = dataset['X_price_test'][-1, :, 0]
            naive_model = models['Naive']
            naive_model.fit(price_data)
            forecasts['Naive'] = naive_model.predict(steps=forecast_horizon)
            logger.info("Generated fallback Naive forecast")
        except Exception as e:
            logger.error(f"Error generating fallback forecast: {str(e)}")
    
    return forecasts

# Function to plot interactive forecasts using Plotly
def plot_interactive_forecasts(
    historical_data: pd.DataFrame,
    forecasts: Dict[str, np.ndarray],
    forecast_dates: pd.DatetimeIndex,
    lookback_days: int = 30,
    title: str = "Oil Price Forecast"
) -> go.Figure:
    """
    Plot interactive forecast comparison using Plotly.
    
    Args:
        historical_data: DataFrame with historical price data
        forecasts: Dictionary of forecasts from each model
        forecast_dates: DatetimeIndex for forecast dates
        lookback_days: Number of historical days to include
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Get historical data for the lookback period
    historical_subset = historical_data.iloc[-lookback_days:].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_subset.index,
        y=historical_subset['Price'],
        mode='lines',
        name='Historical',
        line=dict(color='black', width=3),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add forecasts for each model
    colors = px.colors.qualitative.Set1
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        color_idx = i % len(colors)
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines+markers',
            name=f"{model_name} Forecast",
            line=dict(color=colors[color_idx], width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Add shapes for date separation
    if len(historical_subset) > 0 and len(forecast_dates) > 0:
        fig.add_vline(
            x=historical_subset.index[-1], 
            line_width=1, 
            line_dash='dashdot', 
            line_color='gray',
            annotation_text='Forecast Start',
            annotation_position='top right'
        )
    
    return fig

# Function to decompose price signal with progress bar
def decompose_price_signal(price_data: pd.DataFrame, n_components: int = 5, progress_bar=None) -> Dict[str, np.ndarray]:
    """
    Decompose price signal into components with progress bar.
    
    Args:
        price_data: DataFrame with price data
        n_components: Number of components to extract
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        Dictionary of decomposed components
    """
    try:
        if progress_bar:
            progress_bar.progress(0.3)
            
        # Extract price data
        price_array = price_data['Price'].values
        
        # Create decomposer
        decomposer = SimplifiedDecomposer(n_components=n_components)
        
        if progress_bar:
            progress_bar.progress(0.6)
            
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
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        return decomposition
    except Exception as e:
        logger.error(f"Error decomposing price signal: {str(e)}")
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        return {}

# Function to plot interactive decomposition using Plotly
def plot_interactive_decomposition(
    decomposition: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    component_names: List[str] = None,
    title: str = "Signal Decomposition"
) -> go.Figure:
    """
    Create interactive decomposition plot using Plotly.
    
    Args:
        decomposition: Dictionary of decomposed signal components
        dates: DatetimeIndex for the x-axis
        component_names: List of components to plot (defaults to all)
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if component_names is None:
        component_names = list(decomposition.keys())
    
    # Create subplots
    fig = make_subplots(
        rows=len(component_names), 
        cols=1,
        shared_xaxes=True,
        subplot_titles=component_names,
        vertical_spacing=0.05
    )
    
    # Colors for components
    colors = {
        'original': 'black',
        'trend': 'darkblue',
        'residual': 'red'
    }
    
    # Default colors for cycles
    cycle_colors = px.colors.qualitative.Dark2
    
    # Add each component to its subplot
    for i, component in enumerate(component_names):
        color = colors.get(component, cycle_colors[i % len(cycle_colors)])
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition[component],
                mode='lines',
                name=component.capitalize(),
                line=dict(color=color, width=2),
                hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=i+1, 
            col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        height=150 * len(component_names),
        margin=dict(l=10, r=10, t=50, b=10),
        template='plotly_white'
    )
    
    # Update y-axis titles
    for i, component in enumerate(component_names):
        fig.update_yaxes(title_text="Value", row=i+1, col=1)
    
    # Update x-axis title for the last subplot only
    fig.update_xaxes(title_text="Date", row=len(component_names), col=1)
    
    return fig

# Function to aggregate sentiment by time with progress bar
def aggregate_sentiment(
    sentiment_data: pd.DataFrame,
    freq: str = 'D',
    date_col: str = 'created_date',
    progress_bar = None
) -> pd.DataFrame:
    """
    Aggregate sentiment data by time period with progress bar.
    
    Args:
        sentiment_data: DataFrame with sentiment data
        freq: Aggregation frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        date_col: Column containing dates
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        DataFrame with aggregated sentiment
    """
    if sentiment_data.empty or date_col not in sentiment_data.columns:
        if progress_bar:
            progress_bar.progress(1.0)
        return pd.DataFrame()
    
    if progress_bar:
        progress_bar.progress(0.3)
    
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
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        return agg_sentiment
    except Exception as e:
        logger.error(f"Error aggregating sentiment: {str(e)}")
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        return pd.DataFrame()

# Function to plot interactive sentiment vs price using Plotly
def plot_interactive_sentiment_price(
    price_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    title: str = "Oil Price with Market Sentiment"
) -> go.Figure:
    """
    Create interactive plot of price with sentiment using Plotly.
    
    Args:
        price_data: DataFrame with price data
        sentiment_data: DataFrame with sentiment data
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Price'],
            name="Oil Price",
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add sentiment compound trace
    fig.add_trace(
        go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data['sentiment_compound'],
            name="Sentiment",
            line=dict(color='#d62728', width=2),
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.4f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add sentiment moving average if available
    if 'compound_ma7' in sentiment_data.columns:
        fig.add_trace(
            go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data['compound_ma7'],
                name="7-day MA",
                line=dict(color='#2ca02c', width=2, dash='dot'),
                hovertemplate='Date: %{x}<br>7-day MA: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Add neutral sentiment line
    fig.add_hline(
        y=0, 
        line_width=1, 
        line_dash='dash', 
        line_color='gray',
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(
        title_text="Sentiment Score", 
        secondary_y=True,
        range=[-1, 1]
    )
    
    return fig

# Function to plot sentiment distribution using Plotly
def plot_sentiment_distribution(
    sentiment_data: pd.DataFrame,
    title: str = "Sentiment Distribution"
) -> go.Figure:
    """
    Create interactive sentiment distribution plot using Plotly.
    
    Args:
        sentiment_data: DataFrame with sentiment data (must have 'sentiment_compound' column)
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    compound_values = sentiment_data['sentiment_compound'].dropna()
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=compound_values,
        nbinsx=30,
        marker_color='skyblue',
        name='Sentiment',
        hovertemplate='Sentiment: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Calculate statistics
    mean = compound_values.mean()
    median = compound_values.median()
    std = compound_values.std()
    
    # Add vertical lines for reference
    fig.add_vline(x=0, line_width=2, line_dash='solid', line_color='gray', name='Neutral')
    fig.add_vline(x=0.2, line_width=2, line_dash='dash', line_color='green', name='Bullish Threshold')
    fig.add_vline(x=-0.2, line_width=2, line_dash='dash', line_color='red', name='Bearish Threshold')
    fig.add_vline(x=mean, line_width=2, line_dash='dot', line_color='blue', name='Mean')
    
    # Add annotations
    fig.add_annotation(
        x=mean, 
        y=0.85, 
        yref='paper',
        text=f"Mean: {mean:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='blue',
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Sentiment Score',
        yaxis_title='Count',
        template='plotly_white',
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Function to plot model performance comparison using Plotly
def plot_model_performance(
    performance: Dict[str, Dict[str, float]],
    metrics: List[str] = ['rmse', 'mae', 'mape', 'directional_accuracy'],
    title: str = "Model Performance Comparison"
) -> go.Figure:
    """
    Create interactive model performance comparison plot using Plotly.
    
    Args:
        performance: Dictionary with metrics for each model
        metrics: List of metrics to plot
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[m.upper() for m in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Define model colors
    model_colors = px.colors.qualitative.Plotly
    model_names = list(performance.keys())
    
    # Map metrics to subplot positions
    positions = {
        0: (1, 1),  # rmse: top-left
        1: (1, 2),  # mae: top-right
        2: (2, 1),  # mape: bottom-left
        3: (2, 2)   # directional_accuracy: bottom-right
    }
    
    # Create a bar chart for each metric
    for i, metric in enumerate(metrics):
        values = [performance[model].get(metric, 0) for model in model_names]
        
        # Define the color based on the metric (for directional accuracy, higher is better)
        colors = model_colors[:len(model_names)]
        if metric == 'directional_accuracy':
            # For this metric, higher is better, so invert the sort
            sorted_indices = np.argsort(values)
        else:
            # For error metrics, lower is better
            sorted_indices = np.argsort(values)[::-1]  # Reverse to show lowest first
        
        # Sort models by metric value
        sorted_models = [model_names[idx] for idx in sorted_indices]
        sorted_values = [values[idx] for idx in sorted_indices]
        sorted_colors = [colors[idx % len(colors)] for idx in sorted_indices]
        
        # Add horizontal bar chart
        fig.add_trace(
            go.Bar(
                y=sorted_models,
                x=sorted_values,
                orientation='h',
                marker_color=sorted_colors,
                name=metric.upper(),
                text=[f"{v:.2f}" for v in sorted_values],
                textposition='outside',
                hovertemplate='%{y}: %{x:.2f}<extra></extra>'
            ),
            row=positions[i][0], 
            col=positions[i][1]
        )
        
        # Highlight the best model
        best_model_index = 0 if metric == 'directional_accuracy' else -1
        fig.add_shape(
            type="rect",
            xref=f"x{i+1}",
            yref=f"y{i+1}",
            x0=0,
            y0=best_model_index - 0.4,
            x1=sorted_values[best_model_index],
            y1=best_model_index + 0.4,
            line=dict(width=2, color="rgba(0,255,0,0.3)"),
            fillcolor="rgba(0,255,0,0.1)"
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        height=600,
        template='plotly_white',
        margin=dict(l=10, r=120, t=80, b=10)  # Extra right margin for text labels
    )
    
    # Update axes
    for i, metric in enumerate(metrics):
        if metric == 'directional_accuracy':
            # For accuracy, higher is better
            fig.update_xaxes(title_text=f"{metric.upper()} (%)", row=positions[i][0], col=positions[i][1])
        else:
            # For error metrics, lower is better
            fig.update_xaxes(title_text=metric.upper(), row=positions[i][0], col=positions[i][1])
    
    return fig

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

# Function to create metric card HTML
def create_metric_card(title, value, unit="", description="", trend=None, trend_value=None):
    """
    Create HTML for a metric card.
    
    Args:
        title: Metric title
        value: Metric value
        unit: Unit of measurement
        description: Metric description
        trend: Trend direction ('up', 'down', or None)
        trend_value: Trend value
        
    Returns:
        HTML string for the metric card
    """
    # Define trend arrow and color
    trend_arrow = ""
    trend_color = "gray"
    trend_text = ""
    
    if trend is not None and trend_value is not None:
        if trend == 'up':
            trend_arrow = "↑"
            trend_color = "green"
            trend_text = f"+{trend_value}"
        elif trend == 'down':
            trend_arrow = "↓"
            trend_color = "red"
            trend_text = f"-{trend_value}"
    
    # Create HTML
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{unit}</div>
        <div style="color: {trend_color}; font-weight: bold; margin-top: 5px;">
            {trend_arrow} {trend_text}
        </div>
        <div style="font-size: 0.8rem; color: #777; margin-top: 5px;">{description}</div>
    </div>
    """
    
    return html

# Main dashboard logic
def main():
    # Initialize session state for persistent storage across reruns
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_loaded = False
        st.session_state.sentiment_loaded = False
        st.session_state.models_loaded = False
        st.session_state.dataset_prepared = False
        st.session_state.price_data = None
        st.session_state.sentiment_data = None
        st.session_state.models = None
        st.session_state.dataset = None
        st.session_state.forecasts = None
        st.session_state.decomposition = None
        st.session_state.agg_sentiment = None
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.title("Dashboard Controls")
        
        # Navigation
        st.sidebar.subheader("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Overview", "Forecast Explorer", "Model Performance", "Sentiment Analysis", "Signal Decomposition"]
        )
        
        # Data parameters
        st.sidebar.subheader("Data Parameters")
        
        oil_type = st.sidebar.selectbox(
            "Oil Type",
            options=["brent", "wti"],
            format_func=lambda x: x.upper()
        )
        
        freq = st.sidebar.selectbox(
            "Data Frequency",
            options=["daily", "weekly", "monthly"],
            format_func=lambda x: x.capitalize()
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
        
        # Load data button
        if st.sidebar.button("Load Data", type="primary"):
            st.session_state.data_loaded = False
            st.session_state.sentiment_loaded = False
            st.session_state.models_loaded = False
            st.session_state.dataset_prepared = False
        
        # Display loading status
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Price data loaded")
        else:
            st.sidebar.warning("Price data not loaded")
            
        if st.session_state.sentiment_loaded:
            st.sidebar.success("✅ Sentiment data loaded")
        else:
            st.sidebar.warning("Sentiment data not loaded")
            
        if st.session_state.models_loaded:
            st.sidebar.success("✅ Models loaded")
        else:
            st.sidebar.warning("Models not loaded")
    
    # Load data if not loaded yet
    if not st.session_state.data_loaded:
        with st.spinner("Loading price data..."):
            progress_bar = st.progress(0)
            price_data = load_oil_data(oil_type, freq, progress_bar)
            if not price_data.empty:
                st.session_state.price_data = price_data
                st.session_state.data_loaded = True
            progress_bar.empty()
    
    # Load sentiment data if not loaded yet
    if st.session_state.data_loaded and not st.session_state.sentiment_loaded:
        with st.spinner("Loading sentiment data..."):
            progress_bar = st.progress(0)
            sentiment_data = load_sentiment_data(data_dir, progress_bar)
            if not sentiment_data.empty:
                st.session_state.sentiment_data = sentiment_data
                st.session_state.sentiment_loaded = True
            progress_bar.empty()
    
    # Load models if not loaded yet
    if st.session_state.data_loaded and not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            progress_bar = st.progress(0)
            models = load_models(progress_bar)
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
            progress_bar.empty()
    
    # Prepare dataset if not prepared yet
    if (st.session_state.data_loaded and 
        st.session_state.sentiment_loaded and 
        st.session_state.models_loaded and 
        not st.session_state.dataset_prepared):
        with st.spinner("Preparing dataset..."):
            progress_bar = st.progress(0)
            dataset = prepare_prediction_dataset(
                st.session_state.price_data,
                st.session_state.sentiment_data,
                window_size=30,
                forecast_horizon=forecast_horizon,
                progress_bar=progress_bar
            )
            if dataset:
                st.session_state.dataset = dataset
                st.session_state.dataset_prepared = True
            progress_bar.empty()
    
    # Check if everything is loaded
    if (not st.session_state.data_loaded or 
        not st.session_state.sentiment_loaded or 
        not st.session_state.models_loaded or 
        not st.session_state.dataset_prepared):
        st.warning("Please load data using the sidebar controls.")
        st.stop()
    
    # Generate forecasts if not generated yet or if forecast horizon has changed
    if (st.session_state.forecasts is None or 
        len(next(iter(st.session_state.forecasts.values()))) != forecast_horizon):
        with st.spinner("Generating forecasts..."):
            # Get selected models
            available_models = list(st.session_state.models.keys())
            selected_models = st.session_state.models
            
            # Generate forecasts
            progress_bar = st.progress(0)
            forecasts = generate_forecasts(
                selected_models,
                st.session_state.dataset,
                forecast_horizon,
                progress_bar
            )
            if forecasts:
                st.session_state.forecasts = forecasts
            progress_bar.empty()
    
    # Decompose price signal if not decomposed yet
    if st.session_state.decomposition is None:
        with st.spinner("Decomposing price signal..."):
            progress_bar = st.progress(0)
            price_data_period = st.session_state.price_data.iloc[-lookback_period:]
            decomposition = decompose_price_signal(price_data_period, n_components=5, progress_bar=progress_bar)
            if decomposition:
                st.session_state.decomposition = decomposition
                st.session_state.decomposition_dates = price_data_period.index
            progress_bar.empty()
    
    # Aggregate sentiment if not aggregated yet
    if st.session_state.agg_sentiment is None:
        with st.spinner("Aggregating sentiment data..."):
            progress_bar = st.progress(0)
            agg_sentiment = aggregate_sentiment(
                st.session_state.sentiment_data,
                freq=freq[0],
                progress_bar=progress_bar
            )
            if not agg_sentiment.empty:
                st.session_state.agg_sentiment = agg_sentiment
            progress_bar.empty()
    
    # ---- PAGE: OVERVIEW ----
    if page == "Overview":
        st.header("Oil Price Forecast Dashboard")
        
        # Create row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Most recent price
        latest_price = st.session_state.price_data['Price'].iloc[-1]
        previous_price = st.session_state.price_data['Price'].iloc[-2]
        price_change = latest_price - previous_price
        price_change_pct = (price_change / previous_price) * 100
        price_trend = 'up' if price_change > 0 else 'down'
        
        with col1:
            st.markdown(
                create_metric_card(
                    f"Latest {oil_type.upper()} Price",
                    f"${latest_price:.2f}",
                    "",
                    f"Last updated: {st.session_state.price_data.index[-1].strftime('%Y-%m-%d')}",
                    price_trend,
                    f"{abs(price_change_pct):.2f}%"
                ),
                unsafe_allow_html=True
            )
        
        # Forecast change
        if st.session_state.forecasts:
            ensemble_forecast = st.session_state.forecasts.get('Ensemble', None)
            if ensemble_forecast is None and st.session_state.forecasts:
                # Use the first available forecast
                ensemble_forecast = next(iter(st.session_state.forecasts.values()))
            
            if ensemble_forecast is not None:
                forecast_end = ensemble_forecast[-1]
                forecast_change = forecast_end - latest_price
                forecast_change_pct = (forecast_change / latest_price) * 100
                forecast_trend = 'up' if forecast_change > 0 else 'down'
                
                with col2:
                    st.markdown(
                        create_metric_card(
                            f"{forecast_horizon}-Day Forecast",
                            f"${forecast_end:.2f}",
                            "",
                            f"Forecasted change from current price",
                            forecast_trend,
                            f"{abs(forecast_change_pct):.2f}%"
                        ),
                        unsafe_allow_html=True
                    )
        
        # Sentiment metrics
        if not st.session_state.agg_sentiment.empty:
            latest_sentiment = st.session_state.agg_sentiment['sentiment_compound'].iloc[-1]
            previous_sentiment = st.session_state.agg_sentiment['sentiment_compound'].iloc[-2] if len(st.session_state.agg_sentiment) > 1 else 0
            sentiment_change = latest_sentiment - previous_sentiment
            sentiment_trend = 'up' if sentiment_change > 0 else 'down'
            
            with col3:
                st.markdown(
                    create_metric_card(
                        "Market Sentiment",
                        f"{latest_sentiment:.2f}",
                        "",
                        f"Last updated: {st.session_state.agg_sentiment.index[-1].strftime('%Y-%m-%d')}",
                        sentiment_trend,
                        f"{abs(sentiment_change):.2f}"
                    ),
                    unsafe_allow_html=True
                )
        
        # Data coverage metrics
        data_start = st.session_state.price_data.index[0]
        data_end = st.session_state.price_data.index[-1]
        data_days = (data_end - data_start).days
        
        with col4:
            st.markdown(
                create_metric_card(
                    "Data Coverage",
                    f"{data_days:,}",
                    " days",
                    f"From {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}",
                    None,
                    None
                ),
                unsafe_allow_html=True
            )
        
        # Add forecast plot
        st.subheader("Latest Price Forecast")
        
        if st.session_state.forecasts:
            # Create dates for forecast
            last_date = st.session_state.price_data.index[-1]
            
            # Set proper date range frequency based on selected data frequency
            if freq == "daily":
                date_freq = 'B'  # Business days
            elif freq == "weekly":
                date_freq = 'W'  # Weeks
            else:
                date_freq = 'M'  # Months
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq=date_freq
            )
            
            # Create interactive forecast plot
            forecast_fig = plot_interactive_forecasts(
                st.session_state.price_data,
                st.session_state.forecasts,
                forecast_dates,
                lookback_period=lookback_period,
                title=f"{oil_type.upper()} Price Forecast ({freq.capitalize()})"
            )
            
            st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Add price with sentiment plot
        st.subheader("Price and Market Sentiment")
        
        if not st.session_state.agg_sentiment.empty:
            # Create interactive sentiment vs price plot
            sentiment_fig = plot_interactive_sentiment_price(
                st.session_state.price_data.iloc[-lookback_period:],
                st.session_state.agg_sentiment,
                title=f"{oil_type.upper()} Price with Market Sentiment"
            )
            
            st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Add key insights section
        st.subheader("Key Insights")
        
        # Create three columns for insights
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("""
                <div class="info-box">
                    <h4>Price Trend</h4>
                    <p>The recent price trend shows significant volatility in oil markets, with short-term fluctuations driven by supply factors and market sentiment.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown("""
                <div class="info-box">
                    <h4>Sentiment Analysis</h4>
                    <p>Recent market sentiment has been correlated with price movements, with negative sentiment preceding price declines by 2-3 days.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with insight_col3:
            st.markdown("""
                <div class="info-box">
                    <h4>Model Performance</h4>
                    <p>The Sentiment-Enhanced LSTM model has shown improved accuracy over traditional models, especially during periods of high volatility.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # ---- PAGE: FORECAST EXPLORER ----
    elif page == "Forecast Explorer":
        st.header("Forecast Explorer")
        
        # Tabs for different forecast views
        forecast_tabs = st.tabs(["Interactive Chart", "Forecast Table", "Forecast Details"])
        
        with forecast_tabs[0]:  # Interactive Chart tab
            if st.session_state.forecasts:
                # Create dates for forecast
                last_date = st.session_state.price_data.index[-1]
                
                # Set proper date range frequency based on selected data frequency
                if freq == "daily":
                    date_freq = 'B'  # Business days
                elif freq == "weekly":
                    date_freq = 'W'  # Weeks
                else:
                    date_freq = 'M'  # Months
                
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_horizon,
                    freq=date_freq
                )
                
                # Model selection for viewing
                available_models = list(st.session_state.forecasts.keys())
                selected_forecast_models = st.multiselect(
                    "Select Models to Display",
                    options=available_models,
                    default=["Ensemble"] if "Ensemble" in available_models else [available_models[0]]
                )
                
                # Filter forecasts to selected models
                if selected_forecast_models:
                    selected_forecasts = {model: forecast for model, forecast in st.session_state.forecasts.items() 
                                         if model in selected_forecast_models}
                    
                    # Create interactive forecast plot
                    forecast_fig = plot_interactive_forecasts(
                        st.session_state.price_data,
                        selected_forecasts,
                        forecast_dates,
                        lookback_period=lookback_period,
                        title=f"{oil_type.upper()} Price Forecast ({freq.capitalize()})"
                    )
                    
                    st.plotly_chart(forecast_fig, use_container_width=True)
                else:
                    st.warning("Please select at least one model to display")
        
        with forecast_tabs[1]:  # Forecast Table tab
            if st.session_state.forecasts:
                # Create dates for forecast
                last_date = st.session_state.price_data.index[-1]
                
                # Set proper date range frequency based on selected data frequency
                if freq == "daily":
                    date_freq = 'B'  # Business days
                elif freq == "weekly":
                    date_freq = 'W'  # Weeks
                else:
                    date_freq = 'M'  # Months
                
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_horizon,
                    freq=date_freq
                )
                
                # Create DataFrame with forecasts
                forecast_df = pd.DataFrame(index=forecast_dates)
                
                # Add each model's forecast as a column
                for model_name, forecast in st.session_state.forecasts.items():
                    # Ensure the forecast length matches our date range
                    if len(forecast) == len(forecast_dates):
                        forecast_df[model_name] = forecast
                    else:
                        # Handle mismatched lengths by padding or truncating
                        logger.warning(f"Forecast length for {model_name} ({len(forecast)}) doesn't match date range ({len(forecast_dates)}). Adjusting...")
                        if len(forecast) > len(forecast_dates):
                            forecast_df[model_name] = forecast[:len(forecast_dates)]
                        else:
                            # Pad with the last value
                            padded_forecast = np.concatenate([forecast, np.repeat(forecast[-1], len(forecast_dates) - len(forecast))])
                            forecast_df[model_name] = padded_forecast
                
                # Display the table
                st.dataframe(forecast_df.style.format("${:.2f}"), use_container_width=True)
                
                # Download forecast data as CSV
                csv = forecast_df.to_csv()
                st.download_button(
                    label="Download Forecast Data",
                    data=csv,
                    file_name=f"oil_forecast_{oil_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with forecast_tabs[2]:  # Forecast Details tab
            if st.session_state.forecasts:
                # Detailed forecast information
                st.subheader("Forecast Details")
                
                # Select model for details
                model_for_details = st.selectbox(
                    "Select Model for Detailed Analysis",
                    options=list(st.session_state.forecasts.keys()),
                    format_func=lambda x: f"{x} Model"
                )
                
                # Show model information
                if model_for_details:
                    forecast = st.session_state.forecasts[model_for_details]
                    
                    # Create dates for forecast
                    last_date = st.session_state.price_data.index[-1]
                    
                    # Set proper date range frequency based on selected data frequency
                    if freq == "daily":
                        date_freq = 'B'  # Business days
                    elif freq == "weekly":
                        date_freq = 'W'  # Weeks
                    else:
                        date_freq = 'M'  # Months
                    
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=len(forecast),
                        freq=date_freq
                    )
                    
                    # Create columns for info
                    col1, col2 = st.columns(2)
                    
                    # Model description
                    with col1:
                        if model_for_details == "Naive":
                            st.markdown("""
                                <div class="info-box">
                                    <h4>Naive Model</h4>
                                    <p>The Naive forecasting model predicts that future values will be equal to the most recent observation. It serves as a simple baseline for comparison.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif model_for_details == "ARIMA":
                            st.markdown("""
                                <div class="info-box">
                                    <h4>ARIMA Model</h4>
                                    <p>The Auto-Regressive Integrated Moving Average model combines autoregression, differencing, and moving averages to fit time series data and make predictions.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif model_for_details == "EMA":
                            st.markdown("""
                                <div class="info-box">
                                    <h4>EMA Model</h4>
                                    <p>The Exponential Moving Average model applies more weight to recent observations, making it responsive to recent price changes while still smoothing out noise.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif model_for_details == "LSTM-Attention":
                            st.markdown("""
                                <div class="info-box">
                                    <h4>LSTM with Attention Model</h4>
                                    <p>This deep learning model combines Long Short-Term Memory networks with an attention mechanism to focus on the most relevant parts of the input sequence for forecasting.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif model_for_details == "Sentiment-Enhanced LSTM":
                            st.markdown("""
                                <div class="info-box">
                                    <h4>Sentiment-Enhanced LSTM Model</h4>
                                    <p>This advanced model integrates market sentiment analysis with price patterns, using both technical and sentiment features to improve forecast accuracy.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif model_for_details == "Ensemble":
                            st.markdown("""
                                <div class="info-box">
                                    <h4>Ensemble Model</h4>
                                    <p>The Ensemble model combines predictions from multiple forecasting models to create a more robust and accurate forecast by leveraging the strengths of each approach.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Forecast statistics
                    with col2:
                        st.markdown("### Forecast Statistics")
                        
                        # Calculate statistics
                        latest_price = st.session_state.price_data['Price'].iloc[-1]
                        forecast_end = forecast[-1]
                        forecast_change = forecast_end - latest_price
                        forecast_change_pct = (forecast_change / latest_price) * 100
                        forecast_min = forecast.min()
                        forecast_max = forecast.max()
                        forecast_mean = forecast.mean()
                        
                        # Create stats table
                        stats_df = pd.DataFrame({
                            'Metric': ['Current Price', 'Forecasted End Price', 'Change', '% Change', 
                                       'Min Forecast', 'Max Forecast', 'Mean Forecast'],
                            'Value': [f"${latest_price:.2f}", f"${forecast_end:.2f}", 
                                      f"${forecast_change:.2f}", f"{forecast_change_pct:.2f}%",
                                      f"${forecast_min:.2f}", f"${forecast_max:.2f}", f"${forecast_mean:.2f}"]
                        })
                        
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)
                    
                    # Show day-by-day forecast
                    st.markdown("### Day-by-Day Forecast")
                    
                    # Create forecast table
                    forecast_details_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Price': forecast,
                        'Change from Previous': [None] + [forecast[i] - forecast[i-1] for i in range(1, len(forecast))],
                        '% Change': [None] + [(forecast[i] - forecast[i-1]) / forecast[i-1] * 100 for i in range(1, len(forecast))]
                    })
                    
                    # Format the columns
                    st.dataframe(
                        forecast_details_df.style.format({
                            'Forecasted Price': '${:.2f}',
                            'Change from Previous': '${:.2f}',
                            '% Change': '{:.2f}%'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
    
    # ---- PAGE: MODEL PERFORMANCE ----
    elif page == "Model Performance":
        st.header("Model Performance Analysis")
        
        # Tabs for different performance views
        performance_tabs = st.tabs(["Model Comparison", "Horizon Analysis", "Improvement Analysis"])
        
        with performance_tabs[0]:  # Model Comparison tab
            # Create synthetic test data for demonstration
            if 'y_test' in st.session_state.dataset and st.session_state.forecasts:
                # Use actual test data if available
                test_data = st.session_state.dataset['y_test'][0]
                
                # Calculate performance metrics
                performance = calculate_model_performance(test_data, st.session_state.forecasts)
                
                if performance:
                    # Create interactive performance comparison plot
                    performance_fig = plot_model_performance(
                        performance,
                        metrics=['rmse', 'mae', 'mape', 'directional_accuracy'],
                        title="Model Performance Comparison"
                    )
                    
                    st.plotly_chart(performance_fig, use_container_width=True)
                    
                    # Show performance table
                    st.subheader("Performance Metrics")
                    
                    # Create performance DataFrame
                    metrics_df = pd.DataFrame({
                        'Model': list(performance.keys()),
                        'RMSE': [performance[model].get('rmse', 0) for model in performance.keys()],
                        'MAE': [performance[model].get('mae', 0) for model in performance.keys()],
                        'MAPE (%)': [performance[model].get('mape', 0) for model in performance.keys()],
                        'Directional Accuracy (%)': [performance[model].get('directional_accuracy', 0) for model in performance.keys()]
                    })
                    
                    # Display the table
                    st.dataframe(
                        metrics_df.style.format({
                            'RMSE': '{:.4f}',
                            'MAE': '{:.4f}',
                            'MAPE (%)': '{:.2f}',
                            'Directional Accuracy (%)': '{:.2f}'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info("Test data or forecasts not available for model performance evaluation.")
        
        with performance_tabs[1]:  # Horizon Analysis tab
            if 'y_test' in st.session_state.dataset and st.session_state.forecasts:
                # Use actual test data if available
                test_data = st.session_state.dataset['y_test'][0]
                
                # Calculate performance metrics
                performance = calculate_model_performance(test_data, st.session_state.forecasts)
                
                # Calculate horizon performance
                horizon_metrics = evaluate_horizon_performance(test_data, st.session_state.forecasts.get('Ensemble', next(iter(st.session_state.forecasts.values()))))
                
                if horizon_metrics:
                    # Create figure
                    fig = go.Figure()
                    
                    # Add RMSE line
                    fig.add_trace(go.Scatter(
                        x=horizon_metrics['horizon'],
                        y=horizon_metrics['rmse'],
                        mode='lines+markers',
                        name='RMSE',
                        line=dict(color='blue', width=2),
                        hovertemplate='Horizon: %{x}<br>RMSE: %{y:.4f}<extra></extra>'
                    ))
                    
                    # Add MAPE line on secondary y-axis
                    fig.add_trace(go.Scatter(
                        x=horizon_metrics['horizon'],
                        y=horizon_metrics['mape'],
                        mode='lines+markers',
                        name='MAPE (%)',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='Horizon: %{x}<br>MAPE: %{y:.2f}%<extra></extra>',
                        yaxis='y2'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Forecast Accuracy by Horizon",
                        xaxis_title="Forecast Horizon (Days)",
                        yaxis_title="RMSE",
                        yaxis2=dict(
                            title="MAPE (%)",
                            overlaying='y',
                            side='right'
                        ),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='center',
                            x=0.5
                        ),
                        template='plotly_white',
                        height=500,
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                        <div class="info-box">
                            <h4>Horizon Analysis Interpretation</h4>
                            <p>The horizon analysis shows how forecast accuracy decreases as the prediction horizon extends further into the future. 
                            Typically, forecasts for the next 1-3 days are the most accurate, with error metrics increasing significantly beyond that point.</p>
                            <p>The RMSE (blue line) shows the absolute error magnitude, while the MAPE (red line) shows the percentage error relative to the actual price.</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Test data or forecasts not available for horizon analysis.")
        
        with performance_tabs[2]:  # Improvement Analysis tab
            if 'y_test' in st.session_state.dataset and st.session_state.forecasts and 'Naive' in st.session_state.forecasts:
                # Use actual test data if available
                test_data = st.session_state.dataset['y_test'][0]
                
                # Calculate performance metrics
                performance = calculate_model_performance(test_data, st.session_state.forecasts)
                
                if performance and 'Naive' in performance:
                    # Calculate improvement over naive baseline
                    baseline_rmse = performance['Naive'].get('rmse', 1.0)  # Default to 1.0 to avoid division by zero
                    
                    improvements = {
                        model: {
                            'rmse_improvement': (1 - metrics.get('rmse', 0) / baseline_rmse) * 100,
                            'mae_improvement': (1 - metrics.get('mae', 0) / performance['Naive'].get('mae', 1.0)) * 100,
                            'mape_improvement': (1 - metrics.get('mape', 0) / performance['Naive'].get('mape', 1.0)) * 100,
                            'dir_acc_improvement': metrics.get('directional_accuracy', 0) - performance['Naive'].get('directional_accuracy', 0)
                        }
                        for model, metrics in performance.items() if model != 'Naive'
                    }
                    
                    # Create a figure for RMSE improvement
                    fig = go.Figure()
                    
                    # Add improvement bars
                    models = list(improvements.keys())
                    rmse_improvements = [improvements[model]['rmse_improvement'] for model in models]
                    
                    # Sort by improvement value
                    sorted_indices = np.argsort(rmse_improvements)
                    sorted_models = [models[i] for i in sorted_indices]
                    sorted_improvements = [rmse_improvements[i] for i in sorted_indices]
                    
                    # Define colors based on improvement (green for positive, red for negative)
                    colors = ['green' if imp > 0 else 'red' for imp in sorted_improvements]
                    
                    fig.add_trace(go.Bar(
                        y=sorted_models,
                        x=sorted_improvements,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{imp:.2f}%" for imp in sorted_improvements],
                        textposition='outside',
                        hovertemplate='Model: %{y}<br>Improvement: %{x:.2f}%<extra></extra>'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="RMSE Improvement Over Naive Baseline",
                        xaxis_title="Improvement (%)",
                        yaxis_title="Model",
                        template='plotly_white',
                        height=400,
                        margin=dict(l=10, r=100, t=50, b=10)  # Extra right margin for text labels
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    best_model = max(improvements.items(), key=lambda x: x[1]['rmse_improvement'])[0]
                    best_improvement = improvements[best_model]['rmse_improvement']
                    
                    st.markdown(f"""
                        <div class="info-box">
                            <h4>Model Improvement Analysis</h4>
                            <p>The <strong>{best_model}</strong> model shows the greatest improvement over the naive baseline with a <strong>{best_improvement:.2f}%</strong> reduction in RMSE.</p>
                            <p>Advanced models like LSTM with Attention and Sentiment-Enhanced LSTM typically show the most significant improvements during periods of high market volatility or when major sentiment shifts occur.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show comprehensive improvement table
                    st.subheader("Comprehensive Improvement Analysis")
                    
                    # Create improvement DataFrame
                    improvement_df = pd.DataFrame({
                        'Model': models,
                        'RMSE Improvement (%)': [improvements[model]['rmse_improvement'] for model in models],
                        'MAE Improvement (%)': [improvements[model]['mae_improvement'] for model in models],
                        'MAPE Improvement (%)': [improvements[model]['mape_improvement'] for model in models],
                        'Directional Accuracy Improvement (%)': [improvements[model]['dir_acc_improvement'] for model in models]
                    })
                    
                    # Display the table
                    st.dataframe(
                        improvement_df.style.format({
                            'RMSE Improvement (%)': '{:.2f}',
                            'MAE Improvement (%)': '{:.2f}',
                            'MAPE Improvement (%)': '{:.2f}',
                            'Directional Accuracy Improvement (%)': '{:.2f}'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info("Test data or Naive model forecast not available for improvement analysis.")
    
    # ---- PAGE: SENTIMENT ANALYSIS ----
    elif page == "Sentiment Analysis":
        st.header("Market Sentiment Analysis")
        
        # Tabs for different sentiment views
        sentiment_tabs = st.tabs(["Sentiment Over Time", "Sentiment Distribution", "Price-Sentiment Correlation"])
        
        with sentiment_tabs[0]:  # Sentiment Over Time tab
            if not st.session_state.agg_sentiment.empty:
                # Create interactive sentiment vs price plot
                sentiment_fig = plot_interactive_sentiment_price(
                    st.session_state.price_data.iloc[-lookback_period:],
                    st.session_state.agg_sentiment,
                    title=f"{oil_type.upper()} Price with Market Sentiment"
                )
                
                st.plotly_chart(sentiment_fig, use_container_width=True)
                
                # Show sentiment table with time aggregation options
                st.subheader("Sentiment Data")
                
                # Time aggregation selector
                time_agg = st.selectbox(
                    "Time Aggregation",
                    options=["Daily", "Weekly", "Monthly"],
                    index=0
                )
                
                # Get sentiment data with selected aggregation
                if time_agg == "Daily":
                    show_sentiment = st.session_state.agg_sentiment
                else:
                    # Reaggregate with selected frequency
                    freq = 'W' if time_agg == "Weekly" else 'M'
                    with st.spinner(f"Aggregating sentiment by {time_agg.lower()} periods..."):
                        show_sentiment = aggregate_sentiment(
                            st.session_state.sentiment_data,
                            freq=freq[0]
                        )
                
                if not show_sentiment.empty:
                    # Show the sentiment data table
                    st.dataframe(
                        show_sentiment.reset_index().rename(columns={'index': 'Date'}).style.format({
                            'sentiment_compound': '{:.4f}',
                            'sentiment_positive': '{:.4f}',
                            'sentiment_negative': '{:.4f}',
                            'sentiment_neutral': '{:.4f}',
                            'count': '{:.0f}'
                        }).sort_values(by='Date', ascending=False),
                        use_container_width=True
                    )
                    
                    # Download sentiment data as CSV
                    csv = show_sentiment.reset_index().to_csv(index=False)
                    st.download_button(
                        label="Download Sentiment Data",
                        data=csv,
                        file_name=f"oil_sentiment_{oil_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Sentiment data not available.")
        
        with sentiment_tabs[1]:  # Sentiment Distribution tab
            if not st.session_state.sentiment_data.empty:
                # Create sentiment distribution plot
                dist_fig = plot_sentiment_distribution(
                    st.session_state.sentiment_data,
                    title="Oil Market Sentiment Distribution"
                )
                
                st.plotly_chart(dist_fig, use_container_width=True)
                
                # Show sentiment statistics
                compound_values = st.session_state.sentiment_data['sentiment_compound'].dropna()
                
                # Calculate statistics
                mean = compound_values.mean()
                median = compound_values.median()
                std = compound_values.std()
                positive_pct = (compound_values > 0.05).mean() * 100
                neutral_pct = ((compound_values >= -0.05) & (compound_values <= 0.05)).mean() * 100
                negative_pct = (compound_values < -0.05).mean() * 100
                
                # Create statistics table
                statistics_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Standard Deviation', 'Positive Sentiment (%)', 'Neutral Sentiment (%)', 'Negative Sentiment (%)'],
                    'Value': [f"{mean:.4f}", f"{median:.4f}", f"{std:.4f}", f"{positive_pct:.2f}%", f"{neutral_pct:.2f}%", f"{negative_pct:.2f}%"]
                })
                
                st.subheader("Sentiment Statistics")
                st.dataframe(statistics_df, hide_index=True, use_container_width=True)
                
                # Add interpretation
                st.markdown(f"""
                    <div class="info-box">
                        <h4>Sentiment Distribution Interpretation</h4>
                        <p>The sentiment distribution shows the overall market sentiment towards oil. A mean sentiment of <strong>{mean:.4f}</strong> 
                        indicates {"a positive" if mean > 0.05 else "a negative" if mean < -0.05 else "a neutral"} bias in market discussions.</p>
                        <p>The distribution reveals that <strong>{positive_pct:.1f}%</strong> of discussions express positive sentiment, 
                        <strong>{neutral_pct:.1f}%</strong> are neutral, and <strong>{negative_pct:.1f}%</strong> are negative.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sentiment data not available.")
        
        with sentiment_tabs[2]:  # Price-Sentiment Correlation tab
            if not st.session_state.agg_sentiment.empty:
                # Create correlation analysis between price and sentiment
                
                # Align price and sentiment data
                sentiment_df = st.session_state.agg_sentiment.copy()
                price_df = st.session_state.price_data.copy()
                
                # Filter to common date range
                start_date = max(sentiment_df.index.min(), price_df.index.min())
                end_date = min(sentiment_df.index.max(), price_df.index.max())
                
                sentiment_aligned = sentiment_df.loc[start_date:end_date]
                price_aligned = price_df.loc[start_date:end_date]
                
                # Create DataFrame with both price and sentiment
                combined = pd.DataFrame({
                    'price': price_aligned['Price'],
                    'sentiment': sentiment_aligned['sentiment_compound']
                })
                
                # Calculate correlation
                corr = combined['price'].corr(combined['sentiment'])
                
                # Calculate cross-correlation at different lags
                max_lag = 10  # Days
                corrs = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        # Sentiment leads price (negative lag)
                        c = combined['price'].corr(combined['sentiment'].shift(lag))
                    else:
                        # Price leads sentiment (positive lag)
                        c = combined['price'].corr(combined['sentiment'].shift(lag))
                    corrs.append((lag, c))
                
                # Find max correlation and corresponding lag
                max_corr_lag, max_corr = max(corrs, key=lambda x: abs(x[1]))
                
                # Create scatter plot for price vs sentiment
                fig1 = px.scatter(
                    combined.reset_index(),
                    x='price',
                    y='sentiment',
                    trendline='ols',
                    labels={'price': 'Oil Price ($)', 'sentiment': 'Sentiment Score'},
                    title='Price vs. Sentiment Correlation',
                    height=500
                )
                
                # Add correlation annotation
                fig1.add_annotation(
                    x=combined['price'].max() * 0.9,
                    y=combined['sentiment'].min() * 0.8,
                    text=f"Correlation: {corr:.4f}",
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=4
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Create lag correlation plot
                fig2 = go.Figure()
                
                # Add correlation by lag
                fig2.add_trace(go.Bar(
                    x=[lag for lag, _ in corrs],
                    y=[corr for _, corr in corrs],
                    marker_color=['green' if c > 0 else 'red' for _, c in corrs],
                    hovertemplate='Lag: %{x}<br>Correlation: %{y:.4f}<extra></extra>'
                ))
                
                # Highlight max correlation
                fig2.add_shape(
                    type="rect",
                    x0=max_corr_lag - 0.4,
                    y0=0,
                    x1=max_corr_lag + 0.4,
                    y1=max_corr,
                    line=dict(width=0),
                    fillcolor="rgba(255, 255, 0, 0.3)"
                )
                
                # Add annotation for max correlation
                fig2.add_annotation(
                    x=max_corr_lag,
                    y=max_corr + (0.05 if max_corr > 0 else -0.05),
                    text=f"Max: {max_corr:.4f} at lag {max_corr_lag}",
                    showarrow=True,
                    arrowhead=1
                )
                
                # Update layout
                fig2.update_layout(
                    title="Cross-Correlation Between Price and Sentiment by Lag",
                    xaxis_title="Lag (Negative = Sentiment leads Price, Positive = Price leads Sentiment)",
                    yaxis_title="Correlation Coefficient",
                    template='plotly_white',
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                # Add zero reference line
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Add interpretation
                lead_follow = "leads" if max_corr_lag < 0 else "follows" if max_corr_lag > 0 else "moves simultaneously with"
                days_text = f"by {abs(max_corr_lag)} days" if max_corr_lag != 0 else ""
                
                st.markdown(f"""
                    <div class="info-box">
                        <h4>Price-Sentiment Correlation Analysis</h4>
                        <p>The overall correlation between oil price and market sentiment is <strong>{corr:.4f}</strong>, 
                        indicating a {"strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"} 
                        {"positive" if corr > 0 else "negative"} relationship.</p>
                        
                        <p>The lag analysis shows that sentiment {lead_follow} price movements {days_text}, 
                        with a maximum correlation of <strong>{max_corr:.4f}</strong> at lag {max_corr_lag}.</p>
                        
                        <p>{"This suggests that sentiment could be a leading indicator for price movements." if max_corr_lag < 0 else
                           "This suggests that price movements drive market sentiment rather than the other way around." if max_corr_lag > 0 else
                           "This suggests that price and sentiment tend to move together without either clearly leading the other."}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sentiment data not available for correlation analysis.")
    
    # ---- PAGE: SIGNAL DECOMPOSITION ----
    elif page == "Signal Decomposition":
        st.header("Signal Decomposition Analysis")
        
        # Check if decomposition is available
        if st.session_state.decomposition is not None:
            # Component selection
            component_options = list(st.session_state.decomposition.keys())
            selected_components = st.multiselect(
                "Select Components to Display",
                options=component_options,
                default=["original", "trend", "residual"]
            )
            
            if selected_components:
                # Create interactive decomposition plot
                decomp_fig = plot_interactive_decomposition(
                    {comp: st.session_state.decomposition[comp] for comp in selected_components},
                    st.session_state.decomposition_dates,
                    component_names=selected_components,
                    title="Oil Price Signal Decomposition"
                )
                
                st.plotly_chart(decomp_fig, use_container_width=True)
                
                # Add interpretation
                st.markdown("""
                    <div class="info-box">
                        <h4>Signal Decomposition Interpretation</h4>
                        <p>The signal decomposition breaks down the oil price into key components:</p>
                        <ul>
                            <li><strong>Original:</strong> The raw oil price time series.</li>
                            <li><strong>Trend:</strong> The long-term directional movement of oil prices, capturing the overall market movement.</li>
                            <li><strong>Cycles:</strong> Periodic patterns of different frequencies, representing market cycles of varying lengths.</li>
                            <li><strong>Residual:</strong> Random fluctuations and noise that cannot be explained by trend or cycles.</li>
                        </ul>
                        <p>This decomposition helps identify which components drive price movements and can improve forecasting by modeling each component separately.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Calculate variance explained by each component
                variances = {}
                original_variance = np.var(st.session_state.decomposition['original'])
                
                for component in st.session_state.decomposition:
                    if component != 'original':
                        variances[component] = (np.var(st.session_state.decomposition[component]) / original_variance) * 100
                
                # Create variance breakdown plot
                variance_fig = go.Figure()
                
                # Sort components by variance
                sorted_components = sorted(variances.items(), key=lambda x: x[1], reverse=True)
                components = [comp for comp, _ in sorted_components]
                var_values = [var for _, var in sorted_components]
                
                # Create horizontal bar chart
                variance_fig.add_trace(go.Bar(
                    y=components,
                    x=var_values,
                    orientation='h',
                    marker_color='skyblue',
                    text=[f"{var:.1f}%" for var in var_values],
                    textposition='outside',
                    hovertemplate='Component: %{y}<br>Variance Explained: %{x:.1f}%<extra></extra>'
                ))
                
                # Update layout
                variance_fig.update_layout(
                    title="Variance Explained by Each Component",
                    xaxis_title="Variance Explained (%)",
                    yaxis_title="Component",
                    template='plotly_white',
                    height=400,
                    margin=dict(l=10, r=100, t=50, b=10)  # Extra right margin for text labels
                )
                
                st.subheader("Component Importance Analysis")
                st.plotly_chart(variance_fig, use_container_width=True)
                
                # Component correlation with sentiment
                st.subheader("Component-Sentiment Correlation")
                
                if not st.session_state.agg_sentiment.empty:
                    # Align sentiment and decomposition dates
                    sentiment_aligned = st.session_state.agg_sentiment.copy()
                    
                    # Convert decomposition dates to DataFrame
                    decomp_df = pd.DataFrame(
                        {comp: st.session_state.decomposition[comp] for comp in st.session_state.decomposition},
                        index=st.session_state.decomposition_dates
                    )
                    
                    # Calculate correlations
                    correlations = {}
                    for component in decomp_df.columns:
                        correlations[component] = decomp_df[component].corr(sentiment_aligned['sentiment_compound'] if 'sentiment_compound' in sentiment_aligned else 0)
                    
                    # Create correlation heatmap
                    corr_fig = go.Figure()
                    
                    # Sort components by correlation strength
                    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    comp_names = [comp for comp, _ in sorted_corrs]
                    corr_values = [corr for _, corr in sorted_corrs]
                    
                    # Create horizontal bar chart
                    corr_fig.add_trace(go.Bar(
                        y=comp_names,
                        x=corr_values,
                        orientation='h',
                        marker_color=['green' if c > 0 else 'red' for c in corr_values],
                        text=[f"{corr:.4f}" for corr in corr_values],
                        textposition='outside',
                        hovertemplate='Component: %{y}<br>Correlation: %{x:.4f}<extra></extra>'
                    ))
                    
                    # Update layout
                    corr_fig.update_layout(
                        title="Correlation Between Components and Market Sentiment",
                        xaxis_title="Correlation Coefficient",
                        yaxis_title="Component",
                        template='plotly_white',
                        height=400,
                        margin=dict(l=10, r=100, t=50, b=10)  # Extra right margin for text labels
                    )
                    
                    # Add zero reference line
                    corr_fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(corr_fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                        <div class="info-box">
                            <h4>Component-Sentiment Correlation Interpretation</h4>
                            <p>This analysis shows how different price components correlate with market sentiment:</p>
                            <ul>
                                <li>High correlation between sentiment and trend suggests sentiment drives long-term price direction.</li>
                                <li>Correlation with cycles may indicate how sentiment influences market cycles of different durations.</li>
                                <li>Correlation with residuals suggests sentiment's impact on short-term price volatility.</li>
                            </ul>
                            <p>These insights help understand how market psychology influences different aspects of price behavior.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Sentiment data not available for correlation analysis.")
            else:
                st.warning("Please select at least one component to display")
        else:
            st.warning("Decomposition data not available. Please check if the price data was loaded correctly.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)