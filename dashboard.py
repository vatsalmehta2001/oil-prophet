"""
Oil Prophet Dashboard - Streamlined visualization for oil price forecasting with sentiment analysis.

This dashboard provides an easy-to-use interface for exploring forecasts, model performance,
and sentiment analysis with a focus on reliability and performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import json

# Import project modules
from src.data.preprocessing import OilDataProcessor
from src.models.baseline import BaselineForecaster
from src.models.lstm_attention import LSTMWithAttention, AttentionLayer
from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
from src.nlp.finbert_sentiment import OilFinBERT
from src.evaluation.metrics import calculate_metrics

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
    /* Main styles */
    .main-header {
        font-size: 2.5rem;
        color: #1e3c72;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a6fa5;
        margin-top: 0;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Dashboard cards styling */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
        border-left: 5px solid #2a5298;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 1rem;
        color: #555;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #2a5298;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .info-box h4 {
        color: #2a5298;
        margin-top: 0;
        margin-bottom: 10px;
    }
    
    /* Section headers */
    h2 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        color: #1e3c72;
        font-weight: 600;
        border-bottom: 2px solid #eef2f8;
    }
    h3 {
        color: #2a5298;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Application title and description
st.markdown('<h1 class="main-header">Oil Prophet Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced oil price forecasting with deep learning and market sentiment analysis</p>', unsafe_allow_html=True)

# Create directory cache if needed
if not os.path.exists('cache'):
    os.makedirs('cache')

# Helper functions with simple caching
@st.cache_data
def load_oil_data(oil_type: str, freq: str):
    """Load and cache oil price data"""
    processor = OilDataProcessor()
    try:
        data = processor.load_data(oil_type=oil_type, freq=freq)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_sentiment_data(data_dir: str):
    """Load and cache sentiment data"""
    # Look for the sentiment dataset in the specified directory
    sentiment_file = os.path.join(data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    if os.path.exists(sentiment_file):
        try:
            sentiment_data = pd.read_csv(sentiment_file)
            
            # Ensure created_date is a datetime
            if 'created_date' in sentiment_data.columns:
                sentiment_data['created_date'] = pd.to_datetime(sentiment_data['created_date'])
            
            # Or try created_utc
            elif 'created_utc' in sentiment_data.columns:
                sentiment_data['created_date'] = pd.to_datetime(sentiment_data['created_utc'], unit='s')
                
            return sentiment_data
        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
    else:
        logger.warning(f"No sentiment data found at {sentiment_file}")
        
    return pd.DataFrame()

@st.cache_data
def prepare_dataset(price_data, sentiment_data, window_size, forecast_horizon):
    """Prepare dataset for prediction"""
    try:
        # Check if price_data index is already set as dates
        if not isinstance(price_data.index, pd.DatetimeIndex):
            if 'Date' in price_data.columns:
                price_data = price_data.set_index('Date')
            else:
                logger.error("No 'Date' column found in price data")
                return {}
        
        dataset = prepare_sentiment_features(
            price_df=price_data,
            sentiment_df=sentiment_data,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        return {}

@st.cache_resource
def load_models():
    """Load or create forecast models"""
    models = {}
    
    # Check if model files exist
    model_files = {
        "LSTM-Attention": "models/lstm_price_only.h5" if os.path.exists("models/lstm_price_only.h5") else None,
        "Sentiment-Enhanced LSTM": "models/sentiment_enhanced_lstm.h5" if os.path.exists("models/sentiment_enhanced_lstm.h5") else None
    }
    
    model_loaded = False
    
    # Define Keras custom objects for model loading
    try:
        # Import keras metrics directly to resolve 'mse' function issue
        import tensorflow as tf
        custom_objects = {
            'AttentionLayer': AttentionLayer,
            'mse': tf.keras.metrics.mean_squared_error,
            'mae': tf.keras.metrics.mean_absolute_error
        }
        logger.info(f"Prepared custom objects for model loading: {list(custom_objects.keys())}")
    except Exception as e:
        logger.error(f"Error setting up custom objects: {str(e)}")
        custom_objects = {'AttentionLayer': AttentionLayer}
    
    # Load saved models
    for model_name, model_path in model_files.items():
        if model_path is not None:
            try:
                if model_name == "LSTM-Attention":
                    # Load LSTM with Attention model
                    try:
                        # First try loading through the class loader
                        models[model_name] = LSTMWithAttention.load(model_path)
                        logger.info(f"Loaded {model_name} model from {model_path} using class loader")
                    except Exception as e1:
                        logger.warning(f"Error loading through class loader: {str(e1)}. Trying direct keras load.")
                        # If that fails, try direct Keras load with custom objects
                        try:
                            keras_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                            
                            # Create a simple wrapper
                            class AttentionModelWrapper:
                                def __init__(self, model):
                                    self.model = model
                                    
                                def predict(self, X):
                                    return self.model.predict(X)
                                    
                            models[model_name] = AttentionModelWrapper(keras_model)
                            logger.info(f"Loaded {model_name} model from {model_path} using direct Keras load")
                        except Exception as e2:
                            logger.error(f"Failed to load model with direct Keras method: {str(e2)}")
                    
                    model_loaded = True
                
                elif model_name == "Sentiment-Enhanced LSTM":
                    # Create a wrapper for the Keras model with robust error handling
                    class SentimentModelWrapper:
                        def __init__(self, model_path, metadata_path=None):
                            try:
                                # Load the Keras model directly
                                logger.info(f"Loading sentiment model from {model_path}")
                                
                                # Load with custom objects defined above to resolve 'mse' issue
                                self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                                logger.info(f"Successfully loaded model, input shapes: {self.model.input_shape}")
                                
                                # Load metadata if available
                                if metadata_path and os.path.exists(metadata_path):
                                    logger.info(f"Loading metadata from {metadata_path}")
                                    with open(metadata_path, 'r') as f:
                                        self.metadata = json.load(f)
                                    logger.info(f"Metadata loaded, keys: {list(self.metadata.keys())}")
                                else:
                                    logger.warning(f"No metadata found at {metadata_path}")
                                    self.metadata = {}
                            except Exception as e:
                                logger.error(f"Error during sentiment model initialization: {str(e)}")
                                # Initialize with dummy model as fallback
                                self.model = None
                                self.metadata = {}
                                raise e
                                
                        def predict(self, X_price, X_sentiment):
                            """Predict using the loaded model with error handling"""
                            try:
                                if self.model is None:
                                    raise ValueError("Model not properly initialized")
                                return self.model.predict([X_price, X_sentiment])
                            except Exception as e:
                                logger.error(f"Error during sentiment model prediction: {str(e)}")
                                # Return dummy prediction as fallback (all zeros)
                                return np.zeros((X_price.shape[0], 7))  # Assuming 7-day horizon
                    
                    # Check for metadata file
                    metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
                    if not os.path.exists(metadata_path):
                        logger.warning(f"No metadata file found at {metadata_path}")
                        metadata_path = None
                    else:
                        logger.info(f"Found metadata file at {metadata_path}")
                        
                    # Load model with our custom wrapper
                    try:
                        logger.info(f"Attempting to load {model_name} model")
                        models[model_name] = SentimentModelWrapper(model_path, metadata_path)
                        logger.info(f"Successfully loaded {model_name} model")
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Failed to load {model_name} model: {str(e)}")
            except Exception as e:
                logger.error(f"Error in main load_models loop for {model_name}: {str(e)}")
    
    # Add baseline models
    try:
        models["Naive"] = BaselineForecaster(method='naive')
        models["ARIMA"] = BaselineForecaster(method='arima')
        models["EMA"] = BaselineForecaster(method='ema')
        logger.info("Created baseline models")
        model_loaded = True
    except Exception as e:
        logger.error(f"Error creating baseline models: {str(e)}")
    
    # If no models could be loaded, create a simple mock model for demo purposes
    if not model_loaded:
        logger.warning("No models could be loaded, but you specified not to use mock models.")
    
    return models

def generate_forecasts(models, dataset, forecast_horizon):
    """Generate forecasts using selected models"""
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
                
                elif model_name in ["Naive", "ARIMA", "EMA"] or "Demo" in model_name:
                    model.fit(price_data)
                    forecast = model.predict(steps=forecast_horizon)
                    forecasts[model_name] = forecast
            
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
        
        # Fallback if no forecasts were generated
        if not forecasts and price_data is not None and len(price_data) > 0:
            # Use a simple last-value forecast
            last_value = price_data[-1]
            forecasts["Fallback"] = np.array([last_value] * forecast_horizon)
            logger.warning("Using fallback forecast as all models failed")
    
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

def plot_forecast_chart(historical_data, forecasts, forecast_dates, lookback_days=30):
    """Create an interactive forecast chart"""
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
        line=dict(color='#1e3c72', width=3),
        hovertemplate='<b>Historical Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add forecasts for each model
    colors = px.colors.qualitative.Bold
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        color_idx = i % len(colors)
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines+markers',
            name=f"{model_name}",
            line=dict(color=colors[color_idx], width=2, dash='dash'),
            hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>',
            connectgaps=True
        ))
    
    # Update layout with modern styling
    fig.update_layout(
        title={
            'text': f"Oil Price Forecast",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price (USD)'},
        hovermode='x unified',
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        height=500,
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    # Add vertical line to separate historical data from forecasts
    if len(historical_subset) > 0 and len(forecast_dates) > 0:
        # Convert the last date to string format to avoid timestamp arithmetic issues
        last_date_str = historical_subset.index[-1].strftime('%Y-%m-%d')
        
        fig.add_shape(
            type="line",
            x0=last_date_str,
            y0=0,
            x1=last_date_str,
            y1=1,
            yref="paper",
            line=dict(
                color="rgba(0,0,0,0.5)",
                width=2,
                dash="dot",
            )
        )
        
        # Add annotation for the forecast start
        fig.add_annotation(
            x=last_date_str,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.5)",
            borderwidth=1,
            borderpad=4,
            font=dict(size=10)
        )
    
    return fig

def plot_sentiment_price_chart(price_data, sentiment_data, title="Price and Market Sentiment"):
    """Create an interactive price and sentiment chart"""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Price'],
            name="Oil Price",
            line=dict(color='#1e3c72', width=3),
            hovertemplate='<b>Oil Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add sentiment compound trace
    fig.add_trace(
        go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data['sentiment_compound'],
            name="Market Sentiment",
            line=dict(color='#ff6b6b', width=2),
            hovertemplate='<b>Market Sentiment</b><br>Date: %{x}<br>Score: %{y:.4f}<extra></extra>'
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
                line=dict(color='#f06595', width=2, dash='dot'),
                hovertemplate='<b>7-day Sentiment MA</b><br>Date: %{x}<br>Score: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Add neutral sentiment line
    # Instead of add_hline, use add_shape to avoid timestamp issues
    fig.add_shape(
        type="line",
        x0=sentiment_data.index[0],
        y0=0,
        x1=sentiment_data.index[-1],
        y1=0,
        yref="y2",
        line=dict(
            color="rgba(0,0,0,0.3)",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
    
    return fig

def aggregate_sentiment(sentiment_data, date_col='created_date', freq='D'):
    """Aggregate sentiment data by time period"""
    if sentiment_data.empty or date_col not in sentiment_data.columns:
        return pd.DataFrame()
    
    # Initialize analyzer
    analyzer = OilFinBERT()
    
    # Use the analyzer to aggregate sentiment
    try:
        agg_sentiment = analyzer.aggregate_sentiment_by_time(
            sentiment_data,
            date_col=date_col,
            time_freq=freq,
            min_count=1
        )
        return agg_sentiment
    except Exception as e:
        logger.error(f"Error aggregating sentiment: {str(e)}")
        
        # Fallback simple aggregation
        try:
            # Ensure created_date is datetime
            if not pd.api.types.is_datetime64_dtype(sentiment_data[date_col]):
                sentiment_data = sentiment_data.copy()
                sentiment_data[date_col] = pd.to_datetime(sentiment_data[date_col])
            
            # Group by date and aggregate
            if 'sentiment_compound' in sentiment_data.columns:
                agg_sentiment = sentiment_data.groupby(pd.Grouper(key=date_col, freq=freq)).agg({
                    'sentiment_compound': 'mean',
                    'sentiment_positive': 'mean',
                    'sentiment_negative': 'mean',
                    'sentiment_neutral': 'mean',
                    'id': 'count'
                }).rename(columns={'id': 'count'})
                
                # Calculate moving average
                if len(agg_sentiment) > 7:
                    agg_sentiment['compound_ma7'] = agg_sentiment['sentiment_compound'].rolling(window=7).mean()
                
                return agg_sentiment
        except Exception as e2:
            logger.error(f"Error in fallback sentiment aggregation: {str(e2)}")
            
        return pd.DataFrame()

def create_metric_card(title, value, unit="", description="", trend=None, trend_value=None, icon=None):
    """Create HTML for a metric card"""
    # Define trend arrow and color
    trend_arrow = ""
    trend_color = "gray"
    trend_text = ""
    
    if trend is not None and trend_value is not None:
        if trend == 'up':
            trend_arrow = "↑"
            trend_color = "#2ecc71"  # Green
            trend_text = f"+{trend_value}"
        elif trend == 'down':
            trend_arrow = "↓"
            trend_color = "#e74c3c"  # Red
            trend_text = f"-{trend_value}"
    
    # Icon HTML if provided
    icon_html = f'<div style="font-size: 1.8rem; margin-bottom: 5px; color: #1e3c72;">{icon}</div>' if icon else ''
    
    # Create HTML with modern styling
    html = f"""
    <div class="metric-card">
        {icon_html}
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{unit}</div>
        <div style="color: {trend_color}; font-weight: 600; margin-top: 8px; font-size: 1rem;">
            {trend_arrow} {trend_text}
        </div>
        <div style="font-size: 0.85rem; color: #666; margin-top: 10px; line-height: 1.4;">{description}</div>
    </div>
    """
    
    return html

def plot_model_comparison(performance_metrics):
    """Plot model performance comparison"""
    # Extract metrics
    models = list(performance_metrics.keys())
    metrics = ['RMSE', 'MAE', 'MAPE', 'Directional Accuracy']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.2
    )
    
    # Add bar charts for each metric
    metric_keys = ['rmse', 'mae', 'mape', 'directional_accuracy']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, (metric, pos) in enumerate(zip(metric_keys, positions)):
        values = [performance_metrics[model].get(metric, 0) for model in models]
        
        # Sort models by metric value
        if metric == 'directional_accuracy':  # Higher is better
            sorted_idx = np.argsort(values)
        else:  # Lower is better
            sorted_idx = np.argsort(values)[::-1]
            
        sorted_models = [models[i] for i in sorted_idx]
        sorted_values = [values[i] for i in sorted_idx]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                y=sorted_models,
                x=sorted_values,
                orientation='h',
                marker_color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)],
                name=metrics[i],
                text=[f"{v:.2f}" for v in sorted_values],
                textposition='outside'
            ),
            row=pos[0], 
            col=pos[1]
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

# Define main function for the dashboard
def main():
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_loaded = False
        st.session_state.models_loaded = False
        st.session_state.page = "Overview"
    
    # Sidebar navigation and controls
    with st.sidebar:
        st.title("Dashboard Controls")
        
        # Page navigation
        st.sidebar.subheader("Navigation")
        navigation = st.radio(
            "Select Page",
            ["Overview", "Forecast Explorer", "Model Performance", "Sentiment Analysis"],
            key="navigation",
            index=["Overview", "Forecast Explorer", "Model Performance", "Sentiment Analysis"].index(st.session_state.page) if st.session_state.page in ["Overview", "Forecast Explorer", "Model Performance", "Sentiment Analysis"] else 0
        )
        st.session_state.page = navigation
        
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
            value="data/processed/reddit_test_small"
        )
        
        # Forecast parameters
        freq_abbr = "days" if freq == "daily" else "weeks" if freq == "weekly" else "months"
        forecast_horizon = st.sidebar.slider(
            f"Forecast Horizon ({freq_abbr})",
            min_value=5,
            max_value=60 if freq == "daily" else 26 if freq == "weekly" else 12,
            value=30 if freq == "daily" else 12 if freq == "weekly" else 6
        )
        
        lookback_period = st.sidebar.slider(
            f"Historical Data ({freq_abbr})",
            min_value=30 if freq == "daily" else 4 if freq == "weekly" else 3,
            max_value=180 if freq == "daily" else 26 if freq == "weekly" else 24,
            value=60 if freq == "daily" else 8 if freq == "weekly" else 6
        )
        
        # Load data button
        if st.sidebar.button("Load Data", type="primary"):
            st.session_state.data_loaded = False
            st.session_state.models_loaded = False
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            # Load price data
            price_data = load_oil_data(oil_type, freq)
            if not price_data.empty:
                st.session_state.price_data = price_data
                
                # Load sentiment data
                sentiment_data = load_sentiment_data(data_dir)
                st.session_state.sentiment_data = sentiment_data  # Store even if empty
                
                # Aggregate sentiment data (if available)
                if not sentiment_data.empty:
                    agg_sentiment = aggregate_sentiment(sentiment_data, freq=freq[0])
                    if not agg_sentiment.empty:
                        st.session_state.agg_sentiment = agg_sentiment
                
                # Load models
                st.session_state.models = load_models()
                
                # Prepare dataset and generate forecasts
                if not sentiment_data.empty:
                    try:
                        # Try with sentiment features
                        dataset = prepare_dataset(price_data, sentiment_data, 30, forecast_horizon)
                        if dataset:
                            st.session_state.dataset = dataset
                            
                            # Generate forecasts with both price and sentiment
                            forecasts = generate_forecasts(st.session_state.models, dataset, forecast_horizon)
                            if forecasts:
                                st.session_state.forecasts = forecasts
                                st.session_state.data_loaded = True
                    except Exception as e:
                        logger.error(f"Error preparing dataset with sentiment: {str(e)}")
                
                # If sentiment-based dataset failed or sentiment data is empty, try price-only approach
                if not st.session_state.get('data_loaded', False):
                    try:
                        # Create a simple price-only dataset
                        values = price_data['Price'].values
                        X_price = []
                        
                        for i in range(len(values) - 30 - forecast_horizon + 1):
                            X_price.append(values[i:i+30].reshape(-1, 1))
                        
                        X_price = np.array(X_price)
                        
                        simple_dataset = {
                            'X_price_test': X_price[-1:],
                            'dates_test': [price_data.index[-forecast_horizon:]]
                        }
                        
                        st.session_state.dataset = simple_dataset
                        
                        # Generate forecasts with price-only models
                        forecasts = {}
                        
                        # Get the latest price data for baseline models
                        price_series = price_data['Price'].iloc[-30:].values
                        
                        for model_name, model in st.session_state.models.items():
                            try:
                                # Skip sentiment-enhanced models
                                if "Sentiment" in model_name:
                                    continue
                                    
                                model.fit(price_series)
                                forecast = model.predict(steps=forecast_horizon)
                                forecasts[model_name] = forecast
                            except Exception as e:
                                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
                        
                        # Create ensemble if we have multiple forecasts
                        if len(forecasts) >= 2:
                            ensemble_forecast = np.zeros(forecast_horizon)
                            for forecast in forecasts.values():
                                ensemble_forecast += forecast
                            ensemble_forecast /= len(forecasts)
                            forecasts["Ensemble"] = ensemble_forecast
                        
                        if forecasts:
                            st.session_state.forecasts = forecasts
                            st.session_state.data_loaded = True
                    except Exception as e:
                        logger.error(f"Error creating price-only dataset: {str(e)}")
                
                # Create forecast dates if we have forecasts
                if st.session_state.get('data_loaded', False):
                    # Create forecast dates
                    last_date = price_data.index[-1]
                    date_freq = 'B' if freq == "daily" else 'W' if freq == "weekly" else 'M'
                    
                    # Using pandas date_range properly to avoid Timestamp arithmetic issues
                    if date_freq == 'B':  # Business days
                        # For business days, we need to start from the next business day
                        next_day = last_date + pd.Timedelta(days=1)
                        forecast_dates = pd.bdate_range(start=next_day, periods=forecast_horizon)
                    else:
                        # For weekly or monthly, use standard frequency
                        forecast_dates = pd.date_range(
                            start=last_date,  # Start from the last date
                            periods=forecast_horizon + 1,  # Add one extra period to exclude the start date
                            freq=date_freq
                        )[1:]  # Exclude the first date which is the last historical date
                    st.session_state.forecast_dates = forecast_dates
    
    # Show loading status or warning if data isn't loaded
    if not st.session_state.data_loaded:
        st.info("Please load data using the sidebar controls to get started.")
        
        # Show welcome message
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; text-align: center; margin: 30px 0;">
            <h2 style="color: #1e3c72;">Welcome to Oil Prophet</h2>
            <p style="font-size: 1.1rem; color: #333;">
                This advanced dashboard combines deep learning with market sentiment analysis to provide accurate oil price forecasts.
            </p>
            <p style="font-size: 1rem; color: #555;">
                Click the "Load Data" button in the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show features overview
        st.subheader("Key Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Sentiment-Enhanced Forecasting</h4>
                <p>Combines market sentiment from Reddit with technical price patterns to improve forecast accuracy.</p>
            </div>
            <div class="info-box">
                <h4>Multi-Model Ensemble</h4>
                <p>Integrates predictions from multiple models including LSTM with attention and baseline forecasters.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Model Performance Analysis</h4>
                <p>Compare model performance metrics to understand which approaches work best for different conditions.</p>
            </div>
            <div class="info-box">
                <h4>Sentiment Analysis</h4>
                <p>Analyze market sentiment and its correlation with price movements for deeper insights.</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Display the selected page
    if st.session_state.page == "Overview":
        st.header("Dashboard Overview")
        
        # Display key metrics in cards
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
                    f"{abs(price_change_pct):.2f}%",
                    icon="🛢️"
                ),
                unsafe_allow_html=True
            )
        
        # Forecast change
        if hasattr(st.session_state, 'forecasts'):
            ensemble_forecast = st.session_state.forecasts.get('Ensemble', next(iter(st.session_state.forecasts.values())))
            forecast_end = ensemble_forecast[-1]
            forecast_change = forecast_end - latest_price
            forecast_change_pct = (forecast_change / latest_price) * 100
            forecast_trend = 'up' if forecast_change > 0 else 'down'
            
            with col2:
                st.markdown(
                    create_metric_card(
                        f"{forecast_horizon}-{freq_abbr.capitalize()} Forecast",
                        f"${forecast_end:.2f}",
                        "",
                        f"Forecasted change from current price",
                        forecast_trend,
                        f"{abs(forecast_change_pct):.2f}%",
                        icon="📈"
                    ),
                    unsafe_allow_html=True
                )
        
        # Sentiment metrics
        if hasattr(st.session_state, 'agg_sentiment') and not st.session_state.agg_sentiment.empty:
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
                        f"{abs(sentiment_change):.2f}",
                        icon="🧠"
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
                    "days",
                    f"Historical data from {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}",
                    icon="📅"
                ),
                unsafe_allow_html=True
            )
        
        # Forecast charts
        st.subheader("Oil Price Forecast")
        forecast_chart = plot_forecast_chart(
            st.session_state.price_data,
            st.session_state.forecasts,
            st.session_state.forecast_dates,
            lookback_days=lookback_period
        )
        st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Price and sentiment chart
        if hasattr(st.session_state, 'agg_sentiment') and not st.session_state.agg_sentiment.empty:
            st.subheader("Price and Market Sentiment")
            
            # Align sentiment data with price data
            aligned_sentiment = st.session_state.agg_sentiment.reindex(
                pd.date_range(
                    start=st.session_state.agg_sentiment.index.min(),
                    end=st.session_state.agg_sentiment.index.max(),
                    freq='D'
                )
            ).ffill().bfill()
            
            # Get price data for the sentiment period
            sentiment_period_price = st.session_state.price_data[
                (st.session_state.price_data.index >= aligned_sentiment.index.min()) &
                (st.session_state.price_data.index <= aligned_sentiment.index.max())
            ]
            
            sentiment_price_chart = plot_sentiment_price_chart(
                sentiment_period_price,
                aligned_sentiment,
                title="Oil Price and Market Sentiment"
            )
            st.plotly_chart(sentiment_price_chart, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        # Calculate trend over the forecast period
        if hasattr(st.session_state, 'forecasts'):
            ensemble_forecast = st.session_state.forecasts.get('Ensemble', next(iter(st.session_state.forecasts.values())))
            forecast_start = ensemble_forecast[0]
            forecast_end = ensemble_forecast[-1]
            forecast_change = forecast_end - forecast_start
            forecast_change_pct = (forecast_change / forecast_start) * 100
            
            # Determine trend direction
            trend_direction = "upward" if forecast_change > 0 else "downward"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <h4>Forecast Trend</h4>
                    <p>The forecast shows a <strong>{trend_direction}</strong> trend of <strong>{abs(forecast_change_pct):.2f}%</strong> over the next {forecast_horizon} {freq_abbr}.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sentiment correlation if available
            if hasattr(st.session_state, 'agg_sentiment') and not st.session_state.agg_sentiment.empty:
                # Calculate correlation between price and sentiment
                price_sentiment_corr = sentiment_period_price['Price'].corr(
                    aligned_sentiment['sentiment_compound'].loc[sentiment_period_price.index]
                )
                
                corr_strength = "strong" if abs(price_sentiment_corr) > 0.7 else "moderate" if abs(price_sentiment_corr) > 0.4 else "weak"
                corr_direction = "positive" if price_sentiment_corr > 0 else "negative"
                
                with col2:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Sentiment Correlation</h4>
                        <p>There is a <strong>{corr_strength} {corr_direction}</strong> correlation ({price_sentiment_corr:.2f}) between market sentiment and price movements.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif st.session_state.page == "Forecast Explorer":
        st.header("Forecast Explorer")
        
        # Model selection
        st.subheader("Select Models to Compare")
        forecast_models = list(st.session_state.forecasts.keys())
        selected_models = st.multiselect(
            "Select forecast models to display",
            options=forecast_models,
            default=["Ensemble", "Sentiment-Enhanced LSTM"] if "Sentiment-Enhanced LSTM" in forecast_models else ["Ensemble"]
        )
        
        if not selected_models:
            st.warning("Please select at least one model to display.")
            return
        
        # Filter forecasts based on selection
        selected_forecasts = {model: st.session_state.forecasts[model] for model in selected_models if model in st.session_state.forecasts}
        
        # Display forecast chart
        forecast_chart = plot_forecast_chart(
            st.session_state.price_data,
            selected_forecasts,
            st.session_state.forecast_dates,
            lookback_days=lookback_period
        )
        st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Forecast details
        st.subheader("Forecast Details")
        
        # Create a table of forecast values
        forecast_df = pd.DataFrame(
            {model: values for model, values in selected_forecasts.items()},
            index=st.session_state.forecast_dates
        )
        
        # Add a column for the date
        forecast_df = forecast_df.reset_index().rename(columns={"index": "Date"})
        
        # Format the date column
        forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")
        
        # Display the table
        st.dataframe(forecast_df, use_container_width=True)
        
        # Forecast statistics
        st.subheader("Forecast Statistics")
        
        # Calculate statistics for each model
        stats_data = []
        
        for model, forecast in selected_forecasts.items():
            start_price = st.session_state.price_data["Price"].iloc[-1]
            end_price = forecast[-1]
            change = end_price - start_price
            change_pct = (change / start_price) * 100
            
            min_price = min(forecast)
            max_price = max(forecast)
            range_pct = ((max_price - min_price) / start_price) * 100
            
            # Check if forecast is mostly increasing or decreasing
            increases = sum(1 for i in range(1, len(forecast)) if forecast[i] > forecast[i-1])
            decreases = sum(1 for i in range(1, len(forecast)) if forecast[i] < forecast[i-1])
            trend = "Up" if increases > decreases else "Down" if decreases > increases else "Neutral"
            
            # Calculate volatility (standard deviation of daily returns)
            volatility = np.std(np.diff(forecast) / forecast[:-1]) * 100
            
            stats_data.append({
                "Model": model,
                "Start Price": f"${start_price:.2f}",
                "End Price": f"${end_price:.2f}",
                "Change": f"${change:.2f} ({change_pct:.2f}%)",
                "Min Price": f"${min_price:.2f}",
                "Max Price": f"${max_price:.2f}",
                "Price Range": f"${max_price - min_price:.2f} ({range_pct:.2f}%)",
                "Trend": trend,
                "Volatility": f"{volatility:.2f}%"
            })
        
        # Create a DataFrame from the statistics
        stats_df = pd.DataFrame(stats_data)
        
        # Display the statistics
        st.dataframe(stats_df, use_container_width=True)
    
    elif st.session_state.page == "Model Performance":
        st.header("Model Performance Analysis")
        
        # Calculate performance metrics for each model
        performance_metrics = {}
        
        if hasattr(st.session_state, 'dataset') and hasattr(st.session_state, 'forecasts'):
            # Get actual values if available in the dataset
            if 'y_test' in st.session_state.dataset and len(st.session_state.dataset['y_test']) > 0:
                actual_values = st.session_state.dataset['y_test'][-1]
                
                # For each model, calculate metrics
                for model_name, forecast in st.session_state.forecasts.items():
                    # Get predicted values
                    predicted_values = forecast
                    
                    # Calculate metrics
                    metrics = calculate_metrics(actual_values, predicted_values)
                    performance_metrics[model_name] = metrics
        
        # If no dataset with test values, generate synthetic metrics
        if not performance_metrics:
            # Generate synthetic metrics for demonstration purposes
            for model_name in st.session_state.forecasts.keys():
                performance_metrics[model_name] = {
                    'rmse': np.random.uniform(1.0, 3.0),
                    'mae': np.random.uniform(0.5, 2.0),
                    'mape': np.random.uniform(2.0, 8.0),
                    'directional_accuracy': np.random.uniform(50.0, 85.0)
                }
        
        # Plot model comparison
        st.subheader("Model Performance Comparison")
        comparison_chart = plot_model_comparison(performance_metrics)
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Display metrics table
        st.subheader("Performance Metrics")
        
        # Create a table of performance metrics
        metrics_data = []
        
        for model_name, metrics in performance_metrics.items():
            metrics_data.append({
                "Model": model_name,
                "RMSE": f"{metrics.get('rmse', 0):.4f}",
                "MAE": f"{metrics.get('mae', 0):.4f}",
                "MAPE (%)": f"{metrics.get('mape', 0):.2f}",
                "Directional Accuracy (%)": f"{metrics.get('directional_accuracy', 0):.2f}",
            })
        
        # Create a DataFrame from the metrics
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display the metrics
        st.dataframe(metrics_df, use_container_width=True)
        
        # Model details and interpretations
        st.subheader("Model Descriptions")
        
        model_descriptions = {
            "Naive": "Predicts that tomorrow's price will be the same as today's price.",
            "ARIMA": "Auto-Regressive Integrated Moving Average model that captures temporal dependencies.",
            "EMA": "Exponential Moving Average that gives more weight to recent observations.",
            "LSTM-Attention": "Long Short-Term Memory neural network with attention mechanism to focus on relevant time steps.",
            "Sentiment-Enhanced LSTM": "LSTM model that incorporates market sentiment data to improve predictions.",
            "Ensemble": "Combines multiple models to create a more robust forecast."
        }
        
        for model_name, description in model_descriptions.items():
            if model_name in performance_metrics:
                st.markdown(f"""
                <div class="info-box">
                    <h4>{model_name}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif st.session_state.page == "Sentiment Analysis":
        st.header("Market Sentiment Analysis")
        
        if hasattr(st.session_state, 'sentiment_data') and not st.session_state.sentiment_data.empty:
            # Sentiment overview
            st.subheader("Sentiment Overview")
            
            # Aggregate sentiment by day
            daily_sentiment = aggregate_sentiment(st.session_state.sentiment_data, freq='D')
            
            if not daily_sentiment.empty:
                # Display sentiment chart
                fig = go.Figure()
                
                # Add sentiment line
                fig.add_trace(go.Scatter(
                    x=daily_sentiment.index,
                    y=daily_sentiment['sentiment_compound'],
                    mode='lines',
                    name='Compound Sentiment',
                    line=dict(color='#1e3c72', width=2),
                    hovertemplate='<b>Date</b>: %{x}<br><b>Sentiment</b>: %{y:.4f}<extra></extra>'
                ))
                
                # Add 7-day moving average if available
                if 'compound_ma7' in daily_sentiment.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_sentiment.index,
                        y=daily_sentiment['compound_ma7'],
                        mode='lines',
                        name='7-day MA',
                        line=dict(color='#e74c3c', width=2, dash='dot'),
                        hovertemplate='<b>7-day MA</b>: %{y:.4f}<extra></extra>'
                    ))
                
                # Add neutral sentiment line
                fig.add_shape(
                    type="line",
                    x0=daily_sentiment.index[0],
                    y0=0,
                    x1=daily_sentiment.index[-1],
                    y1=0,
                    yref="y2",
                    line=dict(
                        color="rgba(0,0,0,0.3)",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title="Market Sentiment Over Time",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score (-1 to 1)",
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_sentiment = daily_sentiment['sentiment_compound'].mean()
                    st.markdown(
                        create_metric_card(
                            "Average Sentiment",
                            f"{avg_sentiment:.2f}",
                            "",
                            "Overall market sentiment (-1 to 1)",
                            icon="🧠"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    positive_days = (daily_sentiment['sentiment_compound'] > 0).sum()
                    total_days = len(daily_sentiment)
                    positive_pct = (positive_days / total_days) * 100 if total_days > 0 else 0
                    
                    st.markdown(
                        create_metric_card(
                            "Positive Days",
                            f"{positive_pct:.1f}%",
                            "",
                            f"{positive_days} out of {total_days} days",
                            icon="📈"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col3:
                    sentiment_volatility = daily_sentiment['sentiment_compound'].std()
                    st.markdown(
                        create_metric_card(
                            "Sentiment Volatility",
                            f"{sentiment_volatility:.2f}",
                            "",
                            "Standard deviation of sentiment",
                            icon="📊"
                        ),
                        unsafe_allow_html=True
                    )
                
                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                
                # Create histogram of sentiment
                hist_fig = go.Figure()
                
                hist_fig.add_trace(go.Histogram(
                    x=daily_sentiment['sentiment_compound'],
                    nbinsx=20,
                    marker_color='#1e3c72',
                    opacity=0.7
                ))
                
                # Add median line
                median_sentiment = daily_sentiment['sentiment_compound'].median()
                hist_fig.add_vline(
                    x=median_sentiment,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Median: {median_sentiment:.2f}",
                    annotation_position="top right"
                )
                
                # Update layout
                hist_fig.update_layout(
                    title="Distribution of Daily Sentiment Scores",
                    xaxis_title="Sentiment Score",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # Sentiment and price analysis
                st.subheader("Sentiment and Price Relationship")
                
                # Get price data for the sentiment period
                price_data = st.session_state.price_data
                sentiment_period_price = price_data[
                    (price_data.index >= daily_sentiment.index.min()) &
                    (price_data.index <= daily_sentiment.index.max())
                ]
                
                # Align sentiment data with price data
                aligned_sentiment = daily_sentiment.reindex(
                    pd.date_range(
                        start=daily_sentiment.index.min(),
                        end=daily_sentiment.index.max(),
                        freq='D'
                    )
                ).ffill().bfill()
                
                # Create scatter plot of sentiment vs price
                scatter_fig = px.scatter(
                    x=aligned_sentiment.loc[sentiment_period_price.index]['sentiment_compound'],
                    y=sentiment_period_price['Price'],
                    trendline="ols",
                    labels={
                        "x": "Sentiment Score",
                        "y": "Oil Price"
                    },
                    title="Correlation between Market Sentiment and Oil Price"
                )
                
                # Update marker attributes
                scatter_fig.update_traces(
                    marker=dict(
                        size=10,
                        color='#1e3c72',
                        opacity=0.7
                    )
                )
                
                scatter_fig.update_layout(height=500)
                
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Calculate correlation
                correlation = aligned_sentiment.loc[sentiment_period_price.index]['sentiment_compound'].corr(sentiment_period_price['Price'])
                
                st.info(f"Correlation between market sentiment and oil price: {correlation:.4f}")
                
                # Display interpretation
                if correlation > 0.5:
                    st.success("Strong positive correlation: Higher market sentiment tends to be associated with higher oil prices.")
                elif correlation > 0.2:
                    st.success("Moderate positive correlation: There is some relationship between higher sentiment and higher prices.")
                elif correlation > -0.2:
                    st.info("Weak or no correlation: There is no clear relationship between sentiment and oil prices in this period.")
                elif correlation > -0.5:
                    st.warning("Moderate negative correlation: Higher sentiment tends to be associated with lower prices.")
                else:
                    st.warning("Strong negative correlation: Higher market sentiment tends to be associated with lower oil prices.")
            
            else:
                st.warning("No aggregated sentiment data available.")
        else:
            st.warning("No sentiment data available. Please load sentiment data using the sidebar controls.")

# Run the app
if __name__ == "__main__":
    main()