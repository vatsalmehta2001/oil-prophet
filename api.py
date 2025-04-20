"""
Oil Prophet API

This module implements a RESTful API for the Oil Prophet forecasting system using FastAPI.
It provides endpoints to access oil price forecasts, model performance metrics,
and sentiment analysis.

Key features:
- Oil price forecasting with multiple models (LSTM, sentiment-enhanced LSTM, and statistical baselines)
- Market sentiment analysis for oil discussions
- Model performance comparison
- Pipeline execution control
- Secure API access with API key authentication

Usage example:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Path, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator

# Import project modules
from src.data.preprocessing import OilDataProcessor
from src.nlp.finbert_sentiment import OilFinBERT
from src.models.lstm_attention import LSTMWithAttention
from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
from src.models.baseline import BaselineForecaster
from src.evaluation.metrics import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Oil Prophet API",
    description="API for advanced oil price forecasting with sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Simple API key security
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("OIL_PROPHET_API_KEY", "development_key")  # Set via environment variable in production
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key if security is enabled."""
    if API_KEY != "development_key" and api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key

# Cached model and data instances
_price_data = {}
_sentiment_data = None
_models = {}

# ---- Pydantic Models for Request/Response ----

class ForecastRequest(BaseModel):
    """Request model for forecast generation."""
    oil_type: str = Field("brent", description="Oil type: 'brent' or 'wti'")
    frequency: str = Field("daily", description="Data frequency: 'daily', 'weekly', or 'monthly'")
    model: str = Field("ensemble", description="Model to use: 'lstm', 'sentiment', 'baseline', or 'ensemble'")
    forecast_horizon: int = Field(7, description="Number of periods to forecast", ge=1, le=60)
    include_history: bool = Field(False, description="Include historical data in response")
    history_periods: int = Field(30, description="Number of historical periods to include", ge=0, le=180)
    
    @validator('oil_type')
    def validate_oil_type(cls, v):
        if v.lower() not in ['brent', 'wti']:
            raise ValueError('Oil type must be either "brent" or "wti"')
        return v.lower()
    
    @validator('frequency')
    def validate_frequency(cls, v):
        if v.lower() not in ['daily', 'weekly', 'monthly']:
            raise ValueError('Frequency must be one of: "daily", "weekly", or "monthly"')
        return v.lower()
    
    @validator('model')
    def validate_model(cls, v):
        valid_models = ['lstm', 'sentiment', 'baseline', 'ensemble']
        if v.lower() not in valid_models:
            raise ValueError(f'Model must be one of: {", ".join(valid_models)}')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "oil_type": "brent",
                "frequency": "daily",
                "model": "ensemble",
                "forecast_horizon": 14,
                "include_history": True,
                "history_periods": 30
            }
        }

class ForecastPoint(BaseModel):
    """Data point in forecast response."""
    date: str
    price: float
    is_forecast: bool

class ForecastResponse(BaseModel):
    """Response model for forecast data."""
    oil_type: str
    frequency: str
    model: str
    forecast_start_date: str
    forecast_end_date: str
    current_price: float
    forecasted_end_price: float
    price_change: float
    price_change_percentage: float
    data: List[ForecastPoint]
    confidence_interval: Optional[Dict[str, List[float]]] = None
    model_metrics: Optional[Dict[str, float]] = None
    last_updated: str

class SentimentMetrics(BaseModel):
    """Model for sentiment metrics."""
    mean: float
    median: float
    positive_percentage: float
    neutral_percentage: float
    negative_percentage: float
    volatility: float
    momentum: float
    current: float
    period_change: float

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    timeframe: str
    period_start: str
    period_end: str
    metrics: SentimentMetrics
    sentiment_data: Optional[List[Dict[str, Any]]] = None
    correlation_with_price: Optional[float] = None
    last_updated: str

class ModelPerformance(BaseModel):
    """Response model for model performance metrics."""
    model: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    comparison_to_baseline: Dict[str, float]

class PerformanceResponse(BaseModel):
    """Response model for all models' performance."""
    models: List[ModelPerformance]
    best_model: str
    test_period: str
    last_updated: str

class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str
    code: int
    details: Optional[str] = None

# ---- Helper Functions ----

def load_price_data(oil_type: str, freq: str) -> pd.DataFrame:
    """Load price data with caching."""
    cache_key = f"{oil_type}_{freq}"
    
    if cache_key not in _price_data:
        try:
            processor = OilDataProcessor()
            data = processor.load_data(oil_type=oil_type, freq=freq)
            _price_data[cache_key] = data
            logger.info(f"Loaded {oil_type} {freq} price data")
            return data
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load price data: {str(e)}"
            )
    return _price_data[cache_key]

def load_sentiment_data() -> pd.DataFrame:
    """Load sentiment data with caching."""
    global _sentiment_data
    
    if _sentiment_data is None:
        # Try to find sentiment data in the data directory
        data_dir = "data/processed/reddit_test_small"
        analyzed_file = os.path.join(data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
        
        if os.path.exists(analyzed_file):
            try:
                _sentiment_data = pd.read_csv(analyzed_file)
                # Ensure created_date is a datetime
                if 'created_date' in _sentiment_data.columns:
                    _sentiment_data['created_date'] = pd.to_datetime(_sentiment_data['created_date'])
                elif 'created_utc' in _sentiment_data.columns:
                    _sentiment_data['created_date'] = pd.to_datetime(_sentiment_data['created_utc'], unit='s')
                logger.info(f"Loaded sentiment data from {analyzed_file}")
            except Exception as e:
                logger.error(f"Error loading sentiment data: {str(e)}")
                _sentiment_data = pd.DataFrame()  # Empty DataFrame as fallback
        else:
            logger.warning(f"Sentiment data file not found: {analyzed_file}")
            _sentiment_data = pd.DataFrame()  # Empty DataFrame as fallback
    
    return _sentiment_data

def load_model(model_name: str) -> object:
    """Load model with caching."""
    if model_name not in _models:
        try:
            if model_name == "lstm":
                model_path = "models/lstm_price_only.h5"
                if os.path.exists(model_path):
                    _models[model_name] = LSTMWithAttention.load(model_path)
                    logger.info(f"Loaded LSTM model from {model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            elif model_name == "sentiment":
                model_path = "models/sentiment_enhanced_lstm.h5"
                if os.path.exists(model_path):
                    _models[model_name] = SentimentEnhancedLSTM.load(model_path)
                    logger.info(f"Loaded Sentiment-Enhanced LSTM model from {model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            elif model_name == "baseline":
                _models[model_name] = BaselineForecaster(method='arima')
                logger.info("Created ARIMA baseline model")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    return _models.get(model_name)

def prepare_dataset(
    price_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    window_size: int = 30,
    forecast_horizon: int = 7
) -> Dict[str, np.ndarray]:
    """Prepare dataset for prediction."""
    try:
        if sentiment_data.empty:
            # Create minimal dataset with just price data
            # This is a simplified version for API use when sentiment data is unavailable
            values = price_data['Price'].values
            X_price = []
            
            for i in range(len(values) - window_size - forecast_horizon + 1):
                X_price.append(values[i:i+window_size].reshape(-1, 1))
            
            X_price = np.array(X_price)
            
            return {
                'X_price_test': X_price[-1:],
                'dates_test': [price_data.index[-forecast_horizon:]]
            }
        else:
            # Use the full preparation function
            dataset = prepare_sentiment_features(
                price_df=price_data,
                sentiment_df=sentiment_data,
                window_size=window_size,
                forecast_horizon=forecast_horizon
            )
            return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare dataset: {str(e)}"
        )

def generate_model_forecast(
    model_name: str,
    dataset: Dict[str, np.ndarray],
    forecast_horizon: int
) -> np.ndarray:
    """Generate forecast with specified model."""
    try:
        # Load model
        model = load_model(model_name)
        
        if model_name == "lstm":
            if 'X_price_test' in dataset:
                return model.predict(dataset['X_price_test'][-1:])[0]
            raise ValueError("Price test data not available")
        
        elif model_name == "sentiment":
            if 'X_price_test' in dataset and 'X_sentiment_test' in dataset:
                return model.predict(dataset['X_price_test'][-1:], dataset['X_sentiment_test'][-1:])[0]
            raise ValueError("Price or sentiment test data not available")
        
        elif model_name == "baseline":
            if 'X_price_test' in dataset:
                # Extract the last window of price data
                price_window = dataset['X_price_test'][-1, :, 0]
                model.fit(price_window)
                return model.predict(steps=forecast_horizon)
            raise ValueError("Price test data not available")
        
        raise ValueError(f"Unsupported model: {model_name}")
    
    except Exception as e:
        logger.error(f"Error generating forecast with {model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast: {str(e)}"
        )

def get_forecast_dates(last_date: datetime, frequency: str, horizon: int) -> List[datetime]:
    """Generate forecast dates based on frequency."""
    if frequency == "daily":
        # Use business days
        dates = []
        current_date = last_date
        for _ in range(horizon):
            current_date = current_date + timedelta(days=1)
            # Skip weekends (simplified approach)
            while current_date.weekday() > 4:  # 5=Saturday, 6=Sunday
                current_date = current_date + timedelta(days=1)
            dates.append(current_date)
        return dates
    
    elif frequency == "weekly":
        # Add weeks
        return [last_date + timedelta(weeks=i+1) for i in range(horizon)]
    
    elif frequency == "monthly":
        # Add months (approximate)
        return [last_date + timedelta(days=30*(i+1)) for i in range(horizon)]
    
    raise ValueError(f"Unsupported frequency: {frequency}")

def generate_confidence_interval(
    forecast: np.ndarray, 
    price_volatility: float,
    confidence: float = 0.95
) -> Dict[str, List[float]]:
    """Generate confidence intervals for the forecast."""
    # Calculate z-score for the confidence level
    # For 95% confidence, z-score is approximately 1.96
    z_score = 1.96 if confidence == 0.95 else 2.58 if confidence == 0.99 else 1.645
    
    # Calculate interval width based on historical volatility
    interval_width = z_score * price_volatility * np.sqrt(np.arange(1, len(forecast) + 1))
    
    # Calculate upper and lower bounds
    upper_bound = forecast + interval_width
    lower_bound = forecast - interval_width
    
    return {
        "upper": upper_bound.tolist(),
        "lower": lower_bound.tolist()
    }

def calculate_model_metrics(models_to_evaluate: List[str], dataset: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Calculate performance metrics for models."""
    if 'y_test' not in dataset or dataset['y_test'].size == 0:
        return {}
    
    metrics = {}
    test_data = dataset['y_test'][0]  # Use first test sample for evaluation
    
    for model_name in models_to_evaluate:
        try:
            if model_name in ["lstm", "sentiment", "baseline"]:
                # Generate forecast with the model
                forecast = generate_model_forecast(model_name, dataset, len(test_data))
                
                # Calculate metrics
                model_metrics = calculate_metrics(test_data, forecast)
                metrics[model_name] = model_metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
    
    return metrics

def aggregate_sentiment_metrics(sentiment_data: pd.DataFrame, timeframe: str = "30d") -> SentimentMetrics:
    """Aggregate sentiment data metrics for the specified timeframe."""
    if sentiment_data.empty:
        # Return default values if no data
        return SentimentMetrics(
            mean=0.0,
            median=0.0,
            positive_percentage=0.0,
            neutral_percentage=0.0,
            negative_percentage=0.0,
            volatility=0.0,
            momentum=0.0,
            current=0.0,
            period_change=0.0
        )
    
    try:
        # Parse timeframe (e.g., "30d" for 30 days)
        value = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        
        # Calculate cutoff date
        now = datetime.now()
        if unit == 'd':
            cutoff = now - timedelta(days=value)
        elif unit == 'w':
            cutoff = now - timedelta(weeks=value)
        elif unit == 'm':
            cutoff = now - timedelta(days=30*value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
        
        # Filter sentiment data
        date_col = 'created_date' if 'created_date' in sentiment_data.columns else None
        
        if date_col and not pd.api.types.is_datetime64_dtype(sentiment_data[date_col]):
            sentiment_data[date_col] = pd.to_datetime(sentiment_data[date_col])
        
        recent_sentiment = sentiment_data[sentiment_data[date_col] >= cutoff] if date_col else sentiment_data
        
        if recent_sentiment.empty:
            raise ValueError("No sentiment data available for the specified timeframe")
        
        # Calculate metrics
        compound_values = recent_sentiment['sentiment_compound'].dropna()
        
        mean = float(compound_values.mean())
        median = float(compound_values.median())
        positive_pct = float((compound_values > 0.05).mean() * 100)
        neutral_pct = float(((compound_values >= -0.05) & (compound_values <= 0.05)).mean() * 100)
        negative_pct = float((compound_values < -0.05).mean() * 100)
        volatility = float(compound_values.std())
        
        # Calculate momentum (change over the period)
        if len(compound_values) >= 2:
            # Group by day and calculate daily average
            if date_col:
                daily_sentiment = recent_sentiment.groupby(pd.Grouper(key=date_col, freq='D'))['sentiment_compound'].mean()
                if len(daily_sentiment) >= 2:
                    momentum = float(daily_sentiment.iloc[-1] - daily_sentiment.iloc[0])
                    current = float(daily_sentiment.iloc[-1])
                    period_change = float(daily_sentiment.iloc[-1] - daily_sentiment.iloc[0])
                else:
                    momentum = 0.0
                    current = float(compound_values.iloc[-1]) if not compound_values.empty else 0.0
                    period_change = 0.0
            else:
                momentum = 0.0
                current = float(compound_values.iloc[-1]) if not compound_values.empty else 0.0
                period_change = 0.0
        else:
            momentum = 0.0
            current = float(compound_values.iloc[-1]) if not compound_values.empty else 0.0
            period_change = 0.0
        
        return SentimentMetrics(
            mean=mean,
            median=median,
            positive_percentage=positive_pct,
            neutral_percentage=neutral_pct,
            negative_percentage=negative_pct,
            volatility=volatility,
            momentum=momentum,
            current=current,
            period_change=period_change
        )
    
    except Exception as e:
        logger.error(f"Error aggregating sentiment metrics: {str(e)}")
        # Return default values on error
        return SentimentMetrics(
            mean=0.0,
            median=0.0,
            positive_percentage=0.0,
            neutral_percentage=0.0,
            negative_percentage=0.0,
            volatility=0.0,
            momentum=0.0,
            current=0.0,
            period_change=0.0
        )

# ---- API Endpoints ----

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Oil Prophet API",
        "version": "1.0.0",
        "description": "API for advanced oil price forecasting with sentiment analysis",
        "endpoints": {
            "forecast": "/api/v1/forecast",
            "models": "/api/v1/models",
            "sentiment": "/api/v1/sentiment",
            "docs": "/docs"
        }
    }

@app.post("/api/v1/forecast", response_model=ForecastResponse, responses={
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def generate_forecast(
    request: ForecastRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate oil price forecast based on the specified parameters.
    
    This endpoint uses the Oil Prophet forecasting models to predict future oil prices.
    It can use different models including LSTM, sentiment-enhanced LSTM, baseline models,
    or an ensemble of models.
    """
    try:
        # Load price data
        price_data = load_price_data(request.oil_type, request.frequency)
        
        # Load sentiment data if needed
        sentiment_data = pd.DataFrame()  # Default empty
        if request.model in ["sentiment", "ensemble"]:
            sentiment_data = load_sentiment_data()
            if sentiment_data.empty:
                logger.warning("Sentiment data not available, falling back to LSTM model")
                if request.model == "sentiment":
                    request.model = "lstm"  # Fallback to LSTM if sentiment data not available
        
        # Prepare the dataset
        dataset = prepare_dataset(
            price_data,
            sentiment_data,
            window_size=30,  # Fixed for now
            forecast_horizon=request.forecast_horizon
        )
        
        # Generate forecast
        forecast = None
        if request.model == "ensemble":
            # Generate forecasts from multiple models and average them
            forecasts = []
            available_models = ["lstm", "baseline"]
            
            if not sentiment_data.empty:
                available_models.append("sentiment")
            
            for model_name in available_models:
                try:
                    model_forecast = generate_model_forecast(model_name, dataset, request.forecast_horizon)
                    forecasts.append(model_forecast)
                except Exception as e:
                    logger.warning(f"Error with {model_name} forecast, skipping: {str(e)}")
            
            if not forecasts:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate any forecasts for ensemble"
                )
            
            # Average the forecasts
            forecast = np.mean(forecasts, axis=0)
        else:
            # Generate forecast with the specified model
            forecast = generate_model_forecast(request.model, dataset, request.forecast_horizon)
        
        # Get the last date and current price
        last_date = price_data.index[-1]
        current_price = float(price_data['Price'].iloc[-1])
        
        # Generate forecast dates
        forecast_dates = get_forecast_dates(last_date, request.frequency, request.forecast_horizon)
        
        # Calculate confidence intervals
        price_volatility = price_data['Price'].pct_change().std() * current_price
        confidence_interval = generate_confidence_interval(forecast, price_volatility)
        
        # Prepare the data points for response
        data_points = []
        
        # Add historical data if requested
        if request.include_history and request.history_periods > 0:
            history_start = max(0, len(price_data) - request.history_periods)
            for date, row in price_data.iloc[history_start:].iterrows():
                data_points.append(ForecastPoint(
                    date=date.strftime("%Y-%m-%d"),
                    price=float(row['Price']),
                    is_forecast=False
                ))
        
        # Add forecast data points
        for date, price in zip(forecast_dates, forecast):
            data_points.append(ForecastPoint(
                date=date.strftime("%Y-%m-%d"),
                price=float(price),
                is_forecast=True
            ))
        
        # Calculate metrics from test data if available
        model_metrics = None
        if 'y_test' in dataset and len(dataset['y_test']) > 0:
            metrics_dict = calculate_model_metrics([request.model], dataset)
            model_metrics = metrics_dict.get(request.model)
        
        # Create the response
        response = ForecastResponse(
            oil_type=request.oil_type,
            frequency=request.frequency,
            model=request.model,
            forecast_start_date=forecast_dates[0].strftime("%Y-%m-%d"),
            forecast_end_date=forecast_dates[-1].strftime("%Y-%m-%d"),
            current_price=current_price,
            forecasted_end_price=float(forecast[-1]),
            price_change=float(forecast[-1] - current_price),
            price_change_percentage=float((forecast[-1] - current_price) / current_price * 100),
            data=data_points,
            confidence_interval=confidence_interval,
            model_metrics=model_metrics,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return response
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast: {str(e)}"
        )

@app.get("/api/v1/models/performance", response_model=PerformanceResponse, responses={
    500: {"model": ErrorResponse}
})
async def get_model_performance(
    oil_type: str = Query("brent", description="Oil type: 'brent' or 'wti'"),
    frequency: str = Query("daily", description="Data frequency: 'daily', 'weekly', or 'monthly'"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get performance metrics for all available forecasting models.
    
    This endpoint evaluates the performance of different models on the test dataset
    and returns comparative metrics.
    """
    try:
        # Load price data
        price_data = load_price_data(oil_type, frequency)
        
        # Load sentiment data
        sentiment_data = load_sentiment_data()
        
        # Prepare the dataset (using a fixed horizon for evaluation)
        dataset = prepare_dataset(
            price_data,
            sentiment_data,
            window_size=30,
            forecast_horizon=7  # Fixed for evaluation
        )
        
        # Check if test data is available
        if 'y_test' not in dataset or len(dataset['y_test']) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Test data not available for performance evaluation"
            )
        
        # Models to evaluate
        models_to_evaluate = ["baseline", "lstm"]
        if not sentiment_data.empty:
            models_to_evaluate.append("sentiment")
        
        # Calculate metrics
        metrics = calculate_model_metrics(models_to_evaluate, dataset)
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to calculate model metrics"
            )
        
        # Identify the baseline model
        baseline_metrics = metrics.get("baseline")
        if not baseline_metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Baseline model metrics not available"
            )
        
        # Create response objects
        performance_results = []
        
        for model_name, model_metrics in metrics.items():
            # Calculate improvement over baseline
            comparison = {}
            for metric, value in model_metrics.items():
                if metric in baseline_metrics:
                    if metric == "directional_accuracy":
                        # For directional accuracy, higher is better
                        comparison[metric] = value - baseline_metrics[metric]
                    else:
                        # For error metrics, lower is better
                        comparison[metric] = (1 - value / baseline_metrics[metric]) * 100
            
            performance_results.append(ModelPerformance(
                model=model_name,
                rmse=model_metrics.get("rmse", 0.0),
                mae=model_metrics.get("mae", 0.0),
                mape=model_metrics.get("mape", 0.0),
                directional_accuracy=model_metrics.get("directional_accuracy", 0.0),
                comparison_to_baseline=comparison
            ))
        
        # Determine best model based on RMSE
        best_model = min(
            [(model_name, metrics[model_name]["rmse"]) for model_name in metrics],
            key=lambda x: x[1]
        )[0]
        
        # Get test period dates
        test_start = dataset['dates_test'][0][0].strftime("%Y-%m-%d") if 'dates_test' in dataset else "N/A"
        test_end = dataset['dates_test'][0][-1].strftime("%Y-%m-%d") if 'dates_test' in dataset else "N/A"
        
        # Create response
        response = PerformanceResponse(
            models=performance_results,
            best_model=best_model,
            test_period=f"{test_start} to {test_end}",
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model performance: {str(e)}"
        )