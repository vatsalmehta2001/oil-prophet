"""
Oil Prophet API Client

This module provides a Python client for interacting with the Oil Prophet API.
It simplifies making requests to the API and handling the responses.
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any
from datetime import datetime, timedelta


class OilProphetClient:
    """
    Client for the Oil Prophet API that provides methods for forecasting,
    sentiment analysis, and model performance evaluation.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the client with the API base URL and API key.
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Handle empty response
            if not response.text:
                return {}
            
            # Parse JSON response
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            try:
                error_data = response.json()
                error_message = error_data.get('detail', str(e))
                raise ValueError(f"API error ({response.status_code}): {error_message}")
            except json.JSONDecodeError:
                raise ValueError(f"API error ({response.status_code}): {str(e)}")
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection error: {str(e)}")
        
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response.text}")
    
    def get_forecast(
        self,
        oil_type: str = "brent",
        frequency: str = "daily",
        model: str = "ensemble",
        forecast_horizon: int = 7,
        include_history: bool = True,
        history_periods: int = 30
    ) -> Dict:
        """
        Generate a forecast for oil prices.
        
        Args:
            oil_type: Oil type ('brent' or 'wti')
            frequency: Data frequency ('daily', 'weekly', or 'monthly')
            model: Model to use for forecasting ('lstm', 'sentiment', 'baseline', 'ensemble')
            forecast_horizon: Number of periods to forecast
            include_history: Whether to include historical data in response
            history_periods: Number of historical periods to include
            
        Returns:
            Dictionary with forecast data
        """
        data = {
            "oil_type": oil_type,
            "frequency": frequency,
            "model": model,
            "forecast_horizon": forecast_horizon,
            "include_history": include_history,
            "history_periods": history_periods
        }
        
        return self._make_request("POST", "/api/v1/forecast", data=data)
    
    def plot_forecast(self, forecast_data: Dict) -> plt.Figure:
        """
        Plot the forecast data.
        
        Args:
            forecast_data: Forecast data from get_forecast()
            
        Returns:
            Matplotlib figure object
        """
        # Extract data points
        data_points = forecast_data.get('data', [])
        
        if not data_points:
            raise ValueError("No data points in forecast")
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(data_points)
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Split historical and forecast data
        historical = df[df['is_forecast'] == False]
        forecast = df[df['is_forecast'] == True]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical['date'], historical['price'], 'b-', label='Historical', linewidth=2)
        
        # Plot forecast data
        ax.plot(forecast['date'], forecast['price'], 'r--', label=f"{forecast_data.get('model', 'Model')} Forecast", linewidth=2)
        
        # Add confidence interval if available
        if 'confidence_interval' in forecast_data and forecast_data['confidence_interval']:
            ci = forecast_data['confidence_interval']
            if 'upper' in ci and 'lower' in ci:
                ax.fill_between(
                    forecast['date'],
                    ci['lower'],
                    ci['upper'],
                    alpha=0.2,
                    color='r',
                    label='95% Confidence Interval'
                )
        
        # Format plot
        ax.set_title(f"{forecast_data.get('oil_type', 'Oil').upper()} Price Forecast ({forecast_data.get('frequency', 'daily')})")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format date axis
        fig.autofmt_xdate()
        
        # Add summary information
        if 'current_price' in forecast_data and 'forecasted_end_price' in forecast_data:
            current_price = forecast_data['current_price']
            end_price = forecast_data['forecasted_end_price']
            change = forecast_data.get('price_change', end_price - current_price)
            change_pct = forecast_data.get('price_change_percentage', (change / current_price) * 100 if current_price else 0)
            
            summary_text = (
                f"Current: ${current_price:.2f}\n"
                f"Forecast: ${end_price:.2f}\n"
                f"Change: ${change:.2f} ({change_pct:.1f}%)"
            )
            
            # Add text box with summary
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(
                0.02, 0.97, summary_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=props
            )
        
        plt.tight_layout()
        
        return fig
    
    def get_sentiment(self, timeframe: str = "30d", include_data: bool = False) -> Dict:
        """
        Get market sentiment analysis.
        
        Args:
            timeframe: Timeframe for sentiment analysis (e.g., '30d' for 30 days)
            include_data: Whether to include individual sentiment data points
            
        Returns:
            Dictionary with sentiment analysis results
        """
        params = {
            "timeframe": timeframe,
            "include_data": str(include_data).lower()
        }
        
        return self._make_request("GET", "/api/v1/sentiment", params=params)
    
    def plot_sentiment(self, sentiment_data: Dict) -> plt.Figure:
        """
        Plot sentiment metrics.
        
        Args:
            sentiment_data: Sentiment data from get_sentiment()
            
        Returns:
            Matplotlib figure object
        """
        metrics = sentiment_data.get('metrics', {})
        
        if not metrics:
            raise ValueError("No metrics in sentiment data")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart of sentiment percentages
        labels = ['Positive', 'Neutral', 'Negative']
        values = [
            metrics.get('positive_percentage', 0),
            metrics.get('neutral_percentage', 0),
            metrics.get('negative_percentage', 0)
        ]
        
        colors = ['green', 'gray', 'red']
        
        ax.bar(labels, values, color=colors, alpha=0.7)
        
        # Add data labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Format plot
        timeframe = sentiment_data.get('timeframe', '30d')
        period_start = sentiment_data.get('period_start', '')
        period_end = sentiment_data.get('period_end', '')
        
        ax.set_title(f"Oil Market Sentiment Distribution ({timeframe})\n{period_start} to {period_end}")
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, max(values) * 1.2)  # Add some space for labels
        
        # Add mean sentiment line
        mean = metrics.get('mean', 0)
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.3)
        
        # Add summary text
        summary_text = (
            f"Mean: {mean:.3f}\n"
            f"Current: {metrics.get('current', 0):.3f}\n"
            f"Momentum: {metrics.get('momentum', 0):.3f}\n"
            f"Volatility: {metrics.get('volatility', 0):.3f}"
        )
        
        # Add text box with summary
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(
            0.02, 0.97, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )
        
        plt.tight_layout()
        
        return fig
    
    def get_model_performance(self, oil_type: str = "brent", frequency: str = "daily") -> Dict:
        """
        Get performance metrics for forecasting models.
        
        Args:
            oil_type: Oil type ('brent' or 'wti')
            frequency: Data frequency ('daily', 'weekly', or 'monthly')
            
        Returns:
            Dictionary with model performance metrics
        """
        params = {
            "oil_type": oil_type,
            "frequency": frequency
        }
        
        return self._make_request("GET", "/api/v1/models/performance", params=params)
    
    def plot_model_performance(self, performance_data: Dict) -> plt.Figure:
        """
        Plot model performance comparison.
        
        Args:
            performance_data: Performance data from get_model_performance()
            
        Returns:
            Matplotlib figure object
        """
        models_data = performance_data.get('models', [])
        
        if not models_data:
            raise ValueError("No model data in performance results")
        
        # Extract model names and metrics
        model_names = [model['model'] for model in models_data]
        rmse_values = [model['rmse'] for model in models_data]
        mae_values = [model['mae'] for model in models_data]
        mape_values = [model['mape'] for model in models_data]
        accuracy_values = [model['directional_accuracy'] for model in models_data]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot RMSE
        axs[0, 0].bar(model_names, rmse_values, color='skyblue')
        axs[0, 0].set_title('RMSE (lower is better)')
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot MAE
        axs[0, 1].bar(model_names, mae_values, color='lightgreen')
        axs[0, 1].set_title('MAE (lower is better)')
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot MAPE
        axs[1, 0].bar(model_names, mape_values, color='salmon')
        axs[1, 0].set_title('MAPE % (lower is better)')
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot Directional Accuracy
        axs[1, 1].bar(model_names, accuracy_values, color='purple')
        axs[1, 1].set_title('Directional Accuracy % (higher is better)')
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        # Add text to highlight best model
        best_model = performance_data.get('best_model', '')
        test_period = performance_data.get('test_period', '')
        
        fig.suptitle(f"Model Performance Comparison\nBest Model: {best_model}\nTest Period: {test_period}")
        
        plt.tight_layout()
        
        return fig
    
    def list_models(self) -> List[Dict]:
        """
        List all available forecasting models.
        
        Returns:
            List of model information dictionaries
        """
        response = self._make_request("GET", "/api/v1/models")
        return response
    
    def run_pipeline(
        self,
        scrape_data: bool = False,
        analyze_sentiment: bool = False,
        train_models: bool = False,
        generate_forecast: bool = True,
        oil_type: str = "brent",
        freq: str = "daily",
        forecast_horizon: int = 7,
        data_dir: str = "data/processed/reddit_test_small"
    ) -> Dict:
        """
        Run the forecast pipeline in the background.
        
        Args:
            scrape_data: Whether to run data scraping stage
            analyze_sentiment: Whether to run sentiment analysis stage
            train_models: Whether to run model training stage
            generate_forecast: Whether to generate forecasts
            oil_type: Oil type ('brent' or 'wti')
            freq: Data frequency ('daily', 'weekly', or 'monthly')
            forecast_horizon: Number of periods to forecast
            data_dir: Data directory
            
        Returns:
            Dictionary with pipeline execution information
        """
        params = {
            "scrape_data": str(scrape_data).lower(),
            "analyze_sentiment": str(analyze_sentiment).lower(),
            "train_models": str(train_models).lower(),
            "generate_forecast": str(generate_forecast).lower(),
            "oil_type": oil_type,
            "freq": freq,
            "forecast_horizon": forecast_horizon,
            "data_dir": data_dir
        }
        
        return self._make_request("POST", "/api/v1/pipeline/run", params=params)
    
    def to_dataframe(self, forecast_data: Dict) -> pd.DataFrame:
        """
        Convert forecast data to a pandas DataFrame.
        
        Args:
            forecast_data: Forecast data from get_forecast()
            
        Returns:
            DataFrame with forecast data
        """
        data_points = forecast_data.get('data', [])
        
        if not data_points:
            raise ValueError("No data points in forecast")
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        return df