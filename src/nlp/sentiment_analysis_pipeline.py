"""
Oil Market Sentiment Analysis Pipeline

This module provides a comprehensive pipeline for extracting, analyzing, and
integrating sentiment data with oil price forecasting models.

Features:
1. Historical data collection (back to 1987)
2. Multi-model ensemble sentiment analysis
3. Domain-specific oil market adaptation
4. Temporal alignment with price data
5. Enhanced feature creation
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local modules
from .finbert_sentiment import OilFinBERT
from .reddit_historical_scraper import HistoricalDataScraper
from .config_setup import check_reddit_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OilSentimentPipeline:
    """
    Pipeline for collecting and analyzing oil market sentiment data.
    
    This pipeline orchestrates the complete workflow:
    1. Historical data collection 
    2. Sentiment analysis
    3. Temporal aggregation and integration with price data
    4. Feature engineering for forecasting models
    """
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        cache_dir: str = "cache",
        models: List[str] = ["finbert", "finbert-tone"],
        device: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Initialize the sentiment pipeline.
        
        Args:
            output_dir: Directory for saving output data
            cache_dir: Directory for caching API responses
            models: List of sentiment models to use
            device: Device to run models on ('cpu' or 'cuda')
            batch_size: Batch size for processing
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/sentiment", exist_ok=True)
        
        # Initialize components
        logger.info("Initializing sentiment pipeline components")
        
        # Historical data scraper
        self.scraper = HistoricalDataScraper(
            output_dir=f"{output_dir}/historical",
            cache_dir=cache_dir
        )
        
        # Sentiment analyzer
        self.analyzer = OilFinBERT(
            models=models,
            device=device,
            domain_adaptation=True,
            batch_size=batch_size,
            oil_term_weighting=1.5
        )
        
        logger.info("Sentiment pipeline initialized")
    
    def run_complete_pipeline(
        self,
        start_year: int = 1987,
        end_year: Optional[int] = None,
        reddit_start_year: int = 2008,
        oil_type: str = "brent",
        time_freq: str = "D",
        window_size: int = 30,
        forecast_horizon: int = 7,
        generate_visualizations: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            start_year: Start year for historical data
            end_year: End year for historical data (defaults to current year)
            reddit_start_year: Year to start Reddit data collection (2008 default)
            oil_type: Type of oil price data ('brent' or 'wti')
            time_freq: Frequency for time aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
            window_size: Window size for feature creation
            forecast_horizon: Forecast horizon for target values
            generate_visualizations: Whether to generate visualization plots
            
        Returns:
            Dictionary with processed datasets
        """
        if not end_year:
            end_year = datetime.now().year
        
        logger.info(f"Running complete sentiment pipeline from {start_year} to {end_year}")
        
        # Step 1: Collect historical data
        logger.info("Step 1: Collecting historical data")
        historical_data = self._collect_historical_data(
            start_year=start_year,
            end_year=end_year,
            reddit_start_year=reddit_start_year
        )
        
        # Step 2: Analyze sentiment
        logger.info("Step 2: Analyzing sentiment")
        sentiment_results = self._analyze_sentiment(historical_data)
        
        # Step 3: Temporally aggregate sentiment
        logger.info("Step 3: Aggregating sentiment by time")
        aggregated_sentiment = self._aggregate_sentiment(
            sentiment_results,
            time_freq=time_freq
        )
        
        # Step 4: Align with price data
        logger.info("Step 4: Aligning with price data")
        price_data = self._load_price_data(oil_type)
        aligned_data = self._align_with_price_data(
            price_data,
            aggregated_sentiment
        )
        
        # Step 5: Create enhanced features
        logger.info("Step 5: Creating enhanced features")
        feature_data = self._create_enhanced_features(
            aligned_data,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        
        # Step 6: Generate visualizations
        if generate_visualizations:
            logger.info("Step 6: Generating visualizations")
            self._generate_visualizations(
                price_data, 
                aggregated_sentiment,
                aligned_data
            )
        
        # Return all processed datasets
        return {
            "historical_data": historical_data,
            "sentiment_results": sentiment_results,
            "aggregated_sentiment": aggregated_sentiment,
            "aligned_data": aligned_data,
            "feature_data": feature_data
        }
    
    def _collect_historical_data(
        self,
        start_year: int = 1987,
        end_year: int = None,
        reddit_start_year: int = 2008
    ) -> pd.DataFrame:
        """
        Collect historical sentiment data from multiple sources.
        
        Args:
            start_year: Start year for historical data
            end_year: End year for historical data
            reddit_start_year: Year to start Reddit data collection
            
        Returns:
            DataFrame with combined historical data
        """
        if not end_year:
            end_year = datetime.now().year
        
        # Collect comprehensive data
        historical_df = self.scraper.create_comprehensive_dataset(
            start_year=start_year,
            end_year=end_year,
            output_file="oil_sentiment_dataset_complete.csv",
            include_reddit=True,
            include_news=True,
            include_financial_datasets=True
        )
        
        # If we succeeded in collecting data
        if not historical_df.empty:
            # Save coverage visualization
            self.scraper.visualize_data_coverage(
                historical_df,
                output_file="sentiment_data_coverage.png"
            )
            
            # Get statistics on collected data
            coverage_stats = self.scraper.analyze_data_coverage(historical_df)
            coverage_path = os.path.join(self.output_dir, "sentiment/data_coverage_stats.json")
            
            try:
                with open(coverage_path, 'w') as f:
                    json.dump(coverage_stats, f, indent=4)
                logger.info(f"Saved data coverage statistics to {coverage_path}")
            except Exception as e:
                logger.error(f"Error saving coverage statistics: {str(e)}")
            
            logger.info(f"Collected {len(historical_df)} historical data points from {start_year} to {end_year}")
            return historical_df
        else:
            logger.warning("No historical data collected. Attempting to load existing data.")
            
            # Try to load existing data
            existing_path = os.path.join(self.output_dir, "historical/oil_sentiment_dataset_complete.csv")
            if os.path.exists(existing_path):
                try:
                    historical_df = pd.read_csv(existing_path)
                    logger.info(f"Loaded {len(historical_df)} historical data points from existing file")
                    return historical_df
                except Exception as e:
                    logger.error(f"Error loading existing data: {str(e)}")
            
            logger.error("Failed to collect or load historical data")
            return pd.DataFrame()
    
    def _analyze_sentiment(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in the dataset.
        
        Args:
            data_df: DataFrame with text data
            
        Returns:
            DataFrame with sentiment analysis results
        """
        if data_df.empty:
            logger.error("Empty DataFrame provided for sentiment analysis")
            return pd.DataFrame()
        
        # Check if sentiment analysis has already been run
        sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_compound']
        if all(col in data_df.columns for col in sentiment_cols):
            logger.info("Sentiment analysis already present in data")
            return data_df
        
        # Determine the appropriate columns based on data type
        title_col = 'title' if 'title' in data_df.columns else None
        
        # Check for content type column
        if 'source_type' in data_df.columns:
            # Process Reddit posts and comments separately
            processed_dfs = []
            
            # Posts (with titles)
            posts_df = data_df[data_df['source_type'] == 'post']
            if not posts_df.empty:
                posts_with_sentiment = self.analyzer.analyze_text_batch(
                    posts_df,
                    text_col='text',
                    title_col=title_col
                )
                processed_dfs.append(posts_with_sentiment)
            
            # Comments (no titles)
            comments_df = data_df[data_df['source_type'] == 'comment']
            if not comments_df.empty:
                comments_with_sentiment = self.analyzer.analyze_text_batch(
                    comments_df,
                    text_col='text',
                    title_col=None
                )
                processed_dfs.append(comments_with_sentiment)
            
            # News articles (with titles)
            news_df = data_df[data_df['source_type'] == 'news']
            if not news_df.empty:
                news_with_sentiment = self.analyzer.analyze_text_batch(
                    news_df,
                    text_col='text',
                    title_col=title_col
                )
                processed_dfs.append(news_with_sentiment)
            
            # Financial datasets (no titles)
            financial_df = data_df[data_df['source_type'] == 'financial-dataset']
            if not financial_df.empty:
                financial_with_sentiment = self.analyzer.analyze_text_batch(
                    financial_df,
                    text_col='text',
                    title_col=None
                )
                processed_dfs.append(financial_with_sentiment)
            
            # Other data types
            other_df = data_df[~data_df['source_type'].isin(['post', 'comment', 'news', 'financial-dataset'])]
            if not other_df.empty:
                other_with_sentiment = self.analyzer.analyze_text_batch(
                    other_df,
                    text_col='text',
                    title_col=title_col
                )
                processed_dfs.append(other_with_sentiment)
            
            # Combine all processed data
            result_df = pd.concat(processed_dfs, ignore_index=False)
            
            # Restore original order
            result_df = result_df.reindex(data_df.index)
        else:
            # Process all data at once
            result_df = self.analyzer.analyze_text_batch(
                data_df,
                text_col='text',
                title_col=title_col
            )
        
        # Save the results
        output_path = os.path.join(self.output_dir, "sentiment/sentiment_analysis_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved sentiment analysis results for {len(result_df)} items to {output_path}")
        
        # Get sentiment distribution statistics
        sentiment_stats = self.analyzer.get_sentiment_distribution(result_df)
        stats_path = os.path.join(self.output_dir, "sentiment/sentiment_distribution_stats.json")
        
        try:
            with open(stats_path, 'w') as f:
                json.dump(sentiment_stats, f, indent=4)
            logger.info(f"Saved sentiment distribution statistics to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving sentiment statistics: {str(e)}")
        
        return result_df
    
    def _aggregate_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        time_freq: str = 'D',
        min_count: int = 1
    ) -> pd.DataFrame:
        """
        Aggregate sentiment by time period.
        
        Args:
            sentiment_df: DataFrame with sentiment results
            time_freq: Time frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
            min_count: Minimum count of items to include in aggregation
            
        Returns:
            DataFrame with aggregated sentiment by time period
        """
        if sentiment_df.empty:
            logger.error("Empty DataFrame provided for sentiment aggregation")
            return pd.DataFrame()
        
        # Determine the date column
        date_col = None
        for col in ['created_date', 'datetime', 'date']:
            if col in sentiment_df.columns:
                date_col = col
                break
        
        if date_col is None and 'created_utc' in sentiment_df.columns:
            # Convert Unix timestamp to datetime
            sentiment_df['datetime'] = pd.to_datetime(sentiment_df['created_utc'], unit='s')
            date_col = 'datetime'
        
        if date_col is None:
            logger.error("No date column found for sentiment aggregation")
            return pd.DataFrame()
        
        # Aggregate sentiment by time
        agg_sentiment = self.analyzer.aggregate_sentiment_by_time(
            sentiment_df,
            date_col=date_col,
            time_freq=time_freq,
            min_count=min_count
        )
        
        if agg_sentiment.empty:
            logger.error("Failed to aggregate sentiment data")
            return pd.DataFrame()
        
        # Add sentiment indicators
        agg_sentiment = self._calculate_sentiment_indicators(agg_sentiment)
        
        # Save aggregated sentiment
        output_path = os.path.join(self.output_dir, f"sentiment/oil_market_sentiment_{time_freq}.csv")
        agg_sentiment.to_csv(output_path)
        logger.info(f"Saved aggregated sentiment data with {len(agg_sentiment)} records to {output_path}")
        
        return agg_sentiment
    
    def _calculate_sentiment_indicators(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional sentiment indicators.
        
        Args:
            sentiment_df: DataFrame with aggregated sentiment
            
        Returns:
            DataFrame with additional sentiment indicators
        """
        if sentiment_df.empty:
            return sentiment_df
        
        # Make a copy to avoid modifying the original
        result_df = sentiment_df.copy()
        
        # Calculate sentiment moving averages
        windows = [3, 7, 14, 30]
        for window in windows:
            if len(result_df) > window:
                result_df[f'sentiment_ma{window}'] = result_df['sentiment_compound'].rolling(window=window).mean()
        
        # Calculate sentiment momentum (rate of change)
        result_df['sentiment_momentum'] = result_df['sentiment_compound'].diff()
        
        # Calculate sentiment volatility
        result_df['sentiment_volatility'] = result_df['sentiment_compound'].rolling(window=7).std()
        
        # Calculate bullish/bearish bias
        result_df['bullish_bias'] = result_df['sentiment_positive'] / (result_df['sentiment_positive'] + result_df['sentiment_negative'])
        result_df['bearish_bias'] = result_df['sentiment_negative'] / (result_df['sentiment_positive'] + result_df['sentiment_negative'])
        
        # Handle division by zero
        result_df['bullish_bias'] = result_df['bullish_bias'].fillna(0.5)
        result_df['bearish_bias'] = result_df['bearish_bias'].fillna(0.5)
        
        # Calculate sentiment signal (simple threshold-based)
        result_df['sentiment_signal'] = 0
        result_df.loc[result_df['sentiment_compound'] > 0.2, 'sentiment_signal'] = 1  # Bullish
        result_df.loc[result_df['sentiment_compound'] < -0.2, 'sentiment_signal'] = -1  # Bearish
        
        return result_df
    
    def _load_price_data(self, oil_type: str = 'brent') -> pd.DataFrame:
        """
        Load oil price data.
        
        Args:
            oil_type: Type of oil price data ('brent' or 'wti')
            
        Returns:
            DataFrame with price data
        """
        try:
            # First try to use the OilDataProcessor if available
            try:
                from src.data.preprocessing import OilDataProcessor
                processor = OilDataProcessor()
                price_data = processor.load_data(oil_type=oil_type, freq="daily")
                logger.info(f"Loaded {len(price_data)} {oil_type} price records using OilDataProcessor")
                return price_data
            except (ImportError, ModuleNotFoundError):
                logger.warning("OilDataProcessor not available, falling back to direct file loading")
            
            # If OilDataProcessor is not available, load directly from file
            file_path = f"data/raw/{oil_type.lower()}-daily.csv"
            if not os.path.exists(file_path):
                logger.error(f"Price data file not found: {file_path}")
                return pd.DataFrame()
            
            # Load price data
            price_df = pd.read_csv(file_path)
            
            # Ensure Date column is datetime
            if 'Date' in price_df.columns:
                price_df['Date'] = pd.to_datetime(price_df['Date'])
                price_df = price_df.set_index('Date')
            
            logger.info(f"Loaded {len(price_df)} {oil_type} price records from file")
            return price_df
            
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}")
            return pd.DataFrame()
    
    def _align_with_price_data(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align sentiment data with price data.
        
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
        
        # Rename sentiment columns to avoid conflicts
        sentiment_cols = sentiment_df.columns.tolist()
        sentiment_df_renamed = sentiment_df.copy()
        for col in sentiment_cols:
            if col in price_df.columns:
                sentiment_df_renamed = sentiment_df_renamed.rename(columns={col: f"sentiment_{col}"})
        
        # Merge the dataframes on date index
        merged_df = price_df.join(sentiment_df_renamed, how='left')
        
        # Handle NaN values in sentiment columns
        sentiment_cols = [col for col in merged_df.columns if col.startswith('sentiment_')]
        
        # Fill missing sentiment values (forward fill first)
        merged_df[sentiment_cols] = merged_df[sentiment_cols].ffill()
        
        # Then backward fill any remaining NaNs
        merged_df[sentiment_cols] = merged_df[sentiment_cols].bfill()
        
        # For any remaining NaNs, fill with neutral sentiment (for early historical periods)
        if 'sentiment_compound' in merged_df.columns:
            merged_df['sentiment_compound'] = merged_df['sentiment_compound'].fillna(0)
        
        if 'sentiment_positive' in merged_df.columns:
            merged_df['sentiment_positive'] = merged_df['sentiment_positive'].fillna(0.33)
        
        if 'sentiment_negative' in merged_df.columns:
            merged_df['sentiment_negative'] = merged_df['sentiment_negative'].fillna(0.33)
        
        if 'sentiment_neutral' in merged_df.columns:
            merged_df['sentiment_neutral'] = merged_df['sentiment_neutral'].fillna(0.34)
        
        # Save the aligned data
        output_path = os.path.join(self.output_dir, "sentiment/price_with_sentiment.csv")
        merged_df.to_csv(output_path)
        logger.info(f"Saved aligned price and sentiment data with {len(merged_df)} records to {output_path}")
        
        return merged_df
    
    def _create_enhanced_features(
        self,
        aligned_df: pd.DataFrame,
        window_size: int = 30,
        forecast_horizon: int = 7
    ) -> Dict[str, np.ndarray]:
        """
        Create enhanced features for forecasting models.
        
        Args:
            aligned_df: DataFrame with aligned price and sentiment data
            window_size: Size of the sliding window
            forecast_horizon: Number of steps to predict
            
        Returns:
            Dictionary with feature arrays
        """
        if aligned_df.empty:
            logger.error("Empty DataFrame provided for feature creation")
            return {}
        
        # Ensure 'Price' column exists
        if 'Price' not in aligned_df.columns:
            logger.error("No 'Price' column found in aligned data")
            return {}
        
        # Select important sentiment features
        sentiment_features = []
        for col in aligned_df.columns:
            if col.startswith('sentiment_'):
                sentiment_features.append(col)
        
        if not sentiment_features:
            logger.error("No sentiment features found in aligned data")
            return {}
        
        # Create sliding windows with both price and sentiment features
        X, y = [], []
        feature_info = {'features': []}
        
        for i in range(len(aligned_df) - window_size - forecast_horizon + 1):
            # Get window of data
            window_df = aligned_df.iloc[i:i + window_size]
            
            # Create feature vector with price and sentiment
            feature_vector = []
            
            # Add price features
            price_features = window_df['Price'].values.reshape(-1, 1)
            feature_vector.append(price_features)
            if 'Price' not in feature_info['features']:
                feature_info['features'].append('Price')
            
            # Add sentiment features
            for feature in sentiment_features:
                if feature in window_df.columns:
                    sentiment_feature = window_df[feature].values.reshape(-1, 1)
                    feature_vector.append(sentiment_feature)
                    if feature not in feature_info['features']:
                        feature_info['features'].append(feature)
            
            # Combine all features
            combined_features = np.hstack(feature_vector)
            X.append(combined_features)
            
            # Get target values (future prices)
            target = aligned_df.iloc[i + window_size:i + window_size + forecast_horizon]['Price'].values
            y.append(target)
        
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Save the feature information
        feature_info['X_shape'] = X_array.shape
        feature_info['y_shape'] = y_array.shape
        feature_info['window_size'] = window_size
        feature_info['forecast_horizon'] = forecast_horizon
        
        info_path = os.path.join(self.output_dir, "sentiment/feature_info.json")
        try:
            with open(info_path, 'w') as f:
                json.dump(feature_info, f, indent=4)
            logger.info(f"Saved feature information to {info_path}")
        except Exception as e:
            logger.error(f"Error saving feature information: {str(e)}")
        
        # Save the feature arrays
        try:
            np.save(os.path.join(self.output_dir, "sentiment/X_features.npy"), X_array)
            np.save(os.path.join(self.output_dir, "sentiment/y_targets.npy"), y_array)
            logger.info(f"Saved feature arrays with shapes X: {X_array.shape}, y: {y_array.shape}")
        except Exception as e:
            logger.error(f"Error saving feature arrays: {str(e)}")
        
        return {
            'X': X_array,
            'y': y_array,
            'feature_info': feature_info
        }
    
    def _generate_visualizations(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        aligned_df: pd.DataFrame
    ) -> None:
        """
        Generate visualization plots.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            aligned_df: DataFrame with aligned data
        """
        try:
            # Create output directory for plots
            plots_dir = os.path.join("notebooks", "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Plot sentiment over time
            if not sentiment_df.empty:
                logger.info("Generating sentiment over time plot")
                fig = self.analyzer.plot_sentiment_over_time(
                    sentiment_df,
                    title="Oil Market Sentiment Over Time"
                )
                if fig:
                    fig.savefig(os.path.join(plots_dir, "sentiment_over_time.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # 2. Plot sentiment distribution
            if 'sentiment_compound' in aligned_df.columns:
                logger.info("Generating sentiment distribution plot")
                fig = self.analyzer.plot_sentiment_distribution(
                    aligned_df,
                    title="Oil Market Sentiment Distribution"
                )
                if fig:
                    fig.savefig(os.path.join(plots_dir, "sentiment_distribution.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # 3. Plot price with sentiment
            if not price_df.empty and not sentiment_df.empty:
                logger.info("Generating price with sentiment plot")
                
                # Focus on most recent 2 years for clarity
                end_date = price_df.index.max()
                if isinstance(end_date, pd.Timestamp):
                    start_date = end_date - pd.DateOffset(years=2)
                    recent_price_df = price_df.loc[start_date:end_date]
                else:
                    recent_price_df = price_df.iloc[-500:]  # Just take the last 500 days
                
                fig = self.plot_price_with_sentiment(
                    recent_price_df,
                    sentiment_df,
                    title="Oil Price with Market Sentiment (Last 2 Years)"
                )
                if fig:
                    fig.savefig(os.path.join(plots_dir, "price_with_sentiment.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # 4. Plot sentiment correlation with price
            if not price_df.empty and not sentiment_df.empty:
                logger.info("Generating sentiment-price correlation plot")
                corr_df = self.analyzer.calculate_correlation_with_price(
                    sentiment_df,
                    price_df,
                    date_col_price='Date' if 'Date' in price_df.columns else None,
                    price_col='Price',
                    max_lag=10
                )
                
                if not corr_df.empty:
                    fig = self.analyzer.plot_correlation_lags(
                        corr_df,
                        title="Correlation: Sentiment vs. Price at Different Lags"
                    )
                    if fig:
                        fig.savefig(os.path.join(plots_dir, "sentiment_price_correlation.png"), dpi=300, bbox_inches='tight')
                        plt.close(fig)
            
            logger.info("Visualization generation complete")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def plot_price_with_sentiment(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8),
        title: str = "Oil Price with Market Sentiment"
    ) -> plt.Figure:
        """
        Plot oil price with sentiment overlay.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # First, try using the analyzer's built-in method
        try:
            return self.analyzer.plot_sentiment_vs_price(
                sentiment_df,
                price_df,
                date_col_price='Date' if 'Date' in price_df.columns else None,
                price_col='Price',
                title=title
            )
        except Exception as e:
            logger.warning(f"Error using analyzer's plot method: {str(e)}")
        
        # If that fails, implement our own plotting logic
        try:
            # Align data
            aligned_df = self._align_with_price_data(price_df, sentiment_df)
            
            if aligned_df.empty:
                logger.error("Failed to create plot due to empty aligned data")
                return None
            
            # Create figure with two y-axes
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Plot price on primary y-axis
            ax1.plot(aligned_df.index, aligned_df['Price'], 'b-', linewidth=2, label='Oil Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price ($)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Create secondary y-axis for sentiment
            ax2 = ax1.twinx()
            
            # Plot sentiment on secondary y-axis
            sentiment_col = 'sentiment_compound' if 'sentiment_compound' in aligned_df.columns else next(
                (col for col in aligned_df.columns if 'sentiment' in col.lower()), None
            )
            
            if sentiment_col:
                ax2.plot(aligned_df.index, aligned_df[sentiment_col], 'r-', linewidth=1.5, label='Sentiment')
                ax2.set_ylabel('Sentiment Score', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                
                # Add moving average if available
                ma_col = 'sentiment_ma7'
                if ma_col in aligned_df.columns:
                    ax2.plot(aligned_df.index, aligned_df[ma_col], 'g--', linewidth=1.5, label='7-day MA')
            
            # Add a horizontal line at neutral sentiment (0)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            # Set title
            plt.title(title)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price with sentiment plot: {str(e)}")
            return None

    def generate_sentiment_features(
        self,
        start_year: int = 1987,
        end_year: Optional[int] = None,
        oil_type: str = 'brent',
        time_freq: str = 'D',
        window_size: int = 30,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate sentiment-enhanced features for a specific timeframe.
        
        This is a convenience function that runs key parts of the pipeline
        and returns sentiment-enhanced features suitable for model training.
        
        Args:
            start_year: Start year for data
            end_year: End year for data
            oil_type: Type of oil price data ('brent' or 'wti')
            time_freq: Time frequency for aggregation
            window_size: Window size for feature creation
            output_path: Optional path to save the features
            
        Returns:
            DataFrame with sentiment-enhanced features
        """
        # Run key steps of the pipeline
        historical_data = self._collect_historical_data(start_year, end_year)
        sentiment_results = self._analyze_sentiment(historical_data)
        aggregated_sentiment = self._aggregate_sentiment(sentiment_results, time_freq=time_freq)
        price_data = self._load_price_data(oil_type)
        aligned_data = self._align_with_price_data(price_data, aggregated_sentiment)
        
        # Create enhanced feature set
        features_df = aligned_data.copy()
        
        # Add technical indicators (if price data available)
        if 'Price' in features_df.columns:
            # Add price-based indicators
            features_df['price_ma7'] = features_df['Price'].rolling(window=7).mean()
            features_df['price_ma30'] = features_df['Price'].rolling(window=30).mean()
            features_df['price_volatility'] = features_df['Price'].rolling(window=14).std()
            features_df['price_momentum'] = features_df['Price'].pct_change(periods=7)
            
            # Add simple price-sentiment interaction features
            if 'sentiment_compound' in features_df.columns:
                features_df['price_sentiment_interaction'] = features_df['Price'] * features_df['sentiment_compound']
                features_df['price_change_sentiment_align'] = np.sign(features_df['Price'].pct_change()) * np.sign(features_df['sentiment_compound'])
        
        # Fill NaN values
        features_df = features_df.dropna()
        
        # Save features if path provided
        if output_path:
            features_df.to_csv(output_path)
            logger.info(f"Saved sentiment-enhanced features to {output_path}")
        
        return features_df


def run_historical_sentiment_analysis(
    start_year: int = 1987,
    end_year: int = None,
    subreddits: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    include_news: bool = True,
    use_finbert: bool = True,
    output_path: str = "data/processed/sentiment/historical_sentiment.csv"
):
    """
    Run a complete historical sentiment analysis workflow.
    
    This is a standalone function to run the entire pipeline with defaults.
    
    Args:
        start_year: Start year for data collection
        end_year: End year for data collection (defaults to current year)
        subreddits: List of subreddits to scrape (defaults to predefined list)
        keywords: List of keywords to search (defaults to predefined list)
        include_news: Whether to include news articles
        use_finbert: Whether to use FinBERT for sentiment (vs standard BERT)
        output_path: Path to save the resulting sentiment data
        
    Returns:
        DataFrame with historical sentiment data
    """
    # Initialize the pipeline
    pipeline = OilSentimentPipeline(
        output_dir="data/processed",
        models=["finbert", "finbert-tone"] if use_finbert else ["ProsusAI/finbert"]
    )
    
    # Set up Reddit config
    check_reddit_config()
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline(
        start_year=start_year,
        end_year=end_year,
        reddit_start_year=max(2008, start_year),  # Reddit data starts ~2008
        generate_visualizations=True
    )
    
    # Return the aligned data (price with sentiment)
    if results and 'aligned_data' in results and not results['aligned_data'].empty:
        # Save the final result
        results['aligned_data'].to_csv(output_path)
        logger.info(f"Saved historical sentiment analysis results to {output_path}")
        return results['aligned_data']
    
    logger.warning("Historical sentiment analysis did not produce valid results")
    return pd.DataFrame()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Oil Market Sentiment Analysis Pipeline")
    parser.add_argument("--start-year", type=int, default=2010, help="Start year for data collection")
    parser.add_argument("--end-year", type=int, help="End year for data collection (defaults to current year)")
    parser.add_argument("--oil-type", type=str, default="brent", choices=["brent", "wti"], help="Type of oil data to use")
    parser.add_argument("--time-freq", type=str, default="D", choices=["D", "W", "M"], 
                        help="Time frequency (D=daily, W=weekly, M=monthly)")
    parser.add_argument("--no-news", action="store_true", help="Skip news article collection")
    parser.add_argument("--no-visualizations", action="store_true", help="Skip generating visualizations")
    parser.add_argument("--output", type=str, default="data/processed/sentiment/historical_sentiment.csv", 
                        help="Path to save sentiment data")
    
    args = parser.parse_args()
    
    # Run the sentiment analysis pipeline
    run_historical_sentiment_analysis(
        start_year=args.start_year,
        end_year=args.end_year,
        include_news=not args.no_news,
        use_finbert=True,
        output_path=args.output
    )