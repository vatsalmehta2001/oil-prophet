#!/usr/bin/env python3
"""
Run Oil Market Sentiment Analysis Pipeline

This script runs the complete sentiment analysis pipeline for the Oil Prophet project.
It collects historical data, performs sentiment analysis, and integrates it with oil price data.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the project directory to the Python path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the pipeline module
from src.nlp.sentiment_analysis_pipeline import OilSentimentPipeline, run_historical_sentiment_analysis
from src.nlp.config_setup import check_reddit_config

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


def main():
    """Main function to run the sentiment analysis pipeline."""
    parser = argparse.ArgumentParser(description="Oil Market Sentiment Analysis Pipeline")
    parser.add_argument("--start-year", type=int, default=1987, help="Start year for data collection")
    parser.add_argument("--end-year", type=int, help="End year for data collection (defaults to current year)")
    parser.add_argument("--oil-type", type=str, default="brent", choices=["brent", "wti"], 
                      help="Type of oil data to use")
    parser.add_argument("--time-freq", type=str, default="D", choices=["D", "W", "M"], 
                      help="Time frequency (D=daily, W=weekly, M=monthly)")
    parser.add_argument("--window-size", type=int, default=30, 
                      help="Window size for feature creation")
    parser.add_argument("--forecast-horizon", type=int, default=7, 
                      help="Forecast horizon in days")
    parser.add_argument("--no-news", action="store_true", help="Skip news article collection")
    parser.add_argument("--no-visualizations", action="store_true", 
                      help="Skip generating visualizations")
    parser.add_argument("--output", type=str, 
                      default="data/processed/sentiment/historical_sentiment.csv", 
                      help="Path to save sentiment data")
    parser.add_argument("--models", type=str, default="finbert,finbert-tone", 
                      help="Comma-separated list of sentiment models to use")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited data")
    
    args = parser.parse_args()
    
    # Check Reddit config file
    check_reddit_config()
    
    # Parse models if provided
    models = [m.strip() for m in args.models.split(',')]
    
    # In test mode, use a smaller dataset
    if args.test:
        args.start_year = max(args.start_year, 2020)  # Use more recent data
        logger.info(f"Running in TEST MODE with limited data from {args.start_year}")
    
    logger.info(f"Starting sentiment analysis pipeline with models: {', '.join(models)}")
    
    # Initialize the pipeline
    pipeline = OilSentimentPipeline(
        output_dir="data/processed",
        models=models
    )
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        reddit_start_year=max(2008, args.start_year),  # Reddit data starts ~2008
        oil_type=args.oil_type,
        time_freq=args.time_freq,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        generate_visualizations=not args.no_visualizations
    )
    
    # Check results
    if results and 'aligned_data' in results and not results['aligned_data'].empty:
        # Save the final result
        if args.output:
            # Ensure output directory exists
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            results['aligned_data'].to_csv(args.output)
            logger.info(f"Saved aligned price and sentiment data to {args.output}")
        
        # Print summary statistics
        aligned_data = results['aligned_data']
        
        print("\n=== Sentiment Analysis Results ===")
        print(f"Time period: {aligned_data.index.min()} to {aligned_data.index.max()}")
        print(f"Total records: {len(aligned_data)}")
        
        if 'sentiment_compound' in aligned_data.columns:
            sentiment = aligned_data['sentiment_compound']
            print(f"\nSentiment Statistics:")
            print(f"  Mean: {sentiment.mean():.4f}")
            print(f"  Median: {sentiment.median():.4f}")
            print(f"  Min: {sentiment.min():.4f}")
            print(f"  Max: {sentiment.max():.4f}")
            
            # Calculate sentiment distribution
            positive = (sentiment > 0.05).mean() * 100
            neutral = ((sentiment >= -0.05) & (sentiment <= 0.05)).mean() * 100
            negative = (sentiment < -0.05).mean() * 100
            
            print(f"\nSentiment Distribution:")
            print(f"  Positive: {positive:.1f}%")
            print(f"  Neutral: {neutral:.1f}%")
            print(f"  Negative: {negative:.1f}%")
            
            # Print correlation with price if available
            if 'Price' in aligned_data.columns:
                price_corr = aligned_data['Price'].corr(sentiment)
                print(f"\nCorrelation with Price: {price_corr:.4f}")
                
                # Print lag correlations if significant
                try:
                    corr_df = pipeline.analyzer.calculate_correlation_with_price(
                        results['aggregated_sentiment'],
                        results['aligned_data'],
                        price_col='Price',
                        max_lag=10
                    )
                    
                    if not corr_df.empty:
                        # Find max correlation
                        max_corr_idx = corr_df['correlation'].abs().idxmax()
                        max_corr = corr_df.loc[max_corr_idx]
                        
                        print(f"Max correlation at lag {max_corr['lag']} days: {max_corr['correlation']:.4f}")
                        print(f"Direction: {max_corr['direction']}")
                except Exception as e:
                    logger.warning(f"Could not calculate lag correlations: {str(e)}")
        
        print("\nFeature data shapes:")
        if 'feature_data' in results and 'X' in results['feature_data']:
            x_shape = results['feature_data']['X'].shape
            y_shape = results['feature_data']['y'].shape
            print(f"  X: {x_shape}")
            print(f"  y: {y_shape}")
            
            # Print feature info
            if 'feature_info' in results['feature_data']:
                feature_info = results['feature_data']['feature_info']
                if 'features' in feature_info:
                    print(f"\nFeatures used: {len(feature_info['features'])}")
                    for i, feature in enumerate(feature_info['features'][:10]):  # Show only first 10
                        print(f"  {i+1}. {feature}")
                    if len(feature_info['features']) > 10:
                        print(f"  ... and {len(feature_info['features']) - 10} more")
        
        print("\nVisualization plots saved to: notebooks/plots/")
        print("=============================")
    else:
        logger.error("Pipeline did not return valid results")
        print("\n⚠️ ERROR: The sentiment analysis pipeline did not produce valid results.")
        print("Check the log file for more details: sentiment_pipeline.log")


if __name__ == "__main__":
    main()