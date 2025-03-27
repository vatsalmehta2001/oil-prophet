#!/usr/bin/env python3
"""
Oil Market Sentiment Analysis Pipeline

This script runs the complete sentiment analysis pipeline:
1. Retrieves Reddit data (using existing or new scraping)
2. Analyzes sentiment using FinBERT
3. Aggregates sentiment by time period
4. Creates visualizations of sentiment trends
5. Prepares sentiment features for forecasting

Usage:
    python run_sentiment_pipeline.py --data_dir=data/processed/reddit_test_small --analyze_only
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Add the project directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from src.nlp.finbert_sentiment import OilFinBERT
from src.nlp.reddit_historical_scraper import EnhancedRedditScraper
from src.data.preprocessing import OilDataProcessor

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


def analyze_sentiment_data(data_dir, output_dir=None, models=None):
    """
    Analyze sentiment in Reddit data.
    
    Args:
        data_dir: Directory containing Reddit data
        output_dir: Directory to save analyzed data (defaults to data_dir)
        models: List of FinBERT models to use (defaults to ["finbert"])
    
    Returns:
        DataFrame with sentiment analysis results
    """
    # Set defaults
    if output_dir is None:
        output_dir = data_dir
    
    if models is None:
        models = ["finbert"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for comprehensive dataset first
    comprehensive_file = os.path.join(data_dir, "comprehensive_sentiment_dataset.csv")
    
    if os.path.exists(comprehensive_file):
        logger.info(f"Loading comprehensive dataset from {comprehensive_file}")
        data = pd.read_csv(comprehensive_file)
    else:
        # Try to find reddit_combined_data.csv
        combined_file = os.path.join(data_dir, "reddit_combined_data.csv")
        
        if os.path.exists(combined_file):
            logger.info(f"Loading combined Reddit data from {combined_file}")
            data = pd.read_csv(combined_file)
        else:
            # Look for individual subreddit files and combine them
            logger.info(f"Looking for individual subreddit files in {data_dir}")
            post_files = [f for f in os.listdir(data_dir) if f.startswith("reddit_posts_")]
            comment_files = [f for f in os.listdir(data_dir) if f.startswith("reddit_comments_")]
            
            all_data = []
            
            # Load posts
            for file in post_files:
                file_path = os.path.join(data_dir, file)
                logger.info(f"Loading posts from {file}")
                df = pd.read_csv(file_path)
                all_data.append(df)
            
            # Load comments
            for file in comment_files:
                file_path = os.path.join(data_dir, file)
                logger.info(f"Loading comments from {file}")
                df = pd.read_csv(file_path)
                all_data.append(df)
            
            if not all_data:
                logger.error(f"No Reddit data found in {data_dir}")
                return None
            
            # Combine all data
            data = pd.concat(all_data, ignore_index=True)
            
            # Save combined data
            combined_output = os.path.join(output_dir, "reddit_combined_data.csv")
            data.to_csv(combined_output, index=False)
            logger.info(f"Saved combined data to {combined_output}")
    
    # Check if sentiment analysis is needed
    if 'sentiment_compound' in data.columns:
        logger.info("Sentiment analysis already present in data")
        return data
    
    # Initialize FinBERT analyzer
    logger.info(f"Initializing OilFinBERT with models: {', '.join(models)}")
    analyzer = OilFinBERT(models=models)
    
    # Analyze sentiment
    logger.info(f"Analyzing sentiment for {len(data)} items...")
    
    # Determine columns to use
    text_col = 'text' if 'text' in data.columns else 'body'
    title_col = 'title' if 'title' in data.columns else None
    
    # Check if this is Reddit data with posts and comments
    if 'type' in data.columns and set(data['type'].unique()).intersection({'post', 'comment'}):
        analyzed_data = analyzer.analyze_reddit_data(
            data,
            text_col=text_col,
            title_col=title_col,
            batch_size=32
        )
    else:
        # Generic text data
        analyzed_data = analyzer.analyze_text_batch(
            data,
            text_col=text_col,
            title_col=title_col,
            batch_size=32
        )
    
    # Save analyzed data
    output_file = os.path.join(output_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    analyzed_data.to_csv(output_file, index=False)
    logger.info(f"Saved analyzed data to {output_file}")
    
    # Calculate sentiment statistics
    sentiment_stats = analyzer.get_sentiment_distribution(analyzed_data)
    
    # Log sentiment statistics
    logger.info("Sentiment Distribution:")
    logger.info(f"  Mean: {sentiment_stats.get('mean', 0):.3f}")
    logger.info(f"  Positive: {sentiment_stats.get('positive_pct', 0):.1f}%")
    logger.info(f"  Neutral: {sentiment_stats.get('neutral_pct', 0):.1f}%")
    logger.info(f"  Negative: {sentiment_stats.get('negative_pct', 0):.1f}%")
    
    return analyzed_data


def create_sentiment_visualizations(sentiment_data, price_data=None, output_dir=None):
    """
    Create visualizations from sentiment data.
    
    Args:
        sentiment_data: DataFrame with sentiment analysis results
        price_data: Optional DataFrame with price data
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = "notebooks/plots"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize FinBERT for visualizations
    analyzer = OilFinBERT()
    
    # Create sentiment distribution plot
    logger.info("Creating sentiment distribution visualization...")
    dist_plot = analyzer.plot_sentiment_distribution(
        sentiment_data,
        title="Oil Market Sentiment Distribution"
    )
    
    dist_plot_path = os.path.join(output_dir, "sentiment_distribution.png")
    dist_plot.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved sentiment distribution plot to {dist_plot_path}")
    
    # Create sentiment over time plot
    logger.info("Creating sentiment over time visualization...")
    time_plot = analyzer.plot_sentiment_over_time(
        sentiment_data,
        date_col='created_date' if 'created_date' in sentiment_data.columns else None,
        time_freq='D',
        title="Oil Market Sentiment Over Time"
    )
    
    if time_plot is not None:
        time_plot_path = os.path.join(output_dir, "sentiment_over_time.png")
        time_plot.savefig(time_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sentiment over time plot to {time_plot_path}")
    
    # Create sentiment vs price plot if price data is available
    if price_data is not None:
        logger.info("Creating sentiment vs price visualization...")
        
        # Prepare price data
        if isinstance(price_data, str):
            # Load price data if a file path is provided
            price_processor = OilDataProcessor()
            price_df = price_processor.load_data(oil_type="brent", freq="daily")
            price_df = price_df.reset_index()
        else:
            price_df = price_data.copy()
            if 'Date' not in price_df.columns and price_df.index.name != 'Date':
                price_df = price_df.reset_index()
        
        # Create plot
        price_plot = analyzer.plot_sentiment_vs_price(
            sentiment_df=sentiment_data,
            price_df=price_df,
            date_col_sentiment='created_date' if 'created_date' in sentiment_data.columns else None,
            date_col_price='Date',
            price_col='Price',
            title='Oil Market Sentiment vs Price'
        )
        
        if price_plot is not None:
            price_plot_path = os.path.join(output_dir, "price_with_sentiment.png")
            price_plot.savefig(price_plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment vs price plot to {price_plot_path}")
    
    # Extract and visualize keywords
    logger.info("Extracting top keywords...")
    keywords = analyzer.extract_keywords(
        sentiment_data,
        text_col='text' if 'text' in sentiment_data.columns else 'body',
        title_col='title' if 'title' in sentiment_data.columns else None,
        n_keywords=30
    )
    
    keyword_plot = analyzer.plot_keyword_frequencies(
        keywords,
        title="Top Keywords in Oil Market Discussions"
    )
    
    keyword_plot_path = os.path.join(output_dir, "top_keywords.png")
    keyword_plot.savefig(keyword_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved keyword frequency plot to {keyword_plot_path}")


def scrape_reddit_data(output_dir=None, start_year=2015, end_year=None):
    """
    Scrape historical Reddit data for oil market sentiment analysis.
    
    Args:
        output_dir: Directory to save scraped data
        start_year: Start year for scraping
        end_year: End year for scraping (defaults to current year)
    
    Returns:
        DataFrame with scraped data
    """
    if output_dir is None:
        output_dir = "data/processed/reddit_scraped"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set end year to current year if not provided
    if end_year is None:
        end_year = datetime.now().year
    
    # Initialize scraper
    logger.info(f"Initializing Reddit scraper for {start_year}-{end_year}...")
    scraper = EnhancedRedditScraper(output_dir=output_dir)
    
    # Focus on finance and investing subreddits most relevant to oil
    subreddits = [
        "investing", "stocks", "stockmarket", "wallstreetbets",
        "economics", "economy", "finance", "oil", "energy"
    ]
    
    # Set oil-specific keywords
    keywords = [
        "oil", "crude oil", "petroleum", "brent crude", "wti crude", 
        "oil price", "oil market", "opec", "oil production", "oil demand",
        "oil supply", "barrel", "energy price"
    ]
    
    # Scrape Reddit data
    logger.info(f"Starting Reddit scraping from {start_year} to {end_year}...")
    logger.info(f"Target subreddits: {', '.join(subreddits)}")
    
    data = scraper.fetch_reddit_data(
        subreddits=subreddits,
        keywords=keywords,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        include_comments=True,
        filter_quality=True,
        min_score=3,
        min_relevance=0.3
    )
    
    if data.empty:
        logger.error("No data collected from Reddit")
        return None
    
    logger.info(f"Collected {len(data)} items from Reddit")
    
    # Create comprehensive dataset
    comprehensive_file = os.path.join(output_dir, "comprehensive_sentiment_dataset.csv")
    data.to_csv(comprehensive_file, index=False)
    logger.info(f"Saved comprehensive dataset to {comprehensive_file}")
    
    return data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Oil Market Sentiment Analysis Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/processed/reddit_test_small", 
                        help="Directory containing Reddit data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output (defaults to data_dir)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing data, don't scrape new data")
    parser.add_argument("--visualize_only", action="store_true",
                        help="Only create visualizations from analyzed data")
    parser.add_argument("--start_year", type=int, default=2015,
                        help="Start year for Reddit scraping")
    parser.add_argument("--end_year", type=int, default=None,
                        help="End year for Reddit scraping (defaults to current year)")
    parser.add_argument("--models", type=str, default="finbert",
                        help="Comma-separated list of FinBERT models to use")
    parser.add_argument("--oil_type", type=str, default="brent", choices=["brent", "wti"],
                        help="Oil type for price data")
    parser.add_argument("--freq", type=str, default="daily", choices=["daily", "weekly", "monthly"],
                        help="Frequency of price data")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_dir
    
    # Parse models
    models = args.models.split(",")
    
    # Load oil price data for visualizations
    logger.info(f"Loading {args.oil_type} {args.freq} price data...")
    processor = OilDataProcessor()
    try:
        price_data = processor.load_data(oil_type=args.oil_type, freq=args.freq)
    except Exception as e:
        logger.warning(f"Could not load price data: {str(e)}")
        price_data = None
    
    # Step 1: Scrape data if needed
    sentiment_data = None
    
    if not args.analyze_only and not args.visualize_only:
        logger.info("Scraping new data from Reddit...")
        sentiment_data = scrape_reddit_data(
            output_dir=output_dir,
            start_year=args.start_year,
            end_year=args.end_year
        )
    
    # Step 2: Analyze sentiment
    if not args.visualize_only:
        logger.info("Analyzing sentiment...")
        sentiment_data = analyze_sentiment_data(
            data_dir=args.data_dir,
            output_dir=output_dir,
            models=models
        )
    else:
        # Load analyzed data for visualization
        analyzed_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
        if os.path.exists(analyzed_file):
            logger.info(f"Loading analyzed data from {analyzed_file}")
            sentiment_data = pd.read_csv(analyzed_file)
        else:
            logger.error(f"No analyzed data found at {analyzed_file}")
            return
    
    # Step 3: Create visualizations
    if sentiment_data is not None:
        logger.info("Creating visualizations...")
        create_sentiment_visualizations(
            sentiment_data=sentiment_data,
            price_data=price_data,
            output_dir=os.path.join(output_dir, "..", "..", "notebooks", "plots")
        )
        
        logger.info("Sentiment analysis pipeline completed successfully!")
    else:
        logger.error("No sentiment data available for analysis")


if __name__ == "__main__":
    main()