#!/usr/bin/env python3
"""
Historical Reddit Data Scraper for Oil Market Sentiment

This script runs the Reddit historical scraper component of the Oil Prophet
sentiment analysis pipeline to collect data going back to the specified year.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the scraper module
from src.nlp.reddit_historical_scraper import HistoricalDataScraper
from src.nlp.config_setup import check_reddit_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reddit_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the Reddit scraper."""
    parser = argparse.ArgumentParser(description="Historical Reddit Data Scraper for Oil Market Sentiment")
    parser.add_argument("--start-year", type=int, default=2010, help="Start year for data collection")
    parser.add_argument("--end-year", type=int, help="End year for data collection (defaults to current year)")
    parser.add_argument("--output-dir", type=str, default="data/processed/historical", help="Output directory")
    parser.add_argument("--subreddits", type=str, help="Comma-separated list of subreddits to scrape")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum posts to collect")
    parser.add_argument("--skip-comments", action="store_true", help="Skip collecting comments")
    parser.add_argument("--skip-news", action="store_true", help="Skip collecting news articles")
    
    args = parser.parse_args()
    
    # Check Reddit config file
    check_reddit_config()
    
    # Set the end year if not provided
    if not args.end_year:
        args.end_year = datetime.now().year
    
    # Parse subreddits if provided
    subreddits = None
    if args.subreddits:
        subreddits = [s.strip() for s in args.subreddits.split(',')]

    # Initialize the scraper
    scraper = HistoricalDataScraper(output_dir=args.output_dir)
    
    # First option: run comprehensive dataset collection
    logger.info(f"Starting comprehensive data collection from {args.start_year} to {args.end_year}")
    
    dataset = scraper.create_comprehensive_dataset(
        start_year=args.start_year,
        end_year=args.end_year,
        output_file="oil_sentiment_dataset_complete.csv",
        include_reddit=True,
        include_news=not args.skip_news,
        include_financial_datasets=True
    )
    
    if not dataset.empty:
        logger.info(f"Successfully collected {len(dataset)} items")
        
        # Analyze and visualize the data coverage
        coverage_stats = scraper.analyze_data_coverage(dataset)
        logger.info(f"Data coverage: {coverage_stats}")
        
        scraper.visualize_data_coverage(
            dataset,
            output_file="oil_sentiment_coverage.png"
        )
    else:
        logger.warning("Comprehensive data collection failed or returned no data")
        
        # Fallback: try collecting Reddit data only
        logger.info("Trying to collect Reddit data only...")
        
        # Use default or provided subreddits
        if not subreddits:
            subreddits = scraper.DEFAULT_OIL_SUBREDDITS
            
        logger.info(f"Collecting Reddit data from subreddits: {', '.join(subreddits)}")
        
        reddit_data = scraper.fetch_historical_data_by_year(
            subreddits=subreddits,
            keywords=scraper.DEFAULT_OIL_KEYWORDS,
            start_year=args.start_year,
            end_year=args.end_year,
            include_comments=not args.skip_comments,
            save_yearly=True
        )
        
        if isinstance(reddit_data, dict):
            total_items = sum(len(df) for df in reddit_data.values())
            logger.info(f"Collected {total_items} items from Reddit")
        elif not reddit_data.empty:
            logger.info(f"Collected {len(reddit_data)} items from Reddit")
        else:
            logger.error("Failed to collect Reddit data")
    
    # Save statistics
    scraper.save_stats()
    
    logger.info("Reddit data collection complete")


if __name__ == "__main__":
    main()