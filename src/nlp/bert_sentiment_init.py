"""
Initialization module for BERT sentiment analysis.

This module provides a simplified way to initialize and use the BERT sentiment analyzer.
"""

import os
import logging
from .bert_sentiment import OilMarketSentimentAnalyzer
from .config_setup import check_reddit_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def initialize_sentiment_analyzer(
    config_file: str = "reddit_config.json",
    model_name: str = "ProsusAI/finbert"
) -> OilMarketSentimentAnalyzer:
    """
    Initialize the BERT sentiment analyzer with Reddit credentials.
    
    Args:
        config_file: Path to the Reddit API config file
        model_name: Name of the BERT model to use
        
    Returns:
        Initialized OilMarketSentimentAnalyzer
    """
    # Make sure we have a valid config file
    check_reddit_config(config_file)
    
    # Initialize the analyzer
    analyzer = OilMarketSentimentAnalyzer(
        config_file=config_file,
        model_name=model_name
    )
    
    if analyzer.initialized:
        logger.info("BERT sentiment analyzer initialized successfully")
    else:
        logger.warning("BERT sentiment analyzer initialization incomplete")
    
    return analyzer

def run_sentiment_analysis_pipeline(
    output_dir: str = 'data/processed',
    visualize_dir: str = 'notebooks/plots',
    days_back: int = 30,
    config_file: str = "reddit_config.json",
    model_name: str = "ProsusAI/finbert"
) -> None:
    """
    Run the complete sentiment analysis pipeline.
    
    Args:
        output_dir: Directory to save processed data
        visualize_dir: Directory to save visualizations
        days_back: Number of days to look back for Reddit posts
        config_file: Path to the Reddit API config file
        model_name: Name of the BERT model to use
    """
    # Initialize the analyzer
    analyzer = initialize_sentiment_analyzer(config_file, model_name)
    
    if not analyzer.initialized:
        logger.error("Cannot run pipeline without initialized analyzer")
        return
    
    # Import the run_sentiment_analysis function from bert_sentiment
    from .bert_sentiment import run_sentiment_analysis
    
    # Run the analysis
    sentiment_data = run_sentiment_analysis(
        config_file=config_file,
        model_name=model_name,
        output_dir=output_dir,
        visualize_dir=visualize_dir,
        days_back=days_back
    )
    
    if sentiment_data is not None:
        logger.info("Sentiment analysis pipeline completed successfully")
        logger.info(f"Results saved to {output_dir} and {visualize_dir}")
    else:
        logger.error("Sentiment analysis pipeline failed")

if __name__ == "__main__":
    # Example usage
    print("Initializing BERT sentiment analyzer...")
    analyzer = initialize_sentiment_analyzer()
    
    print("\nTo run the full sentiment analysis pipeline:")
    print("from src.nlp.bert_sentiment_init import run_sentiment_analysis_pipeline")
    print("run_sentiment_analysis_pipeline()")