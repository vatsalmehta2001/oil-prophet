#!/usr/bin/env python3
"""
Oil Prophet: Complete Forecasting Pipeline CLI

This script provides a unified command-line interface to run all stages of the
Oil Prophet forecasting pipeline, including:
1. Data scraping from Reddit
2. Sentiment analysis
3. Model training and evaluation
4. Forecast generation and visualization

Usage Examples:
    # Run the complete pipeline with default settings
    python run_forecast_pipeline.py --run-all
    
    # Analyze sentiment on existing data only
    python run_forecast_pipeline.py --analyze-sentiment --data-dir data/processed/reddit_test_small
    
    # Train and evaluate models only using existing sentiment data
    python run_forecast_pipeline.py --train-models --data-dir data/processed/reddit_test_small
    
    # Generate forecasts using existing models
    python run_forecast_pipeline.py --generate-forecast --model-path models/sentiment_enhanced_lstm.h5
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("forecast_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Oil Prophet Forecasting Pipeline")
    
    # Pipeline stage selection
    parser.add_argument("--run-all", action="store_true", 
                        help="Run the complete pipeline from scraping to forecasting")
    parser.add_argument("--scrape-data", action="store_true", 
                        help="Scrape data from Reddit")
    parser.add_argument("--analyze-sentiment", action="store_true", 
                        help="Analyze sentiment in the data")
    parser.add_argument("--train-models", action="store_true", 
                        help="Train and evaluate forecasting models")
    parser.add_argument("--generate-forecast", action="store_true", 
                        help="Generate forecasts using trained models")
    
    # Data options
    parser.add_argument("--data-dir", type=str, default="data/processed/reddit_test_small",
                      help="Directory containing Reddit data")
    parser.add_argument("--oil-type", type=str, default="brent", choices=["brent", "wti"],
                      help="Oil type for price data")
    parser.add_argument("--freq", type=str, default="daily", choices=["daily", "weekly", "monthly"],
                      help="Frequency of price data")
    
    # Scraping options
    parser.add_argument("--start-year", type=int, default=2015,
                      help="Start year for Reddit scraping")
    parser.add_argument("--end-year", type=int, default=None,
                      help="End year for Reddit scraping (defaults to current year)")
    parser.add_argument("--subreddits", type=str, 
                      help="Comma-separated list of subreddits to scrape (optional)")
    
    # Sentiment analysis options
    parser.add_argument("--sentiment-models", type=str, default="finbert",
                      help="Comma-separated list of FinBERT models to use")
    
    # Model training options
    parser.add_argument("--window-size", type=int, default=30,
                      help="Window size for LSTM model")
    parser.add_argument("--forecast-horizon", type=int, default=7,
                      help="Number of days to forecast")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--model-path", type=str, default="models/sentiment_enhanced_lstm.h5",
                      help="Path to save/load model")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="notebooks/plots",
                      help="Directory to save output plots")
    
    args = parser.parse_args()
    
    # If no specific stage is selected, run everything
    if not (args.run_all or args.scrape_data or args.analyze_sentiment or 
            args.train_models or args.generate_forecast):
        args.run_all = True
    
    return args


def scrape_data(args):
    """Run Reddit data scraping stage."""
    from src.nlp.reddit_historical_scraper import EnhancedRedditScraper
    
    logger.info("Starting Reddit data scraping...")
    
    # Ensure output directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Parse subreddits if provided
    subreddits = None
    if args.subreddits:
        subreddits = [s.strip() for s in args.subreddits.split(',')]
    
    # Initialize scraper
    scraper = EnhancedRedditScraper(output_dir=args.data_dir)
    
    # Run scraping
    data = scraper.fetch_reddit_data(
        subreddits=subreddits,
        start_date=f"{args.start_year}-01-01",
        end_date=f"{args.end_year or datetime.now().year}-12-31",
        include_comments=True,
        min_relevance=0.3
    )
    
    # Save comprehensive dataset
    if not data.empty:
        output_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset.csv")
        data.to_csv(output_file, index=False)
        logger.info(f"Saved {len(data)} items to {output_file}")
        
        # Visualize data coverage
        scraper.visualize_data_coverage(
            data,
            output_file="coverage_visualization.png"
        )
        
        return data
    else:
        logger.error("No data collected from Reddit")
        return None


def analyze_sentiment(args, data=None):
    """Run sentiment analysis stage."""
    from src.nlp.finbert_sentiment import OilFinBERT, run_sentiment_analysis
    
    logger.info("Starting sentiment analysis...")
    
    # Parse sentiment models
    models = args.sentiment_models.split(',')
    
    # If data is not provided, load it from file
    if data is None:
        sentiment_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset.csv")
        
        if os.path.exists(sentiment_file):
            logger.info(f"Loading data from {sentiment_file}")
            data = pd.read_csv(sentiment_file)
        else:
            logger.error(f"Sentiment data file not found: {sentiment_file}")
            return None
    
    # Check if sentiment analysis is needed
    if 'sentiment_compound' in data.columns:
        logger.info("Sentiment analysis already present in data")
        return data
    
    # Run sentiment analysis
    analyzed_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    analyzed_data = run_sentiment_analysis(
        input_path=os.path.join(args.data_dir, "comprehensive_sentiment_dataset.csv"),
        output_path=analyzed_file,
        models=models,
        text_col='text',
        title_col='title' if 'title' in data.columns else None
    )
    
    # Create visualizations
    if not analyzed_data.empty:
        # Initialize analyzer
        analyzer = OilFinBERT()
        
        # Create visualization directory
        vis_dir = os.path.join(args.output_dir)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create sentiment distribution plot
        dist_plot = analyzer.plot_sentiment_distribution(
            analyzed_data,
            title="Oil Market Sentiment Distribution"
        )
        
        dist_plot.savefig(os.path.join(vis_dir, "sentiment_distribution.png"), dpi=300)
        
        # Create sentiment over time plot
        time_plot = analyzer.plot_sentiment_over_time(
            analyzed_data,
            date_col='created_date' if 'created_date' in analyzed_data.columns else None,
            title="Oil Market Sentiment Over Time"
        )
        
        if time_plot:
            time_plot.savefig(os.path.join(vis_dir, "sentiment_over_time.png"), dpi=300)
        
        logger.info(f"Sentiment analysis completed and visualizations saved to {vis_dir}")
        
    return analyzed_data


def train_models(args):
    """Train and evaluate forecasting models."""
    from src.data.preprocessing import OilDataProcessor
    from src.nlp.finbert_sentiment import OilFinBERT
    from src.models.lstm_attention import LSTMWithAttention
    from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
    
    logger.info("Starting model training and evaluation...")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load oil price data
    logger.info(f"Loading {args.oil_type} {args.freq} price data...")
    processor = OilDataProcessor()
    price_data = processor.load_data(oil_type=args.oil_type, freq=args.freq)
    
    # Load sentiment data
    analyzed_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    if os.path.exists(analyzed_file):
        logger.info(f"Loading analyzed sentiment data from {analyzed_file}")
        sentiment_data = pd.read_csv(analyzed_file)
    else:
        logger.error(f"Analyzed sentiment data not found: {analyzed_file}")
        logger.info("Run with --analyze-sentiment first or provide analyzed data.")
        return None
    
    # Prepare dataset
    logger.info("Preparing combined dataset...")
    try:
        dataset = prepare_sentiment_features(
            price_df=price_data,
            sentiment_df=sentiment_data,
            window_size=args.window_size,
            forecast_horizon=args.forecast_horizon
        )
        
        # Log dataset information
        logger.info(f"Dataset prepared with {len(dataset['X_price_train'])} training samples")
        logger.info(f"Price input shape: {dataset['X_price_train'].shape}")
        logger.info(f"Sentiment input shape: {dataset['X_sentiment_train'].shape}")
        logger.info(f"Target shape: {dataset['y_train'].shape}")
        
        # Create and train price-only model
        logger.info("Training price-only LSTM model...")
        price_model = LSTMWithAttention(
            input_shape=(dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2]),
            output_dim=dataset['y_train'].shape[1],
            lstm_units=[128, 64],
            dropout_rate=0.2,
            bidirectional=True
        )
        
        # Set a lower number of epochs and higher batch size for faster training
        price_history = price_model.fit(
            dataset['X_price_train'],
            dataset['y_train'],
            dataset['X_price_val'],
            dataset['y_val'],
            batch_size=64,
            epochs=30,  # Reduced for faster training
            patience=5,  # Reduced for faster training
            model_path='models/lstm_price_only.h5',
            verbose=1
        )
        
        # Create and train sentiment-enhanced model
        logger.info("Training sentiment-enhanced LSTM model...")
        sentiment_model = SentimentEnhancedLSTM(
            price_input_shape=(dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2]),
            sentiment_input_shape=(dataset['X_sentiment_train'].shape[1], dataset['X_sentiment_train'].shape[2]),
            output_dim=dataset['y_train'].shape[1],
            lstm_units=[128, 64],
            dropout_rate=0.2,
            bidirectional=True
        )
        
        sentiment_history = sentiment_model.fit(
            dataset['X_price_train'],
            dataset['X_sentiment_train'],
            dataset['y_train'],
            dataset['X_price_val'],
            dataset['X_sentiment_val'],
            dataset['y_val'],
            batch_size=64,
            epochs=30,  # Reduced for faster training
            patience=5,  # Reduced for faster training
            model_path='models/sentiment_enhanced_lstm.h5',
            verbose=1
        )
        
        # Evaluate models on test set
        logger.info("Evaluating models on test data...")
        price_metrics = price_model.evaluate(dataset['X_price_test'], dataset['y_test'])
        
        sentiment_metrics = sentiment_model.evaluate(
            dataset['X_price_test'],
            dataset['X_sentiment_test'],
            dataset['y_test']
        )
        
        # Save evaluation results
        evaluation_results = {
            'price_only_model': {
                'metrics': price_metrics,
                'history': price_history
            },
            'sentiment_enhanced_model': {
                'metrics': sentiment_metrics,
                'history': sentiment_history
            },
            'dataset_info': {
                'train_size': len(dataset['X_price_train']),
                'val_size': len(dataset['X_price_val']),
                'test_size': len(dataset['X_price_test']),
                'window_size': args.window_size,
                'forecast_horizon': args.forecast_horizon
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save evaluation results
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_path = os.path.join(args.output_dir, f"model_evaluation_{date_str}.json")
        
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4, default=str)
        
        logger.info(f"Evaluation results saved to {evaluation_path}")
        
        # Generate and save comparative visualization
        logger.info("Generating model comparison visualization...")
        comparison_fig = sentiment_model.compare_with_price_only_model(
            price_model,
            dataset['X_price_test'],
            dataset['X_sentiment_test'],
            dataset['y_test'],
            dates=[dates[-1] for dates in dataset['dates_test']],
            save_path=os.path.join(args.output_dir, 'model_comparison.png')
        )
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_forecast(args, models=None):
    """Generate forecasts using trained models."""
    from src.data.preprocessing import OilDataProcessor
    from src.models.lstm_attention import LSTMWithAttention
    from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
    
    logger.info("Generating forecasts...")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If models aren't provided, load them
    if models is None:
        # Check if model file exists
        if not os.path.exists(args.model_path):
            logger.error(f"Model file not found: {args.model_path}")
            logger.info("Run with --train-models first or provide a valid model path.")
            return None
        
        # Load the sentiment-enhanced model
        logger.info(f"Loading model from {args.model_path}")
        sentiment_model = SentimentEnhancedLSTM.load(args.model_path)
        
        # Load price data and sentiment data
        logger.info(f"Loading data for forecasting...")
        processor = OilDataProcessor()
        price_data = processor.load_data(oil_type=args.oil_type, freq=args.freq)
        
        analyzed_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
        if not os.path.exists(analyzed_file):
            logger.error(f"Analyzed sentiment data not found: {analyzed_file}")
            return None
        
        sentiment_data = pd.read_csv(analyzed_file)
        
        # Prepare dataset
        dataset = prepare_sentiment_features(
            price_df=price_data,
            sentiment_df=sentiment_data,
            window_size=args.window_size,
            forecast_horizon=args.forecast_horizon
        )
    else:
        # Use the provided models and dataset
        sentiment_model = models['sentiment_model']
        dataset = models['dataset']
    
    # Generate future forecast
    # For this demo, we'll use the most recent data point to generate a forecast
    
    # Get the most recent window of data
    X_price_recent = dataset['X_price_test'][-1:].copy()
    X_sentiment_recent = dataset['X_sentiment_test'][-1:].copy()
    
    # Generate prediction
    forecast = sentiment_model.predict(X_price_recent, X_sentiment_recent)[0]
    
    # Get the last date from the test set
    if 'dates_test' in dataset and dataset['dates_test']:
        last_date = dataset['dates_test'][-1][-1]
        # Generate forecast dates (next N days)
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, len(forecast)+1)]
    else:
        forecast_dates = [f"Day {i+1}" for i in range(len(forecast))]
    
    # Create forecast visualization
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, forecast, 'r-o', linewidth=2, label='Forecast')
    
    # Add last few actual prices for context
    if 'y_test' in dataset:
        last_actual = dataset['y_test'][-1]
        last_actual_dates = dataset['dates_test'][-1] if 'dates_test' in dataset else [f"Day -{i}" for i in range(len(last_actual), 0, -1)]
        plt.plot(last_actual_dates, last_actual, 'b-o', linewidth=2, label='Actual')
    
    plt.title(f"Oil Price Forecast ({args.forecast_horizon} days ahead)")
    plt.xlabel("Date")
    plt.ylabel(f"Predicted Price ({args.oil_type.upper()})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format dates on x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save forecast plot
    forecast_path = os.path.join(args.output_dir, "oil_price_forecast.png")
    plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved forecast visualization to {forecast_path}")
    
    # Save forecast data to CSV
    forecast_data = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted_Price': forecast
    })
    
    forecast_csv = os.path.join(args.output_dir, "oil_price_forecast.csv")
    forecast_data.to_csv(forecast_csv, index=False)
    logger.info(f"Saved forecast data to {forecast_csv}")
    
    # Print forecast summary
    print("\n" + "="*50)
    print(f"OIL PRICE FORECAST ({args.oil_type.upper()}, {args.forecast_horizon} days ahead)")
    print("="*50)
    for i, (date, price) in enumerate(zip(forecast_dates, forecast)):
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
        print(f"Day {i+1} ({date_str}): ${price:.2f}")
    print("="*50)
    
    return forecast_data


def main():
    """Main function to run the complete pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    print("\n" + "="*50)
    print("OIL PROPHET FORECASTING PIPELINE")
    print("="*50 + "\n")
    
    # Create output directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Track progress
    progress = {
        'data_scraped': False,
        'sentiment_analyzed': False,
        'models_trained': False,
        'forecast_generated': False
    }
    
    # Results storage
    results = {}
    
    # Run selected pipeline stages
    if args.run_all or args.scrape_data:
        logger.info("\n" + "-"*50)
        logger.info("STAGE 1: DATA SCRAPING")
        logger.info("-"*50)
        
        data = scrape_data(args)
        progress['data_scraped'] = data is not None
        results['data'] = data
    
    if args.run_all or args.analyze_sentiment:
        logger.info("\n" + "-"*50)
        logger.info("STAGE 2: SENTIMENT ANALYSIS")
        logger.info("-"*50)
        
        data = results.get('data') if 'data' in results else None
        analyzed_data = analyze_sentiment(args, data)
        progress['sentiment_analyzed'] = analyzed_data is not None
        results['analyzed_data'] = analyzed_data
    
    if args.run_all or args.train_models:
        logger.info("\n" + "-"*50)
        logger.info("STAGE 3: MODEL TRAINING AND EVALUATION")
        logger.info("-"*50)
        
        model_results = train_models(args)
        progress['models_trained'] = model_results is not None
        results['models'] = model_results
    
    if args.run_all or args.generate_forecast:
        logger.info("\n" + "-"*50)
        logger.info("STAGE 4: FORECAST GENERATION")
        logger.info("-"*50)
        
        models = results.get('models') if 'models' in results else None
        forecast = generate_forecast(args, models)
        progress['forecast_generated'] = forecast is not None
        results['forecast'] = forecast
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    for stage, status in progress.items():
        print(f"{stage.replace('_', ' ').title()}: {'✅ Completed' if status else '❌ Not run or failed'}")
    print("="*50)
    
    # Print locations of outputs
    if any(progress.values()):
        print("\nOutputs:")
        if progress['data_scraped']:
            print(f"- Scraped data: {args.data_dir}/comprehensive_sentiment_dataset.csv")
        if progress['sentiment_analyzed']:
            print(f"- Analyzed sentiment data: {args.data_dir}/comprehensive_sentiment_dataset_analyzed.csv")
        if progress['models_trained']:
            print(f"- Trained models: models/lstm_price_only.h5, {args.model_path}")
        if progress['forecast_generated']:
            print(f"- Forecast visualization: {args.output_dir}/oil_price_forecast.png")
            print(f"- Forecast data: {args.output_dir}/oil_price_forecast.csv")
        print(f"- Visualizations: {args.output_dir}")
        print(f"- Logs: forecast_pipeline.log")
        
    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()