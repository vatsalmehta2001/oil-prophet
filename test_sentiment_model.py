#!/usr/bin/env python3
"""
Test Script for Sentiment-Enhanced LSTM Model

This script loads pre-scraped Reddit data and oil price data, runs sentiment analysis
if needed, and tests the sentiment-enhanced LSTM model against a baseline price-only model.

Usage:
    python test_sentiment_model.py --data_dir=data/processed/reddit_test_small
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add the project directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from src.data.preprocessing import OilDataProcessor
from src.nlp.finbert_sentiment import OilFinBERT
from src.models.lstm_attention import LSTMWithAttention
from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_model_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Sentiment-Enhanced LSTM Model")
    parser.add_argument("--data_dir", type=str, default="data/processed/reddit_test_small", 
                        help="Directory containing Reddit data")
    parser.add_argument("--oil_type", type=str, default="brent", choices=["brent", "wti"],
                        help="Oil type for price data")
    parser.add_argument("--freq", type=str, default="daily", choices=["daily", "weekly", "monthly"],
                        help="Frequency of price data")
    parser.add_argument("--window_size", type=int, default=30,
                        help="Window size for LSTM model")
    parser.add_argument("--forecast_horizon", type=int, default=7,
                        help="Number of days to forecast")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="notebooks/plots",
                        help="Directory to save output plots")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load oil price data
    logger.info(f"Loading {args.oil_type} {args.freq} price data...")
    processor = OilDataProcessor()
    price_data = processor.load_data(oil_type=args.oil_type, freq=args.freq)
    
    # Load sentiment data
    sentiment_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset.csv")
    analyzed_sentiment_file = os.path.join(args.data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    if os.path.exists(analyzed_sentiment_file):
        logger.info(f"Loading pre-analyzed sentiment data from {analyzed_sentiment_file}")
        sentiment_data = pd.read_csv(analyzed_sentiment_file)
    elif os.path.exists(sentiment_file):
        logger.info(f"Loading sentiment data from {sentiment_file}")
        sentiment_data = pd.read_csv(sentiment_file)
        
        # Check if sentiment analysis needs to be performed
        if 'sentiment_compound' not in sentiment_data.columns:
            logger.info("Running sentiment analysis...")
            
            # Initialize FinBERT analyzer
            analyzer = OilFinBERT(models=["finbert"])
            
            # Analyze sentiment
            sentiment_data = analyzer.analyze_reddit_data(
                sentiment_data,
                text_col='text',
                title_col='title' if 'title' in sentiment_data.columns else None,
                batch_size=32
            )
            
            # Save analyzed data
            sentiment_data.to_csv(analyzed_sentiment_file, index=False)
            logger.info(f"Sentiment analysis complete and saved to {analyzed_sentiment_file}")
    else:
        logger.error(f"No sentiment data found in {args.data_dir}")
        sys.exit(1)
    
    # Prepare combined dataset
    logger.info("Preparing combined dataset...")
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
    
    # Create and train price-only LSTM model for comparison
    logger.info("Training price-only LSTM model for comparison...")
    price_only_model = LSTMWithAttention(
        input_shape=(dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2]),
        output_dim=dataset['y_train'].shape[1],
        lstm_units=[128, 64],
        dropout_rate=0.2,
        bidirectional=True
    )
    
    price_only_history = price_only_model.fit(
        dataset['X_price_train'],
        dataset['y_train'],
        dataset['X_price_val'],
        dataset['y_val'],
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.epochs // 5,
        model_path='models/lstm_price_only.h5'
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
        batch_size=args.batch_size,
        epochs=args.epochs,  # Fixed: removed extraneous bracket
        patience=args.epochs // 5,
        model_path='models/sentiment_enhanced_lstm.h5'
    )
    
    # Evaluate both models
    logger.info("Evaluating models...")
    price_only_metrics = price_only_model.evaluate(dataset['X_price_test'], dataset['y_test'])
    sentiment_metrics = sentiment_model.evaluate(
        dataset['X_price_test'], 
        dataset['X_sentiment_test'], 
        dataset['y_test']
    )
    
    # Print comparison of metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"{'Metric':<10} {'Price-Only':<15} {'Price+Sentiment':<15} {'Improvement':<10}")
    print("-"*50)
    
    for key in price_only_metrics:
        price_val = price_only_metrics[key]
        sent_val = sentiment_metrics[key]
        improvement = (1 - sent_val / price_val) * 100  # Negative is better for these metrics
        print(f"{key:<10} {price_val:<15.4f} {sent_val:<15.4f} {improvement:>+10.2f}%")
    
    print("="*50 + "\n")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    axes[0].plot(price_only_history['loss'], 'b-', label='Price-Only Training')
    axes[0].plot(price_only_history['val_loss'], 'b--', label='Price-Only Validation')
    axes[0].plot(sentiment_history['loss'], 'r-', label='Price+Sentiment Training')
    axes[0].plot(sentiment_history['val_loss'], 'r--', label='Price+Sentiment Validation')
    axes[0].set_title('Model Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(price_only_history['mae'], 'b-', label='Price-Only Training')
    axes[1].plot(price_only_history['val_mae'], 'b--', label='Price-Only Validation')
    axes[1].plot(sentiment_history['mae'], 'r-', label='Price+Sentiment Training')
    axes[1].plot(sentiment_history['val_mae'], 'r--', label='Price+Sentiment Validation')
    axes[1].set_title('Model MAE Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_plot_path = os.path.join(args.output_dir, "model_history_comparison.png")
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training history comparison to {history_plot_path}")
    
    # Plot prediction comparison
    logger.info("Generating prediction comparison plots...")
    comparison_plot = sentiment_model.compare_with_price_only_model(
        price_only_model,
        dataset['X_price_test'],
        dataset['X_sentiment_test'],
        dataset['y_test'],
        dates=[dates[-1] for dates in dataset['dates_test']],  # Fixed: This line might cause an error - 'dates' is not defined
        save_path=os.path.join(args.output_dir, "model_prediction_comparison.png")
    )
    
    # Plot individual model predictions
    price_only_model.plot_predictions(
        dataset['X_price_test'], 
        dataset['y_test'],
        n_samples=10,
        save_path=os.path.join(args.output_dir, "price_only_predictions.png")
    )
    
    sentiment_model.plot_predictions(
        dataset['X_price_test'],
        dataset['X_sentiment_test'],
        dataset['y_test'],
        n_samples=10,
        save_path=os.path.join(args.output_dir, "sentiment_enhanced_predictions.png")
    )
    
    # Create a visualization of sentiment vs price
    if 'sentiment_compound' in sentiment_data.columns:
        logger.info("Creating sentiment vs price visualization...")
        
        # Initialize FinBERT for visualization
        analyzer = OilFinBERT()
        
        # Create sentiment vs price plot
        sentiment_price_fig = analyzer.plot_sentiment_vs_price(
            sentiment_df=sentiment_data,
            price_df=price_data.reset_index(),
            date_col_sentiment='created_date' if 'created_date' in sentiment_data.columns else None,
            date_col_price='Date',
            price_col='Price',
            title=f'Oil Market Sentiment vs {args.oil_type.upper()} Price ({args.freq})'
        )
        
        sentiment_price_path = os.path.join(args.output_dir, "price_with_sentiment.png")
        sentiment_price_fig.savefig(sentiment_price_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sentiment vs price visualization to {sentiment_price_path}")
    
    logger.info("Test completed successfully!")
    print(f"\nAll visualizations saved to {args.output_dir}")
    print(f"Model evaluation metrics logged to sentiment_model_test.log")


if __name__ == "__main__":
    main()