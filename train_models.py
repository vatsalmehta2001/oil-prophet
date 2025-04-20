#!/usr/bin/env python3
"""
Oil Prophet Model Training Script

This script focuses exclusively on training and optimizing the forecasting models
without any demo or GUI aspects. It includes:

1. Data loading and preprocessing
2. Model architecture optimization
3. Hyperparameter tuning
4. Comprehensive model evaluation
5. Model persistence

Usage:
    python train_models.py --oil-type brent --freq daily --window 30 --horizon 7
"""

import os
import logging
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional, Concatenate, Lambda
from tensorflow.keras.models import Model

# Import project modules
from src.data.preprocessing import OilDataProcessor
from src.models.lstm_attention import LSTMWithAttention, AttentionLayer
from src.models.sentiment_enhanced_lstm import SentimentEnhancedLSTM, prepare_sentiment_features
from src.nlp.finbert_sentiment import OilFinBERT
from src.evaluation.metrics import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(oil_type, freq, data_dir):
    """Load and prepare price and sentiment data"""
    # Load price data
    logger.info(f"Loading {oil_type} {freq} price data...")
    processor = OilDataProcessor()
    price_data = processor.load_data(oil_type=oil_type, freq=freq)
    
    if price_data.empty:
        logger.error("Failed to load price data")
        return None, None
    
    # Load sentiment data
    analyzed_file = os.path.join(data_dir, "comprehensive_sentiment_dataset_analyzed.csv")
    
    if os.path.exists(analyzed_file):
        logger.info(f"Loading analyzed sentiment data from {analyzed_file}")
        sentiment_data = pd.read_csv(analyzed_file)
    else:
        logger.error(f"Analyzed sentiment data not found: {analyzed_file}")
        return price_data, None
    
    return price_data, sentiment_data

def prepare_datasets(price_data, sentiment_data, window_size, forecast_horizon):
    """Prepare datasets for model training"""
    logger.info("Preparing datasets for model training...")
    
    try:
        # Prepare combined dataset with sentiment
        if sentiment_data is not None:
            dataset = prepare_sentiment_features(
                price_df=price_data,
                sentiment_df=sentiment_data,
                window_size=window_size,
                forecast_horizon=forecast_horizon
            )
            
            logger.info(f"Dataset prepared with {len(dataset['X_price_train'])} training samples")
            logger.info(f"Price input shape: {dataset['X_price_train'].shape}")
            logger.info(f"Sentiment input shape: {dataset['X_sentiment_train'].shape}")
            logger.info(f"Target shape: {dataset['y_train'].shape}")
            
            return dataset
        else:
            # Fallback to price-only dataset
            logger.info("Sentiment data not available, creating price-only dataset")
            dataset = processor.prepare_dataset(
                oil_type=oil_type,
                freq=freq,
                window_size=window_size,
                forecast_horizon=forecast_horizon
            )
            
            # Convert to expected structure
            return {
                'X_price_train': dataset['X_train'],
                'X_price_val': dataset['X_val'],
                'X_price_test': dataset['X_test'],
                'y_train': dataset['y_train'],
                'y_val': dataset['y_val'],
                'y_test': dataset['y_test'],
                'dates_test': dataset.get('dates_test', None)
            }
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def train_price_only_model(dataset, model_params, training_params):
    """Train and save price-only LSTM model"""
    logger.info("Training price-only LSTM model...")
    
    # Create model with specified parameters
    model = LSTMWithAttention(
        input_shape=(dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2]),
        output_dim=dataset['y_train'].shape[1],
        lstm_units=model_params['lstm_units'],
        dropout_rate=model_params['dropout_rate'],
        learning_rate=model_params['learning_rate'],
        bidirectional=model_params['bidirectional']
    )
    
    # Train model
    history = model.fit(
        dataset['X_price_train'],
        dataset['y_train'],
        dataset['X_price_val'],
        dataset['y_val'],
        batch_size=training_params['batch_size'],
        epochs=training_params['epochs'],
        patience=training_params['patience'],
        model_path=training_params['model_path'],
        verbose=training_params['verbose']
    )
    
    # Evaluate model
    metrics = model.evaluate(dataset['X_price_test'], dataset['y_test'])
    logger.info(f"Price-only model evaluation: {metrics}")
    
    # Save model metadata
    metadata_path = f"{os.path.splitext(training_params['model_path'])[0]}_metadata.json"
    model.save(training_params['model_path'], metadata_path=metadata_path)
    
    return model, history, metrics

def train_sentiment_model(dataset, model_params, training_params):
    """Train and save sentiment-enhanced LSTM model"""
    logger.info("Training sentiment-enhanced LSTM model...")
    
    # Set up input shapes and dimensions
    price_input_shape = (dataset['X_price_train'].shape[1], dataset['X_price_train'].shape[2])
    sentiment_input_shape = (dataset['X_sentiment_train'].shape[1], dataset['X_sentiment_train'].shape[2])
    output_dim = dataset['y_train'].shape[1]
    lstm_units = model_params['lstm_units']
    dropout_rate = model_params['dropout_rate']
    learning_rate = model_params['learning_rate']
    bidirectional = model_params['bidirectional']
    price_weight = model_params['price_weight']
    sentiment_weight = model_params['sentiment_weight']
    
    # Build a simple sentiment-enhanced model using Functional API
    # Price branch
    price_input = Input(shape=price_input_shape, name='price_input')
    x_price = price_input
    
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        
        if bidirectional:
            x_price = Bidirectional(LSTM(units, return_sequences=return_sequences))(x_price)
        else:
            x_price = LSTM(units, return_sequences=return_sequences)(x_price)
        
        x_price = Dropout(dropout_rate)(x_price)
    
    # Sentiment branch  
    sentiment_input = Input(shape=sentiment_input_shape, name='sentiment_input')
    x_sentiment = sentiment_input
    
    # Process sentiment with LSTM
    if bidirectional:
        x_sentiment = Bidirectional(LSTM(32, return_sequences=False))(x_sentiment)
    else:
        x_sentiment = LSTM(32, return_sequences=False)(x_sentiment)
    
    x_sentiment = Dropout(dropout_rate)(x_sentiment)
    
    # Apply weights
    x_price_weighted = Lambda(lambda x: x * price_weight)(x_price)
    x_sentiment_weighted = Lambda(lambda x: x * sentiment_weight)(x_sentiment)
    
    # Concatenate features
    combined = Concatenate()([x_price_weighted, x_sentiment_weighted])
    
    # Final dense layers
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)
    
    # Output layer
    output = Dense(output_dim, activation='linear')(combined)
    
    # Create and compile model
    model = Model(inputs=[price_input, sentiment_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Built Sentiment-Enhanced LSTM model with {len(lstm_units)} LSTM layers")
    logger.info(f"Price weight: {price_weight}, Sentiment weight: {sentiment_weight}")
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(training_params['model_path']), exist_ok=True)
    
    # Prepare callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_params['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            training_params['model_path'],
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=training_params['patience'] // 2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    logger.info(f"Starting model training with {len(dataset['X_price_train'])} samples")
    history = model.fit(
        [dataset['X_price_train'], dataset['X_sentiment_train']], 
        dataset['y_train'],
        validation_data=([dataset['X_price_val'], dataset['X_sentiment_val']], dataset['y_val']),
        batch_size=training_params['batch_size'],
        epochs=training_params['epochs'],
        callbacks=callbacks,
        verbose=training_params['verbose']
    )
    
    # Evaluate model
    metrics = model.evaluate(
        [dataset['X_price_test'], dataset['X_sentiment_test']], 
        dataset['y_test'],
        verbose=0
    )
    
    loss, mae = metrics
    y_pred = model.predict([dataset['X_price_test'], dataset['X_sentiment_test']])
    mse = np.mean(np.square(dataset['y_test'] - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((dataset['y_test'] - y_pred) / np.maximum(np.abs(dataset['y_test']), 1e-7))) * 100
    
    evaluation_metrics = {
        'loss': float(loss),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape)
    }
    
    logger.info(f"Sentiment-enhanced model evaluation: {evaluation_metrics}")
    
    # Save model metadata
    metadata_path = f"{os.path.splitext(training_params['model_path'])[0]}_metadata.json"
    metadata = {
        'price_input_shape': price_input_shape,
        'sentiment_input_shape': sentiment_input_shape,
        'output_dim': output_dim,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'bidirectional': bidirectional,
        'price_weight': price_weight,
        'sentiment_weight': sentiment_weight,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    
    return model, history.history, evaluation_metrics

def save_evaluation_results(price_model_results, sentiment_model_results, dataset_info, output_dir):
    """Save comprehensive evaluation results"""
    logger.info("Saving evaluation results...")
    
    # Extract data
    price_model, price_history, price_metrics = price_model_results
    sentiment_model, sentiment_history, sentiment_metrics = sentiment_model_results
    
    # Create evaluation results dictionary
    evaluation_results = {
        'price_only_model': {
            'metrics': price_metrics,
            'history': price_history
        },
        'sentiment_enhanced_model': {
            'metrics': sentiment_metrics,
            'history': sentiment_history
        },
        'dataset_info': dataset_info,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add improvement metrics
    improvement = {}
    for key in price_metrics:
        if key in sentiment_metrics and price_metrics[key] > 0:
            imp_pct = (1 - sentiment_metrics[key] / price_metrics[key]) * 100
            improvement[key] = f"{imp_pct:+.2f}%"
    
    evaluation_results['improvement'] = improvement
    
    # Save to file
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_path = os.path.join(output_dir, f"model_evaluation_{date_str}.json")
    
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4, default=str)
    
    logger.info(f"Evaluation results saved to {evaluation_path}")
    
    # Generate and save comparison visualization
    try:
        logger.info("Generating model comparison visualization...")
        output_path = os.path.join(output_dir, f"model_comparison_{date_str}.png")
        
        # Generate predictions
        X_price_test = dataset_info['X_price_test']
        X_sentiment_test = dataset_info['X_sentiment_test']
        y_test = dataset_info['y_test']
        
        # Get predictions from both models
        price_only_pred = price_model.predict(X_price_test)
        sentiment_pred = sentiment_model.predict([X_price_test, X_sentiment_test])
        
        # Create figure with subplots for visualization
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        
        # Use dates if provided, otherwise use indices
        if dataset_info.get('dates_test') is not None:
            time_index = [dates[-1] for dates in dataset_info['dates_test']]
        else:
            time_index = np.arange(y_test.shape[1])
        
        # Plot first 5 samples
        for i in range(5):
            ax = axes[i]
            
            # Plot actual values
            ax.plot(time_index[:y_test.shape[1]], y_test[i], 'k-', linewidth=2, label='Actual')
            
            # Plot price-only model predictions
            ax.plot(time_index[:price_only_pred.shape[1]], price_only_pred[i], 'b--', linewidth=1.5, label='Price-Only')
            
            # Plot combined model predictions
            ax.plot(time_index[:sentiment_pred.shape[1]], sentiment_pred[i], 'r--', linewidth=1.5, label='Price+Sentiment')
            
            # Add legend for first subplot only
            if i == 0:
                ax.legend()
            
            # Add y-label showing the sample index
            ax.set_ylabel(f'Sample {i+1}')
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        # Add common labels
        fig.text(0.5, 0.02, 'Time', ha='center', va='center', fontsize=12)
        fig.text(0.02, 0.5, 'Price', ha='center', va='center', rotation='vertical', fontsize=12)
        
        # Add title
        fig.suptitle('Model Comparison: Price-Only vs Price+Sentiment', fontsize=16)
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function for model training"""
    parser = argparse.ArgumentParser(description="Train Oil Prophet forecasting models")
    
    # Data parameters
    parser.add_argument("--oil-type", type=str, default="brent", choices=["brent", "wti"], 
                        help="Type of oil price data to use")
    parser.add_argument("--freq", type=str, default="daily", 
                        choices=["daily", "weekly", "monthly"], 
                        help="Frequency of price data")
    parser.add_argument("--data-dir", type=str, default="data/processed/reddit_test_small", 
                        help="Directory containing sentiment data")
    
    # Model parameters
    parser.add_argument("--window", type=int, default=30, 
                        help="Window size for input sequences")
    parser.add_argument("--horizon", type=int, default=7, 
                        help="Forecast horizon (prediction length)")
    parser.add_argument("--lstm-units", type=str, default="128,64", 
                        help="Comma-separated list of LSTM layer units")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout rate for regularization")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate for optimization")
    parser.add_argument("--price-weight", type=float, default=0.7, 
                        help="Weight for price features (sentiment model)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Maximum number of epochs for training")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--output-dir", type=str, default="models", 
                        help="Directory to save models and results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    price_data, sentiment_data = load_data(args.oil_type, args.freq, args.data_dir)
    
    if price_data is None:
        logger.error("Failed to load price data. Exiting.")
        return
    
    # Prepare dataset
    dataset = prepare_datasets(
        price_data, 
        sentiment_data, 
        args.window, 
        args.horizon
    )
    
    if dataset is None:
        logger.error("Failed to prepare dataset. Exiting.")
        return
    
    # Parse LSTM units
    lstm_units = [int(units) for units in args.lstm_units.split(",")]
    
    # Set up model parameters
    model_params = {
        'lstm_units': lstm_units,
        'dropout_rate': args.dropout,
        'learning_rate': args.learning_rate,
        'bidirectional': True,
        'sentiment_dense_units': [32],
        'price_weight': args.price_weight,
        'sentiment_weight': 1.0 - args.price_weight
    }
    
    # Set up training parameters for price-only model
    price_training_params = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'model_path': os.path.join(args.output_dir, "lstm_price_only.h5"),
        'verbose': 1
    }
    
    # Set up training parameters for sentiment-enhanced model
    sentiment_training_params = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'model_path': os.path.join(args.output_dir, "sentiment_enhanced_lstm.h5"),
        'verbose': 1
    }
    
    # Train price-only model
    price_model_results = train_price_only_model(dataset, model_params, price_training_params)
    
    # Train sentiment-enhanced model (if sentiment data available)
    if 'X_sentiment_train' in dataset:
        sentiment_model_results = train_sentiment_model(dataset, model_params, sentiment_training_params)
        
        # Save evaluation results
        dataset_info = {
            'train_size': len(dataset['X_price_train']),
            'val_size': len(dataset['X_price_val']),
            'test_size': len(dataset['X_price_test']),
            'window_size': args.window,
            'forecast_horizon': args.horizon,
            'X_price_test': dataset['X_price_test'],
            'X_sentiment_test': dataset['X_sentiment_test'],
            'y_test': dataset['y_test'],
            'dates_test': dataset.get('dates_test', None)
        }
        
        save_evaluation_results(
            price_model_results,
            sentiment_model_results,
            dataset_info,
            args.output_dir
        )
        
        logger.info("Model training and evaluation completed successfully")
    else:
        logger.warning("Sentiment data not available, only price-only model was trained")
        
    logger.info(f"Trained models saved to {args.output_dir}")

if __name__ == "__main__":
    main() 