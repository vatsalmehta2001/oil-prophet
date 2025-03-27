"""
FinBERT Oil Market Sentiment Analysis

This module provides enhanced sentiment analysis for oil market texts using 
specialized financial BERT models. It includes multi-model consensus analysis,
domain adaptation for oil markets, and utilities for analyzing historical sentiment.
"""

import torch
import pandas as pd
import numpy as np
import re
import os
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification
)
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("finbert_sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OilFinBERT:
    """
    Enhanced financial BERT model specialized for oil market sentiment analysis.
    
    This class combines multiple BERT models trained on financial and market data
    to provide more accurate sentiment analysis specifically for oil market texts.
    """
    
    # Available models with their characteristics
    MODELS = {
        "finbert": {
            "name": "ProsusAI/finbert",
            "labels": ["positive", "negative", "neutral"],
            "domain": "finance",
            "specialization": "general finance"
        },
        "finbert-tone": {
            "name": "yiyanghkust/finbert-tone",
            "labels": ["neutral", "positive", "negative"],
            "domain": "finance",
            "specialization": "earnings calls"
        },
        "finance-sentiment": {
            "name": "soleimanian/financial-roberta-large-sentiment",
            "labels": ["Neutral", "Positive", "Negative"],
            "domain": "finance",
            "specialization": "news"
        },
        "esg-sentiment": {
            "name": "yiyanghkust/finbert-esg",
            "labels": ["neutral", "positive", "negative"],
            "domain": "finance-esg",
            "specialization": "environmental social governance"
        },
        "commodity-sentiment": {
            "name": "nlptown/bert-base-multilingual-uncased-sentiment",
            "labels": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
            "domain": "general",
            "specialization": "rating based sentiment"
        }
    }
    
    # Oil market domain words for adaptive weighting
    OIL_DOMAIN_TERMS = [
        "oil", "crude", "petroleum", "barrel", "wti", "brent", "opec",
        "shale", "refinery", "gasoline", "drilling", "rig", "offshore",
        "onshore", "fracking", "production", "reserves", "supply", "demand",
        "futures", "prices", "price", "market", "energy", "fossil", "petrol",
        "diesel", "jet fuel", "commodity", "commodities"
    ]
    
    def __init__(
        self,
        models: List[str] = ["finbert", "finbert-tone"],
        device: Optional[str] = None,
        domain_adaptation: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
        oil_term_weighting: float = 1.5
    ):
        """
        Initialize the OilFinBERT sentiment analyzer.
        
        Args:
            models: List of model names to use (from OilFinBERT.MODELS)
            device: Device to run models on ('cpu' or 'cuda')
            domain_adaptation: Whether to apply oil domain adaptation
            batch_size: Batch size for inference
            max_length: Maximum text length for tokenization
            oil_term_weighting: Weight multiplier for texts with oil terms
        """
        self.model_keys = models
        self.batch_size = batch_size
        self.max_length = max_length
        self.domain_adaptation = domain_adaptation
        self.oil_term_weighting = oil_term_weighting
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self.models = {}
        self.tokenizers = {}
        
        logger.info(f"Initializing OilFinBERT with models: {', '.join(models)}")
        logger.info(f"Using device: {self.device}")
        
        for model_key in models:
            if model_key not in self.MODELS:
                logger.warning(f"Model '{model_key}' not recognized, skipping")
                continue
                
            model_info = self.MODELS[model_key]
            logger.info(f"Loading model: {model_info['name']}")
            
            try:
                # Load tokenizer
                self.tokenizers[model_key] = AutoTokenizer.from_pretrained(model_info["name"])
                
                # Load model
                self.models[model_key] = AutoModelForSequenceClassification.from_pretrained(model_info["name"])
                self.models[model_key].to(self.device)
                self.models[model_key].eval()
                
                logger.info(f"Successfully loaded {model_key}")
            except Exception as e:
                logger.error(f"Error loading model {model_key}: {str(e)}")
                # Remove from model keys
                if model_key in self.model_keys:
                    self.model_keys.remove(model_key)
        
        # Initialize NLTK for text processing
        self._initialize_nltk()
        
        # Compile regex patterns
        self.oil_pattern = re.compile(r'\b(' + '|'.join(self.OIL_DOMAIN_TERMS) + r')\b', re.IGNORECASE)
        
        if not self.model_keys:
            logger.error("No models loaded successfully")
            raise ValueError("Failed to load any models")
        
        logger.info(f"OilFinBERT initialized with {len(self.model_keys)} models")
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK components for text processing."""
        try:
            # Download required resources if not already present
            for resource in ['punkt', 'stopwords']:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            
            # Get English stopwords
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK components initialized")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {str(e)}")
            self.stop_words = set()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # We don't remove special characters and numbers as they may be relevant
        # for financial sentiment (e.g., prices, dates, percentages)
        
        return text
    
    def _analyze_with_model(
        self, 
        texts: List[str], 
        model_key: str
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment using a specific model.
        
        Args:
            texts: List of texts to analyze
            model_key: Key of the model to use
            
        Returns:
            List of sentiment dictionaries
        """
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        model_info = self.MODELS[model_key]
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Process each result
                for j, prob in enumerate(probs):
                    scores = prob.cpu().numpy()
                    
                    # Map to standard sentiment keys based on model's label order
                    labels = model_info["labels"]
                    
                    sentiment_result = {
                        "positive": 0.0,
                        "negative": 0.0,
                        "neutral": 0.0
                    }
                    
                    # Map model-specific labels to standard keys
                    for idx, label in enumerate(labels):
                        label_lower = label.lower()
                        
                        if "positive" in label_lower or label in ["4 stars", "5 stars"]:
                            sentiment_result["positive"] += float(scores[idx])
                        elif "negative" in label_lower or label in ["1 star", "2 stars"]:
                            sentiment_result["negative"] += float(scores[idx])
                        elif "neutral" in label_lower or label in ["3 stars"]:
                            sentiment_result["neutral"] += float(scores[idx])
                    
                    # Calculate compound score (positive - negative)
                    sentiment_result["compound"] = sentiment_result["positive"] - sentiment_result["negative"]
                    
                    # Add model-specific sentiment
                    sentiment_result[f"{model_key}_positive"] = sentiment_result["positive"]
                    sentiment_result[f"{model_key}_negative"] = sentiment_result["negative"]
                    sentiment_result[f"{model_key}_neutral"] = sentiment_result["neutral"]
                    sentiment_result[f"{model_key}_compound"] = sentiment_result["compound"]
                    
                    results.append(sentiment_result)
        
        return results
    
    def _calculate_domain_relevance(self, text: str) -> float:
        """
        Calculate relevance of text to oil domain.
        
        Args:
            text: Input text
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Count oil domain terms
        oil_terms = self.oil_pattern.findall(text.lower())
        oil_term_count = len(oil_terms)
        
        # Calculate density (terms per 100 words)
        words = text.split()
        word_count = max(1, len(words))  # Avoid division by zero
        oil_term_density = (oil_term_count / word_count) * 100
        
        # Convert to relevance score (0.0 to 1.0)
        # Higher density = higher relevance, capped at 1.0
        relevance = min(1.0, oil_term_density / 10.0)
        
        return relevance
    
    def analyze_sentiment(
        self, 
        texts: Union[str, List[str]], 
        verbose: bool = False
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Analyze sentiment for one or more texts.
        
        Args:
            texts: Single text or list of texts to analyze
            verbose: Whether to include detailed model outputs
            
        Returns:
            Dictionary or list of dictionaries with sentiment scores
        """
        # Handle single text input
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        
        # Preprocess texts
        preprocessed_texts = [self._preprocess_text(text) for text in texts]
        
        # Calculate domain relevance if domain adaptation is enabled
        domain_weights = None
        if self.domain_adaptation:
            domain_weights = [self._calculate_domain_relevance(text) for text in preprocessed_texts]
        
        # Get sentiment from each model
        all_model_results = {}
        for model_key in self.model_keys:
            try:
                model_results = self._analyze_with_model(preprocessed_texts, model_key)
                all_model_results[model_key] = model_results
            except Exception as e:
                logger.error(f"Error analyzing with model {model_key}: {str(e)}")
        
        # Combine results from all models
        combined_results = []
        for i, text in enumerate(preprocessed_texts):
            # Get sentiment from each model
            sentiments = {
                model_key: all_model_results[model_key][i]
                for model_key in all_model_results
            }
            
            # Initialize combined result
            combined = {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "compound": 0.0
            }
            
            # Add domain info if enabled
            if self.domain_adaptation:
                domain_relevance = domain_weights[i]
                combined["domain_relevance"] = domain_relevance
            
            # Determine model weights
            model_weights = {}
            total_weight = 0.0
            
            for model_key in sentiments:
                model_info = self.MODELS[model_key]
                
                # Base weight (all models start equal)
                weight = 1.0
                
                # Adjust weight based on domain
                if self.domain_adaptation and domain_weights:
                    # Finance-specific models get higher weight for financial texts
                    if model_info["domain"] == "finance":
                        weight *= 1.2
                    
                    # ESG models get higher weight for environmental texts
                    if model_info["domain"] == "finance-esg" and domain_weights[i] > 0.5:
                        weight *= 1.3
                
                model_weights[model_key] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for model_key in model_weights:
                    model_weights[model_key] /= total_weight
            
            # Combine sentiments from all models using weights
            for model_key, sentiment in sentiments.items():
                weight = model_weights[model_key]
                combined["positive"] += sentiment["positive"] * weight
                combined["negative"] += sentiment["negative"] * weight
                combined["neutral"] += sentiment["neutral"] * weight
                combined["compound"] += sentiment["compound"] * weight
            
            # Apply domain adaptation if enabled
            if self.domain_adaptation and domain_weights and domain_weights[i] > 0.2:
                # Enhance sentiment signal for oil-relevant texts
                weight_factor = 1.0 + (domain_weights[i] * (self.oil_term_weighting - 1.0))
                combined["compound"] *= weight_factor
                
                # Recalibrate positive/negative based on adjusted compound
                if combined["compound"] > 0:
                    positive_boost = combined["compound"] * 0.5
                    combined["positive"] = min(1.0, combined["positive"] + positive_boost)
                    combined["negative"] = max(0.0, combined["negative"] - positive_boost * 0.5)
                elif combined["compound"] < 0:
                    negative_boost = abs(combined["compound"]) * 0.5
                    combined["negative"] = min(1.0, combined["negative"] + negative_boost)
                    combined["positive"] = max(0.0, combined["positive"] - negative_boost * 0.5)
                
                # Ensure neutrality is adjusted accordingly
                total = combined["positive"] + combined["negative"]
                if total > 1.0:
                    combined["neutral"] = 0.0
                    # Normalize positive and negative
                    combined["positive"] /= total
                    combined["negative"] /= total
                else:
                    combined["neutral"] = 1.0 - total
            
            # Add verbose model details if requested
            if verbose:
                combined["models"] = sentiments
                combined["model_weights"] = model_weights
            
            combined_results.append(combined)
        
        # Return single result if input was a single string
        if single_input:
            return combined_results[0]
        
        return combined_results
    
    def analyze_text_batch(
        self, 
        df: pd.DataFrame, 
        text_col: str = 'text', 
        title_col: Optional[str] = 'title',
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts to analyze
            text_col: Column name containing text
            title_col: Optional column name containing titles
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with added sentiment columns
        """
        result_df = df.copy()
        
        # Create a list of texts to analyze
        texts = []
        for _, row in df.iterrows():
            # Combine title and text if both are available
            text = row[text_col] if pd.notna(row[text_col]) else ""
            if title_col and title_col in row and pd.notna(row[title_col]):
                text = f"{row[title_col]}. {text}"
            
            texts.append(text)
        
        # Process in batches to avoid memory issues
        all_sentiments = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i+batch_size]
            batch_sentiments = self.analyze_sentiment(batch_texts)
            all_sentiments.extend(batch_sentiments)
        
        # Add sentiment columns to DataFrame
        result_df["sentiment_positive"] = [s["positive"] for s in all_sentiments]
        result_df["sentiment_negative"] = [s["negative"] for s in all_sentiments]
        result_df["sentiment_neutral"] = [s["neutral"] for s in all_sentiments]
        result_df["sentiment_compound"] = [s["compound"] for s in all_sentiments]
        
        # Add domain relevance if available
        if self.domain_adaptation and "domain_relevance" in all_sentiments[0]:
            result_df["oil_relevance"] = [s.get("domain_relevance", 0.0) for s in all_sentiments]
        
        return result_df
    
    def analyze_reddit_data(
        self, 
        df: pd.DataFrame, 
        text_col: str = 'text', 
        title_col: Optional[str] = 'title',
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Analyze sentiment for Reddit data, handling both posts and comments.
        
        Args:
            df: DataFrame containing Reddit data
            text_col: Column name containing post text or comment body
            title_col: Column name containing post titles
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with added sentiment columns
        """
        # Check if the DataFrame has a 'type' column to distinguish posts and comments
        if 'type' in df.columns:
            # Create a copy of the DataFrame to avoid modifying the original
            result_df = df.copy()
            
            # Get posts and comments separately
            posts_df = result_df[result_df['type'] == 'post']
            comments_df = result_df[result_df['type'] == 'comment']
            
            # Process posts (with titles if available)
            if not posts_df.empty:
                posts_df = self.analyze_text_batch(
                    posts_df,
                    text_col=text_col,
                    title_col=title_col,
                    batch_size=batch_size
                )
            
            # Process comments (no titles)
            if not comments_df.empty:
                comments_df = self.analyze_text_batch(
                    comments_df,
                    text_col=text_col,
                    title_col=None,  # Comments don't have titles
                    batch_size=batch_size
                )
            
            # Combine the results
            combined_df = pd.concat([posts_df, comments_df], ignore_index=False)
            
            # Sort by original index to preserve order
            combined_df = combined_df.sort_index()
            
            return combined_df
        else:
            # If there's no 'type' column, treat all data as posts
            return self.analyze_text_batch(
                df,
                text_col=text_col,
                title_col=title_col,
                batch_size=batch_size
            )
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate sentiment distribution statistics.
        
        Args:
            df: DataFrame with sentiment columns
            
        Returns:
            Dictionary with sentiment distribution statistics
        """
        if df.empty or not all(col in df.columns for col in ['sentiment_compound', 'sentiment_positive', 'sentiment_negative']):
            logger.error("DataFrame missing required sentiment columns")
            return {}
        
        # Get compound sentiment statistics
        compound = df['sentiment_compound'].dropna()
        
        # Calculate sentiment distribution
        distribution = {
            'mean': float(compound.mean()),
            'median': float(compound.median()),
            'std': float(compound.std()),
            'min': float(compound.min()),
            'max': float(compound.max()),
            'positive_pct': float((compound > 0.05).mean() * 100),
            'neutral_pct': float((compound.between(-0.05, 0.05)).mean() * 100),
            'negative_pct': float((compound < -0.05).mean() * 100)
        }
        
        return distribution
    
    def aggregate_sentiment_by_time(
        self, 
        df: pd.DataFrame, 
        date_col: str = 'created_date',
        time_freq: str = 'D',
        min_count: int = 3
    ) -> pd.DataFrame:
        """
        Aggregate sentiment by time period.
        
        Args:
            df: DataFrame with sentiment columns
            date_col: Column containing dates
            time_freq: Time frequency for aggregation (D=daily, W=weekly, M=monthly)
            min_count: Minimum count of items to include in aggregation
            
        Returns:
            DataFrame with aggregated sentiment by time period
        """
        # Ensure date column is datetime type
        if date_col not in df.columns:
            logger.error(f"Date column '{date_col}' not found in DataFrame")
            return pd.DataFrame()
        
        # Create a copy with datetime index
        if pd.api.types.is_datetime64_dtype(df[date_col]):
            time_df = df.set_index(date_col).copy()
        else:
            # Try to convert to datetime
            try:
                if 'created_utc' in df.columns:
                    # Convert Unix timestamp to datetime
                    time_df = df.copy()
                    time_df['datetime'] = pd.to_datetime(time_df['created_utc'], unit='s')
                    time_df = time_df.set_index('datetime')
                else:
                    # Try to parse the date column
                    time_df = df.copy()
                    time_df['datetime'] = pd.to_datetime(time_df[date_col])
                    time_df = time_df.set_index('datetime')
            except Exception as e:
                logger.error(f"Error converting date column: {str(e)}")
                return pd.DataFrame()
        
        # Aggregate by time period
        agg_df = time_df.resample(time_freq).agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'id': 'count'  # Count of items in each period
        }).rename(columns={'id': 'count'})
        
        # Filter periods with too few items
        agg_df = agg_df[agg_df['count'] >= min_count]
        
        # Calculate moving averages
        for window in [3, 7, 14, 30]:
            if len(agg_df) > window:
                agg_df[f'compound_ma{window}'] = agg_df['sentiment_compound'].rolling(window=window).mean()
        
        # Calculate momentum (day-over-day change)
        agg_df['sentiment_momentum'] = agg_df['sentiment_compound'].diff()
        
        # Calculate volatility (rolling standard deviation)
        if len(agg_df) > 7:
            agg_df['sentiment_volatility'] = agg_df['sentiment_compound'].rolling(window=7).std()
        
        return agg_df
    
    def plot_sentiment_over_time(
        self, 
        df: pd.DataFrame,
        date_col: str = 'created_date',
        time_freq: str = 'D',
        figsize: Tuple[int, int] = (12, 8),
        title: str = 'Oil Market Sentiment Over Time',
        show_ma: bool = True
    ) -> plt.Figure:
        """
        Plot sentiment over time.
        
        Args:
            df: DataFrame with sentiment columns
            date_col: Column containing dates
            time_freq: Time frequency for aggregation
            figsize: Figure size
            title: Plot title
            show_ma: Whether to show moving averages
            
        Returns:
            Matplotlib figure
        """
        # Aggregate sentiment by time
        agg_df = self.aggregate_sentiment_by_time(df, date_col, time_freq)
        
        if agg_df.empty:
            logger.error("No data available for sentiment time plot")
            return None
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot compound sentiment
        ax1.plot(agg_df.index, agg_df['sentiment_compound'], 'b-', linewidth=2, label='Compound Sentiment')
        
        # Add moving averages if requested and available
        if show_ma:
            if 'compound_ma7' in agg_df.columns:
                ax1.plot(agg_df.index, agg_df['compound_ma7'], 'r--', linewidth=1.5, label='7-day MA')
            
            if 'compound_ma14' in agg_df.columns:
                ax1.plot(agg_df.index, agg_df['compound_ma14'], 'g--', linewidth=1.5, label='14-day MA')
        
        # Add horizontal lines at meaningful levels
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax1.axhline(y=0.2, color='g', linestyle=':', alpha=0.3, label='Bullish Threshold')
        ax1.axhline(y=-0.2, color='r', linestyle=':', alpha=0.3, label='Bearish Threshold')
        
        # Format first subplot
        ax1.set_title(title)
        ax1.set_ylabel('Sentiment Score')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot post/comment count in second subplot
        ax2.bar(agg_df.index, agg_df['count'], color='steelblue', alpha=0.7, label='Post/Comment Count')
        
        # Format second subplot
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Oil Market Sentiment Distribution'
    ) -> plt.Figure:
        """
        Plot sentiment distribution.
        
        Args:
            df: DataFrame with sentiment columns
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if df.empty or 'sentiment_compound' not in df.columns:
            logger.error("DataFrame missing required sentiment columns")
            return None
        
        # Get compound sentiment values
        compound = df['sentiment_compound'].dropna()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(
            compound,
            bins=50,
            color='skyblue',
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add vertical lines at meaningful levels
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Neutral')
        ax.axvline(x=0.2, color='g', linestyle='--', alpha=0.5, label='Bullish Threshold')
        ax.axvline(x=-0.2, color='r', linestyle='--', alpha=0.5, label='Bearish Threshold')
        
        # Add statistics
        mean = compound.mean()
        median = compound.median()
        std = compound.std()
        
        stats_text = f"Mean: {mean:.3f}\nMedian: {median:.3f}\nStd Dev: {std:.3f}"
        ax.text(
            0.02, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Format plot
        ax.set_title(title)
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_sentiment_vs_price(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        date_col_sentiment: str = 'created_date',
        date_col_price: str = 'Date',
        price_col: str = 'Price',
        time_freq: str = 'D',
        figsize: Tuple[int, int] = (14, 8),
        title: str = 'Oil Market Sentiment vs. Price'
    ) -> plt.Figure:
        """
        Plot sentiment vs. price over time.
        
        Args:
            sentiment_df: DataFrame with sentiment columns
            price_df: DataFrame with price data
            date_col_sentiment: Date column in sentiment DataFrame
            date_col_price: Date column in price DataFrame
            price_col: Price column name
            time_freq: Time frequency for aggregation
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Aggregate sentiment by time
        agg_sentiment = self.aggregate_sentiment_by_time(sentiment_df, date_col_sentiment, time_freq)
        
        if agg_sentiment.empty:
            logger.error("No sentiment data available for price comparison plot")
            return None
        
        # Prepare price data
        if date_col_price not in price_df.columns or price_col not in price_df.columns:
            logger.error(f"Required columns not found in price DataFrame: {date_col_price}, {price_col}")
            return None
        
        # Convert price dates to datetime and set as index
        price_df = price_df.copy()
        if not pd.api.types.is_datetime64_dtype(price_df[date_col_price]):
            price_df[date_col_price] = pd.to_datetime(price_df[date_col_price])
        
        price_df = price_df.set_index(date_col_price)
        
        # Resample price data to match sentiment frequency
        price_resampled = price_df[price_col].resample(time_freq).mean()
        
        # Find overlapping date range
        start_date = max(agg_sentiment.index.min(), price_resampled.index.min())
        end_date = min(agg_sentiment.index.max(), price_resampled.index.max())
        
        # Filter to common date range
        agg_sentiment = agg_sentiment.loc[start_date:end_date]
        price_resampled = price_resampled.loc[start_date:end_date]
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot price on primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(price_resampled.index, price_resampled.values, color=color, linewidth=2, label='Oil Price')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for sentiment
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Sentiment Score', color=color)
        ax2.plot(agg_sentiment.index, agg_sentiment['sentiment_compound'], color=color, linewidth=1.5, label='Sentiment')
        
        # Add moving average if available
        if 'compound_ma7' in agg_sentiment.columns:
            ax2.plot(agg_sentiment.index, agg_sentiment['compound_ma7'], 'tab:green', linewidth=1.5, linestyle='--', label='Sentiment 7-day MA')
        
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add horizontal line at neutral sentiment
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
    
    def extract_keywords(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        title_col: Optional[str] = 'title',
        n_keywords: int = 30,
        min_word_length: int = 3,
        additional_stopwords: Optional[List[str]] = None
    ) -> List[Tuple[str, int]]:
        """
        Extract most frequent keywords from texts.
        
        Args:
            df: DataFrame containing texts
            text_col: Column name containing text
            title_col: Optional column name containing titles
            n_keywords: Number of top keywords to extract
            min_word_length: Minimum word length to include
            additional_stopwords: Additional stopwords to filter out
            
        Returns:
            List of (keyword, count) tuples
        """
        # Create combined text corpus
        texts = []
        for _, row in df.iterrows():
            # Combine title and text if both are available
            text = row[text_col] if pd.notna(row[text_col]) else ""
            if title_col and title_col in row and pd.notna(row[title_col]):
                text = f"{row[title_col]}. {text}"
            
            texts.append(text)
        
        # Combine all texts
        corpus = " ".join(texts)
        
        # Tokenize
        words = word_tokenize(corpus.lower())
        
        # Add additional stopwords if provided
        stop_words = self.stop_words.copy()
        if additional_stopwords:
            stop_words.update(additional_stopwords)
        
        # Filter words
        filtered_words = [
            word for word in words
            if (
                word.isalpha() and  # Only alphabetic
                len(word) >= min_word_length and  # Minimum length
                word not in stop_words  # Not a stopword
            )
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top keywords
        return word_counts.most_common(n_keywords)
    
    def plot_keyword_frequencies(
        self,
        keywords: List[Tuple[str, int]],
        figsize: Tuple[int, int] = (12, 8),
        title: str = 'Top Keywords in Oil Market Discussions'
    ) -> plt.Figure:
        """
        Plot keyword frequencies.
        
        Args:
            keywords: List of (keyword, count) tuples
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not keywords:
            logger.error("No keywords provided for plotting")
            return None
        
        # Unpack keywords and counts
        words, counts = zip(*keywords)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot with blue to red color gradient based on frequency
        y_pos = range(len(words))
        cmap = plt.cm.get_cmap('Blues')
        colors = [cmap(count / max(counts)) for count in counts]
        
        ax.barh(y_pos, counts, color=colors)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Most frequent at top
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            ax.text(count + 0.5, i, str(count), va='center')
        
        # Format plot
        ax.set_title(title)
        ax.set_xlabel('Frequency')
        
        plt.tight_layout()
        
        return fig
    
    def calculate_correlation_with_price(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        date_col_sentiment: str = 'created_date',
        date_col_price: str = 'Date',
        price_col: str = 'Price',
        time_freq: str = 'D',
        max_lag: int = 10
    ) -> pd.DataFrame:
        """
        Calculate correlation between sentiment and price with different lags.
        
        Args:
            sentiment_df: DataFrame with sentiment columns
            price_df: DataFrame with price data
            date_col_sentiment: Date column in sentiment DataFrame
            date_col_price: Date column in price DataFrame
            price_col: Price column name
            time_freq: Time frequency for aggregation
            max_lag: Maximum lag days to test
            
        Returns:
            DataFrame with correlation results
        """
        # Aggregate sentiment by time
        agg_sentiment = self.aggregate_sentiment_by_time(sentiment_df, date_col_sentiment, time_freq)
        
        if agg_sentiment.empty:
            logger.error("No sentiment data available for correlation analysis")
            return pd.DataFrame()
        
        # Prepare price data
        if date_col_price not in price_df.columns or price_col not in price_df.columns:
            logger.error(f"Required columns not found in price DataFrame: {date_col_price}, {price_col}")
            return pd.DataFrame()
        
        # Convert price dates to datetime and set as index
        price_df = price_df.copy()
        if not pd.api.types.is_datetime64_dtype(price_df[date_col_price]):
            price_df[date_col_price] = pd.to_datetime(price_df[date_col_price])
        
        price_df = price_df.set_index(date_col_price)
        
        # Resample price data to match sentiment frequency
        price_resampled = price_df[price_col].resample(time_freq).mean()
        
        # Find overlapping date range
        start_date = max(agg_sentiment.index.min(), price_resampled.index.min())
        end_date = min(agg_sentiment.index.max(), price_resampled.index.max())
        
        # Filter to common date range
        agg_sentiment = agg_sentiment.loc[start_date:end_date]
        price_resampled = price_resampled.loc[start_date:end_date]
        
        # Compute correlations at different lags
        correlations = []
        
        # Sentiment leading price (negative lag)
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Sentiment leads price (e.g., -3 means sentiment from 3 days ago)
                sentiment_shifted = agg_sentiment['sentiment_compound'].shift(lag)
                corr = price_resampled.corr(sentiment_shifted)
            elif lag > 0:
                # Price leads sentiment (e.g., 3 means price from 3 days ago)
                price_shifted = price_resampled.shift(lag)
                corr = agg_sentiment['sentiment_compound'].corr(price_shifted)
            else:
                # Contemporaneous
                corr = price_resampled.corr(agg_sentiment['sentiment_compound'])
            
            correlations.append({
                'lag': lag,
                'correlation': corr,
                'direction': 'Sentiment leads Price' if lag < 0 else ('Price leads Sentiment' if lag > 0 else 'Same day')
            })
        
        result_df = pd.DataFrame(correlations)
        result_df = result_df.sort_values('lag')
        
        return result_df
    
    def plot_correlation_lags(
        self,
        correlation_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6),
        title: str = 'Cross-Correlation: Sentiment vs. Price at Different Lags'
    ) -> plt.Figure:
        """
        Plot correlation between sentiment and price at different lags.
        
        Args:
            correlation_df: DataFrame with correlation results
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if correlation_df.empty or not all(col in correlation_df.columns for col in ['lag', 'correlation']):
            logger.error("Correlation DataFrame missing required columns")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot correlation by lag
        lags = correlation_df['lag'].values
        corrs = correlation_df['correlation'].values
        
        # Plot with color gradient based on correlation direction (red for negative, blue for positive)
        colors = ['r' if c < 0 else 'b' for c in corrs]
        
        ax.bar(lags, corrs, color=colors, alpha=0.7)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Add vertical line at lag 0
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        
        # Find max correlation and highlight it
        max_corr_idx = np.argmax(np.abs(corrs))
        max_corr_lag = lags[max_corr_idx]
        max_corr_val = corrs[max_corr_idx]
        
        ax.plot(max_corr_lag, max_corr_val, 'ko', markersize=10)
        ax.annotate(
            f'Max correlation: {max_corr_val:.3f} at lag {max_corr_lag}',
            xy=(max_corr_lag, max_corr_val),
            xytext=(max_corr_lag + (-3 if max_corr_lag > 0 else 3), max_corr_val + 0.1),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
        )
        
        # Format plot
        ax.set_title(title)
        ax.set_xlabel('Lag (Negative = Sentiment leads Price, Positive = Price leads Sentiment)')
        ax.set_ylabel('Correlation Coefficient')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def create_sentiment_features(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        date_col_sentiment: str = 'created_date',
        date_col_price: str = 'Date',
        price_col: str = 'Price',
        time_freq: str = 'D',
        window_size: int = 30,
        smoothing_window: int = 7
    ) -> pd.DataFrame:
        """
        Create sentiment features for forecasting models.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            price_df: DataFrame with price data
            date_col_sentiment: Date column in sentiment DataFrame
            date_col_price: Date column in price DataFrame
            price_col: Price column in price DataFrame
            time_freq: Time frequency for aggregation
            window_size: Window size for feature creation
            smoothing_window: Window size for sentiment smoothing
            
        Returns:
            DataFrame with combined price and sentiment features
        """
        # Aggregate sentiment by time
        agg_sentiment = self.aggregate_sentiment_by_time(sentiment_df, date_col_sentiment, time_freq)
        
        if agg_sentiment.empty:
            logger.error("No sentiment data available for feature creation")
            return pd.DataFrame()
        
        # Prepare price data
        if date_col_price not in price_df.columns or price_col not in price_df.columns:
            logger.error(f"Required columns not found in price DataFrame: {date_col_price}, {price_col}")
            return pd.DataFrame()
        
        # Convert price dates to datetime and set as index
        price_df = price_df.copy()
        if not pd.api.types.is_datetime64_dtype(price_df[date_col_price]):
            price_df[date_col_price] = pd.to_datetime(price_df[date_col_price])
        
        price_df = price_df.set_index(date_col_price)
        
        # Resample price data to match sentiment frequency
        price_resampled = price_df[price_col].resample(time_freq).mean()
        
        # Create combined DataFrame
        combined = pd.DataFrame(index=price_resampled.index)
        combined['price'] = price_resampled
        
        # Add sentiment data
        combined = combined.join(
            agg_sentiment[['sentiment_compound', 'sentiment_positive', 'sentiment_negative']], 
            how='left'
        )
        
        # Create smoothed sentiment features
        if smoothing_window > 1:
            combined['sentiment_ma'] = combined['sentiment_compound'].rolling(window=smoothing_window).mean()
        else:
            combined['sentiment_ma'] = combined['sentiment_compound']
        
        # Fill missing sentiment values with neutral (0)
        sentiment_cols = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_ma']
        combined[sentiment_cols] = combined[sentiment_cols].fillna(0)
        
        # Create lagged price and sentiment features
        for lag in range(1, window_size + 1):
            # Price lags
            combined[f'price_lag_{lag}'] = combined['price'].shift(lag)
            
            # Sentiment lags (using smoothed sentiment)
            combined[f'sentiment_lag_{lag}'] = combined['sentiment_ma'].shift(lag)
        
        # Drop rows with missing values (from the lags)
        combined = combined.dropna()
        
        return combined
    
    def save_model(self, path: str) -> None:
        """
        Save model information (not the models themselves).
        
        Args:
            path: Path to save model information
        """
        # We can't save the actual models easily, but we can save the configuration
        model_info = {
            'model_keys': self.model_keys,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'domain_adaptation': self.domain_adaptation,
            'oil_term_weighting': self.oil_term_weighting,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(model_info, f, indent=4)
            logger.info(f"Saved model information to {path}")
        except Exception as e:
            logger.error(f"Error saving model information: {str(e)}")
    
    @classmethod
    def from_saved_config(cls, path: str) -> 'OilFinBERT':
        """
        Create OilFinBERT instance from saved configuration.
        
        Args:
            path: Path to saved model information
            
        Returns:
            OilFinBERT instance
        """
        try:
            with open(path, 'r') as f:
                model_info = json.load(f)
            
            logger.info(f"Loaded model information from {path}")
            
            # Create new instance with saved configuration
            instance = cls(
                models=model_info.get('model_keys', ["finbert", "finbert-tone"]),
                device=model_info.get('device', None),
                domain_adaptation=model_info.get('domain_adaptation', True),
                batch_size=model_info.get('batch_size', 16),
                max_length=model_info.get('max_length', 512),
                oil_term_weighting=model_info.get('oil_term_weighting', 1.5)
            )
            
            return instance
        except Exception as e:
            logger.error(f"Error loading model information: {str(e)}")
            raise


def run_sentiment_analysis(
    input_path: str,
    output_path: str = None,
    models: List[str] = ["finbert", "finbert-tone"],
    text_col: str = 'text',
    title_col: Optional[str] = 'title',
    batch_size: int = 32,
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Run sentiment analysis on a dataset and save results.
    
    Args:
        input_path: Path to input data (CSV)
        output_path: Path to save results (CSV)
        models: List of model names to use
        text_col: Column name containing text
        title_col: Optional column name containing titles
        batch_size: Batch size for processing
        device: Device to run models on ('cpu' or 'cuda')
        
    Returns:
        DataFrame with sentiment analysis results
    """
    try:
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows from {input_path}")
        
        # Initialize sentiment analyzer
        analyzer = OilFinBERT(models=models, device=device, batch_size=batch_size)
        
        # Run sentiment analysis
        if 'type' in df.columns and any(df['type'] == 'post') and any(df['type'] == 'comment'):
            # Reddit data with posts and comments
            result_df = analyzer.analyze_reddit_data(
                df,
                text_col=text_col,
                title_col=title_col,
                batch_size=batch_size
            )
        else:
            # Generic text data
            result_df = analyzer.analyze_text_batch(
                df,
                text_col=text_col,
                title_col=title_col,
                batch_size=batch_size
            )
        
        # Save results if output path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(result_df)} rows to {output_path}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error running sentiment analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Oil Market Sentiment Analysis with FinBERT")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument("--text_col", type=str, default="text", help="Column containing text")
    parser.add_argument("--title_col", type=str, help="Column containing titles (optional)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--models", type=str, default="finbert,finbert-tone", help="Comma-separated list of models to use")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to run models on")
    
    args = parser.parse_args()
    
    # Parse models list
    models = args.models.split(",")
    
    # Run sentiment analysis
    results = run_sentiment_analysis(
        input_path=args.input,
        output_path=args.output,
        models=models,
        text_col=args.text_col,
        title_col=args.title_col,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Print summary
    if not results.empty:
        sentiment_distribution = results['sentiment_compound'].describe()
        print("\nSentiment Distribution:")
        print(sentiment_distribution)
        
        # Calculate sentiment proportions
        positive_pct = (results['sentiment_compound'] > 0.05).mean() * 100
        neutral_pct = ((results['sentiment_compound'] >= -0.05) & (results['sentiment_compound'] <= 0.05)).mean() * 100
        negative_pct = (results['sentiment_compound'] < -0.05).mean() * 100
        
        print(f"\nPositive sentiment: {positive_pct:.1f}%")
        print(f"Neutral sentiment: {neutral_pct:.1f}%")
        print(f"Negative sentiment: {negative_pct:.1f}%")
        
        if 'oil_relevance' in results.columns:
            avg_relevance = results['oil_relevance'].mean()
            print(f"\nAverage oil relevance score: {avg_relevance:.3f}")