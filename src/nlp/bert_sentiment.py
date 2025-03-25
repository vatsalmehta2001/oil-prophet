"""
BERT-based sentiment analysis module for Oil Prophet.

This module provides functionality to extract and analyze sentiment from Reddit posts
and news articles related to oil markets using BERT, which can be used as additional 
features for forecasting models.
"""

import pandas as pd
import numpy as np
import praw
import re
import time
import os
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BERTSentimentAnalyzer:
    """
    Class for analyzing sentiment using BERT models.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512
    ):
        """
        Initialize the BERT sentiment analyzer.
        
        Args:
            model_name: Name of the BERT model to use (default: FinBERT)
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model
        self.initialized = False
        self.tokenizer = None
        self.model = None
        
        try:
            self._initialize_model()
            
            # Initialize NLTK for text preprocessing
            self._initialize_nltk()
            
            logger.info(f"BERT sentiment analyzer initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {str(e)}")
    
    def _initialize_model(self) -> None:
        """Initialize BERT model and tokenizer."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            raise
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK components."""
        try:
            # Download NLTK resources if needed
            nltk_resources = ['punkt', 'stopwords']
            for resource in nltk_resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
                except LookupError:
                    nltk.download(resource, quiet=True)
            
            # Get stopwords
            self.stop_words = set(stopwords.words('english'))
            
            logger.info("NLTK components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLTK components: {str(e)}")
    
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
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores for each text
        """
        if not self.initialized:
            logger.error("BERT sentiment analyzer not initialized")
            return [{} for _ in texts]
        
        # Preprocess texts
        preprocessed_texts = [self._preprocess_text(text) for text in texts]
        
        # Handle empty texts
        results = []
        valid_indices = []
        valid_texts = []
        
        for i, text in enumerate(preprocessed_texts):
            if text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                # Add empty result for empty text
                results.append({
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,  # Default to neutral
                    'compound': 0.0
                })
        
        if not valid_texts:
            return results
        
        # Tokenize texts
        inputs = self.tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Process scores based on model's label order
        # FinBERT: ['positive', 'negative', 'neutral']
        # Other models may have different label orders
        if self.model_name == "ProsusAI/finbert":
            for i, idx in enumerate(valid_indices):
                score = scores[i].cpu().numpy()
                results.insert(idx, {
                    'positive': float(score[0]),
                    'negative': float(score[1]),
                    'neutral': float(score[2]),
                    'compound': float(score[0] - score[1])  # Simple compound score
                })
        else:
            # Default handling for other models (assuming binary sentiment)
            for i, idx in enumerate(valid_indices):
                score = scores[i].cpu().numpy()
                results.insert(idx, {
                    'positive': float(score[1]) if len(score) > 1 else float(score[0]),
                    'negative': float(score[0]) if len(score) > 1 else 1.0 - float(score[0]),
                    'neutral': 0.0,  # No neutral class in binary models
                    'compound': float(score[1] - score[0]) if len(score) > 1 else float(score[0] * 2 - 1)
                })
        
        return results
    
    def analyze_sentiment(self, df: pd.DataFrame, text_cols: List[str] = ['text']) -> pd.DataFrame:
        """
        Analyze sentiment for texts in DataFrame.
        
        Args:
            df: DataFrame containing texts to analyze
            text_cols: Columns containing text to analyze
            
        Returns:
            DataFrame with added sentiment scores
        """
        if not self.initialized:
            logger.error("BERT sentiment analyzer not initialized")
            return df
        
        if df.empty:
            logger.warning("Empty DataFrame provided for sentiment analysis")
            return df
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Combine text columns for each row
        texts = []
        for _, row in result_df.iterrows():
            # For posts, combine title and text
            combined_text = ""
            if 'type' in row and row['type'] == 'post' and 'title' in row:
                combined_text += row['title'] + ". "
            
            # Add text from each specified column
            for col in text_cols:
                if col in row and row[col]:
                    combined_text += str(row[col]) + " "
            
            texts.append(combined_text.strip())
        
        # Initialize sentiment columns
        result_df['sentiment_pos'] = 0.0
        result_df['sentiment_neg'] = 0.0
        result_df['sentiment_neu'] = 0.0
        result_df['sentiment_compound'] = 0.0
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_results = self.analyze_batch(batch_texts)
            
            for j, result in enumerate(batch_results):
                idx = i + j
                if idx < len(result_df):
                    result_df.at[idx, 'sentiment_pos'] = result['positive']
                    result_df.at[idx, 'sentiment_neg'] = result['negative']
                    result_df.at[idx, 'sentiment_neu'] = result['neutral']
                    result_df.at[idx, 'sentiment_compound'] = result['compound']
        
        logger.info(f"Analyzed sentiment for {len(result_df)} items using BERT")
        
        return result_df


class RedditScraper:
    """
    Class for scraping posts from Reddit.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        """
        Initialize the Reddit scraper.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
            config_file: Path to config file with Reddit API credentials
        """
        self.credentials = {}
        
        # Try to load credentials from config file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.credentials = json.load(f)
        
        # Override with provided credentials if any
        if client_id:
            self.credentials['client_id'] = client_id
        if client_secret:
            self.credentials['client_secret'] = client_secret
        if user_agent:
            self.credentials['user_agent'] = user_agent
        
        self.reddit = None
        self.initialized = False
        
        # Check if we have all required credentials
        if self._check_credentials():
            self._initialize_reddit()
    
    def _check_credentials(self) -> bool:
        """
        Check if we have all required credentials.
        
        Returns:
            True if all credentials are available, False otherwise
        """
        required_keys = ['client_id', 'client_secret', 'user_agent']
        has_all_keys = all(key in self.credentials for key in required_keys)
        
        if not has_all_keys:
            missing_keys = [key for key in required_keys if key not in self.credentials]
            logger.warning(f"Missing Reddit API credentials: {', '.join(missing_keys)}")
            logger.info("Please set up credentials before using Reddit scraper")
        
        return has_all_keys
    
    def _initialize_reddit(self) -> None:
        """Initialize Reddit API client."""
        try:
            self.reddit = praw.Reddit(
                client_id=self.credentials['client_id'],
                client_secret=self.credentials['client_secret'],
                user_agent=self.credentials['user_agent']
            )
            self.initialized = True
            logger.info("Reddit API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API client: {str(e)}")
    
    def extract_posts(
        self,
        subreddits: List[str],
        keywords: List[str],
        time_period: str = 'week',
        limit: int = 100,
        include_comments: bool = True,
        comments_limit: int = 20
    ) -> pd.DataFrame:
        """
        Extract posts from specified subreddits matching keywords.
        
        Args:
            subreddits: List of subreddit names to search
            keywords: List of keywords to filter posts
            time_period: Time period to search ('day', 'week', 'month', 'year')
            limit: Maximum number of posts to retrieve per subreddit
            include_comments: Whether to include comments
            comments_limit: Maximum number of comments to retrieve per post
            
        Returns:
            DataFrame with extracted posts and metadata
        """
        if not self.initialized:
            logger.error("Reddit scraper not initialized")
            return pd.DataFrame()
        
        # Convert time period to appropriate filter
        time_filter = {
            'day': 'day',
            'week': 'week',
            'month': 'month',
            'year': 'year',
            'all': 'all'
        }.get(time_period.lower(), 'week')
        
        # Compile regex pattern for keywords
        pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b', re.IGNORECASE)
        
        posts_data = []
        comments_data = []
        
        # Iterate through subreddits
        for subreddit_name in subreddits:
            logger.info(f"Scraping r/{subreddit_name} for posts about {', '.join(keywords)}")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get top posts from the time period
                for post in subreddit.top(time_filter=time_filter, limit=limit):
                    # Check if post contains any of the keywords
                    if (
                        pattern.search(post.title) or 
                        (post.selftext and pattern.search(post.selftext))
                    ):
                        # Extract post data
                        post_data = {
                            'id': post.id,
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext,
                            'author': str(post.author),
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'url': post.url,
                            'type': 'post'
                        }
                        
                        posts_data.append(post_data)
                        
                        # Get comments if requested
                        if include_comments:
                            post.comments.replace_more(limit=0)  # Ignore "load more comments" links
                            for comment in list(post.comments)[:comments_limit]:
                                comment_data = {
                                    'id': comment.id,
                                    'post_id': post.id,
                                    'subreddit': subreddit_name,
                                    'text': comment.body,
                                    'author': str(comment.author),
                                    'score': comment.score,
                                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                                    'type': 'comment'
                                }
                                
                                comments_data.append(comment_data)
                
                # Avoid hitting Reddit API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
        
        # Combine posts and comments into a single DataFrame
        all_data = posts_data + comments_data
        
        if not all_data:
            logger.warning("No matching posts found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Sort by creation time
        df = df.sort_values('created_utc', ascending=False)
        
        logger.info(f"Extracted {len(posts_data)} posts and {len(comments_data)} comments")
        
        return df


class OilMarketSentimentAnalyzer:
    """
    Main class for oil market sentiment analysis.
    """
    
    def __init__(
        self,
        # Reddit API credentials
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        config_file: Optional[str] = "reddit_config.json",
        # BERT model settings
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Initialize the oil market sentiment analyzer.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
            config_file: Path to config file with Reddit API credentials
            model_name: Name of the BERT model to use
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Batch size for inference
        """
        # Initialize Reddit scraper
        self.scraper = RedditScraper(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            config_file=config_file
        )
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = BERTSentimentAnalyzer(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        
        self.initialized = self.scraper.initialized and self.sentiment_analyzer.initialized
        
        if self.initialized:
            logger.info("Oil market sentiment analyzer initialized")
        else:
            logger.warning("Oil market sentiment analyzer initialization incomplete")
    
    def extract_and_analyze(
        self,
        subreddits: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        time_period: str = 'week',
        limit: int = 100,
        include_comments: bool = True
    ) -> pd.DataFrame:
        """
        Extract posts and analyze sentiment.
        
        Args:
            subreddits: List of subreddit names to search
            keywords: List of keywords to filter posts
            time_period: Time period to search ('day', 'week', 'month', 'year')
            limit: Maximum number of posts to retrieve per subreddit
            include_comments: Whether to include comments
            
        Returns:
            DataFrame with extracted posts and sentiment scores
        """
        if not self.initialized:
            logger.error("Oil market sentiment analyzer not fully initialized")
            return pd.DataFrame()
        
        # Default subreddits related to oil markets
        if subreddits is None:
            subreddits = [
                'investing', 
                'stocks', 
                'wallstreetbets', 
                'energy', 
                'oil', 
                'commodities',
                'economics',
                'finance'
            ]
        
        # Default keywords related to oil
        if keywords is None:
            keywords = [
                'oil', 'crude', 'petroleum', 'brent', 'wti', 'opec',
                'shale', 'drilling', 'refinery', 'barrel',
                'energy', 'fossil fuel', 'gasoline', 'petrol',
                'diesel', 'production cut', 'supply', 'reserves'
            ]
        
        # Extract posts
        df = self.scraper.extract_posts(
            subreddits=subreddits,
            keywords=keywords,
            time_period=time_period,
            limit=limit,
            include_comments=include_comments
        )
        
        if df.empty:
            logger.warning("No posts extracted, skipping sentiment analysis")
            return df
        
        # Analyze sentiment
        sentiment_df = self.sentiment_analyzer.analyze_sentiment(df)
        
        return sentiment_df
    
    def extract_keywords(
        self, 
        df: pd.DataFrame, 
        n_keywords: int = 20,
        additional_stopwords: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Extract most frequent keywords from posts and comments.
        
        Args:
            df: DataFrame with posts and comments
            n_keywords: Number of top keywords to extract
            additional_stopwords: Additional stopwords to filter out
            
        Returns:
            Dictionary with keyword counts
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for keyword extraction")
            return {}
        
        # Combine all text
        all_text = ""
        
        for _, row in df.iterrows():
            # Add title for posts
            if row['type'] == 'post' and 'title' in row:
                all_text += row['title'] + " "
            
            # Add text content
            if 'text' in row and row['text']:
                all_text += row['text'] + " "
        
        # Tokenize
        tokens = word_tokenize(all_text.lower())
        
        # Extended stopwords
        stop_words = self.sentiment_analyzer.stop_words.copy()
        if additional_stopwords:
            stop_words.update(additional_stopwords)
        
        # Filter tokens
        # - Remove stopwords
        # - Remove short tokens
        # - Remove non-alphabetic tokens
        filtered_tokens = [
            token
            for token in tokens
            if (
                token not in stop_words and
                len(token) > 2 and
                token.isalpha()
            )
        ]
        
        # Count frequencies
        word_counts = Counter(filtered_tokens)
        
        # Get top keywords
        top_keywords = word_counts.most_common(n_keywords)
        
        logger.info(f"Extracted {len(top_keywords)} keywords")
        
        return dict(top_keywords)
    
    def aggregate_sentiment(
        self, 
        df: pd.DataFrame, 
        freq: str = 'D',
        weighted: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by time period.
        
        Args:
            df: DataFrame with sentiment scores
            freq: Frequency for aggregation ('D' for daily, 'W' for weekly, etc.)
            weighted: Whether to weight sentiment by post/comment score
            
        Returns:
            DataFrame with aggregated sentiment over time
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for sentiment aggregation")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        agg_df = df.copy()
        
        # Ensure datetime index
        agg_df['date'] = agg_df['created_utc'].dt.date
        
        # Create weight column if using weighted aggregation
        if weighted:
            # For posts and comments, use score as weight
            agg_df['weight'] = agg_df['score'].clip(lower=1)  # Ensure positive weight
            
            # Weighted average function
            def weighted_avg(series, weights):
                return np.average(series, weights=weights)
            
            # Group by date and calculate weighted average
            result = agg_df.groupby(pd.Grouper(key='date', freq=freq)).agg({
                'sentiment_compound': lambda x: weighted_avg(x, agg_df.loc[x.index, 'weight']),
                'sentiment_pos': lambda x: weighted_avg(x, agg_df.loc[x.index, 'weight']),
                'sentiment_neg': lambda x: weighted_avg(x, agg_df.loc[x.index, 'weight']),
                'sentiment_neu': lambda x: weighted_avg(x, agg_df.loc[x.index, 'weight']),
                'id': 'count'  # Count of posts/comments
            })
        else:
            # Simple average
            result = agg_df.groupby(pd.Grouper(key='date', freq=freq)).agg({
                'sentiment_compound': 'mean',
                'sentiment_pos': 'mean',
                'sentiment_neg': 'mean',
                'sentiment_neu': 'mean',
                'id': 'count'  # Count of posts/comments
            })
        
        # Rename count column
        result = result.rename(columns={'id': 'count'})
        
        # Fill missing dates
        result = result.resample(freq).asfreq().fillna(0)
        
        logger.info(f"Aggregated sentiment data to {freq} frequency")
        
        return result
    
    def calculate_sentiment_indicators(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment indicators from aggregated sentiment data.
        
        Args:
            sentiment_df: DataFrame with aggregated sentiment
            
        Returns:
            DataFrame with added sentiment indicators
        """
        if sentiment_df.empty:
            logger.warning("Empty DataFrame provided for calculating sentiment indicators")
            return sentiment_df
        
        # Make a copy to avoid modifying the original
        result_df = sentiment_df.copy()
        
        # Calculate sentiment moving averages
        windows = [3, 7, 14]
        for window in windows:
            result_df[f'sentiment_ma{window}'] = result_df['sentiment_compound'].rolling(window=window).mean()
        
        # Calculate sentiment momentum (rate of change)
        result_df['sentiment_momentum'] = result_df['sentiment_compound'].diff()
        
        # Calculate sentiment volatility
        result_df['sentiment_volatility'] = result_df['sentiment_compound'].rolling(window=7).std()
        
        # Calculate bullish/bearish bias
        result_df['bullish_bias'] = result_df['sentiment_pos'] / (result_df['sentiment_pos'] + result_df['sentiment_neg'])
        result_df['bearish_bias'] = result_df['sentiment_neg'] / (result_df['sentiment_pos'] + result_df['sentiment_neg'])
        
        # Handle division by zero
        result_df['bullish_bias'] = result_df['bullish_bias'].fillna(0.5)
        result_df['bearish_bias'] = result_df['bearish_bias'].fillna(0.5)
        
        # Calculate sentiment signal (simple threshold-based)
        result_df['sentiment_signal'] = 0
        result_df.loc[result_df['sentiment_compound'] > 0.2, 'sentiment_signal'] = 1  # Bullish
        result_df.loc[result_df['sentiment_compound'] < -0.2, 'sentiment_signal'] = -1  # Bearish
        
        logger.info("Calculated sentiment indicators")
        
        return result_df
    
    def save_sentiment_data(
        self, 
        sentiment_df: pd.DataFrame, 
        filename: str = 'sentiment_data.csv',
        output_dir: str = 'data/processed'
    ) -> None:
        """
        Save sentiment data to a CSV file.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            filename: Output filename
            output_dir: Output directory
        """
        if sentiment_df.empty:
            logger.warning("Empty DataFrame provided for saving")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, filename)
        sentiment_df.to_csv(output_path)
        
        logger.info(f"Saved sentiment data to {output_path}")
    
    def load_sentiment_data(
        self, 
        filename: str = 'sentiment_data.csv',
        input_dir: str = 'data/processed'
    ) -> pd.DataFrame:
        """
        Load sentiment data from a CSV file.
        
        Args:
            filename: Input filename
            input_dir: Input directory
            
        Returns:
            DataFrame with sentiment data
        """
        input_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_path):
            logger.warning(f"Sentiment data file not found: {input_path}")
            return pd.DataFrame()
        
        # Load from CSV
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        
        logger.info(f"Loaded sentiment data from {input_path}")
        
        return df
    
    def plot_sentiment_over_time(
        self, 
        sentiment_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot sentiment over time.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if sentiment_df.empty:
            logger.warning("Empty DataFrame provided for plotting")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot compound sentiment
        ax1.plot(sentiment_df.index, sentiment_df['sentiment_compound'], 'b-', linewidth=2, label='Compound Sentiment')
        
        # Plot sentiment moving averages if available
        if 'sentiment_ma7' in sentiment_df.columns:
            ax1.plot(sentiment_df.index, sentiment_df['sentiment_ma7'], 'r--', linewidth=1.5, label='7-day MA')
        
        if 'sentiment_ma14' in sentiment_df.columns:
            ax1.plot(sentiment_df.index, sentiment_df['sentiment_ma14'], 'g--', linewidth=1.5, label='14-day MA')
        
        # Add horizontal lines at meaningful levels
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax1.axhline(y=0.2, color='g', linestyle=':', alpha=0.3, label='Bullish Threshold')
        ax1.axhline(y=-0.2, color='r', linestyle=':', alpha=0.3, label='Bearish Threshold')
        
        # Format first subplot
        ax1.set_title('Oil Market Sentiment Analysis (BERT)')
        ax1.set_ylabel('Sentiment Score')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot post/comment count in second subplot
        ax2.bar(sentiment_df.index, sentiment_df['count'], color='steelblue', alpha=0.7, label='Post/Comment Count')
        
        # Format second subplot
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment plot to {save_path}")
        
        return fig
    
    def plot_sentiment_distribution(
        self,
        sentiment_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of sentiment scores.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if sentiment_df.empty:
            logger.warning("Empty DataFrame provided for plotting")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram of compound sentiment
        ax.hist(
            sentiment_df['sentiment_compound'],
            bins=20,
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add vertical lines at meaningful levels
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Neutral')
        ax.axvline(x=0.2, color='g', linestyle='--', alpha=0.5, label='Bullish Threshold')
        ax.axvline(x=-0.2, color='r', linestyle='--', alpha=0.5, label='Bearish Threshold')
        
        # Format plot
        ax.set_title('Distribution of Oil Market Sentiment Scores')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment distribution plot to {save_path}")
        
        return fig
    
    def plot_keyword_frequency(
        self,
        keywords: Dict[str, int],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot frequency of extracted keywords.
        
        Args:
            keywords: Dictionary with keyword counts
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not keywords:
            logger.warning("Empty keywords dictionary provided for plotting")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort keywords by frequency
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        words, counts = zip(*sorted_keywords)
        
        # Limit to top 20 if more than 20 keywords
        if len(words) > 20:
            words = words[:20]
            counts = counts[:20]
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, color='steelblue', alpha=0.7)
        
        # Add word labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(count + 0.5, i, str(count), va='center')
        
        # Format plot
        ax.set_title('Top Keywords in Oil Market Discussions')
        ax.set_xlabel('Frequency')
        ax.invert_yaxis()  # Words read top-to-bottom
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved keyword frequency plot to {save_path}")
        
        return fig


def run_sentiment_analysis(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    user_agent: Optional[str] = None,
    config_file: str = 'reddit_config.json',
    model_name: str = "ProsusAI/finbert",
    output_dir: str = 'data/processed',
    visualize_dir: str = 'notebooks/plots',
    days_back: int = 30
) -> Optional[pd.DataFrame]:
    """
    Run the complete sentiment analysis pipeline.
    
    Args:
        client_id: Reddit API client ID
        client_secret: Reddit API client secret
        user_agent: Reddit API user agent
        config_file: Path to config file with Reddit API credentials
        model_name: Name of the BERT model to use
        output_dir: Output directory for results
        visualize_dir: Directory for visualization outputs
        days_back: Number of days to analyze
        
    Returns:
        DataFrame with sentiment data or None if analysis fails
    """
    try:
        # Initialize the sentiment analyzer
        analyzer = OilMarketSentimentAnalyzer(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            config_file=config_file,
            model_name=model_name
        )
        
        if not analyzer.initialized:
            logger.error("Failed to initialize sentiment analyzer")
            return None
        
        # Determine time period based on days_back
        if days_back <= 1:
            time_period = 'day'
        elif days_back <= 7:
            time_period = 'week'
        elif days_back <= 31:
            time_period = 'month'
        else:
            time_period = 'year'
        
        # Extract posts and analyze sentiment
        sentiment_data = analyzer.extract_and_analyze(
            time_period=time_period,
            limit=200  # Increase limit for more data
        )
        
        if sentiment_data.empty:
            logger.error("No sentiment data extracted")
            return None
        
        # Extract keywords
        keywords = analyzer.extract_keywords(
            sentiment_data,
            n_keywords=30,
            additional_stopwords=['oil', 'price', 'prices', 'market', 'markets']
        )
        
        # Aggregate sentiment by day
        daily_sentiment = analyzer.aggregate_sentiment(
            sentiment_data,
            freq='D',
            weighted=True
        )
        
        # Calculate sentiment indicators
        indicators = analyzer.calculate_sentiment_indicators(daily_sentiment)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        
        # Save sentiment data
        analyzer.save_sentiment_data(
            sentiment_data,
            filename='raw_sentiment_data.csv',
            output_dir=output_dir
        )
        
        # Save aggregated sentiment
        analyzer.save_sentiment_data(
            indicators,
            filename='oil_market_sentiment_indicators.csv',
            output_dir=output_dir
        )
        
        # Generate visualizations
        analyzer.plot_sentiment_over_time(
            indicators,
            save_path=f"{visualize_dir}/oil_sentiment_trend.png"
        )
        
        analyzer.plot_sentiment_distribution(
            sentiment_data,
            save_path=f"{visualize_dir}/oil_sentiment_distribution.png"
        )
        
        analyzer.plot_keyword_frequency(
            keywords,
            save_path=f"{visualize_dir}/oil_keyword_frequency.png"
        )
        
        logger.info("Sentiment analysis completed successfully")
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline: {str(e)}")
        return None


if __name__ == "__main__":
    """
    Example usage of the sentiment analysis module.
    
    For actual usage, create a reddit_config.json file with your Reddit API credentials:
    {
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "user_agent": "your_user_agent"
    }
    
    Or provide credentials directly when calling run_sentiment_analysis().
    """
    # Check if we have GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")
    
    # Create a dummy Reddit config if not found (for demonstration)
    config_path = 'reddit_config.json'
    if not os.path.exists(config_path):
        logger.warning(f"Reddit config file not found at {config_path}")
        logger.info("Creating a placeholder config file - you need to update it with real credentials")
        
        with open(config_path, 'w') as f:
            json.dump({
                "client_id": "YOUR_CLIENT_ID",
                "client_secret": "YOUR_CLIENT_SECRET",
                "user_agent": "YOUR_USER_AGENT (by /u/YOUR_USERNAME)"
            }, f, indent=4)
    
    # Run analysis using FinBERT model
    print("To run sentiment analysis, uncomment the following lines and provide valid credentials:")
    print("# sentiment_data = run_sentiment_analysis(")
    print("#     model_name='ProsusAI/finbert',")
    print("#     days_back=30")
    print("# )")
    
    # Explain how to use
    print("\nTo use this module:")
    print("1. Update reddit_config.json with your Reddit API credentials")
    print("2. Install required packages: transformers, torch, praw")
    print("3. Run this script to analyze oil market sentiment")