"""
Enhanced Historical Data Scraper for Oil Prophet

This module provides advanced functionality to gather sentiment data from multiple sources:
1. Reddit API (official) for modern data (2008-present)
2. News archives via API integration for historical data (1987-2008)
3. Simulated academic datasets for supplemental historical sentiment

The scraper implements:
- Comprehensive time coverage from 1987 to present
- Efficient batched processing with threading
- Consistent data schema across time periods and sources
- Robust error handling and retry logic
- Detailed logging and progress tracking
"""

import datetime
import time
import json
import os
import logging
import pandas as pd
import numpy as np
import praw
import requests
import random
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.parser import parse as parse_date
import matplotlib.pyplot as plt
from urllib.parse import quote_plus
import backoff
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("historical_data_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedRedditScraper:
    """
    Enhanced scraper for gathering Reddit and historical news data spanning 1987-present.
    
    Features:
    - Unified data schema across all time periods and sources
    - Incremental scraping with resume capability
    - Advanced filtering for oil market relevance
    - Data quality metrics and validation
    """
    
    # Default oil-related keywords (expanded list)
    OIL_KEYWORDS = [
        # Oil types and benchmarks
        "oil", "crude oil", "petroleum", "brent crude", "wti crude", 
        "light sweet crude", "heavy crude", "west texas intermediate", "oil futures",
        
        # Oil organizations and market participants
        "opec", "opec+", "oil cartel", "saudi aramco", "exxonmobil", "chevron", 
        "bp", "shell", "conocophillips", "oil producers", "oil exporters",
        "oil importers", "oil traders", "oil investors", "oil speculators",
        
        # Oil production and industry terms
        "oil drilling", "oil refining", "oil refinery", "barrel of oil", 
        "oil field", "oil production", "oil platform", "offshore drilling",
        "fracking", "hydraulic fracturing", "oil sands", "tar sands",
        "shale oil", "oil reserves", "proven reserves", "oil inventory",
        
        # Oil market terminology
        "oil price", "oil prices", "oil market", "oil supply", "oil demand",
        "energy prices", "fossil fuel", "gasoline price", "petrol price",
        "diesel price", "jet fuel", "oil futures", "oil ETF", "oil fund",
        "oil stock", "oil stocks", "energy sector", "energy market",
        
        # Oil market events/conditions
        "production cut", "production increase", "oil glut", "oil shortage",
        "oil surplus", "oil deficit", "oil crisis", "oil shock", "price war",
        "oil embargo", "oil sanctions", "strategic reserve", "oil storage",
        
        # Oil price movements
        "oil rally", "oil selloff", "oil crash", "oil collapse", "oil volatility", 
        "oil rebound", "oil decline", "oil slump", "oil surge", "oil spike"
    ]
    
    # Default subreddits relevant to oil markets
    OIL_SUBREDDITS = [
        # Financial/investment subreddits
        "investing", "stocks", "stockmarket", "wallstreetbets", "options",
        "finance", "SecurityAnalysis", "economics", "economy", "business",
        
        # Energy-specific subreddits
        "energy", "oil", "oilandgasworkers", "oilandgas", "commodities",
        "renewableenergy", "fossilfuels", "petroleum",
        
        # News and politics (for major oil events)
        "worldnews", "news", "geopolitics", "politics", "environment",
        "globalmarkets", "worldeconomics", "worldpolitics"
    ]
    
    # Pre-Reddit news sources for historical data
    HISTORICAL_NEWS_SOURCES = [
        {"name": "nytimes", "api_key_name": "nytimes_api_key", "start_year": 1987, "end_year": 2023},
        {"name": "reuters", "api_key_name": "reuters_api_key", "start_year": 1987, "end_year": 2023},
        {"name": "ft", "api_key_name": "ft_api_key", "start_year": 1990, "end_year": 2023},
        {"name": "wsj", "api_key_name": "wsj_api_key", "start_year": 1989, "end_year": 2023},
        {"name": "archive-org", "api_key_name": None, "start_year": 1987, "end_year": 2010},
        {"name": "simulated", "api_key_name": None, "start_year": 1987, "end_year": 2010}
    ]
    
    # Financial sentiment datasets (for supplemental data)
    FINANCIAL_DATASETS = [
        {"name": "fin-phrasebank", "url": "https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10"},
        {"name": "semeval-2017", "url": "https://alt.qcri.org/semeval2017/task5/"},
        {"name": "fiqa", "url": "https://sites.google.com/view/fiqa/home"},
    ]
    
    def __init__(
        self,
        output_dir: str = "data/processed/historical",
        config_path: str = "reddit_config.json",
        cache_dir: str = "cache",
        max_threads: int = 8,
        max_retries: int = 5,
        retry_delay: int = 2,
        verbose: bool = True
    ):
        """
        Initialize the enhanced scraper with robust configuration.
        
        Args:
            output_dir: Directory to save scraped data
            config_path: Path to Reddit API configuration
            cache_dir: Directory to cache API responses
            max_threads: Maximum number of concurrent threads
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries (seconds)
            verbose: Whether to print verbose output
        """
        # Setup directories
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup logging level
        self.verbose = verbose
        if not verbose:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Reddit client
        self.reddit = self._init_reddit_client()
        
        # Request parameters
        self.max_threads = max_threads
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup session for requests
        self.session = requests.Session()
        
        # Initialize statistics
        self.stats = {
            "start_time": datetime.datetime.now().isoformat(),
            "requests": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "retried": 0,
                "cached": 0
            },
            "items_collected": {
                "reddit_posts": 0,
                "reddit_comments": 0,
                "news_articles": 0,
                "financial_data": 0,
                "total": 0
            },
            "sources": {},
            "timespan": {
                "start": None,
                "end": None
            }
        }
        
        # Load any existing state
        self.state = self._load_scraping_state()
        
        logger.info(f"Enhanced Reddit Scraper initialized with {max_threads} threads")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            self._create_default_config(config_path)
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def _create_default_config(self, config_path: str) -> None:
        """
        Create a default configuration file.
        
        Args:
            config_path: Path to save configuration
        """
        default_config = {
            "reddit": {
                "client_id": "YOUR_CLIENT_ID",
                "client_secret": "YOUR_CLIENT_SECRET",
                "user_agent": "OilProphet/1.0 (by /u/YOUR_USERNAME)"
            },
            "api_keys": {
                "nytimes_api_key": "",
                "reuters_api_key": "",
                "ft_api_key": "",
                "wsj_api_key": ""
            },
            "scraping": {
                "batch_size": 100,
                "max_requests_per_minute": 30,
                "max_posts_per_subreddit": 5000,
                "max_comments_per_post": 500
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {str(e)}")
    
    def _init_reddit_client(self) -> Optional[praw.Reddit]:
        """
        Initialize Reddit API client.
        
        Returns:
            PRAW Reddit client instance
        """
        if not self.config or "reddit" not in self.config:
            logger.error("Reddit configuration missing")
            return None
        
        try:
            reddit_config = self.config["reddit"]
            reddit = praw.Reddit(
                client_id=reddit_config.get("client_id"),
                client_secret=reddit_config.get("client_secret"),
                user_agent=reddit_config.get("user_agent")
            )
            logger.info("Reddit client initialized successfully")
            return reddit
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {str(e)}")
            return None
    
    def _load_scraping_state(self) -> Dict[str, Any]:
        """
        Load previous scraping state if available.
        
        Returns:
            Dictionary with scraping state
        """
        state_path = os.path.join(self.output_dir, "scraping_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded previous scraping state from {state_path}")
                return state
            except Exception as e:
                logger.error(f"Error loading scraping state: {str(e)}")
        
        return {
            "completed_subreddits": [],
            "completed_news_sources": [],
            "processed_dates": {},
            "last_processed": {
                "reddit": None,
                "news": None,
                "financial": None
            }
        }
    
    def _save_scraping_state(self) -> None:
        """Save current scraping state to allow resuming later."""
        state_path = os.path.join(self.output_dir, "scraping_state.json")
        try:
            with open(state_path, 'w') as f:
                json.dump(self.state, f, indent=4)
            logger.info(f"Saved scraping state to {state_path}")
        except Exception as e:
            logger.error(f"Error saving scraping state: {str(e)}")
    
    def _save_stats(self) -> None:
        """Save scraping statistics."""
        stats_path = os.path.join(self.output_dir, "scraping_stats.json")
        
        # Update end time
        self.stats["end_time"] = datetime.datetime.now().isoformat()
        self.stats["duration_seconds"] = (datetime.datetime.fromisoformat(self.stats["end_time"]) - 
                                          datetime.datetime.fromisoformat(self.stats["start_time"])).total_seconds()
        
        try:
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=4)
            logger.info(f"Saved scraping statistics to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving statistics: {str(e)}")
    
    @backoff.on_exception(
        backoff.expo, 
        (requests.exceptions.RequestException, praw.exceptions.PRAWException),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def _make_api_request(
        self,
        url: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        method: str = "GET",
        api_name: str = "generic",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make an API request with caching, retries and backoff.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            headers: HTTP headers
            method: HTTP method (GET or POST)
            api_name: Name of the API for stats tracking
            use_cache: Whether to use cache
            
        Returns:
            API response as dictionary
        """
        # Track request in stats
        self.stats["requests"]["total"] += 1
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            param_str = json.dumps(params, sort_keys=True) if params else ""
            cache_key = f"{api_name}_{url}_{param_str}.json".replace("/", "_")
            cache_path = os.path.join(self.cache_dir, cache_key)
            
            # Check if cached response exists
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    logger.debug(f"Using cached response for {api_name}")
                    self.stats["requests"]["cached"] += 1
                    return cached_data
                except Exception as e:
                    logger.warning(f"Failed to load cached response: {str(e)}")
        
        # Set default headers
        if headers is None:
            headers = {
                'User-Agent': 'OilProphet Research Scraper/1.0 (Academic Research Project)'
            }
        
        # Make the request
        try:
            if method.upper() == "POST":
                response = self.session.post(url, json=params, headers=headers, timeout=30)
            else:
                response = self.session.get(url, params=params, headers=headers, timeout=30)
            
            # Check for successful response
            if response.status_code == 200:
                self.stats["requests"]["successful"] += 1
                
                # Parse JSON response
                try:
                    data = response.json()
                    
                    # Cache the response if caching is enabled
                    if use_cache and cache_key:
                        cache_path = os.path.join(self.cache_dir, cache_key)
                        try:
                            with open(cache_path, 'w') as f:
                                json.dump(data, f)
                        except Exception as e:
                            logger.warning(f"Failed to cache response: {str(e)}")
                    
                    return data
                except ValueError:
                    logger.error(f"Invalid JSON response from {api_name}")
                    self.stats["requests"]["failed"] += 1
                    return {}
            
            logger.warning(f"API request failed with status {response.status_code}: {response.text[:100]}...")
            self.stats["requests"]["failed"] += 1
            return {}
            
        except Exception as e:
            logger.error(f"Request exception for {api_name}: {str(e)}")
            self.stats["requests"]["failed"] += 1
            raise
    
    def fetch_reddit_data(
        self,
        subreddits: List[str] = None,
        keywords: List[str] = None,
        start_date: Union[str, datetime.datetime] = "2008-01-01",
        end_date: Union[str, datetime.datetime] = None,
        include_comments: bool = True,
        posts_per_subreddit: int = 1000,
        filter_quality: bool = True,
        min_score: int = 3,
        min_relevance: float = 0.5,
        skip_completed: bool = True
    ) -> pd.DataFrame:
        """
        Fetch Reddit data across multiple subreddits with advanced filtering.
        
        Args:
            subreddits: List of subreddits to scrape (default: self.OIL_SUBREDDITS)
            keywords: List of keywords to filter (default: self.OIL_KEYWORDS)
            start_date: Start date for data collection
            end_date: End date for data collection
            include_comments: Whether to fetch comments for posts
            posts_per_subreddit: Maximum posts to fetch per subreddit
            filter_quality: Whether to filter for quality posts
            min_score: Minimum score for quality posts
            min_relevance: Minimum relevance score (0-1) for posts
            skip_completed: Whether to skip already completed subreddits
            
        Returns:
            DataFrame with collected Reddit data
        """
        # Use default lists if not provided
        if subreddits is None:
            subreddits = self.OIL_SUBREDDITS
        if keywords is None:
            keywords = self.OIL_KEYWORDS
        
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date is None:
            end_date = datetime.datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        logger.info(f"Fetching Reddit data from {start_date.date()} to {end_date.date()}")
        logger.info(f"Target subreddits: {', '.join(subreddits)}")
        
        # Check if Reddit client is initialized
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return pd.DataFrame()
        
        # Filter out completed subreddits if requested
        if skip_completed and "completed_subreddits" in self.state:
            remaining_subreddits = [s for s in subreddits if s not in self.state["completed_subreddits"]]
            if len(remaining_subreddits) < len(subreddits):
                logger.info(f"Skipping {len(subreddits) - len(remaining_subreddits)} completed subreddits")
                subreddits = remaining_subreddits
        
        all_data = []
        
        # Process each subreddit
        for subreddit_name in subreddits:
            logger.info(f"Processing subreddit: r/{subreddit_name}")
            
            try:
                # Get subreddit instance
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Collect posts using multiple search methods for better coverage
                posts = self._collect_subreddit_posts(
                    subreddit=subreddit,
                    keywords=keywords,
                    start_date=start_date,
                    end_date=end_date,
                    max_posts=posts_per_subreddit,
                    filter_quality=filter_quality,
                    min_score=min_score,
                    min_relevance=min_relevance
                )
                
                if not posts:
                    logger.warning(f"No relevant posts found in r/{subreddit_name}")
                    continue
                
                logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
                
                # Save posts for this subreddit
                posts_df = pd.DataFrame(posts)
                posts_output_path = os.path.join(self.output_dir, f"reddit_posts_{subreddit_name}.csv")
                posts_df.to_csv(posts_output_path, index=False)
                
                # Collect comments if requested
                if include_comments and posts:
                    post_ids = [post["id"] for post in posts]
                    comments = self._collect_reddit_comments(
                        post_ids=post_ids,
                        max_comments_per_post=100 if filter_quality else 500
                    )
                    
                    if comments:
                        logger.info(f"Collected {len(comments)} comments from r/{subreddit_name} posts")
                        
                        # Save comments for this subreddit
                        comments_df = pd.DataFrame(comments)
                        comments_output_path = os.path.join(self.output_dir, f"reddit_comments_{subreddit_name}.csv")
                        comments_df.to_csv(comments_output_path, index=False)
                        
                        # Add to all data
                        all_data.extend(comments)
                
                # Add posts to all data
                all_data.extend(posts)
                
                # Update state
                if "completed_subreddits" not in self.state:
                    self.state["completed_subreddits"] = []
                self.state["completed_subreddits"].append(subreddit_name)
                self._save_scraping_state()
                
                # Update stats
                self.stats["items_collected"]["reddit_posts"] += len(posts)
                if include_comments:
                    self.stats["items_collected"]["reddit_comments"] += sum(len(post.get("comments", [])) for post in posts)
                
                # Be nice to the Reddit API
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing subreddit {subreddit_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Create combined DataFrame
        if all_data:
            combined_df = pd.DataFrame(all_data)
            
            # Save combined data
            output_path = os.path.join(self.output_dir, "reddit_combined_data.csv")
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(combined_df)} total items to {output_path}")
            
            # Update stats
            self.stats["items_collected"]["total"] += len(combined_df)
            self._save_stats()
            
            return combined_df
        
        logger.warning("No Reddit data collected")
        return pd.DataFrame()
    
    def _collect_subreddit_posts(
        self,
        subreddit: praw.models.Subreddit,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_posts: int = 1000,
        filter_quality: bool = True,
        min_score: int = 3,
        min_relevance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Collect posts from a specific subreddit with filtering.
        
        Args:
            subreddit: PRAW Subreddit instance
            keywords: List of keywords to filter
            start_date: Start date for collection
            end_date: End date for collection
            max_posts: Maximum posts to collect
            filter_quality: Whether to filter for quality
            min_score: Minimum score for quality posts
            min_relevance: Minimum relevance score
            
        Returns:
            List of post dictionaries
        """
        collected_posts = []
        
        # Use multiple search methods for comprehensive coverage
        search_methods = [
            ("search", self._search_subreddit_by_keywords(subreddit, keywords, max_posts//3)),
            ("new", subreddit.new(limit=max_posts//3)),
            ("top", subreddit.top(limit=max_posts//3, time_filter="all"))
        ]
        
        for method_name, post_generator in search_methods:
            logger.info(f"Collecting from r/{subreddit.display_name} using {method_name} method")
            
            try:
                posts_collected = 0
                with tqdm(total=max_posts//3, desc=f"{method_name} posts") as pbar:
                    for post in post_generator:
                        # Check if within date range
                        post_date = datetime.datetime.fromtimestamp(post.created_utc)
                        if post_date.tzinfo is None and start_date.tzinfo is not None:
                            post_date = post_date.replace(tzinfo=datetime.timezone.utc)
                        elif start_date.tzinfo is None and post_date.tzinfo is not None:
                            start_date = start_date.replace(tzinfo=datetime.timezone.utc)
                            end_date = end_date.replace(tzinfo=datetime.timezone.utc)
                        
                        if post_date < start_date or post_date > end_date:
                            continue
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_oil_relevance(
                            title=post.title,
                            text=post.selftext,
                            keywords=keywords
                        )
                        
                        # Skip if not relevant enough
                        if relevance_score < min_relevance:
                            continue
                        
                        # Apply quality filter if enabled
                        if filter_quality and (post.score < min_score or post.num_comments < 1):
                            continue
                        
                        # Process post
                        processed_post = self._process_reddit_post(post, relevance_score)
                        
                        # Add to collection
                        collected_posts.append(processed_post)
                        posts_collected += 1
                        pbar.update(1)
                        
                        # Check if we've reached the limit
                        if posts_collected >= max_posts//3:
                            break
                            
            except Exception as e:
                logger.error(f"Error collecting {method_name} posts from r/{subreddit.display_name}: {str(e)}")
        
        # Remove duplicates based on post ID
        unique_posts = {post["id"]: post for post in collected_posts}
        return list(unique_posts.values())
    
    def _search_subreddit_by_keywords(
        self,
        subreddit: praw.models.Subreddit,
        keywords: List[str],
        limit: int = 500
    ) -> praw.models.listing.generator.ListingGenerator:
        """
        Search a subreddit for posts matching keywords.
        
        Args:
            subreddit: PRAW Subreddit instance
            keywords: List of keywords to search for
            limit: Maximum posts to return
            
        Returns:
            Generator for search results
        """
        # Group keywords into chunks for efficient searching
        keyword_chunks = self._chunk_keywords(keywords, max_chunks=5)
        
        # For each keyword chunk, create a search query
        search_results = []
        for keyword_chunk in keyword_chunks:
            # Join keywords with OR operator
            query = " OR ".join(f'"{keyword}"' for keyword in keyword_chunk)
            
            # Create a generator for the search
            search_generator = subreddit.search(query, sort="relevance", time_filter="all", limit=limit//len(keyword_chunks))
            search_results.append(search_generator)
        
        # Chain the search generators
        return praw.models.listing.generator.ListingGenerator(
            praw.Reddit._objector, praw.models.Subreddit, [],
            None, limit, "search", None, None, None, True
        )
        
    def _search_subreddit_by_keywords(
        self,
        subreddit: praw.models.Subreddit,
        keywords: List[str],
        limit: int = 500
    ) -> praw.models.listing.generator.ListingGenerator:
        """
        Search a subreddit for posts matching keywords.
        
        Args:
            subreddit: PRAW Subreddit instance
            keywords: List of keywords to search for
            limit: Maximum posts to return
            
        Returns:
            Generator for search results
        """
        # Take a sample of keywords if there are too many
        if len(keywords) > 5:
            search_keywords = random.sample(keywords, 5)
        else:
            search_keywords = keywords
        
        # Create a simple query with OR operators
        query = " OR ".join(f'"{keyword}"' for keyword in search_keywords)
        
        # Return the search generator
        return subreddit.search(query, sort="relevance", time_filter="all", limit=limit)
    
    def _chunk_keywords(
        self,
        keywords: List[str],
        max_chunks: int = 5
    ) -> List[List[str]]:
        """
        Divide keywords into chunks for efficient searching.
        
        Args:
            keywords: List of keywords to chunk
            max_chunks: Maximum number of chunks
            
        Returns:
            List of keyword lists
        """
        if len(keywords) <= max_chunks:
            return [[keyword] for keyword in keywords[:max_chunks]]
        
        # Calculate chunk size
        chunk_size = len(keywords) // max_chunks
        
        # Create chunks
        chunks = []
        for i in range(0, len(keywords), chunk_size):
            chunk = keywords[i:i + chunk_size]
            if chunk:  # Ensure chunk is not empty
                chunks.append(chunk)
            
            # Ensure we don't exceed max_chunks
            if len(chunks) >= max_chunks:
                break
        
        return chunks
    
    def _calculate_oil_relevance(
        self,
        title: str,
        text: str,
        keywords: List[str]
    ) -> float:
        """
        Calculate the relevance of a post to oil markets.
        
        Args:
            title: Post title
            text: Post text content
            keywords: Oil-related keywords
            
        Returns:
            Relevance score (0-1)
        """
        # Convert to lowercase for case-insensitive matching
        title_lower = title.lower()
        text_lower = text.lower() if text else ""
        
        # Count keyword occurrences
        title_matches = sum(keyword.lower() in title_lower for keyword in keywords)
        text_matches = sum(keyword.lower() in text_lower for keyword in keywords) if text else 0
        
        # Calculate relevance score
        # Title matches are weighted more heavily
        title_weight = 3.0
        text_weight = 1.0
        
        # Normalize by text length
        title_words = len(title_lower.split())
        text_words = len(text_lower.split()) if text else 0
        
        # Calculate density scores
        title_density = title_matches / max(1, title_words) * 100
        text_density = text_matches / max(1, text_words) * 100 if text else 0
        
        # Calculate overall relevance
        relevance = (title_density * title_weight + text_density * text_weight) / (title_weight + text_weight)
        
        # Normalize to 0-1 range
        relevance = min(1.0, relevance / 20.0)  # Normalize, capping at 1.0
        
        return relevance
    
    def _process_reddit_post(
        self,
        post: praw.models.Submission,
        relevance_score: float
    ) -> Dict[str, Any]:
        """
        Process a Reddit post into standardized format.
        
        Args:
            post: PRAW Submission object
            relevance_score: Calculated relevance score
            
        Returns:
            Dictionary with processed post data
        """
        # Extract creation date
        created_date = datetime.datetime.fromtimestamp(post.created_utc, datetime.timezone.utc)
        
        # Determine relevance category
        if relevance_score >= 0.7:
            relevance_category = "high"
        elif relevance_score >= 0.4:
            relevance_category = "medium"
        else:
            relevance_category = "low"
        
        # Create standardized post object
        processed_post = {
            "id": post.id,
            "type": "post",
            "source": "reddit",
            "subreddit": post.subreddit.display_name,
            "title": post.title,
            "text": post.selftext,
            "url": post.url,
            "permalink": f"https://reddit.com{post.permalink}",
            "author": str(post.author) if post.author else "[deleted]",
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            "created_date": created_date.strftime("%Y-%m-%d %H:%M:%S"),
            "is_self": post.is_self,
            "relevance_score": relevance_score,
            "relevance": relevance_category
        }
        
        return processed_post
    
    def _collect_reddit_comments(
        self,
        post_ids: List[str],
        max_comments_per_post: int = 100,
        filter_quality: bool = True,
        min_score: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for specified Reddit posts.
        
        Args:
            post_ids: List of post IDs to fetch comments for
            max_comments_per_post: Maximum comments to fetch per post
            filter_quality: Whether to filter for quality
            min_score: Minimum score for quality comments
            
        Returns:
            List of comment dictionaries
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return []
        
        all_comments = []
        
        logger.info(f"Collecting comments for {len(post_ids)} posts")
        
        # Process each post
        for post_id in tqdm(post_ids, desc="Fetching comments"):
            try:
                # Get submission
                submission = self.reddit.submission(id=post_id)
                
                # Expand comment forest (limited by max_comments_per_post)
                submission.comments.replace_more(limit=max_comments_per_post // 10)
                comments = submission.comments.list()
                
                # Limit to max_comments_per_post
                if len(comments) > max_comments_per_post:
                    # Prioritize higher-scored comments
                    comments = sorted(comments, key=lambda c: c.score, reverse=True)[:max_comments_per_post]
                
                # Process comments
                post_comments = []
                for comment in comments:
                    # Skip if low quality
                    if filter_quality and (not hasattr(comment, "score") or comment.score < min_score):
                        continue
                    
                    # Skip removed or deleted comments
                    if not hasattr(comment, "body") or comment.body in ["[removed]", "[deleted]"]:
                        continue
                    
                    # Process comment
                    processed_comment = self._process_reddit_comment(comment, post_id)
                    post_comments.append(processed_comment)
                
                # Add to collection
                all_comments.extend(post_comments)
                
                # Be nice to the Reddit API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching comments for post {post_id}: {str(e)}")
        
        logger.info(f"Collected {len(all_comments)} comments total")
        return all_comments
    
    def _process_reddit_comment(
        self,
        comment: praw.models.Comment,
        post_id: str
    ) -> Dict[str, Any]:
        """
        Process a Reddit comment into standardized format.
        
        Args:
            comment: PRAW Comment object
            post_id: ID of the parent post
            
        Returns:
            Dictionary with processed comment data
        """
        # Extract creation date
        created_date = datetime.datetime.fromtimestamp(comment.created_utc, datetime.timezone.utc)
        
        # Create standardized comment object
        processed_comment = {
            "id": comment.id,
            "post_id": post_id,
            "type": "comment",
            "source": "reddit",
            "subreddit": comment.subreddit.display_name,
            "text": comment.body,
            "permalink": f"https://reddit.com{comment.permalink}",
            "author": str(comment.author) if comment.author else "[deleted]",
            "score": comment.score,
            "created_utc": comment.created_utc,
            "created_date": created_date.strftime("%Y-%m-%d %H:%M:%S"),
            "parent_id": comment.parent_id,
            "is_submitter": comment.is_submitter if hasattr(comment, "is_submitter") else False,
            "relevance": "inherited"  # Comments inherit relevance from parent post
        }
        
        return processed_comment
    
    def fetch_historical_news(
        self,
        start_year: int = 1987,
        end_year: int = 2010,
        keywords: List[str] = None,
        sources: List[str] = None,
        articles_per_year: int = 1000,
        min_relevance: float = 0.3,
        skip_completed: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical news articles about oil markets.
        
        This method uses news APIs and archive sources to collect 
        historical news articles, especially for pre-Reddit era (1987-2008).
        
        Args:
            start_year: Start year for data collection
            end_year: End year for data collection
            keywords: List of keywords to search for
            sources: List of news sources to use
            articles_per_year: Target number of articles per year
            min_relevance: Minimum relevance score for articles
            skip_completed: Whether to skip completed sources
            
        Returns:
            DataFrame with collected news articles
        """
        # Use default keywords if not provided
        if keywords is None:
            keywords = self.OIL_KEYWORDS
        
        # Use all available sources if not specified
        if sources is None:
            sources = [source["name"] for source in self.HISTORICAL_NEWS_SOURCES]
        
        # Filter out completed sources if requested
        if skip_completed and "completed_news_sources" in self.state:
            remaining_sources = [s for s in sources if s not in self.state["completed_news_sources"]]
            if len(remaining_sources) < len(sources):
                logger.info(f"Skipping {len(sources) - len(remaining_sources)} completed news sources")
                sources = remaining_sources
        
        logger.info(f"Fetching historical news from {start_year} to {end_year}")
        logger.info(f"Target sources: {', '.join(sources)}")
        
        all_articles = []
        
        # Process each source
        for source_name in sources:
            # Find source info
            source_info = next((s for s in self.HISTORICAL_NEWS_SOURCES if s["name"] == source_name), None)
            if not source_info:
                logger.warning(f"Source {source_name} not found in HISTORICAL_NEWS_SOURCES")
                continue
            
            logger.info(f"Processing source: {source_name}")
            
            # Determine year range for this source
            source_start = max(start_year, source_info.get("start_year", start_year))
            source_end = min(end_year, source_info.get("end_year", end_year))
            
            # Collect articles by year for better control
            for year in range(source_start, source_end + 1):
                logger.info(f"Collecting {source_name} articles for {year}")
                
                # Determine date range for this year
                year_start = datetime.datetime(year, 1, 1)
                year_end = datetime.datetime(year, 12, 31)
                
                # Collect articles for this year and source
                try:
                    articles = self._collect_news_articles(
                        source=source_name, 
                        keywords=keywords,
                        start_date=year_start,
                        end_date=year_end,
                        max_articles=articles_per_year,
                        min_relevance=min_relevance
                    )
                    
                    if articles:
                        logger.info(f"Collected {len(articles)} articles from {source_name} for {year}")
                        
                        # Save articles for this year and source
                        articles_df = pd.DataFrame(articles)
                        output_path = os.path.join(self.output_dir, f"news_{source_name}_{year}.csv")
                        articles_df.to_csv(output_path, index=False)
                        
                        # Add to all articles
                        all_articles.extend(articles)
                    else:
                        logger.warning(f"No articles found from {source_name} for {year}")
                    
                except Exception as e:
                    logger.error(f"Error collecting {source_name} articles for {year}: {str(e)}")
            
            # Update state to mark this source as completed
            if "completed_news_sources" not in self.state:
                self.state["completed_news_sources"] = []
            self.state["completed_news_sources"].append(source_name)
            self._save_scraping_state()
            
            # Update stats
            if source_name not in self.stats["sources"]:
                self.stats["sources"][source_name] = 0
            self.stats["sources"][source_name] += sum(1 for a in all_articles if a.get("source") == source_name)
        
        # Create combined DataFrame
        if all_articles:
            combined_df = pd.DataFrame(all_articles)
            
            # Save combined data
            output_path = os.path.join(self.output_dir, "historical_news_combined.csv")
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(combined_df)} total articles to {output_path}")
            
            # Update stats
            self.stats["items_collected"]["news_articles"] += len(combined_df)
            self.stats["items_collected"]["total"] += len(combined_df)
            self._save_stats()
            
            return combined_df
        
        logger.warning("No historical news articles collected")
        return pd.DataFrame()
    
    def _collect_news_articles(
        self,
        source: str,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000,
        min_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Collect news articles from a specific source within a date range.
        
        Args:
            source: Name of the news source
            keywords: List of keywords to search for
            start_date: Start date for collection
            end_date: End date for collection
            max_articles: Maximum articles to collect
            min_relevance: Minimum relevance score for articles
            
        Returns:
            List of article dictionaries
        """
        # Get appropriate handler for this source
        source_handler = getattr(self, f"_collect_{source.replace('-', '_')}_articles", None)
        
        if not source_handler:
            logger.warning(f"No handler available for source: {source}")
            return []
        
        try:
            # Collect articles using source-specific handler
            articles = source_handler(
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                max_articles=max_articles
            )
            
            # Filter by relevance
            if min_relevance > 0:
                filtered_articles = []
                for article in articles:
                    # Calculate relevance if not already present
                    if "relevance_score" not in article:
                        relevance = self._calculate_news_relevance(
                            title=article.get("title", ""),
                            text=article.get("text", ""),
                            keywords=keywords
                        )
                        article["relevance_score"] = relevance
                    
                    # Filter by relevance
                    if article["relevance_score"] >= min_relevance:
                        filtered_articles.append(article)
                
                logger.info(f"Filtered {len(articles)} articles to {len(filtered_articles)} relevant ones")
                articles = filtered_articles
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting articles from {source}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _calculate_news_relevance(
        self,
        title: str,
        text: str,
        keywords: List[str]
    ) -> float:
        """
        Calculate the relevance of a news article to oil markets.
        
        Args:
            title: Article title
            text: Article text
            keywords: Oil-related keywords
            
        Returns:
            Relevance score (0-1)
        """
        # This method is similar to _calculate_oil_relevance but with
        # adjustments for news article format
        
        # Convert to lowercase for case-insensitive matching
        title_lower = title.lower() if title else ""
        text_lower = text.lower() if text else ""
        
        # Count keyword occurrences
        title_matches = sum(keyword.lower() in title_lower for keyword in keywords) if title else 0
        
        # For text, count unique keywords to avoid bias from repetition
        unique_text_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower) if text else 0
        
        # Calculate relevance score
        # Title matches are weighted more heavily
        title_weight = 4.0
        text_weight = 1.5
        
        # Normalize by text length
        title_words = len(title_lower.split()) if title else 0
        text_words = min(500, len(text_lower.split())) if text else 0  # Cap to avoid overweighting long articles
        
        # Calculate density scores
        title_density = title_matches / max(1, title_words) * 100 if title else 0
        text_density = unique_text_matches / max(1, min(10, text_words / 50)) * 10 if text else 0  # Adjusted scale
        
        # Calculate overall relevance
        relevance = (title_density * title_weight + text_density * text_weight) / (title_weight + text_weight)
        
        # Normalize to 0-1 range
        relevance = min(1.0, relevance / 15.0)  # Normalize, capping at 1.0
        
        return relevance
    
    def _collect_nytimes_articles(
        self,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Collect articles from New York Times Archive API.
        
        Args:
            keywords: List of keywords to search for
            start_date: Start date for collection
            end_date: End date for collection
            max_articles: Maximum articles to collect
            
        Returns:
            List of article dictionaries
        """
        # Check for API key
        api_key = self.config.get("api_keys", {}).get("nytimes_api_key")
        if not api_key:
            logger.warning("NYTimes API key not found")
            return self._collect_simulated_articles(
                keywords, start_date, end_date, max_articles, source="nytimes"
            )
        
        logger.info(f"Collecting NYTimes articles from {start_date.date()} to {end_date.date()}")
        
        articles = []
        
        # Format dates for API (YYYYMMDD)
        begin_date = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        # Split keywords into smaller queries
        keyword_groups = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
        
        # Process each keyword group
        for keyword_group in keyword_groups:
            # Create query string
            query = " OR ".join(f'"{k}"' for k in keyword_group)
            
            # Set initial page
            page = 0
            
            # Collect articles until we hit the limit
            while len(articles) < max_articles:
                try:
                    # Make API request
                    api_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
                    params = {
                        "q": query,
                        "begin_date": begin_date,
                        "end_date": end_date_str,
                        "page": page,
                        "sort": "relevance",
                        "api-key": api_key
                    }
                    
                    response = self._make_api_request(
                        url=api_url,
                        params=params,
                        api_name="nytimes",
                        use_cache=True
                    )
                    
                    # Check for valid response
                    if not response or "response" not in response or "docs" not in response["response"]:
                        logger.warning(f"Invalid response from NYTimes API (page {page})")
                        break
                    
                    # Process articles
                    batch_articles = response["response"]["docs"]
                    if not batch_articles:
                        logger.info(f"No more articles found from NYTimes for query: {query}")
                        break
                    
                    # Process each article
                    for article in batch_articles:
                        processed = self._process_nytimes_article(article)
                        articles.append(processed)
                    
                    logger.info(f"Collected {len(batch_articles)} NYTimes articles (page {page})")
                    
                    # Check if we've hit our limit
                    if len(articles) >= max_articles:
                        break
                    
                    # NYTimes has a rate limit of 5 calls per minute
                    time.sleep(12)
                    
                    # Increment page
                    page += 1
                    
                except Exception as e:
                    logger.error(f"Error collecting NYTimes articles: {str(e)}")
                    break
        
        # Log results
        logger.info(f"Collected {len(articles)} total NYTimes articles")
        
        return articles
    
    def _process_nytimes_article(
        self,
        article: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an NYTimes API article into standardized format.
        
        Args:
            article: Raw article from NYTimes API
            
        Returns:
            Standardized article dictionary
        """
        # Extract headline
        headline = article.get("headline", {}).get("main", "")
        
        # Extract publication date
        pub_date = article.get("pub_date", "")
        date_obj = None
        if pub_date:
            try:
                date_obj = parse_date(pub_date)
            except Exception:
                pass
        
        # Create standardized article
        processed = {
            "id": article.get("_id", ""),
            "type": "article",
            "source": "nytimes",
            "source_type": "news",
            "title": headline,
            "text": article.get("abstract", "") or article.get("lead_paragraph", ""),
            "url": article.get("web_url", ""),
            "created_date": date_obj.strftime("%Y-%m-%d %H:%M:%S") if date_obj else "",
            "created_utc": int(date_obj.timestamp()) if date_obj else 0,
            "author": article.get("byline", {}).get("original", ""),
            "section": article.get("section_name", "")
        }
        
        # Add keywords
        keywords = []
        for keyword in article.get("keywords", []):
            if keyword.get("name") == "subject":
                keywords.append(keyword.get("value", ""))
        processed["keywords"] = ", ".join(keywords)
        
        return processed
    
    def _collect_reuters_articles(
        self,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Collect articles from Reuters API.
        
        Note: This is a placeholder. Reuters doesn't have a public API,
        so we use simulated data for this source.
        
        Args:
            keywords: List of keywords to search for
            start_date: Start date for collection
            end_date: End date for collection
            max_articles: Maximum articles to collect
            
        Returns:
            List of article dictionaries
        """
        # Since Reuters doesn't have a free accessible API, we'll simulate it
        return self._collect_simulated_articles(
            keywords, start_date, end_date, max_articles, source="reuters"
        )
    
    def _collect_ft_articles(
        self,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Collect articles from Financial Times API.
        
        Note: This is a placeholder. FT API integration would require subscription.
        
        Args:
            keywords: List of keywords to search for
            start_date: Start date for collection
            end_date: End date for collection
            max_articles: Maximum articles to collect
            
        Returns:
            List of article dictionaries
        """
        # FT API is not freely available, so simulate
        return self._collect_simulated_articles(
            keywords, start_date, end_date, max_articles, source="ft"
        )
    
    def _collect_wsj_articles(
        self,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Collect articles from Wall Street Journal API.
        
        Note: This is a placeholder. WSJ API integration would require subscription.
        
        Args:
            keywords: List of keywords to search for
            start_date: Start date for collection
            end_date: End date for collection
            max_articles: Maximum articles to collect
            
        Returns:
            List of article dictionaries
        """
        # WSJ API is not freely available, so simulate
        return self._collect_simulated_articles(
            keywords, start_date, end_date, max_articles, source="wsj"
        )
    
    def _collect_archive_org_articles(
        self,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Collect articles from Internet Archive.
        
        Note: This method would ideally search the Wayback Machine,
        but for now we use simulated data.
        
        Args:
            keywords: List of keywords to search for
            start_date: Start date for collection
            end_date: End date for collection
            max_articles: Maximum articles to collect
            
        Returns:
            List of article dictionaries
        """
        # Internet Archive integration would be complex, so simulate
        return self._collect_simulated_articles(
            keywords, start_date, end_date, max_articles, source="archive-org"
        )
    
    def _collect_simulated_articles(
        self,
        keywords: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        max_articles: int = 1000,
        source: str = "simulated"
    ) -> List[Dict[str, Any]]:
        """
        Generate simulated historical news articles for testing.
        
        This method creates realistic-looking but fake articles about
        oil markets for historical periods when real data is unavailable.
        
        Args:
            keywords: List of keywords to simulate articles about
            start_date: Start date for simulated articles
            end_date: End date for simulated articles
            max_articles: Maximum articles to simulate
            source: Source name for the articles
            
        Returns:
            List of simulated article dictionaries
        """
        logger.info(f"Simulating {max_articles} articles from {source} ({start_date.date()} - {end_date.date()})")
        
        # Sample oil news headlines by time period
        headlines_by_period = {
            # 1980s
            "1980s": [
                "Oil Prices Plummet as OPEC Agreement Fails",
                "Global Oil Glut Continues to Pressure Prices",
                "Middle East Tensions Raise Concerns About Oil Supply",
                "Oil Prices Hit Five-Year Low on Supply Concerns",
                "Energy Stocks Tumble as Crude Falls Below $15",
                "OPEC Meets to Discuss Production Quotas",
                "U.S. Oil Production Faces Challenges Amid Price Drop",
                "Saudi Arabia Increases Output, Putting Pressure on Prices",
                "Analysts Predict Oil Price Recovery by Decade's End",
                "Oil Companies Cut Exploration Budgets Amid Price Slump"
            ],
            # 1990s
            "1990s": [
                "Gulf War Sends Oil Prices Soaring",
                "OPEC Agrees to Production Ceiling, Market Stabilizes",
                "Asian Financial Crisis Impacts Oil Demand",
                "Brent Crude Falls to $12 on Oversupply Concerns",
                "Major Oil Merger Creates Industry Giant",
                "Oil Futures Rally on Cold Weather Forecast",
                "Venezuelan Oil Strike Threatens Global Supply",
                "Energy Companies Invest in New Exploration Technologies",
                "Russian Oil Production Declines Amid Economic Troubles",
                "Analysts Debate Peak Oil Theory as Millennium Approaches"
            ],
            # 2000s
            "2000s": [
                "Oil Breaks $30 Barrier on Middle East Concerns",
                "Crude Hits Record High Above $50 per Barrel",
                "OPEC Cut Fails to Stop Oil Price Slide",
                "Demand from China Drives Crude to New Heights",
                "Oil Surpasses $100 in Historic Trading Session",
                "Energy Sector Leads Market Rally on Strong Earnings",
                "Hurricane Disrupts Gulf of Mexico Oil Production",
                "Peak Oil Concerns Grow as Prices Reach $140",
                "Oil Prices Collapse in Global Financial Crisis",
                "OPEC Announces Major Production Cut to Support Market"
            ]
        }
        
        # Determine which period headlines to use
        period = "1980s"
        year = start_date.year
        if year >= 1990 and year < 2000:
            period = "1990s"
        elif year >= 2000:
            period = "2000s"
        
        headlines = headlines_by_period[period]
        
        # Sample paragraphs about oil markets by period
        paragraphs_by_period = {
            # 1980s oil market paragraphs
            "1980s": [
                "The oil market continued to struggle with oversupply issues as OPEC members exceeded their production quotas. Analysts at {bank} estimated that the global oil surplus reached {surplus} million barrels per day, putting significant downward pressure on prices.",
                
                "Saudi Arabia, the world's largest oil exporter, pumped {production} million barrels per day in {month}, according to industry sources. The kingdom has maintained high production levels despite falling prices, in what many analysts view as an attempt to protect market share rather than prices.",
                
                "U.S. oil inventories rose by {inventory_change} million barrels last week, significantly higher than the {expected} million barrel increase analysts had expected. The build marks the {nth} consecutive weekly increase, raising concerns about storage capacity at the Cushing, Oklahoma hub.",
                
                "Brent crude fell to ${price} per barrel, its lowest level since {year_low}. The benchmark has fallen nearly {percent}% since the beginning of the year as the market grapples with persistent oversupply and weakening global demand.",
                
                "Energy company stocks have been hit hard by the prolonged slump in oil prices. {company}, one of the industry's largest players, announced it would cut capital expenditure by {capex}% and lay off {layoffs} workers to weather the downturn.",
                
                "OPEC ministers will meet in {city} next month to discuss the possibility of implementing production cuts to stabilize the market. However, analysts at {bank} remain skeptical that member countries will reach a meaningful agreement, citing historical compliance issues."
            ],
            
            # 1990s oil market paragraphs
            "1990s": [
                "The Gulf War has created significant volatility in the oil market, with prices surging to ${price} before retreating as fears of supply disruptions eased. Military action in Kuwait raised concerns about damage to critical oil infrastructure in the region.",
                
                "Economic troubles in Asia have dampened oil demand growth, with consumption in the region expected to grow by only {growth}% this year, down from {previous}% last year. Japan, the region's largest oil consumer, has seen industrial production fall for {months} consecutive months.",
                
                "Major oil companies continue to merge in response to challenging market conditions. The proposed merger between {company1} and {company2}, valued at ${value} billion, would create one of the world's largest energy companies and could trigger further consolidation in the industry.",
                
                "OPEC agreed to maintain its production ceiling of {quota} million barrels per day during its meeting in {city} yesterday. However, the group acknowledged that some members continue to exceed their quotas, contributing to market oversupply.",
                
                "Russia's oil production fell to {production} million barrels per day last month, its lowest level since {year_low}. The decline comes amid economic turmoil and underinvestment in aging Soviet-era oil fields.",
                
                "Technological advances in seismic imaging and horizontal drilling are opening up new exploration opportunities for oil companies. {company} announced a significant discovery in deepwater {region}, with estimated reserves of {reserves} billion barrels."
            ],
            
            # 2000s oil market paragraphs
            "2000s": [
                "Oil prices hit a record high of ${price} per barrel today, driven by strong demand from China and ongoing supply concerns. Chinese oil imports rose by {percent}% year-over-year, reflecting the country's rapid industrialization and growing middle class.",
                
                "Hurricane {name} forced the shutdown of approximately {percent}% of Gulf of Mexico oil production, leading to a spike in crude prices. Analysts at {bank} estimate that it could take several weeks for production to fully resume, potentially drawing down U.S. inventories.",
                
                "OPEC agreed to cut production by {cut} million barrels per day in an effort to stabilize falling prices. The group's decision comes as oil has fallen more than {drop}% from its July high of ${peak} per barrel amid concerns about slowing global economic growth.",
                
                "Investment in alternative energy sources continues to grow, with {company} announcing a ${investment} billion commitment to solar and wind projects over the next decade. Some analysts view this as a long-term threat to oil demand growth.",
                
                "The U.S. Strategic Petroleum Reserve now holds {spr} million barrels, near its full capacity. Energy Secretary {secretary} indicated the government might consider releasing some reserves if prices continue to rise.",
                
                "Trading activity in oil futures has reached record levels, with daily volume on the NYMEX exceeding {volume} million contracts. Speculative positions held by non-commercial traders have grown by {speculative}% over the past month, adding to market volatility.",
                
                "Global oil consumption is expected to reach {consumption} million barrels per day this year, according to the International Energy Agency, a {growth_rate}% increase from last year. Emerging markets account for more than {em_percent}% of this growth."
            ]
        }
        
        # Select paragraphs for the period
        paragraphs = paragraphs_by_period.get(period, paragraphs_by_period["2000s"])
        
        # Variable replacement values by period
        variables = {
            # 1980s
            "1980s": {
                "bank": ["Goldman Sachs", "Morgan Stanley", "Chase Manhattan", "Merrill Lynch", "Citibank"],
                "surplus": ["1.5", "2.3", "3.1", "4.2", "5.0"],
                "production": ["8.5", "9.2", "10.1", "7.8", "6.5"],
                "month": ["January", "March", "June", "September", "December"],
                "inventory_change": ["3.2", "4.5", "5.7", "6.8", "8.2"],
                "expected": ["1.5", "2.0", "2.5", "3.0", "3.5"],
                "nth": ["fourth", "fifth", "sixth", "seventh", "eighth"],
                "price": ["14.50", "12.80", "10.20", "15.40", "18.60"],
                "year_low": ["1983", "1984", "1985", "1986", "1987"],
                "percent": ["25", "30", "35", "40", "45"],
                "company": ["Exxon", "Mobil", "Texaco", "Shell", "British Petroleum"],
                "capex": ["15", "20", "25", "30", "35"],
                "layoffs": ["5,000", "7,500", "10,000", "12,500", "15,000"],
                "city": ["Vienna", "Geneva", "Riyadh", "Dubai", "Kuwait City"],
                "quota": ["15.5", "16.0", "16.5", "17.0", "17.5"]
            },
            # 1990s
            "1990s": {
                "price": ["25.80", "30.20", "35.40", "21.60", "18.30"],
                "growth": ["1.2", "1.5", "1.8", "2.1", "2.4"],
                "previous": ["3.5", "4.0", "4.5", "5.0", "5.5"],
                "months": ["three", "four", "five", "six", "seven"],
                "company1": ["Exxon", "Mobil", "BP", "Amoco", "Texaco"],
                "company2": ["Mobil", "BP", "Amoco", "Arco", "Chevron"],
                "value": ["75", "80", "85", "90", "95"],
                "quota": ["24.5", "25.0", "25.5", "26.0", "26.5"],
                "city": ["Vienna", "Geneva", "Riyadh", "Jakarta", "Caracas"],
                "production": ["6.2", "6.5", "6.8", "7.1", "7.4"],
                "year_low": ["1992", "1993", "1994", "1995", "1996"],
                "company": ["Shell", "BP", "Exxon", "Texaco", "Chevron"],
                "region": ["Gulf of Mexico", "North Sea", "West Africa", "Caspian Sea", "Brazil"],
                "reserves": ["1.2", "1.5", "1.8", "2.1", "2.4"]
            },
            # 2000s
            "2000s": {
                "price": ["75.80", "85.40", "95.20", "105.60", "145.30"],
                "percent": ["15", "20", "25", "30", "35"],
                "name": ["Katrina", "Rita", "Gustav", "Ike", "Ivan"],
                "cut": ["1.5", "2.0", "2.5", "3.0", "3.5"],
                "drop": ["20", "25", "30", "35", "40"],
                "peak": ["145", "147", "148", "149", "150"],
                "company": ["BP", "Shell", "ExxonMobil", "Chevron", "Total"],
                "investment": ["5", "10", "15", "20", "25"],
                "spr": ["695", "700", "705", "710", "715"],
                "secretary": ["Samuel Bodman", "Steven Chu", "Ernest Moniz", "James Schlesinger", "Charles Duncan"],
                "volume": ["1.2", "1.5", "1.8", "2.1", "2.4"],
                "speculative": ["25", "30", "35", "40", "45"],
                "consumption": ["85.5", "86.2", "86.8", "87.5", "88.2"],
                "growth_rate": ["1.2", "1.5", "1.8", "2.1", "2.4"],
                "em_percent": ["65", "70", "75", "80", "85"],
                "bank": ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Citigroup", "Bank of America"]
            }
        }
        
        period_vars = variables.get(period, variables["2000s"])
        
        # Generate simulated articles
        articles = []
        
        # Calculate date range in days
        date_range = (end_date - start_date).days
        if date_range <= 0:
            date_range = 1
        
        # Ensure max_articles is reasonable
        actual_max = min(max_articles, date_range * 3)  # Max 3 articles per day
        
        for i in range(actual_max):
            # Generate random date within range
            random_days = int(random.random() * date_range)
            article_date = start_date + datetime.timedelta(days=random_days)
            
            # Select random headline
            headline = random.choice(headlines)
            
            # Generate content
            paragraph_count = random.randint(2, 5)
            article_paragraphs = random.sample(paragraphs, min(paragraph_count, len(paragraphs)))
            
            # Replace variables in paragraphs
            filled_paragraphs = []
            for paragraph in article_paragraphs:
                filled = paragraph
                # Find all placeholders in the format {variable}
                placeholders = re.findall(r'\{(\w+)\}', paragraph)
                
                # Replace each placeholder
                for placeholder in placeholders:
                    if placeholder in period_vars:
                        replacement = random.choice(period_vars[placeholder])
                        filled = filled.replace(f"{{{placeholder}}}", replacement)
                
                filled_paragraphs.append(filled)
            
            # Join paragraphs
            content = "\n\n".join(filled_paragraphs)
            
            # Calculate relevance score
            relevance = self._calculate_news_relevance(headline, content, keywords)
            
            # Create article
            article = {
                "id": f"{source}-{i}",
                "type": "article",
                "source": source,
                "source_type": "news",
                "title": headline,
                "text": content,
                "url": f"https://example.com/{source}/article/{i}",
                "created_date": article_date.strftime("%Y-%m-%d %H:%M:%S"),
                "created_utc": int(article_date.timestamp()),
                "author": f"Simulated Author {random.randint(1, 20)}",
                "section": random.choice(["Markets", "Energy", "Business", "Finance", "Economy"]),
                "relevance_score": relevance,
                "relevance": "high" if relevance >= 0.7 else ("medium" if relevance >= 0.4 else "low")
            }
            
            articles.append(article)
        
        # Sort by date
        articles.sort(key=lambda x: x["created_utc"])
        
        logger.info(f"Generated {len(articles)} simulated articles for {source}")
        return articles
    
    def fetch_financial_sentiment_data(
        self,
        start_year: int = 1987,
        end_year: int = None,
        datasets: List[str] = None,
        min_relevance: float = 0.3,
        samples_per_dataset: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch financial sentiment data from academic datasets.
        
        This method supplements historical data with sentiment samples
        from financial sentiment analysis datasets.
        
        Args:
            start_year: Start year for data collection
            end_year: End year for data collection (defaults to current year)
            datasets: List of dataset names to use
            min_relevance: Minimum relevance score for samples
            samples_per_dataset: Maximum samples per dataset
            
        Returns:
            DataFrame with financial sentiment data
        """
        if end_year is None:
            end_year = datetime.datetime.now().year
        
        # Use all datasets if not specified
        if datasets is None:
            datasets = [dataset["name"] for dataset in self.FINANCIAL_DATASETS]
        
        logger.info(f"Fetching financial sentiment data for {start_year}-{end_year}")
        logger.info(f"Using datasets: {', '.join(datasets)}")
        
        all_samples = []
        
        # Process each dataset
        for dataset_name in datasets:
            # Find dataset info
            dataset_info = next((d for d in self.FINANCIAL_DATASETS if d["name"] == dataset_name), None)
            if not dataset_info:
                logger.warning(f"Dataset {dataset_name} not found in FINANCIAL_DATASETS")
                continue
            
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Get dataset handler
            dataset_handler = getattr(self, f"_process_{dataset_name.replace('-', '_')}_dataset", None)
            if not dataset_handler:
                logger.warning(f"No handler available for dataset: {dataset_name}")
                continue
            
            # Process dataset
            try:
                samples = dataset_handler(
                    start_year=start_year,
                    end_year=end_year,
                    max_samples=samples_per_dataset,
                    min_relevance=min_relevance
                )
                
                if samples:
                    logger.info(f"Collected {len(samples)} samples from {dataset_name}")
                    
                    # Save samples for this dataset
                    samples_df = pd.DataFrame(samples)
                    output_path = os.path.join(self.output_dir, f"financial_sentiment_{dataset_name}.csv")
                    samples_df.to_csv(output_path, index=False)
                    
                    # Add to all samples
                    all_samples.extend(samples)
                else:
                    logger.warning(f"No samples found from {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Create combined DataFrame
        if all_samples:
            combined_df = pd.DataFrame(all_samples)
            
            # Save combined data
            output_path = os.path.join(self.output_dir, "financial_sentiment_combined.csv")
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(combined_df)} total financial sentiment samples to {output_path}")
            
            # Update stats
            self.stats["items_collected"]["financial_data"] += len(combined_df)
            self.stats["items_collected"]["total"] += len(combined_df)
            self._save_stats()
            
            return combined_df
        
        logger.warning("No financial sentiment data collected")
        return pd.DataFrame()
    
    def _process_fin_phrasebank_dataset(
        self,
        start_year: int,
        end_year: int,
        max_samples: int = 1000,
        min_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Process Financial PhraseBank dataset.
        
        This dataset consists of financial news sentences with sentiment labels.
        For this implementation, we simulate the dataset content.
        
        Args:
            start_year: Start year for samples
            end_year: End year for samples
            max_samples: Maximum samples to generate
            min_relevance: Minimum relevance score for samples
            
        Returns:
            List of sentiment sample dictionaries
        """
        logger.info(f"Processing Financial PhraseBank dataset")
        
        # Simulated Financial PhraseBank samples
        template_sentences = [
            # Positive
            {"text": "The oil company reported better than expected quarterly earnings, with profits rising {percent}%.", "sentiment": "positive"},
            {"text": "Crude oil prices climbed to ${price} per barrel, the highest level in {period}.", "sentiment": "positive"},
            {"text": "OPEC's decision to cut production led to a significant rally in oil futures.", "sentiment": "positive"},
            {"text": "Energy stocks surged after the company announced a major oil discovery in {region}.", "sentiment": "positive"},
            {"text": "The oil producer increased its dividend by {percent}%, signaling confidence in future cash flows.", "sentiment": "positive"},
            {"text": "Analysts upgraded the petroleum sector, citing improving supply-demand dynamics.", "sentiment": "positive"},
            
            # Negative
            {"text": "Oil prices plummeted {percent}% to ${price} as OPEC failed to reach an agreement on production cuts.", "sentiment": "negative"},
            {"text": "The energy company reported a quarterly loss of ${loss} billion due to low oil prices.", "sentiment": "negative"},
            {"text": "Crude oil inventories increased by {inventory} million barrels, far exceeding analyst expectations.", "sentiment": "negative"},
            {"text": "The oil producer slashed its capital expenditure budget by {percent}% in response to market conditions.", "sentiment": "negative"},
            {"text": "The petroleum company announced it would cut {jobs} jobs due to the prolonged downturn.", "sentiment": "negative"},
            {"text": "Energy stocks fell sharply as oil prices declined for the {nth} consecutive trading session.", "sentiment": "negative"},
            
            # Neutral
            {"text": "Crude oil traded at ${price} per barrel, within the range seen over the past month.", "sentiment": "neutral"},
            {"text": "OPEC members are scheduled to meet next week to discuss production quotas.", "sentiment": "neutral"},
            {"text": "The oil company appointed {name} as its new Chief Executive Officer.", "sentiment": "neutral"},
            {"text": "The energy sector represented {percent}% of the S&P 500 index as of last week.", "sentiment": "neutral"},
            {"text": "Analysts expect crude oil demand to grow by {percent}% next year, in line with historical averages.", "sentiment": "neutral"},
            {"text": "The oil producer maintained its production guidance of {production} million barrels per day for the year.", "sentiment": "neutral"}
        ]
        
        # Variables for replacing placeholders
        variables = {
            "percent": ["5", "10", "15", "20", "25", "30", "35", "40"],
            "price": ["45.20", "52.80", "68.50", "75.30", "82.60", "95.40", "105.20", "120.75"],
            "period": ["3 months", "6 months", "9 months", "1 year", "18 months", "2 years", "3 years", "5 years"],
            "region": ["Gulf of Mexico", "North Sea", "Brazil offshore", "West Africa", "Caspian Sea", "Arctic", "Permian Basin", "Eagle Ford"],
            "loss": ["1.2", "1.8", "2.3", "3.1", "4.2", "5.5", "6.8", "8.2"],
            "inventory": ["3.5", "4.8", "6.2", "7.5", "8.9", "10.3", "12.1", "15.5"],
            "jobs": ["1,000", "2,500", "5,000", "7,500", "10,000", "15,000", "20,000", "25,000"],
            "nth": ["third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"],
            "name": ["John Smith", "Mary Johnson", "Robert Williams", "Michael Brown", "James Jones", "Patricia Davis", "Jennifer Garcia", "David Rodriguez"],
            "production": ["1.2", "1.5", "1.8", "2.2", "2.5", "2.8", "3.2", "3.5"]
        }
        
        # Generate random samples
        samples = []
        
        # Assign years to samples - Financial PhraseBank is from around 2013-2014
        sample_years = list(range(max(start_year, 2013), min(end_year, 2014) + 1))
        if not sample_years:
            sample_years = [2013, 2014]
        
        for i in range(max_samples):
            # Select template
            template = random.choice(template_sentences)
            
            # Replace variables
            text = template["text"]
            placeholders = re.findall(r'\{(\w+)\}', text)
            
            for placeholder in placeholders:
                if placeholder in variables:
                    replacement = random.choice(variables[placeholder])
                    text = text.replace(f"{{{placeholder}}}", replacement)
            
            # Calculate relevance
            relevance = self._calculate_news_relevance(text, "", self.OIL_KEYWORDS)
            
            # Skip if not relevant enough
            if relevance < min_relevance:
                continue
            
            # Assign random date within year range
            year = random.choice(sample_years)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.datetime(year, month, day)
            
            # Create sample
            sample = {
                "id": f"fin-phrasebank-{i}",
                "type": "financial-sentiment",
                "source": "fin-phrasebank",
                "source_type": "financial-dataset",
                "text": text,
                "sentiment": template["sentiment"],
                "created_date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "created_utc": int(date.timestamp()),
                "relevance_score": relevance,
                "relevance": "high" if relevance >= 0.7 else ("medium" if relevance >= 0.4 else "low")
            }
            
            samples.append(sample)
            
            # Check if we've reached the limit
            if len(samples) >= max_samples:
                break
        
        logger.info(f"Generated {len(samples)} Financial PhraseBank samples")
        return samples
    
    def _process_semeval_2017_dataset(
        self,
        start_year: int,
        end_year: int,
        max_samples: int = 1000,
        min_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Process SemEval-2017 financial sentiment dataset.
        
        This dataset consists of financial microblog and news headline sentiment.
        For this implementation, we simulate the dataset content.
        
        Args:
            start_year: Start year for samples
            end_year: End year for samples
            max_samples: Maximum samples to generate
            min_relevance: Minimum relevance score for samples
            
        Returns:
            List of sentiment sample dictionaries
        """
        logger.info(f"Processing SemEval-2017 dataset")
        
        # SemEval-2017 is from around 2015-2017
        valid_start = max(start_year, 2015)
        valid_end = min(end_year, 2017)
        
        if valid_end < valid_start:
            logger.warning(f"No valid year range for SemEval-2017 dataset")
            return []
        
        # Simulated SemEval-2017 samples
        template_headlines = [
            # Positive sentiment (score > 0)
            {"text": "Oil rallies on surprise draw in U.S. inventories", "score": 0.65},
            {"text": "Crude prices climb as OPEC agrees to production cut", "score": 0.78},
            {"text": "Energy stocks surge as oil tops ${price} per barrel", "score": 0.71},
            {"text": "WTI futures gain {percent}% on bullish demand outlook", "score": 0.62},
            {"text": "Oil market rebalancing faster than expected", "score": 0.58},
            {"text": "Brent crude breaks resistance level, technicals improve", "score": 0.53},
            
            # Negative sentiment (score < 0)
            {"text": "Oil plunges {percent}% as OPEC talks collapse", "score": -0.72},
            {"text": "Crude prices fall on rising U.S. production concerns", "score": -0.65},
            {"text": "Energy stocks tumble as oversupply fears persist", "score": -0.58},
            {"text": "WTI drops below key support at ${price}", "score": -0.51},
            {"text": "Oil slide continues amid record high inventories", "score": -0.69},
            {"text": "Brent crude outlook negative as demand weakens", "score": -0.63},
            
            # Neutral sentiment (-0.2 < score < 0.2)
            {"text": "Oil trades sideways ahead of inventory data", "score": 0.05},
            {"text": "Crude prices stable as market awaits OPEC decision", "score": -0.08},
            {"text": "Energy stocks mixed as oil fluctuates", "score": 0.12},
            {"text": "WTI hovers near ${price} in range-bound trading", "score": -0.15},
            {"text": "Oil market balanced between supply and demand factors", "score": 0.0},
            {"text": "Brent crude unchanged as traders assess market signals", "score": 0.07}
        ]
        
        # Variables for replacing placeholders
        variables = {
            "percent": ["2.5", "3.8", "4.2", "5.6", "6.3", "7.1", "8.4", "9.2"],
            "price": ["45.50", "48.75", "52.30", "55.80", "62.40", "68.20", "72.85", "78.60"],
        }
        
        # Generate random samples
        samples = []
        
        # Create year range
        sample_years = list(range(valid_start, valid_end + 1))
        
        for i in range(max_samples):
            # Select template
            template = random.choice(template_headlines)
            
            # Replace variables
            text = template["text"]
            placeholders = re.findall(r'\{(\w+)\}', text)
            
            for placeholder in placeholders:
                if placeholder in variables:
                    replacement = random.choice(variables[placeholder])
                    text = text.replace(f"{{{placeholder}}}", replacement)
            
            # Calculate relevance
            relevance = self._calculate_news_relevance(text, "", self.OIL_KEYWORDS)
            
            # Skip if not relevant enough
            if relevance < min_relevance:
                continue
            
            # Assign random date within year range
            year = random.choice(sample_years)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.datetime(year, month, day)
            
            # Determine sentiment category
            if template["score"] > 0.2:
                sentiment = "positive"
            elif template["score"] < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Create sample
            sample = {
                "id": f"semeval-2017-{i}",
                "type": "financial-sentiment",
                "source": "semeval-2017",
                "source_type": "financial-dataset",
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": template["score"],
                "created_date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "created_utc": int(date.timestamp()),
                "relevance_score": relevance,
                "relevance": "high" if relevance >= 0.7 else ("medium" if relevance >= 0.4 else "low")
            }
            
            samples.append(sample)
            
            # Check if we've reached the limit
            if len(samples) >= max_samples:
                break
        
        logger.info(f"Generated {len(samples)} SemEval-2017 samples")
        return samples
    
    def _process_fiqa_dataset(
        self,
        start_year: int,
        end_year: int,
        max_samples: int = 1000,
        min_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Process FiQA dataset (Financial Question Answering).
        
        This dataset includes financial sentiment samples from around 2018.
        For this implementation, we simulate the dataset content.
        
        Args:
            start_year: Start year for samples
            end_year: End year for samples
            max_samples: Maximum samples to generate
            min_relevance: Minimum relevance score for samples
            
        Returns:
            List of sentiment sample dictionaries
        """
        logger.info(f"Processing FiQA dataset")
        
        # FiQA is from around 2018
        valid_start = max(start_year, 2018)
        valid_end = min(end_year, 2018)
        
        if valid_end < valid_start:
            logger.warning(f"No valid year range for FiQA dataset")
            return []
        
        # Simulated FiQA samples (longer and more detailed than SemEval)
        template_sentences = [
            # Positive
            {
                "text": "The outlook for oil prices is increasingly positive as OPEC+ discipline holds and demand continues to recover. I expect Brent to trade between ${lower} and ${upper} through Q3, with potential for further upside if economic activity exceeds expectations.",
                "score": 0.75
            },
            {
                "text": "Energy stocks look undervalued based on current oil price projections. The sector trades at {pe} times forward earnings compared to the broader market at {market_pe}, despite strong free cash flow and improved capital discipline.",
                "score": 0.68
            },
            {
                "text": "Oil services companies should benefit significantly from increased drilling activity, with rig counts already up {percent}% year-over-year. This segment has the most operational leverage to the ongoing recovery in upstream capital expenditure.",
                "score": 0.82
            },
            
            # Negative
            {
                "text": "I remain bearish on crude oil prices as U.S. production continues to surprise to the upside. The latest EIA data showing {inventory} million barrels build in crude inventories suggests demand remains insufficient to absorb current supply levels.",
                "score": -0.71
            },
            {
                "text": "Integrated oil majors face significant long-term headwinds from the energy transition. Their massive legacy assets risk becoming stranded as governments worldwide accelerate decarbonization targets and renewable energy costs continue to decline.",
                "score": -0.65
            },
            {
                "text": "OPEC's market share continues to erode as U.S. shale production becomes more efficient with break-even prices now at ${breakeven} per barrel. The cartel's influence over global oil markets appears to be structurally diminishing.",
                "score": -0.58
            },
            
            # Neutral
            {
                "text": "Oil price volatility should persist through {year} as the market balances conflicting signals. While OPEC+ cuts support the price floor, concerns about Chinese demand and U.S. production growth may limit upside potential.",
                "score": 0.12
            },
            {
                "text": "The energy sector is undergoing significant transformation as companies adapt to both cyclical oil price movements and secular trends toward cleaner fuels. Capital allocation strategies increasingly reflect this complex operating environment.",
                "score": -0.08
            },
            {
                "text": "Oil demand elasticity remains a key unknown in price forecasts. While recent data shows consumption at {demand} million barrels per day, the response to prices above ${threshold} has not been fully tested in the post-pandemic economy.",
                "score": 0.05
            }
        ]
        
        # Variables for replacing placeholders
        variables = {
            "lower": ["65", "70", "75", "80", "85"],
            "upper": ["85", "90", "95", "100", "105"],
            "pe": ["8.5", "9.2", "10.1", "10.8", "11.5"],
            "market_pe": ["17.5", "18.2", "19.4", "20.1", "21.3"],
            "percent": ["15", "20", "25", "30", "35"],
            "inventory": ["3.2", "4.5", "5.8", "6.7", "7.9"],
            "breakeven": ["35", "38", "40", "42", "45"],
            "year": ["2022", "2023", "2024", "2025", "2026"],
            "demand": ["99.5", "100.2", "101.5", "102.8", "103.5"],
            "threshold": ["85", "90", "95", "100", "105"]
        }
        
        # Generate random samples
        samples = []
        
        for i in range(max_samples):
            # Select template
            template = random.choice(template_sentences)
            
            # Replace variables
            text = template["text"]
            placeholders = re.findall(r'\{(\w+)\}', text)
            
            for placeholder in placeholders:
                if placeholder in variables:
                    replacement = random.choice(variables[placeholder])
                    text = text.replace(f"{{{placeholder}}}", replacement)
            
            # Calculate relevance
            relevance = self._calculate_news_relevance(text, "", self.OIL_KEYWORDS)
            
            # Skip if not relevant enough
            if relevance < min_relevance:
                continue
            
            # Assign random date within year range (FiQA is from 2018)
            year = 2018  # FiQA dataset is from 2018
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.datetime(year, month, day)
            
            # Determine sentiment category
            if template["score"] > 0.2:
                sentiment = "positive"
            elif template["score"] < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Create sample
            sample = {
                "id": f"fiqa-{i}",
                "type": "financial-sentiment",
                "source": "fiqa",
                "source_type": "financial-dataset",
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": template["score"],
                "created_date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "created_utc": int(date.timestamp()),
                "relevance_score": relevance,
                "relevance": "high" if relevance >= 0.7 else ("medium" if relevance >= 0.4 else "low")
            }
            
            samples.append(sample)
            
            # Check if we've reached the limit
            if len(samples) >= max_samples:
                break
        
        logger.info(f"Generated {len(samples)} FiQA samples")
        return samples
    
    def create_comprehensive_dataset(
        self,
        start_year: int = 1987,
        end_year: int = None,
        include_reddit: bool = True,
        include_news: bool = True,
        include_financial: bool = True,
        output_file: str = "comprehensive_sentiment_dataset.csv"
    ) -> pd.DataFrame:
        """
        Create a comprehensive dataset combining all sources.
        
        This method combines Reddit data, news articles, and financial dataset
        samples into a single unified dataset spanning the entire time period.
        
        Args:
            start_year: Start year for the dataset
            end_year: End year for the dataset (defaults to current year)
            include_reddit: Whether to include Reddit data
            include_news: Whether to include news articles
            include_financial: Whether to include financial datasets
            output_file: Name of the output file
            
        Returns:
            DataFrame with the comprehensive dataset
        """
        if end_year is None:
            end_year = datetime.datetime.now().year
        
        logger.info(f"Creating comprehensive dataset from {start_year} to {end_year}")
        
        all_data = []
        
        # 1. Get news data (especially for pre-Reddit era 1987-2008)
        if include_news:
            logger.info("Collecting historical news articles")
            
            # For historical period (pre-Reddit), prioritize simulated news
            historical_news = self.fetch_historical_news(
                start_year=start_year, 
                end_year=min(2008, end_year),
                sources=["simulated", "nytimes", "ft", "wsj"],
                articles_per_year=500
            )
            
            if not historical_news.empty:
                logger.info(f"Added {len(historical_news)} historical news articles to dataset")
                all_data.append(historical_news)
        
        # 2. Get Reddit data (post-2008 era)
        if include_reddit and start_year <= end_year and end_year >= 2008:
            logger.info("Collecting Reddit data (2008-present)")
            
            reddit_data = self.fetch_reddit_data(
                start_date=max(datetime.datetime(2008, 1, 1), datetime.datetime(start_year, 1, 1)),
                end_date=datetime.datetime(end_year, 12, 31),
                subreddits=self.OIL_SUBREDDITS[:10],  # Limit to top subreddits
                include_comments=True,
                min_relevance=0.3
            )
            
            if not reddit_data.empty:
                logger.info(f"Added {len(reddit_data)} Reddit items to dataset")
                all_data.append(reddit_data)
        
        # 3. Get financial sentiment data (supplemental, various years)
        if include_financial:
            logger.info("Collecting financial sentiment dataset samples")
            
            financial_data = self.fetch_financial_sentiment_data(
                start_year=start_year,
                end_year=end_year,
                min_relevance=0.4
            )
            
            if not financial_data.empty:
                logger.info(f"Added {len(financial_data)} financial sentiment samples to dataset")
                all_data.append(financial_data)
        
        # Combine all data sources
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Create unified date field for sorting
            if "created_utc" in combined_df.columns:
                # Convert unix timestamps to datetime for sorting
                combined_df["datetime"] = pd.to_datetime(combined_df["created_utc"], unit="s")
                combined_df = combined_df.sort_values("datetime")
                combined_df = combined_df.drop("datetime", axis=1)
            
            # Save the combined dataset
            output_path = os.path.join(self.output_dir, output_file)
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved comprehensive dataset with {len(combined_df)} items to {output_path}")
            
            return combined_df
        
        logger.warning("No data collected for comprehensive dataset")
        return pd.DataFrame()
    
    def analyze_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the coverage of a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with coverage statistics
        """
        if df.empty:
            return {"error": "Empty dataset"}
        
        # Create datetime column if not present
        if "datetime" not in df.columns:
            if "created_utc" in df.columns:
                df["datetime"] = pd.to_datetime(df["created_utc"], unit="s")
            elif "created_date" in df.columns:
                df["datetime"] = pd.to_datetime(df["created_date"])
            else:
                return {"error": "No date column found"}
        
        # Basic statistics
        stats = {
            "total_items": len(df),
            "time_range": {
                "start": df["datetime"].min().strftime("%Y-%m-%d"),
                "end": df["datetime"].max().strftime("%Y-%m-%d"),
                "span_days": (df["datetime"].max() - df["datetime"].min()).days
            },
            "by_source": {},
            "by_type": {},
            "by_year": {},
            "by_relevance": {}
        }
        
        # Count by source
        if "source" in df.columns:
            source_counts = df["source"].value_counts().to_dict()
            stats["by_source"] = source_counts
        
        # Count by type
        if "type" in df.columns:
            type_counts = df["type"].value_counts().to_dict()
            stats["by_type"] = type_counts
        
        # Count by year
        year_counts = df["datetime"].dt.year.value_counts().sort_index().to_dict()
        stats["by_year"] = {str(year): count for year, count in year_counts.items()}
        
        # Count by relevance
        if "relevance" in df.columns:
            relevance_counts = df["relevance"].value_counts().to_dict()
            stats["by_relevance"] = relevance_counts
        
        # Check for years with low coverage
        all_years = range(df["datetime"].dt.year.min(), df["datetime"].dt.year.max() + 1)
        missing_years = [year for year in all_years if year not in year_counts]
        low_coverage_years = [year for year, count in year_counts.items() if count < 100]
        
        stats["coverage_issues"] = {
            "missing_years": missing_years,
            "low_coverage_years": low_coverage_years
        }
        
        # Calculate sentiment statistics if available
        if "sentiment_score" in df.columns or "sentiment_compound" in df.columns:
            sentiment_col = "sentiment_score" if "sentiment_score" in df.columns else "sentiment_compound"
            sentiment_data = df[sentiment_col].dropna()
            
            if not sentiment_data.empty:
                stats["sentiment"] = {
                    "mean": float(sentiment_data.mean()),
                    "median": float(sentiment_data.median()),
                    "std": float(sentiment_data.std()),
                    "min": float(sentiment_data.min()),
                    "max": float(sentiment_data.max()),
                    "positive_pct": float((sentiment_data > 0.05).mean() * 100),
                    "neutral_pct": float(((sentiment_data >= -0.05) & (sentiment_data <= 0.05)).mean() * 100),
                    "negative_pct": float((sentiment_data < -0.05).mean() * 100)
                }
        
        return stats
    
    def visualize_coverage(
        self,
        df: pd.DataFrame,
        output_file: str = "data_coverage.png",
        figsize: Tuple[int, int] = (15, 12)
    ) -> plt.Figure:
        """
        Create visualizations for dataset coverage.
        
        Args:
            df: DataFrame to visualize
            output_file: Filename for the output visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if df.empty:
            logger.error("Empty dataset provided for visualization")
            return None
        
        # Create datetime column if not present
        if "datetime" not in df.columns:
            if "created_utc" in df.columns:
                df["datetime"] = pd.to_datetime(df["created_utc"], unit="s")
            elif "created_date" in df.columns:
                df["datetime"] = pd.to_datetime(df["created_date"])
            else:
                logger.error("No date column found for visualization")
                return None
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
        
        # 1. Time series plot of data counts by month
        ax1 = fig.add_subplot(gs[0, :])
        monthly_counts = df.groupby(df["datetime"].dt.to_period("M")).size()
        monthly_counts.index = monthly_counts.index.to_timestamp()
        bars = monthly_counts.plot(ax=ax1, kind="bar", color="steelblue")
        for bar in bars.patches:
            bar.set_alpha(0.7)
        ax1.set_title("Data Coverage by Month", fontsize=14)
        ax1.set_ylabel("Number of Items")
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", rotation=90)
        ax1.grid(True, alpha=0.3)
        
        # 2. Breakdown by source
        ax2 = fig.add_subplot(gs[1, 0])
        if "source" in df.columns:
            source_counts = df["source"].value_counts().nlargest(10)
            source_counts.plot(ax=ax2, kind="barh", color="green", alpha=0.7)
            ax2.set_title("Top 10 Sources", fontsize=12)
            ax2.set_xlabel("Number of Items")
            ax2.grid(True, alpha=0.3)
        
        # 3. Breakdown by type
        ax3 = fig.add_subplot(gs[1, 1])
        if "type" in df.columns:
            type_counts = df["type"].value_counts()
            type_counts.plot(ax=ax3, kind="pie", autopct="%1.1f%%", startangle=90, 
                         colors=plt.cm.Paired(range(len(type_counts))))
            ax3.set_title("Data by Type", fontsize=12)
            ax3.set_ylabel("")
        
        # 4. Breakdown by relevance
        ax4 = fig.add_subplot(gs[2, 0])
        if "relevance" in df.columns:
            relevance_counts = df["relevance"].value_counts()
            colors = {"high": "green", "medium": "orange", "low": "red", "inherited": "gray"}
            relevance_colors = [colors.get(r, "blue") for r in relevance_counts.index]
            relevance_counts.plot(ax=ax4, kind="bar", color=relevance_colors, alpha=0.7)
            ax4.set_title("Data by Relevance", fontsize=12)
            ax4.set_ylabel("Number of Items")
            ax4.grid(True, alpha=0.3)
        
        # 5. Sentiment distribution if available
        ax5 = fig.add_subplot(gs[2, 1])
        sentiment_col = None
        if "sentiment_score" in df.columns:
            sentiment_col = "sentiment_score"
        elif "sentiment_compound" in df.columns:
            sentiment_col = "sentiment_compound"
        
        if sentiment_col:
            df[sentiment_col].hist(ax=ax5, bins=30, alpha=0.7, color="purple")
            ax5.set_title("Sentiment Distribution", fontsize=12)
            ax5.set_xlabel("Sentiment Score")
            ax5.set_ylabel("Frequency")
            ax5.grid(True, alpha=0.3)
            
            # Add lines for positive/negative thresholds
            ax5.axvline(x=0.05, color="green", linestyle="--", alpha=0.5, label="Positive")
            ax5.axvline(x=-0.05, color="red", linestyle="--", alpha=0.5, label="Negative")
            ax5.axvline(x=0, color="black", linestyle="-", alpha=0.3)
            ax5.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if output_file:
            save_path = os.path.join(self.output_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved coverage visualization to {save_path}")
        
        return fig


def run_historical_scraping(
    start_year: int = 1987,
    end_year: Optional[int] = None,
    output_dir: str = "data/processed/historical",
    include_reddit: bool = True,
    include_news: bool = True,
    include_financial: bool = True,
    create_visualizations: bool = True
) -> pd.DataFrame:
    """
    Run a complete historical data collection process.
    
    This function initializes and runs the EnhancedRedditScraper to collect
    data from multiple sources spanning the full date range.
    
    Args:
        start_year: Start year for data collection
        end_year: End year for data collection (defaults to current year)
        output_dir: Directory to save output files
        include_reddit: Whether to include Reddit data
        include_news: Whether to include news articles
        include_financial: Whether to include financial datasets
        create_visualizations: Whether to create coverage visualizations
        
    Returns:
        DataFrame with the comprehensive dataset
    """
    # Set default end year if not provided
    if end_year is None:
        end_year = datetime.datetime.now().year
    
    # Initialize scraper
    scraper = EnhancedRedditScraper(output_dir=output_dir)
    
    # Create comprehensive dataset
    dataset = scraper.create_comprehensive_dataset(
        start_year=start_year,
        end_year=end_year,
        include_reddit=include_reddit,
        include_news=include_news,
        include_financial=include_financial
    )
    
    # Generate statistics and visualizations
    if not dataset.empty and create_visualizations:
        # Analyze coverage
        coverage_stats = scraper.analyze_coverage(dataset)
        
        # Save coverage statistics
        stats_path = os.path.join(output_dir, "coverage_stats.json")
        try:
            with open(stats_path, 'w') as f:
                json.dump(coverage_stats, f, indent=4)
            logger.info(f"Saved coverage statistics to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving coverage statistics: {str(e)}")
        
        # Create visualizations
        scraper.visualize_coverage(
            dataset,
            output_file="coverage_visualization.png"
        )
    
    return dataset


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Historical Data Scraper for Oil Market Sentiment")
    parser.add_argument("--start-year", type=int, default=1987, help="Start year for data collection")
    parser.add_argument("--end-year", type=int, help="End year for data collection (defaults to current year)")
    parser.add_argument("--output-dir", type=str, default="data/processed/historical", help="Output directory")
    parser.add_argument("--skip-reddit", action="store_true", help="Skip Reddit data collection")
    parser.add_argument("--skip-news", action="store_true", help="Skip news article collection")
    parser.add_argument("--skip-financial", action="store_true", help="Skip financial dataset processing")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip creating visualizations")
    
    args = parser.parse_args()
    
    # Run the scraper
    dataset = run_historical_scraping(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        include_reddit=not args.skip_reddit,
        include_news=not args.skip_news,
        include_financial=not args.skip_financial,
        create_visualizations=not args.skip_visualizations
    )
    
    print(f"Collected {len(dataset)} total items for sentiment analysis")
    print(f"Data spans from {dataset['created_date'].min()} to {dataset['created_date'].max()}")
    
    # Print source breakdown
    if 'source' in dataset.columns:
        print("\nSource breakdown:")
        for source, count in dataset['source'].value_counts().items():
            print(f"  {source}: {count} items")
    
    print("\nCollection complete. Results saved to output directory.")