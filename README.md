# Oil Prophet

![Project Status](https://img.shields.io/badge/status-in_development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

## Advanced Oil Price Forecasting System

Oil Prophet is a sophisticated forecasting system that combines advanced time-series decomposition techniques with deep learning (LSTM with attention mechanism) and alternative data sources to predict oil price movements with higher accuracy than traditional methods.

> 🚀 **Development Progress**: Core components including data processing, signal decomposition, forecasting models, and historical data scraping have been implemented and successfully tested. Reddit historical scraping with comments is now fully functional, and sentiment analysis integration is under active development.

## Features

### Implemented
- **Data Processing Pipeline**: Robust data loading and preprocessing for multi-timeframe analysis
- **Signal Decomposition**: Time series decomposition into trend, cyclical, and residual components using advanced filtering techniques
- **LSTM with Attention Model**: Deep learning forecasting with attention mechanisms to focus on relevant parts of the input sequence
- **Baseline Models**: Simple forecasting models for performance benchmarking
- **Ensemble Approach**: Combines predictions from multiple models for more robust forecasts
- **Enhanced Reddit Historical Scraper**: Comprehensive system for collecting oil-related discussions from Reddit spanning from 2008 to present
  - Scrapes both posts and comments for each relevant discussion
  - Implements relevance filtering to focus on oil-related content
  - Maintains relationship between posts and their comments
  - Supports multi-subreddit collection with configurable parameters
- **Data Coverage Visualization**: Tools to analyze and visualize the coverage of collected data across different time periods
- **Visualization System**: Comprehensive visualization tools for time series data, decomposition, forecasts, and model comparison
- **Evaluation Framework**: Rigorous model evaluation with multiple metrics and statistical significance testing

### In Progress
- **Sentiment Analysis Integration**: Currently enhancing FinBERT-based sentiment analysis for oil market-specific content
- **Historical News Data Collection**: Working on methods to collect pre-Reddit era (1987-2008) news articles
- **Sentiment-Enhanced Forecasting**: Integrating sentiment features with price data for improved prediction accuracy

### Planned
- **Interactive Dashboard**: Web-based interface for exploring forecasts and model performance
- **API Development**: Exposing model predictions through a REST API
- **Temporal Sentiment Aggregation**: Methods to aggregate sentiment across different timeframes for time-series analysis

## Project Structure

```
oil-prophet/
├── data/
│   ├── raw/                 # Raw oil price data files (WTI, Brent crude)
│   └── processed/           # Processed datasets and sentiment data
│       └── reddit_test_small/   # Historical Reddit data with posts and comments
├── models/                  # Saved model files and evaluation results
├── notebooks/
│   └── plots/               # Generated visualization plots
├── src/
│   ├── data/
│   │   └── preprocessing.py # Data loading and preprocessing
│   ├── models/
│   │   ├── ceemdan.py       # Signal decomposition implementation
│   │   ├── lstm_attention.py # LSTM with attention implementation
│   │   ├── ensemble.py      # Ensemble forecasting models
│   │   └── baseline.py      # Baseline forecasting models
│   ├── evaluation/
│   │   └── metrics.py       # Evaluation metrics and testing
│   ├── visualization/
│   │   └── plots.py         # Visualization functions
│   ├── nlp/
│   │   ├── reddit_historical_scraper.py # Enhanced Reddit data scraper
│   │   ├── finbert_sentiment.py         # FinBERT sentiment analysis (in progress)
│   │   ├── bert_sentiment.py            # BERT sentiment analysis
│   │   ├── config_setup.py              # Reddit API configuration
│   │   └── sentiment_demo.py            # Sentiment integration demo
│   └── api/                 # (Planned) API implementation
├── tests/
├── requirements.txt
├── setup_checker.py         # System setup verification
├── .gitignore
├── README.md
└── LICENSE
```

## Recent Achievements

✅ **Enhanced Reddit Scraper with Comments**: Successfully implemented a comprehensive Reddit scraper that collects both posts and their associated comments across multiple financial subreddits, with proper relationship maintenance between posts and comments.

✅ **Multi-Subreddit Coverage**: The scraper successfully collects data from r/investing, r/stocks, r/stockmarket, r/wallstreetbets, r/options, r/finance, r/SecurityAnalysis, r/economics, r/economy, and r/business, providing a broad spectrum of market sentiment.

✅ **Data Coverage Analysis**: Built tools to analyze and visualize the coverage of collected data, helping identify gaps in historical sentiment data.

✅ **FinBERT Integration**: Began integration of specialized financial BERT models for improved sentiment analysis of oil market-specific content.

## Roadmap

- [x] Project structure setup
- [x] Data collection (historical oil prices)
- [x] Data preprocessing pipeline
- [x] Signal decomposition implementation
- [x] LSTM with attention implementation
- [x] Baseline models implementation
- [x] Ensemble model development
- [x] Visualization system
- [x] Model evaluation framework
- [x] Historical Reddit data scraper with comments
- [ ] Complete sentiment analysis implementation
- [ ] Historical news data collection (pre-Reddit era)
- [ ] Sentiment-enhanced feature creation
- [ ] Interactive visualization dashboard
- [ ] API development

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies listed in requirements.txt
- Reddit API credentials (for historical data scraping)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/oil-prophet.git
   cd oil-prophet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify setup:
   ```bash
   python setup_checker.py
   ```

4. Set up Reddit API credentials:
   ```bash
   python -m src.nlp.config_setup
   ```
   Update the generated `reddit_config.json` with your credentials.

### Usage Examples

1. Data preprocessing:
   ```bash
   python -m src.data.preprocessing
   ```

2. Run signal decomposition:
   ```bash
   python -m src.models.ceemdan
   ```

3. Generate visualizations:
   ```bash
   python -m src.visualization.plots
   ```

4. Collect historical Reddit data with comments:
   ```bash
   python -m src.nlp.reddit_historical_scraper --start-year=2020 --end-year=2023 --skip-news --skip-financial --output-dir=data/processed/reddit_data
   ```

5. Evaluate model performance:
   ```bash
   python -m src.evaluation.metrics
   ```

## Data Collection Details

### Reddit Historical Scraper

The Reddit historical scraper collects data from financial subreddits with these key features:

1. **Post and Comment Collection**: For each relevant post, the scraper also collects all associated comments, providing a complete picture of the discussion.

2. **Relevance Filtering**: Uses keyword matching and relevance scoring to focus on oil-related content, with configurable relevance thresholds.

3. **Data Structure**:
   - Posts include title, text content, author, score, comment count, and creation date
   - Comments include text, author, score, parent post ID, and creation date
   - Relationship between posts and comments is maintained through post_id references

4. **Comprehensive Coverage**: Collects data from multiple subreddits focused on finance, investing, and economics to capture diverse perspectives.

5. **Historical Range**: Can collect data from Reddit's inception (2008) to present day.

6. **Quality Filters**: Includes options to filter by post score and comment quality to focus on significant discussions.

7. **Data Storage**: Saves data in both individual files per subreddit and a combined comprehensive dataset.

### Planned News Data Collection

To complement Reddit data (which only goes back to 2008), the system is being enhanced to collect historical news articles about oil markets from 1987-2008 using:

1. News archive APIs (NYTimes, etc.)
2. Financial news databases
3. Simulated historical data for periods with limited digital records

## Applications

This forecasting system is valuable for:

- **Investment Decisions**: Helping investors make informed decisions for commodities trading
- **Risk Management**: Assisting companies in hedging strategies based on expected price movements
- **Budget Planning**: Supporting businesses in financial planning that depends on oil price forecasts
- **Market Research**: Providing insights into the relationship between market sentiment and oil prices

## Current Development Focus

### Enhanced Sentiment Analysis

We're currently focused on improving the sentiment analysis component of the system:

1. **FinBERT Integration**: Implementing specialized financial BERT models that are pre-trained on financial texts for better sentiment understanding of oil market discussions.

2. **Comment-Level Sentiment Analysis**: Developing methods to analyze sentiment at the comment level and aggregate it for each discussion topic.

3. **Historical Data Coverage**: Working on methods to fill gaps in historical sentiment data, especially for the pre-Reddit era (1987-2008).

4. **Oil Market Domain Adaptation**: Adapting sentiment models to better understand oil-specific terminology and market dynamics.

5. **Multi-Source Sentiment Aggregation**: Developing methods to combine sentiment signals from multiple sources (Reddit, news, financial datasets).

6. **Temporal Sentiment Features**: Creating time-based sentiment features that can capture market sentiment shifts over different timeframes.

### Future Work

The next major milestone is to complete the sentiment analysis pipeline and integrate it with the forecasting models. This will allow the system to generate enhanced feature vectors that combine technical price patterns with market sentiment indicators, potentially improving prediction accuracy during periods of high market emotion.

## License

This project is licensed under the MIT License

## Acknowledgments

- This project's approach is inspired by recent research in hybrid forecasting models
- Oil price data sourced from public datasets