# Oil Prophet

![Project Status](https://img.shields.io/badge/status-in_development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

## Advanced Oil Price Forecasting System

Oil Prophet is a sophisticated forecasting system that combines advanced time-series decomposition techniques with deep learning (LSTM with attention mechanism) and market sentiment analysis to predict oil price movements with higher accuracy than traditional methods.

> 🚀 **Development Progress**: Core components including data processing, signal decomposition, forecasting models, and sentiment analysis integration have been implemented. Reddit historical scraping with comments is fully functional, and sentiment-enhanced LSTM model now successfully integrates market sentiment with price patterns. Interactive dashboard is currently under development.

## Features

### Implemented
- **Data Processing Pipeline**: Robust data loading and preprocessing for multi-timeframe analysis
- **Signal Decomposition**: Time series decomposition into trend, cyclical, and residual components using advanced filtering techniques
- **LSTM with Attention Model**: Deep learning forecasting with attention mechanisms to focus on relevant parts of the input sequence
- **Sentiment-Enhanced LSTM**: Advanced model that combines price patterns with market sentiment indicators for improved forecasting
- **Baseline Models**: Simple forecasting models for performance benchmarking
- **Ensemble Approach**: Combines predictions from multiple models for more robust forecasts
- **Enhanced Reddit Historical Scraper**: Comprehensive system for collecting oil-related discussions from Reddit spanning from 2008 to present
  - Scrapes both posts and comments for each relevant discussion
  - Implements relevance filtering to focus on oil-related content
  - Maintains relationship between posts and their comments
  - Supports multi-subreddit collection with configurable parameters
- **Sentiment Analysis Pipeline**: FinBERT-based sentiment analysis specifically adapted for oil market content
- **Data Coverage Visualization**: Tools to analyze and visualize the coverage of collected data across different time periods
- **Visualization System**: Comprehensive visualization tools for time series data, decomposition, forecasts, and model comparison
- **Evaluation Framework**: Rigorous model evaluation with multiple metrics and statistical significance testing

### In Progress
- **Interactive Dashboard**: Web-based interface for exploring forecasts, model performance, and sentiment analysis (currently under active development)
- **Historical News Data Collection**: Working on methods to collect pre-Reddit era (1987-2008) news articles
- **Temporal Sentiment Aggregation**: Methods to aggregate sentiment across different timeframes for time-series analysis

### Planned
- **API Development**: Exposing model predictions through a REST API
- **Automated Retraining Pipeline**: System for periodically retraining models with new data
- **Forecast Uncertainty Quantification**: Methods to provide confidence intervals for predictions

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
│   │   ├── sentiment_enhanced_lstm.py # Sentiment-enhanced LSTM model
│   │   ├── ensemble.py      # Ensemble forecasting models
│   │   └── baseline.py      # Baseline forecasting models
│   ├── evaluation/
│   │   └── metrics.py       # Evaluation metrics and testing
│   ├── visualization/
│   │   └── plots.py         # Visualization functions
│   ├── nlp/
│   │   ├── reddit_historical_scraper.py # Enhanced Reddit data scraper
│   │   ├── finbert_sentiment.py         # FinBERT sentiment analysis
│   │   ├── bert_sentiment.py            # BERT sentiment analysis
│   │   ├── config_setup.py              # Reddit API configuration
│   │   └── sentiment_demo.py            # Sentiment integration demo
│   └── api/                 # (Planned) API implementation
├── dashboard.py             # Interactive Streamlit dashboard (in development)
├── run_forecast_pipeline.py # CLI for running the full forecasting pipeline
├── test_sentiment_model.py  # Testing script for sentiment-enhanced models
├── tests/
├── requirements.txt
├── setup_checker.py         # System setup verification
├── .gitignore
├── README.md
└── LICENSE
```

## Recent Achievements

✅ **Sentiment-Enhanced LSTM Implementation**: Successfully integrated market sentiment features with technical price patterns in a combined forecasting model, showing improved accuracy over baseline models.

✅ **Enhanced Reddit Scraper with Comments**: Implemented a comprehensive Reddit scraper that collects both posts and their associated comments across multiple financial subreddits, with proper relationship maintenance between posts and comments.

✅ **Multi-Subreddit Coverage**: The scraper successfully collects data from r/investing, r/stocks, r/stockmarket, r/wallstreetbets, r/options, r/finance, r/SecurityAnalysis, r/economics, r/economy, and r/business, providing a broad spectrum of market sentiment.

✅ **FinBERT Integration**: Successfully integrated specialized financial BERT models for improved sentiment analysis of oil market-specific content.

✅ **CLI Interface**: Developed a comprehensive command-line interface for running the complete forecasting pipeline.

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
- [x] Complete sentiment analysis implementation
- [x] Sentiment-enhanced LSTM model
- [ ] Interactive visualization dashboard (in progress)
- [ ] Historical news data collection (pre-Reddit era)
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

1. Run the complete forecasting pipeline:
   ```bash
   python run_forecast_pipeline.py --run-all
   ```

2. Test the sentiment-enhanced LSTM model:
   ```bash
   python test_sentiment_model.py --data_dir=data/processed/reddit_test_small
   ```

3. Run the interactive dashboard (currently in development):
   ```bash
   streamlit run dashboard.py
   ```

4. Data preprocessing:
   ```bash
   python -m src.data.preprocessing
   ```

5. Run signal decomposition:
   ```bash
   python -m src.models.ceemdan
   ```

6. Generate visualizations:
   ```bash
   python -m src.visualization.plots
   ```

7. Collect historical Reddit data with comments:
   ```bash
   python -m src.nlp.reddit_historical_scraper --start-year=2020 --end-year=2023 --skip-news --skip-financial --output-dir=data/processed/reddit_data
   ```

8. Analyze sentiment in collected data:
   ```bash
   python -m src.nlp.finbert_sentiment --input=data/processed/reddit_test_small/comprehensive_sentiment_dataset.csv
   ```

## Sentiment Analysis Integration

The sentiment analysis pipeline now fully integrates with the forecasting models:

1. **FinBERT-Based Analysis**: Uses specialized financial BERT models pre-trained on financial texts for better sentiment understanding of oil market discussions.

2. **Domain Adaptation**: Implements oil market-specific term weighting to improve sentiment accuracy for industry terminology.

3. **Temporal Aggregation**: Aggregates sentiment by time period (daily, weekly, monthly) for alignment with price data.

4. **Feature Integration**: Combines sentiment features with price data in the input sequence for LSTM models.

5. **Performance Enhancement**: Testing shows improved forecasting accuracy, especially during periods of high market volatility or significant news events.

## Interactive Dashboard (In Development)

The interactive Streamlit dashboard is currently under development and includes:

1. **Forecast Explorer**: Compare predictions from different models, including sentiment-enhanced forecasts.

2. **Model Performance Analysis**: Evaluate and compare model performance with various metrics.

3. **Sentiment Analysis Visualization**: Explore the relationship between market sentiment and price movements.

4. **Signal Decomposition Analysis**: Visualize the components of price signals and their contributions.

**Note**: The dashboard is still in active development and may have stability issues. We're working to resolve these and improve the user experience.

## Applications

This forecasting system is valuable for:

- **Investment Decisions**: Helping investors make informed decisions for commodities trading
- **Risk Management**: Assisting companies in hedging strategies based on expected price movements
- **Budget Planning**: Supporting businesses in financial planning that depends on oil price forecasts
- **Market Research**: Providing insights into the relationship between market sentiment and oil prices

## Future Work

Our next major milestones include:

1. **Completing the Interactive Dashboard**: Finishing and stabilizing the Streamlit dashboard for easy exploration of forecasts and models.

2. **Historical News Data Collection**: Expanding data sources to include pre-Reddit era (1987-2008) news articles.

3. **API Development**: Building a REST API for integrating predictions with other systems.

4. **Forecast Uncertainty Quantification**: Adding confidence intervals and probability distributions to predictions.

## License

This project is licensed under the MIT License

## Acknowledgments

- This project's approach is inspired by recent research in hybrid forecasting models
- Oil price data sourced from public datasets