# Oil Prophet

![Project Status](https://img.shields.io/badge/status-in_development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

## Advanced Oil Price Forecasting System

Oil Prophet is a sophisticated forecasting system that combines advanced time-series decomposition techniques with deep learning (LSTM with attention mechanism) and alternative data sources to predict oil price movements with higher accuracy than traditional methods.

> 🚀 **Development Progress**: Core components including data processing, signal decomposition, and forecasting models have been implemented. The system now supports visualization and evaluation of model performance.

## Features

### Implemented
- **Data Processing Pipeline**: Robust data loading and preprocessing for multi-timeframe analysis
- **Signal Decomposition**: Time series decomposition into trend, cyclical, and residual components using advanced filtering techniques
- **LSTM with Attention Model**: Deep learning forecasting with attention mechanisms to focus on relevant parts of the input sequence
- **Baseline Models**: Simple forecasting models for performance benchmarking
- **Ensemble Approach**: Combines predictions from multiple models for more robust forecasts
- **Visualization System**: Comprehensive visualization tools for time series data, decomposition, forecasts, and model comparison
- **Evaluation Framework**: Rigorous model evaluation with multiple metrics and statistical significance testing

### Planned
- **Market Sentiment Integration**: Analyzing sentiment from Reddit financial communities to incorporate market psychology
- **Interactive Dashboard**: Web-based interface for exploring forecasts and model performance
- **API Development**: Exposing model predictions through a REST API

## Project Structure

```
oil-prophet/
├── data/
│   ├── raw/                 # Raw oil price data files
│   └── processed/           # Processed datasets
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
│   ├── nlp/                 # (Planned) Sentiment analysis
│   └── api/                 # (Planned) API implementation
├── tests/
├── requirements.txt
├── setup_checker.py         # System setup verification
├── .gitignore
├── README.md
└── LICENSE
```

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
- [ ] Reddit sentiment analysis integration
- [ ] Interactive visualization dashboard
- [ ] API development

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies listed in requirements.txt

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

4. Evaluate model performance:
   ```bash
   python -m src.evaluation.metrics
   ```

## Applications

This forecasting system is valuable for:

- **Investment Decisions**: Helping investors make informed decisions for commodities trading
- **Risk Management**: Assisting companies in hedging strategies based on expected price movements
- **Budget Planning**: Supporting businesses in financial planning that depends on oil price forecasts
- **Market Research**: Providing insights into the relationship between market factors and oil prices

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project's approach is inspired by recent research in hybrid forecasting models
- Oil price data sourced from public datasets