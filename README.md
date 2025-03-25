# Oil Prophet

![Project Status](https://img.shields.io/badge/status-in_development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

## Advanced Oil Price Forecasting System

Oil Prophet is a sophisticated forecasting system that combines advanced time-series decomposition techniques (CEEMDAN) with deep learning (LSTM with attention mechanism) and alternative data sources (Reddit sentiment) to predict oil price movements with higher accuracy than traditional methods.

> ⚠️ **This project is currently under development**. Many features are not yet implemented, and the code is subject to significant changes.

## Features (Planned)

- **Hybrid CEEMDAN-LSTM Architecture**: Leverages Complete Ensemble Empirical Mode Decomposition with Adaptive Noise for signal decomposition and Long Short-Term Memory networks with attention mechanisms for accurate predictions
- **Multi-timeframe Analysis**: Support for daily, weekly, monthly, and yearly price data for both WTI and Brent crude oil
- **Market Sentiment Integration**: Analyzes sentiment from Reddit financial communities to incorporate market psychology into price forecasts
- **Ensemble Approach**: Combines technical forecasts with sentiment signals for more robust predictions
- **Interactive Visualizations**: Clear visualization of forecasts, model performance, and sentiment indicators

## Project Structure

```
oil-prophet/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── ceemdan.py
│   │   ├── lstm_attention.py
│   │   ├── ensemble.py
│   │   └── baseline.py
│   ├── nlp/
│   ├── evaluation/
│   ├── visualization/
│   └── api/
├── tests/
├── requirements.txt
├── setup.py
├── .gitignore
├── README.md
└── LICENSE
```

## Roadmap

- [x] Project structure setup
- [x] Data collection (historical oil prices)
- [ ] Data preprocessing pipeline
- [ ] CEEMDAN implementation
- [ ] LSTM with attention implementation
- [ ] Reddit sentiment analysis integration
- [ ] Ensemble model development
- [ ] Model evaluation framework
- [ ] Interactive visualization dashboard
- [ ] API development

## Getting Started

As this project is still in early development, full installation and usage instructions will be provided once a working prototype is available.

### Prerequisites (Planned)

- Python 3.8+
- TensorFlow 2.x
- PyEMD
- PRAW (Python Reddit API Wrapper)
- Pandas, NumPy, Matplotlib
- Scikit-learn

## Applications

When completed, this forecasting system will be valuable for:

- **Investment Decisions**: Helping investors make informed decisions for commodities trading
- **Risk Management**: Assisting companies in hedging strategies based on expected price movements
- **Budget Planning**: Supporting businesses in financial planning that depends on oil price forecasts
- **Market Research**: Providing insights into the relationship between social sentiment and oil prices

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project's approach is inspired by recent research in hybrid forecasting models
- Oil price data sourced from public datasets