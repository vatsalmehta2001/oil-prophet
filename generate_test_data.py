"""
Generate synthetic oil price data for testing.

This script creates synthetic data files with realistic oil price patterns
for testing the Oil Prophet forecasting system.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_oil_prices(
    start_date: str = '2010-01-01',
    end_date: str = '2023-12-31',
    base_price: float = 60.0,
    volatility: float = 0.15,
    trend: float = 0.01,
    seasonality: float = 10.0,
    random_seed: int = 42
):
    """
    Generate synthetic oil price data with trend, seasonality, and noise.
    
    Args:
        start_date: Start date for the time series
        end_date: End date for the time series
        base_price: Base price level
        volatility: Volatility of price changes
        trend: Long-term trend component
        seasonality: Magnitude of seasonal effects
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with dates and prices
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Create time index
    t = np.arange(n_days)
    
    # Generate components
    
    # Trend: linear trend with some variation
    trend_component = base_price + trend * t
    
    # Seasonality: annual cycle (365 days)
    seasonal_component = seasonality * np.sin(2 * np.pi * t / 365)
    
    # Add a business cycle (approximately 3-year cycle)
    business_cycle = seasonality * 1.5 * np.sin(2 * np.pi * t / (365 * 3))
    
    # Random component (white noise)
    random_component = np.random.normal(0, volatility * base_price, n_days)
    
    # OPEC shocks (random spikes or drops)
    n_shocks = int(n_days / 365)  # About one shock per year
    shock_indices = np.random.choice(n_days, n_shocks, replace=False)
    shock_component = np.zeros(n_days)
    for idx in shock_indices:
        magnitude = np.random.normal(0, base_price * 0.2)
        decay_length = np.random.randint(30, 90)  # Shock effects last 1-3 months
        
        # Exponential decay of shock effect
        for i in range(min(decay_length, n_days - idx)):
            decay_factor = np.exp(-i / (decay_length / 3))
            if idx + i < n_days:
                shock_component[idx + i] += magnitude * decay_factor
    
    # Combine all components
    prices = trend_component + seasonal_component + business_cycle + random_component + shock_component
    
    # Ensure prices are positive
    prices = np.maximum(prices, 10.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Price': prices
    })
    
    return df

def create_different_frequencies(daily_data):
    """
    Create weekly, monthly, and yearly data from daily data.
    
    Args:
        daily_data: DataFrame with daily data
        
    Returns:
        Dictionary with different frequency DataFrames
    """
    # Set date as index for resampling
    data = daily_data.copy()
    data.set_index('Date', inplace=True)
    
    # Create weekly data (mean of each week)
    weekly = data.resample('W').mean().reset_index()
    
    # Create monthly data
    monthly = data.resample('M').mean().reset_index()
    
    # Create yearly data
    yearly = data.resample('Y').mean().reset_index()
    
    # Reset index on daily data
    daily = data.reset_index()
    
    return {
        'daily': daily,
        'weekly': weekly,
        'monthly': monthly,
        'year': yearly
    }

def save_data_files(output_dir: str = 'data/raw'):
    """
    Generate and save synthetic oil price data files.
    
    Args:
        output_dir: Directory to save the data files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data for different oil types with slightly different parameters
    oil_types = {
        'brent': {
            'base_price': 65.0,
            'volatility': 0.15,
            'trend': 0.01,
            'random_seed': 42
        },
        'wti': {
            'base_price': 60.0,
            'volatility': 0.17,
            'trend': 0.008,
            'random_seed': 43
        }
    }
    
    for oil_type, params in oil_types.items():
        # Generate daily data
        daily_data = generate_oil_prices(**params)
        
        # Create different frequencies
        frequency_data = create_different_frequencies(daily_data)
        
        # Save each frequency to a file
        for freq, data in frequency_data.items():
            filename = f"{oil_type}-{freq}.csv"
            filepath = os.path.join(output_dir, filename)
            data.to_csv(filepath, index=False)
            print(f"Saved {len(data)} records to {filepath}")

if __name__ == "__main__":
    print("Generating synthetic oil price data...")
    save_data_files()
    print("Done!")