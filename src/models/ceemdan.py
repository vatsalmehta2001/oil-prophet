"""
Simplified decomposition module for Oil Prophet.

This module implements a simplified version of EMD (Empirical Mode Decomposition)
without requiring the PyEMD package. It uses a combination of filtering techniques
to decompose time series signals into components similar to IMFs.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union
import logging
import joblib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimplifiedDecomposer:
    """
    Class for decomposing time series data using a simplified approach.
    
    This class uses a combination of moving averages and filtering to decompose
    a signal into trend, cyclical, and noise components, which approximate IMFs.
    """
    
    def __init__(
        self, 
        n_components: int = 5, 
        random_state: int = 42
    ):
        """
        Initialize the decomposer.
        
        Args:
            n_components: Number of components to extract
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.components = None
        self.trend = None
        
    def _smooth(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply smoothing to the signal using Savitzky-Golay filter.
        
        Args:
            signal: Input signal
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed signal
        """
        if window_size % 2 == 0:
            window_size += 1  # Make sure window size is odd
        
        # Use minimum polynomial order
        poly_order = min(3, window_size - 1)
        
        try:
            return savgol_filter(signal, window_size, poly_order)
        except Exception as e:
            logger.warning(f"Savgol filter failed, using moving average instead: {e}")
            return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    
    def _extract_trend(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract the trend component from a signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Trend component
        """
        # Use a relatively large window size for the trend
        window_size = max(int(len(signal) * 0.1), 5)
        window_size = min(window_size, len(signal) - 1)
        
        return self._smooth(signal, window_size)
    
    def _extract_cyclical(self, signal: np.ndarray, period: int) -> np.ndarray:
        """
        Extract a cyclical component with a specific period.
        
        Args:
            signal: Input signal
            period: Period of the cycle
            
        Returns:
            Cyclical component
        """
        # Design bandpass filter for the specific period
        fs = 1.0  # Normalized frequency
        lowcut = 0.5 / period
        highcut = 2.0 / period
        
        # Create a bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)  # Ensure it's below Nyquist
        
        # Use a butterwoth filter with safe order
        order = min(3, len(signal) // 10)
        order = max(order, 1)
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            return signal.filtfilt(b, a, signal)
        except Exception as e:
            logger.warning(f"Butterworth filter failed, using simple bandpass: {e}")
            # Simple bandpass using difference of moving averages
            ma_short = self._smooth(signal, max(3, period // 4))
            ma_long = self._smooth(signal, period * 2)
            return ma_short - ma_long
    
    def _extract_noise(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Extract high-frequency noise component.
        
        Args:
            signal: Input signal
            window_size: Size of the smoothing window
            
        Returns:
            Noise component
        """
        # Get smoothed signal
        smoothed = self._smooth(signal, window_size)
        
        # Noise is the difference
        return signal - smoothed
    
    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        Decompose a signal into components.
        
        Args:
            signal: The time series signal to decompose
            
        Returns:
            Array containing all components
        """
        # Ensure signal is 1D
        signal = np.ravel(signal)
        
        logger.info(f"Decomposing signal of length {len(signal)}")
        
        try:
            # Initialize components list
            components = []
            residual = signal.copy()
            
            # Extract trend component
            trend = self._extract_trend(residual)
            components.append(trend)
            residual = residual - trend
            
            # Extract cyclical components with different periods
            signal_length = len(signal)
            
            # Calculate periods based on signal length
            if self.n_components <= 2:
                periods = [signal_length // 4]
            else:
                min_period = max(4, int(signal_length * 0.01))
                max_period = min(int(signal_length * 0.2), signal_length // 2)
                periods = np.geomspace(min_period, max_period, self.n_components - 2)
                periods = periods.astype(int)
            
            for period in periods:
                if period < 3:  # Skip very short periods
                    continue
                    
                cyclical = self._extract_cyclical(residual, period)
                components.append(cyclical)
                residual = residual - cyclical
            
            # Add the residual (noise component)
            components.append(residual)
            
            # Store components
            self.components = np.array(components)
            self.trend = trend
            
            logger.info(f"Decomposition complete. Extracted {len(self.components)} components")
            
            return self.components
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            raise
    
    def get_components(self) -> np.ndarray:
        """
        Get the decomposed components.
        
        Returns:
            Array of components
        """
        if self.components is None:
            raise ValueError("No components available. Run decompose() first.")
        
        return self.components
    
    def get_trend(self) -> np.ndarray:
        """
        Get the trend component.
        
        Returns:
            Trend array
        """
        if self.trend is None:
            raise ValueError("No trend available. Run decompose() first.")
        
        return self.trend
    
    def plot_decomposition(
        self, 
        time_index: Optional[np.ndarray] = None, 
        figsize: Tuple[int, int] = (12, 15),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the original signal and components.
        
        Args:
            time_index: Optional time indices for x-axis
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.components is None:
            raise ValueError("No components available. Run decompose() first.")
        
        n_components = len(self.components)
        
        # Create x-axis values if not provided
        if time_index is None:
            time_index = np.arange(len(self.components[0]))
        
        # Create figure
        fig, axes = plt.subplots(n_components + 1, 1, figsize=figsize, sharex=True)
        
        # Plot original signal
        original_signal = np.sum(self.components, axis=0)
        axes[0].plot(time_index, original_signal, 'k')
        axes[0].set_title('Original Signal')
        axes[0].set_ylabel('Amplitude')
        
        # Plot components
        for i, component in enumerate(self.components):
            if i == 0:
                axes[i+1].plot(time_index, component, 'r')
                axes[i+1].set_title(f'Trend')
            elif i == n_components - 1:
                axes[i+1].plot(time_index, component, 'g')
                axes[i+1].set_title(f'Residual (Noise)')
            else:
                axes[i+1].plot(time_index, component)
                axes[i+1].set_title(f'Cyclical Component {i}')
            axes[i+1].set_ylabel('Amplitude')
        
        # Set x-label for the last subplot
        axes[-1].set_xlabel('Time')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved decomposition plot to {save_path}")
        
        return fig
    
    def save(self, filepath: str) -> None:
        """
        Save the decomposer to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'n_components': self.n_components,
            'random_state': self.random_state,
            'components': self.components,
            'trend': self.trend
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
        logger.info(f"Decomposition model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SimplifiedDecomposer':
        """
        Load a decomposer from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            SimplifiedDecomposer instance
        """
        model_data = joblib.load(filepath)
        
        decomposer = cls(
            n_components=model_data['n_components'],
            random_state=model_data['random_state']
        )
        
        decomposer.components = model_data['components']
        decomposer.trend = model_data['trend']
        
        logger.info(f"Decomposition model loaded from {filepath}")
        return decomposer


# Alias for backward compatibility with original code
CEEMDANDecomposer = SimplifiedDecomposer


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import OilDataProcessor
    
    # Load and prepare data
    processor = OilDataProcessor()
    data = processor.load_data(oil_type="brent", freq="daily")
    signal = data['Price'].values
    
    # Run decomposition
    decomposer = SimplifiedDecomposer(n_components=5)
    components = decomposer.decompose(signal)
    
    # Plot and save the decomposition
    fig = decomposer.plot_decomposition(time_index=data.index)
    plt.show()
    
    # Save the decomposer
    decomposer.save('models/decomposition_brent_daily.pkl')