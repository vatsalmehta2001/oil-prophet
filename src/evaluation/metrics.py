"""
Evaluation metrics module for Oil Prophet.

This module provides metrics and evaluation functions for assessing
forecasting performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    scale_factor: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics for forecasting evaluation.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        scale_factor: Optional scale factor for percentage metrics
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure arrays are the same shape
    if y_true.shape != y_pred.shape:
        # If y_pred is 1D and y_true is 2D, reshape y_pred
        if len(y_pred.shape) == 1 and len(y_true.shape) == 2:
            y_pred = y_pred.reshape(-1, 1)
        # If y_true is 1D and y_pred is 2D, reshape y_true
        elif len(y_true.shape) == 1 and len(y_pred.shape) == 2:
            y_true = y_true.reshape(-1, 1)
        else:
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate percentage errors
    if scale_factor is None:
        # Use mean of true values as scale factor if not provided
        scale_factor = np.mean(np.abs(y_true))
    
    mape = 100.0 * np.mean(np.abs((y_true - y_pred) / y_true))
    
    # Calculate normalized errors
    nrmse = rmse / scale_factor * 100.0  # RMSE as percentage of scale
    nmae = mae / scale_factor * 100.0  # MAE as percentage of scale
    
    # Calculate R^2 score
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    # For each consecutive pair of points, check if direction matches
    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true, axis=0))
        pred_dir = np.sign(np.diff(y_pred, axis=0))
        dir_accuracy = np.mean(true_dir == pred_dir) * 100.0
    else:
        dir_accuracy = np.nan
    
    # Return all metrics as a dictionary
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'nrmse': float(nrmse),
        'nmae': float(nmae),
        'r2': float(r2),
        'directional_accuracy': float(dir_accuracy)
    }
    
    return metrics


def evaluate_horizon_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, List[float]]:
    """
    Evaluate performance separately for each forecasting horizon.
    
    Args:
        y_true: Actual values (shape: samples, horizons)
        y_pred: Predicted values (shape: samples, horizons)
        
    Returns:
        Dictionary with metrics for each horizon
    """
    # Check if inputs are multi-horizon
    if len(y_true.shape) < 2 or len(y_pred.shape) < 2:
        raise ValueError("Inputs must have shape (samples, horizons) for horizon evaluation")
    
    # Get number of horizons
    n_horizons = y_true.shape[1]
    
    # Calculate metrics for each horizon
    horizon_metrics = {
        'horizon': list(range(1, n_horizons + 1)),
        'rmse': [],
        'mae': [],
        'mape': []
    }
    
    for h in range(n_horizons):
        # Get true and predicted values for this horizon
        y_true_h = y_true[:, h]
        y_pred_h = y_pred[:, h]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_h, y_pred_h)
        
        # Store metrics
        horizon_metrics['rmse'].append(metrics['rmse'])
        horizon_metrics['mae'].append(metrics['mae'])
        horizon_metrics['mape'].append(metrics['mape'])
    
    return horizon_metrics


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    scale_factor: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple models.
    
    Args:
        y_true: Actual values
        predictions: Dictionary of model predictions {model_name: predictions}
        scale_factor: Optional scale factor for percentage metrics
        
    Returns:
        Dictionary with metrics for each model
    """
    comparison = {}
    
    for model_name, y_pred in predictions.items():
        # Calculate metrics for this model
        metrics = calculate_metrics(y_true, y_pred, scale_factor)
        
        # Store metrics
        comparison[model_name] = metrics
    
    return comparison


def evaluate_ensemble(
    y_true: np.ndarray,
    individual_predictions: Dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
    scale_factor: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate performance of individual models and their ensemble.
    
    Args:
        y_true: Actual values
        individual_predictions: Dictionary of individual model predictions
        ensemble_prediction: Predictions from the ensemble model
        scale_factor: Optional scale factor for percentage metrics
        
    Returns:
        Dictionary with metrics for each model and the ensemble
    """
    # Evaluate individual models
    comparison = compare_models(y_true, individual_predictions, scale_factor)
    
    # Evaluate ensemble
    ensemble_metrics = calculate_metrics(y_true, ensemble_prediction, scale_factor)
    comparison['Ensemble'] = ensemble_metrics
    
    return comparison


def calculate_diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    horizon: int = 1
) -> Tuple[float, float]:
    """
    Calculate Diebold-Mariano test for comparing forecast accuracy.
    
    The Diebold-Mariano test assesses whether two forecasting methods have
    equal predictive accuracy.
    
    Args:
        y_true: Actual values
        y_pred1: Predictions from first model
        y_pred2: Predictions from second model
        horizon: Forecast horizon (for calculating autocorrelation adjustment)
        
    Returns:
        DM statistic and p-value
    """
    from scipy import stats
    
    # Calculate squared errors for each model
    err1 = (y_true - y_pred1) ** 2
    err2 = (y_true - y_pred2) ** 2
    
    # Calculate loss differential
    d = err1 - err2
    
    # Calculate mean loss differential
    d_bar = np.mean(d)
    
    # Calculate autocovariance of loss differential
    n = len(d)
    gamma_0 = np.sum((d - d_bar) ** 2) / n
    
    # Calculate autocorrelation adjustments for h-step ahead forecasts
    gamma = np.zeros(horizon)
    for i in range(horizon):
        gamma[i] = np.sum((d[i+1:] - d_bar) * (d[:-(i+1)] - d_bar)) / n
    
    # Calculate variance of mean loss differential
    var_d_bar = (gamma_0 + 2 * np.sum(gamma)) / n
    
    # Calculate DM statistic
    dm_stat = d_bar / np.sqrt(var_d_bar)
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def statistical_significance_tests(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    baseline_model: str,
    alpha: float = 0.05
) -> Dict[str, Dict[str, Union[float, bool]]]:
    """
    Perform statistical significance tests for model comparisons.
    
    Args:
        y_true: Actual values
        predictions: Dictionary of model predictions {model_name: predictions}
        baseline_model: Name of the baseline model for comparison
        alpha: Significance level
        
    Returns:
        Dictionary with test results for each model comparison
    """
    if baseline_model not in predictions:
        raise ValueError(f"Baseline model '{baseline_model}' not found in predictions")
    
    baseline_pred = predictions[baseline_model]
    results = {}
    
    for model_name, y_pred in predictions.items():
        if model_name == baseline_model:
            continue
        
        # Perform Diebold-Mariano test
        dm_stat, p_value = calculate_diebold_mariano_test(y_true, baseline_pred, y_pred)
        
        # Store results
        results[model_name] = {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'significant_improvement': bool(p_value < alpha and dm_stat > 0),
            'significant_degradation': bool(p_value < alpha and dm_stat < 0)
        }
    
    return results


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str,
    model_name: str = "model",
    include_timestamp: bool = True
) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary with evaluation results
        output_path: Directory to save the results
        model_name: Name of the model for the filename
        include_timestamp: Whether to include timestamp in the filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create filename
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_evaluation_{timestamp}.json"
    else:
        filename = f"{model_name}_evaluation.json"
    
    filepath = os.path.join(output_path, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved evaluation results to {filepath}")


def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary with evaluation results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded evaluation results from {filepath}")
    return results


def plot_horizon_performance(
    horizon_metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance metrics across forecast horizons.
    
    Args:
        horizon_metrics: Dictionary with metrics for each horizon
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get horizons
    horizons = horizon_metrics['horizon']
    
    # Plot RMSE
    ax1.plot(horizons, horizon_metrics['rmse'], 'b-o', linewidth=2)
    ax1.set_title('RMSE by Forecast Horizon')
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    
    # Plot MAPE
    ax2.plot(horizons, horizon_metrics['mape'], 'r-o', linewidth=2)
    ax2.set_title('MAPE by Forecast Horizon')
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('MAPE (%)')
    ax2.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved horizon performance plot to {save_path}")
    
    return fig


def plot_model_comparison_metrics(
    comparison_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['rmse', 'mae', 'directional_accuracy'],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models based on evaluation metrics.
    
    Args:
        comparison_results: Dictionary with metrics for each model
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Get number of models and metrics
    n_models = len(comparison_results)
    n_metrics = len(metrics)
    
    # Create figure
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Convert to list if only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Get model names and colors
    model_names = list(comparison_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Get metric values for each model
        values = [comparison_results[model][metric] for model in model_names]
        
        # Create bar chart
        bars = axes[i].bar(
            np.arange(n_models),
            values,
            color=colors,
            alpha=0.7
        )
        
        # Add metric name as title
        axes[i].set_title(metric.upper())
        
        # Add model names as x-tick labels
        axes[i].set_xticks(np.arange(n_models))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add grid
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                rotation=0
            )
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import OilDataProcessor
    
    # Create output directory for results
    os.makedirs('models/evaluation', exist_ok=True)
    
    # Load data
    processor = OilDataProcessor()
    try:
        # Prepare dataset
        dataset = processor.prepare_dataset(oil_type="brent", freq="daily")
        
        # Get test data
        y_test = dataset['y_test']
        
        # Create dummy predictions for demonstration
        np.random.seed(42)
        
        # Naive model (just use the previous value)
        y_naive = y_test.copy()
        
        # Simple model (add small random noise)
        y_simple = y_test + np.random.normal(0, 0.01, size=y_test.shape)
        
        # Complex model (add larger random noise)
        y_complex = y_test + np.random.normal(0, 0.05, size=y_test.shape)
        
        # Ensemble (average of all models)
        y_ensemble = (y_naive + y_simple + y_complex) / 3
        
        # Calculate metrics for individual models
        predictions = {
            'Naive': y_naive,
            'Simple': y_simple,
            'Complex': y_complex
        }
        
        # Compare models
        comparison = evaluate_ensemble(
            y_test,
            predictions,
            y_ensemble
        )
        
        print("Model Comparison:")
        for model, metrics in comparison.items():
            print(f"  {model}:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    MAPE: {metrics['mape']:.2f}%")
            print(f"    Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        # Calculate horizon-specific performance
        horizon_metrics = evaluate_horizon_performance(y_test, y_complex)
        
        # Perform statistical significance tests
        significance = statistical_significance_tests(y_test, predictions, 'Naive')
        
        # Save evaluation results
        results = {
            'comparison': comparison,
            'horizon_metrics': horizon_metrics,
            'significance_tests': significance
        }
        
        save_evaluation_results(results, 'models/evaluation', 'example')
        
        print("\nEvaluation results saved to models/evaluation/")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")