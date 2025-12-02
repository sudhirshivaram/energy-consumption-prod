import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Union


def read_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read YAML configuration file.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path where to save the config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to: {config_path}")


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object of the directory
    """
    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def load_data(file_path: Union[str, Path], file_format: str = 'csv') -> pd.DataFrame:
    """
    Load data from CSV or Parquet file.
    
    Args:
        file_path: Path to the data file
        file_format: Format of the file ('csv' or 'parquet')
        
    Returns:
        DataFrame containing the data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Loaded data from {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_data(df: pd.DataFrame, file_path: Union[str, Path], file_format: str = 'csv') -> None:
    """
    Save DataFrame to CSV or Parquet file.
    
    Args:
        df: DataFrame to save
        file_path: Path where to save the file
        file_format: Format of the file ('csv' or 'parquet')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == 'csv':
        df.to_csv(file_path, index=False)
    elif file_format == 'parquet':
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Data saved to {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing metric names and values
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset (e.g., "Training", "Test")
    """
    print(f"\n{dataset_name} Performance Metrics:")
    print("=" * 60)
    for metric_name, metric_value in metrics.items():
        if metric_name == 'R2':
            print(f"{metric_name}: {metric_value:.6f}")
        elif metric_name == 'MAPE':
            print(f"{metric_name}: {metric_value:.4f}%")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
