import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class DataManager:
    """
    Manages data loading, saving, and database operations for the energy forecast project.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataManager with configuration.
        
        Args:
            config: Configuration dictionary containing data paths
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.raw_data_config = self.data_config.get('raw_data', {})
        self.prod_data_config = self.data_config.get('prod_data', {})
        
        self.raw_csv_path = Path(self.raw_data_config.get('csv_path', 'data/raw_data/csv/energy-efficiency-data.csv'))
        self.raw_parquet_path = Path(self.raw_data_config.get('parquet_path', 'data/raw_data/parquet/energy-efficiency-data.parquet'))
        self.prod_csv_path = Path(self.prod_data_config.get('csv_path', 'data/prod_data/csv/predictions.csv'))
        self.prod_parquet_path = Path(self.prod_data_config.get('parquet_path', 'data/prod_data/parquet/predictions.parquet'))
    
    def initialize_prod_database(self) -> None:
        """
        Initialize production database by loading historical raw data.
        Creates parquet version of CSV data if it doesn't exist.
        """
        print("\nInitializing production database...")
        print("=" * 60)
        
        if self.raw_csv_path.exists():
            print(f"Loading raw data from: {self.raw_csv_path}")
            df = pd.read_csv(self.raw_csv_path)
            print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            
            self.raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.raw_parquet_path, index=False)
            print(f"Saved parquet version to: {self.raw_parquet_path}")
            
        elif self.raw_parquet_path.exists():
            print(f"Loading raw data from: {self.raw_parquet_path}")
            df = pd.read_parquet(self.raw_parquet_path)
            print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            raise FileNotFoundError(f"Raw data not found at {self.raw_csv_path} or {self.raw_parquet_path}")
        
        print("Production database initialized successfully")
    
    def load_raw_data(self, format: str = 'csv') -> pd.DataFrame:
        """
        Load raw training data.
        
        Args:
            format: Data format ('csv' or 'parquet')
            
        Returns:
            DataFrame containing raw data
        """
        if format == 'csv':
            if not self.raw_csv_path.exists():
                raise FileNotFoundError(f"Raw CSV data not found: {self.raw_csv_path}")
            df = pd.read_csv(self.raw_csv_path)
        elif format == 'parquet':
            if not self.raw_parquet_path.exists():
                raise FileNotFoundError(f"Raw Parquet data not found: {self.raw_parquet_path}")
            df = pd.read_parquet(self.raw_parquet_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"\nLoaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def save_predictions(self, df: pd.DataFrame, format: str = 'csv') -> None:
        """
        Save prediction results to production database.
        
        Args:
            df: DataFrame containing predictions
            format: Output format ('csv' or 'parquet')
        """
        if format == 'csv':
            self.prod_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.prod_csv_path, index=False)
            print(f"\nPredictions saved to: {self.prod_csv_path}")
        elif format == 'parquet':
            self.prod_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.prod_parquet_path, index=False)
            print(f"\nPredictions saved to: {self.prod_parquet_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_predictions(self, format: str = 'csv') -> Optional[pd.DataFrame]:
        """
        Load existing predictions from production database.
        
        Args:
            format: Data format ('csv' or 'parquet')
            
        Returns:
            DataFrame containing predictions, or None if file doesn't exist
        """
        if format == 'csv':
            if not self.prod_csv_path.exists():
                print(f"No existing predictions found at: {self.prod_csv_path}")
                return None
            df = pd.read_csv(self.prod_csv_path)
        elif format == 'parquet':
            if not self.prod_parquet_path.exists():
                print(f"No existing predictions found at: {self.prod_parquet_path}")
                return None
            df = pd.read_parquet(self.prod_parquet_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"\nLoaded predictions: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def append_predictions(self, new_predictions: pd.DataFrame, format: str = 'csv') -> None:
        """
        Append new predictions to existing production database.
        
        Args:
            new_predictions: DataFrame with new predictions
            format: Data format ('csv' or 'parquet')
        """
        existing_predictions = self.load_predictions(format=format)
        
        if existing_predictions is not None:
            combined = pd.concat([existing_predictions, new_predictions], ignore_index=True)
            print(f"Appending {new_predictions.shape[0]} new predictions to existing {existing_predictions.shape[0]} rows")
        else:
            combined = new_predictions
            print(f"Creating new predictions database with {new_predictions.shape[0]} rows")
        
        self.save_predictions(combined, format=format)
    
    def get_latest_prediction(self, format: str = 'csv') -> Optional[pd.Series]:
        """
        Get the most recent prediction from the database.
        
        Args:
            format: Data format ('csv' or 'parquet')
            
        Returns:
            Series containing the latest prediction, or None if no predictions exist
        """
        predictions = self.load_predictions(format=format)
        
        if predictions is None or predictions.empty:
            return None
        
        if 'timestamp' in predictions.columns:
            predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
            latest = predictions.loc[predictions['timestamp'].idxmax()]
        else:
            latest = predictions.iloc[-1]
        
        return latest
    
    def clear_predictions(self, format: str = 'csv') -> None:
        """
        Clear all predictions from the production database.
        
        Args:
            format: Data format ('csv' or 'parquet')
        """
        if format == 'csv':
            if self.prod_csv_path.exists():
                self.prod_csv_path.unlink()
                print(f"Cleared predictions from: {self.prod_csv_path}")
        elif format == 'parquet':
            if self.prod_parquet_path.exists():
                self.prod_parquet_path.unlink()
                print(f"Cleared predictions from: {self.prod_parquet_path}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of raw and prediction data.
        
        Returns:
            Dictionary containing data summary
        """
        summary = {
            'raw_data': {},
            'predictions': {}
        }
        
        try:
            raw_df = self.load_raw_data(format='csv')
            summary['raw_data'] = {
                'rows': raw_df.shape[0],
                'columns': raw_df.shape[1],
                'file_path': str(self.raw_csv_path)
            }
        except Exception as e:
            summary['raw_data']['error'] = str(e)
        
        try:
            pred_df = self.load_predictions(format='csv')
            if pred_df is not None:
                summary['predictions'] = {
                    'rows': pred_df.shape[0],
                    'columns': pred_df.shape[1],
                    'file_path': str(self.prod_csv_path)
                }
            else:
                summary['predictions'] = {'status': 'No predictions available'}
        except Exception as e:
            summary['predictions']['error'] = str(e)
        
        return summary
