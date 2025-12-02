import pandas as pd
import numpy as np
from typing import Dict, Any, List


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline for energy efficiency data.
    Creates lag features from specified columns to capture temporal patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineering pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing feature engineering parameters
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        self.lag_features_config = self.feature_config.get('lag_features', {})
        self.fill_method = self.feature_config.get('fill_method', 'bfill')
    
    def create_lag_features(self, df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for a specific column.
        
        Args:
            df: Input DataFrame
            column: Column name to create lag features from
            lags: List of lag periods (e.g., [1, 2, 3])
            
        Returns:
            DataFrame with added lag features
        """
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame, skipping")
            return df
        
        df_with_lags = df.copy()
        
        for lag in lags:
            lag_column_name = f"{column}_lag_{lag}"
            df_with_lags[lag_column_name] = df_with_lags[column].shift(lag)
            print(f"Created lag feature: {lag_column_name}")
        
        return df_with_lags
    
    def create_all_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all lag features based on configuration.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all lag features added
        """
        if not self.lag_features_config:
            print("No lag features configuration found")
            return df
        
        df_with_all_lags = df.copy()
        
        for column, lags in self.lag_features_config.items():
            if not lags:
                continue
            
            print(f"\nCreating lag features for column: {column}")
            print(f"Lag periods: {lags}")
            
            df_with_all_lags = self.create_lag_features(df_with_all_lags, column, lags)
        
        return df_with_all_lags
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values created by lag features.
        
        Args:
            df: Input DataFrame with lag features
            
        Returns:
            DataFrame with filled missing values
        """
        lag_columns = [col for col in df.columns if '_lag_' in col]
        
        if not lag_columns:
            print("No lag columns found, skipping fill")
            return df
        
        df_filled = df.copy()
        
        initial_nulls = df_filled[lag_columns].isnull().sum().sum()
        
        if self.fill_method == 'bfill':
            df_filled[lag_columns] = df_filled[lag_columns].bfill()
            print(f"Filled {initial_nulls} missing values using backward fill (bfill)")
        elif self.fill_method == 'ffill':
            df_filled[lag_columns] = df_filled[lag_columns].ffill()
            remaining_nulls_after_ffill = df_filled[lag_columns].isnull().sum().sum()
            if remaining_nulls_after_ffill > 0:
                df_filled[lag_columns] = df_filled[lag_columns].bfill()
            print(f"Filled {initial_nulls} missing values using forward fill (ffill) with bfill fallback")
        elif self.fill_method == 'mean':
            for col in lag_columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            print(f"Filled {initial_nulls} missing values using mean")
        elif self.fill_method == 'median':
            for col in lag_columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            print(f"Filled {initial_nulls} missing values using median")
        elif self.fill_method == 'zero':
            df_filled[lag_columns] = df_filled[lag_columns].fillna(0)
            print(f"Filled {initial_nulls} missing values with zeros")
        else:
            print(f"Unknown fill method: {self.fill_method}, skipping fill")
        
        remaining_nulls = df_filled[lag_columns].isnull().sum().sum()
        print(f"Remaining missing values: {remaining_nulls}")
        
        return df_filled
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Input preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        print(f"\nInput shape: {df.shape}")
        print(f"Input columns: {list(df.columns)}")
        
        df_engineered = df.copy()
        
        df_engineered = self.create_all_lag_features(df_engineered)
        
        df_engineered = self.fill_missing_values(df_engineered)
        
        lag_columns = [col for col in df_engineered.columns if '_lag_' in col]
        print(f"\nCreated {len(lag_columns)} lag features: {lag_columns}")
        
        print(f"\nOutput shape: {df_engineered.shape}")
        print(f"Output columns: {list(df_engineered.columns)}")
        print("\nFeature engineering pipeline completed successfully")
        print("="*60)
        
        return df_engineered
    
    def get_lag_features_config(self) -> Dict[str, List[int]]:
        """
        Get the lag features configuration.
        
        Returns:
            Dictionary mapping column names to lag periods
        """
        return self.lag_features_config
    
    def get_lag_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of lag feature column names.
        
        Args:
            df: DataFrame to extract lag feature names from
            
        Returns:
            List of lag feature column names
        """
        return [col for col in df.columns if '_lag_' in col]
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame before feature engineering.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")
        
        if df.shape[0] == 0:
            raise ValueError("Input DataFrame has no rows")
        
        missing_columns = []
        for column in self.lag_features_config.keys():
            if column not in df.columns:
                missing_columns.append(column)
        
        if missing_columns:
            print(f"Warning: Columns for lag features not found: {missing_columns}")
        
        print(f"Input validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
        return True
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about engineered features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary containing feature statistics
        """
        lag_columns = self.get_lag_feature_names(df)
        
        stats = {
            'total_features': df.shape[1],
            'original_features': df.shape[1] - len(lag_columns),
            'lag_features': len(lag_columns),
            'lag_feature_names': lag_columns,
            'missing_values': df[lag_columns].isnull().sum().sum() if lag_columns else 0
        }
        
        return stats