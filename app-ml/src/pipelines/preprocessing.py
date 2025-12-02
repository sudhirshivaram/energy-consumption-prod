import pandas as pd
from typing import Dict, Any, List, Optional


class PreprocessingPipeline:
    """
    Preprocessing pipeline for energy efficiency data.
    Handles column renaming, dropping unnecessary columns, and index reset.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessing pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.column_mapping = self.preprocessing_config.get('column_mapping', {})
        self.columns_to_drop = self.preprocessing_config.get('columns_to_drop', [])
        self.reset_index = self.preprocessing_config.get('reset_index', True)
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns based on column mapping configuration.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with renamed columns
        """
        if not self.column_mapping:
            print("No column mapping found, skipping column renaming")
            return df
        
        df_renamed = df.rename(columns=self.column_mapping)
        
        renamed_cols = [col for col in df.columns if col in self.column_mapping]
        print(f"Renamed {len(renamed_cols)} columns: {renamed_cols}")
        
        return df_renamed
    
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unnecessary columns based on configuration.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with specified columns dropped
        """
        if not self.columns_to_drop:
            print("No columns to drop")
            return df
        
        columns_to_drop_existing = [col for col in self.columns_to_drop if col in df.columns]
        
        if not columns_to_drop_existing:
            print(f"Columns to drop {self.columns_to_drop} not found in DataFrame")
            return df
        
        df_dropped = df.drop(columns=columns_to_drop_existing)
        
        print(f"Dropped {len(columns_to_drop_existing)} columns: {columns_to_drop_existing}")
        print(f"Remaining columns: {list(df_dropped.columns)}")
        
        return df_dropped
    
    def reset_dataframe_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reset DataFrame index.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with reset index
        """
        if not self.reset_index:
            print("Index reset disabled")
            return df
        
        df_reset = df.reset_index(drop=True)
        print("DataFrame index reset")
        
        return df_reset
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Input raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        print(f"\nInput shape: {df.shape}")
        print(f"Input columns: {list(df.columns)}")
        
        df_processed = df.copy()
        
        df_processed = self.rename_columns(df_processed)
        
        df_processed = self.drop_columns(df_processed)
        
        df_processed = self.reset_dataframe_index(df_processed)
        
        print(f"\nOutput shape: {df_processed.shape}")
        print(f"Output columns: {list(df_processed.columns)}")
        print("\nPreprocessing pipeline completed successfully")
        print("="*60)
        
        return df_processed
    
    def get_column_mapping(self) -> Dict[str, str]:
        """
        Get the column mapping configuration.
        
        Returns:
            Dictionary of column mappings
        """
        return self.column_mapping
    
    def get_columns_to_drop(self) -> List[str]:
        """
        Get the list of columns to drop.
        
        Returns:
            List of column names to drop
        """
        return self.columns_to_drop
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame before preprocessing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")
        
        if df.shape[0] == 0:
            raise ValueError("Input DataFrame has no rows")
        
        if df.shape[1] == 0:
            raise ValueError("Input DataFrame has no columns")
        
        print(f"Input validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
        return True