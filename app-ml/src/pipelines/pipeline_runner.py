from typing import Dict, Any, Tuple
from pathlib import Path
from catboost import CatBoostRegressor

from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.postprocessing import PostprocessingPipeline
from common.data_manager import DataManager


class PipelineRunner:
    """
    Orchestrates the complete training pipeline for energy efficiency forecasting.
    Runs preprocessing, feature engineering, training, and postprocessing in sequence.
    """
    
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initialize pipeline runner with configuration and data manager.
        
        Args:
            config: Configuration dictionary
            data_manager: DataManager instance for data operations
        """
        self.config = config
        self.data_manager = data_manager
        
        self.preprocessing_pipeline = PreprocessingPipeline(config)
        self.feature_engineering_pipeline = FeatureEngineeringPipeline(config)
        self.training_pipeline = TrainingPipeline(config)
        self.postprocessing_pipeline = PostprocessingPipeline(config)
        
        self.trained_model = None
        self.metrics = None
    
    def run_training(self) -> Tuple[CatBoostRegressor, Dict[str, Any]]:
        """
        Run the complete training pipeline from raw data to trained model.
        
        Returns:
            Tuple of (trained_model, metrics)
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE TRAINING PIPELINE")
        print("="*80)
        
        print("\nStep 1: Loading raw data...")
        raw_data = self.data_manager.load_raw_data(format='csv')
        print(f"Loaded {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
        
        print("\nStep 2: Running preprocessing pipeline...")
        preprocessed_data = self.preprocessing_pipeline.run(raw_data)
        
        print("\nStep 3: Running feature engineering pipeline...")
        engineered_data = self.feature_engineering_pipeline.run(preprocessed_data)
        
        print("\nStep 4: Running training pipeline...")
        self.trained_model, self.metrics = self.training_pipeline.run(engineered_data)
        
        print("\nStep 5: Running postprocessing pipeline...")
        feature_names = self.training_pipeline.feature_names
        self.postprocessing_pipeline.run(self.trained_model, self.metrics, feature_names)
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        self._print_summary()
        
        return self.trained_model, self.metrics
    
    def _print_summary(self) -> None:
        """
        Print summary of training pipeline execution.
        """
        print("\nPIPELINE EXECUTION SUMMARY:")
        print("-"*80)
        
        if self.metrics:
            print("\nModel Performance:")
            print(f"  Training R2: {self.metrics['train']['R2']:.6f}")
            print(f"  Test R2: {self.metrics['test']['R2']:.6f}")
            print(f"  Training RMSE: {self.metrics['train']['RMSE']:.4f}")
            print(f"  Test RMSE: {self.metrics['test']['RMSE']:.4f}")
        
        if self.trained_model:
            model_path = Path(self.config['postprocessing']['model_save_path'])
            print(f"\nModel saved to: {model_path}")
            
        print("-"*80)
    
    def get_model(self) -> CatBoostRegressor:
        """
        Get the trained model.
        
        Returns:
            Trained CatBoost model
        """
        if self.trained_model is None:
            raise ValueError("Model has not been trained yet. Run run_training() first.")
        
        return self.trained_model
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the training metrics.
        
        Returns:
            Dictionary containing training and test metrics
        """
        if self.metrics is None:
            raise ValueError("Metrics not available. Run run_training() first.")
        
        return self.metrics
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.training_pipeline.model is None:
            raise ValueError("Model has not been trained yet. Run run_training() first.")
        
        importance_df = self.training_pipeline.get_feature_importance(top_n=top_n)

        if importance_df.empty:
            return {}
            
        return dict(zip(importance_df['feature'], importance_df['importance']))