import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import joblib


class PostprocessingPipeline:
    """
    Postprocessing pipeline for energy efficiency forecasting.
    Handles model persistence and prediction formatting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize postprocessing pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing postprocessing parameters
        """
        self.config = config
        self.postprocessing_config = config.get('postprocessing', {})
        self.model_save_path = self.postprocessing_config.get('model_save_path', 'models/prod/energy_forecast_model.cbm')
        self.prediction_columns = self.postprocessing_config.get('prediction_columns', ['timestamp', 'predicted_heating_load'])
        self.time_increment_minutes = self.postprocessing_config.get('time_increment_minutes', 60)
    
    def save_model(self, model, model_path: Optional[str] = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained CatBoost model
            model_path: Optional custom path to save model (uses config path if not provided)
        """
        if model is None:
            raise ValueError("Model is None, cannot save")
        
        save_path = Path(model_path) if model_path else Path(self.model_save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if model has save_model method (CatBoost)
        if hasattr(model, 'save_model'):
            model.save_model(str(save_path))
        else:
            # Use joblib for sklearn models (LinearRegression, etc.)
            joblib.dump(model, str(save_path))
        
        print(f"\nModel saved successfully to: {save_path}")
        print(f"Model file size: {save_path.stat().st_size / 1024:.2f} KB")
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load trained model from disk.
        
        Args:
            model_path: Optional custom path to load model (uses config path if not provided)
            
        Returns:
            Loaded CatBoost model
        """
        load_path = Path(model_path) if model_path else Path(self.model_save_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Try loading with joblib first (sklearn models)
        try:
            model = joblib.load(str(load_path))
        except:
            # Fall back to CatBoost load method
            model = CatBoostRegressor()
            model.load_model(str(load_path))
        
        print(f"\nModel loaded successfully from: {load_path}")
        
        return model
    
    def format_prediction(self, prediction: float, timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """
        Format a single prediction as a DataFrame.
        
        Args:
            prediction: Predicted value
            timestamp: Optional timestamp for the prediction (uses current time if not provided)
            
        Returns:
            DataFrame with formatted prediction
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        prediction_df = pd.DataFrame({
            self.prediction_columns[0]: [timestamp],
            self.prediction_columns[1]: [prediction]
        })
        
        return prediction_df
    
    def format_predictions_batch(self, predictions: np.ndarray, 
                                 start_timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """
        Format multiple predictions as a DataFrame with timestamps.
        
        Args:
            predictions: Array of predicted values
            start_timestamp: Starting timestamp (uses current time if not provided)
            
        Returns:
            DataFrame with formatted predictions
        """
        if start_timestamp is None:
            start_timestamp = datetime.now()
        
        timestamps = [
            start_timestamp + timedelta(minutes=i * self.time_increment_minutes) 
            for i in range(len(predictions))
        ]
        
        predictions_df = pd.DataFrame({
            self.prediction_columns[0]: timestamps,
            self.prediction_columns[1]: predictions
        })
        
        return predictions_df
    
    def create_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary containing prediction summary statistics
        """
        pred_col = self.prediction_columns[1]
        
        if pred_col not in predictions_df.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in DataFrame")
        
        summary = {
            'count': len(predictions_df),
            'mean': predictions_df[pred_col].mean(),
            'std': predictions_df[pred_col].std(),
            'min': predictions_df[pred_col].min(),
            'max': predictions_df[pred_col].max(),
            'median': predictions_df[pred_col].median(),
            'q25': predictions_df[pred_col].quantile(0.25),
            'q75': predictions_df[pred_col].quantile(0.75)
        }
        
        return summary
    
    def save_training_metadata(self, metrics: Dict[str, Any], 
                               feature_names: list,
                               model_path: Optional[str] = None) -> None:
        """
        Save training metadata alongside the model.
        
        Args:
            metrics: Training and test metrics
            feature_names: List of feature names used in training
            model_path: Optional custom path for metadata (uses model path if not provided)
        """
        save_path = Path(model_path) if model_path else Path(self.model_save_path)
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.txt"
        
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL TRAINING METADATA\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {save_path}\n\n")
            
            f.write("TRAINING METRICS:\n")
            f.write("-"*40 + "\n")
            for metric_name, value in metrics.get('train', {}).items():
                if metric_name == 'R2':
                    f.write(f"{metric_name}: {value:.6f}\n")
                elif metric_name == 'MAPE':
                    f.write(f"{metric_name}: {value:.4f}%\n")
                else:
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            f.write("\nTEST METRICS:\n")
            f.write("-"*40 + "\n")
            for metric_name, value in metrics.get('test', {}).items():
                if metric_name == 'R2':
                    f.write(f"{metric_name}: {value:.6f}\n")
                elif metric_name == 'MAPE':
                    f.write(f"{metric_name}: {value:.4f}%\n")
                else:
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            f.write(f"\nFEATURES ({len(feature_names)}):\n")
            f.write("-"*40 + "\n")
            for i, feature in enumerate(feature_names, 1):
                f.write(f"{i}. {feature}\n")
        
        print(f"Training metadata saved to: {metadata_path}")
    
    def run(self, model, metrics: Dict[str, Any], 
            feature_names: list) -> None:
        """
        Run the complete postprocessing pipeline.
        
        Args:
            model: Trained CatBoost model
            metrics: Training and test metrics
            feature_names: List of feature names used in training
        """
        print("\n" + "="*60)
        print("POSTPROCESSING PIPELINE")
        print("="*60)
        
        self.save_model(model)
        
        self.save_training_metadata(metrics, feature_names)
        
        print("\nPostprocessing pipeline completed successfully")
        print("="*60)
    
    def validate_model(self, model) -> bool:
        """
        Validate model before saving.
        
        Args:
            model: Model to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if model is None:
            raise ValueError("Model is None")        
        
        if hasattr(model, 'is_fitted') and not model.is_fitted():
            raise ValueError("Model has not been trained")
        
        print("Model validation passed")
        return True
    
    def get_model_info(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a saved model.
        
        Args:
            model_path: Optional custom path to model (uses config path if not provided)
            
        Returns:
            Dictionary containing model information
        """
        load_path = Path(model_path) if model_path else Path(self.model_save_path)
        
        if not load_path.exists():
            return {'status': 'Model file not found', 'path': str(load_path)}
        
        info = {
            'path': str(load_path),
            'file_size_kb': load_path.stat().st_size / 1024,
            'modified_date': datetime.fromtimestamp(load_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'exists': True
        }
        
        metadata_path = load_path.parent / f"{load_path.stem}_metadata.txt"
        if metadata_path.exists():
            info['metadata_exists'] = True
            info['metadata_path'] = str(metadata_path)
        else:
            info['metadata_exists'] = False
        
        return info