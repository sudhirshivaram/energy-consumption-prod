"""
Training Entrypoint for Energy Efficiency Forecasting System

This script serves as the main entry point for training the energy forecast model.
It orchestrates the complete ML pipeline from raw data to a trained, saved model.

Pipeline Steps:
1. Load configuration from config.yaml
2. Initialize production database with raw energy efficiency data
3. Run preprocessing pipeline (column renaming, dropping, index reset)
4. Run feature engineering pipeline (lag feature creation)
5. Run training pipeline (model training with CatBoost)
6. Run postprocessing pipeline (model saving, metadata creation)

Usage:
    python app-ml/entrypoint/train.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)


from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager


def main():
    """
    Main function to execute the complete training pipeline.
    """
    print("\n" + "="*80)
    print("ENERGY EFFICIENCY FORECASTING - MODEL TRAINING")
    print("="*80)
    
    config_path = project_root / 'config' / 'config.yaml'
    print(f"\nLoading configuration from: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = read_config(config_path)
    print("Configuration loaded successfully")
    
    print("\nInitializing data manager...")
    data_manager = DataManager(config)
    data_manager.initialize_prod_database()
    
    print("\nInitializing pipeline runner...")
    pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)
    
    print("\nStarting training pipeline...")
    model, metrics = pipeline_runner.run_training()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\nFinal Model Performance:")
    print(f"  Training R2: {metrics['train']['R2']:.6f}")
    print(f"  Test R2: {metrics['test']['R2']:.6f}")
    print(f"  Training RMSE: {metrics['train']['RMSE']:.4f}")
    print(f"  Test RMSE: {metrics['test']['RMSE']:.4f}")
    print(f"  Test MAE: {metrics['test']['MAE']:.4f}")
    print(f"  Test MAPE: {metrics['test']['MAPE']:.4f}%")
    
    model_path = Path(config['postprocessing']['model_save_path'])
    print(f"\nModel saved to: {model_path}")
    print(f"Model file size: {model_path.stat().st_size / 1024:.2f} KB")
    
    print("\nTop 10 Most Important Features:")
    top_features = pipeline_runner.get_feature_importance(top_n=10)
    if top_features:  # Check if dict is not empty
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    else:
        print("  Feature importance not available for this model type")
            
    print("\n" + "="*80)
    print("Training pipeline execution completed.")
    print("="*80 + "\n")
    
    return model, metrics


if __name__ == "__main__":
    try:
        model, metrics = main()
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Training pipeline failed")
        print("="*80)
        print(f"\nError details: {str(e)}")
        print("\nPlease check:")
        print("  1. Configuration file exists at config/config.yaml")
        print("  2. Raw data file exists at data/raw_data/csv/energy-efficiency-data.csv")
        print("  3. All required directories are created")
        print("  4. All dependencies are installed")
        print("\n" + "="*80 + "\n")
        sys.exit(1)