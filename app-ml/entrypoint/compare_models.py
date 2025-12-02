"""
Model Comparison Script for Energy Efficiency Forecasting

This script trains and compares different regression models on the energy efficiency dataset:
- CatBoost
- XGBoost
- LightGBM
- RandomForest
- GradientBoosting

It outputs a comparison table with performance metrics for each model.

Usage:
    python app-ml/entrypoint/compare_models.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)

from common.utils import read_config
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline


def prepare_data(config):
    """
    Prepare data by running preprocessing and feature engineering.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Preprocessed and feature-engineered DataFrame
    """
    print("\nPreparing data...")
    
    data_manager = DataManager(config)
    data_manager.initialize_prod_database()
    raw_data = data_manager.load_raw_data(format='csv')
    
    preprocessing_pipeline = PreprocessingPipeline(config)
    preprocessed_data = preprocessing_pipeline.run(raw_data)
    
    feature_engineering_pipeline = FeatureEngineeringPipeline(config)
    engineered_data = feature_engineering_pipeline.run(preprocessed_data)
    
    return engineered_data


def train_and_evaluate_model(model_type, config, engineered_data):
    """
    Train and evaluate a single model type.
    
    Args:
        model_type: Type of model to train
        config: Configuration dictionary
        engineered_data: Preprocessed and engineered DataFrame
        
    Returns:
        Dictionary containing model name, metrics, and training time
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type}")
    print('='*80)
    
    model_config = config.copy()
    model_config['training']['model']['type'] = model_type
    
    training_pipeline = TrainingPipeline(model_config)
    
    start_time = time.time()
    
    try:
        model, metrics = training_pipeline.run(engineered_data)
        training_time = time.time() - start_time
        
        result = {
            'Model': model_type,
            'Train_R2': metrics['train']['R2'],
            'Test_R2': metrics['test']['R2'],
            'Train_RMSE': metrics['train']['RMSE'],
            'Test_RMSE': metrics['test']['RMSE'],
            'Train_MAE': metrics['train']['MAE'],
            'Test_MAE': metrics['test']['MAE'],
            'Test_MAPE': metrics['test']['MAPE'],
            'Training_Time_sec': training_time,
            'Status': 'Success'
        }
        
        print(f"\n{model_type} Training Completed:")
        print(f"  Test R2: {metrics['test']['R2']:.6f}")
        print(f"  Test RMSE: {metrics['test']['RMSE']:.4f}")
        print(f"  Training Time: {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n{model_type} Training Failed: {str(e)}")
        result = {
            'Model': model_type,
            'Train_R2': None,
            'Test_R2': None,
            'Train_RMSE': None,
            'Test_RMSE': None,
            'Train_MAE': None,
            'Test_MAE': None,
            'Test_MAPE': None,
            'Training_Time_sec': None,
            'Status': f'Failed: {str(e)}'
        }
    
    return result


def main():
    """
    Main function to compare different models.
    """
    print("\n" + "="*80)
    print("ENERGY EFFICIENCY FORECASTING - MODEL COMPARISON")
    print("="*80)
    
    config_path = project_root / 'config' / 'config.yaml'
    print(f"\nLoading configuration from: {config_path}")
    config = read_config(config_path)
    
    engineered_data = prepare_data(config)
    
    models_to_compare = [
        'CatBoostRegressor',
        'XGBRegressor',
        'RandomForestRegressor',
        'GradientBoostingRegressor'
    ]
    
    results = []
    
    for model_type in models_to_compare:
        result = train_and_evaluate_model(model_type, config, engineered_data)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print("\n" + results_df.to_string(index=False))
    
    successful_models = results_df[results_df['Status'] == 'Success']
    
    if len(successful_models) > 0:
        print("\n" + "="*80)
        print("RANKINGS")
        print("="*80)
        
        best_r2 = successful_models.loc[successful_models['Test_R2'].idxmax()]
        print(f"\nBest Test R2: {best_r2['Model']} ({best_r2['Test_R2']:.6f})")
        
        best_rmse = successful_models.loc[successful_models['Test_RMSE'].idxmin()]
        print(f"Best Test RMSE: {best_rmse['Model']} ({best_rmse['Test_RMSE']:.4f})")
        
        fastest = successful_models.loc[successful_models['Training_Time_sec'].idxmin()]
        print(f"Fastest Training: {fastest['Model']} ({fastest['Training_Time_sec']:.2f} sec)")
        
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        
        if best_r2['Model'] == best_rmse['Model']:
            print(f"\nRecommended Model: {best_r2['Model']}")
            print(f"  - Best performance on both R2 and RMSE")
            print(f"  - Test R2: {best_r2['Test_R2']:.6f}")
            print(f"  - Test RMSE: {best_r2['Test_RMSE']:.4f}")
        else:
            print(f"\nTop Candidates:")
            print(f"  1. {best_r2['Model']} - Best R2 ({best_r2['Test_R2']:.6f})")
            print(f"  2. {best_rmse['Model']} - Best RMSE ({best_rmse['Test_RMSE']:.4f})")
    
    output_path = project_root / 'models' / 'experiments' / 'model_comparison_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*80 + "\n")
    
    return results_df


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Model comparison failed")
        print("="*80)
        print(f"\nError details: {str(e)}")
        print("\n" + "="*80 + "\n")
        sys.exit(1)