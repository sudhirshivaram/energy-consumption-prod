import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
from catboost import CatBoostRegressor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager
from common.utils import read_config


@pytest.fixture
def integration_config(tmp_path):
    """Create configuration using actual project paths but temporary output paths"""
    return {
        'preprocessing': {
            'column_mapping': {
                'X1': 'relative_compactness',
                'X2': 'surface_area',
                'X3': 'wall_area',
                'X4': 'roof_area',
                'X5': 'overall_height',
                'X6': 'orientation',
                'X7': 'glazing_area',
                'X8': 'glazing_area_distribution',
                'Y1': 'heating_load',
                'Y2': 'cooling_load'
            },
            'columns_to_drop': ['cooling_load'],
            'reset_index': True
        },
        'feature_engineering': {
            'lag_features': {
                'heating_load': [1, 2, 3, 5, 10],
                'relative_compactness': [1, 2],
                'surface_area': [1, 2],
                'overall_height': [1, 2]
            },
            'fill_method': 'bfill'
        },
        'training': {
            'target_column': 'heating_load',
            'target_shift': 1,
            'test_size': 0.2,
            'random_state': 42,
            'shuffle': False,
            'model': {
                'type': 'CatBoostRegressor',
                'params': {
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'random_seed': 42,
                    'verbose': 0
                },
                'early_stopping_rounds': 20
            },
            'hyperparameter_tuning': {
                'enabled': False
            }
        },
        'postprocessing': {
            'model_save_path': str(tmp_path / 'models/prod/energy_forecast_model.cbm'),
            'prediction_columns': ['timestamp', 'predicted_heating_load'],
            'time_increment_minutes': 60
        },
        'data': {
            'raw_data': {
                'csv_path': str(project_root / 'data/raw_data/csv/energy-efficiency-data.csv'),
                'parquet_path': str(tmp_path / 'data/raw_data/parquet/energy-efficiency-data.parquet')
            },
            'prod_data': {
                'csv_path': str(tmp_path / 'data/prod_data/csv/predictions.csv'),
                'parquet_path': str(tmp_path / 'data/prod_data/parquet/predictions.parquet')
            }
        }
    }


@pytest.fixture
def check_actual_data_exists():
    """Check if actual energy efficiency data exists"""
    data_path = project_root / 'data/raw_data/csv/energy-efficiency-data.csv'
    
    if not data_path.exists():
        pytest.skip(f"Actual data file not found: {data_path}")
    
    return data_path


class TestEndToEndTrainingWithActualData:
    """Integration tests using actual energy efficiency dataset"""
    
    def test_complete_training_pipeline_with_actual_data(self, integration_config, check_actual_data_exists):
        """Test complete training pipeline with actual energy efficiency data"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        
        model, metrics = runner.run_training()
        
        assert isinstance(model, CatBoostRegressor)
        assert model.is_fitted()
        
        assert 'train' in metrics
        assert 'test' in metrics
        
        assert metrics['train']['R2'] > 0.7
        assert metrics['test']['R2'] > 0.7
        
        print(f"\nTraining R2: {metrics['train']['R2']:.6f}")
        print(f"Test R2: {metrics['test']['R2']:.6f}")
        print(f"Training RMSE: {metrics['train']['RMSE']:.4f}")
        print(f"Test RMSE: {metrics['test']['RMSE']:.4f}")
    
    def test_model_saves_correctly_with_actual_data(self, integration_config, check_actual_data_exists):
        """Test that model and metadata files are saved correctly"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        runner.run_training()
        
        model_path = Path(integration_config['postprocessing']['model_save_path'])
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.txt"
        
        assert model_path.exists()
        assert metadata_path.exists()
        
        assert model_path.stat().st_size > 0
        assert metadata_path.stat().st_size > 0
    
    def test_trained_model_can_make_predictions(self, integration_config, check_actual_data_exists):
        """Test that trained model can make predictions on actual data"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        model, metrics = runner.run_training()
        
        X_test = runner.training_pipeline.X_test
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p > 0 for p in predictions)
        assert all(p < 100 for p in predictions)
    
    def test_feature_importance_with_actual_data(self, integration_config, check_actual_data_exists):
        """Test feature importance extraction with actual data"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        runner.run_training()
        
        importance = runner.get_feature_importance(top_n=10)
        
        assert len(importance) == 10
        assert all(score >= 0 for score in importance.values())
        
        print("\nTop 10 Feature Importances:")
        for feature, score in importance.items():
            print(f"  {feature}: {score:.4f}")
    
    def test_actual_data_dimensions(self, integration_config, check_actual_data_exists):
        """Test that actual data has expected dimensions"""
        data_manager = DataManager(integration_config)
        raw_data = data_manager.load_raw_data(format='csv')
        
        assert raw_data.shape[0] == 768
        assert raw_data.shape[1] == 10
        
        expected_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']
        assert list(raw_data.columns) == expected_columns
    
    def test_preprocessing_with_actual_data(self, integration_config, check_actual_data_exists):
        """Test preprocessing pipeline with actual data"""
        data_manager = DataManager(integration_config)
        raw_data = data_manager.load_raw_data(format='csv')
        
        runner = PipelineRunner(integration_config, data_manager)
        preprocessed_data = runner.preprocessing_pipeline.run(raw_data)
        
        assert preprocessed_data.shape[0] == 768
        assert 'heating_load' in preprocessed_data.columns
        assert 'cooling_load' not in preprocessed_data.columns
        assert 'relative_compactness' in preprocessed_data.columns
    
    def test_feature_engineering_with_actual_data(self, integration_config, check_actual_data_exists):
        """Test feature engineering pipeline with actual data"""
        data_manager = DataManager(integration_config)
        raw_data = data_manager.load_raw_data(format='csv')
        
        runner = PipelineRunner(integration_config, data_manager)
        preprocessed_data = runner.preprocessing_pipeline.run(raw_data)
        engineered_data = runner.feature_engineering_pipeline.run(preprocessed_data)
        
        lag_columns = [col for col in engineered_data.columns if '_lag_' in col]
        
        assert len(lag_columns) == 11
        assert 'heating_load_lag_1' in engineered_data.columns
        assert 'heating_load_lag_10' in engineered_data.columns
        
        assert engineered_data[lag_columns].isnull().sum().sum() == 0
    
    def test_model_performance_metrics_reasonable(self, integration_config, check_actual_data_exists):
        """Test that model performance metrics are in reasonable ranges"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        model, metrics = runner.run_training()
        
        assert 0 <= metrics['train']['R2'] <= 1
        assert 0 <= metrics['test']['R2'] <= 1
        
        assert metrics['train']['RMSE'] > 0
        assert metrics['test']['RMSE'] > 0
        
        assert metrics['train']['MAE'] > 0
        assert metrics['test']['MAE'] > 0
        
        assert 0 <= metrics['train']['MAPE'] <= 100
        assert 0 <= metrics['test']['MAPE'] <= 100
    
    def test_model_overfitting_check(self, integration_config, check_actual_data_exists):
        """Test that model is not significantly overfitting"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        model, metrics = runner.run_training()
        
        train_r2 = metrics['train']['R2']
        test_r2 = metrics['test']['R2']
        
        r2_difference = train_r2 - test_r2
        
        assert r2_difference < 0.15
        
        print(f"\nR2 Difference (Train - Test): {r2_difference:.6f}")
    
    def test_saved_model_can_be_reloaded(self, integration_config, check_actual_data_exists):
        """Test that saved model can be reloaded and used"""
        data_manager = DataManager(integration_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(integration_config, data_manager)
        original_model, metrics = runner.run_training()
        
        X_test = runner.training_pipeline.X_test
        original_predictions = original_model.predict(X_test)
        
        reloaded_model = runner.postprocessing_pipeline.load_model()
        reloaded_predictions = reloaded_model.predict(X_test)
        
        np.testing.assert_array_almost_equal(original_predictions, reloaded_predictions, decimal=5)


class TestDataQualityChecks:
    """Integration tests for data quality checks"""
    
    def test_no_missing_values_in_actual_data(self, integration_config, check_actual_data_exists):
        """Test that actual data has no missing values"""
        data_manager = DataManager(integration_config)
        raw_data = data_manager.load_raw_data(format='csv')
        
        assert raw_data.isnull().sum().sum() == 0
    
    def test_target_variable_range(self, integration_config, check_actual_data_exists):
        """Test that target variable is in expected range"""
        data_manager = DataManager(integration_config)
        raw_data = data_manager.load_raw_data(format='csv')
        
        assert raw_data['Y1'].min() > 0
        assert raw_data['Y1'].max() < 100
        
        print(f"\nHeating Load Range: [{raw_data['Y1'].min():.2f}, {raw_data['Y1'].max():.2f}]")
    
    def test_feature_distributions(self, integration_config, check_actual_data_exists):
        """Test that features have reasonable distributions"""
        data_manager = DataManager(integration_config)
        raw_data = data_manager.load_raw_data(format='csv')
        
        assert raw_data['X1'].between(0, 1).all()
        
        assert raw_data['X6'].isin([2, 3, 4, 5]).all()
        
        assert raw_data['X8'].isin([0, 1, 2, 3, 4, 5]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])