import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
from catboost import CatBoostRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager


@pytest.fixture
def sample_config(tmp_path):
    """Create sample configuration for pipeline runner"""
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
                'heating_load': [1, 2, 3],
                'relative_compactness': [1, 2],
                'surface_area': [1, 2]
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
                    'iterations': 20,
                    'learning_rate': 0.1,
                    'depth': 3,
                    'l2_leaf_reg': 3,
                    'random_seed': 42,
                    'verbose': 0
                },
                'early_stopping_rounds': 10
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
                'csv_path': str(tmp_path / 'data/raw_data/csv/energy-efficiency-data.csv'),
                'parquet_path': str(tmp_path / 'data/raw_data/parquet/energy-efficiency-data.parquet')
            },
            'prod_data': {
                'csv_path': str(tmp_path / 'data/prod_data/csv/predictions.csv'),
                'parquet_path': str(tmp_path / 'data/prod_data/parquet/predictions.parquet')
            }
        }
    }


@pytest.fixture
def sample_energy_data():
    """Create sample energy efficiency data"""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'X1': np.random.rand(n_samples),
        'X2': np.random.rand(n_samples) * 1000,
        'X3': np.random.rand(n_samples) * 500,
        'X4': np.random.rand(n_samples) * 200,
        'X5': np.random.choice([3.5, 7.0], n_samples),
        'X6': np.random.choice([2, 3, 4, 5], n_samples),
        'X7': np.random.choice([0.0, 0.1, 0.25, 0.4], n_samples),
        'X8': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        'Y1': np.random.rand(n_samples) * 40 + 5,
        'Y2': np.random.rand(n_samples) * 40 + 10
    })


@pytest.fixture
def data_manager_with_data(sample_config, sample_energy_data):
    """Create data manager with sample data"""
    data_manager = DataManager(sample_config)
    
    csv_path = Path(sample_config['data']['raw_data']['csv_path'])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_energy_data.to_csv(csv_path, index=False)
    
    data_manager.initialize_prod_database()
    
    return data_manager


class TestPipelineRunnerInitialization:
    """Test cases for PipelineRunner initialization"""
    
    def test_initialization_with_config(self, sample_config, data_manager_with_data):
        """Test pipeline runner initialization"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        assert runner.config == sample_config
        assert runner.data_manager == data_manager_with_data
        assert runner.preprocessing_pipeline is not None
        assert runner.feature_engineering_pipeline is not None
        assert runner.training_pipeline is not None
        assert runner.postprocessing_pipeline is not None
    
    def test_initialization_pipelines_created(self, sample_config, data_manager_with_data):
        """Test that all pipeline instances are created"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        assert hasattr(runner, 'preprocessing_pipeline')
        assert hasattr(runner, 'feature_engineering_pipeline')
        assert hasattr(runner, 'training_pipeline')
        assert hasattr(runner, 'postprocessing_pipeline')
    
    def test_initialization_model_metrics_none(self, sample_config, data_manager_with_data):
        """Test that model and metrics are None before training"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        assert runner.trained_model is None
        assert runner.metrics is None


class TestRunTraining:
    """Test cases for run_training method"""
    
    def test_run_training_complete_pipeline(self, sample_config, data_manager_with_data):
        """Test running complete training pipeline"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model, metrics = runner.run_training()
        
        assert isinstance(model, CatBoostRegressor)
        assert 'train' in metrics
        assert 'test' in metrics
    
    def test_run_training_stores_model(self, sample_config, data_manager_with_data):
        """Test that trained model is stored"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model, metrics = runner.run_training()
        
        assert runner.trained_model is not None
        assert runner.trained_model == model
    
    def test_run_training_stores_metrics(self, sample_config, data_manager_with_data):
        """Test that metrics are stored"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model, metrics = runner.run_training()
        
        assert runner.metrics is not None
        assert runner.metrics == metrics
    
    def test_run_training_saves_model_file(self, sample_config, data_manager_with_data):
        """Test that model file is saved"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        runner.run_training()
        
        model_path = Path(sample_config['postprocessing']['model_save_path'])
        assert model_path.exists()
    
    def test_run_training_saves_metadata(self, sample_config, data_manager_with_data):
        """Test that metadata file is saved"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        runner.run_training()
        
        model_path = Path(sample_config['postprocessing']['model_save_path'])
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.txt"
        assert metadata_path.exists()
    
    def test_run_training_metrics_structure(self, sample_config, data_manager_with_data):
        """Test that metrics have correct structure"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model, metrics = runner.run_training()
        
        assert 'MSE' in metrics['train']
        assert 'RMSE' in metrics['train']
        assert 'MAE' in metrics['train']
        assert 'R2' in metrics['train']
        assert 'MAPE' in metrics['train']
        
        assert 'MSE' in metrics['test']
        assert 'RMSE' in metrics['test']
        assert 'MAE' in metrics['test']
        assert 'R2' in metrics['test']
        assert 'MAPE' in metrics['test']
    
    def test_run_training_model_can_predict(self, sample_config, data_manager_with_data):
        """Test that trained model can make predictions"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model, metrics = runner.run_training()
        
        n_features = len(runner.training_pipeline.feature_names)
        X_test = np.random.rand(5, n_features)
        X_test_df = pd.DataFrame(X_test, columns=runner.training_pipeline.feature_names)
        predictions = model.predict(X_test_df)
        
        assert len(predictions) == 5


class TestGetModel:
    """Test cases for get_model method"""
    
    def test_get_model_after_training(self, sample_config, data_manager_with_data):
        """Test getting model after training"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        runner.run_training()
        
        model = runner.get_model()
        
        assert isinstance(model, CatBoostRegressor)
    
    def test_get_model_before_training_raises_error(self, sample_config, data_manager_with_data):
        """Test that getting model before training raises error"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            runner.get_model()
    
    def test_get_model_returns_same_model(self, sample_config, data_manager_with_data):
        """Test that get_model returns the same model instance"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        model1, _ = runner.run_training()
        model2 = runner.get_model()
        
        assert model1 is model2


class TestGetMetrics:
    """Test cases for get_metrics method"""
    
    def test_get_metrics_after_training(self, sample_config, data_manager_with_data):
        """Test getting metrics after training"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        runner.run_training()
        
        metrics = runner.get_metrics()
        
        assert 'train' in metrics
        assert 'test' in metrics
    
    def test_get_metrics_before_training_raises_error(self, sample_config, data_manager_with_data):
        """Test that getting metrics before training raises error"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        with pytest.raises(ValueError, match="Metrics not available"):
            runner.get_metrics()
    
    def test_get_metrics_returns_same_metrics(self, sample_config, data_manager_with_data):
        """Test that get_metrics returns the same metrics"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        _, metrics1 = runner.run_training()
        metrics2 = runner.get_metrics()
        
        assert metrics1 == metrics2


class TestGetFeatureImportance:
    """Test cases for get_feature_importance method"""
    
    def test_get_feature_importance_after_training(self, sample_config, data_manager_with_data):
        """Test getting feature importance after training"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        runner.run_training()
        
        importance = runner.get_feature_importance(top_n=5)
        
        assert isinstance(importance, dict)
        assert len(importance) == 5
    
    def test_get_feature_importance_before_training_raises_error(self, sample_config, data_manager_with_data):
        """Test that getting importance before training raises error"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            runner.get_feature_importance()
    
    def test_get_feature_importance_all_features(self, sample_config, data_manager_with_data):
        """Test getting all feature importance"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        runner.run_training()
        
        importance = runner.get_feature_importance(top_n=None)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_get_feature_importance_values(self, sample_config, data_manager_with_data):
        """Test that feature importance values are valid"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        runner.run_training()
        
        importance = runner.get_feature_importance(top_n=5)
        
        for feature, score in importance.items():
            assert isinstance(feature, str)
            assert isinstance(score, (int, float, np.number))
            assert score >= 0


class TestPrintSummary:
    """Test cases for _print_summary method"""
    
    def test_print_summary_with_metrics(self, sample_config, data_manager_with_data, capsys):
        """Test that summary is printed with metrics"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        runner.run_training()
        
        captured = capsys.readouterr()
        
        assert "PIPELINE EXECUTION SUMMARY" in captured.out
        assert "Training R2" in captured.out
        assert "Test R2" in captured.out
        assert "Training RMSE" in captured.out
        assert "Test RMSE" in captured.out


class TestEndToEndIntegration:
    """Test cases for end-to-end pipeline integration"""
    
    def test_complete_pipeline_execution(self, sample_config, data_manager_with_data):
        """Test complete pipeline execution from start to finish"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model, metrics = runner.run_training()
        
        assert isinstance(model, CatBoostRegressor)
        assert model.is_fitted()
        assert 'R2' in metrics['train']
        assert 'R2' in metrics['test']
        
        model_path = Path(sample_config['postprocessing']['model_save_path'])
        assert model_path.exists()
    
    def test_pipeline_with_minimal_data(self, sample_config, tmp_path):
        """Test pipeline with minimal dataset"""
        minimal_data = pd.DataFrame({
            'X1': np.random.rand(30),
            'X2': np.random.rand(30) * 1000,
            'X3': np.random.rand(30) * 500,
            'X4': np.random.rand(30) * 200,
            'X5': [7.0] * 30,
            'X6': [2] * 30,
            'X7': [0.0] * 30,
            'X8': [0] * 30,
            'Y1': np.random.rand(30) * 40 + 5,
            'Y2': np.random.rand(30) * 40 + 10
        })
        
        csv_path = Path(sample_config['data']['raw_data']['csv_path'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        minimal_data.to_csv(csv_path, index=False)
        
        data_manager = DataManager(sample_config)
        data_manager.initialize_prod_database()
        
        runner = PipelineRunner(sample_config, data_manager)
        model, metrics = runner.run_training()
        
        assert isinstance(model, CatBoostRegressor)
        assert 'train' in metrics
        assert 'test' in metrics
    
    def test_multiple_training_runs(self, sample_config, data_manager_with_data):
        """Test running training multiple times"""
        runner = PipelineRunner(sample_config, data_manager_with_data)
        
        model1, metrics1 = runner.run_training()
        model2, metrics2 = runner.run_training()
        
        assert isinstance(model2, CatBoostRegressor)
        assert runner.trained_model == model2


class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_pipeline_with_missing_data(self, sample_config, tmp_path):
        """Test pipeline with missing data file"""
        data_manager = DataManager(sample_config)
        runner = PipelineRunner(sample_config, data_manager)
        
        with pytest.raises(FileNotFoundError):
            runner.run_training()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])