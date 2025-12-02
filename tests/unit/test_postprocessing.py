import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import tempfile
import shutil

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from pipelines.postprocessing import PostprocessingPipeline


@pytest.fixture
def sample_config(tmp_path):
    """Create sample postprocessing configuration"""
    return {
        'postprocessing': {
            'model_save_path': str(tmp_path / 'models/prod/energy_forecast_model.cbm'),
            'prediction_columns': ['timestamp', 'predicted_heating_load'],
            'time_increment_minutes': 60
        }
    }


@pytest.fixture
def sample_trained_model():
    """Create a simple trained CatBoost model"""
    np.random.seed(42)
    X_train = np.random.rand(50, 5)
    y_train = np.random.rand(50) * 40 + 5
    
    model = CatBoostRegressor(iterations=10, learning_rate=0.1, depth=3, random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    
    return model


@pytest.fixture
def sample_metrics():
    """Create sample training metrics"""
    return {
        'train': {
            'MSE': 2.5,
            'RMSE': 1.58,
            'MAE': 1.2,
            'R2': 0.95,
            'MAPE': 5.5
        },
        'test': {
            'MSE': 3.2,
            'RMSE': 1.79,
            'MAE': 1.4,
            'R2': 0.92,
            'MAPE': 6.2
        }
    }


@pytest.fixture
def sample_feature_names():
    """Create sample feature names"""
    return ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height']


class TestPostprocessingPipelineInitialization:
    """Test cases for PostprocessingPipeline initialization"""
    
    def test_initialization_with_config(self, sample_config):
        """Test pipeline initialization with valid config"""
        pipeline = PostprocessingPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.postprocessing_config == sample_config['postprocessing']
        assert 'energy_forecast_model.cbm' in pipeline.model_save_path
        assert pipeline.prediction_columns == ['timestamp', 'predicted_heating_load']
        assert pipeline.time_increment_minutes == 60
    
    def test_initialization_with_defaults(self):
        """Test pipeline initialization with default values"""
        config = {'postprocessing': {}}
        pipeline = PostprocessingPipeline(config)
        
        assert pipeline.model_save_path == 'models/prod/energy_forecast_model.cbm'
        assert pipeline.prediction_columns == ['timestamp', 'predicted_heating_load']
        assert pipeline.time_increment_minutes == 60


class TestSaveModel:
    """Test cases for save_model method"""
    
    def test_save_model_basic(self, sample_config, sample_trained_model):
        """Test basic model saving"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_model(sample_trained_model)
        
        model_path = Path(pipeline.model_save_path)
        assert model_path.exists()
        assert model_path.suffix == '.cbm'
    
    def test_save_model_creates_directory(self, sample_config, sample_trained_model):
        """Test that save_model creates parent directories"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_model(sample_trained_model)
        
        model_path = Path(pipeline.model_save_path)
        assert model_path.parent.exists()
    
    def test_save_model_custom_path(self, sample_config, sample_trained_model, tmp_path):
        """Test saving model to custom path"""
        pipeline = PostprocessingPipeline(sample_config)
        custom_path = tmp_path / 'custom_model.cbm'
        
        pipeline.save_model(sample_trained_model, model_path=str(custom_path))
        
        assert custom_path.exists()
    
    def test_save_model_none_raises_error(self, sample_config):
        """Test that saving None model raises error"""
        pipeline = PostprocessingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Model is None"):
            pipeline.save_model(None)


class TestLoadModel:
    """Test cases for load_model method"""
    
    def test_load_model_basic(self, sample_config, sample_trained_model):
        """Test basic model loading"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_model(sample_trained_model)
        
        loaded_model = pipeline.load_model()
        
        assert isinstance(loaded_model, CatBoostRegressor)
    
    def test_load_model_custom_path(self, sample_config, sample_trained_model, tmp_path):
        """Test loading model from custom path"""
        pipeline = PostprocessingPipeline(sample_config)
        custom_path = tmp_path / 'custom_model.cbm'
        pipeline.save_model(sample_trained_model, model_path=str(custom_path))
        
        loaded_model = pipeline.load_model(model_path=str(custom_path))
        
        assert isinstance(loaded_model, CatBoostRegressor)
    
    def test_load_model_can_predict(self, sample_config, sample_trained_model):
        """Test that loaded model can make predictions"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_model(sample_trained_model)
        
        loaded_model = pipeline.load_model()
        X_test = np.random.rand(5, 5)
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == 5
    
    def test_load_model_nonexistent_raises_error(self, sample_config):
        """Test loading non-existent model raises error"""
        pipeline = PostprocessingPipeline(sample_config)
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_model()


class TestFormatPrediction:
    """Test cases for format_prediction method"""
    
    def test_format_prediction_with_timestamp(self, sample_config):
        """Test formatting single prediction with timestamp"""
        pipeline = PostprocessingPipeline(sample_config)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        result = pipeline.format_prediction(25.5, timestamp=timestamp)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 1
        assert 'timestamp' in result.columns
        assert 'predicted_heating_load' in result.columns
        assert result['predicted_heating_load'].iloc[0] == 25.5
    
    def test_format_prediction_without_timestamp(self, sample_config):
        """Test formatting prediction without timestamp uses current time"""
        pipeline = PostprocessingPipeline(sample_config)
        
        result = pipeline.format_prediction(25.5)
        
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert result['timestamp'].iloc[0] <= datetime.now()
    
    def test_format_prediction_column_names(self, sample_config):
        """Test that prediction uses configured column names"""
        pipeline = PostprocessingPipeline(sample_config)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        result = pipeline.format_prediction(25.5, timestamp=timestamp)
        
        assert list(result.columns) == ['timestamp', 'predicted_heating_load']


class TestFormatPredictionsBatch:
    """Test cases for format_predictions_batch method"""
    
    def test_format_predictions_batch_basic(self, sample_config):
        """Test formatting multiple predictions"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions = np.array([15.5, 20.8, 19.5, 17.0, 28.5])
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        result = pipeline.format_predictions_batch(predictions, start_timestamp=start_time)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 5
        assert 'timestamp' in result.columns
        assert 'predicted_heating_load' in result.columns
    
    def test_format_predictions_batch_timestamps(self, sample_config):
        """Test that timestamps are correctly incremented"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions = np.array([15.5, 20.8, 19.5])
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        result = pipeline.format_predictions_batch(predictions, start_timestamp=start_time)
        
        assert result['timestamp'].iloc[0] == start_time
        assert result['timestamp'].iloc[1] == start_time + timedelta(minutes=60)
        assert result['timestamp'].iloc[2] == start_time + timedelta(minutes=120)
    
    def test_format_predictions_batch_no_timestamp(self, sample_config):
        """Test batch formatting without start timestamp"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions = np.array([15.5, 20.8, 19.5])
        
        result = pipeline.format_predictions_batch(predictions)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3


class TestCreatePredictionSummary:
    """Test cases for create_prediction_summary method"""
    
    def test_create_prediction_summary_basic(self, sample_config):
        """Test creating prediction summary"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'predicted_heating_load': [15.5, 20.8, 19.5, 17.0, 28.5, 22.0, 24.5, 18.0, 21.5, 26.0]
        })
        
        summary = pipeline.create_prediction_summary(predictions_df)
        
        assert 'count' in summary
        assert 'mean' in summary
        assert 'std' in summary
        assert 'min' in summary
        assert 'max' in summary
        assert 'median' in summary
        assert 'q25' in summary
        assert 'q75' in summary
        assert summary['count'] == 10
    
    def test_create_prediction_summary_values(self, sample_config):
        """Test that summary values are correct"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'predicted_heating_load': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        summary = pipeline.create_prediction_summary(predictions_df)
        
        assert summary['mean'] == 30.0
        assert summary['min'] == 10.0
        assert summary['max'] == 50.0
        assert summary['median'] == 30.0
    
    def test_create_prediction_summary_missing_column(self, sample_config):
        """Test summary with missing prediction column"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'other_column': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="Prediction column"):
            pipeline.create_prediction_summary(predictions_df)


class TestSaveTrainingMetadata:
    """Test cases for save_training_metadata method"""
    
    def test_save_training_metadata_basic(self, sample_config, sample_metrics, sample_feature_names):
        """Test saving training metadata"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_training_metadata(sample_metrics, sample_feature_names)
        
        model_path = Path(pipeline.model_save_path)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.txt"
        
        assert metadata_path.exists()
    
    def test_save_training_metadata_content(self, sample_config, sample_metrics, sample_feature_names):
        """Test that metadata file contains expected content"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_training_metadata(sample_metrics, sample_feature_names)
        
        model_path = Path(pipeline.model_save_path)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.txt"
        
        with open(metadata_path, 'r') as f:
            content = f.read()
        
        assert 'MODEL TRAINING METADATA' in content
        assert 'TRAINING METRICS' in content
        assert 'TEST METRICS' in content
        assert 'FEATURES' in content
        assert 'relative_compactness' in content
    
    def test_save_training_metadata_custom_path(self, sample_config, sample_metrics, sample_feature_names, tmp_path):
        """Test saving metadata to custom path"""
        pipeline = PostprocessingPipeline(sample_config)
        custom_path = tmp_path / 'custom_model.cbm'
        
        pipeline.save_training_metadata(sample_metrics, sample_feature_names, model_path=str(custom_path))
        
        metadata_path = custom_path.parent / f"{custom_path.stem}_metadata.txt"
        assert metadata_path.exists()


class TestRunPipeline:
    """Test cases for run method (complete pipeline)"""
    
    def test_run_complete_pipeline(self, sample_config, sample_trained_model, sample_metrics, sample_feature_names):
        """Test running the complete postprocessing pipeline"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.run(sample_trained_model, sample_metrics, sample_feature_names)
        
        model_path = Path(pipeline.model_save_path)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.txt"
        
        assert model_path.exists()
        assert metadata_path.exists()
    
    def test_run_creates_both_files(self, sample_config, sample_trained_model, sample_metrics, sample_feature_names):
        """Test that run creates both model and metadata files"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.run(sample_trained_model, sample_metrics, sample_feature_names)
        
        model_path = Path(pipeline.model_save_path)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.txt"
        
        assert model_path.exists()
        assert metadata_path.exists()
        assert model_path.stat().st_size > 0
        assert metadata_path.stat().st_size > 0


class TestValidateModel:
    """Test cases for validate_model method"""
    
    def test_validate_model_valid(self, sample_config, sample_trained_model):
        """Test validation with valid model"""
        pipeline = PostprocessingPipeline(sample_config)
        result = pipeline.validate_model(sample_trained_model)
        
        assert result == True
    
    def test_validate_model_none(self, sample_config):
        """Test validation with None model"""
        pipeline = PostprocessingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Model is None"):
            pipeline.validate_model(None)
    
    def test_validate_model_wrong_type(self, sample_config):
        """Test validation with wrong model type"""
        pipeline = PostprocessingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Model must be CatBoostRegressor"):
            pipeline.validate_model("not a model")
    
    def test_validate_model_untrained(self, sample_config):
        """Test validation with untrained model"""
        pipeline = PostprocessingPipeline(sample_config)
        untrained_model = CatBoostRegressor(iterations=10, verbose=0)
        
        with pytest.raises(ValueError, match="not been trained"):
            pipeline.validate_model(untrained_model)


class TestGetModelInfo:
    """Test cases for get_model_info method"""
    
    def test_get_model_info_existing(self, sample_config, sample_trained_model):
        """Test getting info for existing model"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.save_model(sample_trained_model)
        
        info = pipeline.get_model_info()
        
        assert 'path' in info
        assert 'file_size_kb' in info
        assert 'modified_date' in info
        assert 'exists' in info
        assert info['exists'] == True
    
    def test_get_model_info_nonexistent(self, sample_config):
        """Test getting info for non-existent model"""
        pipeline = PostprocessingPipeline(sample_config)
        
        info = pipeline.get_model_info()
        
        assert 'status' in info
        assert 'Model file not found' in info['status']
    
    def test_get_model_info_with_metadata(self, sample_config, sample_trained_model, sample_metrics, sample_feature_names):
        """Test getting info when metadata exists"""
        pipeline = PostprocessingPipeline(sample_config)
        pipeline.run(sample_trained_model, sample_metrics, sample_feature_names)
        
        info = pipeline.get_model_info()
        
        assert 'metadata_exists' in info
        assert info['metadata_exists'] == True
        assert 'metadata_path' in info
    
    def test_get_model_info_custom_path(self, sample_config, sample_trained_model, tmp_path):
        """Test getting info for custom path"""
        pipeline = PostprocessingPipeline(sample_config)
        custom_path = tmp_path / 'custom_model.cbm'
        pipeline.save_model(sample_trained_model, model_path=str(custom_path))
        
        info = pipeline.get_model_info(model_path=str(custom_path))
        
        assert info['exists'] == True


class TestEdgeCases:
    """Test cases for edge cases and error handling"""
    
    def test_save_and_load_round_trip(self, sample_config, sample_trained_model):
        """Test that model can be saved and loaded successfully"""
        pipeline = PostprocessingPipeline(sample_config)
        
        X_test = np.random.rand(10, 5)
        original_predictions = sample_trained_model.predict(X_test)
        
        pipeline.save_model(sample_trained_model)
        loaded_model = pipeline.load_model()
        loaded_predictions = loaded_model.predict(X_test)
        
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)
    
    def test_large_prediction_batch(self, sample_config):
        """Test formatting large batch of predictions"""
        pipeline = PostprocessingPipeline(sample_config)
        predictions = np.random.rand(1000) * 40 + 5
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        result = pipeline.format_predictions_batch(predictions, start_timestamp=start_time)
        
        assert result.shape[0] == 1000
        assert result['timestamp'].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])