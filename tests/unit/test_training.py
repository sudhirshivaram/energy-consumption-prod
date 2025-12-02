import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from catboost import CatBoostRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from pipelines.training import TrainingPipeline


@pytest.fixture
def sample_config():
    """Create sample training configuration"""
    return {
        'training': {
            'target_column': 'heating_load',
            'target_shift': 1,
            'test_size': 0.2,
            'random_state': 42,
            'shuffle': False,
            'model': {
                'type': 'CatBoostRegressor',
                'params': {
                    'iterations': 50,
                    'learning_rate': 0.1,
                    'depth': 4,
                    'l2_leaf_reg': 3,
                    'random_seed': 42,
                    'verbose': 0
                },
                'early_stopping_rounds': 10
            },
            'hyperparameter_tuning': {
                'enabled': False,
                'n_trials': 10,
                'params': {
                    'iterations': {'min': 10, 'max': 100},
                    'learning_rate': {'min': 0.01, 'max': 0.3, 'log': True},
                    'depth': {'min': 3, 'max': 6},
                    'l2_leaf_reg': {'min': 1, 'max': 5}
                }
            }
        }
    }


@pytest.fixture
def sample_engineered_data():
    """Create sample data with engineered features"""
    np.random.seed(42)
    n_samples = 50
    
    return pd.DataFrame({
        'relative_compactness': np.random.rand(n_samples),
        'surface_area': np.random.rand(n_samples) * 1000,
        'wall_area': np.random.rand(n_samples) * 500,
        'roof_area': np.random.rand(n_samples) * 200,
        'overall_height': np.random.choice([3.5, 7.0], n_samples),
        'orientation': np.random.choice([2, 3, 4, 5], n_samples),
        'glazing_area': np.random.choice([0.0, 0.1, 0.25, 0.4], n_samples),
        'glazing_area_distribution': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        'heating_load': np.random.rand(n_samples) * 40 + 5,
        'heating_load_lag_1': np.random.rand(n_samples) * 40 + 5,
        'heating_load_lag_2': np.random.rand(n_samples) * 40 + 5,
        'relative_compactness_lag_1': np.random.rand(n_samples),
        'surface_area_lag_1': np.random.rand(n_samples) * 1000
    })


class TestTrainingPipelineInitialization:
    """Test cases for TrainingPipeline initialization"""
    
    def test_initialization_with_config(self, sample_config):
        """Test pipeline initialization with valid config"""
        pipeline = TrainingPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.training_config == sample_config['training']
        assert pipeline.target_column == 'heating_load'
        assert pipeline.target_shift == 1
        assert pipeline.test_size == 0.2
        assert pipeline.random_state == 42
        assert pipeline.shuffle == False
    
    def test_initialization_with_defaults(self):
        """Test pipeline initialization with default values"""
        config = {'training': {}}
        pipeline = TrainingPipeline(config)
        
        assert pipeline.target_column == 'heating_load'
        assert pipeline.target_shift == 1
        assert pipeline.test_size == 0.2
        assert pipeline.random_state == 42
        assert pipeline.shuffle == False
    
    def test_initialization_model_params(self, sample_config):
        """Test that model parameters are correctly loaded"""
        pipeline = TrainingPipeline(sample_config)
        
        assert pipeline.model_params['iterations'] == 50
        assert pipeline.model_params['learning_rate'] == 0.1
        assert pipeline.early_stopping_rounds == 10


class TestCreateTargetVariable:
    """Test cases for create_target_variable method"""
    
    def test_create_target_variable_shift_1(self, sample_config, sample_engineered_data):
        """Test creating target with shift=1"""
        pipeline = TrainingPipeline(sample_config)
        result = pipeline.create_target_variable(sample_engineered_data)
        
        assert 'heating_load_target' in result.columns
        assert result.shape[0] == sample_engineered_data.shape[0] - 1
    
    def test_create_target_variable_values(self, sample_config, sample_engineered_data):
        """Test that target values are correctly shifted"""
        pipeline = TrainingPipeline(sample_config)
        result = pipeline.create_target_variable(sample_engineered_data)
        
        assert result['heating_load_target'].iloc[0] == sample_engineered_data['heating_load'].iloc[1]
    
    def test_create_target_variable_shift_2(self, sample_engineered_data):
        """Test creating target with shift=2"""
        config = {'training': {'target_column': 'heating_load', 'target_shift': 2}}
        pipeline = TrainingPipeline(config)
        result = pipeline.create_target_variable(sample_engineered_data)
        
        assert result.shape[0] == sample_engineered_data.shape[0] - 2
    
    def test_create_target_missing_column(self, sample_config):
        """Test with missing target column"""
        df = pd.DataFrame({'other_column': [1, 2, 3]})
        pipeline = TrainingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Target column"):
            pipeline.create_target_variable(df)


class TestPrepareFeaturesAndTarget:
    """Test cases for prepare_features_and_target method"""
    
    def test_prepare_features_and_target(self, sample_config, sample_engineered_data):
        """Test preparing features and target"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        
        assert 'heating_load_target' not in X.columns
        assert 'heating_load' in X.columns
        assert len(y) == len(X)
    
    def test_feature_names_stored(self, sample_config, sample_engineered_data):
        """Test that feature names are stored"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        
        assert pipeline.feature_names is not None
        assert len(pipeline.feature_names) == X.shape[1]
    
    def test_prepare_missing_target(self, sample_config):
        """Test with missing target column"""
        df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        pipeline = TrainingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Target column"):
            pipeline.prepare_features_and_target(df)


class TestSplitData:
    """Test cases for split_data method"""
    
    def test_split_data_proportions(self, sample_config, sample_engineered_data):
        """Test train-test split proportions"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        total_samples = len(X)
        assert len(X_train) == int(total_samples * 0.8)
        assert len(X_test) == total_samples - len(X_train)
    
    def test_split_data_no_shuffle(self, sample_config, sample_engineered_data):
        """Test that shuffle=False preserves order"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        assert X_train.index[0] < X_test.index[0]
    
    def test_split_data_stored_in_pipeline(self, sample_config, sample_engineered_data):
        """Test that split data is stored in pipeline"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        assert pipeline.X_train is not None
        assert pipeline.X_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_test is not None


class TestTrainModel:
    """Test cases for train_model method"""
    
    def test_train_model_basic(self, sample_config, sample_engineered_data):
        """Test basic model training"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        model = pipeline.train_model(X_train, y_train, X_test, y_test)
        
        assert isinstance(model, CatBoostRegressor)
        assert hasattr(model, 'best_iteration_')
    
    def test_train_model_with_custom_params(self, sample_config, sample_engineered_data):
        """Test training with custom parameters"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        custom_params = {
            'iterations': 20,
            'learning_rate': 0.2,
            'depth': 3,
            'random_seed': 42,
            'verbose': 0
        }
        
        model = pipeline.train_model(X_train, y_train, X_test, y_test, params=custom_params)
        
        assert isinstance(model, CatBoostRegressor)
    
    def test_train_model_can_predict(self, sample_config, sample_engineered_data):
        """Test that trained model can make predictions"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        
        model = pipeline.train_model(X_train, y_train, X_test, y_test)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestEvaluateModel:
    """Test cases for evaluate_model method"""
    
    def test_evaluate_model_returns_metrics(self, sample_config, sample_engineered_data):
        """Test that evaluation returns proper metrics"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        model = pipeline.train_model(X_train, y_train, X_test, y_test)
        
        metrics = pipeline.evaluate_model(model, X_train, y_train, X_test, y_test)
        
        assert 'train' in metrics
        assert 'test' in metrics
        assert 'MSE' in metrics['train']
        assert 'RMSE' in metrics['train']
        assert 'MAE' in metrics['train']
        assert 'R2' in metrics['train']
        assert 'MAPE' in metrics['train']
    
    def test_evaluate_model_metrics_reasonable(self, sample_config, sample_engineered_data):
        """Test that metrics are in reasonable ranges"""
        pipeline = TrainingPipeline(sample_config)
        df_with_target = pipeline.create_target_variable(sample_engineered_data)
        X, y = pipeline.prepare_features_and_target(df_with_target)
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
        model = pipeline.train_model(X_train, y_train, X_test, y_test)
        
        metrics = pipeline.evaluate_model(model, X_train, y_train, X_test, y_test)
        
        assert metrics['train']['MSE'] >= 0
        assert metrics['train']['RMSE'] >= 0
        assert metrics['train']['MAE'] >= 0
        assert 0 <= metrics['train']['R2'] <= 1


class TestRunPipeline:
    """Test cases for run method (complete pipeline)"""
    
    def test_run_complete_pipeline(self, sample_config, sample_engineered_data):
        """Test running the complete training pipeline"""
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(sample_engineered_data)
        
        assert isinstance(model, CatBoostRegressor)
        assert 'train' in metrics
        assert 'test' in metrics
        assert pipeline.model is not None
    
    def test_run_stores_splits(self, sample_config, sample_engineered_data):
        """Test that run stores train/test splits"""
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(sample_engineered_data)
        
        assert pipeline.X_train is not None
        assert pipeline.X_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_test is not None
    
    def test_run_with_tuning_disabled(self, sample_config, sample_engineered_data):
        """Test run with hyperparameter tuning disabled"""
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(sample_engineered_data)
        
        assert isinstance(model, CatBoostRegressor)


class TestGetFeatureImportance:
    """Test cases for get_feature_importance method"""
    
    def test_get_feature_importance(self, sample_config, sample_engineered_data):
        """Test getting feature importance"""
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(sample_engineered_data)
        
        importance_df = pipeline.get_feature_importance()
        
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(pipeline.feature_names)
    
    def test_get_feature_importance_top_n(self, sample_config, sample_engineered_data):
        """Test getting top N feature importance"""
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(sample_engineered_data)
        
        importance_df = pipeline.get_feature_importance(top_n=5)
        
        assert len(importance_df) == 5
    
    def test_get_feature_importance_sorted(self, sample_config, sample_engineered_data):
        """Test that feature importance is sorted"""
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(sample_engineered_data)
        
        importance_df = pipeline.get_feature_importance()
        
        assert importance_df['importance'].is_monotonic_decreasing
    
    def test_get_feature_importance_no_model(self, sample_config):
        """Test getting importance without trained model"""
        pipeline = TrainingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            pipeline.get_feature_importance()


class TestValidateInput:
    """Test cases for validate_input method"""
    
    def test_validate_input_valid_dataframe(self, sample_config, sample_engineered_data):
        """Test validation with valid DataFrame"""
        pipeline = TrainingPipeline(sample_config)
        result = pipeline.validate_input(sample_engineered_data)
        
        assert result == True
    
    def test_validate_input_none(self, sample_config):
        """Test validation with None DataFrame"""
        pipeline = TrainingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(None)
    
    def test_validate_input_empty(self, sample_config):
        """Test validation with empty DataFrame"""
        pipeline = TrainingPipeline(sample_config)
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(empty_df)
    
    def test_validate_input_missing_target(self, sample_config):
        """Test validation with missing target column"""
        df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        pipeline = TrainingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Target column"):
            pipeline.validate_input(df)
    
    def test_validate_input_insufficient_data(self, sample_config):
        """Test validation with insufficient data"""
        df = pd.DataFrame({
            'heating_load': [1, 2, 3, 4, 5],
            'feature1': [1, 2, 3, 4, 5]
        })
        pipeline = TrainingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            pipeline.validate_input(df)


class TestEdgeCases:
    """Test cases for edge cases and error handling"""
    
    def test_training_with_minimal_data(self, sample_config):
        """Test training with minimal dataset"""
        np.random.seed(42)
        small_df = pd.DataFrame({
            'heating_load': np.random.rand(20) * 40 + 5,
            'feature1': np.random.rand(20),
            'feature2': np.random.rand(20)
        })
        
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(small_df)
        
        assert isinstance(model, CatBoostRegressor)
        assert 'train' in metrics
    
    def test_training_with_many_features(self, sample_config):
        """Test training with many features"""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        data = {f'feature_{i}': np.random.rand(n_samples) for i in range(n_features)}
        data['heating_load'] = np.random.rand(n_samples) * 40 + 5
        
        large_df = pd.DataFrame(data)
        
        pipeline = TrainingPipeline(sample_config)
        model, metrics = pipeline.run(large_df)
        
        assert isinstance(model, CatBoostRegressor)
        assert len(pipeline.feature_names) == n_features + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])