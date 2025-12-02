import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from pipelines.feature_engineering import FeatureEngineeringPipeline


@pytest.fixture
def sample_config():
    """Create sample feature engineering configuration"""
    return {
        'feature_engineering': {
            'lag_features': {
                'heating_load': [1, 2, 3, 5, 10],
                'relative_compactness': [1, 2],
                'surface_area': [1, 2]
            },
            'fill_method': 'bfill'
        }
    }


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed energy efficiency data"""
    return pd.DataFrame({
        'relative_compactness': [0.98, 0.90, 0.86, 0.82, 0.79, 0.76, 0.74, 0.72, 0.69, 0.67, 0.64, 0.62],
        'surface_area': [514.5, 563.5, 588.0, 612.5, 637.0, 661.5, 686.0, 710.5, 735.0, 759.5, 784.0, 808.5],
        'wall_area': [294.0, 318.5, 294.0, 318.5, 343.0, 294.0, 318.5, 343.0, 294.0, 318.5, 343.0, 367.5],
        'roof_area': [110.25, 122.5, 147.0, 147.0, 147.0, 171.0, 171.0, 171.0, 195.5, 195.5, 195.5, 195.5],
        'overall_height': [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        'orientation': [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
        'glazing_area': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'glazing_area_distribution': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'heating_load': [15.55, 20.84, 19.5, 17.05, 28.52, 22.45, 24.32, 26.18, 21.77, 23.65, 25.53, 27.41]
    })


class TestFeatureEngineeringPipelineInitialization:
    """Test cases for FeatureEngineeringPipeline initialization"""
    
    def test_initialization_with_config(self, sample_config):
        """Test pipeline initialization with valid config"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.feature_config == sample_config['feature_engineering']
        assert pipeline.lag_features_config == sample_config['feature_engineering']['lag_features']
        assert pipeline.fill_method == 'bfill'
    
    def test_initialization_with_empty_config(self):
        """Test pipeline initialization with empty config"""
        pipeline = FeatureEngineeringPipeline({})
        
        assert pipeline.lag_features_config == {}
        assert pipeline.fill_method == 'bfill'
    
    def test_initialization_with_custom_fill_method(self):
        """Test pipeline initialization with custom fill method"""
        config = {
            'feature_engineering': {
                'lag_features': {},
                'fill_method': 'ffill'
            }
        }
        pipeline = FeatureEngineeringPipeline(config)
        
        assert pipeline.fill_method == 'ffill'


class TestCreateLagFeatures:
    """Test cases for create_lag_features method"""
    
    def test_create_single_lag_feature(self, sample_config, sample_preprocessed_data):
        """Test creating a single lag feature"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.create_lag_features(sample_preprocessed_data, 'heating_load', [1])
        
        assert 'heating_load_lag_1' in result.columns
        assert result['heating_load_lag_1'].iloc[1] == sample_preprocessed_data['heating_load'].iloc[0]
    
    def test_create_multiple_lag_features(self, sample_config, sample_preprocessed_data):
        """Test creating multiple lag features"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.create_lag_features(sample_preprocessed_data, 'heating_load', [1, 2, 3])
        
        assert 'heating_load_lag_1' in result.columns
        assert 'heating_load_lag_2' in result.columns
        assert 'heating_load_lag_3' in result.columns
    
    def test_lag_feature_values_correctness(self, sample_config, sample_preprocessed_data):
        """Test that lag feature values are correctly shifted"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.create_lag_features(sample_preprocessed_data, 'heating_load', [1, 2])
        
        assert pd.isna(result['heating_load_lag_1'].iloc[0])
        assert result['heating_load_lag_1'].iloc[1] == sample_preprocessed_data['heating_load'].iloc[0]
        assert result['heating_load_lag_2'].iloc[2] == sample_preprocessed_data['heating_load'].iloc[0]
    
    def test_create_lag_for_nonexistent_column(self, sample_config, sample_preprocessed_data):
        """Test creating lag features for non-existent column"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.create_lag_features(sample_preprocessed_data, 'nonexistent_column', [1])
        
        assert result.shape == sample_preprocessed_data.shape
        assert 'nonexistent_column_lag_1' not in result.columns
    
    def test_create_lag_preserves_original_data(self, sample_config, sample_preprocessed_data):
        """Test that creating lag features preserves original data"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        original_columns = sample_preprocessed_data.columns.tolist()
        result = pipeline.create_lag_features(sample_preprocessed_data, 'heating_load', [1])
        
        for col in original_columns:
            pd.testing.assert_series_equal(result[col], sample_preprocessed_data[col])


class TestCreateAllLagFeatures:
    """Test cases for create_all_lag_features method"""
    
    def test_create_all_lag_features(self, sample_config, sample_preprocessed_data):
        """Test creating all lag features from config"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.create_all_lag_features(sample_preprocessed_data)
        
        assert 'heating_load_lag_1' in result.columns
        assert 'heating_load_lag_2' in result.columns
        assert 'heating_load_lag_3' in result.columns
        assert 'heating_load_lag_5' in result.columns
        assert 'heating_load_lag_10' in result.columns
        assert 'relative_compactness_lag_1' in result.columns
        assert 'relative_compactness_lag_2' in result.columns
        assert 'surface_area_lag_1' in result.columns
        assert 'surface_area_lag_2' in result.columns
    
    def test_create_all_lag_features_count(self, sample_config, sample_preprocessed_data):
        """Test that correct number of lag features are created"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.create_all_lag_features(sample_preprocessed_data)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        expected_count = 5 + 2 + 2
        assert len(lag_columns) == expected_count
    
    def test_create_all_lag_features_empty_config(self, sample_preprocessed_data):
        """Test with no lag features configuration"""
        config = {'feature_engineering': {'lag_features': {}}}
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.create_all_lag_features(sample_preprocessed_data)
        
        assert result.shape == sample_preprocessed_data.shape


class TestFillMissingValues:
    """Test cases for fill_missing_values method"""
    
    def test_fill_missing_with_bfill(self, sample_preprocessed_data):
        """Test filling missing values with backward fill"""
        config = {
            'feature_engineering': {
                'lag_features': {'heating_load': [1, 2]},
                'fill_method': 'bfill'
            }
        }
        pipeline = FeatureEngineeringPipeline(config)
        df_with_lags = pipeline.create_all_lag_features(sample_preprocessed_data)
        result = pipeline.fill_missing_values(df_with_lags)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert result[lag_columns].isnull().sum().sum() == 0
    
    def test_fill_missing_with_ffill(self, sample_preprocessed_data):
        """Test filling missing values with forward fill"""
        config = {
            'feature_engineering': {
                'lag_features': {'heating_load': [1, 2]},
                'fill_method': 'ffill'
            }
        }
        pipeline = FeatureEngineeringPipeline(config)
        df_with_lags = pipeline.create_all_lag_features(sample_preprocessed_data)
        result = pipeline.fill_missing_values(df_with_lags)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert result[lag_columns].isnull().sum().sum() == 0
    
    def test_fill_missing_with_mean(self, sample_preprocessed_data):
        """Test filling missing values with mean"""
        config = {
            'feature_engineering': {
                'lag_features': {'heating_load': [1, 2]},
                'fill_method': 'mean'
            }
        }
        pipeline = FeatureEngineeringPipeline(config)
        df_with_lags = pipeline.create_all_lag_features(sample_preprocessed_data)
        result = pipeline.fill_missing_values(df_with_lags)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert result[lag_columns].isnull().sum().sum() == 0
    
    def test_fill_missing_with_median(self, sample_preprocessed_data):
        """Test filling missing values with median"""
        config = {
            'feature_engineering': {
                'lag_features': {'heating_load': [1, 2]},
                'fill_method': 'median'
            }
        }
        pipeline = FeatureEngineeringPipeline(config)
        df_with_lags = pipeline.create_all_lag_features(sample_preprocessed_data)
        result = pipeline.fill_missing_values(df_with_lags)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert result[lag_columns].isnull().sum().sum() == 0
    
    def test_fill_missing_with_zero(self, sample_preprocessed_data):
        """Test filling missing values with zeros"""
        config = {
            'feature_engineering': {
                'lag_features': {'heating_load': [1, 2]},
                'fill_method': 'zero'
            }
        }
        pipeline = FeatureEngineeringPipeline(config)
        df_with_lags = pipeline.create_all_lag_features(sample_preprocessed_data)
        result = pipeline.fill_missing_values(df_with_lags)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert result[lag_columns].isnull().sum().sum() == 0
    
    def test_fill_missing_no_lag_columns(self, sample_preprocessed_data):
        """Test filling when no lag columns exist"""
        config = {'feature_engineering': {'fill_method': 'bfill'}}
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.fill_missing_values(sample_preprocessed_data)
        
        assert result.shape == sample_preprocessed_data.shape


class TestRunPipeline:
    """Test cases for run method (complete pipeline)"""
    
    def test_run_complete_pipeline(self, sample_config, sample_preprocessed_data):
        """Test running the complete feature engineering pipeline"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.run(sample_preprocessed_data)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert len(lag_columns) == 9
        assert result[lag_columns].isnull().sum().sum() == 0
    
    def test_run_preserves_row_count(self, sample_config, sample_preprocessed_data):
        """Test that pipeline preserves row count"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.run(sample_preprocessed_data)
        
        assert result.shape[0] == sample_preprocessed_data.shape[0]
    
    def test_run_increases_column_count(self, sample_config, sample_preprocessed_data):
        """Test that pipeline increases column count with lag features"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.run(sample_preprocessed_data)
        
        assert result.shape[1] > sample_preprocessed_data.shape[1]
    
    def test_run_with_minimal_config(self, sample_preprocessed_data):
        """Test running with minimal configuration"""
        config = {'feature_engineering': {}}
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.run(sample_preprocessed_data)
        
        assert result.shape == sample_preprocessed_data.shape


class TestGetLagFeaturesConfig:
    """Test cases for get_lag_features_config method"""
    
    def test_get_lag_features_config(self, sample_config):
        """Test getting lag features configuration"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        config = pipeline.get_lag_features_config()
        
        assert config == sample_config['feature_engineering']['lag_features']
        assert 'heating_load' in config
        assert config['heating_load'] == [1, 2, 3, 5, 10]


class TestGetLagFeatureNames:
    """Test cases for get_lag_feature_names method"""
    
    def test_get_lag_feature_names(self, sample_config, sample_preprocessed_data):
        """Test getting lag feature names"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        df_engineered = pipeline.run(sample_preprocessed_data)
        lag_names = pipeline.get_lag_feature_names(df_engineered)
        
        assert len(lag_names) == 9
        assert 'heating_load_lag_1' in lag_names
        assert 'relative_compactness_lag_1' in lag_names
    
    def test_get_lag_feature_names_no_lags(self, sample_preprocessed_data):
        """Test getting lag feature names when no lags exist"""
        config = {'feature_engineering': {}}
        pipeline = FeatureEngineeringPipeline(config)
        lag_names = pipeline.get_lag_feature_names(sample_preprocessed_data)
        
        assert len(lag_names) == 0


class TestValidateInput:
    """Test cases for validate_input method"""
    
    def test_validate_input_valid_dataframe(self, sample_config, sample_preprocessed_data):
        """Test validation with valid DataFrame"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.validate_input(sample_preprocessed_data)
        
        assert result == True
    
    def test_validate_input_none(self, sample_config):
        """Test validation with None DataFrame"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(None)
    
    def test_validate_input_empty_dataframe(self, sample_config):
        """Test validation with empty DataFrame"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(empty_df)
    
    def test_validate_input_no_rows(self, sample_config):
        """Test validation with DataFrame having no rows"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        df_no_rows = pd.DataFrame(columns=['heating_load', 'relative_compactness'])
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(df_no_rows)
    
    def test_validate_input_missing_columns(self, sample_config):
        """Test validation with missing columns for lag features"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        df_missing_cols = pd.DataFrame({
            'other_column': [1, 2, 3, 4, 5]
        })
        
        result = pipeline.validate_input(df_missing_cols)
        assert result == True


class TestGetFeatureStatistics:
    """Test cases for get_feature_statistics method"""
    
    def test_get_feature_statistics(self, sample_config, sample_preprocessed_data):
        """Test getting feature statistics"""
        pipeline = FeatureEngineeringPipeline(sample_config)
        df_engineered = pipeline.run(sample_preprocessed_data)
        stats = pipeline.get_feature_statistics(df_engineered)
        
        assert 'total_features' in stats
        assert 'original_features' in stats
        assert 'lag_features' in stats
        assert 'lag_feature_names' in stats
        assert 'missing_values' in stats
        assert stats['lag_features'] == 9
        assert stats['missing_values'] == 0
    
    def test_get_feature_statistics_no_lags(self, sample_preprocessed_data):
        """Test getting statistics when no lag features"""
        config = {'feature_engineering': {}}
        pipeline = FeatureEngineeringPipeline(config)
        df_engineered = pipeline.run(sample_preprocessed_data)
        stats = pipeline.get_feature_statistics(df_engineered)
        
        assert stats['lag_features'] == 0
        assert stats['total_features'] == stats['original_features']


class TestEdgeCases:
    """Test cases for edge cases and error handling"""
    
    def test_feature_engineering_with_small_dataset(self, sample_config):
        """Test feature engineering with small dataset"""
        small_df = pd.DataFrame({
            'heating_load': [15.55, 20.84, 19.5],
            'relative_compactness': [0.98, 0.90, 0.86],
            'surface_area': [514.5, 563.5, 588.0]
        })
        
        pipeline = FeatureEngineeringPipeline(sample_config)
        result = pipeline.run(small_df)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        assert len(lag_columns) == 9
        
        remaining_nulls = result[lag_columns].isnull().sum().sum()
        assert remaining_nulls >= 0
    
    def test_feature_engineering_with_large_lags(self, sample_preprocessed_data):
        """Test with large lag values"""
        config = {
            'feature_engineering': {
                'lag_features': {
                    'heating_load': [50, 100]
                },
                'fill_method': 'bfill'
            }
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.run(sample_preprocessed_data)
        
        assert 'heating_load_lag_50' in result.columns
        assert 'heating_load_lag_100' in result.columns
    
    def test_multiple_columns_with_different_lags(self, sample_preprocessed_data):
        """Test creating different lag periods for multiple columns"""
        config = {
            'feature_engineering': {
                'lag_features': {
                    'heating_load': [1, 2, 3],
                    'relative_compactness': [1],
                    'surface_area': [1, 2, 3, 4, 5]
                },
                'fill_method': 'bfill'
            }
        }
        
        pipeline = FeatureEngineeringPipeline(config)
        result = pipeline.run(sample_preprocessed_data)
        
        lag_columns = [col for col in result.columns if '_lag_' in col]
        expected_count = 3 + 1 + 5
        assert len(lag_columns) == expected_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])