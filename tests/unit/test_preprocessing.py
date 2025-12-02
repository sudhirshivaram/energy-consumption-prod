import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))

from pipelines.preprocessing import PreprocessingPipeline


@pytest.fixture
def sample_config():
    """Create sample preprocessing configuration"""
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
            'columns_to_drop': ['Y2'],
            'reset_index': True
        }
    }


@pytest.fixture
def sample_energy_data():
    """Create sample energy efficiency data"""
    return pd.DataFrame({
        'X1': [0.98, 0.90, 0.86, 0.82, 0.79],
        'X2': [514.5, 563.5, 588.0, 612.5, 637.0],
        'X3': [294.0, 318.5, 294.0, 318.5, 343.0],
        'X4': [110.25, 122.5, 147.0, 147.0, 147.0],
        'X5': [7.0, 7.0, 7.0, 7.0, 7.0],
        'X6': [2, 3, 4, 5, 2],
        'X7': [0.0, 0.0, 0.0, 0.0, 0.0],
        'X8': [0, 0, 0, 0, 0],
        'Y1': [15.55, 20.84, 19.5, 17.05, 28.52],
        'Y2': [21.33, 28.28, 27.3, 23.77, 37.73]
    })


class TestPreprocessingPipelineInitialization:
    """Test cases for PreprocessingPipeline initialization"""
    
    def test_initialization_with_config(self, sample_config):
        """Test pipeline initialization with valid config"""
        pipeline = PreprocessingPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.preprocessing_config == sample_config['preprocessing']
        assert pipeline.column_mapping == sample_config['preprocessing']['column_mapping']
        assert pipeline.columns_to_drop == sample_config['preprocessing']['columns_to_drop']
        assert pipeline.reset_index == sample_config['preprocessing']['reset_index']
    
    def test_initialization_with_empty_config(self):
        """Test pipeline initialization with empty config"""
        pipeline = PreprocessingPipeline({})
        
        assert pipeline.column_mapping == {}
        assert pipeline.columns_to_drop == []
        assert pipeline.reset_index == True
    
    def test_initialization_with_partial_config(self):
        """Test pipeline initialization with partial config"""
        config = {
            'preprocessing': {
                'column_mapping': {'X1': 'feature1'}
            }
        }
        
        pipeline = PreprocessingPipeline(config)
        
        assert pipeline.column_mapping == {'X1': 'feature1'}
        assert pipeline.columns_to_drop == []


class TestRenameColumns:
    """Test cases for rename_columns method"""
    
    def test_rename_columns_with_mapping(self, sample_config, sample_energy_data):
        """Test renaming columns with valid mapping"""
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.rename_columns(sample_energy_data)
        
        assert 'relative_compactness' in result.columns
        assert 'surface_area' in result.columns
        assert 'heating_load' in result.columns
        assert 'X1' not in result.columns
        assert 'X2' not in result.columns
    
    def test_rename_columns_preserves_data(self, sample_config, sample_energy_data):
        """Test that renaming preserves data values"""
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.rename_columns(sample_energy_data)
        
        assert result.shape == sample_energy_data.shape
        assert result['relative_compactness'].equals(sample_energy_data['X1'])
        assert result['heating_load'].equals(sample_energy_data['Y1'])
    
    def test_rename_columns_no_mapping(self, sample_energy_data):
        """Test renaming with no column mapping"""
        config = {'preprocessing': {'column_mapping': {}}}
        pipeline = PreprocessingPipeline(config)
        result = pipeline.rename_columns(sample_energy_data)
        
        assert list(result.columns) == list(sample_energy_data.columns)
    
    def test_rename_columns_partial_mapping(self, sample_energy_data):
        """Test renaming with partial column mapping"""
        config = {
            'preprocessing': {
                'column_mapping': {'X1': 'feature1', 'X2': 'feature2'}
            }
        }
        pipeline = PreprocessingPipeline(config)
        result = pipeline.rename_columns(sample_energy_data)
        
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert 'X3' in result.columns
        assert 'X1' not in result.columns


class TestDropColumns:
    """Test cases for drop_columns method"""
    
    def test_drop_columns_single(self, sample_config, sample_energy_data):
        """Test dropping a single column"""
        config = {
            'preprocessing': {
                'columns_to_drop': ['Y2']
            }
        }
        pipeline = PreprocessingPipeline(config)
        result = pipeline.drop_columns(sample_energy_data)
        
        assert 'Y2' not in result.columns
        assert result.shape[1] == sample_energy_data.shape[1] - 1
    
    def test_drop_columns_multiple(self, sample_energy_data):
        """Test dropping multiple columns"""
        config = {
            'preprocessing': {
                'columns_to_drop': ['Y1', 'Y2']
            }
        }
        pipeline = PreprocessingPipeline(config)
        result = pipeline.drop_columns(sample_energy_data)
        
        assert 'Y1' not in result.columns
        assert 'Y2' not in result.columns
        assert result.shape[1] == sample_energy_data.shape[1] - 2
    
    def test_drop_columns_none(self, sample_energy_data):
        """Test with no columns to drop"""
        config = {'preprocessing': {'columns_to_drop': []}}
        pipeline = PreprocessingPipeline(config)
        result = pipeline.drop_columns(sample_energy_data)
        
        assert result.shape == sample_energy_data.shape
    
    def test_drop_columns_nonexistent(self, sample_energy_data):
        """Test dropping non-existent columns"""
        config = {
            'preprocessing': {
                'columns_to_drop': ['nonexistent_column']
            }
        }
        pipeline = PreprocessingPipeline(config)
        result = pipeline.drop_columns(sample_energy_data)
        
        assert result.shape == sample_energy_data.shape
    
    def test_drop_columns_preserves_data(self, sample_energy_data):
        """Test that dropping columns preserves remaining data"""
        config = {
            'preprocessing': {
                'columns_to_drop': ['Y2']
            }
        }
        pipeline = PreprocessingPipeline(config)
        result = pipeline.drop_columns(sample_energy_data)
        
        assert result['Y1'].equals(sample_energy_data['Y1'])
        assert result['X1'].equals(sample_energy_data['X1'])


class TestResetDataframeIndex:
    """Test cases for reset_dataframe_index method"""
    
    def test_reset_index_enabled(self, sample_config, sample_energy_data):
        """Test resetting index when enabled"""
        df_with_custom_index = sample_energy_data.copy()
        df_with_custom_index.index = [10, 20, 30, 40, 50]
        
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.reset_dataframe_index(df_with_custom_index)
        
        assert list(result.index) == [0, 1, 2, 3, 4]
    
    def test_reset_index_disabled(self, sample_energy_data):
        """Test with index reset disabled"""
        df_with_custom_index = sample_energy_data.copy()
        df_with_custom_index.index = [10, 20, 30, 40, 50]
        
        config = {'preprocessing': {'reset_index': False}}
        pipeline = PreprocessingPipeline(config)
        result = pipeline.reset_dataframe_index(df_with_custom_index)
        
        assert list(result.index) == [10, 20, 30, 40, 50]
    
    def test_reset_index_preserves_data(self, sample_config, sample_energy_data):
        """Test that resetting index preserves data"""
        df_with_custom_index = sample_energy_data.copy()
        df_with_custom_index.index = [10, 20, 30, 40, 50]
        
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.reset_dataframe_index(df_with_custom_index)
        
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), 
            sample_energy_data.reset_index(drop=True)
        )


class TestRunPipeline:
    """Test cases for run method (complete pipeline)"""
    
    def test_run_complete_pipeline(self, sample_config, sample_energy_data):
        """Test running the complete preprocessing pipeline"""
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.run(sample_energy_data)
        
        assert 'relative_compactness' in result.columns
        assert 'heating_load' in result.columns
        assert 'Y2' not in result.columns
        assert list(result.index) == [0, 1, 2, 3, 4]
    
    def test_run_preserves_row_count(self, sample_config, sample_energy_data):
        """Test that pipeline preserves row count"""
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.run(sample_energy_data)
        
        assert result.shape[0] == sample_energy_data.shape[0]
    
    def test_run_reduces_column_count(self, sample_config, sample_energy_data):
        """Test that pipeline maintains column count when column to drop doesn't exist after rename"""
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.run(sample_energy_data)
        
        assert result.shape[1] == sample_energy_data.shape[1]
    
    def test_run_with_minimal_config(self, sample_energy_data):
        """Test running with minimal configuration"""
        config = {'preprocessing': {}}
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run(sample_energy_data)
        
        assert result.shape[0] == sample_energy_data.shape[0]


class TestGetColumnMapping:
    """Test cases for get_column_mapping method"""
    
    def test_get_column_mapping(self, sample_config):
        """Test getting column mapping"""
        pipeline = PreprocessingPipeline(sample_config)
        mapping = pipeline.get_column_mapping()
        
        assert mapping == sample_config['preprocessing']['column_mapping']
        assert 'X1' in mapping
        assert mapping['X1'] == 'relative_compactness'


class TestGetColumnsToDrop:
    """Test cases for get_columns_to_drop method"""
    
    def test_get_columns_to_drop(self, sample_config):
        """Test getting columns to drop"""
        pipeline = PreprocessingPipeline(sample_config)
        columns = pipeline.get_columns_to_drop()
        
        assert columns == ['Y2']
    
    def test_get_columns_to_drop_empty(self):
        """Test getting columns to drop when none specified"""
        config = {'preprocessing': {}}
        pipeline = PreprocessingPipeline(config)
        columns = pipeline.get_columns_to_drop()
        
        assert columns == []


class TestValidateInput:
    """Test cases for validate_input method"""
    
    def test_validate_input_valid_dataframe(self, sample_config, sample_energy_data):
        """Test validation with valid DataFrame"""
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.validate_input(sample_energy_data)
        
        assert result == True
    
    def test_validate_input_none(self, sample_config):
        """Test validation with None DataFrame"""
        pipeline = PreprocessingPipeline(sample_config)
        
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            pipeline.validate_input(None)
    
    def test_validate_input_empty_dataframe(self, sample_config):
        """Test validation with empty DataFrame"""
        pipeline = PreprocessingPipeline(sample_config)
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            pipeline.validate_input(empty_df)
    
    def test_validate_input_no_rows(self, sample_config):
        """Test validation with DataFrame having no rows"""
        pipeline = PreprocessingPipeline(sample_config)
        df_no_rows = pd.DataFrame(columns=['X1', 'X2', 'X3'])
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(df_no_rows)
    
    def test_validate_input_no_columns(self, sample_config):
        """Test validation with DataFrame having no columns"""
        pipeline = PreprocessingPipeline(sample_config)
        df_no_cols = pd.DataFrame(index=[0, 1, 2])
        
        with pytest.raises(ValueError, match="None or empty"):
            pipeline.validate_input(df_no_cols)


class TestEdgeCases:
    """Test cases for edge cases and error handling"""
    
    def test_preprocessing_with_single_row(self, sample_config):
        """Test preprocessing with single row"""
        single_row_df = pd.DataFrame({
            'X1': [0.98], 'X2': [514.5], 'X3': [294.0], 'X4': [110.25],
            'X5': [7.0], 'X6': [2], 'X7': [0.0], 'X8': [0],
            'Y1': [15.55], 'Y2': [21.33]
        })
        
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.run(single_row_df)
        
        assert result.shape[0] == 1
        assert 'heating_load' in result.columns
    
    def test_preprocessing_with_large_dataset(self, sample_config):
        """Test preprocessing with larger dataset"""
        large_df = pd.DataFrame({
            'X1': np.random.rand(1000),
            'X2': np.random.rand(1000) * 1000,
            'X3': np.random.rand(1000) * 500,
            'X4': np.random.rand(1000) * 200,
            'X5': np.random.choice([3.5, 7.0], 1000),
            'X6': np.random.choice([2, 3, 4, 5], 1000),
            'X7': np.random.choice([0.0, 0.1, 0.25, 0.4], 1000),
            'X8': np.random.choice([0, 1, 2, 3, 4, 5], 1000),
            'Y1': np.random.rand(1000) * 40 + 5,
            'Y2': np.random.rand(1000) * 40 + 10
        })
        
        pipeline = PreprocessingPipeline(sample_config)
        result = pipeline.run(large_df)
        
        assert result.shape[0] == 1000
        assert 'heating_load' in result.columns
        assert 'Y2' not in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])