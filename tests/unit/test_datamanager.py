import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from common.data_manager import DataManager


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for testing"""
    config = {
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
    return config


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


@pytest.fixture
def sample_predictions():
    """Create sample prediction data"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'predicted_heating_load': [15.5, 20.8, 19.5, 17.0, 28.5]
    })


class TestDataManagerInitialization:
    """Test cases for DataManager initialization"""
    
    def test_initialization_with_config(self, sample_config):
        """Test DataManager initialization with valid config"""
        dm = DataManager(sample_config)
        
        assert dm.config == sample_config
        assert dm.data_config == sample_config['data']
        assert dm.raw_data_config == sample_config['data']['raw_data']
        assert dm.prod_data_config == sample_config['data']['prod_data']
    
    def test_paths_correctly_set(self, sample_config, tmp_path):
        """Test that file paths are correctly set from config"""
        dm = DataManager(sample_config)
        
        assert str(tmp_path / 'data/raw_data/csv/energy-efficiency-data.csv') in str(dm.raw_csv_path)
        assert str(tmp_path / 'data/raw_data/parquet/energy-efficiency-data.parquet') in str(dm.raw_parquet_path)
        assert str(tmp_path / 'data/prod_data/csv/predictions.csv') in str(dm.prod_csv_path)
        assert str(tmp_path / 'data/prod_data/parquet/predictions.parquet') in str(dm.prod_parquet_path)


class TestInitializeProdDatabase:
    """Test cases for initialize_prod_database method"""
    
    def test_initialize_with_csv(self, sample_config, sample_energy_data, tmp_path):
        """Test initialization when CSV file exists"""
        csv_path = Path(sample_config['data']['raw_data']['csv_path'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_csv(csv_path, index=False)
        
        dm = DataManager(sample_config)
        dm.initialize_prod_database()
        
        assert dm.raw_parquet_path.exists()
        
        df_parquet = pd.read_parquet(dm.raw_parquet_path)
        assert df_parquet.shape == sample_energy_data.shape
    
    def test_initialize_with_parquet_only(self, sample_config, sample_energy_data, tmp_path):
        """Test initialization when only Parquet file exists"""
        parquet_path = Path(sample_config['data']['raw_data']['parquet_path'])
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_parquet(parquet_path, index=False)
        
        dm = DataManager(sample_config)
        dm.initialize_prod_database()
        
        assert parquet_path.exists()
    
    def test_initialize_no_data_raises_error(self, sample_config):
        """Test that initialization raises error when no data exists"""
        dm = DataManager(sample_config)
        
        with pytest.raises(FileNotFoundError):
            dm.initialize_prod_database()


class TestLoadRawData:
    """Test cases for load_raw_data method"""
    
    def test_load_raw_data_csv(self, sample_config, sample_energy_data):
        """Test loading raw data from CSV"""
        csv_path = Path(sample_config['data']['raw_data']['csv_path'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_csv(csv_path, index=False)
        
        dm = DataManager(sample_config)
        df = dm.load_raw_data(format='csv')
        
        assert df.shape == sample_energy_data.shape
        assert list(df.columns) == list(sample_energy_data.columns)
    
    def test_load_raw_data_parquet(self, sample_config, sample_energy_data):
        """Test loading raw data from Parquet"""
        parquet_path = Path(sample_config['data']['raw_data']['parquet_path'])
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_parquet(parquet_path, index=False)
        
        dm = DataManager(sample_config)
        df = dm.load_raw_data(format='parquet')
        
        assert df.shape == sample_energy_data.shape
        assert list(df.columns) == list(sample_energy_data.columns)
    
    def test_load_raw_data_csv_not_found(self, sample_config):
        """Test loading non-existent CSV raises error"""
        dm = DataManager(sample_config)
        
        with pytest.raises(FileNotFoundError):
            dm.load_raw_data(format='csv')
    
    def test_load_raw_data_unsupported_format(self, sample_config, sample_energy_data):
        """Test loading with unsupported format raises error"""
        csv_path = Path(sample_config['data']['raw_data']['csv_path'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_csv(csv_path, index=False)
        
        dm = DataManager(sample_config)
        
        with pytest.raises(ValueError):
            dm.load_raw_data(format='json')


class TestSavePredictions:
    """Test cases for save_predictions method"""
    
    def test_save_predictions_csv(self, sample_config, sample_predictions):
        """Test saving predictions to CSV"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='csv')
        
        assert dm.prod_csv_path.exists()
        
        loaded = pd.read_csv(dm.prod_csv_path)
        assert loaded.shape == sample_predictions.shape
    
    def test_save_predictions_parquet(self, sample_config, sample_predictions):
        """Test saving predictions to Parquet"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='parquet')
        
        assert dm.prod_parquet_path.exists()
        
        loaded = pd.read_parquet(dm.prod_parquet_path)
        assert loaded.shape == sample_predictions.shape
    
    def test_save_predictions_creates_directory(self, sample_config, sample_predictions):
        """Test that save_predictions creates parent directories"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='csv')
        
        assert dm.prod_csv_path.parent.exists()
    
    def test_save_predictions_unsupported_format(self, sample_config, sample_predictions):
        """Test saving with unsupported format raises error"""
        dm = DataManager(sample_config)
        
        with pytest.raises(ValueError):
            dm.save_predictions(sample_predictions, format='xml')


class TestLoadPredictions:
    """Test cases for load_predictions method"""
    
    def test_load_predictions_csv(self, sample_config, sample_predictions):
        """Test loading predictions from CSV"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='csv')
        
        loaded = dm.load_predictions(format='csv')
        
        assert loaded is not None
        assert loaded.shape == sample_predictions.shape
    
    def test_load_predictions_parquet(self, sample_config, sample_predictions):
        """Test loading predictions from Parquet"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='parquet')
        
        loaded = dm.load_predictions(format='parquet')
        
        assert loaded is not None
        assert loaded.shape == sample_predictions.shape
    
    def test_load_predictions_no_file(self, sample_config):
        """Test loading predictions when file doesn't exist"""
        dm = DataManager(sample_config)
        
        loaded = dm.load_predictions(format='csv')
        
        assert loaded is None
    
    def test_load_predictions_unsupported_format(self, sample_config):
        """Test loading with unsupported format raises error"""
        dm = DataManager(sample_config)
        
        with pytest.raises(ValueError):
            dm.load_predictions(format='txt')


class TestAppendPredictions:
    """Test cases for append_predictions method"""
    
    def test_append_to_existing_predictions(self, sample_config):
        """Test appending new predictions to existing ones"""
        dm = DataManager(sample_config)
        
        first_batch = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='h'),
            'predicted_heating_load': [15.5, 20.8, 19.5]
        })
        
        second_batch = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 03:00:00', periods=2, freq='h'),
            'predicted_heating_load': [17.0, 28.5]
        })
        
        dm.save_predictions(first_batch, format='csv')
        dm.append_predictions(second_batch, format='csv')
        
        combined = dm.load_predictions(format='csv')
        
        assert combined.shape[0] == 5
        assert combined.shape[0] == first_batch.shape[0] + second_batch.shape[0]
    
    def test_append_to_empty_database(self, sample_config, sample_predictions):
        """Test appending when no existing predictions"""
        dm = DataManager(sample_config)
        dm.append_predictions(sample_predictions, format='csv')
        
        loaded = dm.load_predictions(format='csv')
        
        assert loaded.shape == sample_predictions.shape


class TestGetLatestPrediction:
    """Test cases for get_latest_prediction method"""
    
    def test_get_latest_with_timestamp(self, sample_config):
        """Test getting latest prediction with timestamp column"""
        dm = DataManager(sample_config)
        
        predictions = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'predicted_heating_load': [15.5, 20.8, 19.5, 17.0, 28.5]
        })
        
        dm.save_predictions(predictions, format='csv')
        latest = dm.get_latest_prediction(format='csv')
        
        assert latest is not None
        assert latest['predicted_heating_load'] == 28.5
    
    def test_get_latest_without_timestamp(self, sample_config):
        """Test getting latest prediction without timestamp column"""
        dm = DataManager(sample_config)
        
        predictions = pd.DataFrame({
            'predicted_heating_load': [15.5, 20.8, 19.5]
        })
        
        dm.save_predictions(predictions, format='csv')
        latest = dm.get_latest_prediction(format='csv')
        
        assert latest is not None
        assert latest['predicted_heating_load'] == 19.5
    
    def test_get_latest_no_predictions(self, sample_config):
        """Test getting latest prediction when no predictions exist"""
        dm = DataManager(sample_config)
        
        latest = dm.get_latest_prediction(format='csv')
        
        assert latest is None


class TestClearPredictions:
    """Test cases for clear_predictions method"""
    
    def test_clear_predictions_csv(self, sample_config, sample_predictions):
        """Test clearing CSV predictions"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='csv')
        
        assert dm.prod_csv_path.exists()
        
        dm.clear_predictions(format='csv')
        
        assert not dm.prod_csv_path.exists()
    
    def test_clear_predictions_parquet(self, sample_config, sample_predictions):
        """Test clearing Parquet predictions"""
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='parquet')
        
        assert dm.prod_parquet_path.exists()
        
        dm.clear_predictions(format='parquet')
        
        assert not dm.prod_parquet_path.exists()
    
    def test_clear_predictions_no_file(self, sample_config):
        """Test clearing predictions when file doesn't exist"""
        dm = DataManager(sample_config)
        
        dm.clear_predictions(format='csv')


class TestGetDataSummary:
    """Test cases for get_data_summary method"""
    
    def test_get_summary_with_data(self, sample_config, sample_energy_data, sample_predictions):
        """Test getting summary when both raw data and predictions exist"""
        csv_path = Path(sample_config['data']['raw_data']['csv_path'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_csv(csv_path, index=False)
        
        dm = DataManager(sample_config)
        dm.save_predictions(sample_predictions, format='csv')
        
        summary = dm.get_data_summary()
        
        assert 'raw_data' in summary
        assert 'predictions' in summary
        assert summary['raw_data']['rows'] == sample_energy_data.shape[0]
        assert summary['raw_data']['columns'] == sample_energy_data.shape[1]
        assert summary['predictions']['rows'] == sample_predictions.shape[0]
    
    def test_get_summary_no_predictions(self, sample_config, sample_energy_data):
        """Test getting summary when no predictions exist"""
        csv_path = Path(sample_config['data']['raw_data']['csv_path'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sample_energy_data.to_csv(csv_path, index=False)
        
        dm = DataManager(sample_config)
        summary = dm.get_data_summary()
        
        assert 'raw_data' in summary
        assert 'predictions' in summary
        assert summary['predictions']['status'] == 'No predictions available'
    
    def test_get_summary_no_raw_data(self, sample_config):
        """Test getting summary when no raw data exists"""
        dm = DataManager(sample_config)
        summary = dm.get_data_summary()
        
        assert 'raw_data' in summary
        assert 'error' in summary['raw_data']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])