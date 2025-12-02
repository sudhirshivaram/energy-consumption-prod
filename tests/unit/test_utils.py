import pytest
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from common.utils import (
    read_config,
    save_config,
    ensure_directory_exists,
    load_data,
    save_data,
    calculate_metrics,
    print_metrics
)


class TestReadConfig:
    """Test cases for read_config function"""
    
    def test_read_config_valid_file(self, tmp_path):
        """Test reading a valid config file"""
        config_data = {
            'preprocessing': {
                'column_mapping': {'X1': 'feature1'},
                'columns_to_drop': ['Y2']
            },
            'training': {
                'test_size': 0.2
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        result = read_config(config_file)
        
        assert result == config_data
        assert 'preprocessing' in result
        assert result['training']['test_size'] == 0.2
    
    def test_read_config_file_not_found(self, tmp_path):
        """Test reading non-existent config file"""
        non_existent_file = tmp_path / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            read_config(non_existent_file)
    
    def test_read_config_empty_file(self, tmp_path):
        """Test reading empty config file"""
        config_file = tmp_path / "empty_config.yaml"
        config_file.touch()
        
        result = read_config(config_file)
        
        assert result is None


class TestSaveConfig:
    """Test cases for save_config function"""
    
    def test_save_config_basic(self, tmp_path):
        """Test saving a basic config file"""
        config_data = {
            'key1': 'value1',
            'key2': {'nested': 'value2'}
        }
        
        config_file = tmp_path / "saved_config.yaml"
        save_config(config_data, config_file)
        
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded == config_data
    
    def test_save_config_creates_directory(self, tmp_path):
        """Test that save_config creates parent directories"""
        config_data = {'test': 'data'}
        config_file = tmp_path / "subdir" / "config.yaml"
        
        save_config(config_data, config_file)
        
        assert config_file.exists()
        assert config_file.parent.exists()


class TestEnsureDirectoryExists:
    """Test cases for ensure_directory_exists function"""
    
    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory"""
        new_dir = tmp_path / "new_directory"
        
        result = ensure_directory_exists(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
    
    def test_existing_directory(self, tmp_path):
        """Test with already existing directory"""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory_exists(existing_dir)
        
        assert existing_dir.exists()
        assert result == existing_dir
    
    def test_nested_directory_creation(self, tmp_path):
        """Test creating nested directories"""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        
        result = ensure_directory_exists(nested_dir)
        
        assert nested_dir.exists()
        assert result == nested_dir


class TestLoadData:
    """Test cases for load_data function"""
    
    def test_load_csv_data(self, tmp_path):
        """Test loading CSV file"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        csv_file = tmp_path / "test_data.csv"
        df.to_csv(csv_file, index=False)
        
        result = load_data(csv_file, file_format='csv')
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_load_parquet_data(self, tmp_path):
        """Test loading Parquet file"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        parquet_file = tmp_path / "test_data.parquet"
        df.to_parquet(parquet_file, index=False)
        
        result = load_data(parquet_file, file_format='parquet')
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_load_data_file_not_found(self, tmp_path):
        """Test loading non-existent file"""
        non_existent = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            load_data(non_existent, file_format='csv')
    
    def test_load_data_unsupported_format(self, tmp_path):
        """Test loading with unsupported format"""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        
        with pytest.raises(ValueError):
            load_data(file_path, file_format='txt')


class TestSaveData:
    """Test cases for save_data function"""
    
    def test_save_csv_data(self, tmp_path):
        """Test saving DataFrame to CSV"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        csv_file = tmp_path / "output.csv"
        save_data(df, csv_file, file_format='csv')
        
        assert csv_file.exists()
        
        loaded = pd.read_csv(csv_file)
        pd.testing.assert_frame_equal(loaded, df)
    
    def test_save_parquet_data(self, tmp_path):
        """Test saving DataFrame to Parquet"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        parquet_file = tmp_path / "output.parquet"
        save_data(df, parquet_file, file_format='parquet')
        
        assert parquet_file.exists()
        
        loaded = pd.read_parquet(parquet_file)
        pd.testing.assert_frame_equal(loaded, df)
    
    def test_save_data_creates_directory(self, tmp_path):
        """Test that save_data creates parent directories"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        file_path = tmp_path / "subdir" / "data.csv"
        
        save_data(df, file_path, file_format='csv')
        
        assert file_path.exists()
        assert file_path.parent.exists()
    
    def test_save_data_unsupported_format(self, tmp_path):
        """Test saving with unsupported format"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        file_path = tmp_path / "output.txt"
        
        with pytest.raises(ValueError):
            save_data(df, file_path, file_format='txt')


class TestCalculateMetrics:
    """Test cases for calculate_metrics function"""
    
    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['MSE'] == 0.0
        assert metrics['RMSE'] == 0.0
        assert metrics['MAE'] == 0.0
        assert metrics['R2'] == 1.0
        assert metrics['MAPE'] == 0.0
    
    def test_calculate_metrics_known_values(self):
        """Test metrics with known values"""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert 'MAPE' in metrics
        
        assert metrics['RMSE'] == np.sqrt(metrics['MSE'])
        assert 0 <= metrics['R2'] <= 1
    
    def test_calculate_metrics_all_keys_present(self):
        """Test that all expected metrics are present"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        expected_keys = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
        for key in expected_keys:
            assert key in metrics
    
    def test_calculate_metrics_types(self):
        """Test that metrics return correct types"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        for key, value in metrics.items():
            assert isinstance(value, (float, np.floating))


class TestPrintMetrics:
    """Test cases for print_metrics function"""
    
    def test_print_metrics_output(self, capsys):
        """Test that print_metrics produces output"""
        metrics = {
            'MSE': 1.5,
            'RMSE': 1.2247,
            'MAE': 1.0,
            'R2': 0.95,
            'MAPE': 5.5
        }
        
        print_metrics(metrics, dataset_name="Test")
        
        captured = capsys.readouterr()
        
        assert "Test Performance Metrics:" in captured.out
        assert "MSE" in captured.out
        assert "RMSE" in captured.out
        assert "MAE" in captured.out
        assert "R2" in captured.out
        assert "MAPE" in captured.out
    
    def test_print_metrics_default_dataset_name(self, capsys):
        """Test print_metrics with default dataset name"""
        metrics = {'MSE': 1.0, 'RMSE': 1.0, 'MAE': 1.0, 'R2': 0.8, 'MAPE': 5.0}
        
        print_metrics(metrics)
        
        captured = capsys.readouterr()
        
        assert "Dataset Performance Metrics:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])