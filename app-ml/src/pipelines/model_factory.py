from typing import Dict, Any, Union
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression   


class ModelFactory:
    """
    Factory class for creating different regression models based on configuration.
    Supports LinearRegression, CatBoost, XGBoost, LightGBM, RandomForest, and GradientBoosting.
    """
    
    SUPPORTED_MODELS = {
        'LinearRegression': LinearRegression,
        'CatBoostRegressor': CatBoostRegressor,
        'XGBoostRegressor': XGBRegressor,
        'XGBRegressor': XGBRegressor,
        'LGBMRegressor': LGBMRegressor,
        'LightGBMRegressor': LGBMRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor
    }
    
    @classmethod
    def create_model(cls, model_type: str, params: Dict[str, Any]) -> Union[
        LinearRegression, CatBoostRegressor, XGBRegressor, LGBMRegressor, RandomForestRegressor, GradientBoostingRegressor
    ]:
        """
        Create a model instance based on model type and parameters.
        
        Args:
            model_type: Type of model to create (e.g., 'CatBoostRegressor', 'XGBRegressor')
            params: Dictionary of model parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls.SUPPORTED_MODELS:
            supported = ', '.join(cls.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported models: {supported}"
            )
        
        model_class = cls.SUPPORTED_MODELS[model_type]
        
        params_copy = params.copy()
        
        if model_type in ['CatBoostRegressor']:
            if 'random_seed' not in params_copy and 'random_state' in params_copy:
                params_copy['random_seed'] = params_copy.pop('random_state')
            if 'verbosity' in params_copy:
                params_copy['verbose'] = params_copy.pop('verbosity')
            if 'n_estimators' in params_copy:
                params_copy['iterations'] = params_copy.pop('n_estimators')
            if 'max_depth' in params_copy and 'depth' not in params_copy:
                params_copy['depth'] = params_copy.pop('max_depth')
            if 'reg_lambda' in params_copy:
                params_copy['l2_leaf_reg'] = params_copy.pop('reg_lambda')
        
        elif model_type in ['XGBoostRegressor', 'XGBRegressor']:
            if 'iterations' in params_copy:
                params_copy['n_estimators'] = params_copy.pop('iterations')
            if 'random_seed' in params_copy:
                params_copy['random_state'] = params_copy.pop('random_seed')
            if 'verbose' in params_copy:
                params_copy['verbosity'] = params_copy.pop('verbose')
            if 'depth' in params_copy:
                params_copy['max_depth'] = params_copy.pop('depth')
            if 'l2_leaf_reg' in params_copy:
                params_copy['reg_lambda'] = params_copy.pop('l2_leaf_reg')
        
        elif model_type in ['LGBMRegressor', 'LightGBMRegressor']:
            if 'iterations' in params_copy:
                params_copy['n_estimators'] = params_copy.pop('iterations')
            if 'random_seed' in params_copy:
                params_copy['random_state'] = params_copy.pop('random_seed')
            if 'verbosity' in params_copy:
                params_copy['verbose'] = params_copy.pop('verbosity')
            if 'depth' in params_copy:
                params_copy['max_depth'] = params_copy.pop('depth')
            if 'l2_leaf_reg' in params_copy:
                params_copy['reg_lambda'] = params_copy.pop('l2_leaf_reg')
        
        elif model_type in ['RandomForestRegressor', 'GradientBoostingRegressor']:
            if 'iterations' in params_copy:
                params_copy['n_estimators'] = params_copy.pop('iterations')
            if 'random_seed' in params_copy:
                params_copy['random_state'] = params_copy.pop('random_seed')
            if 'depth' in params_copy:
                params_copy['max_depth'] = params_copy.pop('depth')
            if 'verbosity' in params_copy:
                params_copy.pop('verbosity')
            if 'verbose' in params_copy:
                params_copy.pop('verbose')
            
            params_copy.pop('reg_lambda', None)
            params_copy.pop('l2_leaf_reg', None)
            
            if model_type == 'RandomForestRegressor':
                params_copy.pop('learning_rate', None)
        
        try:
            model = model_class(**params_copy)
            print(f"Created {model_type} with parameters: {params_copy}")
            return model
        except Exception as e:
            raise ValueError(f"Error creating {model_type}: {str(e)}")
    
    @classmethod
    def get_supported_models(cls) -> list:
        """
        Get list of supported model types.
        
        Returns:
            List of supported model type names
        """
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def is_supported(cls, model_type: str) -> bool:
        """
        Check if a model type is supported.
        
        Args:
            model_type: Model type to check
            
        Returns:
            True if supported, False otherwise
        """
        return model_type in cls.SUPPORTED_MODELS
    
    @classmethod
    def get_default_params(cls, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            'CatBoostRegressor': {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'verbose': 0
            },
            'XGBoostRegressor': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': 0
            },
            'XGBRegressor': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': 0
            },
            'LGBMRegressor': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbose': -1
            },
            'LightGBMRegressor': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbose': -1
            },
            'RandomForestRegressor': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'verbose': 0
            },
            'GradientBoostingRegressor': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbose': 0
            }
        }
        
        return defaults.get(model_type, {})