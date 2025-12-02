import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the path so we can import psai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_df():
    """Creates a simple dataframe for testing."""
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    return df

@pytest.fixture
def sample_config():
    """Creates a basic config dictionary."""
    return {
        'dataset': {
            'train_path': 'dummy.csv',
            'target': 'target',
            'task_type': 'classification',
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 2,
            'verbose': 0
        },
        'models': {
            'lightgbm': {'enabled': False, 'optuna_metric': 'auc', 'optuna_n_jobs': 1},
            'xgboost': {'enabled': False, 'optuna_metric': 'auc', 'optuna_n_jobs': 1},
            'catboost': {'enabled': False, 'optuna_metric': 'auc', 'optuna_n_jobs': 1},
            'random_forest': {'enabled': False, 'optuna_metric': 'auc', 'optuna_n_jobs': 1},
            'pytorch': {'enabled': False, 'optuna_metric': 'auc', 'optuna_n_jobs': 1},
        },
        'preprocessor': {
            "numerical": {"imputer": "mean", "scaler": "standard"},
            "skewed": {"imputer": "median", "scaler": "log"},
            "outlier": {"imputer": "median", "scaler": "log"},
            "low_cardinality": {"imputer": "most_frequent", "encoder": "onehot", "scaler": "none"},
            "high_cardinality": {"imputer": "most_frequent", "encoder": "target", "scaler": "none"},
            "dimension_reduction": {"method": "none"}
        },
        'stacking': {'cv_enabled': False, 'final_enabled': False},
        'voting': {'cv_enabled': False, 'final_enabled': False},
        'mlflow': {'enabled': False},
        'output': {'models_dir': 'tmp_models', 'results_dir': 'tmp_results'}
    }
