import pytest
import pandas as pd
import numpy as np
import os
import shutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from psai.core.psml import psML

def test_initialization(sample_df, sample_config):
    """Test that psML initializes correctly with provided config and data."""
    X = sample_df.drop(columns=['target'])
    y = sample_df[['target']]
    
    model = psML(config=sample_config, X=X, y=y)
    
    assert model.config == sample_config
    assert model.X_train is not None
    assert model.X_test is not None
    assert model.y_train is not None
    assert model.y_test is not None
    assert model.config['dataset']['task_type'] == 'classification'

def test_create_preprocessor(sample_df, sample_config):
    """Test that the preprocessor is created correctly."""
    X = sample_df.drop(columns=['target'])
    y = sample_df[['target']]
    
    model = psML(config=sample_config, X=X, y=y)
    model.create_preprocessor()
    
    assert model.preprocessor is not None
    assert model.columns is not None
    assert 'nskewed_cols' in model.columns
    assert 'skewed_cols' in model.columns

def test_optimize_model_integration(sample_df, sample_config):
    """
    Integration test for optimize_model.
    We enable one fast model (Random Forest) and run 1 trial to ensure the flow works.
    """
    # Enable Random Forest for this test
    sample_config['models']['random_forest']['enabled'] = True
    sample_config['models']['random_forest']['optuna_trials'] = 1
    sample_config['models']['random_forest']['optuna_n_jobs'] = 1
    # Ensure params key exists for the adapter
    if 'params' not in sample_config['models']['random_forest']:
        sample_config['models']['random_forest']['params'] = {'verbose': 0, 'n_jobs': 1}
    
    # Use a temporary directory for outputs to avoid clutter
    sample_config['output']['models_dir'] = 'tmp_test_models'
    sample_config['output']['results_dir'] = 'tmp_test_results'
    
    X = sample_df.drop(columns=['target'])
    y = sample_df[['target']]
    
    model = psML(config=sample_config, X=X, y=y)
    
    # Run optimization for the enabled model
    model.optimize_all_models()
    
    # Check if results were populated
    assert 'random_forest' in model.models
    assert 'cv_score' in model.models['random_forest']
    assert 'best_params' in model.models['random_forest']
    assert 'final_model_random_forest' in model.models
    
    # Cleanup
    if os.path.exists('tmp_test_models'):
        shutil.rmtree('tmp_test_models')
    if os.path.exists('tmp_test_results'):
        shutil.rmtree('tmp_test_results')

def test_scores(sample_df, sample_config):
    """Test the scores method."""
    X = sample_df.drop(columns=['target'])
    y = sample_df[['target']]
    
    model = psML(config=sample_config, X=X, y=y)
    
    # Manually populate some dummy results
    model.models['lightgbm'] = {'cv_score': 0.85}
    model.models['final_model_lightgbm'] = {'score': 0.88}
    model.config['models']['lightgbm']['enabled'] = True # Ensure it's enabled so it shows up
    
    scores = model.scores()
    
    assert 'lightgbm' in scores
    assert scores['lightgbm']['cv_score'] == 0.85
    assert scores['lightgbm']['test_score'] == 0.88

def test_save_load(sample_df, sample_config):
    """Test saving and loading the psML object."""
    X = sample_df.drop(columns=['target'])
    y = sample_df[['target']]
    
    model = psML(config=sample_config, X=X, y=y)
    model.some_attribute = "test_value"
    
    save_path = os.path.join("tmp_test_dir", "tmp_test_model.pkl")
    model.save(save_path)
    
    assert os.path.exists(save_path)
    
    loaded_model = psML.load(save_path)
    
    assert loaded_model.some_attribute == "test_value"
    assert loaded_model.config['dataset']['task_type'] == 'classification'
    
    # Cleanup
    if os.path.exists("tmp_test_dir"):
        shutil.rmtree("tmp_test_dir")
