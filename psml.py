import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import pickle
import os
import lightgbm as lgb
from typing import Dict, Any, Optional, Union, List

from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, f1_score, roc_auc_score, mean_squared_log_error, root_mean_squared_log_error,
    precision_score
)
from sklearn.base import clone

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from psai.scalersencoders import create_preprocessor
from psai.pstorch import PyTorchRegressor, PyTorchClassifier

class psML:
    """
    Machine Learning Pipeline for training, evaluation, and prediction
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        self.models = {}
        
        if config is None:
            try:
                from .config import CONFIG
                self.config = CONFIG
            except ImportError:
                print("Error: Config not provided and config.py not found.")
                return
        else:
            self.config = config
        
        # Read and split data
        if X is None or y is None:

            train_path = self.config['dataset']['train_path']
            if not os.path.exists(train_path):
                print(f"Error: Training data file not found at {train_path}")
                return # Or raise FileNotFoundError(f"Training data file not found at {train_path}")
            df = pd.read_csv(train_path)
            target = self.config['dataset']['target']
            if target not in df.columns:
                print(f"Error: Target column '{target}' not found in the dataset.")
                return
            X = df.drop(columns=[target])
            y = df[[target]]

        if isinstance(y, pd.Series):
            y = y.to_frame()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['dataset']['test_size'], 
            random_state=self.config['dataset']['random_state']
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.columns = {}
        self.preprocessor = None

        # Store min value of y_train for rmsle_safe
        self.y_train_min = self.y_train.min()
        
        # Internal storage for optimization results (replacing globals)
        self.best_cv_scores = {}
        self.best_cv_models = {}
        
        # Initialize best scores with worst possible values
        for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch']:
            if self.config['dataset']['task_type'] == 'classification':
                self.best_cv_scores[model_name] = 0
            else:
                self.best_cv_scores[model_name] = float('inf')
            self.best_cv_models[model_name] = None

        # Optuna setup
        self.pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, 
            max_resource=5,
            reduction_factor=3
        )
        self.sampler = optuna.samplers.TPESampler(seed=42)

    def save_model(self, filepath: str):
        """Save the pSML object to a file using pickle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model successfully saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'psML':
        """Load a pSML object from a file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model successfully loaded from {filepath}")
        return model

    def optimize_all_models(self):
        self.create_preprocessor()
        
        # List of supported models
        supported_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch']
        
        for model_name in supported_models:
            if self.config['models'].get(model_name, {}).get('enabled', False):
                self.optimize_model(model_name)

    def create_preprocessor(self):
        preprocessor, columns = create_preprocessor(self.config['preprocessor'], self.X_train)
        self.preprocessor = preprocessor
        self.columns = columns

    def scores(self, return_json=False):
        results = {}
        for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch']:
            if self.config['models'].get(model_name, {}).get('enabled', False):
                cv_score = self.models.get(model_name, {}).get("cv_score", "N/A")
                test_score = self.models.get(f"final_model_{model_name}", {}).get("score", "N/A")
                results[model_name] = {
                    "cv_score": cv_score,
                    "test_score": test_score
                }
                print(f'{model_name.capitalize()} CV Score: {cv_score}, Test Score: {test_score}')
        
        if return_json:
            return json.dumps(results)

        return results

    def rmsle_safe(self, y_true, y_pred):
        """
        Root Mean Squared Logarithmic Error that handles negative values
        by converting them to the minimum value from the training set.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        min_val = float(self.y_train_min)
        if min_val <= 0:
            min_val = 0.1
        
        # Create copies to avoid modifying original arrays if they were passed by reference
        y_true_safe = np.maximum(y_true, min_val)
        y_pred_safe = np.maximum(y_pred, min_val)
        
        return np.sqrt(mean_squared_error(np.log1p(y_true_safe), np.log1p(y_pred_safe)))

    def get_evaluation_metric(self, metric_name: str):
        metric_mapping = {
            'acc': accuracy_score,
            'f1': lambda y_true, y_pred: f1_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
            'auc': roc_auc_score,
            'prec': lambda y_true, y_pred: precision_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'msle': mean_squared_log_error,
            'rmsle': root_mean_squared_log_error,
            'rmsle_safe': self.rmsle_safe,
            'rmse_safe': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error,
        }
        
        if metric_name not in metric_mapping:
            print(f"Warning: Metric '{metric_name}' not found.")
            if self.config['dataset']['task_type'] == 'regression':
                print("Defaulting to RMSE for regression.")
                return lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            else:
                print("Defaulting to roc_auc_score for classification.")
                return roc_auc_score
        
        return metric_mapping[metric_name]

    def _suggest_param(self, trial, model_name, param_name, default_type=None, **default_kwargs):
        """
        Suggests a parameter value based on config or defaults.
        """
        config_params = self.config['models'][model_name].get('optuna_params', {})
        
        # Check if param is in config
        if param_name in config_params:
            p_config = config_params[param_name]
            p_type = p_config.get('type')
            
            if p_type == 'int':
                return trial.suggest_int(param_name, p_config['low'], p_config['high'], log=p_config.get('log', False))
            elif p_type == 'float':
                return trial.suggest_float(param_name, p_config['low'], p_config['high'], log=p_config.get('log', False))
            elif p_type == 'categorical':
                return trial.suggest_categorical(param_name, p_config['choices'])
        
        # Fallback to default
        if default_type == 'int':
            return trial.suggest_int(param_name, **default_kwargs)
        elif default_type == 'float':
            return trial.suggest_float(param_name, **default_kwargs)
        elif default_type == 'categorical':
            return trial.suggest_categorical(param_name, **default_kwargs)
        
        return None

    def _get_model_params(self, model_name: str, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        """
        Centralized parameter definition. 
        If trial is provided, suggests parameters. 
        If trial is None, expects self.models[model_name]['best_params'] to exist (for final training).
        """
        model_config = self.config['models'][model_name]['params']
        params = {}
        
        # If we are not in a trial (final training), we use the best params found
        if trial is None:
            best_params = self.models[model_name].get('best_params', {})
            # We will merge these later, but for now we need to know if we are suggesting or just retrieving
            pass

        if model_name == 'lightgbm':
            params = {
                "objective": model_config['objective'],
                "device": model_config['device'],
                "metric": model_config['eval_metric'], 
                "verbosity": -1,
                "n_estimators": 10000,
                "num_threads": model_config['num_threads'],
                "verbose": model_config['verbose']
            }
            if trial:
                params.update({
                    "boosting_type": self._suggest_param(trial, model_name, 'boosting_type', 'categorical', choices=['gbdt', 'goss']),
                    "lambda_l1": self._suggest_param(trial, model_name, "lambda_l1", 'float', low=1e-8, high=10.0, log=True),
                    "lambda_l2": self._suggest_param(trial, model_name, "lambda_l2", 'float', low=1e-8, high=10.0, log=True),
                    "num_leaves": self._suggest_param(trial, model_name, "num_leaves", 'int', low=20, high=300),
                    "feature_fraction": self._suggest_param(trial, model_name, "feature_fraction", 'float', low=0.4, high=1.0),
                    "min_child_samples": self._suggest_param(trial, model_name, "min_child_samples", 'int', low=5, high=100),
                    "learning_rate": self._suggest_param(trial, model_name, "learning_rate", 'float', low=0.001, high=0.2, log=True),
                    "min_split_gain": self._suggest_param(trial, model_name, "min_split_gain", 'float', low=1e-8, high=1.0, log=True),
                    "max_depth": self._suggest_param(trial, model_name, "max_depth", 'int', low=3, high=20),
                })
                if params['boosting_type'] != 'goss':
                    params["bagging_fraction"] = self._suggest_param(trial, model_name, "bagging_fraction", 'float', low=0.4, high=1.0)
                    params["bagging_freq"] = self._suggest_param(trial, model_name, "bagging_freq", 'int', low=1, high=7)
            else:
                # Merge best params
                params.update(best_params)

        elif model_name == 'xgboost':
            params = {
                "objective": model_config['objective'],
                "eval_metric": model_config['eval_metric'],
                "device": 'cuda' if model_config['device'] == 'gpu' else 'cpu',
                "early_stopping_rounds": 100,
                "nthread": model_config['nthread'],
                "verbose": model_config['verbose']
            }
            if trial:
                params.update({
                    "booster": self._suggest_param(trial, model_name, 'booster', 'categorical', choices=['gbtree']),
                    "max_depth": self._suggest_param(trial, model_name, 'max_depth', 'int', low=3, high=20),
                    "learning_rate": self._suggest_param(trial, model_name, 'learning_rate', 'float', low=0.001, high=0.2, log=True),
                    "n_estimators": self._suggest_param(trial, model_name, 'n_estimators', 'int', low=500, high=3000),
                    "subsample": self._suggest_param(trial, model_name, 'subsample', 'float', low=0, high=1),
                    "lambda": self._suggest_param(trial, model_name, 'lambda', 'float', low=1e-4, high=5, log=True),
                    "gamma": self._suggest_param(trial, model_name, 'gamma', 'float', low=1e-4, high=5, log=True),
                    "alpha": self._suggest_param(trial, model_name, 'alpha', 'float', low=1e-4, high=5, log=True),
                    "min_child_weight": self._suggest_param(trial, model_name, 'min_child_weight', 'categorical', choices=[0.5, 1, 3, 5]),
                    "colsample_bytree": self._suggest_param(trial, model_name, 'colsample_bytree', 'float', low=0.5, high=1),
                    "colsample_bylevel": self._suggest_param(trial, model_name, 'colsample_bylevel', 'float', low=0.5, high=1),
                })
            else:
                params.update(best_params)

        elif model_name == 'catboost':
            params = {
                'loss_function': model_config['objective'],
                'eval_metric': model_config['eval_metric'],
                'task_type': 'CPU' if model_config['device'] == 'cpu' else 'GPU',
                'random_seed': 42,
                'verbose': model_config['verbose'],
                'thread_count': model_config['thread_count']
            }
            if trial:
                params.update({
                    'n_estimators': self._suggest_param(trial, model_name, 'n_estimators', 'int', low=100, high=3000),
                    'learning_rate': self._suggest_param(trial, model_name, 'learning_rate', 'float', low=0.001, high=0.2, log=True),
                    'depth': self._suggest_param(trial, model_name, 'depth', 'int', low=4, high=10),
                    'l2_leaf_reg': self._suggest_param(trial, model_name, 'l2_leaf_reg', 'float', low=1e-3, high=10.0, log=True),
                    'border_count': self._suggest_param(trial, model_name, 'border_count', 'int', low=32, high=128),
                    'bootstrap_type': self._suggest_param(trial, model_name, 'bootstrap_type', 'categorical', choices=['Bayesian', 'Bernoulli', 'MVS']),
                    'feature_border_type': self._suggest_param(trial, model_name, 'feature_border_type', 'categorical', choices=['Median', 'Uniform', 'GreedyMinEntropy']),
                    'leaf_estimation_iterations': self._suggest_param(trial, model_name, 'leaf_estimation_iterations', 'int', low=1, high=10),
                    'min_data_in_leaf': self._suggest_param(trial, model_name, 'min_data_in_leaf', 'int', low=1, high=30),
                    'random_strength': self._suggest_param(trial, model_name, 'random_strength', 'float', low=1e-9, high=10, log=True),
                    'grow_policy': self._suggest_param(trial, model_name, 'grow_policy', 'categorical', choices=['SymmetricTree', 'Depthwise', 'Lossguide']),
                })
                if params['bootstrap_type'] == 'MVS' and params['task_type'] == 'GPU':
                    print("GPU choose must switch Bootstrap type to Bayesian")
                    params['bootstrap_type'] = 'Bayesian'
                if params['bootstrap_type'] == 'Bernoulli':
                    params['subsample'] = self._suggest_param(trial, model_name, 'subsample', 'float', low=0.6, high=1.0)
                if params['bootstrap_type'] == 'Bayesian':
                    params['bagging_temperature'] = self._suggest_param(trial, model_name, 'bagging_temperature', 'float', low=0, high=1)
                if params['grow_policy'] == 'Lossguide':
                    params['max_leaves'] = self._suggest_param(trial, model_name, 'max_leaves', 'int', low=2, high=32)
            else:
                params.update(best_params)

        elif model_name == 'pytorch':
            # Base params that are always present
            params = {
                "loss": model_config['objective'],
                "embedding_info": self.columns.get('embedding_info'),
                "verbose": model_config['verbose'],
                "device": 'cuda' if model_config['device'] == 'gpu' else 'cpu',
                "num_threads": model_config['num_threads']
            }
            
            if trial:
                # Suggest model type first
                model_type = self._suggest_param(trial, model_name, 'model_type', 'categorical', choices=['mlp', 'ft_transformer'])
                params['model_type'] = model_type
                
                params.update({
                    "learning_rate": self._suggest_param(trial, model_name, 'learning_rate', 'categorical', choices=[0.001]),
                    "optimizer_name": self._suggest_param(trial, model_name, 'optimizer_name', 'categorical', choices=['adam']),
                    "batch_size": self._suggest_param(trial, model_name, 'batch_size', 'categorical', choices=[64, 128, 256]),
                    "weight_init": self._suggest_param(trial, model_name, 'weight_init', 'categorical', choices=['default']),
                    "max_epochs": model_config['train_max_epochs'],
                    "patience": model_config['train_patience'],
                })

                if model_type == 'mlp':
                    params["net"] = self._suggest_param(trial, model_name, 'net', 'categorical', choices=[
                        [
                            {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'layer_norm'},
                            {'type': 'dropout', 'p': 0.1},
                            {'type': 'dense', 'out_features': 16, 'activation': 'gelu', 'norm': 'layer_norm'},
                            {'type': 'dropout', 'p': 0.1},
                            {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': 'layer_norm'}
                        ],
                        [
                            {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'batch_norm'},
                            {'type': 'res_block', 'layers': [
                                {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'batch_norm'},
                                {'type': 'dropout', 'p': 0.1}
                            ]},
                            {'type': 'dense', 'out_features': 16, 'activation': 'gelu', 'norm': 'batch_norm'},
                            {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}
                        ]
                    ])
                elif model_type == 'ft_transformer':
                    params["ft_params"] = {
                        'd_token': self._suggest_param(trial, model_name, 'd_token', 'categorical', choices=[64, 128, 192, 256]),
                        'n_layers': self._suggest_param(trial, model_name, 'n_layers', 'int', low=1, high=4),
                        'n_heads': self._suggest_param(trial, model_name, 'n_heads', 'categorical', choices=[4, 8]),
                        'd_ffn_factor': self._suggest_param(trial, model_name, 'd_ffn_factor', 'float', low=1.0, high=2.0),
                        'attention_dropout': self._suggest_param(trial, model_name, 'attention_dropout', 'float', low=0.0, high=0.3),
                        'ffn_dropout': self._suggest_param(trial, model_name, 'ffn_dropout', 'float', low=0.0, high=0.3),
                        'residual_dropout': self._suggest_param(trial, model_name, 'residual_dropout', 'float', low=0.0, high=0.2),
                        'activation': 'reglu',
                        'n_out': 1
                    }

            else:
                params.update(best_params)
                
                # Re-structure for PyTorch class requirements
                if params.get('model_type') == 'ft_transformer':
                    ft_keys = ['d_token', 'n_layers', 'n_heads', 'd_ffn_factor', 
                               'attention_dropout', 'ffn_dropout', 'residual_dropout']
                    ft_params = {k: params.pop(k) for k in ft_keys if k in params}
                    # Add hardcoded defaults that were used in trial but not optimized
                    ft_params['activation'] = 'reglu'
                    ft_params['n_out'] = 1
                    params['ft_params'] = ft_params

                # Override epochs for final training
                params["max_epochs"] = model_config['final_max_epochs']
                params["patience"] = model_config['final_patience']

        elif model_name == 'random_forest':
            params = {
                'verbose': model_config['verbose'],
                'n_jobs': model_config['n_jobs'],
                'random_state': 42
            }
            if trial:
                params.update({
                    'n_estimators': self._suggest_param(trial, model_name, 'n_estimators', 'int', low=100, high=1000),
                    'max_depth': self._suggest_param(trial, model_name, 'max_depth', 'int', low=3, high=30),
                    'min_samples_split': self._suggest_param(trial, model_name, 'min_samples_split', 'int', low=2, high=20),
                    'min_samples_leaf': self._suggest_param(trial, model_name, 'min_samples_leaf', 'int', low=1, high=10),
                    'max_features': self._suggest_param(trial, model_name, 'max_features', 'categorical', choices=['sqrt', 'log2', None]),
                    'bootstrap': self._suggest_param(trial, model_name, 'bootstrap', 'categorical', choices=[True, False])
                })
                if params['bootstrap']:
                    params['max_samples'] = self._suggest_param(trial, model_name, 'max_samples', 'float', low=0.5, high=1.0)
            else:
                params.update(best_params)

        return params

    def _create_model(self, model_name: str, params: Dict[str, Any]):
        task_type = self.config['dataset']['task_type']
        
        if task_type == 'classification':
            if model_name == 'lightgbm': return LGBMClassifier(**params)
            elif model_name == 'xgboost': return XGBClassifier(**params)
            elif model_name == 'catboost': return CatBoostClassifier(**params)
            elif model_name == 'random_forest': return RandomForestClassifier(**params)
            elif model_name == 'pytorch': return PyTorchClassifier(**params)
        else:
            if model_name == 'lightgbm': return LGBMRegressor(**params)
            elif model_name == 'xgboost': return XGBRegressor(**params)
            elif model_name == 'catboost': return CatBoostRegressor(**params)
            elif model_name == 'random_forest': return RandomForestRegressor(**params)
            elif model_name == 'pytorch': return PyTorchRegressor(**params)
        return None

    def _fit_model(self, model, model_name: str, X_train, y_train, X_val, y_val):
        """Helper to fit models with their specific early stopping syntax"""
        
        # Common fit args
        fit_params = {}
        
        if model_name == 'lightgbm':
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'eval_metric': self.config['models']['lightgbm']['params']['eval_metric'],
                'callbacks': [lgb.early_stopping(stopping_rounds=100, verbose=False)]
            }
        elif model_name == 'xgboost':
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'verbose': False
            }
        elif model_name == 'catboost':
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'early_stopping_rounds': 100
            }
        elif model_name == 'pytorch':
            fit_params = {
                'eval_set': [X_val, y_val]
            }
        elif model_name == 'random_forest':
            fit_params = {} # Random Forest doesn't support early stopping in fit

        # Pipeline handling: The model is the last step in a pipeline
        if isinstance(model, Pipeline):
            # We need to pass fit params to the specific step
            # Scikit-learn pipelines accept step_name__param
            step_fit_params = {f"{model_name}__{k}": v for k, v in fit_params.items()}
            model.fit(X_train, y_train, **step_fit_params)
        else:
            model.fit(X_train, y_train, **fit_params)

    def optimize_model(self, model_name: str):
        self.models[model_name] = {}
        
        def objective(trial):
            params = self._get_model_params(model_name, trial)
            
            if self.config['dataset']['verbose'] > 0:
                print(f'===============  {model_name} training - trial {trial.number+1} / {self.config["models"][model_name]["optuna_trials"]}  =========================')
            if self.config['dataset']['verbose'] >= 2:
                print(f'Optune used params:\n{params}')

            split = self.config['dataset']['cv_folds']
            scores = 0
            
            if self.config['dataset']['task_type'] == 'classification':
                skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=self.config['dataset']['random_state'])
            else:
                skf = KFold(n_splits=split, shuffle=True, random_state=self.config['dataset']['random_state'])
            
            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                if self.config['dataset']['verbose'] > 1:
                    print(f'Fold {fold} ...', end="")

                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

                fold_preprocessor = clone(self.preprocessor)
                fold_preprocessor.fit(X_train_fold, y_train_fold)
                
                # Transform validation data for eval_set
                X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))
                
                clf = self._create_model(model_name, params)
                
                pipeline = Pipeline([
                    ('preprocessor', fold_preprocessor),
                    (model_name, clf)
                ])

                try:
                    self._fit_model(
                        pipeline, model_name, 
                        X_train_fold.reset_index(drop=True), 
                        y_train_fold.reset_index(drop=True), 
                        X_val_transformed, 
                        y_val_fold.reset_index(drop=True)
                    )
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as e:
                    print(f"Error in fold {fold}: {e}")
                    raise optuna.exceptions.TrialPruned()

                # Prediction
                if self.config['dataset']['task_type'] == 'classification':
                    y_pred = pipeline.predict_proba(X_val_fold.reset_index(drop=True))[:, 1]
                else:
                    y_pred = pipeline.predict(X_val_fold.reset_index(drop=True))

                metric_func = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])
                s = metric_func(y_val_fold.reset_index(drop=True), y_pred)

                df_y_pred = pd.DataFrame(y_pred, index=y_val_fold.index, columns=[y_val_fold.columns[0]])
                fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}
                
                scores += s
                if self.config['dataset']['verbose'] > 1:
                    print(f" score: {s}")

                trial.report(np.mean(scores / fold), step=fold) # Report average so far
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            avg_score = scores / split
            
            # Update best scores
            is_better = False
            if self.config['dataset']['task_type'] == 'classification':
                if avg_score > self.best_cv_scores[model_name]:
                    is_better = True
            else:
                if avg_score < self.best_cv_scores[model_name]:
                    is_better = True
            
            if is_better:
                self.best_cv_scores[model_name] = avg_score
                self.best_cv_models[model_name] = fitted_trial_models

            if self.config['dataset']['verbose'] > 0:
                print(f"CV {self.config['models'][model_name]['optuna_metric']} Score:", avg_score)

            return avg_score

        # Create Study
        direction = 'maximize' if self.config['dataset']['task_type'] == 'classification' else 'minimize'
        study = optuna.create_study(
            direction=direction, 
            study_name=f'{model_name} Optimization', 
            pruner=self.pruner, 
            sampler=self.sampler
        )
        
        study.optimize(
            objective, 
            n_trials=max(1, self.config['models'][model_name]['optuna_trials']), 
            timeout=self.config['models'][model_name].get('optuna_timeout', None),
            n_jobs=self.config['models'][model_name]['optuna_n_jobs']
        )

        # Store results
        self.models[model_name]['cv_score'] = self.best_cv_scores[model_name]
        self.models[model_name]['cv_model'] = self.best_cv_models[model_name]
        self.models[model_name]['best_params'] = study.best_trial.params
        self.models[model_name]['study'] = study

        # Final Training on Full Data
        print(f'===============  {model_name} Final Training  ============================')
        
        final_params = self._get_model_params(model_name, trial=None)
        
        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        X_test_transformed = final_preprocessor.transform(self.X_test)
        
        clf_final = self._create_model(model_name, final_params)
        
        final_pipeline = Pipeline([
            ('preprocessor', final_preprocessor),
            (model_name, clf_final)
        ])

        self._fit_model(final_pipeline, model_name, self.X_train, self.y_train, X_test_transformed, self.y_test)

        if self.config['dataset']['task_type'] == 'classification':
            final_ypred = final_pipeline.predict_proba(self.X_test)[:, 1]
        else:
            final_ypred = final_pipeline.predict(self.X_test)
            
        final_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, final_ypred)

        self.models[f'final_model_{model_name}'] = {
            'model': final_pipeline,
            'score': final_score
        }
        print(f'===============  {model_name} test score: {final_score} =============')

    def _get_estimator(self, model_name: str):
        """Helper to get estimator instance for stacking"""
        task_type = self.config['dataset']['task_type']
        random_state = 42
        
        if model_name == 'catboost':
            cls = CatBoostClassifier if task_type == 'classification' else CatBoostRegressor
            return cls(random_state=random_state, verbose=0, thread_count=self.config['models']['catboost']['params']['thread_count'])
        elif model_name == 'lightgbm':
            cls = LGBMClassifier if task_type == 'classification' else LGBMRegressor
            return cls(random_state=random_state, verbose=0, num_threads=self.config['models']['lightgbm']['params']['num_threads'])
        elif model_name == 'xgboost':
            cls = XGBClassifier if task_type == 'classification' else XGBRegressor
            return cls(random_state=random_state, verbose=0, nthread=self.config['models']['xgboost']['params']['nthread'])
        elif model_name == 'linear':
            return RidgeClassifier() if task_type == 'classification' else Ridge()
        return None

    def build_ensemble_cv(self):
        if self.config['dataset']['verbose'] > 0 and (self.config['stacking']['cv_enabled'] or self.config['voting']['cv_enabled']):
            print(f'===============  STARTING BUILD ENSEMBLE FROM CV MODELS  ====================')
        
        if self.config['stacking']['cv_enabled']:
            if self.config['dataset']['verbose'] > 0:
                print(f'Build Stacking from CV models')
            self._build_stacking(use_cv_models=True)
        
        if self.config['voting']['cv_enabled']:
            if self.config['dataset']['verbose'] > 0:
                print(f'Build Voting from CV models')
            self._build_voting(use_cv_models=True)

    def build_ensemble_final(self):
        if self.config['dataset']['verbose'] > 0 and (self.config['stacking']['final_enabled'] or self.config['voting']['final_enabled']):
            print(f'===============  STARTING BUILD ENSEMBLE FROM FINAL MODELS  ====================')
        
        if self.config['stacking']['final_enabled']:
            if self.config['dataset']['verbose'] > 0:
                print(f'Build Stacking from final models')
            self._build_stacking(use_cv_models=False)
        
        if self.config['voting']['final_enabled']:
            if self.config['dataset']['verbose'] > 0:
                print(f'Build Voting from final models')
            self._build_voting(use_cv_models=False)

    def _build_stacking(self, use_cv_models: bool):
        suffix = "cv" if use_cv_models else "final"
        meta_model_name = self.config['stacking']['meta_model']
        final_estimator = self._get_estimator(meta_model_name)
        
        base_estimators = []
        for model_name in self.config['models']:
            if self.config['models'][model_name]['enabled']:
                if use_cv_models:
                    # Add all fold models
                    for i in range(len(self.models[model_name]['cv_model'])):
                        model = self.models[model_name]['cv_model'][f'fold{i+1}']['model']
                        if model_name == 'xgboost':
                            model.named_steps[model_name].set_params(early_stopping_rounds=None)
                        base_estimators.append((f"{model_name}_{i+1}", model))
                else:
                    # Add final model
                    model = self.models[f'final_model_{model_name}']['model']
                    if model_name == 'xgboost':
                        model.named_steps[model_name].set_params(early_stopping_rounds=None)
                    base_estimators.append((model_name, model))

        cls = StackingClassifier if self.config['dataset']['task_type'] == 'classification' else StackingRegressor
        st = cls(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=self.config['stacking']['cv_folds'] if not self.config['stacking']['prefit'] else 'prefit',
            n_jobs=-1
        )
        
        print(f"Training Stacking ({suffix})...")
        st.fit(self.X_train, self.y_train)
        
        if self.config['dataset']['task_type'] == 'classification':
            y_pred = st.predict_proba(self.X_test)[:, 1]
        else:
            y_pred = st.predict(self.X_test)
            
        # Use metric from first enabled model as reference
        ref_model = next(iter(self.config['models']))
        score = self.get_evaluation_metric(self.config['models'][ref_model]['optuna_metric'])(self.y_test, y_pred)
        
        self.models[f'ensemble_stacking_{suffix}'] = {'model': st, 'score': score}
        print(f'Stacking {suffix} score: {score}')

    def _build_voting(self, use_cv_models: bool):
        suffix = "cv" if use_cv_models else "final"
        base_estimators = []
        
        for model_name in self.config['models']:
            if self.config['models'][model_name]['enabled']:
                if use_cv_models:
                    for i in range(len(self.models[model_name]['cv_model'])):
                        base_estimators.append(
                            (f"{model_name}_{i+1}", self.models[model_name]['cv_model'][f'fold{i+1}']['model'])
                        )
                else:
                    base_estimators.append(
                        (model_name, self.models[f'final_model_{model_name}']['model'])
                    )

        cls = VotingClassifier if self.config['dataset']['task_type'] == 'classification' else VotingRegressor
        kwargs = {'voting': 'soft'} if self.config['dataset']['task_type'] == 'classification' else {}
        
        vt = cls(estimators=base_estimators, n_jobs=-1, **kwargs)
        
        print(f"Training Voting ({suffix})...")
        if not self.config['voting']['prefit']:
            vt.fit(self.X_train, self.y_train)
        else:
            # If prefit, we need to manually set estimators_ (sklearn doesn't support prefit=True in constructor for Voting)
            # Note: VotingClassifier/Regressor doesn't strictly support 'prefit' in the same way Stacking does in its init.
            # Usually you just fit it. If you want to use pre-trained models, you just fit it and it will re-fit them unless you hack it.
            # However, for Voting, fitting just means fitting the sub-estimators. 
            # If they are already fitted (and we want to keep them), we can't easily use VotingRegressor's fit.
            # But since the original code had logic for this, I'll preserve the intent:
            # "vt.estimators_ = base_estimators"
            # This relies on the list being just models, not (name, model) tuples for estimators_ attribute
            vt.estimators_ = [m for _, m in base_estimators]
            # We also need to populate named_estimators_
            vt.named_estimators_ = dict(base_estimators)
            
        if self.config['dataset']['task_type'] == 'classification':
            y_pred = vt.predict_proba(self.X_test)[:, 1]
        else:
            y_pred = vt.predict(self.X_test)
            
        ref_model = next(iter(self.config['models']))
        score = self.get_evaluation_metric(self.config['models'][ref_model]['optuna_metric'])(self.y_test, y_pred)
        
        self.models[f'ensemble_voting_{suffix}'] = {'model': vt, 'score': score}
        print(f'Voting {suffix} score: {score}')
