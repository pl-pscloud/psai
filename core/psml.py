import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import pickle
import os
import logging
import lightgbm as lgb
from typing import Dict, Any, Optional, Union, List, Callable, Tuple
import time

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, f1_score, roc_auc_score, mean_squared_log_error, root_mean_squared_log_error,
    precision_score, recall_score
)
from sklearn.base import clone

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from psai.core.scalersencoders import create_preprocessor
from psai.core.pstorch import PyTorchRegressor, PyTorchClassifier
from psai.models.lightgbm import LightGBMAdapter
from psai.models.xgboost import XGBoostAdapter
from psai.models.catboost import CatBoostAdapter
from psai.models.sklearn import RandomForestAdapter
from psai.models.pytorch import PyTorchAdapter

try:
    from scipy.optimize import BracketError
except ImportError:
    try:
        from scipy.optimize._optimize import BracketError
    except ImportError:
        class BracketError(Exception): pass

class psML:
    """
    Machine Learning Pipeline for training, evaluation, and prediction
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, X: Optional[pd.DataFrame] = None, y: Optional[Union[pd.Series, pd.DataFrame]] = None, experiment_name: Optional[str] = None):
        self.models: Dict[str, Any] = {}
        
        if config is None:
            try:
                from .config import CONFIG
                self.config = CONFIG
            except ImportError:
                logger.error("Config not provided and config.py not found.")
                raise ValueError("Config not provided and config.py not found.")
        else:
            self.config = config
        
        # Read and split data
        if X is None or y is None:

            train_path = self.config['dataset']['train_path']
            if not os.path.exists(train_path):
                logger.error(f"Training data file not found at {train_path}")
                raise FileNotFoundError(f"Training data file not found at {train_path}")
            df = pd.read_csv(train_path)
            target = self.config['dataset']['target']
            if target not in df.columns:
                logger.error(f"Target column '{target}' not found in the dataset.")
                raise ValueError(f"Target column '{target}' not found in the dataset.")
            X = df.drop(columns=[target])
            y = df[[target]]

        if isinstance(y, pd.Series):
            y = y.to_frame()

        if self.config['dataset']['task_type'] == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=self.config['dataset']['test_size'], 
                random_state=self.config['dataset']['random_state'],
                stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=self.config['dataset']['test_size'], 
                random_state=self.config['dataset']['random_state']
            )

        self.columns = {}
        self.preprocessor = None

        # Determine if multiclass
        self.is_multiclass = False
        self.le = None
        if self.config['dataset']['task_type'] == 'classification':
             if self.y_train.iloc[:, 0].nunique() > 2:
                 self.is_multiclass = True
             
             # Encode target
             from sklearn.preprocessing import LabelEncoder
             self.le = LabelEncoder()
             self.y_train.iloc[:, 0] = self.le.fit_transform(self.y_train.iloc[:, 0])
             self.y_test.iloc[:, 0] = self.le.transform(self.y_test.iloc[:, 0])

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

        self.mlflow_experiment_name = None
        self.mlflow_experiment_id = None

        if experiment_name or self.config.get('mlflow', {}).get('enabled', True):

            # MLflow setup
            if experiment_name:
                self.mlflow_experiment_name = experiment_name
            elif self.config.get('mlflow', {}).get('enabled', True):
                self.mlflow_experiment_name = self.config.get('mlflow', {}).get('experiment_name', f"psai_{time.strftime('%Y%m%d%H%M%S')}")
            
            self.mlflow_tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', None)
            
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            if self.mlflow_experiment_name:
                mlflow.set_experiment(self.mlflow_experiment_name)
                self.mlflow_experiment_id = mlflow.get_experiment_by_name(self.mlflow_experiment_name).experiment_id

        # Initialize adapters
        self.adapters = {
            'lightgbm': LightGBMAdapter(self.config['models']['lightgbm'], self.config),
            'xgboost': XGBoostAdapter(self.config['models']['xgboost'], self.config),
            'catboost': CatBoostAdapter(self.config['models']['catboost'], self.config),
            'random_forest': RandomForestAdapter(self.config['models']['random_forest'], self.config),
            'pytorch': PyTorchAdapter(self.config['models']['pytorch'], self.config)
        }
        
        # Set multiclass for adapters
        for adapter in self.adapters.values():
            adapter.set_multiclass(self.is_multiclass)
            if hasattr(adapter, 'set_n_classes') and self.is_multiclass:
                 adapter.set_n_classes(self.y_train.iloc[:, 0].nunique())

    def save(self, filepath: str) -> None:
        """Save the pSML object to a file using pickle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model successfully saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'psML':
        """Load a pSML object from a file"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model successfully loaded from {filepath}")
        return model

    def optimize_all_models(self) -> None:
        self.create_preprocessor()
        
        # List of supported models
        supported_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch']
        
        for model_name in supported_models:
            if self.config['models'].get(model_name, {}).get('enabled', False):
                self.optimize_model(model_name)

    def create_preprocessor(self) -> None:
        preprocessor, columns = create_preprocessor(self.config['preprocessor'], self.X_train)
        self.preprocessor = preprocessor
        self.columns = columns

    def scores(self, return_json: bool = False) -> Union[Dict[str, Any], str]:
        results = {}
        for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch']:
            if self.config['models'].get(model_name, {}).get('enabled', False):
                cv_score = self.models.get(model_name, {}).get("cv_score", "N/A")
                test_score = self.models.get(f"final_model_{model_name}", {}).get("score", "N/A")
                results[model_name] = {
                    "cv_score": cv_score,
                    "test_score": test_score
                }
                logger.info(f'{model_name.capitalize()} CV Score: {cv_score}, Test Score: {test_score}')
        
        # Add ensemble scores
        for ensemble_name in ['ensemble_stacking_cv', 'ensemble_stacking_final', 'ensemble_voting_cv', 'ensemble_voting_final']:
            if ensemble_name in self.models:
                score = self.models[ensemble_name].get('score', "N/A")
                results[ensemble_name] = {
                    "score": score
                }
                logger.info(f'{ensemble_name.replace("_", " ").title()} Score: {score}')
        
        if return_json:
            return json.dumps(results)

        return results

    def rmsle_safe(self, y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
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

    def get_evaluation_metric(self, metric_name: str) -> Callable:
        metric_mapping = {
            'acc': lambda y_true, y_pred: accuracy_score(y_true, np.argmax(y_pred, axis=1)) if self.is_multiclass else accuracy_score(y_true, (y_pred >= 0.5).astype(int)),
            'f1': lambda y_true, y_pred: f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted') if self.is_multiclass else f1_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
            'auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted') if self.is_multiclass else roc_auc_score(y_true, y_pred),
            'prec': lambda y_true, y_pred: precision_score(y_true, np.argmax(y_pred, axis=1), average='weighted') if self.is_multiclass else precision_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
            'rec': lambda y_true, y_pred: recall_score(y_true, np.argmax(y_pred, axis=1), average='weighted') if self.is_multiclass else recall_score(y_true, (y_pred >= 0.5).astype(int), average='binary'),
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
            logger.warning(f"Metric '{metric_name}' not found.")
            if self.config['dataset']['task_type'] == 'regression':
                logger.warning("Defaulting to RMSE for regression.")
                return lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            else:
                logger.warning("Defaulting to roc_auc_score for classification.")
                return roc_auc_score
        
        return metric_mapping[metric_name]

    def _fit_model(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame], X_val: pd.DataFrame, y_val: Union[pd.Series, pd.DataFrame]) -> None:
        """Helper to fit models with their specific early stopping syntax using adapters"""
        adapter = self.adapters[model_name]
        fit_params = adapter.get_fit_params(X_val, y_val)
        
        # Pipeline handling: The model is the last step in a pipeline
        if isinstance(model, Pipeline):
            # We need to pass fit params to the specific step
            # Scikit-learn pipelines accept step_name__param
            step_fit_params = {f"{model_name}__{k}": v for k, v in fit_params.items()}
            model.fit(X_train, y_train, **step_fit_params)
        else:
            model.fit(X_train, y_train, **fit_params)

    def optimize_model(self, model_name: str):
        if self.preprocessor is None:
            self.create_preprocessor()

        self.models[model_name] = {}
        adapter = self.adapters[model_name]

        # Update adapter with embedding info for PyTorch if needed
        if model_name == 'pytorch':
             if hasattr(adapter, 'set_embedding_info'):
                 adapter.set_embedding_info(self.columns.get('embedding_info'))
        
        def objective(trial):
            params = adapter.get_params(trial)
            
            if self.config['dataset']['verbose'] > 0:
                logger.info(f'===============  {model_name} training - trial {trial.number+1} / {self.config["models"][model_name]["optuna_trials"]}  =========================')
            if self.config['dataset']['verbose'] >= 2:
                logger.info(f'Optuna used params:\n{params}')

            split = self.config['dataset']['cv_folds']
            scores = 0
            
            if self.config['dataset']['task_type'] == 'classification':
                skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=self.config['dataset']['random_state'])
            else:
                skf = KFold(n_splits=split, shuffle=True, random_state=self.config['dataset']['random_state'])
            
            fitted_trial_models = {}

            for fold, (train_index, val_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
                try:
                    if self.config['dataset']['verbose'] > 1:
                        logger.info(f'Fold {fold} ...')

                    X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                    y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

                    fold_preprocessor = clone(self.preprocessor)
                    fold_preprocessor.fit(X_train_fold, y_train_fold)
                    
                    # Transform validation data for eval_set
                    X_val_transformed = fold_preprocessor.transform(X_val_fold.reset_index(drop=True))
                    
                    clf = adapter.create_model(params)
                    
                    pipeline = Pipeline([
                        ('preprocessor', fold_preprocessor),
                        (model_name, clf)
                    ])

                    self._fit_model(
                        pipeline, model_name, 
                        X_train_fold.reset_index(drop=True), 
                        y_train_fold.reset_index(drop=True), 
                        X_val_transformed, 
                        y_val_fold.reset_index(drop=True)
                    )

                    # Prediction
                    if self.config['dataset']['task_type'] == 'classification':
                        if self.is_multiclass:
                            y_pred = pipeline.predict_proba(X_val_fold.reset_index(drop=True))
                        else:
                            y_pred = pipeline.predict_proba(X_val_fold.reset_index(drop=True))[:, 1]
                    else:
                        y_pred = pipeline.predict(X_val_fold.reset_index(drop=True))

                    metric_func = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])
                    s = metric_func(y_val_fold.reset_index(drop=True), y_pred)

                    if self.is_multiclass:
                        cols = [f"{y_val_fold.columns[0]}_{i}" for i in range(y_pred.shape[1])]
                        df_y_pred = pd.DataFrame(y_pred, index=y_val_fold.index, columns=cols)
                    else:
                        df_y_pred = pd.DataFrame(y_pred, index=y_val_fold.index, columns=[y_val_fold.columns[0]])
                    fitted_trial_models[f'fold{fold}'] = {'model': pipeline, 'oof': df_y_pred}
                    
                    scores += s
                    if self.config['dataset']['verbose'] > 1:
                        logger.info(f" score: {s}")

                    trial.report(np.mean(scores / fold), step=fold) # Report average so far
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                except optuna.exceptions.TrialPruned:
                    raise
                except BracketError:
                    logger.warning(f"BracketError in fold {fold}. Pruning trial.")
                    raise optuna.exceptions.TrialPruned()
                except Exception as e:
                    logger.error(f"Error in fold {fold}: {e}")
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
                logger.info(f"CV {self.config['models'][model_name]['optuna_metric']} Score: {avg_score}")

            return avg_score

        # Create Study
        direction = 'maximize' if self.config['dataset']['task_type'] == 'classification' else 'minimize'
        study = optuna.create_study(
            direction=direction, 
            study_name=f'{model_name} Optimization', 
            pruner=self.pruner, 
            sampler=self.sampler
        )
        
        # Custom MLflow Callback for Optuna
        callbacks = []
        if self.mlflow_experiment_name or self.config.get('mlflow', {}).get('enabled', True):
            def logging_callback_mlflow(study, frozen_trial):
                with mlflow.start_run(run_name=f"{model_name}_{frozen_trial.number}", experiment_id=self.mlflow_experiment_id, tags={"model_name": model_name}):
                    mlflow.log_params(frozen_trial.params)
                    if frozen_trial.value is not None:
                        mlflow.log_metric(self.config['models'][model_name]['optuna_metric'], frozen_trial.value)
                    mlflow.set_tag("state", frozen_trial.state.name)
            callbacks.append(logging_callback_mlflow)

        study.optimize(
            objective, 
            n_trials=max(1, self.config['models'][model_name]['optuna_trials']), 
            timeout=self.config['models'][model_name].get('optuna_timeout', None),
            n_jobs=self.config['models'][model_name]['optuna_n_jobs'],
            callbacks=callbacks
        )

        # Store results
        try:
            self.models[model_name]['best_params'] = study.best_trial.params
        except ValueError:
            logger.warning(f"No trials completed for {model_name}. Disabling model.")
            self.config['models'][model_name]['enabled'] = False
            return

        self.models[model_name]['cv_score'] = self.best_cv_scores[model_name]
        self.models[model_name]['cv_model'] = self.best_cv_models[model_name]
        self.models[model_name]['study'] = study

        # Final Training on Full Data
        logger.info(f'===============  {model_name} Final Training  ============================')
        
        final_params = adapter.get_params(trial=None)
        
        final_preprocessor = clone(self.preprocessor)
        final_preprocessor.fit(self.X_train, self.y_train)
        X_test_transformed = final_preprocessor.transform(self.X_test)
        
        clf_final = adapter.create_model(final_params)
        
        final_pipeline = Pipeline([
            ('preprocessor', final_preprocessor),
            (model_name, clf_final)
        ])

        self._fit_model(final_pipeline, model_name, self.X_train, self.y_train, X_test_transformed, self.y_test)

        if self.config['dataset']['task_type'] == 'classification':
            if self.is_multiclass:
                final_ypred = final_pipeline.predict_proba(self.X_test)
            else:
                final_ypred = final_pipeline.predict_proba(self.X_test)[:, 1]
        else:
            final_ypred = final_pipeline.predict(self.X_test)
            
        final_score = self.get_evaluation_metric(self.config['models'][model_name]['optuna_metric'])(self.y_test, final_ypred)

        self.models[f'final_model_{model_name}'] = {
            'model': final_pipeline,
            'score': final_score
        }
        logger.info(f'===============  {model_name} test score: {final_score} =============')

        # Log final model to MLflow
        if self.mlflow_experiment_name or self.config.get('mlflow', {}).get('enabled', True):
            with mlflow.start_run(run_name=f"final_{model_name}", experiment_id=self.mlflow_experiment_id, tags={"model_name": model_name}):
                mlflow.log_params(final_params)
                mlflow.log_metric(self.config['models'][model_name]['optuna_metric'], final_score)
                mlflow.log_metric("cv_score", self.models[model_name]['cv_score'])
            
                # Log model artifact
                try:
                    if model_name == 'lightgbm':
                        mlflow.lightgbm.log_model(clf_final, model_name)
                    elif model_name == 'xgboost':
                        mlflow.xgboost.log_model(clf_final, model_name)
                    elif model_name == 'catboost':
                        mlflow.catboost.log_model(clf_final, model_name)
                    elif model_name == 'random_forest':
                        mlflow.sklearn.log_model(clf_final, model_name)
                    elif model_name == 'pytorch':
                        mlflow.pytorch.log_model(clf_final, model_name)
                except Exception as e:
                    logger.warning(f"Failed to log model to MLflow: {e}")

    def _get_estimator(self, model_name: str) -> Optional[Any]:
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

    def build_ensemble_cv(self) -> None:
        if self.config['dataset']['verbose'] > 0 and (self.config['stacking']['cv_enabled'] or self.config['voting']['cv_enabled']):
            logger.info(f'===============  STARTING BUILD ENSEMBLE FROM CV MODELS  ====================')
        
        if self.config['stacking']['cv_enabled']:
            if self.config['dataset']['verbose'] > 0:
                logger.info(f'Build Stacking from CV models')
            self._build_stacking(use_cv_models=True)
        
        if self.config['voting']['cv_enabled']:
            if self.config['dataset']['verbose'] > 0:
                logger.info(f'Build Voting from CV models')
            self._build_voting(use_cv_models=True)

    def build_ensemble_final(self) -> None:
        if self.config['dataset']['verbose'] > 0 and (self.config['stacking']['final_enabled'] or self.config['voting']['final_enabled']):
            logger.info(f'\n===============  STARTING BUILD ENSEMBLE FROM FINAL MODELS  ====================\n')
        
        if self.config['stacking']['final_enabled']:
            if self.config['dataset']['verbose'] > 0:
                logger.info(f'Build Stacking from final models')
            self._build_stacking(use_cv_models=False)
        
        if self.config['voting']['final_enabled']:
            if self.config['dataset']['verbose'] > 0:
                logger.info(f'Build Voting from final models')
            self._build_voting(use_cv_models=False)

    def _build_stacking(self, use_cv_models: bool) -> None:
        suffix = "cv" if use_cv_models else "final"
        meta_model_name = self.config['stacking']['meta_model']
        final_estimator = self._get_estimator(meta_model_name)
        
        base_estimators = []
        # Determine which models to use based on config
        if use_cv_models:
            models_to_use = self.config['stacking'].get('cv_models', [])
            # Fallback if empty or not present: use all enabled models
            if not models_to_use:
                models_to_use = [m for m in self.config['models'] if self.config['models'][m]['enabled']]
        else:
            models_to_use = self.config['stacking'].get('final_models', [])
            # Fallback if empty or not present: use all enabled models
            if not models_to_use:
                models_to_use = [m for m in self.config['models'] if self.config['models'][m]['enabled']]

        for model_name in models_to_use:
            # Ensure model is enabled in general config and exists in self.models
            if self.config['models'].get(model_name, {}).get('enabled', False) and model_name in self.models:
                if use_cv_models:
                    # Add all fold models
                    if 'cv_model' in self.models[model_name] and self.models[model_name]['cv_model']:
                        for i in range(len(self.models[model_name]['cv_model'])):
                            fold_key = f'fold{i+1}'
                            if fold_key in self.models[model_name]['cv_model']:
                                model = self.models[model_name]['cv_model'][fold_key]['model']
                                if model_name == 'xgboost':
                                    model.named_steps[model_name].set_params(early_stopping_rounds=None)
                                base_estimators.append((f"{model_name}_{i+1}", model))
                else:
                    # Add final model
                    final_key = f'final_model_{model_name}'
                    if final_key in self.models:
                        model = self.models[final_key]['model']
                        if model_name == 'xgboost':
                            model.named_steps[model_name].set_params(early_stopping_rounds=None)
                        base_estimators.append((model_name, model))
        
        if self.config['dataset']['verbose'] > 1:
            logger.info(f"Base estimators: {base_estimators}")


        cls = StackingClassifier if self.config['dataset']['task_type'] == 'classification' else StackingRegressor
        st = cls(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=self.config['stacking']['cv_folds'] if not self.config['stacking']['prefit'] else 'prefit',
            n_jobs=-1,
            passthrough=self.config['stacking']['use_features']
        )
        
        print(f"Training Stacking ({suffix})...")
        st.fit(self.X_train, self.y_train)
        
        if self.config['dataset']['task_type'] == 'classification':
            if self.is_multiclass:
                y_pred = st.predict_proba(self.X_test)
            else:
                y_pred = st.predict_proba(self.X_test)[:, 1]
        else:
            y_pred = st.predict(self.X_test)
            
        # Use metric from first enabled model as reference
        ref_model = next(iter(self.config['models']))
        score = self.get_evaluation_metric(self.config['models'][ref_model]['optuna_metric'])(self.y_test, y_pred)
        
        self.models[f'ensemble_stacking_{suffix}'] = {'model': st, 'score': score}
        logger.info(f'Stacking {suffix} score: {score}')

        # Log stacking model to MLflow
        if self.mlflow_experiment_name or self.config.get('mlflow', {}).get('enabled', True):
            with mlflow.start_run(run_name=f"stacking_{suffix}", experiment_id=self.mlflow_experiment_id, tags={"model_name": f"stacking_{suffix}"}):
                mlflow.log_metric(self.config['models'][ref_model]['optuna_metric'], score)
                
                try:
                    mlflow.sklearn.log_model(st, f"stacking_{suffix}")
                except Exception as e:
                    logger.warning(f"Failed to log stacking model to MLflow: {e}")

    def _build_voting(self, use_cv_models: bool) -> None:
        suffix = "cv" if use_cv_models else "final"
        base_estimators = []
        
        # Determine which models to use based on config
        if use_cv_models:
            models_to_use = self.config['voting'].get('cv_models', [])
            # Fallback if empty or not present: use all enabled models
            if not models_to_use:
                models_to_use = [m for m in self.config['models'] if self.config['models'][m]['enabled']]
        else:
            models_to_use = self.config['voting'].get('final_models', [])
            # Fallback if empty or not present: use all enabled models
            if not models_to_use:
                models_to_use = [m for m in self.config['models'] if self.config['models'][m]['enabled']]

        for model_name in models_to_use:
            # Ensure model is enabled in general config and exists in self.models
            if self.config['models'].get(model_name, {}).get('enabled', False) and model_name in self.models:
                if use_cv_models:
                    if 'cv_model' in self.models[model_name] and self.models[model_name]['cv_model']:
                        for i in range(len(self.models[model_name]['cv_model'])):
                            fold_key = f'fold{i+1}'
                            if fold_key in self.models[model_name]['cv_model']:
                                base_estimators.append(
                                    (f"{model_name}_{i+1}", self.models[model_name]['cv_model'][fold_key]['model'])
                                )
                else:
                    final_key = f'final_model_{model_name}'
                    if final_key in self.models:
                        base_estimators.append(
                            (model_name, self.models[final_key]['model'])
                        )

        if self.config['dataset']['verbose'] > 1:
            logger.info(f"Base estimators: {base_estimators}")

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
            if self.is_multiclass:
                y_pred = vt.predict_proba(self.X_test)
            else:
                y_pred = vt.predict_proba(self.X_test)[:, 1]
        else:
            y_pred = vt.predict(self.X_test)
            
        ref_model = next(iter(self.config['models']))
        score = self.get_evaluation_metric(self.config['models'][ref_model]['optuna_metric'])(self.y_test, y_pred)
        
        self.models[f'ensemble_voting_{suffix}'] = {'model': vt, 'score': score}
        logger.info(f'Voting {suffix} score: {score}')

        # Log voting model to MLflow
        if self.mlflow_experiment_name or self.config.get('mlflow', {}).get('enabled', True):
            with mlflow.start_run(run_name=f"voting_{suffix}", experiment_id=self.mlflow_experiment_id, tags={"model_name": f"voting_{suffix}"}):
                mlflow.log_metric(self.config['models'][ref_model]['optuna_metric'], score)
                
                try:
                    mlflow.sklearn.log_model(vt, f"voting_{suffix}")
                except Exception as e:
                    logger.warning(f"Failed to log voting model to MLflow: {e}")
