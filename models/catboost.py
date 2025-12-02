from typing import Any, Dict, Optional
import optuna
import logging
from catboost import CatBoostClassifier, CatBoostRegressor
from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

class CatBoostAdapter(BaseModelAdapter):
    def get_params(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        params = {
            'loss_function': self.model_config['params']['objective'],
            'eval_metric': self.model_config['params']['eval_metric'],
            'task_type': 'CPU' if self.model_config['params']['device'] == 'cpu' else 'GPU',
            'random_seed': 42,
            'verbose': self.model_config['params']['verbose'],
            'thread_count': self.model_config['params']['thread_count']
        }
        
        if trial:
            params.update({
                'n_estimators': self._suggest_param(trial, 'n_estimators', 'int', low=100, high=3000),
                'learning_rate': self._suggest_param(trial, 'learning_rate', 'float', low=0.001, high=0.2, log=True),
                'depth': self._suggest_param(trial, 'depth', 'int', low=4, high=10),
                'l2_leaf_reg': self._suggest_param(trial, 'l2_leaf_reg', 'float', low=1e-3, high=10.0, log=True),
                'border_count': self._suggest_param(trial, 'border_count', 'int', low=32, high=128),
                'bootstrap_type': self._suggest_param(trial, 'bootstrap_type', 'categorical', choices=['Bayesian', 'Bernoulli', 'MVS']),
                'feature_border_type': self._suggest_param(trial, 'feature_border_type', 'categorical', choices=['Median', 'Uniform', 'GreedyMinEntropy']),
                'leaf_estimation_iterations': self._suggest_param(trial, 'leaf_estimation_iterations', 'int', low=1, high=10),
                'min_data_in_leaf': self._suggest_param(trial, 'min_data_in_leaf', 'int', low=1, high=30),
                'random_strength': self._suggest_param(trial, 'random_strength', 'float', low=1e-9, high=10, log=True),
                'grow_policy': self._suggest_param(trial, 'grow_policy', 'categorical', choices=['SymmetricTree', 'Depthwise', 'Lossguide']),
            })
            
            if params['bootstrap_type'] == 'MVS' and params['task_type'] == 'GPU':
                logger.info("GPU choose must switch Bootstrap type to Bayesian")
                params['bootstrap_type'] = 'Bayesian'
            if params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = self._suggest_param(trial, 'subsample', 'float', low=0.6, high=1.0)
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = self._suggest_param(trial, 'bagging_temperature', 'float', low=0, high=1)
            if params['grow_policy'] == 'Lossguide':
                params['max_leaves'] = self._suggest_param(trial, 'max_leaves', 'int', low=2, high=32)
        else:
            best_params = self.model_config.get('best_params', {})
            params.update(best_params)
            if params.get('bootstrap_type') == 'MVS' and params.get('task_type') == 'GPU':
                logger.info("GPU choose must switch Bootstrap type to Bayesian")
                params['bootstrap_type'] = 'Bayesian'
                
        return params

    def create_model(self, params: Dict[str, Any]) -> Any:
        if self.task_type == 'classification':
            return CatBoostClassifier(**params)
        else:
            return CatBoostRegressor(**params)

    def get_fit_params(self, X_val, y_val) -> Dict[str, Any]:
        return {
            'eval_set': [(X_val, y_val)],
            'early_stopping_rounds': 100
        }
