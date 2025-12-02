from typing import Any, Dict, Optional
import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from .base import BaseModelAdapter

class LightGBMAdapter(BaseModelAdapter):
    def get_params(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        params = {
            "objective": self.model_config['params']['objective'],
            "device": self.model_config['params']['device'],
            "metric": self.model_config['params']['eval_metric'], 
            "verbosity": -1,
            "n_estimators": 10000,
            "num_threads": self.model_config['params']['num_threads'],
            "verbose": self.model_config['params']['verbose']
        }
        
        if trial:
            params.update({
                "boosting_type": self._suggest_param(trial, 'boosting_type', 'categorical', choices=['gbdt', 'goss']),
                "lambda_l1": self._suggest_param(trial, "lambda_l1", 'float', low=1e-8, high=10.0, log=True),
                "lambda_l2": self._suggest_param(trial, "lambda_l2", 'float', low=1e-8, high=10.0, log=True),
                "num_leaves": self._suggest_param(trial, "num_leaves", 'int', low=20, high=300),
                "feature_fraction": self._suggest_param(trial, "feature_fraction", 'float', low=0.4, high=1.0),
                "min_child_samples": self._suggest_param(trial, "min_child_samples", 'int', low=5, high=100),
                "learning_rate": self._suggest_param(trial, "learning_rate", 'float', low=0.001, high=0.2, log=True),
                "min_split_gain": self._suggest_param(trial, "min_split_gain", 'float', low=1e-8, high=1.0, log=True),
                "max_depth": self._suggest_param(trial, "max_depth", 'int', low=3, high=20),
            })
            if params['boosting_type'] != 'goss':
                params["bagging_fraction"] = self._suggest_param(trial, "bagging_fraction", 'float', low=0.4, high=1.0)
                params["bagging_freq"] = self._suggest_param(trial, "bagging_freq", 'int', low=1, high=7)
        else:
            best_params = self.model_config.get('best_params', {})
            params.update(best_params)
            
        return params

    def create_model(self, params: Dict[str, Any]) -> Any:
        if self.task_type == 'classification':
            return LGBMClassifier(**params)
        else:
            return LGBMRegressor(**params)

    def get_fit_params(self, X_val, y_val) -> Dict[str, Any]:
        return {
            'eval_set': [(X_val, y_val)],
            'eval_metric': self.model_config['params']['eval_metric'],
            'callbacks': [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        }
