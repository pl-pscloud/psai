from typing import Any, Dict, Optional
import optuna
from xgboost import XGBClassifier, XGBRegressor
from .base import BaseModelAdapter

class XGBoostAdapter(BaseModelAdapter):
    def get_params(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        params = {
            "objective": self.model_config['params']['objective'],
            "eval_metric": self.model_config['params']['eval_metric'],
            "device": 'cuda' if self.model_config['params']['device'] == 'gpu' else 'cpu',
            "early_stopping_rounds": 100,
            "nthread": self.model_config['params']['nthread'],
            "verbose": self.model_config['params']['verbose']
        }
        
        if trial:
            params.update({
                "booster": self._suggest_param(trial, 'booster', 'categorical', choices=['gbtree']),
                "max_depth": self._suggest_param(trial, 'max_depth', 'int', low=3, high=20),
                "learning_rate": self._suggest_param(trial, 'learning_rate', 'float', low=0.001, high=0.2, log=True),
                "n_estimators": self._suggest_param(trial, 'n_estimators', 'int', low=500, high=3000),
                "subsample": self._suggest_param(trial, 'subsample', 'float', low=0, high=1),
                "lambda": self._suggest_param(trial, 'lambda', 'float', low=1e-4, high=5, log=True),
                "gamma": self._suggest_param(trial, 'gamma', 'float', low=1e-4, high=5, log=True),
                "alpha": self._suggest_param(trial, 'alpha', 'float', low=1e-4, high=5, log=True),
                "min_child_weight": self._suggest_param(trial, 'min_child_weight', 'categorical', choices=[0.5, 1, 3, 5]),
                "colsample_bytree": self._suggest_param(trial, 'colsample_bytree', 'float', low=0.5, high=1),
                "colsample_bylevel": self._suggest_param(trial, 'colsample_bylevel', 'float', low=0.5, high=1),
            })
        else:
            best_params = self.model_config.get('best_params', {})
            params.update(best_params)
            
        return params

    def create_model(self, params: Dict[str, Any]) -> Any:
        if self.task_type == 'classification':
            return XGBClassifier(**params)
        else:
            return XGBRegressor(**params)

    def get_fit_params(self, X_val, y_val) -> Dict[str, Any]:
        return {
            'eval_set': [(X_val, y_val)],
            'verbose': False
        }
