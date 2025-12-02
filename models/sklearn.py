from typing import Any, Dict, Optional
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import BaseModelAdapter

class RandomForestAdapter(BaseModelAdapter):
    def get_params(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        params = {
            'verbose': self.model_config['params']['verbose'],
            'n_jobs': self.model_config['params']['n_jobs'],
            'random_state': 42
        }
        
        if trial:
            params.update({
                'n_estimators': self._suggest_param(trial, 'n_estimators', 'int', low=100, high=1000),
                'max_depth': self._suggest_param(trial, 'max_depth', 'int', low=3, high=30),
                'min_samples_split': self._suggest_param(trial, 'min_samples_split', 'int', low=2, high=20),
                'min_samples_leaf': self._suggest_param(trial, 'min_samples_leaf', 'int', low=1, high=10),
                'max_features': self._suggest_param(trial, 'max_features', 'categorical', choices=['sqrt', 'log2', None]),
                'bootstrap': self._suggest_param(trial, 'bootstrap', 'categorical', choices=[True, False])
            })
            if params['bootstrap']:
                params['max_samples'] = self._suggest_param(trial, 'max_samples', 'float', low=0.5, high=1.0)
        else:
            best_params = self.model_config.get('best_params', {})
            params.update(best_params)
            
        return params

    def create_model(self, params: Dict[str, Any]) -> Any:
        if self.task_type == 'classification':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)

    def get_fit_params(self, X_val, y_val) -> Dict[str, Any]:
        return {}
