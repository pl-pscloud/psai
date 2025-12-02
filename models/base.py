from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import numpy as np
import optuna
import logging

logger = logging.getLogger(__name__)

class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters in psML.
    """
    def __init__(self, model_config: Dict[str, Any], global_config: Dict[str, Any]):
        self.model_config = model_config
        self.global_config = global_config
        self.task_type = global_config['dataset']['task_type']
        self.is_multiclass = False
        # We might need to know if it's multiclass, which usually comes from y_train check in psML
        # For now, we'll assume the adapter might need to check y or be told.
        # We can pass is_multiclass in __init__ if needed, or deduce it.
        
    def set_multiclass(self, is_multiclass: bool):
        self.is_multiclass = is_multiclass

    def _suggest_param(self, trial: optuna.Trial, param_name: str, default_type: Optional[str] = None, **default_kwargs: Any) -> Any:
        """
        Suggests a parameter value based on config or defaults.
        """
        config_params = self.model_config.get('optuna_params', {})
        
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

    @abstractmethod
    def get_params(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        """
        Returns the parameters for the model. 
        If trial is provided, suggests parameters using Optuna.
        If trial is None, returns best params or defaults.
        """
        pass

    @abstractmethod
    def create_model(self, params: Dict[str, Any]) -> Any:
        """
        Instantiates the model with the given parameters.
        """
        pass

    @abstractmethod
    def get_fit_params(self, X_val, y_val) -> Dict[str, Any]:
        """
        Returns parameters for the fit method (e.g. eval_set, callbacks).
        """
        pass

    def predict(self, model: Any, X) -> np.ndarray:
        """
        Predicts using the model. 
        For classification, should return probabilities (or class indices if proba not supported/needed, but usually proba for metrics).
        For regression, returns values.
        """
        if self.task_type == 'classification':
            if hasattr(model, 'predict_proba'):
                if self.is_multiclass:
                    return model.predict_proba(X)
                else:
                    return model.predict_proba(X)[:, 1]
            else:
                return model.predict(X)
        else:
            return model.predict(X)
