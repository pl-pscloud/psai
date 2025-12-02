from typing import Any, Dict, Optional
import optuna
import os
from psai.core.pstorch import PyTorchClassifier, PyTorchRegressor
from psai.models.base import BaseModelAdapter

class PyTorchAdapter(BaseModelAdapter):
    def __init__(self, model_config: Dict[str, Any], global_config: Dict[str, Any]):
        super().__init__(model_config, global_config)
        self.embedding_info = None
        self.n_classes = 1

    def set_embedding_info(self, embedding_info):
        self.embedding_info = embedding_info

    def set_n_classes(self, n_classes):
        self.n_classes = n_classes

    def get_params(self, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
        # Only use embedding_info if encoder is ordinal
        encoder = self.global_config['preprocessor']['high_cardinality']['encoder']
        if encoder != 'ordinal':
            self.embedding_info = {}

        params = {
            "loss": self.model_config['params']['objective'],
            "embedding_info": self.embedding_info,
            "verbose": self.model_config['params']['verbose'],
            "device": 'cuda' if self.model_config['params']['device'] == 'gpu' else 'cpu',
            "num_threads": self.model_config['params']['num_threads']
        }
        
        if trial:
            # Suggest model type first
            model_type = self._suggest_param(trial, 'model_type', 'categorical', choices=['mlp', 'ft_transformer'])
            params['model_type'] = model_type
            
            params.update({
                "learning_rate": self._suggest_param(trial, 'learning_rate', 'categorical', choices=[0.001]),
                "optimizer_name": self._suggest_param(trial, 'optimizer_name', 'categorical', choices=['adam']),
                "batch_size": self._suggest_param(trial, 'batch_size', 'categorical', choices=[64, 128, 256]),
                "weight_init": self._suggest_param(trial, 'weight_init', 'categorical', choices=['default']),
                "max_epochs": self.model_config['params']['train_max_epochs'],
                "patience": self.model_config['params']['train_patience'],
            })

            if model_type == 'mlp':
                params["net"] = self._suggest_param(trial, 'net', 'categorical', choices=[
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
                    'd_token': self._suggest_param(trial, 'd_token', 'categorical', choices=[64, 128, 192, 256]),
                    'n_layers': self._suggest_param(trial, 'n_layers', 'int', low=1, high=4),
                    'n_heads': self._suggest_param(trial, 'n_heads', 'categorical', choices=[4, 8]),
                    'd_ffn_factor': self._suggest_param(trial, 'd_ffn_factor', 'float', low=1.0, high=2.0),
                    'attention_dropout': self._suggest_param(trial, 'attention_dropout', 'float', low=0.0, high=0.3),
                    'ffn_dropout': self._suggest_param(trial, 'ffn_dropout', 'float', low=0.0, high=0.3),
                    'residual_dropout': self._suggest_param(trial, 'residual_dropout', 'float', low=0.0, high=0.2),
                    'activation': 'reglu',
                    'n_out': 1
                }
        else:
            best_params = self.model_config.get('best_params', {})
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
            params["max_epochs"] = self.model_config['params']['final_max_epochs']
            params["patience"] = self.model_config['params']['final_patience']

        # Adjust output dimension for multiclass
        n_classes = 1
        if self.is_multiclass:
             n_classes = self.n_classes
        
        if params.get('model_type') == 'ft_transformer':
            if 'ft_params' in params:
                params['ft_params']['n_out'] = n_classes
        elif params.get('model_type') == 'mlp':
            if 'net' in params and isinstance(params['net'], list):
                # Assume last layer is the output layer
                import copy
                params['net'] = copy.deepcopy(params['net'])
                params['net'][-1]['out_features'] = n_classes

        return params

    def create_model(self, params: Dict[str, Any]) -> Any:
        if self.task_type == 'classification':
            return PyTorchClassifier(**params)
        else:
            return PyTorchRegressor(**params)

    def get_fit_params(self, X_val, y_val) -> Dict[str, Any]:
        return {
            'eval_set': [X_val, y_val]
        }
