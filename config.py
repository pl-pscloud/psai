
import os

optuna_trials = 10                                   # Number of trials for Optuna hyperparameter optimization
optuna_n_jobs = 1                                   # Number of parallel Optuna jobs (studies running at once)
optuna_metric = 'rmse_safe'                         # Metric to optimize during Optuna trials (e.g., 'rmse', 'auc')
model_n_jobs = int(os.cpu_count() / optuna_n_jobs)  # Number of threads per model (CPU cores / optuna jobs)
device = 'gpu'                                      # Device to use for training ('cpu' or 'gpu')
verbose = 2                                         # Verbosity level (0: silent, 1: minimal, 2: detailed)
models_enabled = {                                  # Master toggle to enable/disable specific models
    'lightgbm': False,
    'xgboost': False,
    'catboost': False,
    'random_forest': False,
    'pytorch': False,
    'stacking': False,
    'voting': False,
}

# Dataset configuration
DATASET_CONFIG = {
    'train_path': 'train.csv',  # Path to the training CSV file
    'target': 'target',         # Name of the target column to predict
    'id_column': 'id',          # Name of the ID column (will be set as index)
    'test_size': 0.2,           # Proportion of data to use for the hold-out test set
    'task_type': 'regression',  # Type of ML task: 'classification' or 'regression'
    'metric': 'rmse',           # Evaluation metric for final scoring: 'auc', 'rmse', 'mae', etc.
    'cv_folds': 5,              # Number of cross-validation folds
    'random_state': 42,         # Seed for reproducibility
    'verbose': verbose          # Verbosity level inherited from global setting
}

# Preprocessor configuration for create_preprocessor function
# Defines how different column types are handled in the pipeline
PREPROCESSOR_CONFIG = {
    "numerical": {
        "imputer": "mean",
        "scaler": "standard"
    },
    "skewed": {
        "imputer": "median",
        "scaler": "log"
    },
    "outlier": {
        "imputer": "median",
        "scaler": "log"
    },
    "low_cardinality": {
        "imputer": "most_frequent",
        "encoder": "onehot",
        "scaler": "none"
    },
    "high_cardinality": {
        "imputer": "most_frequent",
        "encoder": "target",
        "scaler": "none"
    },
    "dimension_reduction": {
        "method": "none"
    }
}

# Model configurations
MODELS_CONFIG = {
    'lightgbm': {
        'enabled': models_enabled['lightgbm'], # Enable/Disable LightGBM
        'optuna_trials': optuna_trials,        # Number of hyperparameter search trials
        'optuna_timeout': 3600,                # Max time (seconds) for optimization
        'optuna_metric': optuna_metric,        # Metric to optimize
        'optuna_n_jobs': optuna_n_jobs,        # Parallel jobs for Optuna
        'params': {                            # Fixed parameters (not optimized)
            'verbose': verbose,
            'objective': 'rmse',               # Learning objective (e.g., 'rmse', 'binary')
            'device': device,                  # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'rmse',             # Metric used for early stopping
            'num_threads': model_n_jobs,       # Threads for model training
            
        },
        'optuna_params': {                    # Hyperparameter search space
            'boosting_type': {'type': 'categorical', 'choices': ['gbdt', 'goss']},
            'lambda_l1': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'lambda_l2': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 300},
            'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.2, 'log': True},
            'min_split_gain': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
            'bagging_freq': {'type': 'int', 'low': 1, 'high': 7},
            'num_class': 10,                   # Number of classes for multiclass classification (if task_type is 'classification')
        }
    },
    'xgboost': {
        'enabled': models_enabled['xgboost'],
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600,
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': optuna_n_jobs,
        'params': {
            'verbose': verbose,
            'objective': 'reg:squarederror',
            'device': device,
            'eval_metric': 'rmse',
            'nthread': model_n_jobs
        },
        'optuna_params': {                    # Hyperparameter search space
            'booster': {'type': 'categorical', 'choices': ['gbtree']},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.2, 'log': True},
            'n_estimators': {'type': 'int', 'low': 500, 'high': 3000},
            'subsample': {'type': 'float', 'low': 0, 'high': 1},
            'lambda': {'type': 'float', 'low': 1e-4, 'high': 5, 'log': True},
            'gamma': {'type': 'float', 'low': 1e-4, 'high': 5, 'log': True},
            'alpha': {'type': 'float', 'low': 1e-4, 'high': 5, 'log': True},
            'min_child_weight': {'type': 'categorical', 'choices': [0.5, 1, 3, 5]},
            'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1},
            'colsample_bylevel': {'type': 'float', 'low': 0.5, 'high': 1}
        }
    },
    'catboost': {
        'enabled': models_enabled['catboost'],
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600, # Time budget in seconds (1 hour)
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': optuna_n_jobs,
        'params': {
            'verbose': verbose,
            'objective': 'RMSE',
            'device': device,
            'eval_metric': 'RMSE',
            'thread_count': model_n_jobs
        },
        'optuna_params': {                    # Hyperparameter search space
            'n_estimators': {'type': 'int', 'low': 100, 'high': 3000},
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.2, 'log': True},
            'depth': {'type': 'int', 'low': 4, 'high': 10},
            'l2_leaf_reg': {'type': 'float', 'low': 1e-3, 'high': 10.0, 'log': True},
            'border_count': {'type': 'int', 'low': 32, 'high': 128},
            'bootstrap_type': {'type': 'categorical', 'choices': ['Bayesian', 'Bernoulli', 'MVS']},
            'feature_border_type': {'type': 'categorical', 'choices': ['Median', 'Uniform', 'GreedyMinEntropy']},
            'leaf_estimation_iterations': {'type': 'int', 'low': 1, 'high': 10},
            'min_data_in_leaf': {'type': 'int', 'low': 1, 'high': 30},
            'random_strength': {'type': 'float', 'low': 1e-9, 'high': 10, 'log': True},
            'grow_policy': {'type': 'categorical', 'choices': ['SymmetricTree', 'Depthwise', 'Lossguide']},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'bagging_temperature': {'type': 'float', 'low': 0, 'high': 1},
            'max_leaves': {'type': 'int', 'low': 2, 'high': 32}
        }
    },
    'random_forest': {
        'enabled': models_enabled['random_forest'], # Enable/Disable Random Forest models
        'optuna_trials': optuna_trials,             # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                     # 3600 seconds = 1 hour 
        'optuna_metric': optuna_metric,             # 'mae', 'mse', 'rmse', 'rmsle', 'r2', 'mape'
        'optuna_n_jobs': model_n_jobs,              # Number of jobs to run in parallel
        'params': {
            'verbose': verbose,
            'n_jobs': model_n_jobs
        },
        'optuna_params': {                    # Hyperparameter search space for RF
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 30},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
            'bootstrap': {'type': 'categorical', 'choices': [True, False]},
            'max_samples': {'type': 'float', 'low': 0.5, 'high': 1.0}
        }
    },
    'pytorch': {
        'enabled': models_enabled['pytorch'],       # Enable/Disable PyTorch models
        'optuna_trials': optuna_trials,             # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                     # 3600 seconds = 1 hour 
        'optuna_metric': optuna_metric,             # 'mae', 'mse', 'rmse', 'rmsle', 'r2', 'mape'
        'optuna_n_jobs': 1,                         # Number of jobs to run in parallel
        'params': {
            "train_max_epochs": 50,                 # Number of epochs to train for
            "train_patience": 5,                    # Number of epochs to wait before early stopping
            "final_max_epochs": 1000,               # Number of epochs to train for
            "final_patience": 20,                   # Number of epochs to wait before early stopping
            "objective": "mse",                     # 'mae', 'mse', 'rmse', 'rmsle', 'r2', 'mape'
            "device": device,                       # 'cpu', 'gpu'
            'verbose': verbose,
            'embedding_info': ['time_of_day'],      #       
            'num_threads': model_n_jobs,
        },
        'optuna_params': {                         # Hyperparameter search space for PyTorch models
            'model_type': {'type': 'categorical', 'choices': ['mlp']},        #['mlp', 'ft_transformer'] model type
            'optimizer_name': {'type': 'categorical', 'choices': ['adam']},                     #['adam', 'nadam', 'adamax', 'adamw', 'sgd', 'rmsprop] optimizer name
            'learning_rate': {'type': 'categorical', 'choices': [0.01]},                        #['0.01', '0.001'] learning rate
            'batch_size': {'type': 'categorical', 'choices': [64, 128, 256]},                   #['64', '128', '256'] batch size
            'weight_init': {'type': 'categorical', 'choices': ['default']},                     #['default', 'xavier', 'kaiming'] weight initialization
            'net': {'type': 'categorical', 'choices': [                                         
                # MLP ReLU without batch or layer norm
                [
                    {'type': 'dense', 'out_features': 16, 'activation': 'relu', 'norm': None},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}
                ],
                # MLP GELU with layer norm
                [
                    {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': 'layer_norm'}
                ],
                # MLP Swish/SILU with layer norm
                [
                    {'type': 'dense', 'out_features': 64, 'activation': 'swish', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 32, 'activation': 'swish', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': 'layer_norm'}
                ],

            ]},
            # FT-Transformer Params
            'd_token': {'type': 'categorical', 'choices': [64, 128, 192, 256]},
            'n_layers': {'type': 'int', 'low': 1, 'high': 4},
            'n_heads': {'type': 'categorical', 'choices': [4, 8]},
            'd_ffn_factor': {'type': 'float', 'low': 1.0, 'high': 2.0},
            'attention_dropout': {'type': 'float', 'low': 0.0, 'high': 0.3},
            'ffn_dropout': {'type': 'float', 'low': 0.0, 'high': 0.3},
            'residual_dropout': {'type': 'float', 'low': 0.0, 'high': 0.2}
        }
    }
}

# Stacking configuration
STACKING_CONFIG = {
    'cv_enabled': models_enabled['stacking'],   # Enable stacking during Cross-Validation
    'cv_folds': 5,                              # Folds for stacking CV (if not using prefit)
    'final_enabled': models_enabled['stacking'],# Enable stacking for the final model
    'meta_model': 'lightgbm',                   # The model used to aggregate base model predictions
    'use_features': True,                       # If True, feeds original features + predictions to meta-model
    'prefit': True,                             # If True, uses existing trained models (faster). If False, retrains.
}

VOTING_CONFIG = {
    'cv_enabled': models_enabled['voting'],     # Enable voting ensemble during Cross-Validation
    'final_enabled': models_enabled['voting'],  # Enable voting ensemble for the final model
    'use_features': True,                       # (Note: Voting usually just averages predictions, this flag might be custom logic)
    'prefit': True,                             # If True, uses already trained models.
}

# Output configuration
OUTPUT_CONFIG = {
    'models_dir': 'models',         # Directory to save trained model artifacts
    'results_dir': 'outputs',       # Directory to save predictions and logs
    'save_models': True,            # Whether to save the model objects (pickle/joblib)
    'save_predictions': True,       # Whether to save prediction CSVs
    'save_feature_importance': True # Whether to save feature importance plots/CSVs
}

CONFIG = {
    'dataset': DATASET_CONFIG,
    'preprocessor': PREPROCESSOR_CONFIG,
    'models': MODELS_CONFIG,
    'stacking': STACKING_CONFIG,
    'voting': VOTING_CONFIG,
    'output': OUTPUT_CONFIG
}
