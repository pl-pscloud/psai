# psai - Machine Learning Pipeline Library

**psai** is a production-ready, configurable Machine Learning pipeline designed to accelerate the development of high-performance models. It unifies state-of-the-art algorithms (LightGBM, XGBoost, CatBoost, PyTorch) into a single, cohesive interface, automating tedious tasks like feature preprocessing, hyperparameter tuning, and ensemble creation.

---

## 🚀 Key Features

*   **🔍 Automated EDA**: Generate professional, publication-ready Exploratory Data Analysis reports (HTML/PDF) with deep insights into distributions, correlations, and target relationships.
*   **🤖 Multi-Model Intelligence**: First-class support for Gradient Boosting Machines (LightGBM, XGBoost, CatBoost) and Deep Learning (PyTorch Custom Architectures).
*   **⚡ Auto-Optimization**: Integrated **Optuna** engine for intelligent, parallelized hyperparameter search with pruning strategies.
*   **🏗️ Advanced Ensembling**: Built-in Stacking and Voting mechanisms to combine weak learners into robust meta-models.
*   **⚙️ Configuration-Driven**: Entire pipeline behavior controlled via a central `config.py`, ensuring reproducibility and ease of experimentation.
*   **🧠 Deep Learning Ready**: Includes implementations of modern tabular DL architectures like **FT-Transformer** and ResNet-style blocks.

---

## 📂 Project Structure

```text
.
├── config.py               # Central configuration file
├── psai/
│   ├── __init__.py
│   ├── psml.py             # Core Pipeline: Training, Optimization, Ensembling
│   ├── datasets.py         # Automated EDA & Data Reporting
│   ├── pstorch.py          # PyTorch Models (MLP, FT-Transformer)
│   └── scalersencoders.py  # Preprocessing Logic (Imputers, Scalers, Encoders)
└── README.md               # Documentation
```

---

## 📦 Installation

Ensure you have the required dependencies installed. It is recommended to use a virtual environment.

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost torch optuna seaborn matplotlib
```

---

## 📘 Configuration Manual

The heart of `psai` is the `config.py` file. It dictates every aspect of the pipeline. Below is a detailed reference for all configuration sections.

### 1. Dataset Configuration (`DATASET_CONFIG`)

Defines the data source and the nature of the machine learning task.

| Parameter | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `train_path` | `str` | Path to the training CSV file. | `'data/train.csv'` |
| `test_path` | `str` | Path to the test CSV file (for final inference). | `'data/test.csv'` |
| `target_column` | `str` | Name of the target variable column. | `'price'` |
| `id_column` | `str` | Name of the ID column (excluded from training). | `'id'` |
| `task_type` | `str` | Type of ML task: `'classification'` or `'regression'`. | `'regression'` |
| `metric` | `str` | Primary evaluation metric. | `'rmse'`, `'auc'` |
| `cv_folds` | `int` | Number of Cross-Validation folds. | `5` |
| `test_size` | `float` | Fraction of data to hold out for internal validation. | `0.2` |
| `random_state` | `int` | Seed for reproducibility. | `42` |
| `verbose` | `int` | Verbosity level (0=Silent, 1=Info, 2=Debug). | `2` |

### 2. Preprocessor Configuration (`PREPROCESSOR_CONFIG`)

Controls how different feature types are handled. The pipeline automatically detects feature types, but you define the strategy.

**Categories:**
*   `numerical`: Standard numerical features.
*   `skewed`: Numerical features with high skewness.
*   `outlier`: Numerical features with detected outliers.
*   `low_cardinality`: Categorical features with few unique values.
*   `high_cardinality`: Categorical features with many unique values.

**Options per Category:**

| Key | Options | Description |
| :--- | :--- | :--- |
| `imputer` | `'none'`, `'mean'`, `'median'`, `'most_frequent'`, `'constant'`, `'knn'`, `'knn_10'`, `'iterative'` | Strategy for filling missing values. |
| `scaler` | `'none'`, `'standard'`, `'minmax'`, `'robust'`, `'quantile'`, `'yeo-johnson'`, `'box-cox'`, `'log'` | Scaling strategy. `'log'` applies Log1p transform. |
| `encoder` | `'none'`, `'onehot'`, `'ordinal'`, `'label'`, `'target'`, `'target_0.5'`, `'target_1'`, `'target_10'`, `'hashing'` | Encoding for categorical variables. |

**Example:**
```python
'numerical': {
    'imputer': 'mean',
    'scaler': 'standard',
}
```

### 3. Model Configuration (`MODELS_CONFIG`)

Each model key (`lightgbm`, `xgboost`, `catboost`, `random_forest`, `pytorch`) has its own section.

**General Settings (Per Model):**

| Parameter | Description |
| :--- | :--- |
| `enabled` | `bool`. Set to `True` to include this model in optimization and training. |
| `optuna_trials` | `int`. Number of hyperparameter optimization trials to run. |
| `optuna_timeout` | `int`. Maximum time (seconds) for optimization. |
| `optuna_metric` | `str`. Metric used by Optuna to evaluate trials (e.g., `'rmse_safe'`). |
| `params` | `dict`. **Fixed** parameters passed directly to the model constructor. |
| `optuna_params` | `dict`. Search space definition for Optuna. |

**Defining Search Spaces (`optuna_params`):**
The library uses a structured dictionary to define search spaces.
*   **Int**: `{'type': 'int', 'low': 10, 'high': 100}`
*   **Float**: `{'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True}`
*   **Categorical**: `{'type': 'categorical', 'choices': ['option1', 'option2']}`

### 4. Ensemble Configuration

**Stacking (`STACKING_CONFIG`)**
| Parameter | Description |
| :--- | :--- |
| `cv_enabled` | Build ensemble using Cross-Validation models (OOF predictions). |
| `final_enabled` | Build ensemble using models trained on the full dataset. |
| `meta_model` | The model used to combine base predictions (e.g., `'lightgbm'`, `'linear'`). |
| `prefit` | If `True`, assumes base models are already fitted. |

**Voting (`VOTING_CONFIG`)**
Similar to Stacking but averages predictions (soft voting for classification).

---
## SHORT EXAMPLE OF CONFIG

```python

## FULL EXAMPLE of CONFIG

```python

import os
from tabnanny import verbose

optuna_trials = 10
optuna_n_jobs = 1
optuna_metric = 'rmse_safe'
model_n_jobs = int(os.cpu_count() / optuna_n_jobs)
device = 'gpu'
verbose = 2

# Dataset configuration
DATASET_CONFIG = {
    'target_column': 'accident_risk',
    'id_column': 'id',
    'test_size': 0.2,
    'task_type': 'regression',  # 'classification' or 'regression'
    'metric': 'rmse',  # 'auc', 'rmse', 'mae', etc.
    'cv_folds': 5,
    'random_state': 42,
    'verbose': verbose
}

# Model configurations
MODELS_CONFIG = {
    'lightgbm': {
        'enabled': False,
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600,
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': optuna_n_jobs,
        'params': {
            'verbose': verbose,
            'objective': 'rmse',
            'device': device,
            'eval_metric': 'rmse',
            'num_threads': model_n_jobs
        },
    },
    'xgboost': {
        'enabled': False,
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
    },
    'pytorch': {
        'enabled': False,
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600,
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': 1,
        'params': {
            "train_max_epochs": 50,
            "train_patience": 5,
            "final_max_epochs": 1000,
            "final_patience": 20,
            "objective": "mse",
            "device": device,
            'verbose': verbose,
            'num_threads': model_n_jobs if model_n_jobs > 0 else os.cpu_count(),
        },
        'optuna_params': {
            'model_type': {'type': 'categorical', 'choices': ['mlp']},
            'optimizer_name': {'type': 'categorical', 'choices': ['sgd','adam']},
            'learning_rate': {'type': 'categorical', 'choices': [0.001,0.01,0.1]},
            'batch_size': {'type': 'categorical', 'choices': [32,64]},
            'weight_init': {'type': 'categorical', 'choices': ['default']},
            'net': {'type': 'categorical', 'choices': [
                # Gelu, ReLU Architectures
                [
                    {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 16, 'activation': 'gelu', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': 'layer_norm'}
                ],
                [
                    {'type': 'dense', 'out_features': 32, 'activation': 'relu', 'norm': 'none'},
                    {'type': 'dense', 'out_features': 16, 'activation': 'relu', 'norm': 'none'},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': 'none'}
                ],

            ]},
        }
    }
}

# Stacking configuration
STACKING_CONFIG = {
    'cv_enabled': True,
    'cv_folds': 5,
    'final_enabled': True,
    'meta_model': 'lightgbm',  # 'lightgbm', 'xgboost', 'catboost', 'linear'
    'use_features': True,  # Whether to use original features along with base model predictions
    'prefit': True,  # Whether to use original features along with base model predictions
}

VOTING_CONFIG = {
    'cv_enabled': True,
    'final_enabled': True,
    'use_features': True,  # Whether to use original features along with base model predictions
    'prefit': True,  # Whether to use original features along with base model predictions
}

# Output configuration
OUTPUT_CONFIG = {
    'models_dir': 'models',
    'results_dir': 'outputs',
    'save_models': True,
    'save_predictions': True,
    'save_feature_importance': True
}

CONFIG = {
    'dataset': DATASET_CONFIG,
    'preprocessor': PREPROCESSOR_CONFIG,
    'models': MODELS_CONFIG,
    'stacking': STACKING_CONFIG,
    'voting': VOTING_CONFIG,
    'output': OUTPUT_CONFIG
}

```

## FULL EXAMPLE of CONFIG

```python

import os
from tabnanny import verbose

optuna_trials = 2
optuna_n_jobs = 1
optuna_metric = 'rmse_safe'
model_n_jobs = int(os.cpu_count() / optuna_n_jobs)
device = 'gpu'
verbose = 2

# Dataset configuration
DATASET_CONFIG = {
    'train_path': 'kaggle/input/playground-series-s5e10/train.csv',
    'test_path': 'kaggle/input/playground-series-s5e10/test.csv',
    'extra_path': 'kaggle/input/playground-series-s5e10/original.csv',
    'target_column': 'accident_risk',
    'id_column': 'id',
    'test_size': 0.2,
    'task_type': 'regression',  # 'classification' or 'regression'
    'metric': 'rmse',  # 'auc', 'rmse', 'mae', etc.
    'cv_folds': 5,
    'random_state': 42,
    'verbose': verbose
}

# Preprocessor configuration for create_preprocessor function
PREPROCESSOR_CONFIG = {
    'numerical': {
        'imputer': 'mean',  
        'scaler': 'minmax',   
    },
    'skewed': {
        'imputer': 'mean',  
        'scaler': 'log',   
    },
    'outlier': {
        'imputer': 'mean',  
        'scaler': 'log',   
    },
    'low_cardinality': {
        'imputer': 'most_frequent',  
        'encoder': 'onehot',  
        'scaler': 'none',   
    },
    'high_cardinality': {
        'imputer': 'most_frequent',  
        'encoder': 'target_0.5',  
        'scaler': 'standard',   
    },
    'dimension_reduction': {
        'enabled': False,
        'method': 'none',   # 'none', 'pca', etc.
    },
}





# Model configurations
MODELS_CONFIG = {
    'lightgbm': {
        'enabled': False,
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600,
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': optuna_n_jobs,
        'params': {
            'verbose': verbose,
            'objective': 'rmse',
            'device': device,
            'eval_metric': 'rmse',
            'num_threads': model_n_jobs
        },
        'optuna_params': {
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
            'bagging_freq': {'type': 'int', 'low': 1, 'high': 7}
        }
    },
    'xgboost': {
        'enabled': False,
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
        'optuna_params': {
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
        'enabled': False, # True
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600, # Time budget in seconds (1 hour)
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': optuna_n_jobs*5,
        'params': {
            'verbose': verbose,
            'objective': 'RMSE',
            'device': device,
            'eval_metric': 'RMSE',
            'thread_count': model_n_jobs
        },
        'optuna_params': {
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
        'enabled': False,
        'optuna_trials': optuna_trials,
        'optuna_timeout': 3600,
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': model_n_jobs, # RF uses n_jobs in the model itself usually, but we can parallelize trials if memory allows
        'params': {
            'verbose': verbose,
            'n_jobs': model_n_jobs
        },
        'optuna_params': {
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
        'enabled': False,
        'optuna_trials': optuna_trials,#,int(optuna_trials/2),
        'optuna_timeout': 3600,
        'optuna_metric': optuna_metric,
        'optuna_n_jobs': 1,
        'params': {
            "train_max_epochs": 50,
            "train_patience": 5,
            "final_max_epochs": 1000,
            "final_patience": 20,
            "objective": "mse",
            "device": device,
            'verbose': verbose,
            'num_threads': model_n_jobs if model_n_jobs > 0 else os.cpu_count(),
        },
        'optuna_params': {
            'model_type': {'type': 'categorical', 'choices': ['ft_transformer']},
            'optimizer_name': {'type': 'categorical', 'choices': ['adam']},
            'learning_rate': {'type': 'categorical', 'choices': [0.001]},
            'batch_size': {'type': 'categorical', 'choices': [64, 128, 256]},
            'weight_init': {'type': 'categorical', 'choices': ['default']},
            'net': {'type': 'categorical', 'choices': [
                # ReLU Architectures
                [
                    {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 16, 'activation': 'gelu', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': 'layer_norm'}
                ],
                # Residual Block Architecture Example
                [
                    {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'batch_norm'},
                    {'type': 'res_block', 'layers': [
                        {'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'batch_norm'},
                        {'type': 'dropout', 'p': 0.1}
                    ]},
                    {'type': 'dense', 'out_features': 16, 'activation': 'gelu', 'norm': 'batch_norm'},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}
                ]
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
    'cv_enabled': True,
    'cv_folds': 5,
    'final_enabled': True,
    'meta_model': 'lightgbm',  # 'lightgbm', 'xgboost', 'catboost', 'linear'
    'use_features': True,  # Whether to use original features along with base model predictions
    'prefit': True,  # Whether to use original features along with base model predictions
}

VOTING_CONFIG = {
    'cv_enabled': True,
    'final_enabled': True,
    'use_features': True,  # Whether to use original features along with base model predictions
    'prefit': True,  # Whether to use original features along with base model predictions
}

# Output configuration
OUTPUT_CONFIG = {
    'models_dir': 'models',
    'results_dir': 'outputs',
    'save_models': True,
    'save_predictions': True,
    'save_feature_importance': True
}

CONFIG = {
    'dataset': DATASET_CONFIG,
    'preprocessor': PREPROCESSOR_CONFIG,
    'models': MODELS_CONFIG,
    'stacking': STACKING_CONFIG,
    'voting': VOTING_CONFIG,
    'output': OUTPUT_CONFIG
}

```

## 🛠️ Usage Guide

### Step 1: Automated EDA

Before training, understand your data.

```python
import pandas as pd
from psai.datasets import EDAReport

# Load your dataset
df = pd.read_csv('data/train.csv')

# Generate Report
report = EDAReport(df, target='target_col')
report.generate_full_report(save_html='eda_report.html')
```

### Step 2: Initialize & Optimize

Load your config and start the pipeline.

```python
from psai.psml import psML
from config import CONFIG
import pandas as pd

# Prepare Data
train_df = pd.read_csv(CONFIG['dataset']['train_path'])
X = train_df.drop(columns=[CONFIG['dataset']['target_column']])
y = train_df[CONFIG['dataset']['target_column']]

# Initialize Pipeline
pipeline = psML(CONFIG, X, y)

# Run Optimization (Optuna)
# This will optimize all models with 'enabled': True in config
pipeline.optimize_all_models()

# Check Performance
pipeline.scores()
```

### Step 3: Build Ensembles

Combine your optimized models for maximum performance.

```python
# 1. Build Ensemble from CV folds (Robust, reduces overfitting)
pipeline.build_ensemble_cv()

# 2. Build Ensemble from Final models (Trained on full data)
pipeline.build_ensemble_final()
```

### Step 4: Inference & Saving

```python
# Save the entire pipeline
pipeline.save_model('models/my_pipeline.pkl')

# Load and Predict
loaded_pipeline = psML.load_model('models/my_pipeline.pkl')
predictions = loaded_pipeline.models['ensemble_stacking_cv']['model'].predict(X_test)
```

---

## 🧠 Extending PyTorch Models

`psai` supports custom PyTorch architectures via the `pstorch.py` module. To add a new architecture:

1.  Define your `nn.Module` class in `psai/pstorch.py`.
2.  Register it in the `PyTorchRegressor` or `PyTorchClassifier` `_build_model` method.
3.  Add the new architecture name to the `optuna_params` choices in `config.py` under the `pytorch` section.

---

## 📝 License

This project is open-source. Feel free to use and modify it for your machine learning projects.
