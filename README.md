# psai - Machine Learning Pipeline Library

**psai** is a production-ready, configurable Machine Learning pipeline designed to accelerate the development of high-performance models. It unifies state-of-the-art algorithms (LightGBM, XGBoost, CatBoost, PyTorch) into a single, cohesive interface, automating tedious tasks like feature preprocessing, hyperparameter tuning, and ensemble creation.

---

## 🚀 Key Features

*   **🔍 Automated EDA**: Generate professional, publication-ready Exploratory Data Analysis reports (HTML/PDF) with deep insights into distributions, correlations, and target relationships.
*   **🤖 AI Data Scientist Agent**: Built-in LLM-powered agent (using Gemini) that can analyze your data, write feature engineering code, and suggest optimal configurations.
*   **🧠 Multi-Model Intelligence**: First-class support for Gradient Boosting Machines (LightGBM, XGBoost, CatBoost) and Deep Learning (PyTorch Custom Architectures).
*   **⚡ Auto-Optimization**: Integrated **Optuna** engine for intelligent, parallelized hyperparameter search with pruning strategies.
*   **🏗️ Advanced Ensembling**: Built-in Stacking and Voting mechanisms to combine weak learners into robust meta-models.
*   **⚙️ Configuration-Driven**: Entire pipeline behavior controlled via a central `config.py`, ensuring reproducibility and ease of experimentation.
*   **🔥 Deep Learning Ready**: Includes implementations of modern tabular DL architectures like **FT-Transformer** and ResNet-style blocks (GELU, Residual Blocks).

---

## 📂 Project Structure

```text
.
├── psai/
│   ├── __init__.py
│   ├── config.py           # Central configuration file
│   ├── datascientist.py    # AI Agent for automated workflow
│   ├── datasets.py         # Automated EDA & Data Reporting
│   ├── psml.py             # Core Pipeline: Training, Optimization, Ensembling
│   ├── pstorch.py          # PyTorch Models (MLP, FT-Transformer)
│   └── scalersencoders.py  # Preprocessing Logic (Imputers, Scalers, Encoders)
└── README.md               # Documentation
```

---

## 📦 Installation

Ensure you have the required dependencies installed. It is recommended to use a virtual environment.

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost torch optuna seaborn matplotlib langchain-google-genai langgraph python-dotenv
```

*Note: You will need a Google Cloud API Key in your `.env` file (`GOOGLE_API_KEY`) to use the AI Agent.*

---

## 🤖 AI Data Scientist Agent

The `DataScientist` agent leverages Large Language Models (LLMs) to automate complex data science tasks. It acts as a pair programmer that understands your data.

### Capabilities
*   **Automated EDA Analysis**: Generates a textual summary of the EDA report and provides a structured analysis (Data Quality, Feature Engineering, Preprocessing, Model Selection).
*   **Intelligent Feature Engineering**: Writes and executes Python code to create new features based on data insights.
*   **Dynamic Configuration**: Suggests optimal `PREPROCESSOR_CONFIG` and `MODELS_CONFIG` based on the dataset characteristics.
*   **End-to-End Automation**: Orchestrates the entire pipeline from EDA to Model Ensembling with a single method call (`end_to_end_ml_process`).
*   **Comprehensive Reporting**: Generates a detailed HTML report (`save_report`) aggregating EDA, feature engineering logic, model configurations, and final performance metrics.

### Usage Example

You need api key for LLM (Google, OpenAI, Anthropic) or use local LLM (Ollama).

```python
from psai.datascientist import DataScientist
import pandas as pd

# Load Data
df = pd.read_csv('data/train.csv')
target = 'price'

# Initialize the agent
agent = DataScientist(
    df=df, 
    target=target, 
    optuna_metric='auc', 
    optuna_trials=20, 
    task_type='classification', 
    model_provider="google", 
    model_name="gemini-3-pro-preview", 
    api_key=api_key, 
    experiment_name=f"{target}_google"
    )

# Run the complete pipeline
# This performs EDA, Feature Engineering, Preprocessing, Model Selection, Training, and Ensembling
agent.end_to_end_ml_process(execute_code=True, save_report=True)

# The agent automatically saves a report (e.g., report-price-20231027...html)
# You can also access the trained pipeline directly:
agent.psml.scores()
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

### 2. MLFLOW Configuration (`MLFLOW_CONFIG`)

Controls how different feature types are handled. The pipeline automatically detects feature types, but you define the strategy.

| Parameter | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | Whether MLflow tracking is enabled. | `True` |
| `experiment_name` | `str` | Name of the MLflow experiment. | `'my_experiment'` |
| `tracking_uri` | `str` | MLflow tracking URI or local path. | `'http://localhost:5000'` or `'mlruns'` |

### 3. Preprocessor Configuration (`PREPROCESSOR_CONFIG`)

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

**Dimension Reduction Options:**
*   `method`: `'none'`, `'pca'`, `'pca_10'`, `'kpca'`, `'kpca_10'`, `'svd'`, `'svd_10'`

**Example:**
```python
'numerical': {
    'imputer': 'mean',
    'scaler': 'standard',
}
```

### 4. Model Configuration (`MODELS_CONFIG`)

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

### 5. Ensemble Configuration

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

## 🛠️ Manual Usage Guide

If you prefer not to use the AI Agent, you can run the pipeline manually.

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

Load your config and start the pipeline. You can pass your dataframes directly to the `psML` constructor.

```python
import pandas as pd
from psai.psml import psML
from psai.config import CONFIG

# Import data
train_df = pd.read_csv("train.csv", index_col='id')
target = 'target_column'

# Separate Features and Target
X = train_df.drop(columns=[target])
y = train_df[[target]]

# Initialize PSAI Pipeline
# You can pass X and y directly here.
ps = psML(config=CONFIG, X=X, y=y, experiment_name="PSAI Experiment")

# Run Optimization (Optuna)
# This will optimize all models with 'enabled': True in config
ps.optimize_all_models()

# Check Performance
ps.scores()
```

### Step 3: Build Ensembles

Combine your optimized models for maximum performance.

```python
# 1. Build Ensemble from CV folds (Robust, reduces overfitting)
ps.build_ensemble_cv()

# 2. Build Ensemble from Final models (Trained on full data)
ps.build_ensemble_final()
```

### Step 4: Inference & Saving

```python
# Save the entire pipeline
ps.save_model('models/my_pipeline.pkl')

# Load and Predict
loaded_pipeline = psML.load_model('models/my_pipeline.pkl')
# Access specific models for prediction if needed, or use the ensemble
predictions = loaded_pipeline.models['ensemble_stacking_cv']['model'].predict(X_test)
```

---

## 🧠 Extending PyTorch Models

`psai` supports custom PyTorch architectures via the `pstorch.py` module. To add a new architecture:

1.  Define your `nn.Module` class in `psai/pstorch.py`.
2.  Register it in the `PyTorchRegressor` or `PyTorchClassifier` `_build_model` method.
3.  Add the new architecture name to the `optuna_params` choices in `config.py` under the `pytorch` section.

---

## FULL EXAMPLE of CONFIG

```python


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

MLFLOW_CONFIG = {
    'enabled': False,
    'experiment_name': 'Experiment Optuna Tuner',
    'tracking_uri': 'mlruns'
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
        'optuna_metric': optuna_metric,        # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': optuna_n_jobs,        # Parallel jobs for Optuna
        'params': {                            # Fixed parameters (not optimized)
            'verbose': verbose,
            'objective': 'rmse',               # Learning objective (e.g., regression:['mse','mae'], classification:['binary','multiclass'])
            'device': device,                  # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'rmse',             # Metric used for early stopping regression:['mse','mae'], classification:['binary','multiclass'])
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
        'enabled': models_enabled['xgboost'],       # Enable/Disable XGBoost models
        'optuna_trials': optuna_trials,             # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                     # 3600 seconds = 1 hour 
        'optuna_metric': optuna_metric,             # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': optuna_n_jobs,             # Number of jobs to run in parallel
        'params': {
            'verbose': verbose,
            'objective': 'reg:squarederror',        # Learning objective (e.g., regression:['reg:squarederror','reg:absoluteerror'], classification:['binary:logistic','multi:softprob'])
            'device': device,                       # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'rmse',                  # Metric used for early stopping regression:['mse','mae'], classification:['binary','multiclass'])
        'nthread': model_n_jobs,                    # Threads for model training
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
        'enabled': models_enabled['catboost'],   # Enable/Disable CatBoost models
        'optuna_trials': optuna_trials,         # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                 # Time budget in seconds (1 hour)
        'optuna_metric': optuna_metric,         # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': optuna_n_jobs,         # Number of parallel Optuna jobs (studies running at once)
        'params': {
            'verbose': verbose,
            'objective': 'RMSE',                # Learning objective (e.g., regression:['RMSE','MAE'], classification:['Logloss','MultiClass'])
            'device': device,                   # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'RMSE',              # Metric used for early stopping regression:['mse','mae'], classification:['binary','multiclass'])
            'thread_count': model_n_jobs,       # Threads for model training
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
        'optuna_metric': optuna_metric,             # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
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
        'optuna_metric': optuna_metric,             # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': 1,                         # Number of jobs to run in parallel
        'params': {
            "train_max_epochs": 50,                 # Number of epochs to train for
            "train_patience": 5,                    # Number of epochs to wait before early stopping
            "final_max_epochs": 1000,               # Number of epochs to train for
            "final_patience": 20,                   # Number of epochs to wait before early stopping
            "objective": "mse",                     # objective (e.g., regression:['mse','mae','huber','rmsle','mape'], classification:['bce','bcelogit','crossentropy']
            "device": device,                       # 'cpu', 'gpu'
            'verbose': verbose,
            'embedding_info': ['time_of_day'],      #       
            'num_threads': model_n_jobs,
        },
        'optuna_params': {                                                                      # Hyperparameter search space for PyTorch models
            'model_type': {'type': 'categorical', 'choices': ['mlp','ft_transformer']},                          #['mlp', 'ft_transformer'] model type
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
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}
                ],
                # MLP Swish/SILU with layer norm
                [
                    {'type': 'dense', 'out_features': 64, 'activation': 'swish', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 32, 'activation': 'swish', 'norm': 'layer_norm'},
                    {'type': 'dropout', 'p': 0.1},
                    {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}
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
    'cv_enabled': models_enabled['stacking'],                                       # Enable stacking on models from Optuna Cross-Validation best trial if choosen eg. lightbm stacking is builded on all lightbm models from best cv trial during optuna optimalization 
    'cv_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],       # Models for stacking CV
    'cv_folds': 5,                                                                  # Folds for stacking CV (if not using prefit)
    'final_enabled': models_enabled['stacking'],                                    # Enable stacking for the final models
    'final_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],    # Models for stacking final modelss
    'meta_model': 'lightgbm',                                                       # The model used to aggregate base model predictions
    'use_features': False,                                                          # If False the meta-model will only learn from the predictions of the base models, not the original features.
    'prefit': True,                                                                 # If True, uses existing trained models (faster). If False, retrains.
}

VOTING_CONFIG = {
    'cv_enabled': models_enabled['voting'],                                         # Enable voting on finals models builded during optuna optimalization 
    'cv_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],       # Models for voting CV
    'final_enabled': models_enabled['voting'],                                      # Enable voting ensemble for the final model
    'final_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],    # Models for voting final models
    'use_features': False,                                                          # If False the meta-model will only learn from the predictions of the base models, not the original features.
    'prefit': True,                                                                 # If True, uses already trained models.
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
    'mlflow': MLFLOW_CONFIG,
    'dataset': DATASET_CONFIG,
    'preprocessor': PREPROCESSOR_CONFIG,
    'models': MODELS_CONFIG,
    'stacking': STACKING_CONFIG,
    'voting': VOTING_CONFIG,
    'output': OUTPUT_CONFIG
}



```

---



## 📝 License

This project is open-source. Feel free to use and modify it for your machine learning projects.
