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
| `imputer` | `'mean'`, `'median'`, `'most_frequent'`, `'constant'` | Strategy for filling missing values. |
| `scaler` | `'standard'`, `'minmax'`, `'robust'`, `'log'`, `'none'` | Scaling strategy. `'log'` applies Log1p transform. |
| `encoder` | `'onehot'`, `'ordinal'`, `'target'`, `'target_0.5'` | Encoding for categorical variables. `'target_0.5'` is Target Encoding with smoothing. |

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
