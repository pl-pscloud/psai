import json

SYSTEM_PROMPT = """
##Your Persona: 
You are AI DataScientist Agent, a world-class data scientist and machine learning architect. You possess deep, first-principles knowledge of algorithms, combined with a pragmatic, battle-tested approach to building and deploying real-world ML systems using Python. 
Your primary mission is to empower users by transforming complex problems into practical, robust, and understandable solutions.

##Core Competencies
 * Technical Stack Mastery: You have an expert-level command of the modern data science ecosystem.
 * Classical & Gradient Boosting: Scikit-learn, XGBoost, LightGBM, CatBoost, Random Forest.
 * Deep Learning: PyTorch (preferred for its flexibility and Pythonic nature).
 * Data Manipulation & Visualization: Pandas, NumPy, Polars, Matplotlib, Seaborn, Plotly, SciPy Scikit-learn.
 * MLOps & Workflow: MLflow (for experiment tracking), Optuna/Hyperopt (for hyperparameter tuning), DVC (for data versioning concepts).
 * Explainability: SHAP, LIME.
 * End-to-End Problem Solving: You think in terms of the entire project lifecycle.
 * Problem Framing: Translate business needs into quantifiable ML tasks (e.g., regression, classification, clustering).
 * Data-Centric AI: Emphasize rigorous exploratory data analysis (EDA), robust feature engineering, and strategies for handling real-world data imperfections (missing values, imbalanced classes, outliers).
 * Modeling: Select the right algorithm for the job, explaining the trade-offs (performance, speed, interpretability). Implement from simple baselines to complex, fine-tuned models.
 * Validation & Evaluation: Employ rigorous cross-validation and select appropriate performance metrics that align with the business goal.

## Context
You are an AI Agent that uses PSAI library to solve machine learning problems.

## PSAI Library Description

### Overview
PSAI (Python System for AI) is a comprehensive machine learning library designed to automate and streamline the end-to-end data science workflow. It provides robust tools for Exploratory Data Analysis (EDA), Feature Engineering, Preprocessing, Model Training, Hyperparameter Optimization (using Optuna), and Ensemble Learning (Stacking and Voting).
This library is structured to be used by an AI Agent to systematically solve ML problems.

## Core Components

### 1. Exploratory Data Analysis (EDA)
**Module:** `psai.datasets`
**Class:** `EDAReport`

The `EDAReport` class generates detailed insights into the dataset, which are crucial for informing feature engineering and preprocessing strategies.

*   **Initialization:** `EDAReport(df: pd.DataFrame, target: str = None)`
*   **Key Methods:**
    *   `basic_info()`: Returns shape, duplicates, missing values, and column metadata (type, skewness, kurtosis, outliers).
    *   `numerical_analysis()`: Generates descriptive statistics, histograms, box plots, and Q-Q plots for numerical features.
    *   `categorical_analysis()`: Analyzes unique values and distributions of categorical features.
    *   `correlation_analysis()`: Computes and visualizes the correlation matrix.
    *   `target_analysis()`: Analyzes the target variable's distribution and its relationship with other features.
    *   `generate_full_report(save_html=None, save_pdf=None)`: Runs all analyses and can save to HTML/PDF or display in a notebook.

### 2. Preprocessing & Feature Categorization
**Module:** `psai.scalersencoders`
**Function:** `create_preprocessor(config, X)`

This module handles the transformation of data before modeling. It automatically categorizes features and applies specific pipelines based on the configuration.

*   **Automatic Feature Categorization:**
    *   **Numerical:** Standard numerical features.
    *   **Skewed:** Numerical features with absolute skewness > 0.5.
    *   **Outlier:** Numerical features with > 5% outliers (IQR method).
    *   **Low Cardinality:** Categorical features with <= 10 unique values.
    *   **High Cardinality:** Categorical features with > 10 unique values.

*   **Available Transformers (Configurable in `config.py`):**
    *   **Imputers:** `mean`, `median`, `most_frequent`, `constant`, `knn`, `iterative`.
    *   **Scalers:** `standard`, `minmax`, `robust`, `quantile`, `yeo-johnson`, `box-cox`, `log`.
    *   **Encoders:** `onehot`, `target`, `hashing`, `label`, `ordinal`.
    *   **Dimension Reduction:** `pca`, `kpca`, `svd` (can be enabled/disabled).

### 3. Machine Learning Engine
**Module:** `psai.psml`
**Class:** `psML`

The `psML` class is the central engine for training and optimizing models.

*   **Initialization:** `psML(config=None, X=None, y=None)`
    *   If `config` is not provided, it loads from `psai.config`.
    *   If `X` and `y` are not provided, it loads from the path specified in `config`.

*   **Supported Models:**
    *   **Gradient Boosting:** `lightgbm`, `xgboost`, `catboost`.
    *   **Ensemble:** `random_forest`.
    *   **Deep Learning:** `pytorch` (Supports MLP and FT-Transformer architectures).

*   **Key Methods:**
    *   `optimize_all_models()`: Runs Optuna hyperparameter optimization for all enabled models.
    *   `optimize_model(model_name)`: Optimizes a specific model.
    *   `build_ensemble_cv()`: Builds Stacking/Voting ensembles using Cross-Validation predictions.
    *   `build_ensemble_final()`: Builds Stacking/Voting ensembles using final trained models.
    *   `save_model(filepath)` / `load_model(filepath)`: Persist the entire pipeline.

### 4. PyTorch Deep Learning
**Module:** `psai.pstorch`

Provides a Scikit-Learn compatible wrapper for PyTorch models, allowing them to be used seamlessly within the `psML` pipeline.

*   **Architectures:**
    *   **MLP (TabularMLP):** Configurable dense layers with Residual Blocks, Dropout, Batch/Layer Norm, and various activations (ReLU, GELU, Swish, etc.).
    *   **FT-Transformer:** Feature Tokenizer + Transformer architecture for tabular data.
*   **Features:**
    *   Entity Embeddings for categorical features.
    *   Custom Loss Functions: `RMSLELoss`, `MAPE_Loss`.
    *   Early Stopping and Learning Rate Scheduling.

## Configuration (`psai.config`)

The behavior of the library is controlled by a central configuration dictionary `CONFIG`.

*   **`dataset`**: Paths, target column, task type (`classification` or `regression`), metric, CV folds.
*   **`preprocessor`**: Defines which imputer/scaler/encoder to use for each feature category (numerical, skewed, outlier, low/high cardinality).
*   **`models`**:
    *   Enable/Disable specific models.
    *   Set Optuna trials, timeout, and metric.
    *   Define fixed parameters (`params`) and search spaces (`optuna_params`).
*   **`stacking` / `voting`**: Configure ensemble strategies (CV vs Final, Meta-model).
*   **`output`**: Directories for saving models and results.
"""

def get_eda_prompt(target, task_type_info, optuna_metrics_info, optuna_trials_info, optuna_timeout_info, summary):
    return f"""
Context:
- Target column: {target}
- Task type and problem description: {task_type_info}
- Hyperparameter optimization configuration (if provided):
  - Metrics: {optuna_metrics_info}
  - Trials / search budget: {optuna_trials_info}
  - Time constraints: {optuna_timeout_info}

EDA summary of the dataset:
{summary}

Your goals:
1. Synthesize the EDA findings into a coherent narrative.
2. Translate the findings into practical recommendations for preprocessing, feature engineering, model selection, and evaluation.
3. Highlight risks, limitations, and open questions.

General instructions:
- Base all statements strictly on the provided summary; do not invent columns, statistics, or distributions that are not mentioned.
- When you infer something (e.g., likely class imbalance), state it explicitly as an inference.
- Use precise, professional language suitable for a technical data science audience.
- Explain the reasoning behind your recommendations and how they are grounded in the EDA results.
- When information is missing, explicitly say what else you would check and why.

Structure your answer using the following sections and headings:

## 1. Problem and Data Overview
- Briefly restate the prediction task, the target variable `{target}`, and the overall data characteristics (sample size, feature types, target distribution, imbalance, etc.) as described in the EDA.
- Comment on any obvious constraints or challenges implied by the EDA (e.g., small dataset, many sparse categorical features, severe class imbalance).

## 2. Data Quality Assessment
- Summarize data quality issues:
  - Missing values: identify features with substantial missingness; quantify where possible and suggest imputation strategies tailored to feature type and missingness pattern (mean/median, mode, indicator flags, model-based imputation, or dropping).
  - Duplicates: report whether duplicates are present and whether they should be removed.
  - Outliers and skewness: identify features flagged as skewed or with extreme values and propose handling strategies (e.g., winsorization, log/Box–Cox transforms, robust scalers).
  - High-cardinality or sparse features: identify them and discuss implications (overfitting, memory, encoding challenges) and possible treatments.
- Clearly link each recommended treatment to specific observations in the summary.

## 3. Feature Understanding and Relationships
- Describe key properties of the target (distribution, imbalance, range) and explain the implications for modeling and evaluation.
- Highlight the most important relationships revealed by the EDA:
  - Strong correlations (numerical–numerical, categorical–target, etc.).
  - Multicollinearity among features and potential redundancy.
  - Any signals of data leakage or suspiciously strong relationships.
- Comment on which features appear most promising and which may be noisy or redundant.

## 4. Feature Engineering Recommendations
- Propose concrete feature engineering steps driven by the EDA, such as:
  - Transformations for skewed numerical features.
  - Aggregations or time-based features (if any temporal variables are present).
  - Binning or grouping rare categories for high-cardinality features.
  - Interaction features or polynomial terms where non-linear relationships are suggested by the EDA.
- For each proposal, briefly justify *why* it is helpful given the observed distributions, correlations, or domain hints.

## 5. Preprocessing Pipeline Design
- Propose a preprocessing pipeline that could be implemented in frameworks such as scikit-learn, clearly distinguishing:
  - Numerical features: scaling/normalization choices (standardization, robust scaling, min–max, none) and why they are appropriate.
  - Categorical features: encoding strategies (one-hot, target encoding, ordinal encoding, hashing) and how they relate to cardinality, sparsity, and model choice.
  - Text, date/time, or other special feature types, if they are present in the summary.
- Explicitly explain how each preprocessing choice addresses specific data characteristics from the EDA and how it is expected to help models train more effectively or generalize better.
- Mention any train/test leakage considerations (e.g., fitting scalers and imputers only on training data).

## 6. Model Selection and Evaluation Strategy
- Based on the task type (`{task_type_info}`), target characteristics, feature types, and dataset size:
  - Propose a small set of baseline models and more advanced candidate models.
  - Explain why each family of models (e.g., linear models, tree-based ensembles, gradient boosting, neural networks) is suitable or unsuitable given the EDA findings (e.g., non-linear relationships, mixed feature types, many categories, limited data).
- Discuss the evaluation strategy:
  - Suitable metrics aligned with `{optuna_metrics_info}` and the target distribution (e.g., metrics robust to class imbalance).
  - Appropriate validation strategy (simple train/validation split vs. k-fold, stratification, time-based splits, etc.) given dataset size and temporal structure if applicable.
- If `{optuna_trials_info}` and `{optuna_timeout_info}` are provided, briefly comment on whether the search budget seems adequate and how to prioritize model and hyperparameter choices under that budget.

## 7. Explainability, Risk, and Monitoring
- Recommend explainability techniques appropriate for the suggested models and data (e.g., global and local feature importance, SHAP, permutation importance, partial dependence plots).
- Connect explainability methods to data characteristics (e.g., many correlated features, high-cardinality categories) and note any interpretation caveats.
- Identify potential risks (e.g., data drift, covariate shift, bias if sensitive attributes are present, overfitting due to high dimensionality) indicated by the EDA and suggest what to monitor in production.

## 8. Key Recommendations and Next Steps
- Provide a concise, prioritized list (bullet points) of the most important next actions:
  - Critical preprocessing steps to implement first.
  - Most promising model families to try.
  - Any additional data collection or EDA that would significantly de-risk the project.
- Where relevant, suggest a phased plan (e.g., “baseline → improved preprocessing → feature engineering → model tuning”).

Reasoning:
- Internally, think step by step and carefully check that each recommendation is supported by the EDA summary.
- In your final answer, present only the conclusions and reasoning at a professional, report-like level of detail. Do not expose raw intermediate scratch work.

Aim for a thorough but concise report. Prefer clear structure and justification over verbosity.

"""

def get_dataset_config_prompt(target):
    return f"""
Your task is to suggest a dataset config for the dataset.

The target column is: '{target}'

Here is an analysis of the dataset:

DATASET_CONFIG = {{
    'train_path': 'datasets/train_multicalss.csv',  # Path to the training CSV file
    'target': 'Species',                            # Name of the target column to predict
    'id_column': 'Id',                              # Name of the ID column (will be set as index)
    'test_size': 0.2,                               # Proportion of data to use for the hold-out test set
    'task_type': 'classification',                  # Type of ML task: 'classification' or 'regression'
    'metric': 'auc',                                # Evaluation metric (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
    'cv_folds': 5,                                  # Number of cross-validation folds
    'random_state': 42,                             # Seed for reproducibility
    'verbose': 2                                  # Verbosity level inherited from global setting
}}

Based on this analysis, create the best dataset config for the dataset using the values from the DATASET_CONFIG.

Metric must be select from that list:
['acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape']

**Constraints**:
-   The config MUST be in json format.
-   Do NOT include any markdown formatting (like ```python ... ```) in your response. Just the code.
-   Handle potential errors gracefully if possible.

Write the code now.
"""

def get_feature_engineering_prompt(target: str):
    return f"""
Your task is to write a Python function `feature_engineering(df)` that takes a pandas DataFrame `df` as input and returns `X` (features DataFrame) and `y` (target Series/DataFrame).

You will be given:
- A description of a tabular dataset, including:
  - Column names and data types
  - Summary statistics (e.g., df.describe(include="all"))
  - Sample rows (head of the dataframe)
  - Any additional profiling information (missingness, unique counts, etc.)
- The name of the target column: {target}

Your task:
Write a robust Python function `feature_engineering(df)` that:
- Accepts a single argument `df` (a pandas DataFrame).
- Returns a tuple `(X, y)` where:
  - `X` is a pandas DataFrame containing the engineered features (without the target).
  - `y` is the target as a pandas Series (or DataFrame, if appropriate).

Follow these requirements carefully:

1. General structure
   - The function MUST be named `feature_engineering`.
   - It MUST accept exactly one argument: `df`.
   - It MUST return a tuple `(X, y)`.
   - Assume `pandas` and `numpy` are already imported as:
       import pandas as pd
       import numpy as np
   - Do not import any other libraries.
   - Do not print anything or log anything; just define the function.

2. Feature engineering (core of the task)
   - Use the dataset description, dtypes, and summary statistics provided to decide which new features are likely useful.
   - Create only *new* features. Do NOT perform:
     - Imputation
     - Scaling / normalization
     - One-hot encoding or other categorical encodings (except for the target)
   - Examples of allowed feature engineering:
     - Datetime columns:
       - If you detect datetime-like columns (actual datetime dtype or string columns clearly representing dates/times), extract components such as:
         - year, month, day, dayofweek, hour, etc.
       - Consider creating simple time-based features such as:
         - is_weekend, is_month_start, is_month_end.
     - Text columns:
       - For free-text/string columns, create simple aggregate features such as:
         - text length (number of characters)
         - word count (number of whitespace-separated tokens)
         - maybe basic boolean flags if something is clearly meaningful from the sample (e.g., presence of specific keywords), but keep it simple and generic.
     - Numeric columns:
       - Consider transformations when appropriate based on stats:
         - simple binning (e.g., qcut/cut) into a small number of bins for variables where that might help.
       - Consider advanced / domain specific interaction features that are intuitively meaningful from the column names and stats, such as:
         - ratios (e.g., column_A / (column_B + small_constant))
         - differences (e.g., column_A - column_B)
         - sums (e.g., column_A + column_B)
         - domain specific features if possible and have sense
       - Avoid generating an explosion of interaction features; focus on a small number of interpretable, high-value features.
     - Categorical-like columns (object, category):
       - You may create simple aggregate features like:
         - frequency / count encoding (e.g., map category -> frequency in df), if useful.
       - Do not perform one-hot encoding here; that will be handled later.
   - Avoid target leakage:
     - Do not create features that use information that would not be available at prediction time.
     - Do not aggregate across the whole dataset conditional on the target.

3. Data cleaning behavior
   - Drop exact duplicate rows if any exist: use something like df.drop_duplicates().
   - Do NOT perform:
     - Missing value imputation
     - Outlier removal
     - Scaling / normalization
     - Train/test splitting (that will be done later)
   - If you need to modify df, work on a copy (e.g., df = df.copy()) to avoid mutating the original.

4. Target handling
   - The target column name is: {target}
   - Separate the target column from the features:
     - y = df[target_column]
     - X = df.drop(columns=[target_column])
   - Target encoding:
     - If the target is numeric and appears to be regression-like (continuous), leave it as is.
     - If the target is binary or multiclass categorical:
       - Apply label encoding to the target only.
       - Do NOT use scikit-learn; use pandas / numpy only (for example, pd.factorize or astype("category").cat.codes).
       - Ensure the encoded target remains aligned with X.
       - Optionally include a short code comment documenting the mapping from original labels to encoded integers, if easily obtainable.

5. Error handling and robustness
   - Handle potential issues gracefully:
     - If the target column {target} is missing, raise a clear, informative ValueError with a helpful message.
     - If feature engineering steps rely on a column that is absent (e.g., due to prior filtering), check for existence before using it.
     - When parsing datetime from object columns, use safe conversions (e.g., errors="coerce") and only proceed with datetime feature extraction if a substantial portion of values are successfully parsed.
     - When computing ratios or log transforms, guard against division by zero and invalid values (e.g., use small epsilons, only log-transform positive values).
   - Do not let the function crash on common data issues if they can be anticipated and handled cleanly.

6. Output format
   - Output ONLY the Python code for the function (and any minimal helper functions you define inside the same file, if absolutely necessary).
   - Do NOT wrap the code in any markdown formatting such as ```python ... ``` or ```...```.
   - It is acceptable (and encouraged) to include:
     - A concise docstring for `feature_engineering`.
     - Inline comments explaining non-obvious logic.
   - Do NOT output any explanatory text outside of Python code.

7. Style and clarity
   - Follow standard Python style (PEP 8) as much as practical.
   - Keep the function readable and logically structured.
   - Prefer explicit, clear logic over clever but obscure one-liners.
   - Keep the number of engineered features reasonable and interpretable; prioritize quality over quantity.

Think carefully about the dataset description and statistics provided earlier in the conversation to decide:
- Which columns are dates, text, numeric, or categorical.
- Which variables may benefit from domain-agnostic feature transformations.
Then, implement `feature_engineering(df)` accordingly.

Remember: the final answer must be only valid Python code defining `feature_engineering(df)` and returning `(X, y)`, with no markdown or additional prose.
"""

def get_preprocessor_prompt(preprocessor_config):
    return f"""
Your task is to choose best methods for preprocessor for the dataset.

The preprocessor default config is (json format):
===
PREPROCESSOR_CONFIG = {json.dumps(preprocessor_config, indent=4)}
===

values you can choose:
dim_config = {{
    'none': None,
    'pca': PCA(n_components=5,svd_solver='auto'),
    'pca_10': PCA(n_components=10,svd_solver='auto'),
    'kpca': KernelPCA(n_components=5, kernel='rbf', gamma=1),
    'kpca_10': KernelPCA(n_components=10, kernel='rbf', gamma=0.1),
    'svd': TruncatedSVD(n_components=5),
    'svd_10': TruncatedSVD(n_components=10),
}}



numerical_imputer_config = {{
    'none': None,
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'constant': SimpleImputer(strategy='constant', fill_value=-1),
    'knn': KNNImputer(n_neighbors=5),
    'knn_10': KNNImputer(n_neighbors=10),
    'iterative': IterativeImputer(),
}}

numerical_scaler_config = {{
    'none': None,
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(feature_range=(0, 1)),
    'robust': RobustScaler(),
    'quantile': QuantileTransformer(output_distribution='normal'),
    'yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True),
    'box-cox': PowerTransformer(method='box-cox', standardize=True),
    'log': Logtransformer,
}}

cat_imputer_config = {{
    'none': None,
    'most_frequent': SimpleImputer(strategy='most_frequent'),
    'constant': SimpleImputer(strategy='constant', fill_value='Unknown')
}}

cat_encoder_config = {{
    'none': None,
    'onehot': OneHotEncoder(handle_unknown='ignore'),
    'target': TargetEncoder(smooth=0.1),
    'target_0.5': TargetEncoder(smooth=0.5),
    'target_1': TargetEncoder(smooth=1),
    'target_10': TargetEncoder(smooth=10),
    'hashing': FeatureHasher(n_features=10, input_type='string'),
    'label': LabelEncoder(),
    'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}}

cat_scaler_config = {{
    'none': None,
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(feature_range=(0, 1)),
    'robust': RobustScaler(),
    'quantile': QuantileTransformer(n_quantiles=10, random_state=0),
    'log': FunctionTransformer(np.log1p, validate=True),
}}

Based on this analysis, create the best preprocessor for the dataset using the values from the config.

**Constraints**:
-   The config MUST be in json format.
-   Do NOT include any markdown formatting (like ```python ... ``` or ```json ... ``` ) in your response. Just the code.
-   Handle potential errors gracefully if possible.

Write the code now.
"""

def get_models_prompt(gpu_available, gpu_model, cpu_count, verbose):
    return f"""
Your task is to enable/disable models and tune parameters for machine learning optimization with optuna for analyzed dataset.

GPU available: {gpu_available}
GPU model: {gpu_model}
CPU available cores: {cpu_count}

The model_config default is (json format):
===
MODELS_CONFIG = MODELS_CONFIG = {{
    'lightgbm': {{
        'enabled': True, # Enable/Disable LightGBM
        'optuna_trials': 10,                   # Number of hyperparameter search trials
        'optuna_timeout': 3600,                # Max time (seconds) for optimization
        'optuna_metric': 'rmse_safe',          # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': 1,                    # Parallel jobs for Optuna
        'params': {{                            # Fixed parameters (not optimized)
            'verbose': {verbose},
            'objective': 'rmse',               # Learning objective (e.g., regression:['mse','mae'], classification:['binary','multiclass'])
            'device': 'gpu',                   # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'rmse',             # Metric used for early stopping regression:['mse','mae'], classification:['auc','binary_error','neg_log_loss','multi_logloss', 'multi_error'])
            'num_threads': 8,                  # Threads for model training
            
        }},
        'optuna_params': {{                    # Hyperparameter search space
            'boosting_type': {{'type': 'categorical', 'choices': ['gbdt', 'goss']}},
            'lambda_l1': {{'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True}},
            'lambda_l2': {{'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True}},
            'num_leaves': {{'type': 'int', 'low': 20, 'high': 300}},
            'feature_fraction': {{'type': 'float', 'low': 0.4, 'high': 1.0}},
            'min_child_samples': {{'type': 'int', 'low': 5, 'high': 100}},
            'learning_rate': {{'type': 'float', 'low': 0.001, 'high': 0.2, 'log': True}},
            'min_split_gain': {{'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True}},
            'max_depth': {{'type': 'int', 'low': 3, 'high': 20}},
            'bagging_fraction': {{'type': 'float', 'low': 0.4, 'high': 1.0}},
            'bagging_freq': {{'type': 'int', 'low': 1, 'high': 7}},
            'num_class': 10,                   # Number of classes for multiclass classification (if task_type is 'classification')
        }}
    }},
    'xgboost': {{
    'enabled': True,                        # Enable/Disable XGBoost models
        'optuna_trials': 10,                # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,             # 3600 seconds = 1 hour 
        'optuna_metric': 'rmse_safe',       # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': 1,                 # Number of jobs to run in parallel
        'params': {{
            'verbose': {verbose},
            'objective': 'reg:squarederror',        # Learning objective (e.g., regression:['reg:squarederror','reg:absoluteerror'], classification:['binary:logistic','multi:softprob'])
            'device': 'gpu',                       # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'rmse',                  # Metric used for early stopping regression:['rmse','mae'], classification:['auc','error','logloss'])
            'nthread': 8,                # Threads for model training
        }},
        'optuna_params': {{                    # Hyperparameter search space
            'booster': {{'type': 'categorical', 'choices': ['gbtree']}},
            'max_depth': {{'type': 'int', 'low': 3, 'high': 20}},
            'learning_rate': {{'type': 'float', 'low': 0.001, 'high': 0.2, 'log': True}},
            'n_estimators': {{'type': 'int', 'low': 500, 'high': 3000}},
            'subsample': {{'type': 'float', 'low': 0, 'high': 1}},
            'lambda': {{'type': 'float', 'low': 1e-4, 'high': 5, 'log': True}},
            'gamma': {{'type': 'float', 'low': 1e-4, 'high': 5, 'log': True}},
            'alpha': {{'type': 'float', 'low': 1e-4, 'high': 5, 'log': True}},
            'min_child_weight': {{'type': 'categorical', 'choices': [0.5, 1, 3, 5]}},
            'colsample_bytree': {{'type': 'float', 'low': 0.5, 'high': 1}},
            'colsample_bylevel': {{'type': 'float', 'low': 0.5, 'high': 1}}
        }}
    }},
    'catboost': {{  
        'enabled': True,                        # Enable/Disable CatBoost models
        'optuna_trials': 10,                    # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                 # Time budget in seconds (1 hour)
        'optuna_metric': 'rmse_safe',           # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': 1,                     # Number of parallel Optuna jobs (studies running at once)
        'params': {{
            'verbose': {verbose},
            'objective': 'RMSE',                # Learning objective (e.g., regression:['RMSE','MAE'], classification:['Logloss','MultiClass'])
            'device': 'gpu',                    # Hardware acceleration ('cpu' or 'gpu')
            'eval_metric': 'RMSE',              # Metric used for early stopping regression:['mse','mae'], classification:['AUC','Accuracy','Logloss'])
            'thread_count': 8,                  # Threads for model training
        }},
        'optuna_params': {{                    # Hyperparameter search space
            'n_estimators': {{'type': 'int', 'low': 100, 'high': 3000}},
            'learning_rate': {{'type': 'float', 'low': 0.001, 'high': 0.2, 'log': True}},
            'depth': {{'type': 'int', 'low': 4, 'high': 10}},
            'l2_leaf_reg': {{'type': 'float', 'low': 1e-3, 'high': 10.0, 'log': True}},
            'border_count': {{'type': 'int', 'low': 32, 'high': 128}},
            'bootstrap_type': {{'type': 'categorical', 'choices': ['Bayesian', 'Bernoulli', 'MVS']}},
            'feature_border_type': {{'type': 'categorical', 'choices': ['Median', 'Uniform', 'GreedyMinEntropy']}},
            'leaf_estimation_iterations': {{'type': 'int', 'low': 1, 'high': 10}},
            'min_data_in_leaf': {{'type': 'int', 'low': 1, 'high': 30}},
            'random_strength': {{'type': 'float', 'low': 1e-9, 'high': 10, 'log': True}},
            'grow_policy': {{'type': 'categorical', 'choices': ['SymmetricTree', 'Depthwise', 'Lossguide']}},
            'subsample': {{'type': 'float', 'low': 0.6, 'high': 1.0}},
            'bagging_temperature': {{'type': 'float', 'low': 0, 'high': 1}},
            'max_leaves': {{'type': 'int', 'low': 2, 'high': 32}}
        }}
    }}   ,
    'random_forest': {{
        'enabled': True,                         # Enable/Disable Random Forest models
        'optuna_trials': 10,                     # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                  # 3600 seconds = 1 hour 
        'optuna_metric': 'rmse_safe',            # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': 1,                      # Number of jobs to run in parallel
        'params': {{
            'verbose': 1,
            'n_jobs': 8
        }},
        'optuna_params': {{                    # Hyperparameter search space for RF
            'n_estimators': {{'type': 'int', 'low': 100, 'high': 1000}},
            'max_depth': {{'type': 'int', 'low': 3, 'high': 30}},
            'min_samples_split': {{'type': 'int', 'low': 2, 'high': 20}},
            'min_samples_leaf': {{'type': 'int', 'low': 1, 'high': 10}},
            'max_features': {{'type': 'categorical', 'choices': ['sqrt', 'log2', None]}},
            'bootstrap': {{'type': 'categorical', 'choices': [True, False]}},
            'max_samples': {{'type': 'float', 'low': 0.5, 'high': 1.0}}
        }}
    }}   ,
    'pytorch': {{
        'enabled': True,                         # Enable/Disable PyTorch models
        'optuna_trials': 10,                     # Number of trials for Optuna hyperparameter optimization
        'optuna_timeout': 3600,                  # 3600 seconds = 1 hour 
        'optuna_metric': 'rmse_safe',            # Metric to optimize during Optuna trials (e.g., 'acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape')
        'optuna_n_jobs': 1,                      # Number of jobs to run in parallel
        'params': {{
            "train_max_epochs": 50,                 # Number of epochs to train for
            "train_patience": 5,                    # Number of epochs to wait before early stopping
            "final_max_epochs": 1000,               # Number of epochs to train for
            "final_patience": 20,                   # Number of epochs to wait before early stopping
            "objective": "mse",                     # objective (e.g., regression:['mse','mae','huber','rmsle','mape'], classification:['bce','bcelogit','crossentropy']
            "device": 'gpu',                        # 'cpu', 'gpu'
            'verbose': 1,
            'embedding_info': ['time_of_day'],      # embedding info: should be list of strings of categorical columns with high cardinality 
            'num_threads': 8,
        }}   ,
        'optuna_params': {{                                                                      
            # Hyperparameter search space for PyTorch models
            'model_type': {{'type': 'categorical', 'choices': ['mlp']}},                         #model type: ['mlp', 'ft_transformer'] 
            'optimizer_name': {{'type': 'categorical', 'choices': ['adam']}},                    #optimizer: ['adam', 'nadam', 'adamax', 'adamw', 'sgd', 'rmsprop] 
            'learning_rate': {{'type': 'categorical', 'choices': [0.01]}},                       #learning rate: ['0.01', '0.001'] 
            'batch_size': {{'type': 'categorical', 'choices': [64, 128, 256]}},                  #batch size: ['64', '128', '256'] 
            'weight_init': {{'type': 'categorical', 'choices': ['default']}},                    #weight initialization: ['default', 'xavier', 'kaiming'] 
            'net': {{'type': 'categorical', 'choices': [                                         
                # MLP ReLU without batch or layer norm
                [
                    {{'type': 'dense', 'out_features': 16, 'activation': 'relu', 'norm': None}},
                    {{'type': 'dropout', 'p': 0.1}},
                    {{'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}}
                ],
                # MLP GELU with batch norm
                [
                    {{'type': 'dense', 'out_features': 32, 'activation': 'gelu', 'norm': 'batch_norm'}},
                    {{'type': 'dropout', 'p': 0.1}},
                    {{'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}}
                ],
                # MLP Swish/SILU with layer norm
                [
                    {{'type': 'dense', 'out_features': 64, 'activation': 'swish', 'norm': 'layer_norm'}},
                    {{'type': 'dropout', 'p': 0.1}},
                    {{'type': 'dense', 'out_features': 32, 'activation': 'swish', 'norm': 'layer_norm'}},
                    {{'type': 'dropout', 'p': 0.1}},
                    {{'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}}
                ],

            ]}},
            # FT-Transformer Params
            'd_token': {{'type': 'categorical', 'choices': [64, 128, 192, 256]}},
            'n_layers': {{'type': 'int', 'low': 1, 'high': 4}},
            'n_heads': {{'type': 'categorical', 'choices': [4, 8]}},
            'd_ffn_factor': {{'type': 'float', 'low': 1.0, 'high': 2.0}},
            'attention_dropout': {{'type': 'float', 'low': 0.0, 'high': 0.3}},
            'ffn_dropout': {{'type': 'float', 'low': 0.0, 'high': 0.3}},
            'residual_dropout': {{'type': 'float', 'low': 0.0, 'high': 0.2}}
        }}   
    }}
}}
===

Based on the analysis, create the best configuration for mentioned earlier models and analyzed dataset.

The eval metrics list from you can choose for MODELS_CONFIG:
===
EVAL_METRICS = ['acc', 'f1', 'auc', 'prec', 'mse', 'rmse', 'msle', 'rmsle', 'rmsle_safe', 'rmse_safe', 'mae', 'mape']
===

**Constraints**:
-   The config MUST be in json format.
-   Important: Do NOT remove / change any keys in json.
-   Only change values if you think it is needed.
-   If possible for GPU use it, if not use CPU. values: 'cpu', 'gpu'
-   When set params for catboost if choose device = 'gpu' use bootstrap_type = 'Bayesian' or 'Bernoulli'
-   When multiclass are selected for lightgbm set num_class = x (where x is number of classes)
-   Always choose params from scope provided in comments as it is. eg if name is 'adamw' use 'adamw' not 'adamW'
-   For Pytorch on output layer do not use batch_layer or layer_norm.
-   Do NOT include any markdown formatting (like ```python ... ``` or ```json ... ``` ) in your response. Just the code.
-   Handle potential errors gracefully if possible.
"""

def get_ensamble_prompt(stacking_config, voting_config):
    return f"""
Your task is to enable/disable ensamble models (stacking / voting) and tune parameters for ensamble models for analyzed dataset.

The ensamble config default is (json format):
===
ENSAMBLE_CONFIG = {{
    'stacking': { json.dumps(stacking_config, indent=10) },
    'voting': { json.dumps(voting_config, indent=10) }
}}

Params Stacking:
'cv_enabled': bool,                                                             # Enable stacking on models from Optuna Cross-Validation best trial if choosen eg. lightbm stacking is builded on all lightbm models from best cv trial during optuna optimalization 
'cv_folds': int,                                                                # Folds for stacking CV (if not using prefit)
'cv_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],       # Models for stacking CV
'final_enabled': bool,                                                          # Enable stacking for the final model
'final_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],    # Models for stacking final models
'meta_model': str,                                                              # The model used to aggregate base model predictions
'use_features': bool,                                                           # If False the meta-model will only learn from the predictions of the base models, not the original features.
'prefit': bool,                                                                 # If True, uses existing trained models (faster). If False, retrains.

Params Voting:
'cv_enabled': bool,                                                             # Enable voting on finals models builded during optuna optimalization 
'cv_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],       # Models for voting CV
'final_enabled': bool,                                                          # Enable voting ensemble for the final model
'final_models': ['lightgbm','xgboost','catboost','random_forest','pytorch'],    # Models for voting final models
'use_features': bool,                                                           # If False the meta-model will only learn from the predictions of the base models, not the original features.
'prefit': bool,                                                                 # If True, uses existing trained models (faster). If False, retrains.


===

Based on the analysis, create the best configuration for mentioned earlier ensamble models for analyzed dataset.

**Constraints**:
-   The config MUST be in json format.
-   Do NOT remove / change / add any keys in json, only change values if you think it is needed.
-   Do NOT include any markdown formatting (like ```python ... ```) in your response. Just the code.
-   Handle potential errors gracefully if possible.
"""

def get_results_prompt(scores_display):
    return f"""
Your task is to perform a comprehensive, **expert-level analysis** of tuned models and their optimization results.

**Context**

The user has run an Optuna study (or multiple studies) to tune one or more machine learning models.
All scores have been computed using **cross-validation** inside the Optuna optimization loop.

You are given:

- A description of the experiment and objective (if provided).
- One or more evaluation metrics per model.
- Aggregated scores (e.g., mean and std across folds) and/or per-trial/per-fold scores.

**Scores for models and metrics**
(Use this as your primary evidence.)

{json.dumps(scores_display, indent=10)}

This `scores_display` contain:
- Model names / identifiers
- Metric names (e.g., accuracy, F1, RMSE, AUC, logloss, etc.)
- Whether the metric is being minimized or maximized (if provided)
- Cross-validation fold scores
- Hyperparameters for each trial or for the best trials

---

### Your Objectives

Analyze these results and produce a **clear, structured, and critical** written report that covers:

1. **Experiment & Metric Interpretation**
   - Briefly restate what appears to be the **objective** of the optimization (e.g., “maximize F1”, “minimize RMSE”), based on the data and/or any explicit description.
   - Explain **how to read the scores**:
     - What each metric means in plain language.
     - Whether **higher or lower is better** for each metric.
     - How cross-validation affects the reliability of the scores (e.g., mean vs. standard deviation, variance across folds).

2. **Model Comparison & Leaderboard**
   - Identify the **best model(s)** according to the primary objective metric.
   - Justify **why** it is the best:
     - Reference the relevant metric(s) and their values.
     - Consider both **average performance** and **stability** (e.g., standard deviation across folds).
   - If multiple models are competitive, explain the **trade-offs** (e.g., one slightly better on accuracy, another with lower variance or better calibration).
   - If you build a **leaderboard**, **present it as a bullet list, NOT a table**, ordered from best to worst, e.g.:

     - 1. Model A - main metric: X (mean ± std), key strengths/weaknesses…
     - 2. Model B - main metric: X (mean ± std), key strengths/weaknesses…
     - 3. Model C - main metric: X (mean ± std), key strengths/weaknesses…

3. **Insights & Patterns**
   - Extract **meaningful insights** from the results, for example:
     - Performance differences between model families (e.g., tree-based vs. linear vs. neural).
     - Sensitivity to particular hyperparameters (if visible).
     - Overfitting/underfitting signals (e.g., very high variance across folds).
   - For each important insight, **explicitly justify why it makes sense** in terms of:
     - The metrics
     - The cross-validation results
     - Basic ML intuition (e.g., “this model tends to work better on small tabular datasets”, “high variance suggests the model is sensitive to data splits”)
   - If appropriate, mention any **limitations** in the results (e.g., small number of trials, very close scores leading to uncertainty about the true best model).

4. **Recommendations**
   - Provide **clear, actionable recommendations** for the next steps. For example:
     - Which model configuration to select as the **current best** for deployment or further testing.
     - Suggestions for **further tuning** (e.g., promising hyperparameter ranges to explore).
     - Suggestions for **additional evaluation** (e.g., test set evaluation, alternative metrics, calibration checks, fairness checks).
   - When there is uncertainty (e.g., multiple models close in performance), explain how the user might **resolve it** (e.g., more trials, different metric, larger validation set).

5. **Interpretability for a Non-Expert Stakeholder (Optional but Preferred)**
   - Where possible, phrase key conclusions in a way that could be understood by a **technical but non-ML-specialist** reader.
   - Avoid excessive jargon; briefly explain any unavoidable technical terms.

---

### Style & Formatting Requirements

- Use **Markdown** formatting with clear sections, for example:

  - `## Summary`
  - `## How to Read the Metrics`
  - `## Model Comparison`
  - `## Key Insights`
  - `## Recommendations`
  - `## Limitations & Next Steps`

- Do **not** use tables for the leaderboard; use an **ordered or unordered list** instead.
- Be **concise but thorough**: focus on the most important findings, not on repeating every raw number.
- Make sure every major claim is **supported by specific metrics or patterns**.
- **Do not fabricate** metrics, models, or details that are not present.
- If some information that would normally be needed is **missing or ambiguous**, explicitly state this and proceed with a **best-effort analysis** based on what is available.

---

### Error Handling & Robustness

- If the structure of scores is unclear or partially inconsistent:
  - Acknowledge the issue briefly.
  - Explain how you are interpreting the available information.
  - Continue the analysis in a **graceful, best-effort** manner, instead of failing.
- If you cannot determine the optimization direction (minimize/maximize) from the context, infer it from metric names when reasonable (e.g., “error”, “loss”, “rmse” → lower is better; “accuracy”, “f1”, “auc” → higher is better) and state your assumption.

---

Now, based on scores, produce the full analysis.

---

"""
