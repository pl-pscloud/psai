from dotenv import load_dotenv
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root (assuming psai is one level deep)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from typing import Literal, TypedDict, List, Dict, Any, Union, Tuple
import os
import json
import pandas as pd
import numpy as np
import re
import random
import copy
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from psai.datasets import EDAReport
from psai.config import CONFIG
from psai.psml import psML

class DataScientist:
    def __init__(self, model_name: str = "gemini-3-pro-preview"):
        """
        Initialize the DataScientist agent.
        
        Args:
            model_name (str): The name of the Gemini model to use.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
        self.llm_memory = InMemorySaver()
        self.eda_report = None
        self.session = random.randint(100000, 999999)
        self.config = {"configurable": {"thread_id": self.session}}
        self.system_prompt = """
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
        self.agent = create_react_agent(
            model=self.llm,
            tools=[],
            checkpointer=self.llm_memory,
            prompt=self.system_prompt,
            name="data_scientist_agent",
        )

    def _final_message_text(self, result: Dict[str, Any]) -> str:
        try:
            messages = result.get("messages") or []
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Extract concatenated text from content blocks
                    parts = []
                    for block in content:
                        try:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(str(block.get("text", "")))
                        except Exception:
                            continue
                    if parts:
                        return "".join(parts)
                    # Fallback to string of content list
                    return str(content)
        except Exception:
            pass
        return str(result)

        
    def consulteda_summary(self, df: pd.DataFrame, target: str) -> str:
        """
        Generates a text summary of the EDA report to feed into the LLM.
        """
        print("Generating EDA summary...")
        report = EDAReport(df, target)
        # Run analyses to populate report_content
        report.basic_info()
        report.numerical_analysis()
        report.categorical_analysis()
        report.correlation_analysis()
        
        summary = []
        for item in report.report_content:
            if item['type'] == 'header':
                summary.append(f"\n## {item['content']}")
            elif item['type'] == 'text':
                summary.append(item['content'])
            elif item['type'] == 'table':
                if item.get('title'):
                    summary.append(f"\n### {item['title']}")
                # Convert dataframe to markdown string for the LLM
                if isinstance(item['content'], pd.DataFrame):
                    summary.append(item['content'].to_markdown())
                else:
                    summary.append(str(item['content']))
            # Skip plots for text summary
        
        eda_prompt = f"""
Your task is to make comprehensive, professional textual analyse of results of EDA.

The target column is: '{target}'

Here is an analysis of the dataset:
{summary}

Based on this eda summary, perform the following:

## Analysis and Key Findings (Textual)

Based on the summary above, provide a structured analysis:

### Data Quality Assessment
* Identify any high-cardinality categorical features.
* Note features with significant **missing values (NaNs)** and suggest an imputation strategy (e.g., mean/median for numerical, mode for categorical, or dropping the column).
* Report the presence of **duplicates** and **outliers** if flagged in the summary.

### Feature Engineering
* Suggest feature engineering steps based on the EDA results.
* Suggest feature scaling and encoding strategies.

### Preprocessing
* Suggest a preprocessing pipeline based on the EDA results.

### Models Selection
* Suggest a models based on the EDA results.

### Explainability
* Explain in details why you propose some steps, how it works and what user can expect from it.

"""

        # 4. Get Code from LLM
        print("Consulting LLM for EDA...")
        response = self.agent.invoke({"messages": [HumanMessage(content=eda_prompt)]}, config=self.config)
        eda_analysis =  self._final_message_text(response)
        
        print("-" * 40)
        print("Generated EDA Analysis:")
        print("-" * 40)
        print(eda_analysis)
        print("-" * 40)

            
        self.eda_summary = eda_analysis.join("\n").join(summary)
        print("EDA summary generated.")

    def consult_feature_engineering(self, dataset: Union[str, pd.DataFrame], target: str, execute_code: bool = False):
        """
        Analyzes the dataset and performs feature engineering using LLM-generated code.
        
        Args:
            dataset (Union[str, pd.DataFrame]): Path to the dataset csv or a pandas DataFrame.
            target (str): The name of the target column.
            
        Returns:
            Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]: X (features) and y (target).
        """
        # 1. Load Data
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                df = pd.read_csv(dataset)
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset}")
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise ValueError("Dataset must be a file path or a pandas DataFrame.")
            
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")


        # 2. Generate EDA Summary if needed
        if self.eda_summary is None:
            self.eda_summary = self.consulteda_summary(df, target)

        
        # 3. Prompt Engineering
        feature_engineering_prompt = f"""
Your task is to write a Python function `feature_engineering(df)` that takes a pandas DataFrame `df` as input and returns `X` (features DataFrame) and `y` (target Series/DataFrame).

Based on this analysis, perform the following in your code:
1.  **Feature Engineering**: Create new features that might be useful for prediction (e.g., interactions, binning, extraction from text/dates).
2.  **Drop Duplicates**: If any.
3.  **Split X and y**: Separate the target from the features. Drop the target column from X.

**Constraints**:
-   Imputing, scaling, one-hot encoding etc will be done in preprocessor step, not in feature engineering, add only new features.
-   The function MUST be named `feature_engineering`.
-   It MUST accept a single argument `df`.
-   It MUST return a tuple `(X, y)`.
-   Use `pandas` and `numpy`. Assume they are imported as `pd` and `np`.
-   Do NOT include any markdown formatting (like ```python ... ```) in your response. Just the code.
-   Handle potential errors gracefully if possible.

Write the code now.
"""
        
        # 4. Get Code from LLM
        print("Consulting LLM for Feature Engineering...")
        response = self.agent.invoke({"messages": [HumanMessage(content=feature_engineering_prompt)]}, config=self.config)
        code_feature_engineering =  self._final_message_text(response)        
        
        print("-" * 40)
        print("Generated Feature Engineering Code:")
        print("-" * 40)
        print(code_feature_engineering)
        print("-" * 40)
                
        
        try:
            if execute_code:
                # 5. Execute Code
                print("Executing generated code...")
                local_scope = {'pd': pd, 'np': np, 'target': target} # Pass target to local scope
                # Clean code (remove markdown blocks if the LLM ignored instructions)
                code = re.sub(r'^```python\s*', '', code_feature_engineering, flags=re.MULTILINE)
                code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
                code = code.strip()
                exec(code, {}, local_scope)
            else:
                print("Code not executed. Returning code instead.")
                return code_feature_engineering
            
            if 'feature_engineering' not in local_scope:
                raise ValueError("The generated code did not define a 'feature_engineering' function.")
            
            feature_engineering_func = local_scope['feature_engineering']
            X, y = feature_engineering_func(df)
            
            print(f"Feature Engineering complete. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error executing generated code: {e}")
            raise

    def consult_preprocessor(self, dataset: Union[str, pd.DataFrame], target: str, execute_code: bool = False):
        """
        Consults the LLM to generate a preprocessor configuration.
        
        Args:
            dataset (Union[str, pd.DataFrame]): Path to the dataset csv or a pandas DataFrame.
            target (str): The name of the target column.
            execute_code (bool): If True, attempts to parse the LLM's response as JSON. If False, returns the raw string.
            
        Returns:
            Union[Dict[str, Any], str]: The generated preprocessor configuration as a dictionary if `execute_code` is True and parsing succeeds, otherwise the raw string response from the LLM.
        """
        # 1. Load Data
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                df = pd.read_csv(dataset)
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset}")
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise ValueError("Dataset must be a file path or a pandas DataFrame.")
            
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        if self.eda_summary is None:
            self.eda_summary = self.consulteda_summary(df, target)
        
        

        preprocessor_config = {
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
                'method': 'none',
            },
        }

        preprocessor_prompt = f"""
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
-   Do NOT include any markdown formatting (like ```python ... ```) in your response. Just the code.
-   Handle potential errors gracefully if possible.

Write the code now.
"""
        
        # 4. Get Code from LLM
        print("Consulting LLM for Preprocessor...")
        response = self.agent.invoke({"messages": [HumanMessage(content=preprocessor_prompt)]}, config=self.config)
        code_preprocessor =  self._final_message_text(response)       
        
        
        
        print("-" * 40)
        print("Generated Preprocessor Code:")
        print("-" * 40)
        print(code_preprocessor)
        print("-" * 40)
        
        try:
            if execute_code:
                print("Executing generated config to create preprocessor...")
                return json.loads(code_preprocessor)
            else:
                print("Code not executed. Returning preprocessor config (dict).")
                return code_preprocessor
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return code_preprocessor

    def consult_models(self, dataset: Union[str, pd.DataFrame], target: str, execute_code: bool = False):
        """
        Consults the LLM to generate a models configuration.
        
        Args:
            dataset (Union[str, pd.DataFrame]): Path to the dataset csv or a pandas DataFrame.
            target (str): The name of the target column.
            execute_code (bool): If True, attempts to parse the LLM's response as JSON. If False, returns the raw string.
            
        Returns:
            Union[Dict[str, Any], str]: The generated models configuration as a dictionary if `execute_code` is True and parsing succeeds, otherwise the raw string response from the LLM.
        """
        # 1. Load Data
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                df = pd.read_csv(dataset)
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset}")
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise ValueError("Dataset must be a file path or a pandas DataFrame.")
            
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        if self.eda_summary is None:
            self.eda_summary = self.consulteda_summary(df, target)
        
        optuna_trials = 10                                   # Number of trials for Optuna hyperparameter optimization
        optuna_n_jobs = 1                                   # Number of parallel Optuna jobs (studies running at once)
        optuna_metric = 'rmse_safe'                         # Metric to optimize during Optuna trials (e.g., 'rmse', 'auc')
        model_n_jobs = int(os.cpu_count() / optuna_n_jobs)  # Number of threads per model (CPU cores / optuna jobs)
        device = 'gpu'                                      # Device to use for training ('cpu' or 'gpu')
        verbose = 2                                         # Verbosity level (0: silent, 1: minimal, 2: detailed)
        models_enabled = {                                  # Master toggle to enable/disable specific models
            'lightgbm': True,
            'xgboost': False,
            'catboost': False,
            'random_forest': False,
            'pytorch': True,
            'stacking': False,
            'voting': False,
        }

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
                    'num_threads': model_n_jobs        # Threads for model training
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
                    'bagging_freq': {'type': 'int', 'low': 1, 'high': 7}
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
                'optuna_trials': optuna_trials/2,             # Number of trials for Optuna hyperparameter optimization
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
                    'num_threads': model_n_jobs,
                },
                'optuna_params': {                         # Hyperparameter search space for PyTorch models
                    'model_type': {'type': 'categorical', 'choices': ['mlp','ft_transformer']},        #['mlp', 'ft_transformer'] model type
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

        models_prompt = f"""
Your task is to enable/disable models and tune them for machine learning optimization with optuna for the dataset.

The model_config default is (json format):
===
MODELS_CONFIG = {json.dumps(MODELS_CONFIG, indent=10)}
===

Based on the analysis, create the best configuration for mentioned earlier models and analyzed dataset.


**Constraints**:
-   The config MUST be in json format.
-   Do NOT remove / change / add any keys in json, only change values if you think it is needed.
-   Do NOT include any markdown formatting (like ```python ... ```) in your response. Just the code.
-   Handle potential errors gracefully if possible.
"""
        
        # 4. Get Code from LLM
        print("Consulting LLM for Models Config...")
        response = self.agent.invoke({"messages": [HumanMessage(content=models_prompt)]}, config=self.config)
        code_models =  self._final_message_text(response)       
        
        
        
        print("-" * 40)
        print("Generated Models Config:")
        print("-" * 40)
        print(code_models)
        print("-" * 40)
        
        try:
            if execute_code:
                print("Executing generated config to create models...")
                return json.loads(code_models)
            else:
                print("Code not executed. Returning models config (dict).")
                return code_models
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return code_models

    def end_to_end_ml_process(self, dataset: Union[str, pd.DataFrame], target: str, execute_code: bool = True) -> psML:
        """
        End-to-end machine learning process.
        
        Args:
            dataset (Union[str, pd.DataFrame]): Path to the dataset csv or a pandas DataFrame.
            target (str): The name of the target column.
            execute_code (bool): Whether to execute the generated code.
            
        Returns:
            psML: The trained psML object.
        """
        # 1. Load Data
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                df = pd.read_csv(dataset)
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset}")
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise ValueError("Dataset must be a file path or a pandas DataFrame.")
            
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        # 2. EDA
        print("\n" + "="*40)
        print(" STEP 1: Exploratory Data Analysis (EDA) ")
        print("="*40 + "\n")
        self.consulteda_summary(df, target)

        # 3. Feature Engineering
        print("\n" + "="*40)
        print(" STEP 2: Feature Engineering ")
        print("="*40 + "\n")
        # consult_feature_engineering returns X, y
        X, y = self.consult_feature_engineering(df, target, execute_code=execute_code)

        # 4. Preprocessing
        print("\n" + "="*40)
        print(" STEP 3: Preprocessing Configuration ")
        print("="*40 + "\n")
        preprocessor_config = self.consult_preprocessor(df, target, execute_code=execute_code)

        # 5. Model Selection
        print("\n" + "="*40)
        print(" STEP 4: Model Selection & Configuration ")
        print("="*40 + "\n")
        models_config = self.consult_models(df, target, execute_code=execute_code)

        # 6. Configuration Merging
        print("\n" + "="*40)
        print(" STEP 5: Configuring PSML ")
        print("="*40 + "\n")
        
        run_config = copy.deepcopy(CONFIG)
        run_config['dataset']['target'] = target
        
        if isinstance(preprocessor_config, dict):
             run_config['preprocessor'] = preprocessor_config
        
        if isinstance(models_config, dict):
            run_config['models'] = models_config

        # 7. Execution
        print("\n" + "="*40)
        print(" STEP 6: Model Training & Optimization ")
        print("="*40 + "\n")
        
        model = psML(config=run_config, X=X, y=y)
        
        print("Optimizing models...")
        model.optimize_all_models()
        
        print("Building Ensembles...")
        model.build_ensemble_cv()
        model.build_ensemble_final()
        
        print("\n" + "="*40)
        print(" Final Scores ")
        print("="*40 + "\n")
        model.scores()
        
        return model