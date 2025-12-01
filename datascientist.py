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
import markdown
import html
import time

from psai.datasets import EDAReport
from psai.config import CONFIG
from psai.psml import psML

class DataScientist:
    def __init__(self, df: pd.DataFrame, target: str, model_provider: str = "google", model_name: str = "gemini-3-pro-preview", model_temperature: float = 1.0,  api_key: str = None, optuna_metric: str = None, optuna_trials: int = None, task_type: str = None, experiment_name: str = None):
        """
        Initialize the DataScientist agent.
        
        Args:
            model_name (str): The name of the Gemini model to use.
        """
        if api_key is None:
            api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in environment variables. Please check your .env file.")


        if model_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=model_temperature, 
                google_api_key=api_key
            )
        elif model_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model = model_name,
                temperature = model_temperature,
                max_retries = 2,
                api_key = api_key,
            )
        elif model_provider == "antrophic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=model_temperature,
                max_retries=2,
                api_key=api_key,
            )
        elif model_provider == "ollama":
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=model_name,
                validate_model_on_init=True,
                temperature=model_temperature,
            )
        else:
            raise ValueError(f"Provider '{model_provider}' not known, use: ['google','openai','antrophic','ollama']")
        

        self.llm_memory = InMemorySaver()
        self.df = df
        self.target = target
        self.X = None
        self.y = None
        self.eda_report = None
        self.ai_eda_summary = None
        self.ai_dataset_config = None
        self.ai_preprocessor = None
        self.ai_feature_engineering_code = None
        self.ai_models_config = None
        self.ai_ensamble_config = None
        self.ai_results_analysis = None
        self.run_config = None
        self.psml = None
        self.session = random.randint(100000, 999999)
        self.config = {"configurable": {"thread_id": self.session}}
        self.optuna_metric = optuna_metric
        self.optuna_trials = optuna_trials
        self.task_type = task_type
        self.experiment_name = experiment_name
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


        # 4. Get Code from LLM
        print("Saying hello to AI DataScientist...\n")
        response = self.agent.invoke({"messages": [HumanMessage(content="Hello!")]}, config=self.config)
        self._final_message_text(response)
        print(self._final_message_text(response))

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

    def run_analysis(self):

        if self.X is None or self.y is None:
            raise ValueError("X and y must be set before calling analyze.")

        self._configuring_psml()

        if self.run_config is None:
            raise ValueError("run_config must be set before calling analyze.")
        
        self.psml = psML(config=self.run_config, X=self.X, y=self.y, experiment_name=self.experiment_name)
        
        print("Optimizing models...")
        self.psml.optimize_all_models()
        

    def run_analysis_ensamble(self):

        self._configuring_psml()
        
        print("Building Ensembles...")
        self.psml.build_ensemble_cv()
        self.psml.build_ensemble_final()

        
    def consult_eda(self) -> str:
        """
        Generates a text summary of the EDA report to feed into the LLM.
        """
        print("Generating EDA summary...")
        report = EDAReport(self.df, self.target)
        self.eda_report = report # Store report for later use
        # Run analyses to populate report_content
        report.basic_info()
        report.numerical_analysis()
        report.categorical_analysis()
        report.correlation_analysis()

        optuna_metrics_info = f"""The optuna_metric is: '{self.optuna_metric}'""" if self.optuna_metric else ""
        optuna_trials_info = f"""Suggested trails for optuna is: '{self.optuna_trials}'""" if self.optuna_trials else ""
        task_type_info = f"""The task_type is: '{self.task_type}'""" if self.task_type else ""
        
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

The target column is: '{self.target}'

{task_type_info}

{optuna_metrics_info}

{optuna_trials_info}

Here is an analysis of the dataset:
{summary}


Based on this eda summary, perform the following:

## Analysis and Key Findings (Textual)

Based on the summary above, provide a structured analysis:

### Data Quality Assessment
* Note features with significant **missing values (NaNs)** and suggest an imputation strategy (e.g., mean/median for numerical, mode for categorical, or dropping the column).
* Report the presence of **duplicates** and **outliers** if flagged in the summary.
* Identify features which are skewd, outliers or have high-cardinality and suggest a strategy to handle them.

### Feature Engineering
* Suggest feature engineering steps based on the EDA results.
* Suggest feature scaling and encoding strategies.

### Preprocessing
* Suggest a preprocessing pipeline based on the EDA results.
* Explain  why you suggest this preprocessing and explain how choosen preprocessing are related to EDA results and how tranformation help models to perform better.

### Models Selection
* Suggest a models based on the EDA results.
* Explain  why you suggest this models and explain how choosen models are related to EDA results.

### Explainability
* Explain in details why you propose some steps, how it works and what user can expect from it.

### CoT
* Deep think about your analysis and possible best practices.
* Provide a clear chain of thought (CoT) for your analysis.

"""

        # 4. Get Code from LLM
        print("Consulting LLM for EDA...")
        response = self.agent.invoke({"messages": [HumanMessage(content=eda_prompt)]}, config=self.config)
        self.ai_eda_summary =  self._final_message_text(response)
        
        print("-" * 40)
        print("Generated EDA Analysis:")
        print("-" * 40)
        print(self.ai_eda_summary)
        print("-" * 40)
            
        print("EDA summary generated.")

    def consult_dataset_config(self, execute_code: bool = False) -> str:
        """
        Generates a text summary of the EDA report to feed into the LLM.
        """
        print("Generating dataset config...")
        
        dataset_config_prompt = f"""
Your task is to suggest a dataset config for the dataset.

The target column is: '{self.target}'

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

        # 4. Get Code from LLM
        print("Consulting LLM for dataset config...")
        response = self.agent.invoke({"messages": [HumanMessage(content=dataset_config_prompt)]}, config=self.config)
        dataset_config =  self._final_message_text(response)
        
        print("-" * 40)
        print("Generated dataset config:")
        print("-" * 40)
        print(dataset_config)
        print("-" * 40)
            
        try:
            self.ai_dataset_config = json.loads(dataset_config)
            if execute_code:
                print("Executing generated config to create dataset config...")
                return self.ai_dataset_config
            else:
                print("Code not executed. Returning dataset config (dict).")
                return dataset_config
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return dataset_config

    def consult_feature_engineering(self, execute_code: bool = False):
        """
        Analyzes the dataset and performs feature engineering using LLM-generated code.
        
        Args:
            dataset (Union[str, pd.DataFrame]): Path to the dataset csv or a pandas DataFrame.
            target (str): The name of the target column.
            
        Returns:
            Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]: X (features) and y (target).
        """
        # 1. Load Data
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")


        # 2. Generate EDA Summary if needed
        if self.ai_eda_summary is None:
            self.ai_eda_summary = self.consulteda_summary(self.df, self.target)

        
        # 3. Prompt Engineering
        feature_engineering_prompt = f"""
Your task is to write a Python function `feature_engineering(df)` that takes a pandas DataFrame `df` as input and returns `X` (features DataFrame) and `y` (target Series/DataFrame).

Based on this analysis, perform the following in your code:
1.  **Feature Engineering**: Create new features that might be useful for prediction (e.g., interactions, binning, extraction from text/dates).
2.  **Drop Duplicates**: If any.
3.  **Target Label Encoding**: If multiclass target, encode it using label encoding.
4.  **Split X and y**: Separate the target from the features. Drop the target column from X.

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
            self.ai_feature_engineering_code = code_feature_engineering
            if execute_code:
                # 5. Execute Code
                print("Executing generated code...")
                local_scope = {'pd': pd, 'np': np, 'target': self.target} # Pass target to local scope
                # Clean code (remove markdown blocks if the LLM ignored instructions)
                code = re.sub(r'^```python\s*', '', code_feature_engineering, flags=re.MULTILINE)
                code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
                code = code.strip()
                exec(code, local_scope)
            else:
                print("Code not executed. Returning code instead.")
                return code_feature_engineering
            
            if 'feature_engineering' not in local_scope:
                raise ValueError("The generated code did not define a 'feature_engineering' function.")
            
            feature_engineering_func = local_scope['feature_engineering']
            self.X, self.y = feature_engineering_func(self.df)
            
            print(f"Feature Engineering complete. X shape: {self.X.shape}, y shape: {self.y.shape}")
            return self.X, self.y
            
        except Exception as e:
            print(f"Error executing generated code: {e}")
            raise

    def consult_preprocessor(self, execute_code: bool = False):
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
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

        if self.ai_eda_summary is None:
            self.ai_eda_summary = self.consulteda_summary(self.df, self.target)
        
        

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
            self.ai_preprocessor = json.loads(code_preprocessor)
            if execute_code:
                print("Executing generated config to create preprocessor...")
                return self.ai_preprocessor
            else:
                print("Code not executed. Returning preprocessor config (dict).")
                return code_preprocessor
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return code_preprocessor

    def consult_models(self, execute_code: bool = False):
        """
        Consults the LLM to generate a models configuration.
        
        Args:
            df (pd.DataFrame): A pandas DataFrame.
            target (str): The name of the target column.
            execute_code (bool): If True, attempts to parse the LLM's response as JSON. If False, returns the raw string.
            
        Returns:
            Union[Dict[str, Any], str]: The generated models configuration as a dictionary if `execute_code` is True and parsing succeeds, otherwise the raw string response from the LLM.
        """

        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

        if self.ai_eda_summary is None:
            self.ai_eda_summary = self.consulteda_summary(self.df, self.target)

        from psai.config import CONFIG as MODELS_CONFIG

        optuna_n_jobs = 1        
        cpu_count = os.cpu_count()
        model_n_jobs = int(cpu_count / optuna_n_jobs)           # Number of threads per model (CPU cores / optuna jobs)
        
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_model = torch.cuda.get_device_name(0) if gpu_available else None
        device = 'gpu' if gpu_available else 'cpu'              # Device to use for training ('cpu' or 'gpu')

        print("GPU available: ", gpu_available)
        print("GPU model: ", gpu_model)
        print("CPU available cores: ", cpu_count)
        print("\n")
        
        verbose = 2                                             # Verbosity level (0: silent, 1: minimal, 2: detailed)
        models_enabled = {                                      # Master toggle to enable/disable specific models
            'lightgbm': False,
            'xgboost': False,
            'catboost': False,
            'random_forest': False,
            'pytorch': False,
            'stacking': False,
            'voting': False,
        }

        models_prompt = f"""
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
            'verbose': verbose,
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
            'verbose': verbose,
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
            'verbose': verbose,
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
            self.ai_models_config = json.loads(code_models)
            if execute_code:
                print("Executing generated config to create models...")
                return self.ai_models_config
            else:
                print("Code not executed. Returning models config (dict).")
                return code_models
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return code_models



    def consult_ensamble(self, execute_code: bool = False):
        """
        Consults the LLM to generate a ensamble configuration stacking / voting.
        
        Args:
            execute_code (bool): If True, attempts to parse the LLM's response as JSON. If False, returns the raw string.
            
        Returns:
            Union[Dict[str, Any], str]: The generated models configuration as a dictionary if `execute_code` is True and parsing succeeds, otherwise the raw string response from the LLM.
        """

        if self.df is None:
            raise ValueError("Dataset must be loaded before consulting ensamble.")
        
        if self.ai_eda_summary is None:
            self.ai_eda_summary = self.consulteda_summary(self.df, self.target)

        from psai.config import CONFIG as MODELS_CONFIG

        ensamble_prompt = f"""
Your task is to enable/disable ensamble models (stacking / voting) and tune parameters for ensamble models for analyzed dataset.

The ensamble config default is (json format):
===
ENSAMBLE_CONFIG = {{
    'stacking': { json.dumps(CONFIG['stacking'], indent=10) },
    'voting': { json.dumps(CONFIG['voting'], indent=10) }
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
        
        # 4. Get Code from LLM
        print("Consulting LLM for Ensamble Config...")
        response = self.agent.invoke({"messages": [HumanMessage(content=ensamble_prompt)]}, config=self.config)
        ensamble_config =  self._final_message_text(response)       
        
        
        
        print("-" * 40)
        print("Generated Ensamble Config:")
        print("-" * 40)
        print(ensamble_config)
        print("-" * 40)
        
        try:
            self.ai_ensamble_config = json.loads(ensamble_config)
            if execute_code:
                print("Executing generated config to create models...")
                return self.ai_ensamble_config
            else:
                print("Code not executed. Returning models config (dict).")
                return ensamble_config
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return ensamble_config



    def consult_results(self):
        """
        Consults the LLM to analyze the results of the models.
        """
        # 1. Check if PSML is None
        if self.psml is None:
            raise ValueError(f"PSML is None.")
        
        if self.ai_models_config is None:
            raise ValueError(f"AI models config is None.")

        if self.df is None:
            raise ValueError(f"Dataset is None.")           

        if self.target is None:
            raise ValueError(f"Target is None.")           

        if self.ai_preprocessor is None:
            raise ValueError(f"AI preprocessor is None.")

        if self.ai_eda_summary is None:
            raise ValueError(f"AI EDA summary is None.")


        model_scores = self.psml.scores()
        if model_scores:
            try:
                scores_display = json.dumps(model_scores, indent=4)
            except TypeError:
                scores_display = str(model_scores)
        else:
            scores_display = "No scores available for analysis."

        results_prompt = f"""
Your task is to make comprehensive analysis of tuned models and their results of machine learning optimization with optuna.

Explain how to read that scores, explain what is the best model and why.

All scores are calculated with CrossValidation inside loop of optuna.

Scores for models - metrics from models config:
===
{scores_display}
===

**Constraints**:
- Analyze results and get insights
- Explain why that insights have sense
- Explain the interpretation for results
- If builded leaderboard of models show them as list not table.
- Handle potential errors gracefully if possible.
- 
"""
        
        # 4. Get Code from LLM
        print("Consulting LLM for results analysis...")
        response = self.agent.invoke({"messages": [HumanMessage(content=results_prompt)]}, config=self.config)
        ai_results =  self._final_message_text(response)       
        
        
        
        print("-" * 40)
        print("Generated Analysis:")
        print("-" * 40)
        print(self.ai_results_analysis)
        print("-" * 40)
        
        try:
            self.ai_results_analysis = ai_results
            if self.ai_results_analysis:
                print("Analysis executed. Returning analysis (str).")
                return self.ai_results_analysis
            else:
                print("Analysis not executed. Returning analysis (str).")
                return self.ai_results_analysis
            
        except Exception as e:
            print(f"Error parsing or executing generated analysis: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return self.ai_results_analysis


    def end_to_end_ml_process(self, execute_code: bool = True, save_report: bool = False, report_filename: str = "report.html") -> psML:
        """
        End-to-end machine learning process.
        
        Args:
            execute_code (bool): Whether to execute the generated code.
            save_report (bool): Whether to save the report.
            report_filename (str): The name of the report file.
            
        Returns:
            psML: The trained psML object.
        """

        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

        # 2. EDA
        print("\n" + "="*40)
        print(" STEP 1: Exploratory Data Analysis (EDA) ")
        print("="*40 + "\n")
        self.consult_eda()

        # 2. Dataset Config
        print("\n" + "="*40)
        print(" STEP 2: Dataset Config ")
        print("="*40 + "\n")
        self.consult_dataset_config(execute_code=execute_code)

        # 3. Feature Engineering
        print("\n" + "="*40)
        print(" STEP 3: Feature Engineering ")
        print("="*40 + "\n")
        # consult_feature_engineering returns X, y
        self.X, self.y = self.consult_feature_engineering(execute_code=execute_code)

        # 4. Preprocessing
        print("\n" + "="*40)
        print(" STEP 4: Preprocessing Configuration ")
        print("="*40 + "\n")
        self.ai_preprocessor = self.consult_preprocessor(execute_code=execute_code)

        # 5. Model Selection
        print("\n" + "="*40)
        print(" STEP 5: Model Selection & Configuration ")
        print("="*40 + "\n")
        self.ai_models_config = self.consult_models(execute_code=execute_code)
       

        # 7. Configuration Merging
        print("\n" + "="*40)
        print(" STEP 6: Models to run ... ")
        print("="*40 + "\n")

        for model_name, model_config_details in self.ai_models_config.items():
            if model_config_details.get("enabled", False) is True:
                print(f"Enabling model: {model_name}")

        # 8. Execution
        print("\n" + "="*40)
        print(" STEP 7: Model Training & Optimization ")
        print("="*40 + "\n")
        self.run_analysis()
        
        # 9. Results
        print("\n" + "="*40)
        print(" STEP 8: Results for models ")
        print("="*40 + "\n")
        self.consult_results()

        # 6. Ensamble Configuration
        print("\n" + "="*40)
        print(" STEP 9: Ensamble Configuration ")
        print("="*40 + "\n")
        self.consult_ensamble(execute_code=execute_code)

        # 8. Execution
        print("\n" + "="*40)
        print(" STEP 10: Ensamble Training & Optimization")
        print("="*40 + "\n")
        self.run_analysis_ensamble()

        # 9. Results
        print("\n" + "="*40)
        print(" STEP 11: Results for models + ensamble")
        print("="*40 + "\n")
        self.consult_results()

        # 10. Save Report
        ddate = time.strftime("%Y%m%d%H%M%S")
        filename = f"report-{self.target}-{ddate}.html"
        print("\n" + "="*40)
        print(f" STEP 10: Save Report (filename: {filename}) ")
        print("="*40 + "\n")
        self.save_report(filename=filename)
        
        return self.psml

    def _configuring_psml(self):
        self.run_config = copy.deepcopy(CONFIG)
        self.run_config['dataset']['target'] = self.target

        if isinstance(self.ai_dataset_config, dict):
            self.run_config['dataset'] = self.ai_dataset_config
        
        if isinstance(self.ai_preprocessor, dict):
            self.run_config['preprocessor'] = self.ai_preprocessor
        
        if isinstance(self.ai_models_config, dict):
            self.run_config['models'] = self.ai_models_config

        if isinstance(self.ai_ensamble_config, dict):
            self.run_config['stacking'] = self.ai_ensamble_config['stacking']
            self.run_config['voting'] = self.ai_ensamble_config['voting']

        if self.psml == None:
            self.psml = psML(self.run_config, X=self.X, y=self.y, experiment_name=self.experiment_name)
        else:
            self.psml.config = self.run_config

    def save_report(self, filename: str = "report.html"):
        """
        Saves a comprehensive HTML report of the end-to-end process.
        
        Args:
            filename (str): The path to save the HTML report.
        """
        if not self.eda_report:
            print("Warning: No EDA report found. Run consult_eda() or end_to_end_ml_process() first.")
            return

        html_content = ["""
        <html>
        <head>
            <title>Data Scientist AI Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; color: #333; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.1); border-radius: 8px; }
                h1, h2, h3 { color: #2c3e50; }
                h1 { text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 20px; }
                h2 { border-left: 5px solid #3498db; padding-left: 15px; margin-top: 40px; background-color: #ecf0f1; padding: 10px; border-radius: 0 5px 5px 0; }
                pre { background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
                .section { margin-bottom: 50px; }
                
                /* Table Styles */
                table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }
                th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #3498db; color: white; }
                tr:hover { background-color: #f1f1f1; }
                .table-container { overflow-x: auto; max-height: 600px; overflow-y: auto; border: 1px solid #eee; margin: 20px 0; }
                
                /* Plot Styles */
                img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eee; border-radius: 4px; }
                .plot-container { text-align: center; margin: 30px 0; }

                /* Markdown Content Styles */
                .text-content p { margin-bottom: 1em; line-height: 1.6; }
                .text-content ul, .text-content ol { margin-bottom: 1em; padding-left: 20px; }
                .text-content li { margin-bottom: 0.5em; }
                .text-content h3 { margin-top: 1.5em; color: #34495e; }
                .text-content h4 { margin-top: 1.2em; color: #34495e; font-weight: bold; }
                .text-content blockquote { border-left: 4px solid #3498db; padding-left: 15px; color: #7f8c8d; margin: 1em 0; }
                .text-content code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: Consolas, monospace; color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Data Scientist AI - End-to-End Report</h1>
        """]

        # 1. EDA Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>1. Exploratory Data Analysis</h2>")
        if self.ai_eda_summary:
             html_content.append("<h3>AI Analysis Summary</h3>")
             html_content.append(f"<div class='text-content'>{markdown.markdown(html.escape(self.ai_eda_summary))}</div>")
        
        html_content.extend(self.eda_report.get_html_fragments())
        html_content.append("</div>")

        # 2. Feature Engineering Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>2. Feature Engineering</h2>")
        if self.ai_feature_engineering_code:
            html_content.append("<h3>Generated Code</h3>")
            html_content.append(f"<pre><code>{self.ai_feature_engineering_code}</code></pre>")
        else:
            html_content.append("<p>No feature engineering code generated.</p>")
        html_content.append("</div>")

        # 3. Dataset Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>3. Dataset Configuration</h2>")
        if self.ai_dataset_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_dataset_config, indent=4) if isinstance(self.ai_dataset_config, dict) else str(self.ai_dataset_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No dataset configuration generated.</p>")
        html_content.append("</div>")

        # 4. Preprocessing Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>4. Preprocessing Configuration</h2>")
        if self.ai_preprocessor:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_preprocessor, indent=4) if isinstance(self.ai_preprocessor, dict) else str(self.ai_preprocessor)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No preprocessor configuration generated.</p>")
        html_content.append("</div>")

        # 5. Models Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>5. Models Configuration</h2>")
        if self.ai_models_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_models_config, indent=4) if isinstance(self.ai_models_config, dict) else str(self.ai_models_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No models configuration generated.</p>")
        html_content.append("</div>")

        # 6. Ensamble Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>6. Ensamble Configuration</h2>")
        if self.ai_ensamble_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_ensamble_config, indent=4) if isinstance(self.ai_ensamble_config, dict) else str(self.ai_ensamble_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No ensamble configuration generated.</p>")
        html_content.append("</div>")

        # 7. Results Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>7. Results & Analysis</h2>")
        
        if self.psml:
             scores = self.psml.scores()
             if scores:
                 html_content.append("<h3>Model Scores</h3>")
                 # Convert scores dict to dataframe for nice table
                 try:
                     scores_df = pd.DataFrame(scores).T
                     html_content.append(scores_df.to_html(classes='table', border=0))
                 except:
                     html_content.append(f"<pre>{json.dumps(scores, indent=4)}</pre>")

        if self.ai_results_analysis:
            html_content.append("<h3>AI Analysis</h3>")
            html_content.append(f"<div class='text-content'>{markdown.markdown(html.escape(self.ai_results_analysis))}</div>")
        
        html_content.append("</div>")
        
        html_content.append("</div></body></html>")
        
        full_html = '\n'.join(html_content)
        
        with open(filename.lower(), 'w', encoding='utf-8') as f:
            f.write(full_html)