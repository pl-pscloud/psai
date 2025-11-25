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

from psai.datasets import EDAReport
from psai.config import CONFIG
from psai.psml import psML

class DataScientist:
    def __init__(self, df: pd.DataFrame, target: str,model_name: str = "gemini-3-pro-preview"):
        """
        Initialize the DataScientist agent.
        
        Args:
            model_name (str): The name of the Gemini model to use.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2, google_api_key=api_key)
        self.llm_memory = InMemorySaver()
        self.df = df
        self.target = target
        self.X = None
        self.y = None
        self.eda_report = None
        self.ai_eda_summary = None
        self.ai_preprocessor = None
        self.ai_feature_engineering_code = None
        self.ai_models_config = None
        self.ai_ensamble_config = None
        self.ai_results_analysis = None
        self.psml = None
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

    def _analyze(self, config: Dict[str, Any]):

        if self.X is None or self.y is None:
            raise ValueError("X and y must be set before calling _analyze.")
        
        self.psml = psML(config=config, X=self.X, y=self.y)
        
        print("Optimizing models...")
        self.psml.optimize_all_models()
        
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
        
        
        optuna_trials = 10                                      # Number of trials for Optuna hyperparameter optimization
        optuna_n_jobs = 1                                       # Number of parallel Optuna jobs (studies running at once)
        optuna_metric = 'rmse_safe'                             # Metric to optimize during Optuna trials (e.g., 'rmse', 'auc')
        
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
MODELS_CONFIG = {json.dumps(MODELS_CONFIG['models'], indent=10)}
===

Based on the analysis, create the best configuration for mentioned earlier models and analyzed dataset.


**Constraints**:
-   The config MUST be in json format.
-   Important: Do NOT remove / change any keys in json.
-   Only change values if you think it is needed.
-   If possible for GPU use it, if not use CPU. values: 'cpu', 'gpu'
-   When set params for catboost if choose device = 'gpu' use bootstrap_type = 'Bayesian' or 'Bernoulli'
-   When multiclass are selected for lightgbm set num_class = x (where x is number of classes)
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
    
        # Stacking configuration
        STACKING_CONFIG = {
            'cv_enabled': False,                        # Enable stacking for Cross-Validation models
            'cv_folds': 5,                              # Folds for stacking CV (if not using prefit)
            'final_enabled': False,                     # Enable stacking for the final model
            'meta_model': 'lightgbm',                   # The model used to aggregate base model predictions
            'use_features': True,                       # If True, feeds original features + predictions to meta-model
            'prefit': True,                             # If True, uses existing trained models (faster). If False, retrains.
        }

        VOTING_CONFIG = {
            'cv_enabled': False,                        # Enable voting ensemble for Cross-Validation models
            'final_enabled': False,                     # Enable voting ensemble for the final model
            'use_features': True,                       # (Note: Voting usually just averages predictions, this flag might be custom logic)
            'prefit': True,                             # If True, uses already trained models.
        }

        

        ensamble_prompt = f"""
Your task is to enable/disable ensamble models (stacking / voting) and tune parameters for ensamble models for analyzed dataset.

The ensamble config default is (json format):
===
ENSAMBLE_CONFIG = {{
    'stacking': { json.dumps(STACKING_CONFIG, indent=10) },
    'voting': { json.dumps(VOTING_CONFIG, indent=10) }
}}

Params Stacking:
'cv_enabled': bool,          # Enable stacking for Cross-Validation models
'cv_folds': int,             # Folds for stacking CV (if not using prefit)
'final_enabled': bool,       # Enable stacking for the final model
'meta_model': str,           # The model used to aggregate base model predictions
'use_features': bool,        # If True, feeds original features + predictions to meta-model
'prefit': bool,              # If True, uses existing trained models (faster). If False, retrains.

Params Voting:
'cv_enabled': bool,          # Enable voting ensemble for Cross-Validation models
'final_enabled': bool,       # Enable voting ensemble for the final model
'use_features': bool,        # (Note: Voting usually just averages predictions, this flag might be custom logic)
'prefit': bool,              # If True, uses already trained models.


===

Based on the analysis, create the best configuration for mentioned earlier ensamble models and analyzed dataset.

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
-   Handle potential errors gracefully if possible.
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

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        # 2. EDA
        print("\n" + "="*40)
        print(" STEP 1: Exploratory Data Analysis (EDA) ")
        print("="*40 + "\n")
        self.consult_eda()

        # 3. Feature Engineering
        print("\n" + "="*40)
        print(" STEP 2: Feature Engineering ")
        print("="*40 + "\n")
        # consult_feature_engineering returns X, y
        self.X, self.y = self.consult_feature_engineering(execute_code=execute_code)

        # 4. Preprocessing
        print("\n" + "="*40)
        print(" STEP 3: Preprocessing Configuration ")
        print("="*40 + "\n")
        self.ai_preprocessor = self.consult_preprocessor(execute_code=execute_code)

        # 5. Model Selection
        print("\n" + "="*40)
        print(" STEP 4: Model Selection & Configuration ")
        print("="*40 + "\n")
        self.ai_models_config = self.consult_models(execute_code=execute_code)

        # 5. Ensamble Configuration
        print("\n" + "="*40)
        print(" STEP 5: Ensamble Configuration ")
        print("="*40 + "\n")
        self.ai_ensamble_config = self.consult_ensamble(execute_code=execute_code)

        # 6. Configuration Merging
        print("\n" + "="*40)
        print(" STEP 6: Configuring PSML ")
        print("="*40 + "\n")
        
        run_config = copy.deepcopy(CONFIG)
        run_config['dataset']['target'] = self.target
        
        if isinstance(self.ai_preprocessor, dict):
             run_config['preprocessor'] = self.ai_preprocessor
        
        if isinstance(self.ai_models_config, dict):
            run_config['models'] = self.ai_models_config

        if isinstance(self.ai_ensamble_config, dict):
            run_config['stacking'] = self.ai_ensamble_config['stacking']
            run_config['voting'] = self.ai_ensamble_config['voting']

        for model_name, model_config_details in self.ai_models_config.items():
            if model_config_details.get("enabled", False) is True:
                print(f"Enabling model: {model_name}")

        # 7. Execution
        print("\n" + "="*40)
        print(" STEP 7: Model Training & Optimization ")
        print("="*40 + "\n")
        self._analyze(run_config)
        
        print("\n" + "="*40)
        print(" Final Scores ")
        print("="*40 + "\n")
        self.psml.scores()

        # 8. Results
        print("\n" + "="*40)
        print(" STEP 8: Results ")
        print("="*40 + "\n")
        self.consult_results()
        
        return self.psml

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

        # 3. Preprocessing Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>3. Preprocessing Configuration</h2>")
        if self.ai_preprocessor:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_preprocessor, indent=4) if isinstance(self.ai_preprocessor, dict) else str(self.ai_preprocessor)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No preprocessor configuration generated.</p>")
        html_content.append("</div>")

        # 4. Models Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>4. Models Configuration</h2>")
        if self.ai_models_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_models_config, indent=4) if isinstance(self.ai_models_config, dict) else str(self.ai_models_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No models configuration generated.</p>")
        html_content.append("</div>")

        # 5. Ensamble Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>5. Ensamble Configuration</h2>")
        if self.ai_ensamble_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_ensamble_config, indent=4) if isinstance(self.ai_ensamble_config, dict) else str(self.ai_ensamble_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No ensamble configuration generated.</p>")
        html_content.append("</div>")

        # 6. Results Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>6. Results & Analysis</h2>")
        
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
        print(" STEP 1: Exploratory Data Analysis (EDA) ")
        print("="*40 + "\n")
        self.consult_eda()

        # 3. Feature Engineering
        print("\n" + "="*40)
        print(" STEP 2: Feature Engineering ")
        print("="*40 + "\n")
        # consult_feature_engineering returns X, y
        self.X, self.y = self.consult_feature_engineering(execute_code=execute_code)

        # 4. Preprocessing
        print("\n" + "="*40)
        print(" STEP 3: Preprocessing Configuration ")
        print("="*40 + "\n")
        self.ai_preprocessor = self.consult_preprocessor(execute_code=execute_code)

        # 5. Model Selection
        print("\n" + "="*40)
        print(" STEP 4: Model Selection & Configuration ")
        print("="*40 + "\n")
        self.ai_models_config = self.consult_models(execute_code=execute_code)

        # 5. Ensamble Configuration
        print("\n" + "="*40)
        print(" STEP 5: Ensamble Configuration ")
        print("="*40 + "\n")
        self.ai_ensamble_config = self.consult_ensamble(execute_code=execute_code)

        # 6. Configuration Merging
        print("\n" + "="*40)
        print(" STEP 6: Configuring PSML ")
        print("="*40 + "\n")
        
        run_config = copy.deepcopy(CONFIG)
        run_config['dataset']['target'] = self.target
        
        if isinstance(self.ai_preprocessor, dict):
             run_config['preprocessor'] = self.ai_preprocessor
        
        if isinstance(self.ai_models_config, dict):
            run_config['models'] = self.ai_models_config

        if isinstance(self.ai_ensamble_config, dict):
            run_config['stacking'] = self.ai_ensamble_config['stacking']
            run_config['voting'] = self.ai_ensamble_config['voting']

        for model_name, model_config_details in self.ai_models_config.items():
            if model_config_details.get("enabled", False) is True:
                print(f"Enabling model: {model_name}")

        # 7. Execution
        print("\n" + "="*40)
        print(" STEP 7: Model Training & Optimization ")
        print("="*40 + "\n")
        self._analyze(run_config)
        
        print("\n" + "="*40)
        print(" Final Scores ")
        print("="*40 + "\n")
        self.psml.scores()

        # 8. Results
        print("\n" + "="*40)
        print(" STEP 8: Results ")
        print("="*40 + "\n")
        self.consult_results()
        
        return self.psml

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

        # 3. Preprocessing Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>3. Preprocessing Configuration</h2>")
        if self.ai_preprocessor:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_preprocessor, indent=4) if isinstance(self.ai_preprocessor, dict) else str(self.ai_preprocessor)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No preprocessor configuration generated.</p>")
        html_content.append("</div>")

        # 4. Models Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>4. Models Configuration</h2>")
        if self.ai_models_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_models_config, indent=4) if isinstance(self.ai_models_config, dict) else str(self.ai_models_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No models configuration generated.</p>")
        html_content.append("</div>")

        # 5. Ensamble Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>5. Ensamble Configuration</h2>")
        if self.ai_ensamble_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_ensamble_config, indent=4) if isinstance(self.ai_ensamble_config, dict) else str(self.ai_ensamble_config)
            html_content.append(f"<pre><code>{config_str}</code></pre>")
        else:
             html_content.append("<p>No ensamble configuration generated.</p>")
        html_content.append("</div>")

        # 6. Results Section
        html_content.append("<div class='section'>")
        html_content.append("<h2>6. Results & Analysis</h2>")
        
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
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_html)