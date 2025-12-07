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
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import markdown
import html
import time

from psai.core.datasets import EDAReport
from psai.core.config import CONFIG
from psai.core.psml import psML
from psai.agents.prompts import (
    SYSTEM_PROMPT,
    get_eda_prompt,
    get_dataset_config_prompt,
    get_preprocessor_prompt,
    get_models_prompt,
    get_ensamble_prompt,
    get_results_prompt,
    get_feature_engineering_prompt,
    get_shap_prompt,
)

class DataScientist:
    def __init__(self, df: pd.DataFrame, target: str, model_provider: str = "google", model_name: str = "gemini-3-pro-preview", model_temperature: float = 1.0,  api_key: str = None, optuna_metric: str = None, optuna_trials: int = None, optuna_timeout: int = None, task_type: str = None, experiment_name: str = None):
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
                model = model_name, 
                temperature = model_temperature, 
                google_api_key = api_key
            )
        elif model_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model = model_name,
                temperature = model_temperature,
                reasoning_effort = "high",
                max_retries = 2,
                api_key = api_key,
            )
        elif model_provider == "antrophic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model = model_name,
                temperature = model_temperature,
                thinking = {"type": "enabled", "budget_tokens": 5000},
                max_retries = 2,
                api_key = api_key,
            )
        elif model_provider == "ollama":
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model = model_name,
                validate_model_on_init = True,
                temperature = model_temperature,
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
        self.ai_preprocessor_config = None
        self.ai_feature_engineering_code = None
        self.ai_feature_engineering = None
        self.ai_models_config = None
        self.ai_ensamble_config = None
        self.ai_results_analysis = None
        self.run_config = None
        self.psml = None
        self.session = random.randint(100000, 999999)
        self.config = {"configurable": {"thread_id": self.session}}
        self.optuna_metric = optuna_metric
        self.optuna_trials = optuna_trials
        self.optuna_timeout = optuna_timeout
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.system_prompt = SYSTEM_PROMPT
        self.agent = create_agent(
            model=self.llm,
            tools=[],
            checkpointer=self.llm_memory,
            system_prompt=self.system_prompt,
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
        optuna_timeout_info = f"""Suggested timeout for optuna is: '{self.optuna_timeout}'""" if self.optuna_timeout else ""
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
        
        eda_prompt = get_eda_prompt(target=self.target, summary=summary, task_type_info=task_type_info, optuna_metrics_info=optuna_metrics_info, optuna_trials_info=optuna_trials_info, optuna_timeout_info=optuna_timeout_info)

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

    def consult_dataset_config(self, execute_code: bool = True, output_config: bool = False) -> str:
        """
        Generates a text summary of the EDA report to feed into the LLM.
        """
        print("Generating dataset config...")

        optuna_metrics_info = f"""The optuna_metric is: '{self.optuna_metric}'""" if self.optuna_metric else ""
        task_type_info = f"""The task_type is: '{self.task_type}'""" if self.task_type else ""
        
        dataset_config_prompt = get_dataset_config_prompt(target=self.target, optuna_metrics_info=optuna_metrics_info, task_type_info=task_type_info)

        # 4. Get Code from LLM
        print("Consulting LLM for dataset config...")
        response = self.agent.invoke({"messages": [HumanMessage(content=dataset_config_prompt)]}, config=self.config)
        dataset_config =  self._final_message_text(response)
        
        print("-" * 40)
        print("Generating dataset config...")
        try:
            if execute_code:
                print("Executing generated config to create dataset config...")
                self.ai_dataset_config = json.loads(dataset_config)
                print("Dataset config executed.")
                if output_config:
                    print("-" * 40)
                    print(dataset_config)
                    print("-" * 40)
                return self.ai_dataset_config
            else:
                print("-" * 40)
                print(dataset_config)
                print("-" * 40)
                print("Code not executed.")
                return None
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            return None

    def consult_feature_engineering(self, execute_code: bool = True, return_X_y: bool = False, output_config: bool = False):
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
        feature_engineering_prompt = get_feature_engineering_prompt(self.target)
        
        # 4. Get Code from LLM
        print("Consulting LLM for Feature Engineering...")
        response = self.agent.invoke({"messages": [HumanMessage(content=feature_engineering_prompt)]}, config=self.config)
        code_feature_engineering =  self._final_message_text(response)        
        
        print("-" * 40)
        print("Generating Feature Engineering Code...")
                
        
        try:
            self.ai_feature_engineering_code = code_feature_engineering
            if execute_code:
                # 5. Execute Code
                print("Executing generated code...")
                
                # Import required base classes for the exec scope
                from sklearn.base import BaseEstimator, TransformerMixin
                
                local_scope = {
                    'pd': pd, 
                    'np': np, 
                    'BaseEstimator': BaseEstimator, 
                    'TransformerMixin': TransformerMixin
                }
                
                # Clean code (remove markdown blocks if the LLM ignored instructions)
                code = re.sub(r'^```python\s*', '', code_feature_engineering, flags=re.MULTILINE)
                code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
                code = code.strip()
                exec(code, local_scope)

                if output_config:
                    print("-" * 40)
                    print(code_feature_engineering)
                    print("-" * 40)

                if 'FeatureEngTransformer' not in local_scope:
                    raise ValueError("The generated code did not define a 'FeatureEngTransformer' class.")

                # Instantiate the transformer
                TransformerClass = local_scope['FeatureEngTransformer']
                self.ai_feature_engineering = TransformerClass()
                
                # Split X and y normally
                self.y = self.df[self.target]
                self.X = self.df.drop(columns=[self.target])
                
                print(f"Feature Engineering Transformer created. X shape: {self.X.shape}, y shape: {self.y.shape}")
                
                if return_X_y:
                    return self.X, self.y
                else:
                    return None
            else:
                print("Code not executed.")
                print("-" * 40)
                print(code_feature_engineering)
                print("-" * 40)
                return None
            
        except Exception as e:
            print(f"Error executing generated code: {e}")
            raise

    def consult_preprocessor_config(self, execute_code: bool = True, output_config: bool = False):
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

        preprocessor_prompt = get_preprocessor_prompt(preprocessor_config=preprocessor_config)
        
        # 4. Get Code from LLM
        print("Consulting LLM for Preprocessor...")
        response = self.agent.invoke({"messages": [HumanMessage(content=preprocessor_prompt)]}, config=self.config)
        code_preprocessor =  self._final_message_text(response)       
        
        print("-" * 40)
        print("Generating Preprocessor Code...")

        try:
            
            if execute_code:
                print("Executing generated config to create preprocessor...")
                self.ai_preprocessor_config = json.loads(code_preprocessor)
                print("Preprocessor config executed.")
                if output_config:
                    print("-" * 40)
                    print(code_preprocessor)
                    print("-" * 40)
                return self.ai_preprocessor_config
            else:
                print("-" * 40)
                print(code_preprocessor)
                print("-" * 40)
                print("Code not executed.")
                return None
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return None

    def consult_models_config(self, execute_code: bool = True, output_config: bool = False):
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

        from psai.core.config import CONFIG as MODELS_CONFIG

        optuna_metrics_info = f"""The optuna_metric is: '{self.optuna_metric}'""" if self.optuna_metric else ""
        optuna_trials_info = f"""Suggested trails for optuna is: '{self.optuna_trials}'""" if self.optuna_trials else ""
        optuna_timeout_info = f"""Suggested timeout for optuna is: '{self.optuna_timeout}'""" if self.optuna_timeout else ""
        task_type_info = f"""The task_type is: '{self.task_type}'""" if self.task_type else ""

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
        
        print(optuna_metrics_info)
        print(optuna_trials_info)
        print(optuna_timeout_info)
        print(task_type_info)
        
        models_prompt = get_models_prompt(gpu_available=gpu_available, gpu_model=gpu_model, cpu_count=cpu_count, verbose=1, optuna_metrics_info=optuna_metrics_info, optuna_trials_info=optuna_trials_info, optuna_timeout_info=optuna_timeout_info, task_type_info=task_type_info)
        
        # 4. Get Code from LLM
        print("Consulting LLM for Models Config...")
        response = self.agent.invoke({"messages": [HumanMessage(content=models_prompt)]}, config=self.config)
        code_models =  self._final_message_text(response)       
        
        
        
        print("-" * 40)
        print("Generating Models Config...")

        
        try:
            if execute_code:
                print("Executing generated config to create models...")
                self.ai_models_config = json.loads(code_models)
                print("Models config executed.")
                if output_config:
                    print("-" * 40)
                    print(code_models)
                    print("-" * 40)
                return self.ai_models_config
            else:
                print("-" * 40)
                print(code_models)
                print("-" * 40)
                print("Code not executed.")
                return None
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return None



    def consult_ensamble_config(self, execute_code: bool = True, output_config: bool = False):
        """
        Consults the LLM to generate an ensemble configuration (Stacking/Voting).
        
        Args:
            execute_code: If True, attempts to parse the LLM's response as JSON. If False, returns the raw string.
            
        Returns:
            The generated ensemble configuration as a dictionary if `execute_code` is True and parsing succeeds, 
            otherwise the raw string response from the LLM.
        """

        if self.df is None:
            raise ValueError("Dataset must be loaded before consulting ensamble.")
        
        if self.ai_eda_summary is None:
            self.ai_eda_summary = self.consulteda_summary(self.df, self.target)

        from psai.core.config import CONFIG as MODELS_CONFIG

        ensamble_prompt = get_ensamble_prompt(stacking_config=CONFIG['stacking'], voting_config=CONFIG['voting'])
        
        # 4. Get Code from LLM
        print("Consulting LLM for Ensamble Config...")
        response = self.agent.invoke({"messages": [HumanMessage(content=ensamble_prompt)]}, config=self.config)
        ensamble_config =  self._final_message_text(response)       
        
        
        
        print("-" * 40)
        print("Generating Ensamble Config...")

        
        try:
            if execute_code:
                print("Executing generated config to create models...")
                self.ai_ensamble_config = json.loads(ensamble_config)
                print("Ensamble config executed.")
                if output_config:
                    print("-" * 40)
                    print(ensamble_config)
                    print("-" * 40)
                return self.ai_ensamble_config
            else:
                print("-" * 40)
                print(ensamble_config)
                print("-" * 40)
                print("Code not executed")
                return ensamble_config
            
        except Exception as e:
            print(f"Error parsing or executing generated code: {e}")
            # If parsing fails, return the raw string so the user can see what happened
            return ensamble_config



    def consult_results(self):
        """
        Consults the LLM to analyze the results of the trained models.
        
        This method gathers the scores from the psML instance and asks the LLM to provide
        a comprehensive analysis, including model comparison, insights, and interpretation.
        
        Returns:
            The LLM's analysis of the model results.
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

        if self.ai_preprocessor_config is None:
            raise ValueError(f"AI preprocessor config is None.")

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

        results_prompt = get_results_prompt(scores_display=scores_display)
        
        # 4. Get Code from LLM
        print("Consulting LLM for results analysis...")
        response = self.agent.invoke({"messages": [HumanMessage(content=results_prompt)]}, config=self.config)
        ai_results =  self._final_message_text(response)       
        
        self.ai_results_analysis = ai_results
        
        print("-" * 40)
        print("Generated Analysis:")
        print("-" * 40)
        print(self.ai_results_analysis)
        print("-" * 40)
        
    def consult_shap(self, model_name: str):
        """
        Consults the LLM to analyze the SHAP values of the trained models.
        
        This method gathers the SHAP plot image from the psML instance and asks the LLM to provide
        a comprehensive analysis and interpretation.
        
        Returns:
            The LLM's analysis of the SHAP plot for model.
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

        if self.ai_preprocessor_config is None:
            raise ValueError(f"AI preprocessor config is None.")

        if self.ai_eda_summary is None:
            raise ValueError(f"AI EDA summary is None.")

        if self.ai_ensamble_config is None:
            raise ValueError(f"AI ensamble config is None.")

        if self.ai_results_analysis is None:
            raise ValueError(f"AI results analysis is None.")

        shap_prompt = get_shap_prompt(model_name=model_name, task_type=self.task_type)
        
        # 4. Get Code from LLM
        print("Consulting LLM for SHAP plot analysis...")
        
        messages = [HumanMessage(content=shap_prompt)]
        
        # Check if SHAP image exists and add it to the message
        model_key = f'final_model_{model_name}'

        if model_key not in self.psml.models:
            raise ValueError(f"Model {model_key} not found in PSML.")
        else:
            print(f"Model {model_key} found in PSML.")
        
        if 'shap_image' not in self.psml.models[model_key]:
            print(f"Running explain_model for model {model_key}.")
            self.psml.explain_model(model_name=model_name)
            print(f"SHAP image generated for model {model_key}.")
        else:
            print(f"SHAP image already exists for model {model_key}.")

        shap_image_base64 = self.psml.models[model_key]['shap_image']

        if shap_image_base64:
            messages.append(HumanMessage(content=[
                {"type": "image", "base64": shap_image_base64, "mime_type": "image/png"}
            ]))
        else:
            raise ValueError(f"SHAP image not found for model {model_key}")

        response = self.agent.invoke({"messages": messages}, config=self.config)
        ai_shap_analysis =  self._final_message_text(response)
    
        self.ai_shap_analysis = ai_shap_analysis
    
        print("-" * 40)
        print("Generated SHAP Analysis:")
        print("-" * 40)
        print(ai_shap_analysis)
        print("-" * 40)  


    def end_to_end_ml_process(self, execute_code: bool = True, save_report: bool = True, report_filename: str = "report.html", output_config: bool = False):
        """
        Executes the complete end-to-end machine learning process.
        
        This includes:
        1. Exploratory Data Analysis (EDA)
        2. Dataset Configuration
        3. Feature Engineering
        4. Preprocessing Configuration
        5. Model Selection & Configuration
        6. Model Training & Optimization
        7. Results Analysis
        8. Ensemble Configuration & Training
        9. Final Results Analysis
        10. Report Generation (optional)
        
        Args:
            execute_code: Whether to execute the generated code and configurations.
            save_report: Whether to save the final report to an HTML file.
            report_filename: The name of the report file.
            
        Returns:
            The trained psML object.
        """

        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

        # 1. EDA
        print("\n" + "="*40)
        print(" STEP 1: Exploratory Data Analysis (EDA) ")
        print("="*40 + "\n")
        self.consult_eda()

        # 2. Feature Engineering
        print("\n" + "="*40)
        print(" STEP 2: Feature Engineering ")
        print("="*40 + "\n")
        # consult_feature_engineering returns X, y
        self.consult_feature_engineering(output_config=output_config)

        # 3. Dataset Config
        print("\n" + "="*40)
        print(" STEP 3: Dataset Config ")
        print("="*40 + "\n")
        self.consult_dataset_config(output_config=output_config)

        # 4. Preprocessing
        print("\n" + "="*40)
        print(" STEP 4: Preprocessing Configuration ")
        print("="*40 + "\n")
        self.consult_preprocessor_config(output_config=output_config)

        # 5. Model Selection
        print("\n" + "="*40)
        print(" STEP 5: Model Selection & Configuration ")
        print("="*40 + "\n")
        self.consult_models_config(output_config=output_config)
       

        # 6. Configuration Merging
        print("\n" + "="*40)
        print(" STEP 6: Models to run ... ")
        print("="*40 + "\n")

        for model_name, model_config_details in self.ai_models_config.items():
            if model_config_details.get("enabled", False) is True:
                print(f"Enabling model: {model_name}")

        # 7. Execution
        print("\n" + "="*40)
        print(" STEP 7: Model Training & Optimization ")
        print("="*40 + "\n")
        self.run_analysis()
        
        # 8. Results
        print("\n" + "="*40)
        print(" STEP 8: Results for models ")
        print("="*40 + "\n")
        self.consult_results()

        # 9. Ensamble Configuration
        print("\n" + "="*40)
        print(" STEP 9: Ensamble Configuration ")
        print("="*40 + "\n")
        self.consult_ensamble_config(output_config=output_config)

        # 10. Execution
        print("\n" + "="*40)
        print(" STEP 10: Ensamble Training & Optimization")
        print("="*40 + "\n")
        self.run_analysis_ensamble()

        # 11. Results
        print("\n" + "="*40)
        print(" STEP 11: Results for models + ensamble")
        print("="*40 + "\n")
        self.consult_results()

        model_name = self.psml.get_best_single_model()

        if model_name:
            # 12. SHAP Analysis
            print("\n" + "="*40)
            print(f" STEP 12: SHAP Analysis for best model: {model_name}")
            print("="*40 + "\n")
            self.consult_shap(model_name)

        # 13. Save Report
        ddate = time.strftime("%Y%m%d%H%M%S")
        filename = f"report-{self.target}-{ddate}.html"
        print("\n" + "="*40)
        print(f" STEP 13: Save Report (filename: {filename}) ")
        print("="*40 + "\n")
        self.save_report(filename=filename)
        
        return self.psml

    def _configuring_psml(self):
        self.run_config = copy.deepcopy(CONFIG)
        self.run_config['dataset']['target'] = self.target

        if isinstance(self.ai_dataset_config, dict):
            self.run_config['dataset'] = self.ai_dataset_config
        
        if isinstance(self.ai_preprocessor_config, dict):
            self.run_config['preprocessor'] = self.ai_preprocessor_config
        
        if isinstance(self.ai_models_config, dict):
            self.run_config['models'] = self.ai_models_config

        if isinstance(self.ai_ensamble_config, dict):
            self.run_config['stacking'] = self.ai_ensamble_config['stacking']
            self.run_config['voting'] = self.ai_ensamble_config['voting']

        if self.run_config['dataset'].get('task_type') is None:
             if self.task_type:
                 self.run_config['dataset']['task_type'] = self.task_type
             else:
                 # Default logic or error?
                 pass

        if self.psml == None:
            self.psml = psML(self.run_config, X=self.X, y=self.y, experiment_name=self.experiment_name)
            # Pass the FE transformer if it exists
            if self.ai_feature_engineering:
                self.psml.feature_engineering_transformer = self.ai_feature_engineering
        else:
            self.psml.config = self.run_config
            if self.ai_feature_engineering:
                self.psml.feature_engineering_transformer = self.ai_feature_engineering

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
        if self.ai_preprocessor_config:
            html_content.append("<h3>Generated Configuration</h3>")
            config_str = json.dumps(self.ai_preprocessor_config, indent=4) if isinstance(self.ai_preprocessor_config, dict) else str(self.ai_preprocessor_config)
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
            html_content.append("<h2>AI Analysis</h2>")
            html_content.append(f"<div class='text-content'>{markdown.markdown(html.escape(self.ai_results_analysis))}</div>")

        if self.ai_shap_analysis:
            best_model = self.psml.get_best_single_model()
            image = self.psml.models[f"final_model_{best_model}"]["shap_image"]
            html_content.append(f"<h2>SHAP Analysis for best model {best_model}</h2>")
            html_content.append(f"<img src='data:image/png;base64,{image}' alt='SHAP Analysis' />")
            html_content.append(f"<div class='text-content'>{markdown.markdown(html.escape(self.ai_shap_analysis))}</div>")
        
        html_content.append("</div>")
        
        html_content.append("</div></body></html>")
        
        full_html = '\n'.join(html_content)
        
        with open(filename.lower(), 'w', encoding='utf-8') as f:
            f.write(full_html)