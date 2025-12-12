from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import os
import logging
import sys
import importlib.util
import copy
import cloudpickle as pickle

from psai.core.psml import psML
from psai.core.config import CONFIG as DEFAULT_CONFIG
from psai.core.experiment_manager import ExperimentManager
import io
from psai.api.schemas import (
    ConfigInput, 
    PredictionInput, 
    PredictionOutput, 
    ScoreOutput,
    InitResponse,
    TrainResponse,
    TransformerInput,
    ExperimentSaveInput,
    ExperimentSaveInput,
    Activity,
    Activity,
    BatchPredictionOutput,
    AgentInitInput,
    AgentInitResponse,
    AgentStepResponse
)
from psai.agents.datascientist import DataScientist

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="psai REST API", version="0.2.0")
exp_manager = ExperimentManager()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration state
# We initiate with a deep copy of the default config
global_config = copy.deepcopy(DEFAULT_CONFIG)

# Global state
psml_instance: Optional[psML] = None
# Agent state
agent_instance: Optional[DataScientist] = None

is_training: bool = False
feature_transformer_path = os.path.join("psai", "custom", "feature_transformer.py")

# Activity Logging
import uuid
import datetime

activities: List[Activity] = []

def log_activity(message: str, type: str = "info"):
    """Log an activity to the persistent store."""
    activity = Activity(
        id=str(uuid.uuid4()),
        message=message,
        type=type,
        timestamp=datetime.datetime.now().isoformat()
    )
    activities.insert(0, activity)
    # Keep only last 50 activities
    if len(activities) > 50:
        activities.pop()

@app.get("/activities", response_model=List[Activity])
async def get_activities():
    """Get recent activities."""
    return activities

# Helper for deep update
def deep_update(target: Dict, source: Dict) -> Dict:
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target

@app.get("/")
async def root():
    return {"message": "Welcome to psai REST API 0.2.0"}

# --- Configuration Endpoints ---

@app.get("/config")
async def get_config():
    """Get current global configuration"""
    return global_config

@app.post("/config")
async def update_config(config_input: ConfigInput):
    """Update global configuration (partial updates supported)"""
    global global_config
    try:
        deep_update(global_config, config_input.config)
        
        # Also update train path/target if provided specifically at top level
        if config_input.train_path:
            global_config['dataset']['train_path'] = config_input.train_path
        if config_input.target:
            global_config['dataset']['target'] = config_input.target
            
        return {"message": "Configuration updated successfully", "config": global_config}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/feature_transformer")
async def set_feature_transformer(input: TransformerInput):
    """
    Save custom feature transformer code to a file.
    The code must define a class named 'FeatureEngTransformer'.
    """
    try:
        os.makedirs(os.path.dirname(feature_transformer_path), exist_ok=True)
        with open(feature_transformer_path, "w") as f:
            f.write(input.code)
        return {"message": "Feature transformer code saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Experiment Management Endpoints ---

@app.get("/experiments")
async def list_experiments():
    """List all saved experiments."""
    try:
        return {"experiments": exp_manager.list_experiments()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments")
async def save_experiment(input: ExperimentSaveInput):
    """Save current experiment state."""
    global psml_instance, global_config
    try:
        # Get feature transformer code if it exists
        feature_code = None
        if os.path.exists(feature_transformer_path):
            with open(feature_transformer_path, "r") as f:
                feature_code = f.read()
        
        # Save experiment
        result = exp_manager.save_experiment(
            name=input.name,
            config=input.config,
            psml_instance=psml_instance,
            feature_code=feature_code
        )
        return {"message": "Experiment saved successfully", "details": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments/{name}/load")
async def load_experiment(name: str):
    """Load a saved experiment."""
    global psml_instance, global_config
    try:
        data = exp_manager.load_experiment(name)
        
        # 1. Restore Config
        if "config" in data:
            global_config = copy.deepcopy(data["config"])
        
        # 2. Restore Feature Code (if present)
        if "feature_code" in data:
            os.makedirs(os.path.dirname(feature_transformer_path), exist_ok=True)
            with open(feature_transformer_path, "w") as f:
                f.write(data["feature_code"])
            
            # Register module so pickle can find it
            try:
                spec = importlib.util.spec_from_file_location("custom_fe", feature_transformer_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["custom_fe"] = module
                    spec.loader.exec_module(module)
                    logger.info("Reloaded custom_fe module for unpickling")
            except Exception as e:
                logger.error(f"Failed to reload custom_fe module during experiment load: {e}")

        # 3. Restore PSML Instance
        if "model_path" in data and os.path.exists(data["model_path"]):
            with open(data["model_path"], "rb") as f:
                psml_instance = pickle.load(f)
            logger.info(f"Loaded psML instance from {data['model_path']}")
        else:
            # If no model saved, we might need to re-initialize psML if config mandates it?
            pass

        return {
            "message": f"Experiment '{name}' loaded successfully",
            "config": global_config,
            "has_model": psml_instance is not None
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    except Exception as e:
        logger.error(f"Error loading experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_api():
    """
    Reset the API state to clean slate.
    Clears configuration, loaded models, and logs.
    """
    global global_config, psml_instance, is_training, training_logs, log_handler
    
    try:
        # Reset config to default
        global_config = copy.deepcopy(DEFAULT_CONFIG)
        
        # Clear state
        psml_instance = None
        is_training = False
        
        # Clear logs
        training_logs.clear()
        if log_handler:
            log_handler.seen_messages.clear()
            
        return {"message": "API state reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Initialization ---

@app.post("/init", response_model=InitResponse)
async def initialize(config_input: Optional[ConfigInput] = None):
    """
    Initialize psML with the current global config.
    Optionally accepts a config payload to update global config before init.
    """
    global psml_instance, global_config
    try:
        # Update config if provided
        if config_input:
             if config_input.config:
                 deep_update(global_config, config_input.config)
             if config_input.train_path:
                 global_config['dataset']['train_path'] = config_input.train_path
             if config_input.target:
                 global_config['dataset']['target'] = config_input.target

        # Check if train path exists
        train_path = global_config['dataset']['train_path']
        if not os.path.exists(train_path):
             raise HTTPException(status_code=400, detail=f"Training data not found at {train_path}")
        
        # Load Feature Transformer if exists
        fe_transformer_instance = None
        if os.path.exists(feature_transformer_path):
            try:
                spec = importlib.util.spec_from_file_location("custom_fe", feature_transformer_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["custom_fe"] = module
                    spec.loader.exec_module(module)
                    if hasattr(module, "FeatureEngTransformer"):
                        fe_transformer_instance = module.FeatureEngTransformer()
                        logger.info("Loaded custom FeatureEngTransformer")
                    else:
                        logger.warning("FeatureEngTransformer class not found in custom module")
            except Exception as e:
                logger.error(f"Failed to load feature transformer: {e}")
                log_activity(f"Failed to load custom transformer: {e}", "warning")
                # Don't fail completely, just log warning
        
        # Initialize psML
        psml_instance = psML(config=global_config, fe_transformer=fe_transformer_instance)
        log_activity("Pipeline initialized", "info")
        return {"message": "psML initialized successfully", "config": global_config}
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        log_activity(f"Initialization failed: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))


# --- Training ---

# Custom log handler to capture logs
class ListHandler(logging.Handler):
    def __init__(self, log_list, max_entries=100):
        super().__init__()
        self.log_list = log_list
        self.max_entries = max_entries
        self.seen_messages = set()
        
    def emit(self, record):
        msg = self.format(record)
        msg_lower = msg.lower()
        if 'http' in msg_lower or 'options' in msg_lower or 'get /' in msg_lower: return
        
        msg_content = msg.split(' - ', 1)[-1] if ' - ' in msg else msg
        if msg_content in self.seen_messages: return
        self.seen_messages.add(msg_content)
        
        self.log_list.append(msg)
        if len(self.log_list) > self.max_entries:
            self.log_list.pop(0)

# Store logs separately from psML status for persistence across re-inits if needed, 
# although we clear them on train start.
training_logs = []
log_handler = ListHandler(training_logs)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
log_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(log_handler)
logging.getLogger('psai').addHandler(log_handler)
logging.getLogger('optuna').addHandler(log_handler)

def run_training_task():
    global is_training, psml_instance, training_logs
    is_training = True
    training_logs.clear()
    log_handler.seen_messages.clear()
    training_logs.append("Training started...")
    log_activity("Training started", "info")
    
    try:
        if psml_instance:
            # We don't need to manually log "Training models: ..." as psML updates its status now,
            # but we can still log high level events here if we want.
            
            psml_instance.optimize_all_models()
            psml_instance.build_ensemble_cv()
            psml_instance.build_ensemble_final()
            
            logger.info("Training completed successfully")
            training_logs.append("Training completed successfully!")
            log_activity("Training completed successfully", "success")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_logs.append(f"ERROR: {str(e)}")
        log_activity(f"Training failed: {str(e)}", "error")
        if psml_instance:
             psml_instance.training_status["state"] = "failed"
             psml_instance.training_status["message"] = f"Failed: {str(e)}"
    finally:
        is_training = False
        if psml_instance and psml_instance.training_status["state"] != "failed":
             psml_instance.training_status["state"] = "completed"

@app.post("/train", response_model=TrainResponse)
async def train(background_tasks: BackgroundTasks):
    global psml_instance, is_training
    
    if psml_instance is None:
        raise HTTPException(status_code=400, detail="psML not initialized. Call /init first.")
    
    if is_training:
        return {"message": "Training is already in progress", "task_id": "current"}
    
    background_tasks.add_task(run_training_task)
    return {"message": "Training started in background", "task_id": "new"}

@app.get("/status")
async def get_status():
    global is_training, psml_instance, training_logs
    
    status_data = {
        "is_initialized": psml_instance is not None,
        "is_training": is_training,
        "logs": training_logs[-20:],
        "detailed_status": None
    }
    
    if psml_instance:
        status_data["detailed_status"] = psml_instance.training_status
    
    return status_data

@app.get("/scores", response_model=ScoreOutput)
async def get_scores():
    global psml_instance
    if psml_instance is None:
         raise HTTPException(status_code=400, detail="psML not initialized.")
    try:
        scores = psml_instance.scores()
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        datasets_dir = "datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        
        file_path = os.path.join(datasets_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file_path}")
        return {"message": f"File uploaded successfully", "path": file_path}
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_name}", response_model=PredictionOutput)
async def predict(model_name: str, input_data: PredictionInput):
    global psml_instance
    if psml_instance is None:
         raise HTTPException(status_code=400, detail="psML not initialized.")
    
    try:
        df = pd.DataFrame(input_data.data)
        
        pipeline = None
        # Try finding the model in various keys
        # 1. Final model
        if f"final_model_{model_name}" in psml_instance.models:
             pipeline = psml_instance.models[f"final_model_{model_name}"]['model']
        # 2. Ensemble Stacking
        elif f"ensemble_stacking_{model_name}" in psml_instance.models:
             pipeline = psml_instance.models[f"ensemble_stacking_{model_name}"]['model']
        # 3. Ensemble Voting
        elif f"ensemble_voting_{model_name}" in psml_instance.models:
             pipeline = psml_instance.models[f"ensemble_voting_{model_name}"]['model']
        # 4. Direct match (rare but possible) or partial matches if user passes 'stacking_final'
        elif model_name in psml_instance.models and 'model' in psml_instance.models[model_name]:
             pipeline = psml_instance.models[model_name]['model']

        if pipeline is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found or not trained.")

        if psml_instance.config['dataset']['task_type'] == 'classification':
            if psml_instance.is_multiclass:
                preds = pipeline.predict_proba(df)
            else:
                 if hasattr(pipeline, "predict_proba"):
                    preds = pipeline.predict_proba(df)[:, 1]
                 else:
                    preds = pipeline.predict(df)
        else:
            preds = pipeline.predict(df)
        
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
            
        return {"predictions": preds}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{model_name}")
async def explain(model_name: str):
    global psml_instance
    if psml_instance is None:
         raise HTTPException(status_code=400, detail="psML not initialized.")

    try:
        if f'final_model_{model_name}' in psml_instance.models:
            if 'shap_image' in psml_instance.models[f'final_model_{model_name}']:
                return {"image_base64": psml_instance.models[f'final_model_{model_name}']['shap_image']}
        
        psml_instance.explain_model(model_name, plot=True)
        
        if f'final_model_{model_name}' in psml_instance.models:
            if 'shap_image' in psml_instance.models[f'final_model_{model_name}']:
                 return {"image_base64": psml_instance.models[f'final_model_{model_name}']['shap_image']}
        
        raise HTTPException(status_code=404, detail="Could not generate explanation.")

    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_model(path: str):
    global psml_instance
    if psml_instance is None:
         raise HTTPException(status_code=400, detail="psML not initialized.")
    try:
        psml_instance.save(path)
        return {"message": f"Model saved to {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load")
async def load_model(path: str):
    global psml_instance
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found")
        
        psml_instance = psML.load(path)
        return {"message": f"Model loaded from {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch/{model_name}", response_model=BatchPredictionOutput)
async def predict_batch(model_name: str, file: UploadFile = File(...)):
    global psml_instance, global_config
    if psml_instance is None:
         raise HTTPException(status_code=400, detail="psML not initialized.")
    
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
            
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Identify ID column
        id_col = global_config['dataset'].get('id_column', 'id')
        
        ids = []
        if id_col in df.columns:
            ids = df[id_col].tolist()
            # We assume psML handles dropping the ID column via preprocessor or model pipeline 
            # if it wasn't trained with it. But usually preprocessors drop unused cols.
            # To be safe and consistent with single predict, we pass the dataframe as is,
            # trusting the preprocessor to select correct columns.
        else:
            # If ID column missing, use index
            ids = df.index.tolist()
            
        pipeline = None
        # Try finding the model (logic copied from single predict)
        if f"final_model_{model_name}" in psml_instance.models:
             pipeline = psml_instance.models[f"final_model_{model_name}"]['model']
        elif f"ensemble_stacking_{model_name}" in psml_instance.models:
             pipeline = psml_instance.models[f"ensemble_stacking_{model_name}"]['model']
        elif f"ensemble_voting_{model_name}" in psml_instance.models:
             pipeline = psml_instance.models[f"ensemble_voting_{model_name}"]['model']
        elif model_name in psml_instance.models and 'model' in psml_instance.models[model_name]:
             pipeline = psml_instance.models[model_name]['model']

        if pipeline is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found or not trained.")

        if psml_instance.config['dataset']['task_type'] == 'classification':
            if psml_instance.is_multiclass:
                preds = pipeline.predict_proba(df)
            else:
                 if hasattr(pipeline, "predict_proba"):
                    preds = pipeline.predict_proba(df)[:, 1]
                 else:
                    preds = pipeline.predict(df)
        else:
            preds = pipeline.predict(df)
        
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
            
        # Combine IDs and predictions
        results = []
        for i, pred in enumerate(preds):
            results.append({
                "id": ids[i],
                "prediction": pred
            })
            
        return {"predictions": results}

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Agent Endpoints ---

@app.post("/agent/init", response_model=AgentInitResponse)
async def init_agent(input: AgentInitInput):
    """
    Initialize the AI Data Scientist agent.
    """
    global agent_instance, global_config
    
    try:
        # Determine dataset path
        # If not provided, try to use what's in global_config or default
        dataset_path = input.dataset_path
        if not dataset_path:
            dataset_path = global_config['dataset'].get('train_path')
            
        if not dataset_path:
             raise HTTPException(status_code=400, detail="Dataset path not provided and not in config.")
             
        if not os.path.exists(dataset_path):
             raise HTTPException(status_code=400, detail=f"Dataset file not found: {dataset_path}")
             
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset from {dataset_path}, shape: {df.shape}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load CSV: {e}")

        # Determine target
        target = input.target
        if not target:
            target = global_config['dataset'].get('target')
            
        if not target:
             # Try to guess or fail? Let's fail if not provided, agent needs target.
             # Or maybe we can ask LLM to guess later? For now, require it.
             raise HTTPException(status_code=400, detail="Target column not provided.")
             
        if target not in df.columns:
             raise HTTPException(status_code=400, detail=f"Target '{target}' not found in dataset columns.")

        # Initialize DataScientist
        agent_instance = DataScientist(
            df=df,
            target=target,
            model_provider=input.provider,
            model_name=input.model,
            model_temperature=input.temperature,
            optuna_trials=input.optuna_trials,
            optuna_metric=input.optuna_metric,
            task_type=input.task_type,
            api_key=input.api_key
        )
        
        # Sync simple settings back to global config
        global_config['dataset']['train_path'] = dataset_path
        global_config['dataset']['target'] = target
        
        log_activity("AI Agent initialized", "info")
        
        # Say Hello!
        greeting = agent_instance.say_hello()
        
        return AgentInitResponse(config=input, message=greeting)

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/step/{step_name}", response_model=AgentStepResponse)
async def run_agent_step(step_name: str):
    """
    Run a specific step of the AI Agent workflow.
    """
    global agent_instance, psml_instance
    
    if agent_instance is None:
        raise HTTPException(status_code=400, detail="Agent not initialized. Call /agent/init first.")
        
    try:
        response = {}
        
        if step_name == "eda":
            response = agent_instance.consult_eda()
            
        elif step_name == "feature_eng":
            response = agent_instance.consult_feature_engineering(execute_code=True, output_config=False)
            
        elif step_name == "dataset_config":
            response = agent_instance.consult_dataset_config(execute_code=True, output_config=False)
            
        elif step_name == "preprocessor":
            response = agent_instance.consult_preprocessor_config(execute_code=True, output_config=False)
            
        elif step_name == "models":
            response = agent_instance.consult_models_config(execute_code=True, output_config=False)
            
        elif step_name == "training":
            # This step runs analysis which creates psML instance
            # We capture stdout/logic? run_analysis prints a lot but doesn't return text analysis.
            # We wrap it to return a success message.
            try:
                agent_instance.run_analysis()
                # Sync psml instance
                psml_instance = agent_instance.psml
                response = {
                    "message": "Training Completed",
                    "llm_output": "Models have been trained and optimized successfully. Check the dashboard for detailed logs.",
                    "config": None
                }
            except Exception as e:
                response = {
                    "message": "Training Failed",
                    "llm_output": f"Error during training: {str(e)}",
                    "config": None
                }
                raise e # Re-raise to be caught by outer try/except
                
        elif step_name == "ensemble":
            # This runs consult_ensamble_config AND run_analysis_ensamble
            config_response = agent_instance.consult_ensamble_config(execute_code=True)
            if config_response.get('error'):
                 response = config_response
            else:
                 agent_instance.run_analysis_ensamble()
                 # Sync psml
                 psml_instance = agent_instance.psml
                 
                 llm_text = config_response.get('llm_output', '')
                 response = {
                     "message": "Ensemble Building Completed",
                     "llm_output": f"{llm_text}\n\nEnsembles have been built successfully.",
                     "config": config_response.get('config')
                 }
        
        elif step_name == "results":
            response = agent_instance.consult_results()
            
        elif step_name == "shap":
            # For SHAP, the agent's method takes a model_name.
            # We need to pick a model. Usually the best one or let user choose.
            # For simplicity in this step flow, let's try to explain the 'best' model or first available.
            model_name = "unknown"
            if agent_instance.psml and agent_instance.psml.models:
                 # Try to find a valid model name
                 for name in agent_instance.psml.models.keys():
                     if name.startswith("final_model_"):
                         model_name = name.replace("final_model_", "")
                         break
            
            if model_name == "unknown":
                 response = {"message": "SHAP Skipped", "llm_output": "No trained models found to explain."}
            else:
                 response = agent_instance.consult_shap(model_name)
                 
        else:
            raise HTTPException(status_code=400, detail=f"Unknown step: {step_name}")

        # Map response to schema
        # Agent methods return dict with keys: message, llm_output, config, etc.
        # We might need to serialize config
        
        config_val = response.get("config")
        # Ensure config is dict or str or None
        
        return AgentStepResponse(
            message=response.get("message", "Step completed"),
            llm_output=response.get("llm_output", ""),
            config=config_val
        )

    except Exception as e:
        logger.error(f"Agent step '{step_name}' failed: {e}")
        # Return error as valid response or raise?
        # If we raise 500, frontend sees error.
        raise HTTPException(status_code=500, detail=f"Step execution failed: {str(e)}")


@app.post("/agent/chat", response_model=AgentStepResponse)
async def agent_chat(message: str):
    """
    Send a message to the agent and get a response.
    """
    global agent_instance
    if agent_instance is None:
        raise HTTPException(status_code=400, detail="Agent not initialized.")
        
    try:
        from langchain_core.messages import HumanMessage
        response = agent_instance.agent.invoke(
            {"messages": [HumanMessage(content=message)]}, 
            config=agent_instance.config
        )
        text = agent_instance._final_message_text(response)
        return AgentStepResponse(
            message="Message sent",
            llm_output=text
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
