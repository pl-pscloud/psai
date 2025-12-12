from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union

class ConfigInput(BaseModel):
    config: Dict[str, Any]
    train_path: Optional[str] = None
    target: Optional[str] = None

class TransformerInput(BaseModel):
    code: str

class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

class PredictionOutput(BaseModel):
    predictions: List[Union[float, List[float]]]

class ScoreOutput(BaseModel):
    scores: Dict[str, Any]

class InitResponse(BaseModel):
    message: str
    config: Dict[str, Any]

class TrainResponse(BaseModel):
    message: str
    task_id: str

class ModelStatus(BaseModel):
    status: str
    trials_completed: int
    total_trials: int
    best_score: Optional[float] = None
    trial_history: List[Dict[str, Any]] = []

class DetailedStatus(BaseModel):
    state: str
    current_stage: Optional[str] = None
    current_model: Optional[str] = None
    current_trial: int = 0
    total_trials: int = 0
    model_progress: Dict[str, ModelStatus] = {}
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class StatusResponse(BaseModel):
    is_initialized: bool
    is_training: bool
    logs: List[str]
    detailed_status: Optional[DetailedStatus] = None

class ExperimentSaveInput(BaseModel):
    name: str
    config: Dict[str, Any]

class Activity(BaseModel):
    id: str
    message: str
    type: str  # 'info', 'success', 'error', 'warning'
    timestamp: str

class BatchPredictionOutput(BaseModel):
    predictions: List[Dict[str, Any]]

# --- Agent Schemas ---

class AgentInitInput(BaseModel):
    dataset_path: Optional[str] = None
    target: Optional[str] = None
    provider: str = "google"
    model: str = "gemini-2.0-flash"
    temperature: float = 1.0
    task_type: Optional[str] = "regression"
    optuna_trials: Optional[int] = 10
    optuna_metric: Optional[str] = "rmse"
    api_key: Optional[str] = None

class AgentInitResponse(BaseModel):
    config: AgentInitInput
    message: str

class AgentStepResponse(BaseModel):
    message: str
    llm_output: str
    config: Optional[Union[Dict[str, Any], str]] = None
