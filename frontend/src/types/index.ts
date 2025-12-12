// API Types matching psai backend schemas
export interface Activity {
    id: string;
    message: string;
    type: 'info' | 'success' | 'error' | 'warning';
    timestamp: string;
}

export interface DatasetConfig {
    train_path: string;
    target: string;
    id_column?: string;
    test_size: number;
    task_type: 'classification' | 'regression';
    metric: string;
    cv_folds: number;
    random_state: number;
    verbose: number;
}

export interface PreprocessorConfig {
    numerical: {
        imputer: 'mean' | 'median' | 'most_frequent' | 'constant';
        scaler: 'standard' | 'minmax' | 'robust' | 'log' | 'none';
    };
    skewed: {
        imputer: 'mean' | 'median' | 'most_frequent';
        scaler: 'log' | 'standard' | 'minmax' | 'none';
    };
    outlier: {
        imputer: 'median' | 'mean';
        scaler: 'log' | 'standard' | 'minmax' | 'none';
    };
    low_cardinality: {
        imputer: 'most_frequent' | 'constant';
        encoder: 'onehot' | 'ordinal' | 'label';
        scaler: 'none' | 'standard' | 'minmax';
    };
    high_cardinality: {
        imputer: 'most_frequent' | 'constant';
        encoder: 'target' | 'frequency' | 'label';
        scaler: 'none' | 'standard' | 'minmax';
    };
    dimension_reduction: {
        method: 'none' | 'pca' | 'umap';
    };
}

export interface OptunaParams {
    type: 'int' | 'float' | 'categorical';
    low?: number;
    high?: number;
    log?: boolean;
    choices?: (string | number | boolean | null)[];
}

export interface ModelConfig {
    enabled: boolean;
    optuna_trials: number;
    optuna_timeout: number;
    optuna_metric: string;
    optuna_n_jobs: number;
    params: Record<string, unknown>;
    optuna_params: Record<string, OptunaParams>;
}

export interface ModelsConfig {
    lightgbm: ModelConfig;
    xgboost: ModelConfig;
    catboost: ModelConfig;
    random_forest: ModelConfig;
    pytorch: ModelConfig;
}

export interface StackingConfig {
    cv_enabled: boolean;
    cv_models: string[];
    cv_folds: number;
    final_enabled: boolean;
    final_models: string[];
    meta_model: string;
    use_features: boolean;
    prefit: boolean;
}

export interface VotingConfig {
    cv_enabled: boolean;
    cv_models: string[];
    final_enabled: boolean;
    final_models: string[];
    use_features: boolean;
    prefit: boolean;
}

export interface MLflowConfig {
    enabled: boolean;
    experiment_name: string;
    tracking_uri: string;
}

export interface OutputConfig {
    models_dir: string;
    results_dir: string;
    save_models: boolean;
    save_predictions: boolean;
    save_feature_importance: boolean;
}

export interface FullConfig {
    mlflow: MLflowConfig;
    dataset: DatasetConfig;
    preprocessor: PreprocessorConfig;
    models: ModelsConfig;
    stacking: StackingConfig;
    voting: VotingConfig;
    output: OutputConfig;
}

// API Response Types
export interface InitResponse {
    message: string;
    config: FullConfig;
}

export interface TrainResponse {
    message: string;
    task_id: string;
}

export interface ModelStatus {
    status: 'pending' | 'optimizing' | 'training_final' | 'completed' | 'failed';
    trials_completed: number;
    total_trials: number;
    best_score?: number | null;
    trial_history: { trial: number; score: number }[];
}

export interface DetailedStatus {
    state: 'idle' | 'training' | 'completed' | 'failed';
    current_stage?: 'optimization' | 'final_training' | 'ensemble_cv' | 'ensemble_final';
    current_model?: string;
    current_trial: number;
    total_trials: number;
    model_progress: Record<string, ModelStatus>;
    message: string;
    start_time?: number | null;
    end_time?: number | null;
}

export interface StatusResponse {
    is_initialized: boolean;
    is_training: boolean;
    logs: string[];
    detailed_status: DetailedStatus | null;
}

export interface ModelScore {
    cv_score?: number | string;
    test_score?: number | string;
    best_params?: Record<string, unknown>;
}

export interface ScoresResponse {
    scores: Record<string, ModelScore>;
}

export interface PredictionResponse {
    predictions: number[] | number[][];
}

export interface BatchPredictionResponse {
    predictions: { id: string | number; prediction: number | number[] }[];
}

export interface ExplainResponse {
    image_base64: string;
}

export interface UploadResponse {
    message: string;
    path: string;
}

// UI State Types
export interface TrainingProgress {
    model_name: string;
    current_trial: number;
    total_trials: number;
    best_score: number;
    status: 'pending' | 'running' | 'completed' | 'failed';
}

export interface AgentStep {
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    output?: string;
}

// Agent API Types

export interface AgentInitInput {
    dataset_path?: string;
    target?: string;
    provider: string;
    model: string;
    temperature: number;
    task_type?: string;
    optuna_trials?: number;
    optuna_metric?: string;
    api_key?: string;
}

export interface AgentInitResponse {
    config: AgentInitInput;
    message: string;
}

export interface AgentStepResponse {
    message: string;
    llm_output: string;
    config?: Record<string, any> | string | null;
}


// Route names
export type RouteName =
    | 'dashboard'
    | 'configuration'
    | 'training'
    | 'models'
    | 'agent'
    | 'explainability'
    | 'results';
