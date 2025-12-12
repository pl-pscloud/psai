import axios from 'axios';
import type {
    FullConfig,
    InitResponse,
    TrainResponse,
    StatusResponse,
    ScoresResponse,
    PredictionResponse,
    BatchPredictionResponse,
    ExplainResponse,
    UploadResponse,
    Activity,
    AgentInitInput,
    AgentInitResponse,
    AgentStepResponse
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Health check
export const checkHealth = async (): Promise<{ message: string }> => {
    const response = await api.get('/');
    return response.data;
};

// Get current status
export const getStatus = async (): Promise<StatusResponse> => {
    const response = await api.get('/status');
    return response.data;
};

// Get global config
export const getConfig = async (): Promise<FullConfig> => {
    const response = await api.get('/config');
    return response.data;
};

// Update global config
export const updateConfig = async (config: Partial<FullConfig>): Promise<{ message: string; config: FullConfig }> => {
    const response = await api.post('/config', { config });
    return response.data;
};

// Get activities
export const getActivities = async (): Promise<Activity[]> => {
    const response = await api.get('/activities');
    return response.data;
};

// Set feature transformer code
export const setFeatureTransformer = async (code: string): Promise<{ message: string }> => {
    const response = await api.post('/config/feature_transformer', { code });
    return response.data;
};

// Initialize psML with configuration
export const initializePipeline = async (
    config: FullConfig,
    trainPath?: string,
    target?: string
): Promise<InitResponse> => {
    const response = await api.post('/init', {
        config,
        train_path: trainPath,
        target,
    });
    return response.data;
};

// Start training
export const startTraining = async (): Promise<TrainResponse> => {
    const response = await api.post('/train');
    return response.data;
};

// Get model scores
export const getScores = async (): Promise<ScoresResponse> => {
    const response = await api.get('/scores');
    return response.data;
};

// Make predictions
export const predict = async (
    modelName: string,
    data: Record<string, unknown>[]
): Promise<PredictionResponse> => {
    const response = await api.post(`/predict/${modelName}`, { data });
    return response.data;
};

// Make batch predictions from CSV
export const predictBatch = async (
    modelName: string,
    file: File
): Promise<BatchPredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post(`/predict_batch/${modelName}`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

// Get SHAP explanation
export const getExplanation = async (modelName: string): Promise<ExplainResponse> => {
    const response = await api.get(`/explain/${modelName}`);
    return response.data;
};

// Upload dataset
export const uploadDataset = async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload-dataset', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

// Save model
export const saveModel = async (path: string): Promise<{ message: string }> => {
    const response = await api.post('/save_model', { path });
    return response.data;
};

export const loadModel = async (path: string): Promise<{ message: string }> => {
    const response = await api.post('/load_model', { path });
    return response.data;
};

// Experiment Management
export const getExperiments = async (): Promise<{ experiments: any[] }> => {
    const response = await api.get('/experiments');
    return response.data;
};

export const saveExperiment = async (name: string, config: FullConfig): Promise<{ message: string }> => {
    const response = await api.post('/experiments', { name, config });
    return response.data;
};

export const loadExperiment = async (name: string): Promise<{ message: string; config: FullConfig; has_model: boolean }> => {
    const response = await api.post(`/experiments/${name}/load`);
    return response.data;
};

// Reset API
export const resetApi = async (): Promise<{ message: string }> => {
    const response = await api.post('/reset');
    return response.data;
};

// --- Agent API ---

export const initAgent = async (config: AgentInitInput): Promise<AgentInitResponse> => {
    const response = await api.post<AgentInitResponse>('/agent/init', config);
    return response.data;
};

export const runAgentStep = async (stepName: string): Promise<AgentStepResponse> => {
    const response = await api.post(`/agent/step/${stepName}`);
    return response.data;
};

export const agentChat = async (message: string): Promise<AgentStepResponse> => {
    const response = await api.post('/agent/chat', null, { params: { message } });
    return response.data;
};

export default api;
