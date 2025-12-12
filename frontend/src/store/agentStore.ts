import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AgentInitInput } from '../types';

export interface AgentStep {
    id: string;
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    description: string;
}

interface Message {
    role: 'user' | 'agent';
    content: string;
    image?: string;
}

interface AgentState {
    // State
    steps: AgentStep[];
    isInitialized: boolean;
    isRunning: boolean;
    messages: Message[];
    agentConfig: AgentInitInput;

    // Actions
    setSteps: (steps: AgentStep[] | ((prev: AgentStep[]) => AgentStep[])) => void;
    setIsInitialized: (initialized: boolean) => void;
    setIsRunning: (running: boolean) => void;
    setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void;
    addMessage: (role: 'user' | 'agent', content: string, image?: string) => void;
    setAgentConfig: (config: AgentInitInput) => void;
    resetAgent: () => void;
    resetSession: () => void;
}

const defaultSteps: AgentStep[] = [
    { id: 'eda', name: 'EDA Analysis', status: 'pending', description: 'Analyze dataset characteristics' },
    { id: 'feature_eng', name: 'Feature Engineering', status: 'pending', description: 'Generate feature transformations' },
    { id: 'dataset_config', name: 'Dataset Config', status: 'pending', description: 'Configure dataset parameters' },
    { id: 'preprocessor', name: 'Preprocessor Config', status: 'pending', description: 'Configure preprocessing pipeline' },
    { id: 'models', name: 'Model Selection', status: 'pending', description: 'Select and configure models' },
    { id: 'training', name: 'Training', status: 'pending', description: 'Train and optimize models' },
    { id: 'ensemble', name: 'Ensemble Building', status: 'pending', description: 'Build stacking/voting ensembles' },
    { id: 'results', name: 'Results Analysis', status: 'pending', description: 'Analyze model performance' },
    { id: 'shap', name: 'SHAP Explainability', status: 'pending', description: 'Generate model explanations' },
];

const defaultConfig: AgentInitInput = {
    provider: 'google',
    model: 'gemini-2.0-flash',
    temperature: 1.0,
    task_type: 'regression',
    optuna_trials: 10,
    optuna_metric: 'rmse',
};

export const useAgentStore = create<AgentState>()(
    persist(
        (set) => ({
            steps: defaultSteps,
            isInitialized: false,
            isRunning: false,
            messages: [],
            agentConfig: defaultConfig,

            setSteps: (steps) => set((state) => ({
                steps: typeof steps === 'function' ? steps(state.steps) : steps
            })),

            setIsInitialized: (initialized) => set({ isInitialized: initialized }),

            setIsRunning: (running) => set({ isRunning: running }),

            setMessages: (messages) => set((state) => ({
                messages: typeof messages === 'function' ? messages(state.messages) : messages
            })),

            addMessage: (role, content, image) => set((state) => ({
                messages: [...state.messages, { role, content, image }]
            })),

            setAgentConfig: (config) => set((state) => ({
                agentConfig: { ...state.agentConfig, ...config }
            })),

            resetAgent: () => set({
                steps: defaultSteps,
                isInitialized: false,
                isRunning: false,
                messages: [],
                agentConfig: defaultConfig
            }),

            resetSession: () => set({
                steps: defaultSteps,
                isInitialized: false,
                isRunning: false,
                messages: [],
                // agentConfig is intentionally NOT reset to preserve user settings
            }),
        }),
        {
            name: 'psai-agent-storage',
            partialize: (state) => ({
                steps: state.steps,
                isInitialized: state.isInitialized,
                messages: state.messages,
                agentConfig: state.agentConfig,
                // Don't persist isRunning to prevent stuck state on reload
            }),
        }
    )
);
