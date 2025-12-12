import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { FullConfig, StatusResponse, ModelScore, Activity } from '../types';

// Default configuration matching psai config.py
const defaultConfig: FullConfig = {
    mlflow: {
        enabled: false,
        experiment_name: 'Experiment Optuna Tuner',
        tracking_uri: 'mlruns',
    },
    dataset: {
        train_path: 'datasets/train.csv',
        target: 'target',
        id_column: 'id',
        test_size: 0.2,
        task_type: 'regression',
        metric: 'rmse',
        cv_folds: 5,
        random_state: 42,
        verbose: 2,
    },
    preprocessor: {
        numerical: { imputer: 'mean', scaler: 'standard' },
        skewed: { imputer: 'median', scaler: 'log' },
        outlier: { imputer: 'median', scaler: 'log' },
        low_cardinality: { imputer: 'most_frequent', encoder: 'onehot', scaler: 'none' },
        high_cardinality: { imputer: 'most_frequent', encoder: 'target', scaler: 'none' },
        dimension_reduction: { method: 'none' },
    },
    models: {
        lightgbm: {
            enabled: true,
            optuna_trials: 10,
            optuna_timeout: 3600,
            optuna_metric: 'rmse',
            optuna_n_jobs: 1,
            params: { verbose: 2, objective: 'rmse', device: 'gpu', eval_metric: 'rmse', num_threads: 8 },
            optuna_params: {
                boosting_type: { type: 'categorical', choices: ['gbdt', 'goss'] },
                learning_rate: { type: 'float', low: 0.001, high: 0.2, log: true },
                num_leaves: { type: 'int', low: 20, high: 300 },
                max_depth: { type: 'int', low: 3, high: 20 },
            },
        },
        xgboost: {
            enabled: true,
            optuna_trials: 10,
            optuna_timeout: 3600,
            optuna_metric: 'rmse',
            optuna_n_jobs: 1,
            params: { verbose: 2, objective: 'reg:squarederror', device: 'gpu', eval_metric: 'rmse', nthread: 8 },
            optuna_params: {
                booster: { type: 'categorical', choices: ['gbtree'] },
                learning_rate: { type: 'float', low: 0.001, high: 0.2, log: true },
                max_depth: { type: 'int', low: 3, high: 20 },
                n_estimators: { type: 'int', low: 500, high: 3000 },
            },
        },
        catboost: {
            enabled: true,
            optuna_trials: 10,
            optuna_timeout: 3600,
            optuna_metric: 'rmse',
            optuna_n_jobs: 1,
            params: { verbose: 2, objective: 'RMSE', device: 'gpu', eval_metric: 'RMSE', thread_count: 8 },
            optuna_params: {
                learning_rate: { type: 'float', low: 0.001, high: 0.2, log: true },
                depth: { type: 'int', low: 4, high: 10 },
                n_estimators: { type: 'int', low: 100, high: 3000 },
            },
        },
        random_forest: {
            enabled: false,
            optuna_trials: 10,
            optuna_timeout: 3600,
            optuna_metric: 'rmse',
            optuna_n_jobs: 4,
            params: { verbose: 2, n_jobs: 4 },
            optuna_params: {
                n_estimators: { type: 'int', low: 100, high: 1000 },
                max_depth: { type: 'int', low: 3, high: 30 },
            },
        },
        pytorch: {
            enabled: false,
            optuna_trials: 10,
            optuna_timeout: 3600,
            optuna_metric: 'rmse',
            optuna_n_jobs: 1,
            params: { train_max_epochs: 50, train_patience: 5, objective: 'mse', device: 'gpu', verbose: 2 },
            optuna_params: {
                model_type: { type: 'categorical', choices: ['mlp', 'ft_transformer'] },
                learning_rate: { type: 'categorical', choices: [0.01, 0.001] },
                batch_size: { type: 'categorical', choices: [64, 128, 256] },
            },
        },
    },
    stacking: {
        cv_enabled: false,
        cv_models: ['lightgbm', 'xgboost', 'catboost'],
        cv_folds: 5,
        final_enabled: false,
        final_models: ['lightgbm', 'xgboost', 'catboost'],
        meta_model: 'lightgbm',
        use_features: false,
        prefit: true,
    },
    voting: {
        cv_enabled: false,
        cv_models: ['lightgbm', 'xgboost', 'catboost'],
        final_enabled: false,
        final_models: ['lightgbm', 'xgboost', 'catboost'],
        use_features: false,
        prefit: true,
    },
    output: {
        models_dir: 'models',
        results_dir: 'outputs',
        save_models: true,
        save_predictions: true,
        save_feature_importance: true,
    },
};

interface AppState {
    // Config
    config: FullConfig;
    setConfig: (config: Partial<FullConfig>) => void;
    updateDatasetConfig: (updates: Partial<FullConfig['dataset']>) => void;
    updateModelConfig: (model: keyof FullConfig['models'], updates: Partial<FullConfig['models']['lightgbm']>) => void;

    // Status
    status: StatusResponse;
    setStatus: (status: StatusResponse) => void;

    // Scores
    scores: Record<string, ModelScore>;
    setScores: (scores: Record<string, ModelScore>) => void;

    // UI State
    isConnected: boolean;
    setIsConnected: (connected: boolean) => void;

    // Feature Engineering Code
    featureEngCode: string;
    setFeatureEngCode: (code: string) => void;

    // Recent Activity
    activities: Activity[];
    addActivity: (message: string, type?: Activity['type']) => void;
    setActivities: (activities: Activity[]) => void;
}

export const useStore = create<AppState>()(
    persist(
        (set) => ({
            config: defaultConfig,
            setConfig: (newConfig) => set((state) => ({
                config: { ...state.config, ...newConfig }
            })),
            updateDatasetConfig: (updates) => set((state) => ({
                config: {
                    ...state.config,
                    dataset: { ...state.config.dataset, ...updates },
                },
            })),
            updateModelConfig: (model, updates) => set((state) => ({
                config: {
                    ...state.config,
                    models: {
                        ...state.config.models,
                        [model]: { ...state.config.models[model], ...updates },
                    },
                },
            })),

            status: { is_initialized: false, is_training: false, logs: [], detailed_status: null },
            setStatus: (status) => set({ status }),

            scores: {},
            setScores: (scores) => set({ scores }),

            isConnected: false,
            setIsConnected: (connected) => set({ isConnected: connected }),

            featureEngCode: `class FeatureEngTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Add your feature engineering here
        df = X.copy()
        return df`,
            setFeatureEngCode: (code) => set({ featureEngCode: code }),

            activities: [],
            addActivity: (message, type = 'info') => set((state) => ({
                activities: [
                    { id: Date.now().toString(), message, timestamp: new Date().toISOString(), type },
                    ...state.activities
                ].slice(0, 50)
            })),
            setActivities: (activities) => set({ activities }),
        }),
        {
            name: 'psai-config-storage',  // localStorage key
            partialize: (state) => ({
                config: state.config,
                featureEngCode: state.featureEngCode,
                activities: state.activities,
            }),  // Persist config, code, and activities
        }
    )
);
