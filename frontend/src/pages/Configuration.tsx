import { useState } from 'react';
import {
    Save,
    FolderOpen,
    Database,
    Cpu,
    Layers,
    Code,
    Settings2,
    AlertCircle
} from 'lucide-react';
import { useStore } from '../store';
import { initializePipeline, uploadDataset, updateConfig, setFeatureTransformer, getStatus } from '../services/api';
import './Configuration.css';

type ConfigTab = 'dataset' | 'preprocessor' | 'models' | 'ensemble' | 'feature_eng';

export default function Configuration() {
    const [activeTab, setActiveTab] = useState<ConfigTab>('dataset');
    const [isInitializing, setIsInitializing] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
    const {
        config,
        updateDatasetConfig,
        updateModelConfig,
        featureEngCode,
        setFeatureEngCode,
        setConfig,
        setStatus,
        addActivity
    } = useStore();

    const handleInitialize = async () => {
        setIsInitializing(true);
        setMessage(null);
        try {
            // 1. Save global config
            await updateConfig(config);

            // 2. Save feature transformer code if present
            if (featureEngCode.trim()) {
                await setFeatureTransformer(featureEngCode);
            }

            // 3. Initialize pipeline (backend will load the saved config and transformer)
            await initializePipeline(config);

            // 4. Update status in store
            const newStatus = await getStatus();
            setStatus(newStatus);

            setMessage({ type: 'success', text: 'Pipeline initialized successfully!' });

            // Log activity
            const enabledModels = Object.entries(config.models)
                .filter(([_, m]) => m.enabled)
                .map(([k, m]) => `${k} (${m.optuna_trials} trials)`)
                .join(', ');

            addActivity(`Pipeline initialized. Enabled: ${enabledModels}`, 'success');

        } catch (error: unknown) {
            const errMsg = error instanceof Error ? error.message : 'Failed to initialize';
            setMessage({ type: 'error', text: errMsg });
            addActivity(`Initialization failed: ${errMsg}`, 'error');
        } finally {
            setIsInitializing(false);
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            try {
                const result = await uploadDataset(file);
                updateDatasetConfig({ train_path: result.path });
                setMessage({ type: 'success', text: 'Dataset uploaded successfully!' });
                addActivity(`Dataset uploaded successfully: ${result.path}`, 'success');
            } catch {
                setMessage({ type: 'error', text: 'Failed to upload dataset' });
                addActivity('Failed to upload dataset', 'error');
            }
        }
    };

    const tabs = [
        { id: 'dataset', label: 'Dataset', icon: Database },
        { id: 'preprocessor', label: 'Preprocessor', icon: Settings2 },
        { id: 'models', label: 'Models', icon: Cpu },
        { id: 'ensemble', label: 'Ensemble', icon: Layers },
        { id: 'feature_eng', label: 'Feature Eng.', icon: Code },
    ];

    return (
        <div className="configuration">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Configuration</h1>
                    <p className="page-subtitle">Configure your ML pipeline settings</p>
                </div>
                <button
                    className="btn btn-primary"
                    onClick={handleInitialize}
                    disabled={isInitializing}
                >
                    {isInitializing ? 'Initializing...' : 'Initialize Pipeline'}
                    <Save size={18} />
                </button>
            </div>

            {message && (
                <div className={`message ${message.type}`}>
                    <AlertCircle size={18} />
                    {message.text}
                </div>
            )}

            {/* Tabs */}
            <div className="tabs">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id as ConfigTab)}
                    >
                        <tab.icon size={16} />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="tab-content">
                {activeTab === 'dataset' && (
                    <div className="config-section animate-fade-in">
                        <div className="card">
                            <h3 className="card-title mb-4">Dataset Settings</h3>

                            <div className="form-row">
                                <div className="form-group">
                                    <label className="form-label">Training Data Path</label>
                                    <div className="input-with-button">
                                        <input
                                            type="text"
                                            className="input"
                                            value={config.dataset.train_path}
                                            onChange={(e) => updateDatasetConfig({ train_path: e.target.value })}
                                            placeholder="path/to/train.csv"
                                        />
                                        <label className="btn btn-secondary">
                                            <FolderOpen size={16} />
                                            <input type="file" accept=".csv" onChange={handleFileUpload} hidden />
                                        </label>
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Target Column</label>
                                    <input
                                        type="text"
                                        className="input"
                                        value={config.dataset.target}
                                        onChange={(e) => updateDatasetConfig({ target: e.target.value })}
                                        placeholder="target"
                                    />
                                </div>
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label className="form-label">Task Type</label>
                                    <select
                                        className="input select"
                                        value={config.dataset.task_type}
                                        onChange={(e) => {
                                            const newTaskType = e.target.value as 'classification' | 'regression';
                                            updateDatasetConfig({
                                                task_type: newTaskType,
                                                // Reset metric to default for the new task type
                                                metric: newTaskType === 'classification' ? 'auc' : 'rmse'
                                            });
                                        }}
                                    >
                                        <option value="regression">Regression</option>
                                        <option value="classification">Classification</option>
                                    </select>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Evaluation Metric</label>
                                    <select
                                        className="input select"
                                        value={config.dataset.metric}
                                        onChange={(e) => updateDatasetConfig({ metric: e.target.value })}
                                    >
                                        {config.dataset.task_type === 'regression' ? (
                                            <>
                                                <option value="rmse">RMSE</option>
                                                <option value="mae">MAE</option>
                                                <option value="mse">MSE</option>
                                                <option value="rmsle">RMSLE</option>
                                            </>
                                        ) : (
                                            <>
                                                <option value="auc">AUC</option>
                                                <option value="accuracy">Accuracy</option>
                                                <option value="f1">F1 Score</option>
                                                <option value="precision">Precision</option>
                                            </>
                                        )}
                                    </select>
                                </div>
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label className="form-label">Test Size</label>
                                    <input
                                        type="number"
                                        className="input"
                                        value={config.dataset.test_size}
                                        onChange={(e) => updateDatasetConfig({ test_size: parseFloat(e.target.value) })}
                                        min="0.1"
                                        max="0.5"
                                        step="0.05"
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="form-label">CV Folds</label>
                                    <input
                                        type="number"
                                        className="input"
                                        value={config.dataset.cv_folds}
                                        onChange={(e) => updateDatasetConfig({ cv_folds: parseInt(e.target.value) })}
                                        min="2"
                                        max="10"
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Random State</label>
                                    <input
                                        type="number"
                                        className="input"
                                        value={config.dataset.random_state}
                                        onChange={(e) => updateDatasetConfig({ random_state: parseInt(e.target.value) })}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'preprocessor' && (
                    <div className="config-section animate-fade-in">
                        <div className="card">
                            <h3 className="card-title mb-4">Preprocessing Settings</h3>

                            <div className="preprocessor-grid">
                                {/* Numerical */}
                                <div className="preproc-card">
                                    <h4>Numerical Features</h4>
                                    <div className="form-group">
                                        <label className="form-label">Imputer</label>
                                        <select
                                            className="input select"
                                            value={config.preprocessor.numerical.imputer}
                                            onChange={(e) => setConfig({
                                                preprocessor: {
                                                    ...config.preprocessor,
                                                    numerical: { ...config.preprocessor.numerical, imputer: e.target.value as 'mean' | 'median' }
                                                }
                                            })}
                                        >
                                            <option value="mean">Mean</option>
                                            <option value="median">Median</option>
                                            <option value="most_frequent">Most Frequent</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Scaler</label>
                                        <select
                                            className="input select"
                                            value={config.preprocessor.numerical.scaler}
                                            onChange={(e) => setConfig({
                                                preprocessor: {
                                                    ...config.preprocessor,
                                                    numerical: { ...config.preprocessor.numerical, scaler: e.target.value as 'standard' | 'minmax' }
                                                }
                                            })}
                                        >
                                            <option value="standard">Standard</option>
                                            <option value="minmax">MinMax</option>
                                            <option value="robust">Robust</option>
                                            <option value="none">None</option>
                                        </select>
                                    </div>
                                </div>

                                {/* Skewed */}
                                <div className="preproc-card">
                                    <h4>Skewed Features</h4>
                                    <div className="form-group">
                                        <label className="form-label">Imputer</label>
                                        <select
                                            className="input select"
                                            value={config.preprocessor.skewed.imputer}
                                            onChange={(e) => setConfig({
                                                preprocessor: {
                                                    ...config.preprocessor,
                                                    skewed: { ...config.preprocessor.skewed, imputer: e.target.value as 'mean' | 'median' }
                                                }
                                            })}
                                        >
                                            <option value="median">Median</option>
                                            <option value="mean">Mean</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Scaler</label>
                                        <select
                                            className="input select"
                                            value={config.preprocessor.skewed.scaler}
                                            onChange={(e) => setConfig({
                                                preprocessor: {
                                                    ...config.preprocessor,
                                                    skewed: { ...config.preprocessor.skewed, scaler: e.target.value as 'log' | 'standard' }
                                                }
                                            })}
                                        >
                                            <option value="log">Log</option>
                                            <option value="standard">Standard</option>
                                            <option value="none">None</option>
                                        </select>
                                    </div>
                                </div>

                                {/* Low Cardinality */}
                                <div className="preproc-card">
                                    <h4>Low Cardinality</h4>
                                    <div className="form-group">
                                        <label className="form-label">Encoder</label>
                                        <select
                                            className="input select"
                                            value={config.preprocessor.low_cardinality.encoder}
                                            onChange={(e) => setConfig({
                                                preprocessor: {
                                                    ...config.preprocessor,
                                                    low_cardinality: { ...config.preprocessor.low_cardinality, encoder: e.target.value as 'onehot' | 'ordinal' }
                                                }
                                            })}
                                        >
                                            <option value="onehot">One-Hot</option>
                                            <option value="ordinal">Ordinal</option>
                                            <option value="label">Label</option>
                                        </select>
                                    </div>
                                </div>

                                {/* High Cardinality */}
                                <div className="preproc-card">
                                    <h4>High Cardinality</h4>
                                    <div className="form-group">
                                        <label className="form-label">Encoder</label>
                                        <select
                                            className="input select"
                                            value={config.preprocessor.high_cardinality.encoder}
                                            onChange={(e) => setConfig({
                                                preprocessor: {
                                                    ...config.preprocessor,
                                                    high_cardinality: { ...config.preprocessor.high_cardinality, encoder: e.target.value as 'target' | 'frequency' }
                                                }
                                            })}
                                        >
                                            <option value="target">Target Encoding</option>
                                            <option value="frequency">Frequency</option>
                                            <option value="label">Label</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'models' && (
                    <div className="config-section animate-fade-in">
                        <div className="models-grid">
                            {(['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch'] as const).map((modelKey) => {
                                const modelConfig = config.models[modelKey];
                                return (
                                    <div key={modelKey} className={`card model-config-card ${modelConfig.enabled ? 'enabled' : ''}`}>
                                        <div className="model-header">
                                            <h4>{modelKey.replace('_', ' ').toUpperCase()}</h4>
                                            <button
                                                className={`toggle ${modelConfig.enabled ? 'active' : ''}`}
                                                onClick={() => updateModelConfig(modelKey, { enabled: !modelConfig.enabled })}
                                            />
                                        </div>

                                        {modelConfig.enabled && (
                                            <div className="model-settings">
                                                <h5 className="text-secondary font-medium mb-2">General Settings</h5>
                                                <div className="form-grid-3">
                                                    <div className="form-group">
                                                        <label className="form-label">Optuna Trials</label>
                                                        <input
                                                            type="number"
                                                            className="input"
                                                            value={modelConfig.optuna_trials}
                                                            onChange={(e) => updateModelConfig(modelKey, { optuna_trials: parseInt(e.target.value) })}
                                                            min="1"
                                                            max="1000"
                                                        />
                                                    </div>

                                                    <div className="form-group">
                                                        <label className="form-label">Timeout (sec)</label>
                                                        <input
                                                            type="number"
                                                            className="input"
                                                            value={modelConfig.optuna_timeout}
                                                            onChange={(e) => updateModelConfig(modelKey, { optuna_timeout: parseInt(e.target.value) })}
                                                            min="60"
                                                            step="60"
                                                        />
                                                    </div>

                                                    <div className="form-group">
                                                        <label className="form-label">Optuna Metric</label>
                                                        <select
                                                            className="input select"
                                                            value={modelConfig.optuna_metric}
                                                            onChange={(e) => updateModelConfig(modelKey, { optuna_metric: e.target.value })}
                                                        >
                                                            <option value="rmse">RMSE</option>
                                                            <option value="mae">MAE</option>
                                                            <option value="auc">AUC</option>
                                                            <option value="accuracy">Accuracy</option>
                                                        </select>
                                                    </div>
                                                </div>

                                                <h5 className="text-secondary font-medium mt-4 mb-2">Fixed Parameters</h5>
                                                <div className="params-grid">
                                                    {Object.entries(modelConfig.params).map(([key, value]) => {
                                                        const handleParamChange = (newValue: unknown) => {
                                                            updateModelConfig(modelKey, {
                                                                params: { ...modelConfig.params, [key]: newValue }
                                                            });
                                                        };

                                                        if (typeof value === 'boolean') {
                                                            return (
                                                                <div key={key} className="form-group checkbox-group">
                                                                    <label>
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={value}
                                                                            onChange={(e) => handleParamChange(e.target.checked)}
                                                                        />
                                                                        {key}
                                                                    </label>
                                                                </div>
                                                            );
                                                        }

                                                        // Handle complex objects/arrays as JSON string
                                                        if (typeof value === 'object' && value !== null) {
                                                            return (
                                                                <div key={key} className="form-group full-width">
                                                                    <label className="form-label">{key} (JSON)</label>
                                                                    <textarea
                                                                        className="input code-input"
                                                                        rows={3}
                                                                        defaultValue={JSON.stringify(value, null, 2)}
                                                                        onBlur={(e) => {
                                                                            try {
                                                                                const parsed = JSON.parse(e.target.value);
                                                                                handleParamChange(parsed);
                                                                            } catch {
                                                                                // Ignore invalid JSON on blur, maybe show error in future
                                                                                console.error('Invalid JSON');
                                                                            }
                                                                        }}
                                                                    />
                                                                </div>
                                                            );
                                                        }

                                                        return (
                                                            <div key={key} className="form-group">
                                                                <label className="form-label">{key}</label>
                                                                <input
                                                                    type={typeof value === 'number' ? 'number' : 'text'}
                                                                    className="input"
                                                                    value={value as string | number}
                                                                    onChange={(e) => {
                                                                        const val = e.target.value;
                                                                        handleParamChange(typeof value === 'number' ? parseFloat(val) : val);
                                                                    }}
                                                                />
                                                            </div>
                                                        );
                                                    })}
                                                </div>

                                                <h5 className="text-secondary font-medium mt-4 mb-2">Hyperparameter Search Space</h5>
                                                <div className="optuna-params-list">
                                                    {Object.entries(modelConfig.optuna_params).map(([key, param]) => (
                                                        <div key={key} className="optuna-param-row card-sub">
                                                            <div className="param-header">
                                                                <span className="param-name">{key}</span>
                                                                <span className="badge badge-secondary">{param.type}</span>
                                                            </div>

                                                            <div className="param-controls">
                                                                {(param.type === 'int' || param.type === 'float') && (
                                                                    <>
                                                                        <div className="control-group">
                                                                            <label>Min</label>
                                                                            <input
                                                                                type="number"
                                                                                className="input input-sm"
                                                                                value={param.low}
                                                                                step={param.type === 'float' ? '0.0001' : '1'}
                                                                                onChange={(e) => {
                                                                                    updateModelConfig(modelKey, {
                                                                                        optuna_params: {
                                                                                            ...modelConfig.optuna_params,
                                                                                            [key]: { ...param, low: parseFloat(e.target.value) }
                                                                                        }
                                                                                    });
                                                                                }}
                                                                            />
                                                                        </div>
                                                                        <div className="control-group">
                                                                            <label>Max</label>
                                                                            <input
                                                                                type="number"
                                                                                className="input input-sm"
                                                                                value={param.high}
                                                                                step={param.type === 'float' ? '0.0001' : '1'}
                                                                                onChange={(e) => {
                                                                                    updateModelConfig(modelKey, {
                                                                                        optuna_params: {
                                                                                            ...modelConfig.optuna_params,
                                                                                            [key]: { ...param, high: parseFloat(e.target.value) }
                                                                                        }
                                                                                    });
                                                                                }}
                                                                            />
                                                                        </div>
                                                                        <div className="control-group checkbox">
                                                                            <label title="Use Log Scale">
                                                                                <input
                                                                                    type="checkbox"
                                                                                    checked={param.log}
                                                                                    onChange={(e) => {
                                                                                        updateModelConfig(modelKey, {
                                                                                            optuna_params: {
                                                                                                ...modelConfig.optuna_params,
                                                                                                [key]: { ...param, log: e.target.checked }
                                                                                            }
                                                                                        });
                                                                                    }}
                                                                                />
                                                                                Log
                                                                            </label>
                                                                        </div>
                                                                    </>
                                                                )}

                                                                {param.type === 'categorical' && (
                                                                    <div className="control-group full-width">
                                                                        <label>Choices (comma separated or JSON)</label>
                                                                        <textarea
                                                                            className="input input-sm"
                                                                            rows={param.choices && typeof param.choices[0] === 'object' ? 4 : 1}
                                                                            defaultValue={JSON.stringify(param.choices)}
                                                                            onBlur={(e) => {
                                                                                try {
                                                                                    // Try to parse as JSON first
                                                                                    const parsed = JSON.parse(e.target.value);
                                                                                    updateModelConfig(modelKey, {
                                                                                        optuna_params: {
                                                                                            ...modelConfig.optuna_params,
                                                                                            [key]: { ...param, choices: parsed }
                                                                                        }
                                                                                    });
                                                                                } catch {
                                                                                    // Fallback: split by comma if simple string
                                                                                    const val = e.target.value;
                                                                                    if (!val.trim().startsWith('[')) {
                                                                                        const choices = val.split(',').map(s => s.trim());
                                                                                        updateModelConfig(modelKey, {
                                                                                            optuna_params: {
                                                                                                ...modelConfig.optuna_params,
                                                                                                [key]: { ...param, choices: choices }
                                                                                            }
                                                                                        });
                                                                                    }
                                                                                }
                                                                            }}
                                                                        />
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {activeTab === 'ensemble' && (
                    <div className="config-section animate-fade-in">
                        <div className="grid grid-cols-2">
                            {/* Stacking */}
                            <div className="card">
                                <div className="model-header">
                                    <h3 className="card-title">Stacking</h3>
                                    <button
                                        className={`toggle ${config.stacking.final_enabled ? 'active' : ''}`}
                                        onClick={() => setConfig({
                                            stacking: { ...config.stacking, final_enabled: !config.stacking.final_enabled }
                                        })}
                                    />
                                </div>

                                {config.stacking.final_enabled && (
                                    <div className="model-settings mt-4">
                                        <div className="form-group">
                                            <label className="form-label">Meta Model</label>
                                            <select
                                                className="input select"
                                                value={config.stacking.meta_model}
                                                onChange={(e) => setConfig({
                                                    stacking: { ...config.stacking, meta_model: e.target.value }
                                                })}
                                            >
                                                <option value="lightgbm">LightGBM</option>
                                                <option value="xgboost">XGBoost</option>
                                                <option value="ridge">Ridge</option>
                                            </select>
                                        </div>

                                        <div className="checkbox-group">
                                            <label>
                                                <input
                                                    type="checkbox"
                                                    checked={config.stacking.use_features}
                                                    onChange={(e) => setConfig({
                                                        stacking: { ...config.stacking, use_features: e.target.checked }
                                                    })}
                                                />
                                                Use Original Features
                                            </label>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Voting */}
                            <div className="card">
                                <div className="model-header">
                                    <h3 className="card-title">Voting</h3>
                                    <button
                                        className={`toggle ${config.voting.final_enabled ? 'active' : ''}`}
                                        onClick={() => setConfig({
                                            voting: { ...config.voting, final_enabled: !config.voting.final_enabled }
                                        })}
                                    />
                                </div>

                                {config.voting.final_enabled && (
                                    <div className="model-settings mt-4">
                                        <p className="text-sm text-muted">
                                            Voting ensemble will combine predictions from all enabled models.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'feature_eng' && (
                    <div className="config-section animate-fade-in">
                        <div className="card">
                            <h3 className="card-title mb-4">Feature Engineering Transformer</h3>
                            <p className="text-sm text-muted mb-4">
                                Define a custom transformer class that will be applied before preprocessing.
                            </p>
                            <textarea
                                className="code-editor"
                                value={featureEngCode}
                                onChange={(e) => setFeatureEngCode(e.target.value)}
                                spellCheck={false}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
