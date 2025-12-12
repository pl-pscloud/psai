import { useState, useEffect } from 'react';
import {
    Play,
    Square,
    RefreshCw,
    CheckCircle2,
    Clock,
    AlertCircle,
    Loader2
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell
} from 'recharts';
import { useStore } from '../store';
import { startTraining, getStatus, checkHealth } from '../services/api';
import './Training.css';

const modelColors: Record<string, string> = {
    lightgbm: '#a855f7',
    xgboost: '#22d3ee',
    catboost: '#f97316',
    random_forest: '#10b981',
    pytorch: '#ec4899',
};

export default function Training() {
    const { status, setStatus, config, isConnected, setIsConnected } = useStore();
    const [trainingLogs, setTrainingLogs] = useState<string[]>([
        '[INFO] Waiting to start training...'
    ]);

    // Check connection and status on mount
    useEffect(() => {
        const checkConnection = async () => {
            try {
                await checkHealth();
                setIsConnected(true);
                const statusData = await getStatus();
                setStatus(statusData);
            } catch {
                setIsConnected(false);
            }
        };
        checkConnection();
    }, [setIsConnected, setStatus]);

    // Poll for status and logs when training is in progress
    useEffect(() => {
        // Always fetch status initially to get logs
        const fetchStatusAndLogs = async () => {
            try {
                const newStatus = await getStatus();
                setStatus(newStatus);

                // Sync logs from backend
                if (newStatus.logs && newStatus.logs.length > 0) {
                    setTrainingLogs(newStatus.logs.map((l: string) => `[API] ${l}`));
                }
            } catch (e) {
                console.error('Status fetch failed:', e);
            }
        };

        // Fetch immediately
        fetchStatusAndLogs();

        // Set up polling interval - always poll but more frequently during training
        const pollInterval = setInterval(fetchStatusAndLogs, status.is_training ? 2000 : 5000);

        return () => clearInterval(pollInterval);
    }, [status.is_training, setStatus]);

    const handleStartTraining = async () => {
        try {
            setTrainingLogs(['[INFO] Starting training...']);
            await startTraining();
            setTrainingLogs(prev => [...prev, '[INFO] Training started successfully']);
            // Polling will be handled by the useEffect above
        } catch (error) {
            setTrainingLogs(prev => [...prev, `[ERROR] Failed to start training: ${error}`]);
        }
    };

    const handleRefreshLogs = async () => {
        try {
            const newStatus = await getStatus();
            setStatus(newStatus);
            if (newStatus.logs && newStatus.logs.length > 0) {
                setTrainingLogs(newStatus.logs.map((l: string) => `[API] ${l}`));
            }
        } catch {
            setTrainingLogs(['[INFO] Failed to fetch logs']);
        }
    };

    const enabledModels = [
        ...Object.entries(config.models)
            .filter(([, cfg]) => cfg.enabled)
            .map(([name]) => name),
        // Add Ensembles
        ...(config.stacking?.cv_enabled ? ['stacking_cv'] : []),
        ...(config.stacking?.final_enabled ? ['stacking_final'] : []),
        ...(config.voting?.cv_enabled ? ['voting_cv'] : []),
        ...(config.voting?.final_enabled ? ['voting_final'] : []),
    ];

    // Sample trial data for visualization
    const trialData = enabledModels.map(model => {
        const completed = status.detailed_status?.model_progress?.[model]?.trials_completed ?? 0;
        let totalTrials = 1;

        if (model.includes('stacking') || model.includes('voting')) {
            totalTrials = 1;
        } else {
            totalTrials = config.models[model as keyof typeof config.models]?.optuna_trials ?? 1;
        }

        return {
            model: model.replace('_', ' '),
            trials: totalTrials,
            completed: completed,
        };
    });

    return (
        <div className="training">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Training</h1>
                    <p className="page-subtitle">Train and optimize your models with Optuna</p>
                </div>
                <div className="training-actions">
                    {status.is_training ? (
                        <button className="btn btn-secondary" disabled>
                            <Square size={18} />
                            Stop Training
                        </button>
                    ) : (
                        <div title={!isConnected ? "Backend not connected" : !status.is_initialized ? "Pipeline not initialized. Go to Configuration > Initialize." : ""}>
                            <button
                                className="btn btn-success"
                                onClick={handleStartTraining}
                                disabled={!isConnected || !status.is_initialized}
                            >
                                <Play size={18} />
                                Start Training
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Status Cards */}
            <div className="training-status-grid">
                <div className={`status-card ${status.is_initialized ? 'success' : ''}`}>
                    <div className="status-icon">
                        {status.is_initialized ? <CheckCircle2 size={24} /> : <AlertCircle size={24} />}
                    </div>
                    <div className="status-info">
                        <span className="status-label">Pipeline Status</span>
                        <span className="status-value">{status.is_initialized ? 'Initialized' : 'Not Initialized'}</span>
                    </div>
                </div>

                <div className={`status-card ${status.is_training ? 'active' : ''}`}>
                    <div className="status-icon">
                        {status.is_training ? <Loader2 size={24} className="animate-spin" /> : <Clock size={24} />}
                    </div>
                    <div className="status-info">
                        <span className="status-label">Training Status</span>
                        <span className="status-value">{status.is_training ? 'In Progress' : 'Idle'}</span>
                    </div>
                </div>

                <div className="status-card">
                    <div className="status-icon">
                        <RefreshCw size={24} />
                    </div>
                    <div className="status-info">
                        <span className="status-label">Enabled Models</span>
                        <span className="status-value">{enabledModels.length}</span>
                    </div>
                </div>
            </div>

            <div className="training-content">
                {/* Model Progress */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Model Training Progress</h3>
                    </div>

                    <div className="models-progress">
                        {enabledModels.map((model) => {
                            const isEnsemble = model.includes('stacking') || model.includes('voting');
                            let totalTrials = 1;
                            if (!isEnsemble) {
                                totalTrials = config.models[model as keyof typeof config.models]?.optuna_trials;
                            }

                            // Parse detailed status
                            const detailedStatus = status.detailed_status;
                            const modelStatusRaw = detailedStatus?.model_progress?.[model];

                            // Default values
                            let progressPercent = 0;
                            let statusText = 'Waiting';
                            let trialsCompleted = 0;

                            if (modelStatusRaw) {
                                trialsCompleted = modelStatusRaw.trials_completed;
                                if (modelStatusRaw.status === 'completed') {
                                    progressPercent = 100;
                                    statusText = 'Completed';
                                } else if (modelStatusRaw.status === 'optimizing') {
                                    if (modelStatusRaw.total_trials > 0) {
                                        progressPercent = Math.min(99, Math.round((trialsCompleted / modelStatusRaw.total_trials) * 100));
                                    } else {
                                        progressPercent = 0;
                                    }
                                    statusText = `Trial ${trialsCompleted} / ${modelStatusRaw.total_trials}`;
                                } else if (modelStatusRaw.status === 'training_final') {
                                    progressPercent = 99;
                                    statusText = 'Final Training...';
                                } else if (modelStatusRaw.status === 'pending') {
                                    statusText = 'Pending';
                                    progressPercent = 0;
                                }

                                // Special handling for ensembles
                                if (isEnsemble && modelStatusRaw.status !== 'completed' && modelStatusRaw.status !== 'pending') {
                                    progressPercent = 50;
                                    statusText = 'Training Ensemble...';
                                }
                            }

                            const isCurrentModel = detailedStatus?.current_model === model;

                            return (
                                <div key={model} className="model-progress-item">
                                    <div className="model-progress-header">
                                        <span className="model-name">{model.replace('_', ' ').toUpperCase()}</span>
                                        <span className="model-trials">{trialsCompleted} / {totalTrials} trials</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div
                                            className="progress-fill"
                                            style={{
                                                width: `${progressPercent}%`,
                                                background: modelColors[model] || '#a855f7'
                                            }}
                                        />
                                    </div>
                                    <div className="model-progress-footer">
                                        <span className="progress-percent">{progressPercent}%</span>
                                        <span className="progress-status">
                                            {isCurrentModel && (statusText === 'Waiting' || statusText === 'Pending')
                                                ? 'Starting...'
                                                : statusText}
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Trials Chart */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Optuna Trials Configuration</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={trialData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                            <XAxis type="number" stroke="#6b7280" fontSize={12} />
                            <YAxis dataKey="model" type="category" stroke="#6b7280" fontSize={12} width={100} />
                            <Tooltip
                                contentStyle={{
                                    background: '#1a1a24',
                                    border: '1px solid #2a2a3a',
                                    borderRadius: '8px'
                                }}
                            />
                            <Bar dataKey="trials" radius={[0, 4, 4, 0]}>
                                {trialData.map((_, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={modelColors[enabledModels[index]] || '#a855f7'}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Logs */}
            <div className="card mt-4">
                <div className="card-header">
                    <h3 className="card-title">Training Logs</h3>
                    <button
                        className="btn btn-secondary btn-icon"
                        onClick={handleRefreshLogs}
                    >
                        <RefreshCw size={16} />
                    </button>
                </div>
                <div className="logs-container">
                    {trainingLogs.map((log, index) => (
                        <div key={index} className={`log-line ${log.includes('ERROR') ? 'error' : log.includes('INFO') ? 'info' : ''}`}>
                            {log}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
