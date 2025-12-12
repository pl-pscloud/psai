import { useEffect, useState } from 'react';
import { startTraining, getActivities } from '../services/api';
import {
    Activity,
    TrendingUp,
    Clock,
    CheckCircle2,
    Play,
    RefreshCw
} from 'lucide-react';
import {
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area
} from 'recharts';
import { useStore } from '../store';
import { getStatus, getScores, checkHealth } from '../services/api';
import './Dashboard.css';


export default function Dashboard() {
    const { status, setStatus, scores, setScores, isConnected, setIsConnected, activities, setActivities, config } = useStore();

    useEffect(() => {
        const checkConnection = async () => {
            try {
                await checkHealth();
                setIsConnected(true);
                const statusData = await getStatus();
                setStatus(statusData);

                if (statusData.is_initialized) {
                    const scoresData = await getScores();
                    setScores(scoresData.scores);
                }

                try {
                    const activitiesData = await getActivities();
                    if (Array.isArray(activitiesData)) {
                        setActivities(activitiesData);
                    }
                } catch (e) {
                    console.error("Failed to fetch activities", e);
                }
            } catch {
                setIsConnected(false);
            }
        };

        checkConnection();
        const interval = setInterval(checkConnection, 5000);
        return () => clearInterval(interval);
    }, [setIsConnected, setStatus, setScores]);

    const [duration, setDuration] = useState<string>('--');

    useEffect(() => {
        const updateDuration = () => {
            const detailed = status?.detailed_status;
            if (!detailed?.start_time) {
                setDuration('--');
                return;
            }

            let diff: number;
            if (status?.is_training) {
                diff = Date.now() / 1000 - detailed.start_time;
            } else if (detailed.end_time) {
                diff = detailed.end_time - detailed.start_time;
            } else {
                setDuration('--');
                return;
            }

            const h = Math.floor(diff / 3600);
            const m = Math.floor((diff % 3600) / 60);
            const s = Math.floor(diff % 60);
            setDuration(`${h > 0 ? h + 'h ' : ''}${m}m ${s}s`);
        };

        updateDuration();
        const interval = setInterval(updateDuration, 1000);
        return () => clearInterval(interval);
    }, [status]);

    const getBestModelInfo = () => {
        try {
            if (!scores || typeof scores !== 'object') return { name: '--', score: '--' };

            const entries = Object.entries(scores)
                .filter(([, s]) => s && typeof s.test_score === 'number');

            if (entries.length === 0) return { name: '--', score: '--' };

            const metric = config?.dataset?.metric || '';
            const MINIMIZATION_METRICS = ['rmse', 'mae', 'mse', 'rmsle'];
            const isMinimization = MINIMIZATION_METRICS.includes(metric.toLowerCase());

            const best = entries.reduce((a, b) => {
                const scoreA = a[1].test_score!;
                const scoreB = b[1].test_score!;

                if (isMinimization) {
                    return scoreA < scoreB ? a : b;
                } else {
                    return scoreA > scoreB ? a : b;
                }
            });

            // Format name
            const name = best[0].replace('final_model_', '').replace('ensemble_', '').replace(/_/g, ' ');
            const score = best[1].test_score?.toFixed(5);

            return {
                name: name.charAt(0).toUpperCase() + name.slice(1),
                score
            };
        } catch (e) {
            console.error("Error calculating best model:", e);
            return { name: '--', score: '--' };
        }
    };

    const getTrainedModelsCount = () => {
        return Object.values(scores).filter(s => s.test_score !== undefined).length;
    };

    const bestModel = getBestModelInfo();
    const getChartData = () => {
        // Defensive checks
        if (!status || !status.detailed_status || !status.detailed_status.model_progress) {
            return [];
        }

        try {
            const history: Record<number, any> = {};
            // We can safely cast/access here because of the check above
            // Use 'as any' or safer access to allow TS to be happy if types are loose
            const progress = status.detailed_status.model_progress as Record<string, any>;

            // Safe mapping
            const allTrials = Object.values(progress).map((m: any) => m?.trial_history?.length || 0);
            const maxTrials = allTrials.length > 0 ? Math.max(...allTrials, 0) : 0;

            if (maxTrials === 0) return [];

            Object.entries(progress).forEach(([model, modelStatus]) => {
                const ms = modelStatus as any;
                if (ms?.trial_history && Array.isArray(ms.trial_history)) {
                    ms.trial_history.forEach((item: any) => {
                        if (item && typeof item.trial === 'number' && typeof item.score === 'number') {
                            if (!history[item.trial]) {
                                history[item.trial] = { trial: item.trial };
                            }
                            history[item.trial][model] = item.score;
                        }
                    });
                }
            });

            return Object.values(history).sort((a, b) => a.trial - b.trial);
        } catch (e) {
            console.error("Error processing chart data:", e);
            return [];
        }
    };

    const chartData = getChartData();

    const MODEL_CONFIGS = [
        { key: 'lightgbm', label: 'LightGBM', color: '#a855f7' },
        { key: 'xgboost', label: 'XGBoost', color: '#22d3ee' },
        { key: 'catboost', label: 'CatBoost', color: '#f97316' },
        { key: 'random_forest', label: 'Random Forest', color: '#10b981' },
        { key: 'pytorch', label: 'PyTorch', color: '#ef4444' },
    ];

    return (
        <div className="dashboard">
            <div className="page-header">
                <h1 className="page-title">Dashboard</h1>
                <p className="page-subtitle">Overview of your ML pipeline status</p>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'rgba(168, 85, 247, 0.15)' }}>
                        <Activity size={24} color="#a855f7" />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{status.is_training ? 'Training' : status.is_initialized ? 'Ready' : 'Idle'}</span>
                        <span className="stat-label">Pipeline Status</span>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'rgba(34, 211, 238, 0.15)' }}>
                        <TrendingUp size={24} color="#22d3ee" />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{bestModel.score}</span>
                        <span className="stat-label">Best: {bestModel.name}</span>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'rgba(16, 185, 129, 0.15)' }}>
                        <CheckCircle2 size={24} color="#10b981" />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{getTrainedModelsCount()}</span>
                        <span className="stat-label">Trained Models</span>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon" style={{ background: 'rgba(249, 115, 22, 0.15)' }}>
                        <Clock size={24} color="#f97316" />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{duration}</span>
                        <span className="stat-label">Training Time</span>
                    </div>
                </div>
            </div>

            {/* Charts Section */}
            <div className="charts-section">
                <div className="main-column">
                    <div className="card chart-card">
                        <div className="card-header">
                            <h3 className="card-title">Model Performance Over Trials</h3>
                            <button className="btn btn-secondary btn-icon" onClick={() => window.location.reload()}>
                                <RefreshCw size={16} />
                            </button>
                        </div>
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={chartData}>
                                    <defs>
                                        {MODEL_CONFIGS.map(config => (
                                            <linearGradient key={config.key} id={`gradient-${config.key}`} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={config.color} stopOpacity={0.3} />
                                                <stop offset="95%" stopColor={config.color} stopOpacity={0} />
                                            </linearGradient>
                                        ))}
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                                    <XAxis dataKey="trial" stroke="#6b7280" fontSize={12} />
                                    <YAxis stroke="#6b7280" fontSize={12} />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#1a1a24',
                                            border: '1px solid #2a2a3a',
                                            borderRadius: '8px'
                                        }}
                                    />
                                    {MODEL_CONFIGS.map(config => (
                                        <Area
                                            key={config.key}
                                            connectNulls
                                            type="monotone"
                                            dataKey={config.key}
                                            stroke={config.color}
                                            fill={`url(#gradient-${config.key})`}
                                            strokeWidth={2}
                                        />
                                    ))}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="chart-legend">
                            {MODEL_CONFIGS.map(config => (
                                <div key={config.key} className="legend-item">
                                    <span className="legend-color" style={{ background: config.color }}></span>
                                    <span>{config.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Model Scores */}
                    {Object.keys(scores).length > 0 && (
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">Model Scores</h3>
                            </div>
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Model</th>
                                        <th>CV Score</th>
                                        <th>Test Score</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(scores).map(([name, score]) => (
                                        <tr key={name}>
                                            <td>{name.replace('final_model_', '').replace('_', ' ')}</td>
                                            <td>{typeof score.cv_score === 'number' ? score.cv_score.toFixed(5) : (score.cv_score ?? '--')}</td>
                                            <td>{typeof score.test_score === 'number' ? score.test_score.toFixed(5) : (score.test_score ?? '--')}</td>
                                            <td>
                                                <span className="badge badge-success">Trained</span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>

                <div className="card quick-actions-card">
                    <div className="card-header">
                        <h3 className="card-title">Quick Actions</h3>
                    </div>
                    <div className="quick-actions">
                        <button className="btn btn-primary w-full" disabled={!isConnected}
                            onClick={async () => {
                                try {
                                    await startTraining();
                                    // Activity logged by backend
                                } catch (e) {
                                    console.error(e);
                                }
                            }}
                        >
                            <Play size={18} />
                            Start Training
                        </button>
                        <button className="btn btn-secondary w-full" disabled={!isConnected}>
                            <RefreshCw size={18} />
                            Refresh Status
                        </button>
                    </div>

                    <div className="recent-activity">
                        <h4 className="activity-title">Recent Activity</h4>
                        <div className="activity-list">
                            {(!activities || !Array.isArray(activities) || activities.length === 0) ? (
                                <p className="text-secondary text-sm p-4 text-center">No recent activity</p>
                            ) : (
                                activities.map((activity) => (
                                    <div key={activity.id} className="activity-item">
                                        <div className="activity-icon-wrapper">
                                            {activity.type === 'success' ? (
                                                <CheckCircle2 size={16} color="#10b981" />
                                            ) : activity.type === 'info' ? (
                                                <Activity size={16} color="#22d3ee" />
                                            ) : (
                                                <Activity size={16} color="#a855f7" />
                                            )}
                                        </div>
                                        <div className="activity-details">
                                            <span className="activity-message">{activity.message}</span>
                                            <span className="activity-time">
                                                {new Date(activity.timestamp).toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
