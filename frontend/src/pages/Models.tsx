import { useEffect, useState } from 'react';
import {
    Trophy,
    TrendingUp,
    RefreshCw
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
    Legend
} from 'recharts';
import { useStore } from '../store';
import { getScores } from '../services/api';
import './Models.css';

const modelColors: Record<string, string> = {
    lightgbm: '#a855f7',
    xgboost: '#22d3ee',
    catboost: '#f97316',
    random_forest: '#10b981',
    pytorch: '#ec4899',
    stacking: '#eab308',
    voting: '#3b82f6',
};

export default function Models() {
    const { scores, setScores, status, config } = useStore();
    const [isLoading, setIsLoading] = useState(false);

    const refreshScores = async () => {
        if (!status.is_initialized) return;
        setIsLoading(true);
        try {
            const data = await getScores();
            setScores(data.scores);
        } catch (error) {
            console.error('Failed to fetch scores:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        refreshScores();
    }, [status.is_initialized]);

    // Determine optimization direction
    const MINIMIZATION_METRICS = ['rmse', 'mae', 'mse', 'rmsle'];
    const isMinimization = MINIMIZATION_METRICS.includes(config.dataset.metric.toLowerCase());

    // Transform scores for bar chart
    const processedModels = Object.entries(scores)
        .map(([key, value]) => {
            let modelName = key;
            // Handle prefixes if they exist (though psml returns clean base names)
            modelName = modelName.replace('final_model_', '').replace('ensemble_', '');

            // Map color based on model type
            let colorKey = modelName;
            if (modelName.includes('stacking')) colorKey = 'stacking';
            if (modelName.includes('voting')) colorKey = 'voting';

            return {
                model: modelName.replace('_', ' ').toUpperCase(),
                test_score: typeof value.test_score === 'number' ? value.test_score : 0,
                cv_score: typeof value.cv_score === 'number' ? value.cv_score : 0,
                color: modelColors[colorKey] || modelColors[modelName] || '#a855f7',
                originalKey: key
            };
        })
        // Sort based on metric: Best to Worst
        .sort((a, b) => {
            // Determine effective score for sorting (prefer test, fallback to cv)
            const getScore = (item: typeof a) => {
                if (item.test_score !== 0) return item.test_score;
                return item.cv_score;
            };

            const scoreA = getScore(a);
            const scoreB = getScore(b);

            // If both effectively 0, keep stable
            if (scoreA === 0 && scoreB === 0) return 0;
            if (scoreA === 0) return 1; // Put missing scores at bottom
            if (scoreB === 0) return -1;

            if (isMinimization) {
                // Lower is better (Ascending)
                return scoreA - scoreB;
            } else {
                // Higher is better (Descending)
                return scoreB - scoreA;
            }
        });

    // Filter for chart (only show models with scores)
    const barChartData = processedModels.filter(item => item.test_score > 0 || item.cv_score > 0);

    // Find best model (First item in sorted list if it exists and has score)
    const bestModel = barChartData.length > 0 ? barChartData[0] : null;

    // Build radar data for model comparison (normalized)
    // For radar, we might want to normalize so best is outer edge
    // But simplistic approach: just plot values if they are comparable (e.g. 0-1 accuracy)
    // For RMSE, Radar chart is weird (Lower is better). 
    // Maybe skip radar for regression or invert? Keeping as is for now but using sorted data.
    const radarData = barChartData.map(item => ({
        model: item.model,
        [item.model]: item.test_score * (isMinimization ? 1 : 100), // Scale up for acc, keep raw for rmse?
        fullMark: 100,
    }));

    return (
        <div className="models-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Models</h1>
                    <p className="page-subtitle">Compare and analyze trained models</p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={refreshScores}
                    disabled={isLoading || !status.is_initialized}
                >
                    <RefreshCw size={18} className={isLoading ? 'animate-spin' : ''} />
                    Refresh
                </button>
            </div>

            {Object.keys(scores).length === 0 ? (
                <div className="empty-state">
                    <TrendingUp size={48} />
                    <h3>No Models Trained Yet</h3>
                    <p>Train some models first to see comparisons here.</p>
                </div>
            ) : (
                <>
                    {/* Best Model Card */}
                    {bestModel && (
                        <div className="best-model-card">
                            <div className="best-model-icon">
                                <Trophy size={32} />
                            </div>
                            <div className="best-model-info">
                                <span className="best-model-label">Best Model ({config.dataset.metric.toUpperCase()})</span>
                                <span className="best-model-name">{bestModel.model.toUpperCase()}</span>
                                <div className="best-model-scores">
                                    <span className="score-item">
                                        <span className="score-label">Test Score:</span>
                                        <span className="score-value">{bestModel.test_score > 0 ? bestModel.test_score.toFixed(5) : 'N/A'}</span>
                                    </span>
                                    <span className="score-divider">|</span>
                                    <span className="score-item">
                                        <span className="score-label">CV Score:</span>
                                        <span className="score-value">{bestModel.cv_score > 0 ? bestModel.cv_score.toFixed(5) : 'N/A'}</span>
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Charts */}
                    <div className="charts-grid">
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">Model Scores Comparison</h3>
                            </div>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={barChartData} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                                    <XAxis type="number" stroke="#6b7280" fontSize={12} domain={[0, 'auto']} />
                                    <YAxis dataKey="model" type="category" stroke="#6b7280" fontSize={12} width={100} reversed />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#1a1a24',
                                            border: '1px solid #2a2a3a',
                                            borderRadius: '8px'
                                        }}
                                        formatter={(value: number) => value.toFixed(5)}
                                    />
                                    <Legend />
                                    <Bar dataKey="cv_score" name={`CV ${config.dataset.metric.toUpperCase()}`} fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                                    <Bar dataKey="test_score" name={`Test ${config.dataset.metric.toUpperCase()}`} fill="#10b981" radius={[0, 4, 4, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">Performance Overview</h3>
                            </div>
                            <ResponsiveContainer width="100%" height={300}>
                                <RadarChart data={radarData}>
                                    <PolarGrid stroke="#2a2a3a" />
                                    <PolarAngleAxis dataKey="model" stroke="#6b7280" fontSize={11} />
                                    <PolarRadiusAxis stroke="#6b7280" fontSize={10} />
                                    {barChartData.map((item) => (
                                        <Radar
                                            key={item.model}
                                            name={item.model}
                                            dataKey={item.model}
                                            stroke={item.color}
                                            fill={item.color}
                                            fillOpacity={0.3}
                                        />
                                    ))}
                                    <Legend />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Scores Table */}
                    <div className="card mt-4">
                        <div className="card-header">
                            <h3 className="card-title">Detailed Scores</h3>
                        </div>
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>CV Score ({config.dataset.metric.toUpperCase()})</th>
                                    <th>Test Score ({config.dataset.metric.toUpperCase()})</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {processedModels.map((item) => {
                                    const isBest = bestModel && item.model === bestModel.model;
                                    return (
                                        <tr key={item.originalKey} className={isBest ? 'best-row' : ''}>
                                            <td>
                                                <div className="model-cell">
                                                    <span
                                                        className="model-dot"
                                                        style={{ background: item.color }}
                                                    />
                                                    {item.model}
                                                    {isBest && <Trophy size={14} className="trophy-icon" />}
                                                </div>
                                            </td>
                                            <td>{item.cv_score > 0 ? item.cv_score.toFixed(5) : '--'}</td>
                                            <td>{item.test_score > 0 ? item.test_score.toFixed(5) : '--'}</td>
                                            <td><span className="badge badge-success">Trained</span></td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
        </div>
    );
}
