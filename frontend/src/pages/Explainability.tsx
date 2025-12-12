import { useState } from 'react';
import {
    Lightbulb,
    RefreshCw,
    AlertCircle
} from 'lucide-react';
import { useStore } from '../store';
import { getExplanation } from '../services/api';
import './Explainability.css';

export default function Explainability() {
    const { scores, status } = useStore();
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [shapImage, setShapImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const trainedModels = Object.keys(scores)
        .filter(k => ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'pytorch'].includes(k) && scores[k].test_score !== 'N/A');

    const handleExplain = async () => {
        if (!selectedModel) return;

        setIsLoading(true);
        setError(null);
        setShapImage(null);

        try {
            const result = await getExplanation(selectedModel);
            setShapImage(result.image_base64);
        } catch (err) {
            setError('Failed to generate explanation. Ensure the model supports SHAP.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="explainability">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Model Explainability</h1>
                    <p className="page-subtitle">Understand your model's predictions with SHAP</p>
                </div>
            </div>

            {!status.is_initialized ? (
                <div className="card notice-card">
                    <AlertCircle size={24} />
                    <p>Initialize the pipeline and train models first to view explanations.</p>
                </div>
            ) : trainedModels.length === 0 ? (
                <div className="card notice-card">
                    <AlertCircle size={24} />
                    <p>No trained models available. Train some models first.</p>
                </div>
            ) : (
                <>
                    <div className="explain-controls">
                        <div className="form-group">
                            <label className="form-label">Select Model</label>
                            <select
                                className="input select"
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                            >
                                <option value="">Choose a model...</option>
                                {trainedModels.map(model => (
                                    <option key={model} value={model}>
                                        {model.replace('_', ' ').toUpperCase()}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <button
                            className="btn btn-primary"
                            onClick={handleExplain}
                            disabled={!selectedModel || isLoading}
                        >
                            {isLoading ? <RefreshCw size={18} className="animate-spin" /> : <Lightbulb size={18} />}
                            Generate Explanation
                        </button>
                    </div>

                    {error && (
                        <div className="error-message">
                            <AlertCircle size={18} />
                            {error}
                        </div>
                    )}

                    {shapImage && (
                        <div className="card shap-card animate-fade-in">
                            <div className="card-header">
                                <h3 className="card-title">
                                    SHAP Summary Plot - {selectedModel.replace('_', ' ').toUpperCase()}
                                </h3>
                            </div>
                            <div className="shap-image-container">
                                <img
                                    src={`data:image/png;base64,${shapImage}`}
                                    alt="SHAP Summary Plot"
                                    className="shap-image"
                                />
                            </div>
                            <div className="shap-legend">
                                <h4>How to Read This Plot:</h4>
                                <ul>
                                    <li><strong>Y-axis:</strong> Features ranked by importance (most important at top)</li>
                                    <li><strong>X-axis:</strong> SHAP value (impact on model output)</li>
                                    <li><strong>Color:</strong> Feature value (red = high, blue = low)</li>
                                    <li><strong>Spread:</strong> Shows the distribution of impact across samples</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {!shapImage && !error && !isLoading && selectedModel && (
                        <div className="placeholder-card">
                            <Lightbulb size={48} />
                            <p>Click "Generate Explanation" to view SHAP analysis</p>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
