import { useState, useEffect } from 'react';
import {
    FileText,
    Save,
    Upload,
    AlertCircle
} from 'lucide-react';
import { useStore } from '../store';
import { saveExperiment, getExperiments, loadExperiment } from '../services/api';
import './Experiments.css';

export default function Experiments() {
    const { config, setConfig } = useStore();
    const [experiments, setExperiments] = useState<any[]>([]);
    const [expName, setExpName] = useState('');
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        fetchExperiments();
    }, []);

    const fetchExperiments = async () => {
        try {
            const data = await getExperiments();
            setExperiments(data.experiments);
        } catch (e) {
            console.error(e);
        }
    };

    const handleSaveExperiment = async () => {
        if (!expName.trim()) {
            setMessage({ type: 'error', text: 'Please enter an experiment name' });
            return;
        }

        setIsLoading(true);
        try {
            await saveExperiment(expName, config);
            setMessage({ type: 'success', text: `Experiment '${expName}' saved successfully` });
            setExpName('');
            fetchExperiments();
        } catch (e) {
            setMessage({ type: 'error', text: 'Failed to save experiment' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleLoadExperiment = async (name: string) => {
        setIsLoading(true);
        try {
            const data = await loadExperiment(name);
            setConfig(data.config);
            // Optionally refresh status to check if model loaded correctly on backend
            setMessage({ type: 'success', text: `Experiment '${name}' loaded. Model active: ${data.has_model}` });
        } catch (e) {
            setMessage({ type: 'error', text: 'Failed to load experiment' });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="experiments">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Experiments</h1>
                    <p className="page-subtitle">Manage your experiments and model versions</p>
                </div>
            </div>

            {message && (
                <div className={`message ${message.type}`}>
                    <AlertCircle size={18} />
                    {message.text}
                </div>
            )}

            <div className="experiments-grid">
                {/* Save Experiment */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Save size={18} />
                            Save Current Experiment
                        </h3>
                    </div>
                    <div className="p-4">
                        <div className="form-group">
                            <label className="form-label">Experiment Name</label>
                            <div className="input-with-button">
                                <input
                                    type="text"
                                    className="input"
                                    value={expName}
                                    onChange={(e) => setExpName(e.target.value)}
                                    placeholder="e.g., v1-baseline-xgboost"
                                    disabled={isLoading}
                                />
                                <button
                                    className="btn btn-primary"
                                    onClick={handleSaveExperiment}
                                    disabled={isLoading}
                                >
                                    <Save size={16} />
                                    {isLoading ? 'Saving...' : 'Save'}
                                </button>
                            </div>
                            <p className="text-sm text-secondary mt-2">
                                Saves current configuration, feature engineering code, and trained model (if available).
                            </p>
                        </div>
                    </div>
                </div>

                {/* Experiment List */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <FileText size={18} />
                            Saved Experiments
                        </h3>
                    </div>
                    <div className="table-responsive">
                        {experiments.length === 0 ? (
                            <p className="text-secondary p-4">No saved experiments found.</p>
                        ) : (
                            <table className="table">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Created At</th>
                                        <th>Files</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {experiments.map((exp) => (
                                        <tr key={exp.name}>
                                            <td className="font-medium">{exp.name}</td>
                                            <td>{new Date(exp.created_at).toLocaleString()}</td>
                                            <td className="text-sm text-secondary">
                                                {exp.files.filter((f: string) => f !== 'feature_transformer.py').join(', ')}
                                            </td>
                                            <td>
                                                <button
                                                    className="btn btn-secondary btn-sm"
                                                    onClick={() => handleLoadExperiment(exp.name)}
                                                    disabled={isLoading}
                                                >
                                                    <Upload size={14} />
                                                    Load
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
