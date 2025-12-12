import { useState, useRef } from 'react';
import { AlertCircle, Zap, FileText, Upload, Download } from 'lucide-react';
import { useStore } from '../store';
import { predict, predictBatch } from '../services/api';
import './Predictions.css';

interface BatchResult {
    id: string | number;
    prediction: number | number[];
}

export default function Predictions() {
    const { scores } = useStore();
    const [activeTab, setActiveTab] = useState<'json' | 'csv'>('json');

    // Single Prediction State
    const [predictionInput, setPredictionInput] = useState('');
    const [singlePredictions, setSinglePredictions] = useState<number[] | number[][] | null>(null);

    // Batch Prediction State
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [batchResults, setBatchResults] = useState<BatchResult[] | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [selectedModel, setSelectedModel] = useState('');
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const trainedModels = Object.keys(scores);

    const handleSinglePredict = async () => {
        if (!selectedModel || !predictionInput) return;
        setIsLoading(true);
        setMessage(null);
        try {
            const data = JSON.parse(predictionInput);
            const result = await predict(selectedModel, Array.isArray(data) ? data : [data]);
            setSinglePredictions(result.predictions);
            setMessage({ type: 'success', text: 'Predictions generated successfully' });
        } catch (e) {
            setMessage({ type: 'error', text: 'Failed to make predictions. Check JSON format.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
            setBatchResults(null);
            setMessage(null);
        }
    };

    const handleBatchPredict = async () => {
        if (!selectedModel || !selectedFile) return;
        setIsLoading(true);
        setMessage(null);
        try {
            const result = await predictBatch(selectedModel, selectedFile);
            setBatchResults(result.predictions);
            setMessage({ type: 'success', text: `Generated predictions for ${result.predictions.length} rows.` });
        } catch (e) {
            setMessage({ type: 'error', text: 'Batch prediction failed. Check file format.' });
        } finally {
            setIsLoading(false);
        }
    };

    const downloadCSV = () => {
        if (!batchResults) return;

        const headers = ['id', 'prediction'];
        const csvContent = [
            headers.join(','),
            ...batchResults.map(row => `${row.id},${Array.isArray(row.prediction) ? row.prediction.join(';') : row.prediction}`)
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `predictions_${selectedModel}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="predictions-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Make Predictions</h1>
                    <p className="page-subtitle">Generate predictions using trained models via JSON or CSV batch upload</p>
                </div>
            </div>

            {message && (
                <div className={`message ${message.type}`}>
                    <AlertCircle size={18} />
                    {message.text}
                </div>
            )}

            <div className="card prediction-card-full">
                <div className="card-header">
                    <h3 className="card-title">
                        <Zap size={18} />
                        Inference
                    </h3>
                </div>

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
                                {model.replace(/_/g, ' ').toUpperCase()}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="tabs">
                    <button
                        className={`tab ${activeTab === 'json' ? 'active' : ''}`}
                        onClick={() => setActiveTab('json')}
                    >
                        <FileText size={16} /> JSON Input
                    </button>
                    <button
                        className={`tab ${activeTab === 'csv' ? 'active' : ''}`}
                        onClick={() => setActiveTab('csv')}
                    >
                        <Upload size={16} /> Batch CSV
                    </button>
                </div>

                {activeTab === 'json' ? (
                    <div className="tab-content">
                        <div className="form-group">
                            <label className="form-label">Input Data (JSON format)</label>
                            <textarea
                                className="input code-input"
                                value={predictionInput}
                                onChange={(e) => setPredictionInput(e.target.value)}
                                placeholder='[{"feature1": 1.0, "feature2": "value"}]'
                                rows={10}
                            />
                        </div>
                        <button
                            className="btn btn-success w-full"
                            onClick={handleSinglePredict}
                            disabled={!selectedModel || !predictionInput || isLoading}
                        >
                            {isLoading ? 'Processing...' : 'Generate Predictions'}
                        </button>

                        {singlePredictions && (
                            <div className="predictions-output">
                                <h4>Predictions:</h4>
                                <pre>{JSON.stringify(singlePredictions, null, 2)}</pre>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="tab-content">
                        <div className="file-upload-area">
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                ref={fileInputRef}
                                style={{ display: 'none' }}
                            />
                            <div className="upload-placeholder" onClick={() => fileInputRef.current?.click()}>
                                <Upload size={48} className="upload-icon" />
                                <p>{selectedFile ? selectedFile.name : "Click to upload CSV file"}</p>
                                <span className="sub-text">File must contain headers matching training features (and 'id' column if applicable)</span>
                            </div>
                        </div>

                        <button
                            className="btn btn-success w-full mt-4"
                            onClick={handleBatchPredict}
                            disabled={!selectedModel || !selectedFile || isLoading}
                        >
                            {isLoading ? 'Processing Batch...' : 'Generate Batch Predictions'}
                        </button>

                        {batchResults && (
                            <div className="batch-results-container">
                                <div className="results-header">
                                    <h4>Results Preview ({batchResults.length} records)</h4>
                                    <button className="btn btn-primary btn-sm" onClick={downloadCSV}>
                                        <Download size={14} /> Download CSV
                                    </button>
                                </div>
                                <div className="table-responsive">
                                    <table className="results-table">
                                        <thead>
                                            <tr>
                                                <th>ID</th>
                                                <th>Prediction</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {batchResults.slice(0, 10).map((row, idx) => (
                                                <tr key={idx}>
                                                    <td>{row.id}</td>
                                                    <td>{JSON.stringify(row.prediction)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                    {batchResults.length > 10 && (
                                        <p className="more-rows">...and {batchResults.length - 10} more rows</p>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
