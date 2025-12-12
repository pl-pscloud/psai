import {
    Bot,
    Play,
    Settings2,
    CheckCircle2,
    Circle,
    Loader2,
    MessageSquare
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { initAgent, runAgentStep } from '../services/api';
import { useAgentStore } from '../store/agentStore';
import type { AgentStep } from '../store/agentStore';
import './AIAgent.css';

export default function AIAgent() {
    const {
        steps,
        setSteps,
        isInitialized,
        setIsInitialized,
        isRunning,
        setIsRunning,
        messages,
        addMessage,
        agentConfig,
        setAgentConfig
    } = useAgentStore();

    const handleInitialize = async () => {
        setIsRunning(true);
        addMessage('user', 'Initialize Agent');

        try {
            const response = await initAgent(agentConfig);
            setIsInitialized(true);
            addMessage('agent', response.message);
        } catch (error) {
            addMessage('agent', `Error initializing agent: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsRunning(false);
        }
    };

    const handleRunStep = async (stepId: string) => {
        setIsRunning(true);

        // Update step status to running
        setSteps(prev => prev.map(s => s.id === stepId ? { ...s, status: 'running' } : s));
        addMessage('user', `Run step: ${stepId}`);

        try {
            const response = await runAgentStep(stepId);

            // Format output nicely
            let outputMsg = response.llm_output;
            if (response.config) {
                outputMsg += "\n\nGenerated Config:\n```json\n" + JSON.stringify(response.config, null, 2) + "\n```";
            }

            addMessage('agent', outputMsg);

            // Update step status to completed
            setSteps(prev => prev.map(s => s.id === stepId ? { ...s, status: 'completed' } : s));

        } catch (error) {
            const errorMsg = error instanceof Error ? error.message : String(error);
            addMessage('agent', `Error running step ${stepId}: ${errorMsg}`);
            // Update step status to failed
            setSteps(prev => prev.map(s => s.id === stepId ? { ...s, status: 'failed' } : s));
        } finally {
            setIsRunning(false);
        }
    };

    const getStepIcon = (status: AgentStep['status']) => {
        switch (status) {
            case 'completed':
                return <CheckCircle2 size={20} className="step-icon completed" />;
            case 'running':
                return <Loader2 size={20} className="step-icon running animate-spin" />;
            case 'failed':
                return <Circle size={20} className="step-icon failed" />;
            default:
                return <Circle size={20} className="step-icon pending" />;
        }
    };

    return (
        <div className="ai-agent">
            <div className="page-header">
                <div>
                    <h1 className="page-title">AI Data Scientist</h1>
                    <p className="page-subtitle">Granular control over the AI Data Science pipeline</p>
                </div>
                <div className="header-actions">
                    {!isInitialized ? (
                        <button
                            className="btn btn-primary"
                            onClick={handleInitialize}
                            disabled={isRunning}
                        >
                            {isRunning ? <Loader2 size={18} className="animate-spin" /> : <Settings2 size={18} />}
                            {isRunning ? 'Initializing...' : 'Initialize Agent'}
                        </button>
                    ) : (
                        <div className="status-badge success">
                            <CheckCircle2 size={16} />
                            Agent Ready
                        </div>
                    )}
                </div>
            </div>

            <div className="agent-layout">
                {/* Config Panel */}
                <div className="card agent-config">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Settings2 size={18} />
                            Configuration
                        </h3>
                    </div>

                    <div className="config-form">
                        <div className="form-group">
                            <label className="form-label">Target Variable</label>
                            <input
                                type="text"
                                className="input"
                                value={agentConfig.target || ''}
                                onChange={(e) => setAgentConfig({ ...agentConfig, target: e.target.value })}
                                placeholder="Target column name..."
                                disabled={isInitialized || isRunning}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">LLM Provider</label>
                            <select
                                className="input select"
                                value={agentConfig.provider}
                                onChange={(e) => setAgentConfig({ ...agentConfig, provider: e.target.value })}
                                disabled={isInitialized || isRunning}
                            >
                                <option value="google">Google (Gemini)</option>
                                <option value="openai">OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                                <option value="ollama">Ollama (Local)</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">API Key (Optional)</label>
                            <input
                                type="password"
                                className="input"
                                value={agentConfig.api_key || ''}
                                onChange={(e) => setAgentConfig({ ...agentConfig, api_key: e.target.value })}
                                placeholder="Overwrite environmental variable..."
                                disabled={isInitialized || isRunning}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Model</label>
                            <select
                                className="input select"
                                value={agentConfig.model}
                                onChange={(e) => setAgentConfig({ ...agentConfig, model: e.target.value })}
                                disabled={isInitialized || isRunning}
                            >
                                <option value="gemini-2.0-flash">Gemini 2.0 Flash</option>
                                <option value="gemini-2.0-flash-thinking">Gemini Flash Thinking</option>
                                <option value="gemini-3-pro-preview">Gemini 3.0 Pro Preview</option>
                                <option value="gpt-4o">GPT-4o</option>
                                <option value="claude-3.5-sonnet">Claude 3.5 Sonnet</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Temperature</label>
                            <input
                                type="number"
                                className="input"
                                value={agentConfig.temperature}
                                onChange={(e) => setAgentConfig({ ...agentConfig, temperature: parseFloat(e.target.value) })}
                                min="0"
                                max="2"
                                step="0.1"
                                disabled={isInitialized || isRunning}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Task Type</label>
                            <select
                                className="input select"
                                value={agentConfig.task_type}
                                onChange={(e) => setAgentConfig({ ...agentConfig, task_type: e.target.value })}
                                disabled={isInitialized || isRunning}
                            >
                                <option value="regression">Regression</option>
                                <option value="classification">Classification</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Optuna Trials</label>
                            <input
                                type="number"
                                className="input"
                                value={agentConfig.optuna_trials}
                                onChange={(e) => setAgentConfig({ ...agentConfig, optuna_trials: parseInt(e.target.value) })}
                                min="1"
                                disabled={isInitialized || isRunning}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Optuna Metric</label>
                            <select
                                className="input select"
                                value={agentConfig.optuna_metric}
                                onChange={(e) => setAgentConfig({ ...agentConfig, optuna_metric: e.target.value })}
                                disabled={isInitialized || isRunning}
                            >
                                <option value="rmse">RMSE</option>
                                <option value="mae">MAE</option>
                                <option value="auc">AUC</option>
                                <option value="accuracy">Accuracy</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Pipeline Steps */}
                <div className="card agent-steps">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Bot size={18} />
                            Pipeline Steps
                        </h3>
                    </div>

                    <div className="steps-list">
                        {steps.map((step, index) => (
                            <div key={step.id} className={`step-item ${step.status}`}>
                                <div className="step-connector">
                                    {getStepIcon(step.status)}
                                    {index < steps.length - 1 && <div className="connector-line" />}
                                </div>
                                <div className="step-content">
                                    <div className="step-header">
                                        <span className="step-name">{step.name}</span>
                                        {isInitialized && (
                                            <button
                                                className="btn btn-sm btn-secondary run-step-btn"
                                                onClick={() => handleRunStep(step.id)}
                                                disabled={isRunning}
                                            >
                                                {step.status === 'running' ? 'Running...' : (step.id === 'training' ? 'Start Training' : `Consult ${step.name.split(' ')[0]}`)}
                                                {step.status !== 'running' && <Play size={12} style={{ marginLeft: 4 }} />}
                                            </button>
                                        )}
                                    </div>
                                    <span className="step-description">{step.description}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Messages */}
                <div className="card agent-messages">
                    <div className="card-header">
                        <h3 className="card-title">
                            <MessageSquare size={18} />
                            Agent Output
                        </h3>
                    </div>

                    <div className="messages-container">
                        {messages.map((msg, index) => (
                            <div key={index} className={`message ${msg.role}`}>
                                <div className="message-icon">
                                    {msg.role === 'agent' ? <Bot size={16} /> : <MessageSquare size={16} />}
                                </div>
                                <div className="message-content">
                                    <div className="markdown-content">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {msg.content}
                                        </ReactMarkdown>
                                    </div>
                                    {msg.image && (
                                        <div className="message-image">
                                            <img src={`data:image/png;base64,${msg.image}`} alt="SHAP Plot" />
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
