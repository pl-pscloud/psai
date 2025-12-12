import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    Settings,
    Rocket,
    BarChart3,
    Bot,
    Lightbulb,
    FileText,
    Circle,
    Zap,
    RefreshCw
} from 'lucide-react';
import { resetApi } from '../../services/api';
import { useStore } from '../../store';
import { useAgentStore } from '../../store/agentStore';
import './Sidebar.css';

const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/configuration', icon: Settings, label: 'Configuration' },
    { path: '/training', icon: Rocket, label: 'Training' },
    { path: '/models', icon: BarChart3, label: 'Models' },
    { path: '/explainability', icon: Lightbulb, label: 'Explainability' },
    { path: '/experiments', icon: FileText, label: 'Experiments' },
    { path: '/predictions', icon: Zap, label: 'Predictions' },
    { path: '/agent', icon: Bot, label: 'AI Agent' },
];

const modelIndicators = [
    { name: 'LightGBM', key: 'lightgbm', color: '#a855f7' },
    { name: 'XGBoost', key: 'xgboost', color: '#22d3ee' },
    { name: 'CatBoost', key: 'catboost', color: '#f97316' },
    { name: 'Random Forest', key: 'random_forest', color: '#10b981' },
    { name: 'PyTorch', key: 'pytorch', color: '#ec4899' },
];

export default function Sidebar() {
    const { config, status, isConnected } = useStore();
    const { resetSession } = useAgentStore();

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <Zap className="logo-icon" />
                    <span className="logo-text">PSAI</span>
                </div>
                <div className={`status-indicator ${isConnected ? 'connected' : ''}`}>
                    <Circle size={8} fill="currentColor" />
                    <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
            </div>

            <nav className="sidebar-nav">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) => `nav-item ${isActive ? 'active' : ''} ${item.label === 'AI Agent' ? 'nav-item-ai' : ''}`}
                    >
                        <item.icon size={20} />
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="sidebar-section">
                <h4 className="section-title">Models</h4>
                <div className="model-list">
                    {modelIndicators.map((model) => {
                        const modelConfig = config.models[model.key as keyof typeof config.models];
                        const isEnabled = modelConfig?.enabled ?? false;
                        return (
                            <div key={model.key} className="model-indicator">
                                <Circle
                                    size={10}
                                    fill={isEnabled ? model.color : 'transparent'}
                                    stroke={model.color}
                                    strokeWidth={2}
                                />
                                <span className={isEnabled ? '' : 'disabled'}>{model.name}</span>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="sidebar-footer">
                <div className="training-status">
                    {status.is_training ? (
                        <>
                            <div className="pulse-dot" />
                            <span>Training in progress...</span>
                        </>
                    ) : status.is_initialized ? (
                        <>
                            <Circle size={8} fill="#10b981" />
                            <span>Ready</span>
                        </>
                    ) : (
                        <>
                            <Circle size={8} fill="#6b7280" />
                            <span>Not initialized</span>
                        </>
                    )}
                </div>

                <button
                    className="restart-btn"
                    onClick={async () => {
                        if (window.confirm('Are you sure you want to restart the session? This will clear all data.')) {
                            try {
                                await resetApi();
                                resetSession();
                                window.location.reload();
                            } catch (e) {
                                console.error('Failed to reset API', e);
                                alert('Failed to reset session');
                            }
                        }
                    }}
                >
                    <RefreshCw size={14} />
                    <span>Restart session</span>
                </button>
            </div>
        </aside>
    );
}
