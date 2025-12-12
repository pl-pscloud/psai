import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import ErrorBoundary from './components/ErrorBoundary';
import Dashboard from './pages/Dashboard';
import Configuration from './pages/Configuration';
import Training from './pages/Training';
import Models from './pages/Models';
import AIAgent from './pages/AIAgent';
import Explainability from './pages/Explainability';
import Experiments from './pages/Experiments';
import Predictions from './pages/Predictions';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
          <Route path="configuration" element={<ErrorBoundary><Configuration /></ErrorBoundary>} />
          <Route path="training" element={<ErrorBoundary><Training /></ErrorBoundary>} />
          <Route path="models" element={<ErrorBoundary><Models /></ErrorBoundary>} />
          <Route path="agent" element={<ErrorBoundary><AIAgent /></ErrorBoundary>} />
          <Route path="explainability" element={<ErrorBoundary><Explainability /></ErrorBoundary>} />
          <Route path="experiments" element={<ErrorBoundary><Experiments /></ErrorBoundary>} />
          <Route path="predictions" element={<ErrorBoundary><Predictions /></ErrorBoundary>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
