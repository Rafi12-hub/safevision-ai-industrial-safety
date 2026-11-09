import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { LiveMonitoringPage } from './pages/LiveMonitoringPage';
import { RecordsPage } from './pages/RecordsPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { ModelSettingsPage } from './pages/ModelSettingsPage';
import { SystemSettingsPage } from './pages/SystemSettingsPage';
import { api } from './services/api';

export function App() {
  const [currentTab, setCurrentTab] = useState<string>('dashboard');
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(true);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isCustomModel, setIsCustomModel] = useState<boolean>(false);

  // Poll backend health status
  const checkHealth = async () => {
    try {
      const data = await api.getHealth();
      setIsBackendOnline(true);
      setIsCameraActive(data.camera_active ?? false);
      setIsCustomModel(data.is_custom_model ?? false);
    } catch (e) {
      setIsBackendOnline(false);
      setIsCameraActive(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const timer = setInterval(checkHealth, 4000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Top Navbar */}
      <Navbar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        isBackendOnline={isBackendOnline}
        isCameraActive={isCameraActive}
        isCustomModel={isCustomModel}
      />

      {/* Main Page Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {currentTab === 'dashboard' && (
          <DashboardPage
            onNavigateToMonitoring={() => setCurrentTab('monitoring')}
            onNavigateToRecords={() => setCurrentTab('records')}
          />
        )}

        {currentTab === 'monitoring' && <LiveMonitoringPage />}

        {currentTab === 'records' && <RecordsPage />}

        {currentTab === 'analytics' && <AnalyticsPage />}

        {currentTab === 'model' && <ModelSettingsPage />}

        {currentTab === 'settings' && <SystemSettingsPage />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-950 border-t border-slate-900 py-6 text-xs text-slate-500 text-center">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <span>AI-Based Mechanical Laboratory Safety Monitoring System</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span>YOLO11 CPU Inference</span>
            <span>•</span>
            <span>Windows 11 Laptop Webcam</span>
            <span>•</span>
            <span>FastAPI + React TypeScript</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
