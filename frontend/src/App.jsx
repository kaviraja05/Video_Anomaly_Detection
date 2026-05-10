/**
 * Main App Component
 * Root component with sidebar navigation and page routing
 */

import React, { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from './context/ThemeContext';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import LiveAnalysisPage from './pages/LiveAnalysisPage';
import Login from './pages/Login';
import Register from './pages/Register';
import History from './pages/History';

function App() {
  const { theme, toggleTheme } = useTheme();
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));
  const [userName, setUserName] = useState(localStorage.getItem('userName') || '');
  const [userEmail, setUserEmail] = useState(localStorage.getItem('userEmail') || '');

  useEffect(() => {
    // Re-check token when route changes
    const token = localStorage.getItem('token');
    setIsAuthenticated(!!token);
  }, [currentPage]);

  const handleNavigate = (pageId) => {
    if (!isAuthenticated && ['upload', 'live-analysis', 'history', 'results'].includes(pageId)) {
      setCurrentPage('login');
    } else {
      setCurrentPage(pageId);
    }
  };

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setCurrentPage('results');
  };

  const handleLogin = (data) => {
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('userName', data.user_name || 'User');
    localStorage.setItem('userEmail', data.user_email || data.email || '');
    setUserName(data.user_name || 'User');
    setUserEmail(data.user_email || data.email || '');
    setIsAuthenticated(true);
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userName');
    localStorage.removeItem('userEmail');
    setUserName('');
    setUserEmail('');
    setIsAuthenticated(false);
    setCurrentPage('login');
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Home onNavigate={handleNavigate} />;
      case 'login':
        return <Login onLogin={handleLogin} onNavigate={handleNavigate} />;
      case 'register':
        return <Register onRegister={() => setCurrentPage('login')} onNavigate={handleNavigate} />;
      case 'upload':
        return <UploadPage onAnalysisComplete={handleAnalysisComplete} />;      
      case 'live-analysis':
        return <LiveAnalysisPage />;
      case 'results':
        return <ResultsPage analysisData={analysisResults} />;
      case 'history':
        return <History />;
      default:
        return <Home onNavigate={handleNavigate} />;
    }
  };

  return (
    <div className="flex h-screen w-full bg-theme-bg text-theme-text overflow-hidden font-sans transition-colors duration-300" style={{ backgroundImage: 'var(--app-bg-gradient)' }}>
      {/* Sidebar Navigation */}
      <Sidebar 
        currentPage={currentPage} 
        onNavigate={handleNavigate} 
        isAuthenticated={isAuthenticated}
        onLogout={handleLogout}
        userName={userName}
        userEmail={userEmail}
      />
      
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Theme Toggle Button */}
        <button 
          onClick={toggleTheme}
          className="absolute top-6 right-6 z-50 p-2.5 rounded-full bg-theme-card/90 backdrop-blur-md border border-theme-border text-theme-text hover:bg-theme-card transition-all duration-300 flex items-center justify-center shadow-[0_4px_15px_rgba(0,0,0,0.1)] hover:scale-110 hover:shadow-[0_6px_20px_rgba(0,0,0,0.15)]"
          title="Toggle Theme"
        >
          {theme === 'dark' ? <Sun size={22} className="text-amber-400" /> : <Moon size={22} className="text-indigo-600" />}
        </button>

        <div className="flex-1 overflow-y-auto p-6 md:p-8 w-full max-w-[1600px] mx-auto custom-scrollbar pt-20 md:pt-8">
          {renderPage()}
        </div>
        
        {/* Footer */}
        <footer className="w-full py-4 text-center text-xs text-slate-500 border-t border-theme-border backdrop-blur-sm z-10 shrink-0">
          <p>Video Anomaly Detection System © 2026 | Powered by DSM + GNN + RA²R</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
