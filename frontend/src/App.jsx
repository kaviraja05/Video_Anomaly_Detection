/**
 * Main App Component
 * Root component with sidebar navigation and page routing
 */

import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import LiveAnalysisPage from './pages/LiveAnalysisPage';
import Login from './pages/Login';
import Register from './pages/Register';
import History from './pages/History';

function App() {
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
    <div className="flex h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans bg-gradient-dark">
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
        <div className="flex-1 overflow-y-auto p-6 md:p-8 w-full max-w-[1600px] mx-auto custom-scrollbar">
          {renderPage()}
        </div>
        
        {/* Footer */}
        <footer className="w-full py-4 text-center text-xs text-slate-500 border-t border-slate-800/50 backdrop-blur-sm z-10 shrink-0">
          <p>Video Anomaly Detection System © 2026 | Powered by DSM + GNN + RA²R</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
