import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard, 
  UploadCloud, 
  Activity, 
  ListTree, 
  Menu, 
  X,
  ShieldAlert,
  Cpu,
  LogOut,
  Camera,
  Sun,
  Moon
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const Sidebar = ({ currentPage, onNavigate, isAuthenticated, onLogout, userName, userEmail }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const menuItems = [
    { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard', description: 'System Overview' },
    ...(isAuthenticated ? [
      { id: 'upload', icon: UploadCloud, label: 'Upload Video', description: 'Analyze New Video' },
      { id: 'live-analysis', icon: Activity, label: 'Live Analysis', description: 'Real-time Detection' },
      { id: 'history', icon: ListTree, label: 'History', description: 'Past Analyses' }
    ] : [
      { id: 'login', icon: LayoutDashboard, label: 'Login', description: 'Sign in to account' }
    ])
  ];

  const sidebarVariants = {
    expanded: { width: '280px' },
    collapsed: { width: '80px' }
  };

  return (
    <motion.div 
      initial="expanded"
      animate={isCollapsed ? "collapsed" : "expanded"}
      variants={sidebarVariants}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="h-full bg-theme-bg backdrop-blur-xl border-r border-theme-border flex flex-col pt-6 pb-4 relative z-20 shrink-0 shadow-2xl transition-colors duration-300"
    >
      {/* Header / Brand */}
      <div className="flex items-center justify-between px-6 mb-10">
        <AnimatePresence>
          {!isCollapsed && (
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="flex items-center gap-3 overflow-hidden whitespace-nowrap"
            >
              <div className="p-2 bg-blue-500/20 rounded-xl border border-blue-500/30 text-blue-400">
                <Camera size={24} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-theme-text tracking-tight text-glow">VAD System</h1>
                <p className="text-xs text-theme-text opacity-50">AI Surveillance</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        <div className="flex items-center gap-2">
          <button 
            onClick={() => setIsCollapsed(!isCollapsed)}
            className={`p-2 rounded-lg hover:bg-theme-card text-theme-text opacity-70 hover:opacity-100 transition-all duration-300 flex items-center justify-center group`}
          >
            <Menu size={22} className="group-hover:scale-110 transition-transform duration-300" />
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-2 overflow-y-auto custom-scrollbar">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center p-3 rounded-xl transition-all duration-200 group relative
                ${isActive 
                  ? 'bg-blue-600/10 text-theme-accent' 
                  : 'text-theme-text opacity-70 hover:bg-theme-card hover:opacity-100'}`}
              title={isCollapsed ? item.label : ''}
            >
              {<Icon size={22} className={`min-w-[22px] transition-transform duration-300 ${isActive ? 'scale-110 drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]' : 'group-hover:scale-110'}`} />}
              
              <AnimatePresence>
                {!isCollapsed && (
                  <motion.div 
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    className="ml-4 text-left overflow-hidden whitespace-nowrap flex-1"
                  >
                    <div className={`font-medium text-sm ${isActive ? 'text-theme-accent' : 'text-theme-text'}`}>
                      {item.label}
                    </div>
                    {/* <div className="text-[10px] text-slate-500 mt-0.5">{item.description}</div> */}
                  </motion.div>
                )}
              </AnimatePresence>

              {isActive && (
                <motion.div 
                  layoutId="activeIndicator"
                  className="absolute left-0 top-1/4 bottom-1/4 w-1 bg-theme-accent rounded-r-full shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                />
              )}
            </button>
          );
        })}
      </nav>

      {isAuthenticated && (
        <div className="px-4 mt-2 mb-2">
          {!isCollapsed && (
            <div className="mb-4 p-3 bg-theme-card rounded-xl border border-theme-border">
              <div className="text-sm font-medium text-theme-text truncate">{userName || 'User'}</div>
              <div className="text-xs text-theme-text opacity-50 truncate">{userEmail || 'user@example.com'}</div>
            </div>
          )}
          <button
            onClick={onLogout}
            className={`w-full flex items-center p-3 rounded-xl transition-all duration-200 text-red-500 hover:bg-theme-card hover:text-red-400 group`}
            title={isCollapsed ? 'Logout' : ''}
          >
            <LogOut size={22} className="min-w-[22px] transition-transform duration-300 group-hover:scale-110" />
            {!isCollapsed && <span className="ml-4 font-medium text-sm text-left overflow-hidden whitespace-nowrap">Logout</span>}
          </button>
        </div>
      )}

      {/* Footer Details */}
      <div className="px-4 mt-auto pt-6 border-t border-theme-border">
        <AnimatePresence>
          {!isCollapsed ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="bg-theme-input rounded-xl p-4 border border-theme-border"
            >
              <div className="flex items-center gap-3 mb-3 text-sm text-theme-text opacity-80">
                <Cpu size={16} className="text-emerald-500" />
                <span className="font-medium">System Status</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-theme-text opacity-50">Model</span>
                  <span className="text-emerald-500 font-medium flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    Online
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-theme-text opacity-50">Accuracy</span>
                  <span className="text-theme-text opacity-80 font-medium">96.8%</span>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="flex justify-center text-emerald-500 relative group cursor-pointer">
              <Cpu size={20} />
              <span className="absolute top-0 right-0 w-2 h-2 rounded-full bg-emerald-500 animate-pulse border border-theme-bg"></span>
            </div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default Sidebar;
