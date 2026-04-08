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
  Camera
} from 'lucide-react';

const Sidebar = ({ currentPage, onNavigate, isAuthenticated, onLogout, userName, userEmail }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

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
      className="h-full bg-slate-900/80 backdrop-blur-xl border-r border-slate-800 flex flex-col pt-6 pb-4 relative z-20 shrink-0 shadow-2xl"
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
                <h1 className="text-lg font-bold text-slate-100 tracking-tight text-glow">VAD System</h1>
                <p className="text-xs text-slate-400">AI Surveillance</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={`p-2 rounded-lg hover:bg-slate-800 transition-colors text-slate-400 hover:text-slate-200 ${isCollapsed ? 'mx-auto' : ''}`}
        >
          {isCollapsed ? <Menu size={20} /> : <X size={20} />}
        </button>
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
                  ? 'bg-blue-600/10 text-blue-400' 
                  : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'}`}
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
                    <div className={`font-medium text-sm ${isActive ? 'text-blue-400' : 'text-slate-200'}`}>
                      {item.label}
                    </div>
                    {/* <div className="text-[10px] text-slate-500 mt-0.5">{item.description}</div> */}
                  </motion.div>
                )}
              </AnimatePresence>

              {isActive && (
                <motion.div 
                  layoutId="activeIndicator"
                  className="absolute left-0 top-1/4 bottom-1/4 w-1 bg-blue-500 rounded-r-full shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                />
              )}
            </button>
          );
        })}
      </nav>

      {isAuthenticated && (
        <div className="px-4 mt-2 mb-2">
          {!isCollapsed && (
            <div className="mb-4 p-3 bg-slate-800/40 rounded-xl border border-slate-700/50">
              <div className="text-sm font-medium text-slate-200 truncate">{userName || 'User'}</div>
              <div className="text-xs text-slate-500 truncate">{userEmail || 'user@example.com'}</div>
            </div>
          )}
          <button
            onClick={onLogout}
            className={`w-full flex items-center p-3 rounded-xl transition-all duration-200 text-red-400 hover:bg-slate-800/50 hover:text-red-300 group`}
            title={isCollapsed ? 'Logout' : ''}
          >
            <LogOut size={22} className="min-w-[22px] transition-transform duration-300 group-hover:scale-110" />
            {!isCollapsed && <span className="ml-4 font-medium text-sm text-left overflow-hidden whitespace-nowrap">Logout</span>}
          </button>
        </div>
      )}

      {/* Footer Details */}
      <div className="px-4 mt-auto pt-6 border-t border-slate-800/50">
        <AnimatePresence>
          {!isCollapsed ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50"
            >
              <div className="flex items-center gap-3 mb-3 text-sm text-slate-300">
                <Cpu size={16} className="text-emerald-400" />
                <span className="font-medium">System Status</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Model</span>
                  <span className="text-emerald-400 font-medium flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Online
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Accuracy</span>
                  <span className="text-slate-300 font-medium">96.8%</span>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="flex justify-center text-emerald-400 relative group cursor-pointer">
              <Cpu size={20} />
              <span className="absolute top-0 right-0 w-2 h-2 rounded-full bg-emerald-400 animate-pulse border border-slate-900"></span>
            </div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default Sidebar;
