import React from 'react';
import Dashboard from '../components/Dashboard';
import { motion } from 'framer-motion';
import { UploadCloud, Activity } from 'lucide-react';

const Home = ({ onNavigate }) => {
  return (
    <div className="w-full absolute inset-0 overflow-y-auto custom-scrollbar p-6 md:p-8">
      {/* Decorative Background Blob */}
      <div className="fixed top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full bg-blue-600/10 blur-[120px] pointer-events-none"></div>
      
      <Dashboard />
      
      {/* Quick Actions / Getting Started */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-12 mb-8 relative z-10"
      >
        <h2 className="text-xl font-bold text-slate-200 mb-6 flex items-center gap-2">
          <span className="w-1 h-6 bg-blue-500 rounded-full"></span>
          Quick Actions
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Default Upload Action */}
          <motion.div 
            whileHover={{ y: -4, scale: 1.01 }}
            className="group cursor-pointer rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl border border-slate-700/50 p-6 md:p-8 shadow-xl overflow-hidden relative"
            onClick={() => onNavigate && onNavigate('upload')}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-bl-full blur-[3xl] transition-all group-hover:bg-blue-500/20"></div>
            
            <div className="relative z-10 flex items-start gap-6">
              <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(59,130,246,0.3)]">
                <UploadCloud size={32} />
              </div>
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2 group-hover:text-blue-300 transition-colors">Upload Custom Video</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Run full static anomaly detection on an MP4 video to generate detailed offline analytics and temporal breakdowns.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Live Analysis Action */}
          <motion.div 
            whileHover={{ y: -4, scale: 1.01 }}
            className="group cursor-pointer rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl border border-slate-700/50 p-6 md:p-8 shadow-xl overflow-hidden relative"
            onClick={() => onNavigate && onNavigate('live-analysis')}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-bl-full blur-[3xl] transition-all group-hover:bg-amber-500/20"></div>
            
            <div className="relative z-10 flex items-start gap-6">
              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(245,158,11,0.3)]">
                <Activity size={32} />
              </div>
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2 group-hover:text-amber-400 transition-colors">Live Stream Analysis</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Watch synchronized streaming anomaly scores in real time with dynamic graphs highlighting abnormal frames instantly.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default Home;
