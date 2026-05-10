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
        <h2 className="text-xl font-bold text-theme-text mb-6 flex items-center gap-2">
          <span className="w-1 h-6 bg-theme-accent rounded-full"></span>
          Quick Actions
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Default Upload Action */}
          <motion.div 
            whileHover={{ y: -4, scale: 1.01 }}
            className="group cursor-pointer rounded-2xl bg-theme-card backdrop-blur-xl border border-theme-border p-6 md:p-8 shadow-xl overflow-hidden relative transition-colors duration-300"
            onClick={() => onNavigate && onNavigate('upload')}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-theme-accent/10 rounded-bl-full blur-[3xl] transition-all group-hover:bg-theme-accent/20"></div>
            
            <div className="relative z-10 flex items-start gap-6">
              <div className="p-4 rounded-xl bg-theme-accent/10 border border-theme-accent/20 text-theme-accent group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(59,130,246,0.3)]">
                <UploadCloud size={32} />
              </div>
              <div>
                <h3 className="text-xl font-bold text-theme-text mb-2 group-hover:text-theme-accent transition-colors">Upload Custom Video</h3>
                <p className="text-theme-text opacity-70 text-sm leading-relaxed">
                  Run full static anomaly detection on an MP4 video to generate detailed offline analytics and temporal breakdowns.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Live Analysis Action */}
          <motion.div 
            whileHover={{ y: -4, scale: 1.01 }}
            className="group cursor-pointer rounded-2xl bg-theme-card backdrop-blur-xl border border-theme-border p-6 md:p-8 shadow-xl overflow-hidden relative transition-colors duration-300"
            onClick={() => onNavigate && onNavigate('live-analysis')}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-bl-full blur-[3xl] transition-all group-hover:bg-amber-500/20"></div>
            
            <div className="relative z-10 flex items-start gap-6">
              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-500 group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(245,158,11,0.3)]">
                <Activity size={32} />
              </div>
              <div>
                <h3 className="text-xl font-bold text-theme-text mb-2 group-hover:text-amber-500 transition-colors">Live Stream Analysis</h3>
                <p className="text-theme-text opacity-70 text-sm leading-relaxed">
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
