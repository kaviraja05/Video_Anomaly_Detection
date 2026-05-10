import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, BarChart2, Eye, Activity, Loader2 } from 'lucide-react';

const ExplainabilityPanel = ({ explanation, isAnalyzing }) => {
  if (!explanation && !isAnalyzing) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-theme-text opacity-50 min-h-[250px] h-full">
        <Cpu className="mb-4 opacity-30 animate-pulse text-blue-500" size={48} />
        <p className="text-sm font-medium">Awaiting video analysis to generate AI insights...</p>
      </div>
    );
  }

  if (!explanation && isAnalyzing) {
    return (
      <div className="flex flex-col items-center justify-center p-8 min-h-[250px] h-full relative overflow-hidden">
        {/* Animated background glow */}
        <div className="absolute inset-0 bg-blue-500/5 animate-pulse rounded-2xl"></div>
        <Loader2 className="animate-spin text-blue-500 mb-4 drop-shadow-[0_0_10px_rgba(59,130,246,0.5)]" size={32} />
        <span className="text-sm text-blue-400 font-medium tracking-wide">GNN Temporal Analysis & DSM Matching in Progress...</span>
      </div>
    );
  }

  const { reason, feature_importance, temporal_context } = explanation;

  return (
    <div className="flex flex-col h-full text-theme-text">
      <div className="flex items-center gap-2 mb-6 border-b border-theme-border pb-3">
        <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-500">
          <Eye size={18} />
        </div>
        <h3 className="font-bold text-lg text-theme-text">Explainable AI Insights</h3>
      </div>
      
      <div className="space-y-6 flex-1 overflow-y-auto custom-scrollbar pr-2">
        {/* Primary Conclusion */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
          <h4 className="flex items-center gap-2 text-sm font-semibold text-theme-text opacity-80">
            <Activity size={14} className="text-amber-500" /> Primary Conclusion
          </h4>
          <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-500 text-sm leading-relaxed shadow-inner">
            {reason}
          </div>
        </motion.div>

        {/* Temporal Context */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="space-y-2">
          <h4 className="flex items-center gap-2 text-sm font-semibold text-theme-text opacity-80">
            <BarChart2 size={14} className="text-blue-500" /> Temporal Features
          </h4>
          <div className="p-3.5 rounded-xl bg-theme-input border border-theme-border text-theme-text opacity-90 text-sm leading-relaxed">
            {temporal_context}
          </div>
        </motion.div>

        {/* GNN Weightings */}
        {feature_importance && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="space-y-4 pt-2">
            <h4 className="flex items-center gap-2 text-sm font-semibold text-theme-text opacity-80">
              <Cpu size={14} className="text-emerald-500" /> GNN Structural Reasoning Weights
            </h4>
            
            <div className="space-y-4">
              {/* Motion Intensity */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-theme-text opacity-70">Motion Intensity Magnitude</span>
                  <span className="text-emerald-500">{(feature_importance.motion_intensity * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 bg-theme-input rounded-full overflow-hidden border border-theme-border">
                  <motion.div 
                    initial={{ width: 0 }} animate={{ width: `${Math.min(100, feature_importance.motion_intensity * 100)}%` }} transition={{ duration: 1, ease: 'easeOut' }}
                    className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.5)]"
                  ></motion.div>
                </div>
              </div>

              {/* Temporal Patterns */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-theme-text opacity-70">Temporal Anomaly Consistency</span>
                  <span className="text-blue-500">{(feature_importance.temporal_patterns * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 bg-theme-input rounded-full overflow-hidden border border-theme-border">
                  <motion.div 
                    initial={{ width: 0 }} animate={{ width: `${Math.min(100, feature_importance.temporal_patterns * 100)}%` }} transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
                    className="h-full bg-gradient-to-r from-blue-600 to-blue-400 shadow-[0_0_10px_rgba(96,165,250,0.5)]"
                  ></motion.div>
                </div>
              </div>

              {/* GNN Reasoning */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-theme-text opacity-70">Graph Structure Irregularity</span>
                  <span className="text-purple-500">{(feature_importance.gnn_reasoning * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 bg-theme-input rounded-full overflow-hidden border border-theme-border">
                  <motion.div 
                    initial={{ width: 0 }} animate={{ width: `${Math.min(100, feature_importance.gnn_reasoning * 100)}%` }} transition={{ duration: 1, ease: 'easeOut', delay: 0.4 }}
                    className="h-full bg-gradient-to-r from-purple-600 to-purple-400 shadow-[0_0_10px_rgba(192,132,252,0.5)]"
                  ></motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default ExplainabilityPanel;
