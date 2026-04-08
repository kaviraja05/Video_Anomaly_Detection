import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock, AlertTriangle, AlertCircle, Info } from 'lucide-react';

const IntelligentTimeline = ({ segments, onEventClick, currentTime }) => {
  if (!segments || segments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-slate-500 min-h-[150px]">
        <Info className="mb-2 opacity-50" size={32} />
        <p className="text-sm">No anomalous events detected yet.</p>
      </div>
    );
  }

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'high': return <AlertTriangle className="text-red-500" size={16} />;
      case 'medium': return <AlertCircle className="text-amber-500" size={16} />;
      default: return <Info className="text-blue-500" size={16} />;
    }
  };

  return (
    <div className="w-full flex flex-col h-full max-h-[400px]">
      <div className="flex items-center justify-between mb-4 px-2">
        <h3 className="text-slate-200 font-semibold flex items-center gap-2">
          <Clock className="text-blue-400" size={18} />
          Event Timeline
        </h3>
        <span className="text-xs font-medium bg-slate-800 text-slate-300 px-2 py-1 rounded-md border border-slate-700">
          {segments.length} Events
        </span>
      </div>
      
      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-3 pb-2 relative">
        {/* Continuous track line */}
        <div className="absolute left-[15px] top-2 bottom-2 w-px bg-slate-800 -z-10"></div>

        <AnimatePresence>
          {segments.map((segment, index) => {
            const isActive = currentTime >= segment.timestamp_start && currentTime <= segment.timestamp_end;
            
            // Determine styles based on severity and active state
            let bgClass = 'bg-slate-900/60 border-slate-800 hover:border-slate-600';
            let dotClass = 'bg-slate-600 border-slate-900';
            let labelText = 'Minor Irregularity';
            
            if (segment.severity === 'high') {
              bgClass = isActive ? 'bg-red-500/10 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.15)] text-red-100' : 'bg-slate-900/60 border-red-900/50 hover:border-red-500/50';
              dotClass = isActive ? 'bg-red-500 border-red-900 shadow-[0_0_8px_rgba(239,68,68,0.8)]' : 'bg-red-500/50 border-slate-900';
              labelText = 'High Confidence Anomaly';
            } else if (segment.severity === 'medium') {
              bgClass = isActive ? 'bg-amber-500/10 border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.15)] text-amber-100' : 'bg-slate-900/60 border-amber-900/50 hover:border-amber-500/50';
              dotClass = isActive ? 'bg-amber-500 border-amber-900 shadow-[0_0_8px_rgba(245,158,11,0.8)]' : 'bg-amber-500/50 border-slate-900';
              labelText = 'Suspicious Activity';
            }

            return (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                key={index} 
                className={`relative flex items-center group cursor-pointer`}
                onClick={() => onEventClick && onEventClick(segment.timestamp_start)}
              >
                {/* Timeline dot */}
                <div className={`w-3.5 h-3.5 rounded-full border-2 z-10 mr-4 ml-[9px] transition-all duration-300 ${dotClass} ${isActive ? 'scale-125' : 'group-hover:scale-110'}`}></div>
                
                {/* Event Card */}
                <div className={`flex-1 rounded-xl border p-3 transition-all duration-300 backdrop-blur-sm ${bgClass}`}>
                  <div className="flex justify-between items-center mb-1">
                    <div className="text-xs font-mono font-bold tracking-wider text-slate-400 group-hover:text-slate-200 transition-colors">
                      {formatTime(segment.timestamp_start)} - {formatTime(segment.timestamp_end)}
                    </div>
                    {isActive && (
                      <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center justify-between mt-1.5">
                    <div className="flex items-center gap-2 font-medium text-sm">
                      {getSeverityIcon(segment.severity)}
                      <span className="text-slate-200">{labelText}</span>
                    </div>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-md bg-slate-800 text-slate-300">
                      {(segment.confidence * 100).toFixed(0)}% Match
                    </span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default IntelligentTimeline;
