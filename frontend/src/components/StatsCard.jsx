import React from 'react';
import { motion } from 'framer-motion';

const StatsCard = ({ icon, title, value, subtitle, gradientClass }) => {
  return (
    <motion.div 
      whileHover={{ y: -5, scale: 1.02 }}
      transition={{ duration: 0.2 }}
      className="relative overflow-hidden rounded-2xl bg-slate-900/40 backdrop-blur-xl border border-slate-700/50 p-6 flex items-start space-x-4 shadow-xl group"
    >
      {/* Background Gradient Glow on Hover */}
      <div className={`absolute -inset-0.5 opacity-0 group-hover:opacity-20 transition-opacity duration-500 blur-xl ${gradientClass}`}></div>
      
      {/* Icon Container */}
      <div className={`p-4 rounded-xl flex-shrink-0 bg-slate-800/80 border border-slate-700 shadow-inner group-hover:scale-110 transition-transform duration-300 relative z-10 ${gradientClass.replace('from', 'text').split(' ')[0]}`}>
        {icon}
      </div>
      
      {/* Content Container */}
      <div className="flex-1 relative z-10">
        <h3 className="text-slate-400 text-sm font-medium tracking-wide uppercase mb-1">{title}</h3>
        <p className="text-3xl font-bold text-slate-100 tracking-tight drop-shadow-md mb-1">{value}</p>
        {subtitle && <p className="text-slate-500 text-xs">{subtitle}</p>}
      </div>
    </motion.div>
  );
};

export default StatsCard;
