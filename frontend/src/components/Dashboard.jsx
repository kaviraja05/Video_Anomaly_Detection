import React, { useState, useEffect } from 'react';
import { getHealth, getModelInfo } from '../api/api';
import StatsCard from './StatsCard';
import { motion } from 'framer-motion';
import { BrainCircuit, Cpu, Binary, Video, Activity, Loader2 } from 'lucide-react';

const Dashboard = () => {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthData, modelData] = await Promise.all([
          getHealth(),
          getModelInfo()
        ]);
        setHealth(healthData);
        setModelInfo(modelData);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] h-full">
        <Loader2 size={48} className="text-blue-500 animate-spin mb-4 drop-shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
        <p className="text-slate-400 font-medium animate-pulse">Initializing connection to inference engine...</p>
      </div>
    );
  }

  const stats = [
    {
      icon: <BrainCircuit size={28} className="text-blue-400" />,
      title: 'Model Status',
      value: health?.model_loaded ? 'Loaded' : 'Offline',
      subtitle: modelInfo?.model_name || 'ProposedModel',
      gradientClass: 'bg-blue-500' // used for hover glow
    },
    {
      icon: <Cpu size={28} className="text-emerald-400" />,
      title: 'Device',
      value: health?.device?.toUpperCase() || 'CPU',
      subtitle: health?.device === 'cuda' ? 'GPU Accelerated' : 'CPU Mode',
      gradientClass: 'bg-emerald-500'
    },
    {
      icon: <Binary size={28} className="text-purple-400" />,
      title: 'Total Parameters',
      value: modelInfo?.parameters?.total 
        ? `${(modelInfo.parameters.total / 1e6).toFixed(2)}M` 
        : '2.32M',
      subtitle: 'Hybrid Architecture',
      gradientClass: 'bg-purple-500'
    },
    {
      icon: <Video size={28} className="text-amber-400" />,
      title: 'Dataset Features',
      value: health?.features_available || 0,
      subtitle: 'Static Reference Vids',
      gradientClass: 'bg-amber-500'
    }
  ];

  return (
    <div className="w-full">
      {/* Hero Header Section */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10 text-center md:text-left relative z-10"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider mb-4 shadow-[0_0_15px_rgba(59,130,246,0.2)]">
          <Activity size={14} className="animate-pulse" />
          <span>Surveillance Platform Active</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-slate-100 to-slate-400 tracking-tight mb-3">
          Video Anomaly Detection System
        </h1>
        <p className="text-slate-400 text-lg md:text-xl max-w-3xl font-light">
          AI-Powered Intelligent Video Surveillance Platform utilizing Hybrid Weakly Supervised Learning (DSM + RA²R + GNN)
        </p>
        
        {/* System Indicators */}
        <div className="flex flex-wrap items-center gap-4 mt-6">
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800 shadow-inner text-sm">
            <span className={`w-2.5 h-2.5 rounded-full ${health?.status === 'healthy' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-red-500'}`}></span>
            <span className="text-slate-300 font-medium">System Online</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800 shadow-inner text-sm">
            <Cpu size={16} className="text-blue-400" />
            <span className="text-slate-300 font-medium">{health?.device?.toUpperCase() || 'CPU'} Tracking</span>
          </div>
        </div>
      </motion.div>

      {/* Analytics Cards Grid */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {stats.map((stat, index) => (
          <StatsCard key={index} {...stat} />
        ))}
      </motion.div>
    </div>
  );
};

export default Dashboard;
