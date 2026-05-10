import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Clock, Film, BarChart3, Activity, FastForward } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, ReferenceLine } from 'recharts';

const ResultsPage = ({ analysisData }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);

  if (!analysisData) {
    return (
      <div className="w-full h-[600px] flex items-center justify-center">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="glass-panel p-12 text-center max-w-md w-full"
        >
          <div className="mx-auto w-24 h-24 bg-theme-input rounded-full flex items-center justify-center mb-6 border border-theme-border shadow-inner">
            <AlertTriangle size={48} className="text-theme-text opacity-70" />
          </div>
          <h2 className="text-2xl font-bold text-theme-text mb-2">No Analysis Data</h2>
          <p className="text-theme-text opacity-70">Please upload a video from the Dashboard or Upload page to see the results.</p>
        </motion.div>
      </div>
    );
  }

  const {
    status,
    video_info,
    anomaly_detected,
    overall_score,
    anomaly_segments = [],
    frame_scores = [],
    explanation,
    processing_time_ms,
    demo_mode,
    model_confidence
  } = analysisData;

  // Prepare chart data
  const chartData = frame_scores.map((score, index) => ({
    frame: index,
    score: score,
    threshold: 0.5,
    anomaly: score > 0.5
  }));

  // Get top anomaly frames
  const anomalyFrames = frame_scores
    .map((score, index) => ({ frame: index, score }))
    .filter(item => item.score > 0.5)
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-theme-card backdrop-blur-md border border-theme-border p-3 rounded-xl shadow-2xl">
          <p className="text-theme-text text-xs font-semibold mb-1">Frame {data.frame}</p>
          <div className="flex items-center gap-2">
            <span className="text-theme-text opacity-70 text-xs">Score:</span>
            <span className={`font-mono text-sm ${data.anomaly ? 'text-red-400 drop-shadow-[0_0_5px_rgba(248,113,113,0.5)]' : 'text-emerald-400'}`}>
              {(data.score * 100).toFixed(1)}%
            </span>
          </div>
          <p className={`text-xs mt-1 px-2 py-0.5 rounded flex items-center gap-1 w-fit ${data.anomaly ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
            {data.anomaly ? <AlertTriangle size={12} /> : null}
            {data.anomaly ? 'Anomaly' : 'Normal'}
          </p>
        </div>
      );
    }
    return null;
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="w-full max-w-7xl mx-auto space-y-8 relative z-10 pb-10"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-theme-border pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-theme-text flex items-center gap-3 mb-2">
            <BarChart3 className="text-blue-500" size={32} />
            Analysis Results
            {demo_mode && <span className="ml-2 text-xs font-semibold bg-indigo-500/20 text-indigo-300 px-3 py-1 rounded-full border border-indigo-500/30">Demo Mode</span>}
          </h1>
          <p className="text-theme-text opacity-70">Detailed offline post-processing analytics for the uploaded video.</p>
        </div>
        <div className={`flex items-center gap-3 px-6 py-2.5 rounded-full border shadow-lg backdrop-blur-md ${
          anomaly_detected ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
        }`}>
          {anomaly_detected ? <AlertTriangle size={24} className="animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.5)] rounded-full" /> : <CheckCircle size={24} />}
          <div className="flex flex-col">
            <span className="font-bold text-lg leading-tight uppercase tracking-wide">{anomaly_detected ? 'Anomaly' : 'Normal'}</span>
            {model_confidence !== undefined && (
              <span className="text-[10px] font-mono tracking-wider opacity-80 uppercase">
                {(model_confidence * 100).toFixed(1)}% Confidence
              </span>
            )}
          </div>
        </div>
      </motion.div>

      {/* Summary Cards */}
      <motion.div variants={itemVariants} className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
        {[
          { icon: <Activity className="text-blue-400" size={24}/>, label: 'Overall Score', value: `${(overall_score * 100).toFixed(1)}%`, bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
          { icon: <Film className="text-purple-400" size={24}/>, label: 'Total Frames', value: video_info?.total_frames || frame_scores.length, bg: 'bg-purple-500/10', border: 'border-purple-500/20' },
          { icon: <AlertTriangle className="text-amber-400" size={24}/>, label: 'Anomaly Segments', value: anomaly_segments.length, bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
          { icon: <FastForward className="text-emerald-400" size={24}/>, label: 'Processing Time', value: `${(processing_time_ms / 1000).toFixed(2)}s`, bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
        ].map((stat, i) => (
          <div key={i} className="glass-panel p-5 rounded-2xl flex items-center gap-4 hover:scale-[1.02] transition-transform">
            <div className={`p-3 rounded-xl border ${stat.bg} ${stat.border}`}>
              {stat.icon}
            </div>
            <div>
              <p className="text-theme-text opacity-70 text-xs font-medium uppercase tracking-wider mb-1">{stat.label}</p>
              <p className="text-2xl font-bold text-theme-text">{stat.value}</p>
            </div>
          </div>
        ))}
      </motion.div>

      {/* Anomaly Score Graph */}
      <motion.div variants={itemVariants} className="glass-panel p-6 rounded-3xl">
        <h2 className="text-xl font-bold text-theme-text mb-6 flex items-center gap-2">
          <Activity className="text-indigo-400" /> Full Temporal Score Timeline
        </h2>
        <div className="h-[350px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="scoreGradientResults" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.6}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" vertical={false} />
              <XAxis dataKey="frame" stroke="rgba(148, 163, 184, 0.5)" tick={{ fill: 'rgba(148, 163, 184, 0.7)', fontSize: 12 }} />
              <YAxis stroke="rgba(148, 163, 184, 0.5)" tick={{ fill: 'rgba(148, 163, 184, 0.7)', fontSize: 12 }} domain={[0, 1]} ticks={[0, 0.5, 1]} />
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)' }} />
              <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1} label={{ value: 'Threshold', fill: '#ef4444', fontSize: 12, position: 'insideTopLeft' }} />
              <Area type="monotone" dataKey="score" stroke="#8b5cf6" strokeWidth={2} fill="url(#scoreGradientResults)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* AI Explanation & Top Frames Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Explanation Section */}
        {explanation && (
          <motion.div variants={itemVariants} className="lg:col-span-1 space-y-6">
            <h2 className="text-xl font-bold text-theme-text flex items-center gap-2">
              <span className="w-1.5 h-6 bg-blue-500 rounded-full"></span> AI Explanations
            </h2>
            <div className="glass-panel p-6 rounded-3xl space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-theme-text opacity-70 uppercase tracking-wider mb-2">Analysis Summary</h3>
                <p className="text-theme-text text-sm leading-relaxed">{explanation.reason}</p>
              </div>
              <div className="h-px bg-theme-border"></div>
              <div>
                <h3 className="text-sm font-semibold text-theme-text opacity-70 uppercase tracking-wider mb-2">Temporal Context</h3>
                <p className="text-theme-text opacity-90 text-sm leading-relaxed">{explanation.temporal_context}</p>
              </div>
              {explanation.feature_importance && (
                <>
                  <div className="h-px bg-theme-border"></div>
                  <div>
                    <h3 className="text-sm font-semibold text-theme-text opacity-70 uppercase tracking-wider mb-3">Feature Importance</h3>
                    <div className="space-y-3 mt-2">
                      {Object.entries(explanation.feature_importance).map(([key, value]) => (
                        <div key={key}>
                          <div className="flex justify-between text-xs font-medium mb-1">
                            <span className="text-theme-text opacity-90 capitalize">{key.replace(/_/g, ' ')}</span>
                            <span className="text-blue-400">{(value * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-theme-input rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${value * 100}%` }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}

        {/* Anomaly Frames */}
        {anomalyFrames.length > 0 && (
          <motion.div variants={itemVariants} className="lg:col-span-2 space-y-6">
            <h2 className="text-xl font-bold text-theme-text flex items-center gap-2 mb-2">
              <span className="w-1.5 h-6 bg-red-500 rounded-full"></span> Top Anomaly Frames
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              {anomalyFrames.map((frameData) => (
                <div 
                  key={frameData.frame}
                  onClick={() => setSelectedFrame(frameData.frame)}
                  className={`glass-panel p-4 rounded-2xl cursor-pointer hover:scale-105 transition-transform border group relative overflow-hidden flex flex-col items-center
                    ${selectedFrame === frameData.frame ? 'border-red-500/50 shadow-[0_0_20px_rgba(239,68,68,0.2)]' : 'border-theme-border hover:border-theme-accent'}`}
                >
                  <div className="absolute inset-0 bg-red-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  
                  {analysisData.frame_images && analysisData.frame_images[frameData.frame] ? (
                    <img 
                      src={analysisData.frame_images[frameData.frame]} 
                      alt={`Anomaly Frame ${frameData.frame}`} 
                      className="w-full h-24 object-cover rounded-lg mb-3 border border-theme-border shadow-md group-hover:border-red-500/50 transition-colors z-10"
                    />
                  ) : (
                    <Film size={36} strokeWidth={1} className="text-theme-text opacity-50 mb-3 group-hover:text-red-400 transition-colors z-10" />
                  )}
                  
                  <span className="text-sm font-mono text-theme-text opacity-90 z-10">Frame {frameData.frame}</span>
                  <div className="mt-3 text-center z-10 w-full pt-3 border-t border-theme-border">
                    <div className="text-xs text-theme-text opacity-70 mb-1">Score</div>
                    <div className={`text-sm font-bold ${frameData.score > 0.8 ? 'text-red-500' : 'text-orange-400'}`}>
                      {(frameData.score * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>

      {/* Anomaly Segments Table */}
      {anomaly_segments.length > 0 && (
        <motion.div variants={itemVariants} className="space-y-4">
          <h2 className="text-xl font-bold text-theme-text flex items-center gap-2">
            <span className="w-1.5 h-6 bg-emerald-500 rounded-full"></span> Detected Segments Log
          </h2>
          <div className="glass-panel rounded-3xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-theme-text opacity-90">
                <thead className="bg-theme-input border-b border-theme-border text-xs uppercase font-semibold text-theme-text opacity-70 tracking-wider">
                  <tr>
                    <th className="px-6 py-4 rounded-tl-3xl">Segment ID</th>
                    <th className="px-6 py-4">Start Frame</th>
                    <th className="px-6 py-4">End Frame</th>
                    <th className="px-6 py-4">Duration</th>
                    <th className="px-6 py-4">Confidence</th>
                    <th className="px-6 py-4 rounded-tr-3xl">Severity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-theme-border">
                  {anomaly_segments.map((segment, idx) => (
                    <tr key={idx} className="hover:bg-theme-card transition-colors">
                      <td className="px-6 py-4 font-mono">#{idx + 1}</td>
                      <td className="px-6 py-4 font-mono">{segment.start_frame}</td>
                      <td className="px-6 py-4 font-mono">{segment.end_frame}</td>
                      <td className="px-6 py-4">
                        {(segment.timestamp_end - segment.timestamp_start).toFixed(2)}s
                      </td>
                      <td className="px-6 py-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                          {(segment.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                          segment.severity === 'high' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                          segment.severity === 'medium' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                          'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                        }`}>
                          {segment.severity.charAt(0).toUpperCase() + segment.severity.slice(1)} Risk
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ResultsPage;
