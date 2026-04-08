import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts';
import { Activity, AlertTriangle } from 'lucide-react';

const LiveAnomalyGraph = ({ 
  isLiveMode = false,
  staticData = [], 
  currentFrame = 0,
  totalFrames = 0,
  threshold = 0.5,
  onFrameClick
}) => {
  const [graphData, setGraphData] = useState([]);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!isLiveMode && staticData && staticData.length > 0) {
      const formattedData = staticData.map((score, index) => ({
        frame: index,
        score: score,
        anomaly: score > threshold
      }));
      setGraphData(formattedData);
    } else if (isLiveMode) {
      setGraphData([]);
    }
  }, [isLiveMode, staticData, threshold]);

  const addDataPoint = (frameIndex, score) => {
    setGraphData(prev => {
      if (prev.some(d => d.frame === frameIndex)) return prev;
      const newPoint = { frame: frameIndex, score, anomaly: score > threshold };
      return [...prev, newPoint].sort((a, b) => a.frame - b.frame);
    });
  };

  const addDataBatch = (startFrame, scores) => {
    setGraphData(prev => {
      const newPoints = scores.map((score, idx) => ({
        frame: startFrame + idx,
        score,
        anomaly: score > threshold
      }));
      const existingFrames = new Set(prev.map(d => d.frame));
      const uniqueNewPoints = newPoints.filter(p => !existingFrames.has(p.frame));
      return [...prev, ...uniqueNewPoints].sort((a, b) => a.frame - b.frame);
    });
  };

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.addDataPoint = addDataPoint;
      chartRef.current.addDataBatch = addDataBatch;
      chartRef.current.clearData = () => setGraphData([]);
      chartRef.current.setData = (data) => setGraphData(data);
    }
  }, []);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/50 p-3 rounded-xl shadow-2xl">
          <p className="text-slate-300 text-xs font-semibold mb-1">Frame {data.frame}</p>
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-xs">Score:</span>
            <span className={`font-mono text-sm ${data.anomaly ? 'text-red-400 drop-shadow-[0_0_5px_rgba(248,113,113,0.5)]' : 'text-emerald-400'}`}>
              {(data.score * 100).toFixed(1)}%
            </span>
          </div>
          <p className={`text-xs mt-1 px-2 py-0.5 rounded flex items-center gap-1 ${data.anomaly ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
            {data.anomaly ? <AlertTriangle size={12} /> : null}
            {data.anomaly ? 'Anomaly Detected' : 'Normal Activity'}
          </p>
        </div>
      );
    }
    return null;
  };

  const CustomDot = (props) => {
    const { cx, cy, payload } = props;
    if (payload.anomaly) {
      return (
        <circle 
          cx={cx} cy={cy} r={3} 
          fill="#ef4444" stroke="#fff" strokeWidth={1}
          style={{ filter: 'drop-shadow(0 0 4px rgba(239, 68, 68, 0.8))' }}
        />
      );
    }
    return null;
  };

  const handleChartClick = (data) => {
    if (data?.activePayload?.length > 0) {
      const frame = data.activePayload[0].payload.frame;
      if (onFrameClick) onFrameClick(frame);
    }
  };

  return (
    <div className="flex flex-col h-full w-full" ref={chartRef}>
      <div className="flex-1 min-h-[250px] relative">
        {graphData.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
            <Activity className="mb-2 opacity-50 animate-pulse" size={32} />
            <p className="text-sm">{isLiveMode ? 'Waiting for stream...' : 'No data available'}</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={graphData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }} onClick={handleChartClick} className="cursor-crosshair">
              <defs>
                <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.5}/>
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" vertical={false} />
              
              <XAxis 
                dataKey="frame" 
                stroke="rgba(148, 163, 184, 0.4)"
                tick={{ fill: 'rgba(148, 163, 184, 0.6)', fontSize: 10 }}
                tickMargin={8}
                minTickGap={30}
              />
              
              <YAxis 
                stroke="rgba(148, 163, 184, 0.4)"
                tick={{ fill: 'rgba(148, 163, 184, 0.6)', fontSize: 10 }}
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
              />
              
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }} />
              
              <ReferenceLine 
                y={threshold} 
                stroke="#ef4444" 
                strokeDasharray="4 4" 
                strokeWidth={1}
                label={{ value: 'Threshold', position: 'insideTopLeft', fill: '#ef4444', fontSize: 10, opacity: 0.8 }}
              />
              
              {currentFrame > 0 && currentFrame <= graphData.length && (
                <ReferenceLine 
                  x={currentFrame} 
                  stroke="#3b82f6" 
                  strokeWidth={1.5}
                  className="drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]"
                />
              )}
              
              <Area
                type="monotone"
                dataKey="score"
                stroke="#f59e0b"
                strokeWidth={2}
                fill="url(#colorScore)"
                dot={<CustomDot />}
                activeDot={{ r: 5, fill: '#f59e0b', stroke: '#fff', strokeWidth: 2 }}
                animationDuration={isLiveMode ? 300 : 1000}
                isAnimationActive={true}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export const LiveAnomalyGraphRef = React.forwardRef((props, ref) => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (ref) {
      ref.current = {
        addDataPoint: (frame, score) => chartRef.current?.addDataPoint(frame, score),
        addDataBatch: (startFrame, scores) => chartRef.current?.addDataBatch(startFrame, scores),
        clearData: () => chartRef.current?.clearData(),
        setData: (data) => chartRef.current?.setData(data)
      };
    }
  }, [ref]);

  return <LiveAnomalyGraph {...props} ref={chartRef} />;
});

export default LiveAnomalyGraphRef;
