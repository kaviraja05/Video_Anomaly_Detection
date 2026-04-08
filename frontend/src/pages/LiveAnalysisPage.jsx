import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, Video, Activity, AlertCircle, CheckCircle, PlaySquare, Maximize2, ShieldAlert, Target } from 'lucide-react';
import VideoPlayer from '../components/VideoPlayer';
import LiveAnomalyGraph from '../components/LiveAnomalyGraph';
import IntelligentTimeline from '../components/IntelligentTimeline';
import ExplainabilityPanel from '../components/ExplainabilityPanel';

const LiveAnalysisPage = () => {
  // State management
  const [videoFile, setVideoFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState('idle'); // idle, buffering, ready, streaming, complete, error
  const [statusMessage, setStatusMessage] = useState('');
  const [videoInfo, setVideoInfo] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [frameScores, setFrameScores] = useState([]);
  const [finalResults, setFinalResults] = useState(null);
  const [error, setError] = useState(null);
  const [bufferedAnalysis, setBufferedAnalysis] = useState(null);
  const [drawnFrameCount, setDrawnFrameCount] = useState(0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);

  // Refs
  const graphRef = useRef(null);
  const eventSourceRef = useRef(null);
  const fileInputRef = useRef(null);

  // Handlers
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) validateAndSetFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) validateAndSetFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const validateAndSetFile = (file) => {
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    const maxSize = 500 * 1024 * 1024; // 500MB

    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      setError('Please upload a valid video file (MP4, AVI, MOV, MKV)');
      return;
    }

    if (file.size > maxSize) {
      setError('File size exceeds 500MB limit');
      return;
    }

    setVideoFile(file);
    setError(null);
    setAnalysisStatus('buffering');
    setStatusMessage('Preparing analysis buffer...');
    setFrameScores([]);
    setFinalResults(null);
    setVideoInfo(null);
    setCurrentFrame(0);
    setBufferedAnalysis(null);
    setDrawnFrameCount(0);
    
    bufferAnalysis(file);
  };

  const bufferAnalysis = async (file) => {
    try {
      const formData = new FormData();
      formData.append('video', file);

      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      setBufferedAnalysis(data);
      setVideoInfo(data.video_info);
      setAnalysisStatus('ready');
      setStatusMessage('Analysis ready. Play the video to begin visualization.');
    } catch (err) {
      console.error('Buffer error:', err);
      setError(err.message || 'Failed to prepare analysis');
      setAnalysisStatus('error');
    }
  };

  const handleVideoTimeUpdate = (time, frame) => {
    setCurrentTime(time);
    setCurrentFrame(frame);
    
    if (bufferedAnalysis && bufferedAnalysis.frame_scores) {
      if (analysisStatus === 'ready' && isVideoPlaying) {
         setAnalysisStatus('streaming');
         setIsAnalyzing(true);
      }
      
      const targetFrame = Math.min(frame, bufferedAnalysis.frame_scores.length - 1);
      if (targetFrame > drawnFrameCount) {
         setStatusMessage('Streaming detection results...');
         const newScores = bufferedAnalysis.frame_scores.slice(drawnFrameCount, targetFrame + 1);
         if (newScores.length > 0) {
            if (graphRef.current?.addDataBatch) {
              graphRef.current.addDataBatch(drawnFrameCount, newScores);
            }
            setDrawnFrameCount(targetFrame + 1);
            setFrameScores(prev => [...prev, ...newScores]);
            
            if (targetFrame >= bufferedAnalysis.frame_scores.length - 1) {
                setAnalysisStatus('complete');
                setFinalResults(bufferedAnalysis);
                setIsAnalyzing(false);
                setStatusMessage('Analysis visualization complete!');
            }
         }
      } else if (targetFrame < drawnFrameCount) {
         // User scrubbed backwards
         setDrawnFrameCount(targetFrame + 1);
         const slicedScores = bufferedAnalysis.frame_scores.slice(0, targetFrame + 1);
         setFrameScores(slicedScores);
         if (graphRef.current?.setData) {
            graphRef.current.setData(slicedScores.map((score, index) => ({
              frame: index,
              score: score,
              anomaly: score > 0.5
            })));
         }
      }
    }
  };

  const handleGraphFrameClick = (frame) => {
    console.log('Clicked frame:', frame);
  };

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <div className="w-full relative z-10">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-500 tracking-tight mb-2 flex items-center gap-3">
            <Activity className="text-amber-500" size={32} />
            Live Anomaly Detection
          </h1>
          <p className="text-slate-400 text-sm md:text-base">
            Upload a video to analyze in real-time with synchronized spatial-temporal playback.
          </p>
        </motion.div>
        
        {analysisStatus !== 'idle' && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }} 
            animate={{ opacity: 1, scale: 1 }}
            className={`flex items-center gap-3 px-4 py-2 rounded-full border shadow-lg backdrop-blur-md ${
              analysisStatus === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400' :
              analysisStatus === 'complete' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
              'bg-amber-500/10 border-amber-500/30 text-amber-400'
            }`}
          >
            {analysisStatus === 'streaming' && (
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping"></span>
            )}
            {analysisStatus === 'complete' && <CheckCircle size={16} />}
            {analysisStatus === 'error' && <AlertCircle size={16} />}
            <span className="font-medium text-sm">{statusMessage}</span>
          </motion.div>
        )}
      </div>

      {!videoFile ? (
        <motion.div 
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-10 mt-10 md:mt-20 max-w-2xl mx-auto flex flex-col items-center justify-center border-dashed border-2 border-slate-700 hover:border-amber-500/50 transition-colors cursor-pointer group"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="p-5 rounded-full bg-slate-800 text-amber-400 mb-4 shadow-inner group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(245,158,11,0.3)] transition-all">
            <Video size={48} strokeWidth={1.5} />
          </div>
          <h3 className="text-xl font-semibold text-slate-200 mb-2">Select Video File for Live Processing</h3>
          <p className="text-slate-500 mb-4">Drag & drop or click to browse</p>
          <span className="text-xs text-slate-400 px-3 py-1 bg-slate-900 rounded-lg border border-slate-800">MP4, AVI, MOV, MKV (max 500MB)</span>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,.mp4,.avi,.mov,.mkv"
            onChange={handleFileSelect}
            className="hidden"
          />

          {error && (
            <div className="mt-6 flex items-center gap-2 text-red-400 bg-red-500/10 px-4 py-2 rounded-lg border border-red-500/20">
              <AlertCircle size={16} />
              <span className="text-sm">{error}</span>
            </div>
          )}
        </motion.div>
      ) : (
        <div className="flex flex-col gap-6">
          {/* Controls / Info Bar */}
          <motion.div 
            initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-4 flex flex-col md:flex-row items-center justify-between gap-4"
          >
            <div className="flex items-center gap-4">
              <div className="p-2 bg-amber-500/20 text-amber-400 rounded-lg">
                <PlaySquare size={24} />
              </div>
              <div>
                <h3 className="text-slate-200 font-semibold text-sm truncate max-w-[200px] md:max-w-md">{videoFile.name}</h3>
                <p className="text-slate-400 text-xs text-left">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row items-center gap-4 w-full md:w-auto mt-4 md:mt-0">
              {(analysisStatus === 'buffering' || analysisStatus === 'ready' || analysisStatus === 'streaming') && (
                <div className="flex-1 w-full md:w-64">
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>{statusMessage}</span>
                    {analysisStatus === 'streaming' && videoInfo && (
                      <span className="text-amber-400">{((drawnFrameCount / (videoInfo.total_frames || 1)) * 100).toFixed(0)}%</span>
                    )}
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-amber-500 transition-all duration-300"
                      style={{ width: analysisStatus === 'buffering' ? '100%' : `${(drawnFrameCount / (videoInfo?.total_frames || 1)) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              <button 
                className="px-4 py-2 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-colors text-sm whitespace-nowrap"
                onClick={() => {
                  setVideoFile(null);
                  setAnalysisStatus('idle');
                  setIsAnalyzing(false);
                  setFrameScores([]);
                  setBufferedAnalysis(null);
                  setDrawnFrameCount(0);
                  setError(null);
                }}
              >
                Change Video
              </button>
            </div>
          </motion.div>

          {/* Split Screen Analysis */}
          {(analysisStatus === 'buffering' || analysisStatus === 'ready' || analysisStatus === 'streaming' || analysisStatus === 'complete') && (
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-6"
            >
              <div className="flex flex-col gap-6">
                {/* Video Player Box */}
                <div className="glass-panel overflow-hidden border border-slate-700/50 shadow-2xl relative p-1 rounded-2xl group">
                  <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-amber-500/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  <VideoPlayer
                    videoFile={videoFile}
                    currentFrame={currentFrame}
                    totalFrames={videoInfo?.total_frames}
                    fps={videoInfo?.fps || 30}
                    onTimeUpdate={handleVideoTimeUpdate}
                    onPlay={() => setIsVideoPlaying(true)}
                    onPause={() => setIsVideoPlaying(false)}
                  />
                  <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur text-white text-xs px-2 py-1 rounded shadow pointer-events-none">
                    Frame: {currentFrame}
                  </div>
                </div>

                {/* Explainability Panel */}
                <div className="glass-panel p-5 rounded-2xl shadow-xl border border-slate-700/50 min-h-[200px]">
                  <ExplainabilityPanel 
                    explanation={finalResults?.explanation}
                    isAnalyzing={analysisStatus === 'streaming'}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-6">
                {/* Live Graph Box */}
                <div className="glass-panel p-5 rounded-2xl shadow-xl border border-slate-700/50 flex-1 min-h-[350px] flex flex-col">
                  <h3 className="text-slate-200 font-semibold mb-4 flex items-center gap-2">
                    <Activity className="text-amber-500" size={18} />
                    Live Anomaly Score Monitor
                  </h3>
                  <div className="flex-1 relative">
                    <LiveAnomalyGraph
                      ref={graphRef}
                      isLiveMode={analysisStatus === 'streaming'}
                      staticData={analysisStatus === 'complete' ? frameScores : []}
                      currentFrame={currentFrame}
                      totalFrames={videoInfo?.total_frames}
                      threshold={0.5}
                      onFrameClick={handleGraphFrameClick}
                    />
                  </div>
                </div>

                {/* Timeline Box */}
                <div className="glass-panel p-5 rounded-2xl shadow-xl border border-slate-700/50">
                  <IntelligentTimeline 
                    segments={finalResults?.anomaly_segments || videoInfo?.anomaly_segments || []} 
                    currentTime={currentTime}
                    onEventClick={(time) => {
                       console.log("Seeking to timestamp:", time);
                    }}
                  />
                </div>

                {/* Small Stats Row */}
                {videoInfo && (
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="glass-panel p-3 rounded-xl border border-slate-700/50">
                      <p className="text-slate-500 text-xs mb-1">Total Frames</p>
                      <p className="text-slate-200 font-semibold">{videoInfo.total_frames}</p>
                    </div>
                    <div className="glass-panel p-3 rounded-xl border border-slate-700/50">
                      <p className="text-slate-500 text-xs mb-1">Duration</p>
                      <p className="text-slate-200 font-semibold">{videoInfo.duration_seconds?.toFixed(1)}s</p>
                    </div>
                    <div className="glass-panel p-3 rounded-xl border border-slate-700/50">
                      <p className="text-slate-500 text-xs mb-1">Processed</p>
                      <p className="text-amber-400 font-semibold">{frameScores.length}</p>
                    </div>
                    {finalResults && (
                      <div className={`glass-panel p-3 rounded-xl border border-slate-700/50 ${finalResults.anomaly_detected ? 'bg-red-500/10 border-red-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}>
                        <p className={`text-xs mb-1 ${finalResults.anomaly_detected ? 'text-red-400' : 'text-emerald-400'}`}>Status</p>
                        <p className={`font-semibold flex items-center gap-1 ${finalResults.anomaly_detected ? 'text-red-500' : 'text-emerald-500'}`}>
                          {finalResults.anomaly_detected ? <ShieldAlert size={14}/> : <Target size={14}/>}
                          {finalResults.anomaly_detected ? 'Anomaly' : 'Normal'}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {error && (
            <div className="glass-panel bg-red-500/10 border border-red-500/30 p-4 rounded-xl flex items-center gap-3 text-red-400">
              <AlertCircle size={20} className="shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LiveAnalysisPage;
