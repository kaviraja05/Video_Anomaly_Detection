import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle, AlertCircle, Loader2, Video, X, Target, Zap, Search } from 'lucide-react';
import { uploadVideo } from '../api/api';

const UploadPage = ({ onAnalysisComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      setError('Please upload a valid video file (MP4, AVI, MOV, or MKV)');
      return;
    }

    // Validate file size (max 500MB)
    if (file.size > 500 * 1024 * 1024) {
      setError('File size must be less than 500MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setSuccess(false);
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file first');
      return;
    }

    setUploading(true);
    setError(null);
    setProgress(0);

    try {
      // Use real upload progress
      const uploadPromise = uploadVideo(selectedFile, (uploadProgress) => {
        setProgress(Math.min(uploadProgress / 2, 50));
      });
      
      // Analysis takes a while on CPU, simulate progress smoothly from 50% to 95%
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev < 50) return prev;
          if (prev >= 95) return 95;
          return prev + 1;
        });
      }, 2000); // +1% every 2 seconds

      const result = await uploadPromise;
      clearInterval(progressInterval);
      
      // Analysis phase logic
      setProgress(75);
      
      // Frame extraction
      const fps = result.video_info?.fps || 30.0;
      const totalFrames = result.video_info?.total_frames || result.frame_scores?.length || 100;
      
      const topFrames = (result.frame_scores || [])
        .map((score, index) => ({ frame: index, score }))
        .filter(item => item.score > 0.5)
        .sort((a, b) => b.score - a.score)
        .slice(0, 8)
        .map(item => item.frame);
        
      const frameImages = await extractAnomalyFrames(selectedFile, topFrames, fps, totalFrames);
      result.frame_images = frameImages;
      
      setProgress(100);
      setSuccess(true);
      
      setTimeout(() => {
        onAnalysisComplete(result);
      }, 1500);
      
    } catch (err) {
      console.error('Upload error:', err);
      let errorMessage = err.message || 'Upload failed. Please try again.';
      
      if (errorMessage.includes('backend') || errorMessage.includes('connect')) {
        errorMessage += '\n\nPlease ensure the backend server is running on http://localhost:8000';
      }
      
      setError(errorMessage);
      setProgress(0);
    } finally {
      if (!success) setUploading(false);
    }
  };

  const extractAnomalyFrames = async (videoFile, frameIndices, fps, totalFrames) => {
    return new Promise((resolve) => {
      const videoUrl = URL.createObjectURL(videoFile);
      const videoElement = document.createElement('video');
      videoElement.src = videoUrl;
      videoElement.muted = true;
      videoElement.setAttribute('playsinline', '');

      const frameImages = {};
      let currentIndex = 0;

      if (!frameIndices || frameIndices.length === 0) {
        URL.revokeObjectURL(videoUrl);
        resolve(frameImages);
        return;
      }

      videoElement.onloadedmetadata = () => {
        const duration = videoElement.duration || (totalFrames / (fps || 30));
        let actualFps = fps || 30;
        if (totalFrames && duration > 0 && duration !== Infinity) {
           actualFps = totalFrames / duration;
        }

        const captureFrame = () => {
          if (currentIndex >= frameIndices.length) {
            URL.revokeObjectURL(videoUrl);
            resolve(frameImages);
            return;
          }

          const targetFrame = frameIndices[currentIndex];
          const targetTime = targetFrame / actualFps;
          
          videoElement.currentTime = Math.min(targetTime, duration - 0.1);
        };

        videoElement.onseeked = () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.videoWidth || 640;
            canvas.height = videoElement.videoHeight || 360;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            frameImages[frameIndices[currentIndex]] = canvas.toDataURL('image/jpeg', 0.5);
          } catch(e) {
            console.error('Canvas extract error:', e);
          }
          currentIndex++;
          captureFrame();
        };

        captureFrame();
      };

      videoElement.onerror = () => {
        URL.revokeObjectURL(videoUrl);
        resolve({});
      };
    });
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="w-full max-w-4xl mx-auto relative z-10">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <h1 className="text-3xl md:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400 tracking-tight mb-3">
          Upload Video for Analysis
        </h1>
        <p className="text-slate-400 text-lg">
          Upload your surveillance footage to detect anomalies using our advanced AI system.
        </p>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-8 relative overflow-hidden"
      >
        {/* Decorative corner glow */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-[80px] pointer-events-none"></div>

        {/* Drag and Drop Zone */}
        <div
          className={`relative border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 ease-in-out
            ${dragActive ? 'border-blue-400 bg-blue-500/10 shadow-[0_0_30px_rgba(59,130,246,0.2)]' : 'border-slate-700 hover:border-slate-500 bg-slate-900/50'}
            ${selectedFile ? 'border-emerald-500/50 hover:border-emerald-400' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <AnimatePresence mode="wait">
            {!selectedFile ? (
              <motion.div 
                key="upload-prompt"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center space-y-4"
              >
                <div className="p-5 rounded-full bg-slate-800 text-blue-400 mb-2 shadow-inner group-hover:scale-110 transition-transform">
                  <UploadCloud size={48} strokeWidth={1.5} />
                </div>
                <h3 className="text-xl font-semibold text-slate-200">Drag & Drop your video here</h3>
                <p className="text-slate-500 text-sm">or</p>
                <label className="cursor-pointer inline-flex items-center px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium transition-all shadow-lg hover:shadow-blue-500/25 active:scale-95">
                  <input
                    type="file"
                    className="hidden"
                    accept="video/mp4,video/avi,video/mov,video/mkv"
                    onChange={handleFileInput}
                    disabled={uploading}
                  />
                  Browse Files
                </label>
                <p className="text-xs text-slate-500 mt-4">
                  Supported formats: MP4, AVI, MOV, MKV (Max 500MB)
                </p>
              </motion.div>
            ) : (
              <motion.div 
                key="file-info"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex flex-col md:flex-row items-center justify-between bg-slate-800/80 rounded-xl p-6 border border-slate-700"
              >
                <div className="flex items-center gap-4 mb-4 md:mb-0">
                  <div className="p-3 bg-emerald-500/20 text-emerald-400 rounded-lg shrink-0">
                    <Video size={32} />
                  </div>
                  <div className="text-left text-ellipsis overflow-hidden max-w-[200px] md:max-w-md">
                    <h3 className="text-slate-200 font-semibold truncate" title={selectedFile.name}>{selectedFile.name}</h3>
                    <p className="text-slate-400 text-sm">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>

                {!uploading && !success && (
                  <button
                    className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors flex items-center gap-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                      setError(null);
                    }}
                  >
                    <X size={20} />
                    <span className="md:hidden">Remove</span>
                  </button>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Progress & Status Area */}
        <AnimatePresence>
          {uploading && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              className="mt-6 space-y-2 overflow-hidden"
            >
              <div className="flex justify-between text-sm text-slate-300 mb-1">
                <span className="flex items-center gap-2">
                  <Loader2 size={16} className="animate-spin text-blue-400" />
                  {progress < 50 ? 'Uploading video...' : 'Analyzing frames...'}
                </span>
                <span className="font-mono text-blue-400">{Math.round(progress)}%</span>
              </div>
              <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                <motion.div 
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 relative"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }}
                >
                  {/* Shimmer effect inside progress bar */}
                  <div className="absolute inset-0 bg-white/20 w-32 blur-[10px] shimmer-animation"></div>
                </motion.div>
              </div>
            </motion.div>
          )}

          {success && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 flex items-center justify-center gap-3 p-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-xl"
            >
              <CheckCircle size={24} />
              <span className="font-medium">Analysis complete! Redirecting to results...</span>
            </motion.div>
          )}

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-left"
            >
              <AlertCircle size={20} className="shrink-0 mt-0.5" />
              <div className="text-sm whitespace-pre-line leading-relaxed">{error}</div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom Actions */}
        {selectedFile && !success && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-8 flex justify-end gap-4"
          >
            <button
              className="px-6 py-2.5 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors disabled:opacity-50"
              onClick={() => {
                setSelectedFile(null);
                setError(null);
                setProgress(0);
              }}
              disabled={uploading}
            >
              Cancel
            </button>
            <button
              className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium flex items-center gap-2 transition-all shadow-lg hover:shadow-blue-500/25 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Search size={18} />
                  Analyze Video
                </>
              )}
            </button>
          </motion.div>
        )}
      </motion.div>

      {/* Features Grid below upload */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12"
      >
        <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 backdrop-blur-sm text-center">
          <Target size={32} className="mx-auto text-blue-400 mb-4" />
          <h4 className="text-slate-200 font-semibold mb-2">High Accuracy</h4>
          <p className="text-sm text-slate-500">92% anomaly detection accuracy using advanced Hybrid AI architecture</p>
        </div>
        <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 backdrop-blur-sm text-center">
          <Zap size={32} className="mx-auto text-yellow-400 mb-4" />
          <h4 className="text-slate-200 font-semibold mb-2">Fast Processing</h4>
          <p className="text-sm text-slate-500">Optimized inference engines deliver rapid offline analysis</p>
        </div>
        <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 backdrop-blur-sm text-center">
          <Search size={32} className="mx-auto text-purple-400 mb-4" />
          <h4 className="text-slate-200 font-semibold mb-2">Detailed Insights</h4>
          <p className="text-sm text-slate-500">Comprehensive frame-level analysis with Explainable AI transparency</p>
        </div>
      </motion.div>

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
        .shimmer-animation {
          animation: shimmer 1.5s infinite linear;
        }
      `}</style>
    </div>
  );
};

export default UploadPage;
