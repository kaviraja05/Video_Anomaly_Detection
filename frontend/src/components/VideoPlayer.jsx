import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, Maximize2 } from 'lucide-react';

const VideoPlayer = ({ videoFile, currentFrame, totalFrames, fps = 30, onTimeUpdate, onEnded, onPlay, onPause }) => {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoUrl, setVideoUrl] = useState(null);

  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [videoFile]);

  useEffect(() => {
    if (videoRef.current && currentFrame !== undefined && totalFrames && fps) {
      const time = currentFrame / fps;
      if (Math.abs(videoRef.current.currentTime - time) > 0.5) {
        videoRef.current.currentTime = time;
      }
    }
  }, [currentFrame, totalFrames, fps]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);
      const frame = Math.floor(time * fps);
      if (onTimeUpdate) onTimeUpdate(time, frame);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) videoRef.current.pause();
      else videoRef.current.play();
      setIsPlaying(!isPlaying);
    }
  };

  const skip = (seconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
    }
  };

  const handleSeek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    if (videoRef.current) videoRef.current.currentTime = pos * duration;
  };

  const toggleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) videoRef.current.requestFullscreen();
      else if (videoRef.current.webkitRequestFullscreen) videoRef.current.webkitRequestFullscreen();
    }
  };

  const formatTime = (seconds) => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="relative w-full rounded-xl overflow-hidden bg-black aspect-video flex flex-col group/player border border-slate-700/50">
      {/* Video Element */}
      {videoUrl ? (
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-full object-contain cursor-pointer"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => { setIsPlaying(false); if (onEnded) onEnded(); }}
          onClick={togglePlay}
          onPlay={() => { if (onPlay) onPlay(); }}
          onPause={() => { if (onPause) onPause(); }}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-slate-900">
          <p className="text-slate-500">No video loaded</p>
        </div>
      )}

      {/* Center Play Overlay */}
      {videoUrl && !isPlaying && (
        <div 
          className="absolute inset-0 flex items-center justify-center bg-black/40 cursor-pointer backdrop-blur-[2px] transition-all"
          onClick={togglePlay}
        >
          <div className="p-4 rounded-full bg-blue-600/80 text-white backdrop-blur-md shadow-[0_0_30px_rgba(37,99,235,0.5)] transform transition-transform hover:scale-110">
            <Play size={40} className="ml-1" />
          </div>
        </div>
      )}

      {/* Standard Controls (Show on hover or pause) */}
      <div className={`absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col pt-12 pb-2 px-4 transition-opacity duration-300 ${isPlaying ? 'opacity-0 group-hover/player:opacity-100' : 'opacity-100'}`}>
        
        {/* Scrubber */}
        <div 
          className="w-full h-1.5 bg-slate-600/50 rounded-full mb-3 cursor-pointer relative group/scrubber"
          onClick={handleSeek}
        >
          {/* Progress */}
          <div 
            className="absolute top-0 left-0 h-full bg-blue-500 rounded-full group-hover/scrubber:bg-blue-400 pointer-events-none"
            style={{ width: `${(currentTime / duration) * 100 || 0}%` }}
          >
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow opacity-0 group-hover/scrubber:opacity-100 transform translate-x-1/2 transition-opacity"></div>
          </div>
        </div>

        {/* Buttons Row */}
        <div className="flex items-center justify-between text-slate-200">
          <div className="flex items-center gap-4">
            <button onClick={() => skip(-5)} className="hover:text-white transition-colors"><SkipBack size={18} /></button>
            <button onClick={togglePlay} className="hover:text-white transition-colors">
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <button onClick={() => skip(5)} className="hover:text-white transition-colors"><SkipForward size={18} /></button>

            <span className="text-xs font-mono tracking-wider text-slate-300 ml-2">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center gap-4">
            {currentFrame !== undefined && (
              <span className="text-xs font-mono text-blue-400 bg-blue-900/30 px-2 py-1 rounded border border-blue-900/50">
                FR: {currentFrame || 0}
              </span>
            )}
            <button onClick={toggleFullscreen} className="hover:text-white transition-colors"><Maximize2 size={16} /></button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
