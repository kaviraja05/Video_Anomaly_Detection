import React from 'react';
import { motion } from 'framer-motion';

const AnimatedTechIllustration = () => {
  return (
    <div className="relative w-full aspect-square flex items-center justify-center p-8 max-w-[500px] mx-auto">
      {/* SVG Container */}
      <svg className="absolute inset-0 w-full h-full drop-shadow-[0_0_15px_rgba(59,130,246,0.5)]" viewBox="0 0 400 400">
        
        {/* Core Outer Ring */}
        <motion.circle
          cx="200" cy="200" r="140"
          stroke="url(#gradient-ring)" strokeWidth="1" strokeDasharray="6 6" fill="none"
          animate={{ rotate: 360 }}
          transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
          style={{ originX: '200px', originY: '200px' }}
        />
        
        {/* Core Inner Ring */}
        <motion.circle
          cx="200" cy="200" r="100"
          stroke="url(#gradient-ring-reverse)" strokeWidth="1.5" strokeDasharray="10 15" fill="none"
          animate={{ rotate: -360 }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
          style={{ originX: '200px', originY: '200px' }}
        />

        {/* Data Flow Lines */}
        {[
          "M 200,200 L 50,50", "M 200,200 L 350,50", 
          "M 200,200 L 50,350", "M 200,200 L 350,350",
          "M 200,200 L 200,20", "M 200,200 L 200,380",
          "M 200,200 L 20,200", "M 200,200 L 380,200"
        ].map((d, i) => (
          <motion.path
            key={`line-${i}`}
            d={d}
            stroke="rgba(96, 165, 250, 0.2)"
            strokeWidth="2"
            fill="none"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 2, repeat: Infinity, repeatType: "reverse", ease: "easeInOut", delay: i * 0.2 }}
          />
        ))}

        {/* Floating Data Packets */}
        {[
          { d: "M 200,200 L 50,50", delay: 0 },
          { d: "M 200,200 L 350,350", delay: 1 },
          { d: "M 200,200 L 350,50", delay: 0.5 },
          { d: "M 200,200 L 50,350", delay: 1.5 }
        ].map((path, i) => (
           <motion.circle
             key={`packet-${i}`}
             r="4"
             fill="#60A5FA"
             initial={{ offsetDistance: "0%" }}
             animate={{ offsetDistance: "100%" }}
             transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: path.delay }}
             style={{ offsetPath: `path('${path.d}')` }}
             className="shadow-[0_0_10px_#60A5FA]"
           />
        ))}

        <defs>
          <linearGradient id="gradient-ring" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#8B5CF6" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#10B981" stopOpacity="0.8" />
          </linearGradient>
          <linearGradient id="gradient-ring-reverse" x1="100%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#F43F5E" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#3B82F6" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0.8" />
          </linearGradient>
        </defs>

        {/* Central Core */}
        <motion.circle
          cx="200" cy="200" r="30"
          fill="rgba(59,130,246,0.1)"
          stroke="#3B82F6" strokeWidth="2"
          animate={{ scale: [1, 1.1, 1], opacity: [0.8, 1, 0.8] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle
           cx="200" cy="200" r="15"
           fill="#60A5FA"
           className="drop-shadow-[0_0_10px_#60A5FA]"
           animate={{ scale: [1, 1.3, 1] }}
           transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
        />

      </svg>
      
      {/* Glow Effects */}
      <motion.div 
        className="absolute inset-0 bg-blue-500/20 rounded-full blur-[80px]"
        animate={{ scale: [0.8, 1.2, 0.8], opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
};

export default AnimatedTechIllustration;
