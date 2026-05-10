import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';

const AnimatedTechIllustration = () => {
  const { theme } = useTheme();

  // Theme-based colors
  const primaryColor = theme === 'dark' ? '#3B82F6' : '#2563eb';
  const secondaryColor = theme === 'dark' ? '#8B5CF6' : '#7c3aed';
  const glowColor = theme === 'dark' ? 'rgba(59,130,246,0.6)' : 'rgba(37,99,235,0.4)';

  return (
    <div className="relative w-full aspect-square flex items-center justify-center p-8 max-w-[500px] mx-auto" style={{ perspective: '1000px' }}>
      
      {/* 3D Container */}
      <motion.div 
        className="relative w-full h-full flex items-center justify-center"
        style={{ transformStyle: 'preserve-3d' }}
        animate={{ rotateX: [15, 25, 15], rotateY: [0, 360] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      >
        {/* Core Sphere / Core Element */}
        <motion.div 
          className="absolute w-24 h-24 rounded-full border border-theme-accent backdrop-blur-md"
          style={{ 
            background: `radial-gradient(circle at 30% 30%, ${primaryColor}40, transparent)`,
            boxShadow: `0 0 30px ${glowColor}`
          }}
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
        />

        <motion.div 
          className="absolute w-12 h-12 rounded-full"
          style={{ background: primaryColor, boxShadow: `0 0 20px ${glowColor}` }}
          animate={{ scale: [1, 1.3, 1], opacity: [0.8, 1, 0.8] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* Orbiting Rings */}
        <motion.div 
          className="absolute w-56 h-56 rounded-full border-t-2 border-r-2"
          style={{ borderColor: primaryColor }}
          animate={{ rotateZ: 360, rotateX: 60 }}
          transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute w-72 h-72 rounded-full border-b-2 border-l-2"
          style={{ borderColor: secondaryColor }}
          animate={{ rotateZ: -360, rotateY: 60 }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute w-96 h-96 rounded-full border-t border-b border-dashed"
          style={{ borderColor: primaryColor, opacity: 0.5 }}
          animate={{ rotateZ: 360, rotateX: 75, rotateY: 45 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />

        {/* Floating Data Nodes */}
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            style={{ 
              width: i % 2 === 0 ? '8px' : '12px',
              height: i % 2 === 0 ? '8px' : '12px',
              borderRadius: '50%',
              backgroundColor: i % 3 === 0 ? secondaryColor : primaryColor,
              boxShadow: `0 0 15px ${glowColor}`,
            }}
            initial={{ rotate: i * 45, translateX: 110 + (i % 3) * 30 }}
            animate={{ rotate: i * 45 + 360 }}
            transition={{ duration: 8 + i * 1.5, repeat: Infinity, ease: "linear" }}
          />
        ))}
      </motion.div>

      {/* Central Glow Effects */}
      <div 
        className={`absolute inset-0 rounded-full blur-[100px] transition-all duration-1000 -z-10 ${
          theme === 'dark' 
            ? 'bg-blue-600/30 scale-110 animate-pulse' 
            : 'bg-blue-300/40 scale-100'
        }`}
      />
    </div>
  );
};

export default AnimatedTechIllustration;
