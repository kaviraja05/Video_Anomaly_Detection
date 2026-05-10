import React, { useState } from 'react';
import { loginUser } from '../api/api';
import { motion } from 'framer-motion';
import AnimatedTechIllustration from '../components/AnimatedTechIllustration';

const Login = ({ onLogin, onNavigate }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const data = await loginUser({ email, password });
      onLogin(data);
    } catch (err) {
      setError('Mail or password is wrong');
    }
  };

  return (
    <div className="flex w-full h-full min-h-[calc(100vh-140px)] bg-theme-bg/50 overflow-hidden rounded-3xl border border-theme-border shadow-2xl backdrop-blur-xl transition-colors duration-300">
      {/* Left Side - Animation / Illustration (Hidden on mobile) */}
      <div className="hidden lg:flex flex-1 relative items-center justify-center bg-theme-bg/60 border-r border-theme-border overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-theme-bg to-purple-900/20 z-0 opacity-80" />
        <div className="relative z-10 w-full max-w-xl">
          <AnimatedTechIllustration />
        </div>
      </div>

      {/* Right Side - Form */}
      <div className="flex-1 flex justify-center items-center p-8 relative">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="bg-theme-card p-10 rounded-2xl border border-theme-border w-full max-w-md backdrop-blur-2xl shadow-[0_8px_32px_rgba(0,0,0,0.37)] relative z-10 transition-colors duration-300"
        >
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 text-theme-text text-glow">Welcome Back</h2>
            <p className="text-theme-text opacity-70">AI Powered Monitoring System</p>
          </div>
          
          {error && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-red-500/10 text-red-400 border border-red-500/20 p-4 rounded-xl mb-6 text-sm"
            >
              {error}
            </motion.div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5" autoComplete="off">
            <div>
              <label className="block text-theme-text opacity-80 mb-1.5 text-sm font-medium">Email Address</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} required autoComplete="new-email" className="w-full bg-theme-input border border-theme-border rounded-xl p-3 text-theme-text outline-none focus:border-theme-accent focus:ring-1 focus:ring-theme-accent transition-all" placeholder="Enter your email" />
            </div>
            <div>
              <label className="block text-theme-text opacity-80 mb-1.5 text-sm font-medium">Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} required autoComplete="new-password" className="w-full bg-theme-input border border-theme-border rounded-xl p-3 text-theme-text outline-none focus:border-theme-accent focus:ring-1 focus:ring-theme-accent transition-all" placeholder="••••••••" />
            </div>
            <button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium py-3 rounded-xl transition-all shadow-lg hover:shadow-blue-500/25 mt-4">
              Login to System
            </button>
          </form>
          
          <div className="mt-8 text-center text-sm text-theme-text opacity-70">
            Don't have an account? <button onClick={() => onNavigate('register')} className="text-theme-accent hover:opacity-80 hover:underline font-medium transition-colors">Register here</button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Login;
