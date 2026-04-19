import React, { useState } from 'react';
import { registerUser } from '../api/api';
import { motion } from 'framer-motion';
import AnimatedTechIllustration from '../components/AnimatedTechIllustration';

const Register = ({ onRegister, onNavigate }) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    
    // Validation
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\W).{8,}$/;
    if (!passwordRegex.test(password)) {
      setError("Password must be at least 8 characters, include an uppercase letter, lowercase letter, and one special character.");
      return;
    }
    
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      await registerUser({ name, email, password });
      onRegister(); // Navigate to login
    } catch (err) {
      let errorMsg = err.message || 'Registration failed.';
      if (errorMsg.includes("already registered") || errorMsg.includes("400")) {
        errorMsg = "The user is already register";
      }
      setError(errorMsg);
    }
  };

  return (
    <div className="flex w-full h-full min-h-[calc(100vh-140px)] bg-slate-900/50 overflow-hidden rounded-3xl border border-slate-800/60 shadow-2xl backdrop-blur-xl">
      {/* Left Side - Animation / Illustration (Hidden on mobile) */}
      <div className="hidden lg:flex flex-1 relative items-center justify-center bg-slate-900/60 border-r border-slate-800/50 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-tr from-indigo-900/20 via-slate-900/80 to-blue-900/20 z-0" />
        <div className="relative z-10 w-full max-w-xl">
          <AnimatedTechIllustration />
        </div>
      </div>

      {/* Right Side - Form */}
      <div className="flex-1 flex justify-center items-center p-8 relative overflow-y-auto custom-scrollbar">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="bg-slate-900/40 p-10 rounded-2xl border border-slate-700/50 w-full max-w-md backdrop-blur-2xl shadow-[0_8px_32px_rgba(0,0,0,0.37)] relative z-10 my-auto"
        >
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 text-slate-100 text-glow">Create Account</h2>
            <p className="text-slate-400">Start Intelligent Monitoring</p>
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

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-slate-400 mb-1.5 text-sm font-medium">Full Name</label>
              <input type="text" value={name} onChange={e => setName(e.target.value)} required className="w-full bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-slate-200 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all" placeholder="John Doe" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1.5 text-sm font-medium">Email Address</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} required className="w-full bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-slate-200 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all" placeholder="Enter your email" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1.5 text-sm font-medium">New Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} required className="w-full bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-slate-200 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all" placeholder="••••••••" />
            </div>
            <div>
              <label className="block text-slate-400 mb-1.5 text-sm font-medium">Confirm Password</label>
              <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required className="w-full bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-slate-200 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all" placeholder="••••••••" />
            </div>
            <button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium py-3 rounded-xl transition-all shadow-lg hover:shadow-blue-500/25 mt-6">
              Register Account
            </button>
          </form>
          
          <div className="mt-8 text-center text-sm text-slate-400">
            Already have an account? <button onClick={() => onNavigate('login')} className="text-blue-400 hover:text-blue-300 hover:underline font-medium transition-colors">Log in</button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Register;
