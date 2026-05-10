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
    <div className="flex w-full h-full min-h-[calc(100vh-140px)] bg-theme-bg/50 overflow-hidden rounded-3xl border border-theme-border shadow-2xl backdrop-blur-xl transition-colors duration-300">
      {/* Left Side - Animation / Illustration (Hidden on mobile) */}
      <div className="hidden lg:flex flex-1 relative items-center justify-center bg-theme-bg/60 border-r border-theme-border overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-tr from-indigo-900/20 via-theme-bg to-blue-900/20 z-0 opacity-80" />
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
          className="bg-theme-card p-10 rounded-2xl border border-theme-border w-full max-w-md backdrop-blur-2xl shadow-[0_8px_32px_rgba(0,0,0,0.37)] relative z-10 my-auto transition-colors duration-300"
        >
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 text-theme-text text-glow">Create Account</h2>
            <p className="text-theme-text opacity-70">Start Intelligent Analysis</p>
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
              <label className="block text-theme-text opacity-80 mb-1.5 text-sm font-medium">Full Name</label>
              <input type="text" value={name} onChange={e => setName(e.target.value)} required className="w-full bg-theme-input border border-theme-border rounded-xl p-3 text-theme-text outline-none focus:border-theme-accent focus:ring-1 focus:ring-theme-accent transition-all" placeholder="John Doe" />
            </div>
            <div>
              <label className="block text-theme-text opacity-80 mb-1.5 text-sm font-medium">Email Address</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} required className="w-full bg-theme-input border border-theme-border rounded-xl p-3 text-theme-text outline-none focus:border-theme-accent focus:ring-1 focus:ring-theme-accent transition-all" placeholder="Enter your email" />
            </div>
            <div>
              <label className="block text-theme-text opacity-80 mb-1.5 text-sm font-medium">New Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} required className="w-full bg-theme-input border border-theme-border rounded-xl p-3 text-theme-text outline-none focus:border-theme-accent focus:ring-1 focus:ring-theme-accent transition-all" placeholder="••••••••" />
            </div>
            <div>
              <label className="block text-theme-text opacity-80 mb-1.5 text-sm font-medium">Confirm Password</label>
              <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required className="w-full bg-theme-input border border-theme-border rounded-xl p-3 text-theme-text outline-none focus:border-theme-accent focus:ring-1 focus:ring-theme-accent transition-all" placeholder="••••••••" />
            </div>
            <button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium py-3 rounded-xl transition-all shadow-lg hover:shadow-blue-500/25 mt-6">
              Register Account
            </button>
          </form>
          
          <div className="mt-8 text-center text-sm text-theme-text opacity-70">
            Already have an account? <button onClick={() => onNavigate('login')} className="text-theme-accent hover:opacity-80 hover:underline font-medium transition-colors">Log in</button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Register;
