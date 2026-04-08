import React, { useState } from 'react';
import { registerUser } from '../api/api';

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
    <div className="flex justify-center items-center h-full">
      <div className="bg-slate-900/80 p-8 rounded-xl border border-slate-800 w-full max-w-md backdrop-blur-sm shadow-2xl relative z-10">
        <h2 className="text-2xl font-bold mb-6 text-center text-slate-100">Create Account</h2>
        {error && <div className="bg-red-500/20 text-red-300 border border-red-500/30 p-3 rounded mb-4 text-sm">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-slate-400 mb-1 text-sm">Full Name</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} required className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 outline-none focus:border-blue-500 transition-colors" />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 text-sm">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 outline-none focus:border-blue-500 transition-colors" />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 text-sm">New Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 outline-none focus:border-blue-500 transition-colors" />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 text-sm">Confirm Password</label>
            <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 outline-none focus:border-blue-500 transition-colors" />
          </div>
          <button type="submit" className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded transition-colors mt-2 border border-blue-500/50">Register</button>
        </form>
        <div className="mt-6 text-center text-sm text-slate-400">
          Already have an account? <button onClick={() => onNavigate('login')} className="text-blue-400 hover:text-blue-300 hover:underline">Log in</button>
        </div>
      </div>
    </div>
  );
};

export default Register;
