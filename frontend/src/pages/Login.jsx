import React, { useState } from 'react';
import { loginUser } from '../api/api';

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
    <div className="flex justify-center items-center h-full">
      <div className="bg-slate-900/80 p-8 rounded-xl border border-slate-800 w-full max-w-md backdrop-blur-sm shadow-2xl relative z-10">
        <h2 className="text-2xl font-bold mb-6 text-center text-slate-100">Welcome Back</h2>
        {error && <div className="bg-red-500/20 text-red-300 border border-red-500/30 p-3 rounded mb-4 text-sm">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4" autoComplete="off">
          <div>
            <label className="block text-slate-400 mb-1 text-sm">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required autoComplete="new-email" className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 outline-none focus:border-blue-500 transition-colors" />
          </div>
          <div>
            <label className="block text-slate-400 mb-1 text-sm">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required autoComplete="new-password" className="w-full bg-slate-950 border border-slate-800 rounded p-2.5 text-slate-200 outline-none focus:border-blue-500 transition-colors" />
          </div>
          <button type="submit" className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded transition-colors mt-2 border border-blue-500/50">Login</button>
        </form>
        <div className="mt-6 text-center text-sm text-slate-400">
          Don't have an account? <button onClick={() => onNavigate('register')} className="text-blue-400 hover:text-blue-300 hover:underline">Register here</button>
        </div>
      </div>
    </div>
  );
};

export default Login;
