/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#0f172a',
        cardBg: 'rgba(30, 41, 59, 0.7)',
        primaryAccent: '#3b82f6',
        secondaryAccent: '#8b5cf6',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        'theme-bg': 'var(--bg-color)',
        'theme-text': 'var(--text-color)',
        'theme-card': 'var(--card-bg)',
        'theme-border': 'var(--card-border)',
        'theme-input': 'var(--input-bg)',
        'theme-accent': 'var(--accent-color)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-dark': 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
      },
      boxShadow: {
        'glow': '0 0 15px rgba(59, 130, 246, 0.5)',
        'glow-success': '0 0 15px rgba(16, 185, 129, 0.5)',
        'glow-danger': '0 0 15px rgba(239, 68, 68, 0.5)',
      }
    },
  },
  plugins: [],
}
