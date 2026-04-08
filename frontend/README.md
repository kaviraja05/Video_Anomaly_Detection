# Video Anomaly Detection - Frontend

Modern React application for video anomaly detection visualization with explainable AI.

---

## 🎨 Features

- ✅ Real-time system health monitoring
- 📊 Interactive anomaly score charts (Recharts)
- 🧠 Explainable AI visualization
- 📈 Memory bank statistics with circular charts
- 🎯 Responsive design (mobile-friendly)
- 🎨 Modern gradient UI with animations
- ⚡ Fast React 18 with functional components

---

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

---

## 🛠️ Tech Stack

### Core
- **React** 18.2.0
- **React DOM** 18.2.0
- **React Scripts** 5.0.1

### Data Visualization
- **Recharts** 2.7.2 - Line/Area charts for frame scores
- **Chart.js** 4.3.0 - Advanced charting
- **react-chartjs-2** 5.2.0 - Chart.js React wrapper

### HTTP & State
- **Axios** 1.4.0 - API communication
- **React Hooks** - State management (useState, useEffect)

### UI Components
- **React Icons** 4.10.1 - Modern icon library
- **Custom CSS** - Gradient animations, responsive grid

---

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html          # HTML template
│   ├── favicon.ico
│   ├── logo192.png
│   └── manifest.json
│
├── src/
│   ├── components/         # React components
│   │   ├── Header.js       # System health display
│   │   ├── Header.css
│   │   ├── Dashboard.js    # Model statistics
│   │   ├── Dashboard.css
│   │   ├── VideoAnalyzer.js # Analysis trigger
│   │   ├── VideoAnalyzer.css
│   │   ├── ResultsPanel.js  # Charts & results
│   │   ├── ResultsPanel.css
│   │   ├── ExplainabilityPanel.js # AI explanation
│   │   ├── ExplainabilityPanel.css
│   │   ├── StatsPanel.js    # Memory bank stats
│   │   └── StatsPanel.css
│   │
│   ├── api.js              # Axios API service
│   ├── App.js              # Main app component
│   ├── App.css             # Global app styles
│   ├── index.js            # React entry point
│   └── index.css           # Global CSS (gradient theme)
│
├── package.json            # Dependencies & scripts
└── README.md               # This file
```

---

## 🔌 API Integration

### API Service (src/api.js)
```javascript
import api from './api';

// Health check
const health = await api.healthCheck();

// Predict anomaly
const result = await api.predictAnomaly();

// Get memory stats
const stats = await api.getMemoryStats();

// Get model info
const info = await api.getModelInfo();
```

### Configuration
Change backend URL in `src/api.js`:
```javascript
const api = axios.create({
  baseURL: 'http://localhost:8000',  // Update this
  timeout: 30000,
});
```

---

## 🎨 Component Overview

### 1. Header
**Purpose**: System health indicator and branding  
**Props**: `systemHealth` (object)  
**Features**:
- Online/offline status indicator
- Device information display
- Pulsing animation for online status
- Responsive mobile layout

### 2. Dashboard
**Purpose**: Model statistics and architecture info  
**Props**: `modelInfo`, `systemHealth`  
**Features**:
- 4 stat cards (parameters, modules, features, device)
- Hover animations
- Architecture tags (DSM, RA²R, GNN, XAI)
- Auto-fit responsive grid

### 3. VideoAnalyzer
**Purpose**: Video analysis trigger interface  
**Props**: `onAnalyze`, `isLoading`, `error`  
**Features**:
- Large analyze button
- Loading spinner with pipeline steps
- Error display
- Progress bar animation

### 4. ResultsPanel
**Purpose**: Display analysis results with charts  
**Props**: `result` (object)  
**Features**:
- Area chart (Recharts) for frame scores
- Anomaly/Normal badge
- Video information grid
- Anomaly segments with severity badges
- Hover tooltips on chart

### 5. ExplainabilityPanel
**Purpose**: Show why anomalies were detected  
**Props**: `explanation` (object)  
**Features**:
- "Why detected" text explanation
- Contributing frames with progress bars
- Feature importance bar chart
- RA²R retrieval statistics

### 6. StatsPanel
**Purpose**: Memory bank & system metrics  
**Props**: `memoryStats` (object)  
**Features**:
- 3 gradient stat boxes
- Pattern distribution bars
- Circular progress charts (SVG)
- Cache hit rate & memory utilization

---

## 🎨 Styling Guide

### Color Variables
```css
/* Primary Colors */
--primary-purple: #667eea;
--primary-dark: #764ba2;
--gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Status Colors */
--success: #48bb78;
--warning: #ed8936;
--danger: #ef4444;
--info: #4299e1;

/* Neutral Colors */
--gray-50: #f7fafc;
--gray-100: #edf2f7;
--gray-700: #4a5568;
--gray-900: #2d3748;
```

### Animation Classes
```css
.fade-in {
  animation: fadeIn 0.6s ease-in;
}

.pulse {
  animation: pulse 2s ease-in-out infinite;
}

.slide-up {
  animation: slideUp 0.5s ease-out;
}
```

---

## 📱 Responsive Breakpoints

```css
/* Mobile */
@media (max-width: 768px) {
  /* Single column layout */
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}

/* Tablet */
@media (min-width: 769px) and (max-width: 1024px) {
  /* 2 columns */
  .dashboard-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* Desktop */
@media (min-width: 1025px) {
  /* Auto-fit with min 250px */
  .dashboard-grid {
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  }
}
```

---

## 🧪 Testing

### Run Tests
```bash
npm test
```

### Test Coverage
```bash
npm test -- --coverage
```

### Component Testing (Example)
```javascript
import { render, screen } from '@testing-library/react';
import Header from './components/Header';

test('renders system status', () => {
  const systemHealth = {
    model_loaded: true,
    device: 'cpu'
  };
  render(<Header systemHealth={systemHealth} />);
  const status = screen.getByText(/System Online/i);
  expect(status).toBeInTheDocument();
});
```

---

## 🚀 Build & Deploy

### Development Build
```bash
npm start
```
- Hot reload enabled
- Source maps included
- Opens at http://localhost:3000

### Production Build
```bash
npm run build
```
- Minified & optimized
- Output in `build/` directory
- Ready for static hosting

### Deploy to Static Hosting

#### Netlify
```bash
npm run build
npx netlify deploy --prod --dir=build
```

#### Vercel
```bash
npm run build
npx vercel --prod
```

#### Firebase Hosting
```bash
npm run build
firebase init hosting
firebase deploy
```

#### AWS S3 + CloudFront
```bash
npm run build
aws s3 sync build/ s3://your-bucket/
aws cloudfront create-invalidation --distribution-id ID --paths "/*"
```

---

## 🔧 Environment Variables

Create `.env` file:
```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENABLE_ANALYTICS=false
REACT_APP_VERSION=2.0.0
```

Access in code:
```javascript
const apiUrl = process.env.REACT_APP_API_URL;
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Issue: "Port 3000 already in use"
```bash
# Use different port (Windows)
set PORT=3001 && npm start

# Use different port (Linux/Mac)
PORT=3001 npm start
```

### Issue: "API connection refused"
- Ensure backend is running on port 8000
- Check `src/api.js` baseURL matches backend
- Verify CORS settings in backend

### Issue: "Build fails"
```bash
# Increase Node memory
set NODE_OPTIONS=--max_old_space_size=4096 && npm run build
```

---

## 📈 Performance Optimization

### Code Splitting
```javascript
// Lazy load components
import React, { lazy, Suspense } from 'react';

const ResultsPanel = lazy(() => import('./components/ResultsPanel'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ResultsPanel />
    </Suspense>
  );
}
```

### Memoization
```javascript
import React, { memo } from 'react';

const Dashboard = memo(({ modelInfo }) => {
  // Component code
});
```

### Image Optimization
- Use WebP format with fallbacks
- Implement lazy loading
- Add proper width/height attributes

---

## 🎯 Future Enhancements

### Planned Features
- [ ] Video upload & direct processing
- [ ] Historical analysis results dashboard
- [ ] Real-time video stream support
- [ ] User authentication & roles
- [ ] Export results to PDF/Excel
- [ ] Dark mode toggle
- [ ] Multi-language support

### Potential Libraries
- **React Query** - Advanced data fetching
- **Redux Toolkit** - Global state management
- **Formik** - Form handling
- **Framer Motion** - Advanced animations
- **React Table** - Data tables

---

## 📚 Resources

### Documentation
- [React Docs](https://react.dev/)
- [Recharts Guide](https://recharts.org/)
- [Axios Docs](https://axios-http.com/)
- [React Icons](https://react-icons.github.io/react-icons/)

### Tutorials
- Create React App: https://create-react-app.dev/
- React Hooks: https://react.dev/reference/react
- CSS Grid: https://css-tricks.com/snippets/css/complete-guide-grid/

---

## 🤝 Contributing

### Setup Development Environment
```bash
git clone <repo-url>
cd frontend
npm install
npm start
```

### Code Style
- Use functional components with hooks
- Follow ESLint rules (`npm run lint`)
- Use Prettier for formatting
- Write meaningful component names

### Pull Request Process
1. Create feature branch: `git checkout -b feature/AmazingFeature`
2. Make changes and test thoroughly
3. Run `npm test` to ensure tests pass
4. Commit: `git commit -m 'Add AmazingFeature'`
5. Push: `git push origin feature/AmazingFeature`
6. Open Pull Request

---

## 📄 License

This project is part of the Video Anomaly Detection System.  
See main README for license information.

---

## 📞 Support

For frontend-specific issues:
- Check browser console (F12)
- Review Network tab for API errors
- Check React component warnings

For general issues, see main project [README](../README_PRODUCTION.md).

---

**Built with React ⚛️ | Last Updated: January 2024**
