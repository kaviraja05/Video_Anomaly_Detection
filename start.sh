#!/bin/bash

# Video Anomaly Detection - Quick Start Script (Linux/Mac)
# This script starts both backend and frontend servers

echo "🎥 Starting Video Anomaly Detection System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if backend dependencies are installed
echo "🔍 Checking backend dependencies..."
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start backend server in background
echo "🚀 Starting Backend Server (Port 8000)..."
uvicorn backend_api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backend server is running!"
    echo "   URL: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
else
    echo "❌ Backend failed to start!"
    kill $BACKEND_PID
    exit 1
fi

echo ""

# Check if frontend directory exists
if [ -d "frontend" ]; then
    # Check if node_modules exists
    if [ ! -d "frontend/node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        cd frontend
        npm install
        cd ..
    fi
    
    # Start frontend server
    echo "🎨 Starting Frontend (Port 3000)..."
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo "✅ Frontend server is starting..."
    echo "   URL: http://localhost:3000"
else
    echo "⚠️  Frontend directory not found, skipping..."
fi

echo ""
echo "════════════════════════════════════════════════"
echo "🎉 System is running!"
echo "════════════════════════════════════════════════"
echo ""
echo "📱 Frontend:  http://localhost:3000"
echo "🔧 Backend:   http://localhost:8000"
echo "📚 API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Trap Ctrl+C to cleanup
trap cleanup INT

cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Cleanup complete"
    exit 0
}

# Keep script running
wait
