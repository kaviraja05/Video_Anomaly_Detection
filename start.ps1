# Video Anomaly Detection - Quick Start Script
# This script starts both backend and frontend servers

Write-Host "🎥 Starting Video Anomaly Detection System..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Check if backend dependencies are installed
Write-Host "🔍 Checking backend dependencies..." -ForegroundColor Green
$torchInstalled = python -c "import torch; print('ok')" 2>$null
if ($torchInstalled -ne "ok") {
    Write-Host "❌ PyTorch not installed!" -ForegroundColor Red
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Start backend server in background
Write-Host "🚀 Starting Backend Server (Port 8000)..." -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & .\venv\Scripts\python.exe -m uvicorn backend_api:app --host 0.0.0.0 --port 8000
}

# Wait for backend to start
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "✅ Backend server is running!" -ForegroundColor Green
    Write-Host "   URL: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Backend failed to start!" -ForegroundColor Red
    Write-Host "Check logs for errors." -ForegroundColor Yellow
    Stop-Job $backendJob
    Remove-Job $backendJob
    exit 1
}

Write-Host ""

# Check if frontend directory exists
if (Test-Path "frontend") {
    # Check if node_modules exists
    if (-not (Test-Path "frontend\node_modules")) {
        Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Green
        Set-Location frontend
        npm install
        Set-Location ..
    }
    
    # Start frontend server
    Write-Host "🎨 Starting Frontend (Port 3000)..." -ForegroundColor Green
    Set-Location frontend
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm start"
    Set-Location ..
    
    Write-Host "✅ Frontend server is starting..." -ForegroundColor Green
    Write-Host "   URL: http://localhost:3000" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  Frontend directory not found, skipping..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🎉 System is running!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📱 Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "🔧 Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "📚 API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop all servers" -ForegroundColor Yellow
Write-Host ""

# Keep script running to maintain backend job
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "🛑 Stopping servers..." -ForegroundColor Red
    Stop-Job $backendJob
    Remove-Job $backendJob
    Write-Host "✅ Cleanup complete" -ForegroundColor Green
}
