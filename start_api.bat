@echo off
echo ======================================
echo FastAPI Preprocessing Proof Server
echo ======================================
echo.

cd /d "d:\Video_Anomaly_Detection\Video_Anomaly_Detection"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting FastAPI server...
echo.
echo API will be available at:
echo   - Local:   http://localhost:8001
echo   - Swagger: http://localhost:8001/docs
echo   - Endpoint: http://localhost:8001/preprocessing-proof
echo.

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001

pause
