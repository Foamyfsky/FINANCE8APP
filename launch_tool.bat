@echo off
title Volatility Regime Router - Local Server
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python from python.org.
    pause
    exit /b
)

:: Pick a free port (default 8765)
set PORT=8765

echo.
echo  =========================================
echo   Volatility Regime Router  ^|  DATA3888
echo  =========================================
echo.
echo  Starting local server on http://localhost:%PORT%
echo  Opening tool in Chrome...
echo.
echo  Press Ctrl+C in this window to stop the server.
echo.

:: Open Chrome after a short delay (give the server a moment to start)
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start chrome http://localhost:%PORT%/final_tool_v4.html"

:: Start the HTTP server (this blocks until Ctrl+C)
python -m http.server %PORT%

echo.
echo  Server stopped.
pause
