@echo off
echo Starting PSAI Application Services...

:: Start Backend API
:: We activate the virtual environment first
:: We set PYTHONPATH to the parent directory so that 'import psai' works if running from source without installation
start "PSAI Backend API" cmd /k call "C:\Python-ML\python12-ml\Scripts\activate" ^&^& cd /d "%~dp0" ^&^& python run_backend.py

:: Start Frontend
start "PSAI Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Services started in separate windows.
