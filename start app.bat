@echo off
title Heart Risk Predictor
echo ==============================================
echo        Heart Risk Predictor Initializer
echo ==============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit
)

echo Checking dependencies...
python -m pip install -r requirements.txt

echo.
echo Starting the application...
python -m streamlit run app.py

pause
