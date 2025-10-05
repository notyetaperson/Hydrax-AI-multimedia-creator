@echo off
title Hydax AI - Complete Audio & Video Suite
echo.
echo ========================================
echo   Hydax AI - Complete Audio ^& Video Suite
echo ========================================
echo.
echo Launching desktop GUI...
echo.
python hydax_ai.py
if errorlevel 1 (
    echo.
    echo Error launching GUI!
    echo Please make sure Python and all dependencies are installed.
    echo Run: pip install -r requirements.txt
    echo.
    pause
)