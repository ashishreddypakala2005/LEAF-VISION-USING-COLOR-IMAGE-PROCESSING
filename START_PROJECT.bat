@echo off
cd /d "%~dp0"
title LeafVision - Leaf Disease Segmentation
echo Installing project requirements...
py -m pip install -r requirements.txt
if errorlevel 1 (echo Installation failed. Check Python and internet. & pause & exit /b 1)
py -m streamlit run app.py
pause
