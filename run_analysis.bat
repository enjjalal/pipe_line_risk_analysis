@echo off
REM Create a virtual environment
python -m venv .venv

REM Activate the virtual environment and install requirements
call .venv\Scripts\activate
pip install -r requirements.txt

REM Run the analysis script
python spatial_risk_analysis.py

REM Pause to see any error messages
pause
