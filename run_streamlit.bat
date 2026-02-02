@echo off
set "SCRIPT=%~dp0streamlit_dashboard.py"
start "" streamlit run "%SCRIPT%"
