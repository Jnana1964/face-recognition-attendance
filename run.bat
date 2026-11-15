@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo First double-click setup.bat
  pause
  exit /b 1
)
rem Skip Streamlit's one-time "enter your email" prompt
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)
echo Starting FaceAttend... your browser will open at http://localhost:8501
echo Keep this window open. Close it to stop the app.
venv\Scripts\python.exe -m streamlit run app.py
pause
