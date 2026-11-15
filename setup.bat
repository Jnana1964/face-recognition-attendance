@echo off
cd /d "%~dp0"
echo.
echo ===== FaceAttend setup =====
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python from python.org and TICK "Add python.exe to PATH".
  pause
  exit /b 1
)
echo [1/4] Creating virtual environment...
if not exist venv\Scripts\python.exe python -m venv venv
echo [2/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo [3/4] Installing libraries (takes a few minutes the first time)...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Library install FAILED. Check your internet and run setup.bat again.
  pause
  exit /b 1
)
echo [4/4] Checking everything...
venv\Scripts\python.exe download_models.py
venv\Scripts\python.exe check_system.py
echo.
echo Setup finished. Double-click run.bat to start the app.
pause
