@echo off
setlocal

REM Always execute inside the deliverable root folder regardless of caller CWD
cd /d "%~dp0"

echo ========================================================================
echo  AEGIS TACTICAL DISASTER RESPONSE COMMAND TERMINAL (PORTION 5 LAUNCHER)
echo ========================================================================

set "REQUIRED_IMPORTS=import fastapi, uvicorn, pydantic, python_multipart, numpy, PIL, requests, dotenv, networkx, gymnasium, stable_baselines3"
set "PY_CMD="

REM Option 1: Reuse existing local .venv if all required packages import cleanly
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "%REQUIRED_IMPORTS%" >nul 2>&1
    if not errorlevel 1 (
        echo [SETUP] Using existing .venv virtual environment...
        set "PY_CMD=.venv\Scripts\python.exe"
    )
)

REM Locate system Python (python or py -3) for Option 2 or Option 3
set "HOST_PY="
if not defined PY_CMD (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "HOST_PY=python"
    if not defined HOST_PY (
        py -3 -c "import sys" >nul 2>&1
        if not errorlevel 1 set "HOST_PY=py -3"
    )
    if not defined HOST_PY (
        echo [ERROR] Python 3.10+ was not found in PATH or via "py -3".
        echo         Install Python from https://www.python.org/downloads/ and re-run run_demo.bat.
        pause
        exit /b 1
    )
)

REM Option 2: Reuse system Python directly if dependencies are already installed (no download/venv needed)
if not defined PY_CMD (
    %HOST_PY% -c "%REQUIRED_IMPORTS%" >nul 2>&1
    if not errorlevel 1 (
        echo [SETUP] Using existing system Python with pre-installed dependencies...
        set "PY_CMD=%HOST_PY%"
    )
)

REM Option 3: Create .venv and install dependencies from requirements.txt
if not defined PY_CMD (
    if not exist ".venv\Scripts\python.exe" (
        echo [SETUP] Creating .venv virtual environment...
        call %HOST_PY% -m venv .venv
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment.
            pause
            exit /b 1
        )
    )
    set "PY_CMD=.venv\Scripts\python.exe"
    echo [SETUP] Installing required dependencies from requirements.txt...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check Python/network access and try again.
        pause
        exit /b 1
    )
)

set "FORCE_SEED=0"
if /i "%~1"=="--reset" set "FORCE_SEED=1"

if "%FORCE_SEED%"=="1" (
    echo [DB] Resetting data\poc_mission.db baseline database...
    %PY_CMD% seed.py
    if errorlevel 1 (
        echo [ERROR] Failed to reset database.
        pause
        exit /b 1
    )
) else if not exist "data\poc_mission.db" (
    echo [DB] Initializing data\poc_mission.db baseline database...
    %PY_CMD% seed.py
    if errorlevel 1 (
        echo [ERROR] Failed to initialize database.
        pause
        exit /b 1
    )
)

echo [START] Starting FastAPI Tactical Command Console (default http://127.0.0.1:8000)...
%PY_CMD% main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
    exit /b 1
)

endlocal
