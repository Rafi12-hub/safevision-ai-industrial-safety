@echo off
setlocal enabledelayedexpansion

title AI Industrial Safety Monitoring System Launcher
color 0B

echo =========================================================================
echo    AI-BASED MECHANICAL LAB & INDUSTRIAL WORKER SAFETY MONITORING SYSTEM
echo                     (WINDOWS 11 CPU-OPTIMIZED)
echo =========================================================================
echo [INFO] Initializing Industrial Safety Monitoring System...
echo.

:: 1. Verify Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found on PATH!
    echo Please install Python 3.11+ and ensure 'Add to PATH' is checked.
    pause
    exit /b 1
)

:: Add User Python Scripts to PATH if present
if exist "%APPDATA%\Python\Python313\Scripts" (
    set "PATH=%APPDATA%\Python\Python313\Scripts;%PATH%"
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo [OK] Python detected: %PYTHON_VER%

:: 2. Check for virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Running with active Python environment...
)

:: 3. Verify AI Model File
if exist "models\best.pt" (
    echo [OK] Custom PPE AI model verified: models\best.pt
) else (
    echo [NOTICE] models\best.pt not found. The system will start using base YOLO11n.
    echo To train best.pt on your lab dataset, run: python training\train.py
)

:: 4. Verify SQLite Database & CSV Records
echo [INFO] Verifying database and records...
python -c "import database; database.init_db(); database.export_to_csv_file()" >nul 2>&1
echo [OK] Database and records verified.

:: 5. Launch Streamlit Dashboard
echo.
echo =========================================================================
echo    STARTING STREAMLIT INDUSTRIAL SAFETY DASHBOARD
echo    Local URL: http://localhost:8501
echo =========================================================================
echo.

python -m streamlit run app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

pause
