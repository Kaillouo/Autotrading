@echo off
echo Setting up Windows Task Scheduler for trading pipeline...

set PROJECT_DIR=C:\Users\chunk\AgentWorkspace\Autotrading
set PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe
set SCRIPT=%PROJECT_DIR%\pipeline.py

schtasks /create /tn "TradingBot_Pipeline" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc minute /mo 15 /st 00:00 /sd 01/01/2024 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Done. Pipeline will run every 15 minutes when PC is on.
    echo.
    echo Useful commands:
    echo   View task:  schtasks /query /tn "TradingBot_Pipeline" /v /fo LIST
    echo   Run now:    schtasks /run /tn "TradingBot_Pipeline"
    echo   Stop task:  scripts\stop_scheduler.bat
) else (
    echo.
    echo ERROR: Task creation failed. Try running this script as Administrator.
)

pause
