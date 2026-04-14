@echo off
echo Setting up morning report scheduled task...

set PROJECT_DIR=C:\Users\chunk\AgentWorkspace\Autotrading
set BAT=%PROJECT_DIR%\scripts\morning_report.bat

schtasks /create /tn "TradingBot_MorningReport" /tr "\"%BAT%\"" /sc daily /st 08:00 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Done. Morning report will run daily at 08:00.
) else (
    echo.
    echo ERROR: Task creation failed. Try running as Administrator.
)

pause
