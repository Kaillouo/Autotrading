@echo off
echo Stopping TradingBot_Pipeline scheduled task...

schtasks /delete /tn "TradingBot_Pipeline" /f

if %ERRORLEVEL% EQU 0 (
    echo Scheduler stopped.
) else (
    echo ERROR: Could not delete task. It may not exist or you need Administrator rights.
)

pause
