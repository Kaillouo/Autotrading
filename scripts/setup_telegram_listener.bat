@echo off
echo Setting up Telegram listener scheduled task (every 1 minute)...

set PROJECT_DIR=C:\Users\chunk\AgentWorkspace\Autotrading
set BAT=%PROJECT_DIR%\scripts\telegram_listener.bat

schtasks /create /tn "TradingBot_TelegramListener" /tr "\"%BAT%\"" /sc minute /mo 1 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Done. Telegram listener will run every 1 minute.
    echo Commands available: /history [N], /status
) else (
    echo.
    echo ERROR: Task creation failed. Try running as Administrator.
)

pause
