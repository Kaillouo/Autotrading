@echo off
echo Setting up Slow Brain scheduled tasks...

schtasks /create /tn "TradingBot_SlowBrain_6h" /tr "C:\Users\chunk\AgentWorkspace\Autotrading\scripts\slow_brain_6h.bat" /sc hourly /mo 6 /f
schtasks /create /tn "TradingBot_SlowBrain_Weekly" /tr "C:\Users\chunk\AgentWorkspace\Autotrading\scripts\slow_brain_weekly.bat" /sc weekly /d SUN /st 10:00 /f

echo Slow Brain schedulers created.
pause
