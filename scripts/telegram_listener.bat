@echo off
cd /d C:\Users\chunk\AgentWorkspace\Autotrading
venv\Scripts\python.exe scripts\telegram_listener.py >> logs\telegram.log 2>&1
