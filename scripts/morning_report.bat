@echo off
cd /d C:\Users\chunk\AgentWorkspace\Autotrading
venv\Scripts\python.exe scripts\morning_report.py >> logs\telegram.log 2>&1
