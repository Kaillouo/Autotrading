@echo off
cd /d C:\Users\chunk\AgentWorkspace\Autotrading
venv\Scripts\python.exe -m src.ai.slow_brain --mode weekly >> logs\slow_brain.log 2>&1
