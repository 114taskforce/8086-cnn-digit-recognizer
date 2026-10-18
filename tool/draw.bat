@echo off
cd /d %~dp0
python draw_digits.py
if errorlevel 1 pause
