@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0master.ps1" %*
endlocal
