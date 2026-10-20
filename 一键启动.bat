@echo off
chcp 65001 >nul
cd /d "%~dp0"
title AgroLake 本机启动
if not exist "%~dp0.venv-p0\Scripts\python.exe" goto missing
"%~dp0.venv-p0\Scripts\python.exe" -B "%~dp0scripts\local\startup.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto done
:missing
echo [错误] 未找到网页环境 .venv-p0。请先运行一键安装.bat，或按 docs\本地启动.md 准备。
set "EXIT_CODE=1"
:done
if not defined AGROLAKE_NO_PAUSE pause
exit /b %EXIT_CODE%
