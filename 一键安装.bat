@echo off
chcp 65001 >nul
cd /d "%~dp0"
title AgroLake 环境准备
set "SETUP_ARGS=%*"
if "%~1"=="" set "SETUP_ARGS=--install-web"
if exist "%~dp0.venv-p0\Scripts\python.exe" goto existing
py -3.11 --version >nul 2>nul
if errorlevel 1 goto missing
py -3.11 -B "%~dp0scripts\local\setup.py" %SETUP_ARGS%
goto finished
:existing
"%~dp0.venv-p0\Scripts\python.exe" -B "%~dp0scripts\local\setup.py" %SETUP_ARGS%
goto finished
:missing
echo [错误] 缺少 Python 3.11。请按 docs\本地启动.md 指定已有3.11解释器。
set "EXIT_CODE=1"
goto done
:finished
set "EXIT_CODE=%ERRORLEVEL%"
:done
if not defined AGROLAKE_NO_PAUSE pause
exit /b %EXIT_CODE%
