@echo off
rem Starts the SVJSim web app. First run creates a private virtual environment
rem in .venv and installs SVJSim into it (needs internet and git, because the
rem SVJ parser is installed straight from GitHub). Later runs start at once.
rem
rem   start_gui.bat                  open the app, pick a vehicle in the page
rem   start_gui.bat car.svj.json     open the app with that vehicle loaded
rem   start_gui.bat --port 8800      any other `svjsim gui` option works too
rem
rem You can also drag a .svj.json file onto this file.

setlocal
cd /d "%~dp0"
title SVJSim

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    where python >nul 2>nul || (
        echo Python 3.11 or newer was not found on PATH.
        echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || (
        echo SVJSim needs Python 3.11 or newer.
        python --version
        pause
        exit /b 1
    )
    echo First run: creating a virtual environment in %VENV% ...
    python -m venv "%VENV%" || (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

"%PY%" -c "import svjsim, svj" >nul 2>nul || (
    echo Installing SVJSim and its dependencies - this needs internet and git ...
    "%PY%" -m pip install -e . || (
        echo.
        echo Install failed. Check your internet connection and that git is installed.
        pause
        exit /b 1
    )
)

echo Starting SVJSim. Close this window or press Ctrl+C to stop it.
"%PY%" -m svjsim.cli gui %*
if errorlevel 1 pause
endlocal
