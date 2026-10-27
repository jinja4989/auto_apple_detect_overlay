@echo off
rem Apple game helper - one-click setup (Windows 10/11)
rem Installs Python (if missing) and the required packages.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [1/2] Python not found. Installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo Automatic install failed. Install Python from https://www.python.org/downloads/
        echo and CHECK "Add python.exe to PATH" in the installer, then run this file again.
        pause
        exit /b 1
    )
    echo.
    echo Python installed. Please CLOSE this window and run this file once more
    echo so the new PATH is applied.
    pause
    exit /b 0
)

echo [2/2] Installing packages...
python -m pip install --upgrade pip
python -m pip install pyautogui opencv-python numpy
if errorlevel 1 (
    echo.
    echo Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Setup complete. Next: python sum10_pair_finder_v2.py --calibrate
pause
