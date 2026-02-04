@echo off
REM Monitor Training Script
REM This script monitors the training progress on AWS instance

echo ========================================
echo Seriguela Training Monitor
echo ========================================
echo.
echo Instance: 107.20.9.35
echo Key: C:\Users\madeinweb\chave-gpu.pem
echo.

:MENU
echo [1] View real-time training logs
echo [2] Check GPU status
echo [3] Check training progress (last 30 lines)
echo [4] Check process status
echo [5] List output directories
echo [6] Connect via SSH
echo [Q] Quit
echo.
set /p choice="Choose option: "

if /i "%choice%"=="1" goto LOGS
if /i "%choice%"=="2" goto GPU
if /i "%choice%"=="3" goto PROGRESS
if /i "%choice%"=="4" goto PROCESS
if /i "%choice%"=="5" goto OUTPUT
if /i "%choice%"=="6" goto SSH
if /i "%choice%"=="Q" goto END
goto MENU

:LOGS
echo.
echo === Real-time Training Logs (Ctrl+C to stop) ===
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@107.20.9.35 "tail -f ~/training_full.log"
goto MENU

:GPU
echo.
echo === GPU Status ===
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@107.20.9.35 "nvidia-smi"
echo.
pause
goto MENU

:PROGRESS
echo.
echo === Training Progress (last 30 lines) ===
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@107.20.9.35 "tail -30 ~/training_full.log"
echo.
pause
goto MENU

:PROCESS
echo.
echo === Process Status ===
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@107.20.9.35 "ps aux | grep -E 'python.*train' | grep -v grep"
echo.
pause
goto MENU

:OUTPUT
echo.
echo === Output Directories ===
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@107.20.9.35 "ls -lh ~/seriguela/output/"
echo.
pause
goto MENU

:SSH
echo.
echo === Connecting via SSH ===
echo To exit, type: exit
ssh -i C:\Users\madeinweb\chave-gpu.pem ubuntu@107.20.9.35
goto MENU

:END
echo.
echo Goodbye!
