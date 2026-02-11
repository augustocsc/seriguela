@echo off
REM Quick AWS Evaluation Script for Windows
REM Created: 2026-02-10

echo ============================================
echo Quick AWS Evaluation - Base vs Medium
echo ============================================
echo.

REM Convert to Unix-style script execution
bash evaluate_on_aws.sh

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! Evaluation complete.
echo ============================================
echo.
echo Check results in: evaluation_results_aws/
echo.
pause
