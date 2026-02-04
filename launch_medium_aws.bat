@echo off
REM Lanca instancia AWS para treinar GPT-2 Medium

echo ========================================
echo Lancando AWS EC2 para GPT-2 Medium
echo ========================================
echo.

REM Converter script para formato Unix (necessario)
dos2unix scripts\aws\launch_medium_training.sh 2>nul

REM Tornar executavel
chmod +x scripts\aws\launch_medium_training.sh 2>nul

REM Pedir tokens se nao fornecidos
set /p WANDB_KEY="Digite sua WANDB API Key: "
set /p HF_TOKEN="Digite seu HuggingFace Token (ou Enter para pular): "

echo.
echo Lancando instancia g5.xlarge na AWS...
echo Isso vai treinar GPT-2 Medium (355M parametros)
echo.

bash scripts/aws/launch_medium_training.sh --wandb-key "%WANDB_KEY%" --hf-token "%HF_TOKEN%"

echo.
pause
