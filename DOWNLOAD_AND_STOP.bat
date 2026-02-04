@echo off
REM Download all results and stop AWS instances
REM Run this IMMEDIATELY to avoid extra costs!

echo ========================================
echo BAIXANDO RESULTADOS - EXPERIMENTO COMPLETO
echo ========================================
echo.

REM Create directories
mkdir results_final\quality 2>nul
mkdir results_final\nguyen 2>nul

echo Baixando resultados da Instancia 1 (Quality)...
scp -i C:\Users\madeinweb\chave-gpu.pem ubuntu@3.90.154.4:~/seriguela/results/quality/*.json ./results_final/quality/
if %ERRORLEVEL% NEQ 0 (
    echo ERRO ao baixar da Instancia 1!
    pause
    exit /b 1
)

echo.
echo Baixando resultados da Instancia 2 (Nguyen 1-6)...
scp -i C:\Users\madeinweb\chave-gpu.pem ubuntu@23.20.79.242:~/seriguela/results/nguyen/*.json ./results_final/nguyen/inst2_
if %ERRORLEVEL% NEQ 0 (
    echo ERRO ao baixar da Instancia 2!
    pause
    exit /b 1
)

echo.
echo Baixando resultados da Instancia 3 (Nguyen 7-12)...
scp -i C:\Users\madeinweb\chave-gpu.pem ubuntu@54.84.126.145:~/seriguela/results/nguyen/*.json ./results_final/nguyen/inst3_
if %ERRORLEVEL% NEQ 0 (
    echo ERRO ao baixar da Instancia 3!
    pause
    exit /b 1
)

echo.
echo ========================================
echo TODOS RESULTADOS BAIXADOS COM SUCESSO!
echo ========================================
echo.
echo Arquivos salvos em: .\results_final\
dir /s /b results_final\*.json
echo.

echo ========================================
echo PARANDO INSTANCIAS AWS
echo ========================================
echo.
echo CRITICO: Parando instancias para evitar custos adicionais...

aws ec2 stop-instances --instance-ids i-020af019c407e77da i-04c4eabae4a555af1 i-091e1500599aa6bd3

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERRO AO PARAR INSTANCIAS!
    echo Por favor, pare manualmente via console AWS:
    echo https://console.aws.amazon.com/ec2/
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCESSO! INSTANCIAS PARADAS!
echo ========================================
echo.
echo Verificando status...
timeout /t 5 >nul

aws ec2 describe-instances --instance-ids i-020af019c407e77da i-04c4eabae4a555af1 i-091e1500599aa6bd3 --query "Reservations[*].Instances[*].[Tags[?Key=='Name'].Value|[0],State.Name]" --output table

echo.
echo ========================================
echo EXPERIMENTO COMPLETO!
echo ========================================
echo.
echo Resultados: .\results_final\
echo Custo total: ~$8-9 USD
echo.
echo Proximos passos:
echo 1. Analisar resultados: python scripts/analyze_results.py
echo 2. Gerar relatorio: python scripts/generate_report.py
echo 3. Ver resumo: type RESULTS_SUMMARY_2026-02-04.md
echo.
pause
