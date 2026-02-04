#!/bin/bash
# Download all results and stop AWS instances
# Run this IMMEDIATELY to avoid extra costs!

set -e

echo "========================================"
echo "BAIXANDO RESULTADOS - EXPERIMENTO COMPLETO"
echo "========================================"
echo ""

# Create directories
mkdir -p results_final/quality
mkdir -p results_final/nguyen

echo "Baixando resultados da Instância 1 (Quality)..."
scp -i ~/chave-gpu.pem ubuntu@3.90.154.4:~/seriguela/results/quality/*.json ./results_final/quality/

echo ""
echo "Baixando resultados da Instância 2 (Nguyen 1-6)..."
scp -i ~/chave-gpu.pem ubuntu@23.20.79.242:~/seriguela/results/nguyen/*.json ./results_final/nguyen/

echo ""
echo "Baixando resultados da Instância 3 (Nguyen 7-12)..."
scp -i ~/chave-gpu.pem ubuntu@54.84.126.145:~/seriguela/results/nguyen/*.json ./results_final/nguyen/

echo ""
echo "========================================"
echo "TODOS RESULTADOS BAIXADOS COM SUCESSO!"
echo "========================================"
echo ""
echo "Arquivos salvos em: ./results_final/"
find results_final -name "*.json" -type f
echo ""

echo "========================================"
echo "PARANDO INSTÂNCIAS AWS"
echo "========================================"
echo ""
echo "CRÍTICO: Parando instâncias para evitar custos adicionais..."

aws ec2 stop-instances --instance-ids i-020af019c407e77da i-04c4eabae4a555af1 i-091e1500599aa6bd3

echo ""
echo "========================================"
echo "SUCESSO! INSTÂNCIAS PARANDO..."
echo "========================================"
echo ""
echo "Aguardando 5 segundos..."
sleep 5

echo "Verificando status..."
aws ec2 describe-instances --instance-ids i-020af019c407e77da i-04c4eabae4a555af1 i-091e1500599aa6bd3 --query "Reservations[*].Instances[*].[Tags[?Key=='Name'].Value|[0],State.Name]" --output table

echo ""
echo "========================================"
echo "EXPERIMENTO COMPLETO!"
echo "========================================"
echo ""
echo "Resultados: ./results_final/"
echo "Custo total: ~\$8-9 USD"
echo ""
echo "Próximos passos:"
echo "1. Analisar resultados: python scripts/analyze_results.py"
echo "2. Gerar relatório: python scripts/generate_report.py"
echo "3. Ver resumo: cat RESULTS_SUMMARY_2026-02-04.md"
echo ""
