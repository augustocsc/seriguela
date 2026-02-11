# Workflow de Avaliação AWS - Status em Tempo Real

**Iniciado**: 2026-02-10
**Objetivo**: Comparar modelos Base (124M) e Medium (355M) em instância AWS nova

## Status Atual

### Fase 1: Download dos Modelos ⏳ EM ANDAMENTO

- ✅ Instâncias Base e Medium iniciadas temporariamente
  - Base: i-03cb806bdc98e6d36 (34.201.9.89)
  - Medium: i-0567ed93f9e625a89 (100.48.68.1)

- ⏳ Download Base: EM ANDAMENTO (background task b77d5b3)
  - Origem: ubuntu@34.201.9.89:~/seriguela/output/gpt2_base_prefix_682k
  - Destino: ./output/gpt2_base_prefix_682k

- ⏳ Download Medium: EM ANDAMENTO (background task bca7ec9)
  - Origem: ubuntu@100.48.68.1:~/seriguela/output/gpt2_medium_prefix_682k
  - Destino: ./output/gpt2_medium_prefix_682k

### Próximas Fases

- ⏸️ Fase 2: Parar instâncias Base e Medium
- ⏸️ Fase 3: Lançar instância de avaliação (g5.xlarge)
- ⏸️ Fase 4: Upload dos modelos para instância de avaliação
- ⏸️ Fase 5: Executar avaliações
  - Validação rápida (5 amostras)
  - Métricas de qualidade (500 amostras)
  - Análise de complexidade (200 amostras)
  - Comparação Base vs Medium
- ⏸️ Fase 6: Download dos resultados
- ⏸️ Fase 7: Parar instância de avaliação

## Scripts Criados

1. **`evaluate_on_aws.sh`**: Workflow completo automatizado
2. **`quick_evaluate_aws.bat`**: Wrapper Windows para executar workflow
3. **`EVALUATION_WORKFLOW_STATUS.md`**: Este arquivo (status em tempo real)

## Comandos de Monitoramento

```bash
# Checar progresso dos downloads
tail -f C:/Users/MADEIN~1/AppData/Local/Temp/claude/C--Users-madeinweb-seriguela/tasks/b77d5b3.output
tail -f C:/Users/MADEIN~1/AppData/Local/Temp/claude/C--Users-madeinweb-seriguela/tasks/bca7ec9.output

# Verificar se modelos foram baixados
ls -lh ./output/gpt2_*_prefix_682k/

# Checar instâncias AWS
aws ec2 describe-instances --instance-ids i-03cb806bdc98e6d36 i-0567ed93f9e625a89 \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name]" --output table
```

## Custos Estimados

| Item | Tempo Estimado | Taxa | Custo |
|------|---------------|------|-------|
| Base/Medium temporários | ~10 min | $2.012/h | ~$0.34 |
| Download (local) | ~5-10 min | $0 | $0 |
| Instância de avaliação | ~1-2h | $1.006/h | ~$1-2 |
| **TOTAL ESTIMADO** | | | **~$1.50-2.50** |

## Timeline Esperado

- **19:56 UTC**: Iniciadas instâncias Base/Medium para download
- **20:06 UTC** (est.): Downloads completos, instâncias paradas
- **20:10 UTC** (est.): Instância de avaliação lançada
- **20:15 UTC** (est.): Modelos uploaded, avaliações iniciadas
- **21:30 UTC** (est.): Avaliações completas, resultados baixados
- **21:35 UTC** (est.): Instância de avaliação parada

**Duração total estimada**: ~1.5-2 horas

## Avaliações Planejadas

### 1. Validação Rápida
- 5 expressões geradas por modelo
- Verificar sintaxe e validade

### 2. Métricas de Qualidade (500 amostras cada)
- Taxa de validade (%)
- Taxa de parsing (%)
- Aderência a constraints (variáveis/operadores)
- Taxa de diversidade (expressões únicas)

### 3. Análise de Complexidade (200 amostras cada)
- Uso de operações de potência (x², x**n)
- Funções aninhadas (sin(cos(x)))
- Profundidade média
- Profundidade máxima
- Distribuição de operadores

### 4. Comparação Direta
- Base vs Medium no benchmark Nguyen-5
- Comparação estatística de métricas

### 5. Prefix vs Infix (se modelos infix disponíveis)
- Comparar notação prefixada vs infixada
- Identificar qual formato é superior

## Resultados Esperados

### Hipóteses

**H1**: Medium tem maior taxa de validade que Base
- Base: ~80% (baseline)
- Medium: ~85-90% (esperado)

**H2**: Medium gera expressões mais complexas
- Base: depth ~1.5, power ops ~15-20%
- Medium: depth ~2.0, power ops ~30-40%

**H3**: Medium performa melhor em Nguyen-5
- Base: R² ~-1.0 (baseline ruim)
- Medium: R² >0.0 (esperado melhoria)

## Notas

- Modelos serão baixados das instâncias originais de treinamento
- Instâncias Base/Medium serão paradas imediatamente após download
- Avaliação rodará em instância nova isolada
- Resultados salvos em: `./evaluation_results_aws/`

---

**Última atualização**: 2026-02-10 19:56 UTC
**Status**: Downloads em andamento
**Próximo passo**: Aguardar conclusão dos downloads (~5-10 min)
