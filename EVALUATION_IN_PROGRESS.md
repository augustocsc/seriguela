# Avaliação AWS em Andamento

**Hora**: 2026-02-10 20:07 UTC
**Status**: ⏳ Uploads em andamento

## Progresso

### ✅ Fase 1: Download dos Modelos - COMPLETO
- Base downloaded: 1.2MB adapter + checkpoints
- Medium downloaded: 3.1MB adapter + checkpoints
- Instâncias Base/Medium: STOPPED (economia ativa)

### ✅ Fase 2: Instância de Avaliação - LANÇADA
- **Instance ID**: i-0bfa29e0a4e501d09
- **IP**: 3.93.21.231
- **Tipo**: g5.xlarge (NVIDIA A10G, 24GB VRAM)
- **Status**: RUNNING
- **Setup**: Completo (repo clonado, deps instaladas)

### ⏳ Fase 3: Upload dos Modelos - EM ANDAMENTO
- **Base**: Upload iniciado (task b8c7966)
- **Medium**: Upload iniciado (task bdf6287)
- **Tempo estimado**: ~5-10 minutos

### ⏸️ Fase 4: Execução das Avaliações - PRÓXIMO
Será executado após uploads completos:

1. **Validação rápida** (5 expressões/modelo)
2. **Métricas de qualidade** (500 amostras/modelo)
   - Valid rate, constraint adherence, diversity
3. **Análise de complexidade** (200 amostras/modelo)
   - Power ops, nested functions, depth
4. **Comparação Base vs Medium** (Nguyen-5)

**Tempo estimado**: 1-2 horas

### ⏸️ Fase 5: Download de Resultados - PENDENTE

### ⏸️ Fase 6: Parada da Instância - PENDENTE

## Comandos de Monitoramento

```bash
# Checar uploads em progresso
tail -f C:/Users/MADEIN~1/AppData/Local/Temp/claude/C--Users-madeinweb-seriguela/tasks/b8c7966.output
tail -f C:/Users/MADEIN~1/AppData/Local/Temp/claude/C--Users-madeinweb-seriguela/tasks/bdf6287.output

# SSH na instância de avaliação
ssh -i C:/Users/madeinweb/chave-gpu.pem ubuntu@3.93.21.231

# Verificar modelos uploadados
ssh ubuntu@3.93.21.231 'ls -lh ~/seriguela/output/'

# Monitorar avaliações (depois de iniciadas)
ssh ubuntu@3.93.21.231 'tail -f ~/seriguela/evaluation_results_*/evaluation.log'
```

## Custos

| Item | Status | Tempo | Taxa | Custo |
|------|--------|-------|------|-------|
| Base/Medium (download) | ✅ | 10 min | $2.012/h | ~$0.34 |
| Instância avaliação | ▶️ | ~2h est. | $1.006/h | ~$2.00 |
| **TOTAL ESTIMADO** | | | | **~$2.34** |

## Timeline

- **20:00 UTC**: Iniciado workflow
- **20:05 UTC**: ✅ Modelos baixados
- **20:06 UTC**: ✅ Instâncias Base/Medium paradas
- **20:07 UTC**: ✅ Instância avaliação lançada
- **20:07 UTC**: ⏳ Uploads iniciados
- **20:15 UTC** (est.): Uploads completos
- **20:20 UTC** (est.): Avaliações iniciadas
- **22:00 UTC** (est.): Avaliações completas
- **22:05 UTC** (est.): Resultados baixados, instância parada

## Próximas Ações (Automáticas)

1. Aguardar uploads completarem (~5-10 min)
2. Executar script de avaliação remoto
3. Monitorar progresso
4. Baixar resultados quando completo
5. Parar instância

---

**Última atualização**: 2026-02-10 20:07 UTC
**Próxima checagem**: 20:15 UTC (verificar uploads)
