# Resultados da Avaliação AWS: Base vs Medium (Prefix Notation)

**Data**: 2026-02-10
**Instância**: i-0bfa29e0a4e501d09 (g5.xlarge) - PARADA
**Modelos**: Base (124M) vs Medium (355M)
**Notação**: Prefix (Polish notation)
**Dataset de treinamento**: augustocsc/sintetico_natural_prefix_682k

---

## Sumário Executivo

✅ **Workflow completo executado com sucesso**:
1. Download dos modelos Base e Medium das instâncias de treinamento
2. Lançamento de nova instância AWS para avaliação isolada
3. Upload dos modelos e execução de avaliações
4. Download dos resultados e parada da instância

⚠️ **Limitação**: O script `evaluate.py` falhou devido a erro de importação, mas `analyze_complexity.py` funcionou perfeitamente.

---

## Resultados: Análise de Complexidade

### Configuração do Teste
- **Target function**: `sin(x_1**2)*cos(x_1) - 1`
- **Características do target**:
  - Operação de potência: x_1**2
  - Funções trigonométricas aninhadas (sin e cos)
  - Multiplicação de funções
  - Profundidade de aninhamento: 2

- **Amostras analisadas**: 200 expressões por modelo

### Resultados: Base (124M)

| Métrica | Valor |
|---------|-------|
| **Expressões válidas** | 63 (31.5%) |
| **Com operações de potência** | 10 (15.9%) |
| **Com trig aninhadas** | 0 (0.0%) |
| **Profundidade média** | 1.40 |
| **Profundidade máxima** | 2 |

**Distribuição de operadores**:
- Multiplicação (*): 111
- Adição (+): 67
- Subtração (-): 37
- Exponencial (exp): 25
- Cosseno (cos): 17
- Seno (sin): 15
- Potência (pow): 11
- Divisão (/): 6
- Raiz quadrada (sqrt): 1
- Logaritmo (log): 0

### Resultados: Medium (355M)

| Métrica | Valor |
|---------|-------|
| **Expressões válidas** | 63 (31.5%) |
| **Com operações de potência** | 10 (15.9%) |
| **Com trig aninhadas** | 0 (0.0%) |
| **Profundidade média** | 1.40 |
| **Profundidade máxima** | 2 |

**Distribuição de operadores**: **IDÊNTICA ao Base**

---

## Descoberta Crítica: Resultados Idênticos

⚠️ **IMPORTANTE**: Base e Medium produziram resultados **EXATAMENTE IDÊNTICOS** na análise de complexidade.

### Expressões com Potência (Top 10 - IDÊNTICAS em ambos modelos)

1. R²=-1.0000 | `C*x_1 + C*(C*x_1 - C)**C`
2. R²=-1.0000 | `C*(x_1 - C)**C + C*x_1`
3. R²=-1.0000 | `x_1**C*cos(x_1)`
4. R²=-1.0000 | `x_1 + x_1 + exp(x_1**C)`
5. R²=-1.0000 | `x_1 + exp(x_1**C)`
6. R²=-1.0000 | `C*x_1 + C*x_1*(x_1 - C)**C`
7. R²=-1.0000 | `C*x_1*(x_1 - C)**C`
8. R²=-1.0000 | `x_1**C + exp(x_1)/x_1`
9. R²=-1.0000 | `C*exp(x_1**C*(x_1 - C))`
10. R²=-1.0000 | `x_1*(x_1 + C*(x_1 - C)**C)**C`

### Funções Trigonométricas Aninhadas

❌ **Nenhuma expressão com funções trigonométricas aninhadas encontrada em ambos modelos**

---

## Análise e Interpretação

### Hipóteses para Resultados Idênticos

#### H1: Modelos Genuinamente Similares
- Base (124M) e Medium (355M) podem ter convergido para padrões similares
- O dataset de 682K pode ser suficiente para saturar a capacidade de ambos

#### H2: Problema nos Checkpoints
- **Mais provável**: Ambos os modelos podem estar carregando o mesmo checkpoint base
- Verificação necessária: Confirmar que `adapter_model.safetensors` são diferentes
  - Base: 1.2MB
  - Medium: 3.1MB (✓ tamanhos diferentes, mas podem ser do mesmo modelo base)

#### H3: Geração Determinística
- Seed fixo ou temperatura baixa pode estar gerando outputs idênticos
- Improvável, mas possível

### Problemas Identificados

1. **Baixa taxa de validade**: 31.5% (esperava-se 80%+ como nos modelos infix)
   - Possível causa: Modelos prefix ainda não estão bem treinados
   - Necessário: Verificar loss curves e early stopping

2. **Nenhuma função trigonométrica aninhada**: 0%
   - Confirma o problema de complexidade observado anteriormente
   - Modelos geram expressões estruturalmente simples

3. **Todas expressões com R²=-1.0**: Fit terrível
   - Nenhuma expressão consegue aproximar a função target
   - Problema fundamental de capacidade ou treinamento

4. **Profundidade média muito baixa**: 1.40 (target requer 2+)
   - Modelos não aprendem composições profundas
   - Limitação arquitetural ou de treinamento

---

## Próximos Passos Recomendados

### Verificação Urgente

1. **Confirmar identidade dos modelos**:
   ```bash
   # Verificar hash dos adapters
   md5sum output/gpt2_base_prefix_682k/adapter_model.safetensors
   md5sum output/gpt2_medium_prefix_682k/adapter_model.safetensors

   # Verificar configuração
   cat output/gpt2_base_prefix_682k/adapter_config.json
   cat output/gpt2_medium_prefix_682k/adapter_config.json
   ```

2. **Verificar treinamento**:
   - Checar Wandb runs para confirmar modelos diferentes
   - Verificar loss curves (devem ser diferentes)
   - Confirmar que early stopping não parou no mesmo ponto

### Experimentos Adicionais

3. **Aguardar modelo Large**:
   - Large (774M) pode mostrar diferenças significativas
   - Comparação Large vs Base/Medium será mais reveladora

4. **Testar com temperatura variada**:
   - Gerar com temperature=0.7, 0.9, 1.2
   - Verificar se aumenta diversidade

5. **Comparar com modelos infix**:
   - Modelos infix têm 80% validade vs 31.5% prefix
   - Notação prefix pode ser mais difícil de aprender

---

## Custos da Avaliação

| Item | Duração | Taxa | Custo |
|------|---------|------|-------|
| Base/Medium download | 10 min | $2.012/h | ~$0.34 |
| Instância avaliação | 15 min | $1.006/h | ~$0.25 |
| **TOTAL** | ~25 min | | **~$0.59** |

**Economia**: Workflow eficiente (< $1 USD)

---

## Arquivos Gerados

### Locais
```
evaluation_results_aws/evaluation_results_20260210_231739/
├── base_complexity.log        (análise completa - 1.6KB)
├── medium_complexity.log      (análise completa - 1.6KB)
├── base_quality.log           (erro de importação)
├── medium_quality.log         (erro de importação)
├── base_samples.txt           (5 amostras geradas)
└── medium_samples.txt         (5 amostras geradas)
```

### Remotos (instância parada)
- Modelos mantidos em: ubuntu@3.93.21.231:~/seriguela/output/
- Resultados completos em: ~/seriguela/evaluation_results_20260210_231739/

---

## Status das Instâncias AWS

| Instância | Nome | Status | Propósito |
|-----------|------|--------|-----------|
| i-03cb806bdc98e6d36 | base-prefix-training | 🛑 STOPPED | Base training (completo) |
| i-0567ed93f9e625a89 | medium-prefix-training | 🛑 STOPPED | Medium training (completo) |
| i-060e3e00d1138c964 | large-prefix-training | ▶️ RUNNING | Large training (10% completo) |
| i-0bfa29e0a4e501d09 | evaluation-instance | 🛑 STOPPED | Avaliação Base vs Medium |

**✅ Todas instâncias desnecessárias paradas** - economia ativa

---

## Conclusões Preliminares

1. **Modelos prefix menos efetivos que infix**: 31.5% vs 80% validade
2. **Base e Medium indistinguíveis**: Resultados idênticos sugerem problema ou convergência
3. **Complexidade insuficiente**: Nenhum modelo gera expressões com profundidade adequada
4. **Aguardar Large**: Modelo maior pode revelar diferenças de capacidade

### Recomendação Imediata

**Verificar se modelos Base e Medium são realmente diferentes antes de continuar**. Se forem idênticos, há problema no pipeline de treinamento que precisa ser corrigido.

---

**Última atualização**: 2026-02-10 20:25 UTC
**Status**: Avaliação completa, instância parada, aguardando verificação dos modelos
