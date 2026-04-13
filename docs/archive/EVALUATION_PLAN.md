# Plano de Avaliação - Experimento de Model Scaling

**Tipo**: Trabalho Acadêmico - Requer rigor científico e metodologia clara

**Data**: 2026-02-03

**Status**: ⏳ Aguardando Large completar (~2 minutos)

---

## Objetivos da Avaliação

### Pergunta de Pesquisa Principal
**"Modelos GPT-2 maiores (355M, 774M) geram expressões matemáticas mais complexas e válidas do que modelos menores (124M) para regressão simbólica?"**

### Hipóteses a Testar

1. **H1 (Validade)**: Modelos maiores → maior taxa de expressões válidas
2. **H2 (Complexidade)**: Modelos maiores → maior profundidade e aninhamento
3. **H3 (Operações Avançadas)**: Modelos maiores → maior uso de potências (x²)
4. **H4 (Diversidade)**: Modelos maiores → maior variedade de expressões
5. **H5 (Performance)**: Modelos maiores → melhor R² em benchmarks

---

## Fase 1: Avaliação Básica (Paralela - 15 min)

### 1.1 Qualidade de Expressões

**Script**: `scripts/evaluate.py`

**Comando**:
```bash
# Executar em paralelo (3 processos simultâneos)
python scripts/evaluate.py --model_path ./output/gpt2_base_700K_json --num_samples 500 --output_file ./results/base_quality.json &
python scripts/evaluate.py --model_path ./output/gpt2_medium_700K_json --num_samples 500 --output_file ./results/medium_quality.json &
python scripts/evaluate.py --model_path ./output/gpt2_large_700K_json --num_samples 500 --output_file ./results/large_quality.json &
wait
```

**Métricas Coletadas**:
- Valid expression rate (%)
- Parseable rate (%)
- Constraint adherence (%)
  - Uses allowed vars (%)
  - Uses allowed operators (%)
- Diversity rate (%)
- Unique expressions count
- Average expression length
- Top errors (debugging)

**Tempo**: ~5 minutos por modelo (paralelo = 5 min total)

### 1.2 Análise de Complexidade

**Script**: `scripts/analyze_complexity.py`

**Comando**:
```bash
# Executar em paralelo
python scripts/analyze_complexity.py --model_path ./output/gpt2_base_700K_json --num_samples 200 --output_file ./results/base_complexity.json &
python scripts/analyze_complexity.py --model_path ./output/gpt2_medium_700K_json --num_samples 200 --output_file ./results/medium_complexity.json &
python scripts/analyze_complexity.py --model_path ./output/gpt2_large_700K_json --num_samples 200 --output_file ./results/large_complexity.json &
wait
```

**Métricas Coletadas**:
- Power operations usage (x², x**n) - %
- Nested trigonometric functions - %
- Average nesting depth
- Maximum nesting depth
- Operator distribution
- Expression tree depth histogram

**Tempo**: ~5 minutos por modelo (paralelo = 5 min total)

---

## Fase 2: Comparação Direta (5 min)

**Script**: `scripts/compare_trained_models.py`

**Comando**:
```bash
python scripts/compare_trained_models.py \
  --model_base ./output/gpt2_base_700K_json \
  --model_medium ./output/gpt2_medium_700K_json \
  --model_large ./output/gpt2_large_700K_json \
  --dataset data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 10 \
  --output_file ./results/model_comparison.json
```

**Output**: Tabela comparativa lado a lado com todas as métricas

**Tempo**: ~5 minutos

---

## Fase 3: Nguyen Suite Básica (1-2 horas)

### Escopo Reduzido (Viável para Acadêmico)

**Decisão**: Rodar apenas **geração direta (supervised)** sem RL para ter resultados rápidos.

- 3 modelos × 12 benchmarks × 1 método = **36 experimentos**
- Tempo: ~1-2 horas (vs 12-16h para suite completa com RL)

**Script**: `scripts/run_nguyen_basic.sh` (criar)

```bash
#!/bin/bash
# Nguyen suite básica - apenas geração supervised

MODELS=("base" "medium" "large")
BENCHMARKS=(1 2 3 4 5 6 7 8 9 10 11 12)
OUTPUT_DIR="./results/nguyen_basic"

mkdir -p $OUTPUT_DIR

for model in "${MODELS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    echo "Testing: $model + Nguyen-$bench"

    python scripts/evaluate.py \
      --model_path ./output/gpt2_${model}_700K_json \
      --dataset data/benchmarks/nguyen/nguyen_${bench}.csv \
      --num_samples 100 \
      --output_file $OUTPUT_DIR/${model}_nguyen${bench}.json
  done
done

echo "Basic Nguyen suite completed!"
python scripts/aggregate_nguyen_basic.py --input_dir $OUTPUT_DIR --output ./results/nguyen_summary.csv
```

**Métricas por benchmark**:
- Valid expression rate
- Best R² achieved (geração supervised)
- Mean R² (valid expressions)
- Expression complexity metrics

---

## Fase 4: Agregação e Visualização (30 min)

### 4.1 Criar Tabelas Comparativas

**Script**: `scripts/aggregate_all_results.py` (criar)

**Output**:
1. `results/comparison_table.csv` - Tabela mestre com todas métricas
2. `results/hypothesis_tests.csv` - Testes estatísticos por hipótese
3. `results/model_scaling_summary.md` - Resumo executivo

### 4.2 Visualizações

**Gráficos necessários**:
1. Valid Rate vs Model Size (bar chart)
2. Complexity Metrics vs Model Size (multi-line)
3. R² Distribution per Model (box plot)
4. Expression Depth Histogram (3 histogramas sobrepostos)
5. Operator Usage Heatmap (operators × models)
6. Training Loss Curves (3 curvas do Wandb)

**Script**: `scripts/generate_academic_plots.py` (criar)

---

## Fase 5: Relatório Científico (1 hora)

### Estrutura do Relatório

**Arquivo**: `EXPERIMENT_MODEL_SCALING.md` (atualizar)

#### Seções:

1. **Abstract** (150-200 palavras)
   - Contexto, objetivo, método, principais resultados, conclusão

2. **Introduction**
   - Motivação
   - Problema de pesquisa
   - Objetivos
   - Estrutura do documento

3. **Related Work**
   - Model scaling em LLMs (GPT-3, Chinchilla, etc.)
   - Symbolic regression com ML
   - LoRA e parameter-efficient fine-tuning

4. **Methodology**
   - **4.1 Models**: Descrição dos 3 modelos GPT-2
   - **4.2 Training**: Hiperparâmetros fixos, early stopping, LoRA config
   - **4.3 Dataset**: augustocsc/sintetico_natural (700K)
   - **4.4 Evaluation Metrics**: Detalhamento de cada métrica
   - **4.5 Benchmarks**: Nguyen 1-12 suite
   - **4.6 Statistical Tests**: Mann-Whitney U, Kruskal-Wallis

5. **Results**
   - **5.1 Training Performance**: Loss curves, early stopping analysis
   - **5.2 Expression Quality**: Valid rate, constraint adherence, diversity
   - **5.3 Expression Complexity**: Depth, power ops, nesting
   - **5.4 Benchmark Performance**: Nguyen suite results
   - **5.5 Hypothesis Testing**: Tabela com p-values e effect sizes

6. **Discussion**
   - **6.1 Key Findings**: Interpretação dos resultados
   - **6.2 Implications**: O que os resultados significam para o campo
   - **6.3 Limitations**: Early stopping agressivo, LoRA vs full fine-tuning, single seed
   - **6.4 Unexpected Results**: Qualquer descoberta não prevista

7. **Conclusion**
   - Resumo das contribuições
   - Resposta à pergunta de pesquisa
   - Recomendações para prática

8. **Future Work**
   - Testar outros tamanhos (1.5B, 7B)
   - Variar LoRA rank com model size
   - Full fine-tuning comparison
   - Extend to Nguyen 13-20

9. **References**
   - Papers citados (GPT-2, LoRA, symbolic regression)

10. **Appendices**
    - A: Tabelas completas de resultados
    - B: Hiperparâmetros detalhados
    - C: Exemplos de expressões geradas
    - D: Código de reprodução

---

## Fase 6: Documentação Final (30 min)

### Arquivos a Atualizar

1. **TRAINING_LOG_MODEL_SCALING_2025.md**
   - Preencher métricas finais de treinamento
   - Loss curves, early stopping details
   - Custos finais, tempos reais

2. **Model Cards** (3 arquivos)
   - `model_cards/gpt2_base_700K_json_card.md`
   - `model_cards/gpt2_medium_700K_json_card.md`
   - `model_cards/gpt2_large_700K_json_card.md`
   - Preencher performance metrics

3. **CLAUDE.md**
   - Adicionar seção "Model Scaling Study Results"
   - Key findings summary
   - Instruções de reprodução

4. **README_EXPERIMENT.md**
   - Atualizar status para "Completed"
   - Adicionar link para relatório científico

---

## Checklist de Qualidade Acadêmica

### Rigor Metodológico
- [ ] Seed fixo (42) usado em todos experimentos
- [ ] Hiperparâmetros documentados completamente
- [ ] Split train/val claramente definido (90/10)
- [ ] Métricas replicáveis com instruções claras

### Análise Estatística
- [ ] Testes de significância aplicados (Mann-Whitney U)
- [ ] P-values reportados para cada hipótese
- [ ] Effect sizes calculados (Cohen's d)
- [ ] Confidence intervals reportados

### Documentação
- [ ] Código versionado (git commits)
- [ ] Scripts de reprodução fornecidos
- [ ] Logs de treinamento completos salvos
- [ ] Wandb runs públicos e linkados

### Transparência
- [ ] Limitações claramente discutidas
- [ ] Resultados negativos reportados honestamente
- [ ] Early stopping justificado (custo-benefício)
- [ ] Escolhas metodológicas justificadas

### Visualizações
- [ ] Todas figuras com legendas claras
- [ ] Eixos rotulados com unidades
- [ ] Cores colorblind-friendly
- [ ] Resolução adequada para publicação

---

## Cronograma de Execução

| Fase | Tempo Estimado | Pode Rodar em Paralelo? |
|------|----------------|-------------------------|
| 0. Aguardar Large | 2 min | N/A |
| 1. Avaliação Básica | 15 min | ✅ Sim (3 modelos paralelos) |
| 2. Comparação Direta | 5 min | ❌ Não (depende de Fase 1) |
| 3. Nguyen Suite Básica | 1-2h | ✅ Parcial (por modelo) |
| 4. Agregação/Viz | 30 min | ❌ Não (depende de tudo) |
| 5. Relatório Científico | 1h | ❌ Não (análise manual) |
| 6. Documentação Final | 30 min | ❌ Não (review manual) |

**Total**: ~3.5 - 4.5 horas

---

## Decisão: Suite Completa vs Suite Básica

### Opção A: Suite Completa (144 experimentos)
- 3 modelos × 12 benchmarks × 4 algoritmos (Supervised, REINFORCE, GRPO, PPO)
- **Tempo**: 12-16 horas
- **Benefício**: Completo, testa RL optimization
- **Custo**: Muito tempo, pode usar AWS

### Opção B: Suite Básica (36 experimentos) - RECOMENDADO
- 3 modelos × 12 benchmarks × 1 método (apenas supervised)
- **Tempo**: 1-2 horas
- **Benefício**: Resultados mais rápidos, foco no scaling
- **Limitação**: Não testa RL (pode ser future work)

**Recomendação**: Começar com **Opção B** para ter resultados acadêmicos rápidos. Se resultados forem promissores, pode-se adicionar RL como "follow-up study".

---

## Critérios de Sucesso

### Mínimo Viável (Trabalho Acadêmico Aceitável)
- ✅ Todos 3 modelos avaliados com métricas claras
- ✅ Hipóteses testadas com testes estatísticos
- ✅ Resultados documentados em formato científico
- ✅ Reprodutibilidade garantida (scripts + seeds + logs)

### Ideal (Publicação de Qualidade)
- ✅ Diferenças estatisticamente significativas encontradas
- ✅ Effect sizes reportados e interpretados
- ✅ Visualizações profissionais
- ✅ Discussão profunda das implicações
- ✅ Comparação com trabalhos relacionados
- ✅ Limitações honestamente discutidas

---

**Próxima Ação**: Aguardar Large completar (~2 min), depois iniciar Fase 1 (avaliações paralelas).

**Arquivo de Output**: Este plano será seguido para gerar `EXPERIMENT_MODEL_SCALING.md` (relatório científico final).
