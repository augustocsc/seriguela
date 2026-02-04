# 3_evaluation/ - Avaliação de Modelos

Este diretório contém scripts para avaliar modelos treinados em diferentes métricas e benchmarks.

## Estrutura

```
3_evaluation/
├── quality/                # Avaliação de qualidade
│   ├── evaluate_quality_simple.py      # Valid rate, diversity
│   ├── evaluate.py                      # Avaliação completa
│   └── evaluate_experiments.py          # Comparação de experimentos
│
├── benchmarks/             # Avaliação em benchmarks
│   ├── evaluate_nguyen_benchmarks.py    # Nguyen 1-12
│   ├── run_all_nguyen_benchmarks.py     # Suite completa
│   └── run_nguyen_subset.py             # Subset de benchmarks
│
├── comparison/             # Comparação entre modelos
│   ├── compare_models.py                # Comparação básica
│   └── compare_trained_models.py        # Comparação detalhada
│
├── generate.py             # Geração de expressões
└── show_expressions.py     # Visualizar expressões geradas
```

## Métricas de Avaliação

### 1. Qualidade de Expressão

Script: `quality/evaluate_quality_simple.py`

**Métricas**:
- **Valid Rate**: % de expressões sintaticamente corretas
- **Diversity Rate**: % de expressões únicas
- **Parseable Rate**: % de expressões parseáveis
- **Error Types**: Categorização de erros

**Uso**:
```bash
cd quality
python evaluate_quality_simple.py \
  --model_path ../../models/gpt2/medium_700k_json \
  --num_samples 500
```

**Output esperado**:
```
Valid expressions: 496/500 (99.2%)
Diversity: 98.8%
Unique expressions: 494
Average length: 42.3 chars
```

### 2. Benchmarks Nguyen

Script: `benchmarks/evaluate_nguyen_benchmarks.py`

**Métricas**:
- **Valid Rate**: % de expressões válidas geradas
- **Best R²**: Melhor fit entre expressões geradas
- **Mean R²**: R² médio das expressões válidas
- **Samples to Best**: Quantas amostras até melhor R²

**Uso Individual**:
```bash
cd benchmarks
python evaluate_nguyen_benchmarks.py \
  --model_path ../../models/gpt2/large_700k_json \
  --benchmark_name nguyen_5 \
  --num_samples 100
```

**Suite Completa** (todos os 12 benchmarks):
```bash
cd benchmarks
python run_all_nguyen_benchmarks.py \
  --model_path ../../models/gpt2/large_700k_json \
  --output_dir ../../results/2025-02_test/nguyen
```

### 3. Comparação de Modelos

Script: `comparison/compare_trained_models.py`

**Compara**:
- Valid rates
- R² scores
- Complexity metrics (depth, operators)
- Execution time

**Uso**:
```bash
cd comparison
python compare_trained_models.py \
  --model_base ../../models/gpt2/base_700k_json \
  --model_medium ../../models/gpt2/medium_700k_json \
  --model_large ../../models/gpt2/large_700k_json \
  --dataset ../../1_data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 10
```

## Resultados Históricos

### Model Scaling Study (2025-02)

| Modelo | Valid Rate (Quality) | Valid Rate (Nguyen) | Avg R² | Max R² |
|--------|---------------------|---------------------|---------|--------|
| Base | 99.4% | 62.5% | 0.9190 | 0.9994 |
| Medium | 99.2% | 75.2% | 0.9812 | 0.9999 |
| Large | **100%** 🏆 | **89.0%** 🏆 | **0.9852** 🏆 | **1.0000** 🏆 |

**Destaque**: Large model achieving perfect R²=1.0 on Nguyen-8

## Geração de Expressões

Script: `generate.py`

**Uso**:
```bash
python generate.py \
  --model_path ../models/gpt2/large_700k_json \
  --num_generations 50 \
  --validate \
  --temperature 0.7
```

**Parâmetros**:
- `--temperature`: Controla diversidade (0.1-1.0)
- `--top_p`: Nucleus sampling (0.9-0.95)
- `--num_return_sequences`: Batch generation
- `--validate`: Validar expressões com SymPy

## Workflow Típico

### Avaliação Completa de Novo Modelo

```bash
# 1. Avaliar qualidade (500 samples)
cd quality
python evaluate_quality_simple.py \
  --model_path ../../models/novo_modelo \
  --num_samples 500 \
  --output_file ../../results/novo_experimento/quality.json

# 2. Avaliar em Nguyen (36 experiments)
cd ../benchmarks
python run_all_nguyen_benchmarks.py \
  --model_path ../../models/novo_modelo \
  --output_dir ../../results/novo_experimento/nguyen

# 3. Comparar com modelos existentes
cd ../comparison
python compare_trained_models.py \
  --model_base ../../models/gpt2/base_700k_json \
  --model_new ../../models/novo_modelo \
  --dataset ../../1_data/benchmarks/nguyen/nguyen_5.csv
```

## Futuro: Novos Benchmarks

### Feynman Equations (120+ equações)
```bash
# Estrutura planejada
python evaluate_feynman_benchmarks.py \
  --model_path ../../models/gpt2/large_700k_json \
  --category mechanics  # mechanics, electromagnetism, etc
```

### Strogatz Benchmarks
```bash
# Sistemas dinâmicos
python evaluate_strogatz_benchmarks.py \
  --model_path ../../models/novo_modelo \
  --system oscillator
```

## Métricas Detalhadas

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

Onde:
  SS_res = Σ(y_true - y_pred)²  # Soma dos quadrados dos resíduos
  SS_tot = Σ(y_true - y_mean)²  # Variância total

Interpretação:
  R² = 1.0    → Fit perfeito
  R² > 0.99   → Excelente
  R² > 0.95   → Muito bom
  R² > 0.90   → Bom
  R² < 0      → Pior que média
```

### Valid Expression Rate
```
Valid Rate = (Valid / Total) * 100%

Critérios de validade:
  1. Sintaxe correta (parseável)
  2. Semanticamente avaliável
  3. Sem erros de runtime
  4. Usa apenas ops/vars permitidas
```

## Troubleshooting

### Low Valid Rate (<80%)
- Verificar formato de dados de treino
- Aumentar epochs de treinamento
- Verificar temperature (reduzir para 0.6-0.7)

### Low R² Scores
- Gerar mais samples (100 → 200)
- Tentar RL fine-tuning (PPO, GRPO)
- Verificar se benchmark é muito complexo

### OOM Durante Avaliação
- Reduzir `num_samples`
- Usar CPU ao invés de GPU
- Processar em batches menores

## Referências

- Nguyen benchmarks: "Semantically-based crossover in genetic programming" (2012)
- R² metric: scikit-learn documentation
- SymPy validation: SymPy documentation
