# 4_analysis/ - Análise de Resultados

Este diretório contém scripts para análise estatística, visualização e interpretação de resultados.

## Estrutura

```
4_analysis/
├── statistical/            # Análises estatísticas
│   ├── aggregate_nguyen_results.py      # Agregar resultados
│   └── analyze_nguyen_results.py        # Análise completa
│
├── visualization/          # Criação de gráficos
│   └── create_visualizations.py         # Figuras publicáveis
│
└── complexity/             # Análise de complexidade
    ├── analyze_complexity.py            # Métricas de complexidade
    └── analyze_failed_expressions.py    # Análise de erros
```

## Análises Disponíveis

### 1. Análise Estatística

Script: `statistical/analyze_nguyen_results.py`

**Output**:
- Estatísticas descritivas por modelo
- Testes de significância (t-test, ANOVA)
- Effect sizes (Cohen's d)
- Tabelas comparativas

**Uso**:
```bash
cd statistical
python analyze_nguyen_results.py
```

**Métricas calculadas**:
- Valid rate médio e desvio padrão
- R² médio, mediana, min, max
- Testes de hipótese (p-values)
- Intervalos de confiança 95%

### 2. Visualizações

Script: `visualization/create_visualizations.py`

**Gera 4 figuras (300 DPI, publication-ready)**:
1. `fig1_valid_rate_comparison.png` - Bar charts de valid rate
2. `fig2_r2_performance.png` - Comparação de R²
3. `fig3_benchmark_heatmap.png` - Heatmap 12×3
4. `fig4_scaling_progression.png` - Scaling laws

**Uso**:
```bash
cd visualization
python create_visualizations.py
```

**Customização**:
```python
# Editar create_visualizations.py

# Cores personalizadas
colors = ['#FF0000', '#00FF00', '#0000FF']

# Tamanho das figuras
plt.rcParams['figure.figsize'] = (16, 10)

# DPI
plt.savefig('output.png', dpi=600)  # Ultra high-res
```

### 3. Análise de Complexidade

Script: `complexity/analyze_complexity.py`

**Métricas de complexidade**:
- **Depth**: Profundidade de aninhamento
- **Power ops usage**: % de x², x**n
- **Nested trig**: % de sin(cos(x)), etc
- **Operator distribution**: Histograma de operadores
- **Variable usage**: Padrões de uso de variáveis

**Uso**:
```bash
cd complexity
python analyze_complexity.py \
  --model_path ../../models/gpt2/large_700k_json \
  --num_samples 200 \
  --output_file complexity_large.json
```

**Output example**:
```json
{
  "avg_depth": 2.3,
  "max_depth": 5,
  "power_ops_pct": 45.2,
  "nested_trig_pct": 12.5,
  "operator_dist": {
    "+": 0.25,
    "*": 0.30,
    "sin": 0.15,
    "**": 0.10,
    "log": 0.05
  }
}
```

## Análises do Model Scaling Study

### Testes de Significância

**Valid Rate (Base → Large)**:
```
Mean difference: +26.5 percentage points
t-statistic: 4.52
p-value: < 0.001 (highly significant)
Cohen's d: 1.24 (large effect)
```

**R² Score (Base → Large)**:
```
Mean difference: +0.0662
t-statistic: 2.87
p-value: 0.008 (significant)
Cohen's d: 0.79 (medium-large effect)
```

### Scaling Laws Observados

**Valid Rate**:
```
Base (124M):   62.5%
Medium (355M): 75.2% (+12.7 pp)
Large (774M):  89.0% (+13.8 pp)

Linear scaling: ~13 pp per size jump
```

**R² Score**:
```
Base:   0.9190
Medium: 0.9812 (+6.8%)
Large:  0.9852 (+0.4%)

Diminishing returns observed
```

## Workflow de Análise Completa

```bash
# 1. Agregar resultados
cd statistical
python aggregate_nguyen_results.py --input_dir ../../results/2025-02_model_scaling/nguyen

# 2. Análise estatística
python analyze_nguyen_results.py

# 3. Análise de complexidade (3 modelos)
cd ../complexity
for model in base medium large; do
    python analyze_complexity.py \
      --model_path ../../models/gpt2/${model}_700k_json \
      --output_file complexity_${model}.json
done

# 4. Gerar visualizações
cd ../visualization
python create_visualizations.py

# 5. Resultados em: ../../results/2025-02_model_scaling/analysis/
```

## Visualizações Customizadas

### Criar Novo Gráfico

```python
import matplotlib.pyplot as plt
import json

# Carregar dados
with open('../../results/2025-02_model_scaling/nguyen/summary.json') as f:
    data = json.load(f)

# Criar figura
fig, ax = plt.subplots(figsize=(12, 8))

# Plot
models = ['Base', 'Medium', 'Large']
r2_scores = [r['best_r2'] for r in data['results']]
ax.boxplot(r2_scores, labels=models)

# Salvar
plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
```

### Heatmap Personalizado

```python
import seaborn as sns
import numpy as np

# Matriz de resultados (12 benchmarks × 3 models)
matrix = np.array([...])  # Seus dados aqui

# Heatmap
sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=['Base', 'Medium', 'Large'],
            yticklabels=[f'Nguyen-{i+1}' for i in range(12)])

plt.savefig('heatmap.png', dpi=300)
```

## Análise de Erros

Script: `complexity/analyze_failed_expressions.py`

**Categoriza erros**:
- Syntax errors
- Division by zero
- Undefined operations
- Variable/operator misuse
- Incomplete expressions

**Uso**:
```bash
cd complexity
python analyze_failed_expressions.py \
  --results_file ../../results/2025-02_model_scaling/quality/gpt2_base_700K_json_results.json
```

**Output**:
```
Error Analysis:
  Total errors: 3/500 (0.6%)

  By type:
    - Syntax error: 2 (66.7%)
    - Incomplete: 1 (33.3%)

  Common patterns:
    - Missing closing parenthesis: 2 cases
    - Truncated expression: 1 case
```

## Exportar para LaTeX

```python
# Gerar tabela LaTeX
import pandas as pd

df = pd.DataFrame({
    'Model': ['Base', 'Medium', 'Large'],
    'Valid Rate': [62.5, 75.2, 89.0],
    'Avg R²': [0.9190, 0.9812, 0.9852]
})

print(df.to_latex(index=False, float_format='%.2f'))
```

## Reprodução de Resultados

Para reproduzir análises do paper:

```bash
# 1. Verificar dados disponíveis
ls ../../results/2025-02_model_scaling/

# 2. Rodar pipeline completo
bash run_full_analysis.sh

# 3. Outputs em: ../../docs/visualizations/
```

## Referências

- Statistical tests: scipy.stats
- Visualizations: matplotlib + seaborn
- Effect sizes: "Statistical Power Analysis" (Cohen, 1988)
