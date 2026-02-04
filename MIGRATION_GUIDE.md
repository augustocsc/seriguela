# Migration Guide - Nova Estrutura do Projeto

Este documento mapeia a estrutura antiga para a nova estrutura organizada por fases de pesquisa.

## Estrutura Antiga → Nova

### Dados

| Antigo | Novo |
|--------|------|
| `data/benchmarks/nguyen/` | `1_data/benchmarks/nguyen/` |
| `data/processed/` | `1_data/processed/` |
| `data/experiments/` | `1_data/processed/` |

### Scripts de Treinamento

| Antigo | Novo |
|--------|------|
| `scripts/train.py` | `2_training/supervised/train.py` |
| `scripts/train_with_json.py` | `2_training/supervised/train_with_json.py` |
| `scripts/ppo_symbolic.py` | `2_training/reinforcement/ppo_symbolic.py` |
| `scripts/grpo_symbolic.py` | `2_training/reinforcement/grpo_symbolic.py` |
| `scripts/reinforce_*.py` | `2_training/reinforcement/` |
| `configs/` | `2_training/configs/` |

### Scripts de Avaliação

| Antigo | Novo |
|--------|------|
| `scripts/evaluate.py` | `3_evaluation/quality/evaluate.py` |
| `scripts/evaluate_quality_simple.py` | `3_evaluation/quality/` |
| `scripts/evaluate_nguyen_benchmarks.py` | `3_evaluation/benchmarks/` |
| `scripts/compare_models.py` | `3_evaluation/comparison/` |
| `scripts/generate.py` | `3_evaluation/generate.py` |

### Scripts de Análise

| Antigo | Novo |
|--------|------|
| `scripts/analyze_complexity.py` | `4_analysis/complexity/` |
| `scripts/aggregate_nguyen_results.py` | `4_analysis/statistical/` |
| `analyze_nguyen_results.py` | `4_analysis/statistical/` |
| `create_visualizations.py` | `4_analysis/visualization/` |

### Modelos

| Antigo | Novo |
|--------|------|
| `output/gpt2_base_700K_json/` | `models/gpt2/base_700k_json/` |
| `output/gpt2_medium_700K_json/` | `models/gpt2/medium_700k_json/` |
| `output/gpt2_large_700K_json/` | `models/gpt2/large_700k_json/` |

### Resultados

| Antigo | Novo |
|--------|------|
| `results_final/quality/` | `results/2025-02_model_scaling/quality/` |
| `results_nguyen_benchmarks/` | `results/2025-02_model_scaling/nguyen_benchmarks/` |
| `visualizations/` | `results/2025-02_model_scaling/analysis/` (cópia) |

### Documentação

| Antigo | Novo |
|--------|------|
| `SCIENTIFIC_REPORT_MODEL_SCALING.md` | `docs/reports/` |
| `NGUYEN_RESULTS_FINAL.md` | `docs/reports/` |
| `MODEL_CARD_*.md` | `docs/model_cards/` |
| `visualizations/` | `docs/visualizations/` |
| `CLAUDE.md` | `docs/guides/CLAUDE.md` |

### Código Fonte

| Antigo | Novo |
|--------|------|
| `classes/` | `src/seriguela/utils/` |
| `__init__.py` | `src/seriguela/__init__.py` |

## Atualizar Imports

### Antes
```python
from classes.expression import Expression
from classes.dataset import Dataset
```

### Depois
```python
from src.seriguela.utils.expression import Expression
from src.seriguela.utils.dataset import Dataset

# Ou (se instalado como package)
from seriguela.utils import Expression, Dataset
```

## Atualizar Caminhos em Scripts

### Exemplo: Treinamento

**Antes**:
```bash
cd seriguela
python scripts/train_with_json.py \
  --dataset_path ./data/processed/700K \
  --output_dir ./output/test
```

**Depois**:
```bash
cd seriguela/2_training/supervised
python train_with_json.py \
  --dataset_path ../../1_data/processed/700K \
  --output_dir ../../models/gpt2/test
```

### Exemplo: Avaliação

**Antes**:
```bash
python scripts/evaluate_nguyen_benchmarks.py \
  --model_path ./output/gpt2_medium_700K_json \
  --benchmark ./data/benchmarks/nguyen/nguyen_5.csv
```

**Depois**:
```bash
cd 3_evaluation/benchmarks
python evaluate_nguyen_benchmarks.py \
  --model_path ../../models/gpt2/medium_700k_json \
  --benchmark ../../1_data/benchmarks/nguyen/nguyen_5.csv
```

## AWS Scripts

A chave SSH correta é `chave-gpu-nova` (não `chave-gpu`):

```bash
# Atualizar scripts AWS
sed -i 's/chave-gpu/chave-gpu-nova/g' scripts/aws/*.sh
```

## Verificação de Migração

```bash
# Verificar se novos diretórios foram criados
ls -d 1_data 2_training 3_evaluation 4_analysis models results docs src

# Verificar se modelos foram movidos
ls models/gpt2/

# Verificar se READMEs existem
find . -name "README.md" -type f | head -10

# Verificar se resultados foram copiados
ls results/2025-02_model_scaling/
```

## Retrocompatibilidade

Os diretórios antigos (`scripts/`, `output/`, `data/`) ainda existem por enquanto para retrocompatibilidade. Recomendamos:

1. Atualizar scripts para usar nova estrutura
2. Testar workflows completos
3. Depois de validar, remover diretórios antigos:

```bash
# APENAS após validar nova estrutura
rm -rf output/  # (modelos já copiados para models/)
rm -rf data/    # (dados já copiados para 1_data/)
rm -rf results_final/ results_nguyen_benchmarks/  # (já em results/)
```

## Benefícios da Nova Estrutura

1. **Clareza**: Fases numeradas (1→2→3→4) guiam o workflow
2. **Escalabilidade**: Fácil adicionar novos modelos/benchmarks
3. **Documentação**: README em cada diretório
4. **Organização**: Resultados separados por experimento
5. **Manutenibilidade**: Código fonte isolado em `src/`

## Suporte

Para dúvidas sobre a migração, consulte:
- README.md de cada diretório
- docs/guides/CLAUDE.md (guia completo)
- Issues no GitHub
