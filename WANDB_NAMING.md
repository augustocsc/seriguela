# Wandb Naming Standards

Padrão de nomenclatura para experimentos no Weights & Biases (Wandb) no projeto Seriguela.

## Configuração

Use o módulo `configs/wandb_config.py` para nomenclatura padronizada.

```python
from configs.wandb_config import generate_run_name, get_run_tags, get_wandb_project_name

# Gerar nome do run
run_name = generate_run_name("ppo", "medium", "nguyen5")
# Output: seriguela-ppo-medium-nguyen5-20260203-143022

# Obter tags
tags = get_run_tags("ppo", "medium", "nguyen5", success=True)
# Output: ['ppo', 'gpt2-medium', 'nguyen5', 'rl', 'success']

# Nome do projeto
project = get_wandb_project_name()
# Output: "seriguela"
```

## Formato Padrão

### Nomenclatura de Runs

**Formato**: `seriguela-{type}-{model}-{dataset}-{extra}-{timestamp}`

**Componentes**:
- `seriguela`: Prefixo fixo do projeto
- `{type}`: Tipo de experimento (supervised, ppo, grpo, reinforce, iterative-sft)
- `{model}`: Tamanho do modelo (base, medium, large)
- `{dataset}`: Dataset ou benchmark (700K, nguyen5, nguyen7, etc) - opcional
- `{extra}`: Informações extras (lr3e5, batch16, etc) - opcional
- `{timestamp}`: Data/hora no formato YYYYMMDD-HHMMSS

### Projetos Wandb

- **Principal**: `seriguela` - Para todos os experimentos de produção
- **Experimentos**: `seriguela-experiments` - Para testes e desenvolvimento

## Exemplos

### Supervised Fine-Tuning

```python
# Treinamento base no dataset 700K
generate_run_name("supervised", "base", "700K")
# → seriguela-supervised-base-700k-20260203-143022

# Treinamento medium no dataset 700K
generate_run_name("supervised", "medium", "700K")
# → seriguela-supervised-medium-700k-20260203-143022

# Treinamento large no dataset 700K
generate_run_name("supervised", "large", "700K")
# → seriguela-supervised-large-700k-20260203-143022
```

### Reinforcement Learning

```python
# PPO no benchmark Nguyen-5
generate_run_name("ppo", "base", "nguyen5")
# → seriguela-ppo-base-nguyen5-20260203-143022

# GRPO no benchmark Nguyen-7 com learning rate customizado
generate_run_name("grpo", "medium", "nguyen7", "lr5e5")
# → seriguela-grpo-medium-nguyen7-lr5e5-20260203-143022

# REINFORCE no benchmark Nguyen-10
generate_run_name("reinforce", "large", "nguyen10")
# → seriguela-reinforce-large-nguyen10-20260203-143022
```

### Iterative Training

```python
# Iterative SFT (Best-of-N)
generate_run_name("iterative-sft", "medium", "nguyen5")
# → seriguela-iterative-sft-medium-nguyen5-20260203-143022
```

### Evaluation

```python
# Avaliação sem timestamp (para consistência)
generate_run_name("eval", "medium", "nguyen5", include_timestamp=False)
# → seriguela-eval-medium-nguyen5
```

## Tags Padrão

Tags são geradas automaticamente baseadas no tipo de experimento:

```python
get_run_tags("ppo", "medium", "nguyen5", success=True)
# ['ppo', 'gpt2-medium', 'nguyen5', 'rl', 'success']

get_run_tags("supervised", "base", "700K")
# ['supervised', 'gpt2-base', '700k', 'supervised']

get_run_tags("grpo", "large", "nguyen7", success=False)
# ['grpo', 'gpt2-large', 'nguyen7', 'rl', 'failed']
```

## Uso em Scripts de Treinamento

### Exemplo completo

```python
import wandb
from configs.wandb_config import (
    generate_run_name,
    get_run_tags,
    get_wandb_project_name,
    setup_wandb_env
)

# Setup credentials
setup_wandb_env()

# Generate run name and tags
run_name = generate_run_name(
    experiment_type="ppo",
    model_size="medium",
    dataset="nguyen5",
    extra_info="lr3e5-batch16"
)

tags = get_run_tags(
    experiment_type="ppo",
    model_size="medium",
    dataset="nguyen5"
)

# Initialize wandb
wandb.init(
    project=get_wandb_project_name(),
    name=run_name,
    tags=tags,
    config={
        "model_size": "gpt2-medium",
        "learning_rate": 3e-5,
        "batch_size": 16,
        "dataset": "nguyen5",
        "algorithm": "ppo"
    }
)
```

### Integração com scripts existentes

Para scripts que já usam argumentos customizados:

```python
import argparse
from configs.wandb_config import generate_run_name, get_wandb_project_name

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", default="supervised")
parser.add_argument("--model_size", default="base")
parser.add_argument("--dataset", default="700K")
parser.add_argument("--wandb_run_name", default=None)
args = parser.parse_args()

# Use custom name if provided, otherwise generate
run_name = args.wandb_run_name or generate_run_name(
    args.experiment_type,
    args.model_size,
    args.dataset
)

wandb.init(
    project=get_wandb_project_name(),
    name=run_name,
    config=vars(args)
)
```

## Tipos de Experimento

Constantes disponíveis em `EXPERIMENT_TYPES`:

```python
from configs.wandb_config import EXPERIMENT_TYPES, DATASETS

# Usar constantes para consistência
run_name = generate_run_name(
    EXPERIMENT_TYPES["PPO"],
    "medium",
    DATASETS["NGUYEN_5"]
)
# → seriguela-ppo-medium-nguyen5-20260203-143022
```

**Tipos disponíveis**:
- `SUPERVISED` / `SFT`: Supervised Fine-Tuning
- `PPO`: Proximal Policy Optimization
- `GRPO`: Group Relative Policy Optimization
- `REINFORCE`: REINFORCE algorithm
- `ITERATIVE_SFT`: Iterative SFT (Best-of-N)
- `BEST_OF_N`: Best-of-N sampling
- `EVALUATION`: Model evaluation

**Datasets disponíveis**:
- `MAIN_700K`: Dataset principal (700K exemplos)
- `NGUYEN_1` a `NGUYEN_10`: Benchmarks Nguyen
- `CUSTOM`: Dataset customizado

## Organização no Wandb Dashboard

Com esse padrão, experimentos ficam organizados no dashboard:

```
seriguela/
├── seriguela-supervised-base-700k-*
├── seriguela-supervised-medium-700k-*
├── seriguela-supervised-large-700k-*
├── seriguela-ppo-base-nguyen5-*
├── seriguela-ppo-medium-nguyen5-*
├── seriguela-grpo-large-nguyen7-*
└── seriguela-eval-*
```

**Filtros úteis**:
- Por tipo: `name contains "ppo"`
- Por modelo: `tags contains "gpt2-medium"`
- Por dataset: `tags contains "nguyen5"`
- Sucesso: `tags contains "success"`
- RL: `tags contains "rl"`

## Migração de Scripts Antigos

Para atualizar scripts existentes:

1. Adicionar import:
```python
from configs.wandb_config import generate_run_name, get_wandb_project_name
```

2. Substituir nome hardcoded:
```python
# Antes
run_name = f"{model_name}-{dataset}-{approach}"

# Depois
run_name = generate_run_name(approach, model_name, dataset)
```

3. Usar projeto padrão:
```python
# Antes
wandb.init(project="seriguela_experiments", name=run_name)

# Depois
wandb.init(project=get_wandb_project_name(), name=run_name)
```

## Testes

Executar `configs/wandb_config.py` diretamente para ver exemplos:

```bash
python configs/wandb_config.py
```

Output:
```
Wandb Configuration Examples:

1. Supervised training on 700K dataset:
   seriguela-supervised-medium-700k-20260203-143022

2. PPO on Nguyen-5 benchmark:
   seriguela-ppo-base-nguyen5-20260203-143022

3. GRPO with custom learning rate:
   seriguela-grpo-large-nguyen7-lr5e5-20260203-143022

4. Evaluation run (no timestamp):
   seriguela-eval-medium-nguyen5

5. Tags example:
   ['ppo', 'gpt2-medium', 'nguyen5', 'rl', 'success']

6. Setup Wandb environment:
   ✓ Wandb API key loaded from ~/.tokens.txt
```

## Best Practices

1. **Sempre use o padrão**: Facilita busca e comparação
2. **Adicione tags**: Ajuda a filtrar experimentos relacionados
3. **Use timestamps**: Evita conflitos entre runs
4. **Documente no config**: Adicione parâmetros importantes ao wandb.config
5. **Marque sucesso/falha**: Use tags para indicar resultado final

## Ver também

- `CREDENTIALS_SETUP.md` - Configuração de API keys
- `configs/wandb_config.py` - Código fonte da configuração
- `CLAUDE.md` - Documentação geral do projeto
