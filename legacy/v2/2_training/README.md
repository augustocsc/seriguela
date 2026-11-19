# 2_training/ - Treinamento e Fine-tuning

Este diretório contém todos os scripts e configurações para treinamento de modelos LLM para symbolic regression.

## Estrutura

```
2_training/
├── supervised/             # Fine-tuning supervisionado
│   ├── train_with_json.py          # Principal: treino com formato JSON
│   ├── train.py                     # Script base
│   ├── train_experiment.py          # Experimentos controlados
│   └── iterative_sampling_sft.py    # SFT iterativo
│
├── reinforcement/          # Reinforcement Learning
│   ├── ppo_symbolic.py              # Proximal Policy Optimization
│   ├── grpo_symbolic.py             # Group Relative PO
│   ├── reinforce_*.py               # REINFORCE algorithm
│   └── best_of_n_experiment.py      # Best-of-N sampling
│
└── configs/                # Configurações
    └── wandb_config.py              # Wandb naming standards
```

## Métodos de Treinamento

### 1. Supervised Fine-tuning (Recomendado)

Script principal: `supervised/train_with_json.py`

**Características**:
- LoRA fine-tuning (apenas 294K parâmetros treináveis)
- Formato JSON estruturado (80% valid rate)
- Early stopping automático
- Split train/validation 90/10
- Integração com Wandb

**Uso**:
```bash
cd supervised
python train_with_json.py \
  --model_size gpt2-medium \
  --dataset_path ../../1_data/processed/700K \
  --output_dir ../../models/gpt2/medium_test \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4
```

**Modelos suportados**:
- `gpt2` (124M params)
- `gpt2-medium` (355M params)
- `gpt2-large` (774M params)
- GPT-Neo, LLaMA, Phi (futuro)

### 2. Reinforcement Learning

#### PPO (Proximal Policy Optimization)
Script: `reinforcement/ppo_symbolic.py`

**Quando usar**:
- Problemas complexos (Nguyen 4+)
- Otimização de R² score
- Após supervised fine-tuning

**Uso**:
```bash
cd reinforcement
python ppo_symbolic.py \
  --model_path ../../models/gpt2/base_700k_json \
  --dataset ../../1_data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 20
```

#### GRPO (Group Relative Policy Optimization)
Script: `reinforcement/grpo_symbolic.py`

**Vantagens**:
- Mais estável que PPO
- Melhor para multi-modal rewards
- Baseado em DeepSeek-R1

#### REINFORCE
Script: `reinforcement/reinforce_symbolic.py`

**Características**:
- Simples e eficaz
- EMA baseline
- Bom para benchmarks fáceis (Nguyen 1-3)

## Configuração LoRA

Configuração padrão (todos os modelos):
```python
{
  "r": 8,
  "lora_alpha": 32,
  "target_modules": ["c_attn"],
  "lora_dropout": 0.05
}
```

**Resultado**: ~294K parâmetros treináveis (vs 124M-774M total)

## Hiperparâmetros Recomendados

### Por Tamanho de Modelo

| Modelo | Batch Size | Instance | VRAM | Tempo |
|--------|-----------|----------|------|-------|
| GPT-2 Base | 8 | g5.xlarge | 24GB | 2-3h |
| GPT-2 Medium | 4 | g5.xlarge | 24GB | 3-4h |
| GPT-2 Large | 2 | g5.2xlarge | 48GB | 4-5h |

### Outros Hiperparâmetros
```python
learning_rate = 5e-5
num_train_epochs = 3
gradient_accumulation_steps = 4
warmup_steps = 500
weight_decay = 0.01
early_stopping_patience = 3
seed = 42
```

## Formato de Dados

### JSON Format (Recomendado)
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "sin(x_1 + C*x_2)"}
```

**Vantagens**:
- 80% valid expression rate
- Structured boundaries
- Lower loss (0.343 vs 0.415)

## Wandb Tracking

Naming standard: `seriguela-{type}-{model}-{dataset}-{timestamp}`

Exemplos:
```python
# Supervised
seriguela-supervised-medium-700k-20260204-120000

# PPO
seriguela-ppo-large-nguyen5-20260204-120000
```

## Deploy AWS

Scripts disponíveis em: `../../scripts/aws/`

Lançar treinamento:
```bash
# Medium model
bash ../../scripts/aws/launch_medium_training.sh \
  --wandb-key YOUR_KEY \
  --hf-token YOUR_TOKEN

# Large model
bash ../../scripts/aws/launch_large_training.sh \
  --wandb-key YOUR_KEY \
  --hf-token YOUR_TOKEN
```

## Troubleshooting

### OOM (Out of Memory)
- Reduzir `per_device_train_batch_size`
- Usar `gradient_accumulation_steps` maior
- Usar instância maior (g5.2xlarge)

### Low Valid Rate
- Verificar formato de dados (deve ser JSON)
- Aumentar `num_train_epochs`
- Verificar conversão de dados

### Early Stopping Prematuro
- Aumentar `early_stopping_patience`
- Verificar validation loss

## Próximos Modelos (Planejados)

### GPT-Neo (EleutherAI)
- 125M, 1.3B, 2.7B params
- Similar ao GPT-2
- Compatível com mesma pipeline

### LLaMA 2/3 (Meta)
- 7B, 13B, 70B params
- Melhor performance
- Requer mais VRAM

### Phi-2/3 (Microsoft)
- 2.7B params
- Otimizado para reasoning
- Bom para symbolic tasks

## Referências

- LoRA: https://arxiv.org/abs/2106.09685
- PPO: https://arxiv.org/abs/1707.06347
- GRPO: DeepSeek-R1 technical report
- Dataset: https://huggingface.co/datasets/augustocsc/sintetico_natural
