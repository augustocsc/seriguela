# Plano de Treinamento: Modelos Prefix (Base, Medium, Large)

## Objetivo

Treinar três modelos GPT-2 (124M, 355M, 774M parâmetros) usando o **dataset prefix recém-criado** (`augustocsc/sintetico_natural_prefix_682k`) para comparar:

1. **Infix vs Prefix**: Desempenho de modelos treinados em notação infix vs prefix na mesma tarefa
2. **Model Scaling em Prefix**: Impacto do tamanho do modelo na geração de expressões em notação prefix
3. **Reprodutibilidade**: Garantir que os modelos prefix sejam comparáveis aos modelos infix existentes

## Dataset

**Nome**: `augustocsc/sintetico_natural_prefix_682k`

**Splits**:
- Train: 682,429 exemplos (90%)
- Validation: 75,826 exemplos (10%)
- **Total**: 758,255 expressões em notação prefix

**Formato**: JSON (EXP-A)
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "+ * x_1 sin x_2"}
```

**Vantagens**:
- ✅ Mesmo split usado no treinamento dos modelos infix (seed=42)
- ✅ 100% conversão bem-sucedida de infix para prefix
- ✅ Permite comparação direta infix vs prefix
- ✅ Seed fixo = reprodutibilidade perfeita

## Modelos a Treinar

### 1. GPT-2 Base (124M parâmetros)

**Nome de output**: `gpt2_base_prefix_682k`

**Configuração**:
- Model: `gpt2` (124M parâmetros)
- LoRA: r=8, alpha=32, target_modules=["c_attn"]
- Batch size: 8 per device
- Gradient accumulation: 4 steps
- Effective batch size: 32
- Learning rate: 5e-5
- Epochs: 3 (com early stopping, patience=3)
- AWS Instance: **g5.xlarge** (NVIDIA A10G, 24GB VRAM)
- Tempo estimado: 2-3 horas
- Custo estimado: $2-3 USD

**Wandb run name**:
```python
seriguela-supervised-base-prefix682k-YYYYMMDD-HHMMSS
```

### 2. GPT-2 Medium (355M parâmetros)

**Nome de output**: `gpt2_medium_prefix_682k`

**Configuração**:
- Model: `gpt2-medium` (355M parâmetros)
- LoRA: r=8, alpha=32, target_modules=["c_attn"]
- Batch size: 4 per device
- Gradient accumulation: 4 steps
- Effective batch size: 16
- Learning rate: 5e-5
- Epochs: 3 (com early stopping, patience=3)
- AWS Instance: **g5.xlarge** (NVIDIA A10G, 24GB VRAM)
- Tempo estimado: 3-4 horas
- Custo estimado: $3-4 USD

**Wandb run name**:
```python
seriguela-supervised-medium-prefix682k-YYYYMMDD-HHMMSS
```

### 3. GPT-2 Large (774M parâmetros)

**Nome de output**: `gpt2_large_prefix_682k`

**Configuração**:
- Model: `gpt2-large` (774M parâmetros)
- LoRA: r=8, alpha=32, target_modules=["c_attn"]
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Effective batch size: 8
- Learning rate: 5e-5
- Epochs: 3 (com early stopping, patience=3)
- AWS Instance: **g5.2xlarge** (NVIDIA A10G, 48GB VRAM)
- Tempo estimado: 4-5 horas
- Custo estimado: $5-6 USD

**Wandb run name**:
```python
seriguela-supervised-large-prefix682k-YYYYMMDD-HHMMSS
```

## Comando de Treinamento Base

### Treinamento Local (Para Testes)

```bash
python scripts/train_with_json.py \
  --model_name_or_path gpt2 \
  --dataset_repo_id augustocsc/sintetico_natural_prefix_682k \
  --data_column p_prompt_n_converted \
  --output_dir ./output/gpt2_base_prefix_682k \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --weight_decay 0.01 \
  --early_stopping_patience 3 \
  --fp16 \
  --seed 42 \
  --wandb_project seriguela \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_target_modules c_attn \
  --lora_dropout 0.05
```

**⚠️ IMPORTANTE**: Use a coluna `p_prompt_n_converted` (expressões convertidas), NÃO `p_prompt_n` (expressões diferentes)!

### Adaptação para Medium e Large

**Medium**:
```bash
# Trocar apenas:
--model_name_or_path gpt2-medium \
--output_dir ./output/gpt2_medium_prefix_682k \
--per_device_train_batch_size 4
```

**Large**:
```bash
# Trocar apenas:
--model_name_or_path gpt2-large \
--output_dir ./output/gpt2_large_prefix_682k \
--per_device_train_batch_size 2
```

## Scripts AWS (Criar)

### 1. `scripts/aws/launch_base_prefix_training.sh`

```bash
#!/bin/bash
# Launch GPT-2 Base training with prefix dataset on AWS

set -e

# Parse arguments
HF_TOKEN=""
WANDB_KEY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --wandb-key) WANDB_KEY="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
  echo "Usage: $0 --hf-token <token> --wandb-key <key>"
  exit 1
fi

# Instance configuration
INSTANCE_TYPE="g5.xlarge"
IMAGE_ID="ami-0c7217cdde317cfec"  # Ubuntu Deep Learning AMI
KEY_NAME="chave-gpu"
SECURITY_GROUP="sg-0deaa73e23482e3f6"
VOLUME_SIZE=100

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
INSTANCE_NAME="seriguela-base-prefix-training-${TIMESTAMP}"

echo "Launching instance: $INSTANCE_NAME"
echo "Instance type: $INSTANCE_TYPE"

# Create user data script
cat > /tmp/userdata_base_prefix.sh <<'EOF'
#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

sleep 5

# Install dependencies
apt-get update
apt-get install -y python3-pip git

# Clone repository
cd /home/ubuntu
git clone https://github.com/augustocsc/seriguela.git
cd seriguela

# Install Python dependencies
pip3 install -r requirements.txt
pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Setup credentials
echo "huggingface = ${HF_TOKEN}" > /home/ubuntu/.tokens.txt
echo "wandb = ${WANDB_KEY}" >> /home/ubuntu/.tokens.txt
chmod 600 /home/ubuntu/.tokens.txt

# Login to services
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_KEY}"
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_KEY

# Start training
python3 scripts/train_with_json.py \
  --model_name_or_path gpt2 \
  --dataset_repo_id augustocsc/sintetico_natural_prefix_682k \
  --data_column p_prompt_n_converted \
  --output_dir ./output/gpt2_base_prefix_682k \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --weight_decay 0.01 \
  --early_stopping_patience 3 \
  --fp16 \
  --seed 42 \
  --wandb_project seriguela \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_target_modules c_attn \
  --lora_dropout 0.05 \
  > /home/ubuntu/training_base_prefix.log 2>&1

# Mark completion
touch /home/ubuntu/.training_complete
echo "Training complete at $(date)" >> /home/ubuntu/.training_complete

# Upload model to HuggingFace
cd output/gpt2_base_prefix_682k
huggingface-cli repo create gpt2_base_prefix_682k --type model || true
huggingface-cli upload gpt2_base_prefix_682k . . --commit-message "GPT-2 Base trained on prefix dataset (682K)"

EOF

# Replace tokens in user data
sed -i "s|\${HF_TOKEN}|${HF_TOKEN}|g" /tmp/userdata_base_prefix.sh
sed -i "s|\${WANDB_KEY}|${WANDB_KEY}|g" /tmp/userdata_base_prefix.sh

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $IMAGE_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\"}}]" \
  --user-data file:///tmp/userdata_base_prefix.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to start..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "============================================"
echo "Instance ready!"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "============================================"
echo "To monitor training:"
echo "  ssh -i ~/.ssh/chave-gpu.pem ubuntu@${PUBLIC_IP}"
echo "  tail -f ~/training_base_prefix.log"
echo ""
echo "To download model after training:"
echo "  scp -i ~/.ssh/chave-gpu.pem -r ubuntu@${PUBLIC_IP}:~/seriguela/output/gpt2_base_prefix_682k ./"
echo ""
echo "IMPORTANT: Stop instance when done!"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "============================================"

rm /tmp/userdata_base_prefix.sh
```

### 2. `scripts/aws/launch_medium_prefix_training.sh`

Copiar `launch_base_prefix_training.sh` e modificar:
- `INSTANCE_NAME="seriguela-medium-prefix-training-${TIMESTAMP}"`
- `--model_name_or_path gpt2-medium`
- `--output_dir ./output/gpt2_medium_prefix_682k`
- `--per_device_train_batch_size 4`
- `training_medium_prefix.log`
- `gpt2_medium_prefix_682k` no upload

### 3. `scripts/aws/launch_large_prefix_training.sh`

Copiar `launch_base_prefix_training.sh` e modificar:
- `INSTANCE_TYPE="g5.2xlarge"` ⚠️ Importante!
- `INSTANCE_NAME="seriguela-large-prefix-training-${TIMESTAMP}"`
- `--model_name_or_path gpt2-large`
- `--output_dir ./output/gpt2_large_prefix_682k`
- `--per_device_train_batch_size 2`
- `training_large_prefix.log`
- `gpt2_large_prefix_682k` no upload

### 4. `launch_all_prefix_models.sh` (Raiz do projeto)

```bash
#!/bin/bash
# Launch all 3 prefix models in parallel on AWS

set -e

# Load credentials
if [ -f ~/.tokens.txt ]; then
  WANDB_KEY=$(grep wandb ~/.tokens.txt | cut -d= -f2 | tr -d ' ')
  HF_TOKEN=$(grep huggingface ~/.tokens.txt | cut -d= -f2 | tr -d ' ')
else
  echo "Error: ~/.tokens.txt not found"
  exit 1
fi

if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
  echo "Error: Tokens not found in ~/.tokens.txt"
  exit 1
fi

echo "============================================"
echo "Launching 3 prefix models in parallel"
echo "============================================"

# Launch Base (g5.xlarge)
echo "Launching Base (124M)..."
bash scripts/aws/launch_base_prefix_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > launch_base_prefix.log 2>&1 &
BASE_PID=$!

sleep 10  # Stagger launches to avoid AWS rate limits

# Launch Medium (g5.xlarge)
echo "Launching Medium (355M)..."
bash scripts/aws/launch_medium_prefix_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > launch_medium_prefix.log 2>&1 &
MEDIUM_PID=$!

sleep 10

# Launch Large (g5.2xlarge)
echo "Launching Large (774M)..."
bash scripts/aws/launch_large_prefix_training.sh \
  --wandb-key "$WANDB_KEY" \
  --hf-token "$HF_TOKEN" \
  > launch_large_prefix.log 2>&1 &
LARGE_PID=$!

echo ""
echo "All launches initiated!"
echo "Base PID: $BASE_PID"
echo "Medium PID: $MEDIUM_PID"
echo "Large PID: $LARGE_PID"
echo ""
echo "Check logs:"
echo "  tail -f launch_base_prefix.log"
echo "  tail -f launch_medium_prefix.log"
echo "  tail -f launch_large_prefix.log"
echo ""

# Wait for all to complete
wait $BASE_PID $MEDIUM_PID $LARGE_PID

echo ""
echo "============================================"
echo "All instances launched successfully"
echo "============================================"

# Show running instances
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-*-prefix-training-*" \
           "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress,Tags[?Key=='Name'].Value|[0]]" \
  --output table

echo ""
echo "⚠️  IMPORTANT: Remember to STOP all instances when training completes!"
echo ""
echo "To stop all:"
echo "  aws ec2 stop-instances --instance-ids \$(aws ec2 describe-instances \\"
echo "    --filters 'Name=tag:Name,Values=seriguela-*-prefix-training-*' \\"
echo "              'Name=instance-state-name,Values=running' \\"
echo "    --query 'Reservations[*].Instances[*].InstanceId' --output text)"
```

## Comparação Infix vs Prefix

Após treinar os modelos prefix, comparar com os modelos infix existentes:

### Modelos Infix (Já Treinados)
- `augustocsc/Se124M_700K_infix_v3_json` (Base)
- `output/gpt2_medium_700K_json` (Medium)
- `output/gpt2_large_700K_json` (Large)

### Modelos Prefix (A Treinar)
- `output/gpt2_base_prefix_682k` (Base)
- `output/gpt2_medium_prefix_682k` (Medium)
- `output/gpt2_large_prefix_682k` (Large)

### Script de Comparação

**Criar**: `scripts/compare_infix_vs_prefix.py`

```python
#!/usr/bin/env python3
"""
Compare infix vs prefix models on Nguyen benchmarks.
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import numpy as np

def evaluate_model(model_path, dataset_path, num_samples=100):
    """Evaluate a model on a benchmark dataset."""
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Generate expressions and evaluate
    results = []
    # ... (evaluation logic)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infix_base", required=True)
    parser.add_argument("--infix_medium", required=True)
    parser.add_argument("--infix_large", required=True)
    parser.add_argument("--prefix_base", required=True)
    parser.add_argument("--prefix_medium", required=True)
    parser.add_argument("--prefix_large", required=True)
    parser.add_argument("--benchmark", default="data/benchmarks/nguyen/nguyen_5.csv")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    # Evaluate all models
    results = {
        "infix_base": evaluate_model(args.infix_base, args.benchmark, args.num_samples),
        "infix_medium": evaluate_model(args.infix_medium, args.benchmark, args.num_samples),
        "infix_large": evaluate_model(args.infix_large, args.benchmark, args.num_samples),
        "prefix_base": evaluate_model(args.prefix_base, args.benchmark, args.num_samples),
        "prefix_medium": evaluate_model(args.prefix_medium, args.benchmark, args.num_samples),
        "prefix_large": evaluate_model(args.prefix_large, args.benchmark, args.num_samples),
    }

    # Generate comparison table
    df = pd.DataFrame(results).T
    print(df)
    df.to_csv("infix_vs_prefix_comparison.csv")

if __name__ == "__main__":
    main()
```

**Uso**:
```bash
python scripts/compare_infix_vs_prefix.py \
  --infix_base augustocsc/Se124M_700K_infix_v3_json \
  --infix_medium ./output/gpt2_medium_700K_json \
  --infix_large ./output/gpt2_large_700K_json \
  --prefix_base ./output/gpt2_base_prefix_682k \
  --prefix_medium ./output/gpt2_medium_prefix_682k \
  --prefix_large ./output/gpt2_large_prefix_682k \
  --benchmark data/benchmarks/nguyen/nguyen_5.csv
```

## Métricas de Avaliação

### 1. Qualidade Básica
- Valid expression rate (%)
- Parseable rate (%)
- Constraint adherence (%)
- Diversity rate (%)

### 2. Complexidade
- Power operations usage (%)
- Nested function usage (%)
- Average expression depth
- Operator distribution

### 3. Performance em Benchmarks
- Best R² achieved
- Mean R² (valid expressions only)
- Valid rate on benchmark
- Convergence rate

### 4. Comparação Infix vs Prefix
- **Hipótese**: Prefix deve ter maior taxa de validade (sintaxe mais simples)
- **Hipótese**: Infix e Prefix devem ter R² similar (mesma capacidade expressiva)
- **Hipótese**: Prefix pode ter parsing mais rápido (sem precedência de operadores)

## Cronograma Estimado

### Fase 1: Preparação (Dia 1 - Manhã)
- ✅ Conversão dataset concluída (682K prefix)
- ✅ Upload para HuggingFace (pendente autenticação)
- ⏳ Criar scripts AWS (3 scripts + 1 launcher)
- ⏳ Testar localmente (1 epoch, batch pequeno)

**Tempo estimado**: 2-3 horas

### Fase 2: Treinamento Paralelo (Dia 1 - Tarde)
- ⏳ Lançar 3 instâncias AWS em paralelo
- ⏳ Monitorar progresso via Wandb
- ⏳ Esperar conclusão (~4-5h devido ao Large)

**Tempo estimado**: 5-6 horas (paralelo)

### Fase 3: Download e Validação (Dia 2 - Manhã)
- ⏳ Download dos 3 modelos
- ⏳ **PARAR INSTÂNCIAS** imediatamente
- ⏳ Testar geração local
- ⏳ Validar métricas básicas

**Tempo estimado**: 1-2 horas

### Fase 4: Avaliação e Comparação (Dia 2 - Tarde)
- ⏳ Avaliar qualidade (valid rate, complexity)
- ⏳ Comparar infix vs prefix
- ⏳ Testar em Nguyen benchmarks
- ⏳ Gerar tabelas e gráficos

**Tempo estimado**: 3-4 horas

### Fase 5: Documentação (Dia 3)
- ⏳ Criar model cards (3 modelos)
- ⏳ Escrever relatório de comparação
- ⏳ Upload para HuggingFace
- ⏳ Atualizar CLAUDE.md

**Tempo estimado**: 2-3 horas

**Total**: ~2-3 dias (com treinamento paralelo)

## Custos Estimados

| Item | Horas | Custo/hora | Custo Total |
|------|-------|-----------|-------------|
| Base (g5.xlarge) | 2-3h | $1.006 | $2-3 |
| Medium (g5.xlarge) | 3-4h | $1.006 | $3-4 |
| Large (g5.2xlarge) | 4-5h | $1.212 | $5-6 |
| **Total Treinamento** | | | **$10-13 USD** |

**⚠️ IMPORTANTE**: Com early stopping, pode ser 20-30% mais barato se convergir antes de 3 épocas.

## Checklist de Execução

### Pré-Treinamento
- ✅ Dataset convertido (682,429 + 75,826)
- ⏳ Upload para HuggingFace Hub
- ⏳ Criar 3 scripts AWS
- ⏳ Criar script launcher
- ⏳ Testar localmente (1 batch)
- ⏳ Verificar credenciais (~/.tokens.txt)

### Durante Treinamento
- ⏳ Lançar 3 instâncias em paralelo
- ⏳ Verificar Wandb (3 runs iniciados)
- ⏳ Monitorar loss (deve cair < 0.4 após epoch 1)
- ⏳ SSH em uma instância para testar acesso

### Pós-Treinamento (CRÍTICO)
- ⏳ **PARAR TODAS AS INSTÂNCIAS**
- ⏳ Verificar que foram paradas (aws ec2 describe-instances)
- ⏳ Download dos 3 modelos
- ⏳ Testar geração local
- ⏳ Registrar custos finais

### Avaliação
- ⏳ Valid rate > 75% (esperado ~80% como infix)
- ⏳ Comparar com modelos infix
- ⏳ Testar Nguyen-5 benchmark
- ⏳ Análise de complexidade

### Publicação
- ⏳ Upload para HuggingFace (3 modelos)
- ⏳ Criar 3 model cards
- ⏳ Escrever relatório comparativo
- ⏳ Git commit final

## Resultados Esperados

### Hipóteses

**H1 (Valid Rate)**: Prefix ≥ Infix
- **Razão**: Sintaxe prefix sem parênteses é mais simples
- **Baseline Infix**: 80% valid
- **Esperado Prefix**: 80-85% valid

**H2 (R² Performance)**: Prefix ≈ Infix
- **Razão**: Mesma capacidade expressiva, apenas notação diferente
- **Baseline Infix**: R² = -1.0 em Nguyen-5 (Base)
- **Esperado Prefix**: Similar (mesmos problemas de complexidade)

**H3 (Model Scaling)**: Large > Medium > Base
- **Razão**: Modelos maiores capturam melhor composições complexas
- **Esperado**: Depth aumenta (1.4 → 2.0+), power ops aumentam (16% → 40%+)

**H4 (Parsing Speed)**: Prefix > Infix
- **Razão**: Prefix não requer precedência de operadores
- **Esperado**: 10-20% mais rápido na conversão para SymPy

### Tabela de Resultados (Preencher Após Treinamento)

| Métrica | Infix Base | Prefix Base | Infix Medium | Prefix Medium | Infix Large | Prefix Large |
|---------|-----------|-------------|--------------|---------------|-------------|--------------|
| Valid Rate (%) | 80 | ? | ? | ? | ? | ? |
| Power Ops (%) | 15.9 | ? | ? | ? | ? | ? |
| Avg Depth | 1.40 | ? | ? | ? | ? | ? |
| Nested Trig (%) | 0 | ? | ? | ? | ? | ? |
| Best R² (N-5) | -1.0 | ? | ? | ? | ? | ? |

## Próximos Passos Após Treinamento

1. **Se Prefix > Infix**: Considerar usar prefix como padrão para futuros modelos
2. **Se Prefix ≈ Infix**: Escolha depende da aplicação (parsing vs legibilidade)
3. **Se Prefix < Infix**: Investigar se problema está no dataset ou na arquitetura

## Arquivos a Criar

1. ✅ `TRAINING_PLAN_PREFIX_MODELS.md` (este arquivo)
2. ⏳ `scripts/aws/launch_base_prefix_training.sh`
3. ⏳ `scripts/aws/launch_medium_prefix_training.sh`
4. ⏳ `scripts/aws/launch_large_prefix_training.sh`
5. ⏳ `launch_all_prefix_models.sh` (raiz)
6. ⏳ `scripts/compare_infix_vs_prefix.py`
7. ⏳ `TRAINING_LOG_PREFIX_MODELS_2025.md` (após treinamento)
8. ⏳ `EXPERIMENT_INFIX_VS_PREFIX.md` (relatório final)
9. ⏳ `model_cards/gpt2_base_prefix_682k_card.md`
10. ⏳ `model_cards/gpt2_medium_prefix_682k_card.md`
11. ⏳ `model_cards/gpt2_large_prefix_682k_card.md`

---

**Criado**: 2026-02-09
**Versão**: 1.0
**Status**: Planejamento completo, aguardando execução
