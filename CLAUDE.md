# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Seriguela** is an **academic research project** focused on training language models for symbolic regression through fine-tuning and reinforcement learning. The project trains GPT-2 models to generate valid mathematical expressions using LoRA (parameter-efficient fine-tuning) and explores RL algorithms (PPO, GRPO, REINFORCE) for optimizing expression quality.

**Research Context**: This is a graduate-level research project exploring the application of large language models to symbolic regression problems, with focus on:
- Parameter-efficient fine-tuning techniques (LoRA)
- Reinforcement learning for expression optimization
- Model scaling effects on compositional complexity
- Benchmark evaluation (Nguyen benchmarks)

**Current Status:** The JSON structured format (EXP-A) is the recommended approach, achieving 80% valid expressions vs 0.5% with EOS token approach. See `EXPERIMENT_RESULTS.md` for details.

## Core Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv .seriguela
source .seriguela/bin/activate  # Linux/macOS
.seriguela\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 (required for GPU)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Login to experiment tracking (optional)
wandb login
```

### Data Preparation

```bash
# Prepare experiment data (JSON + EOS formats)
python scripts/data/prepare_experiment_data.py \
  --dataset_repo_id augustocsc/sintetico_natural \
  --data_dir 700K \
  --data_column i_prompt_n \
  --output_base_dir ./data/experiments

# Legacy: Add end-of-expression markers (simple format)
python scripts/data/prepare_training_data_fixed.py \
  --dataset_repo_id augustocsc/sintetico_natural \
  --data_dir 700K \
  --data_column i_prompt_n \
  --output_dir ./data/processed/700K_fixed
```

### Training

```bash
# RECOMMENDED: Train with JSON format (EXP-A) - 80% valid expressions
python scripts/train_experiment.py \
  --experiment_name exp_a_json \
  --train_file ./data/experiments/exp_a_json/train.csv \
  --output_dir ./output/exp_a_json \
  --num_train_epochs 3

# Alternative: Train with EOS format (EXP-B) - not recommended, 0.5% valid
python scripts/train_experiment.py \
  --experiment_name exp_b_eos \
  --train_file ./data/experiments/exp_b_eos/train.csv \
  --output_dir ./output/exp_b_eos \
  --use_native_eos

# Legacy: Basic training with original format
python scripts/train.py \
  --model_name_or_path gpt2 \
  --dataset_repo_id augustocsc/sintetico_natural \
  --data_dir 700K \
  --data_column i_prompt_n \
  --output_dir ./output/Se124M_700K_infix \
  --wandb_project seriguela \
  --num_train_epochs 3

# Training with config file
python scripts/train.py --config configs/training_medium.json
```

### Generation & Evaluation

```bash
# Evaluate experiment results
python scripts/evaluate_experiments.py \
  --model_path ./output/exp_a_json \
  --num_samples 200

# Generate expressions
python scripts/generate.py \
  --model_path ./output/exp_a_json \
  --num_generations 50 \
  --validate

# Evaluate model
python scripts/evaluate.py \
  --model_path ./output/exp_a_json \
  --num_samples 100

# Compare two models
python scripts/compare_models.py \
  --model1 ./output/model_v1 \
  --model2 ./output/model_v2
```

### AWS Infrastructure

**⚠️ CONVENÇÃO DE NOMES IMPORTANTE**: Todas as instâncias AWS criadas por este projeto usam o prefixo **"augusto-"** para evitar conflitos e garantir que apenas suas instâncias sejam gerenciadas.

#### Scripts Seguros de Gerenciamento

```bash
# ✅ RECOMENDADO: Listar apenas suas instâncias (prefixo "augusto-")
bash scripts/aws/list_augusto_instances.sh

# ✅ RECOMENDADO: Parar TODAS as suas instâncias (pede confirmação)
bash scripts/aws/stop_augusto_instances.sh

# ✅ RECOMENDADO: Parar TODAS as suas instâncias (sem confirmação)
bash scripts/aws/stop_augusto_instances.sh --confirm

# ✅ RECOMENDADO: Parar instância específica (valida prefixo "augusto-")
bash scripts/aws/stop_instance.sh INSTANCE_ID
```

#### Lançar Instâncias de Treinamento

Todas as instâncias criadas automaticamente recebem o prefixo "augusto-seriguela-":

```bash
# Base (124M) - g5.xlarge
bash scripts/aws/launch_base_training.sh \
  --hf-token <token> \
  --wandb-key <key>
# Cria: augusto-seriguela-base-training

# Medium (355M) - g5.xlarge
bash scripts/aws/launch_medium_training.sh \
  --hf-token <token> \
  --wandb-key <key>
# Cria: augusto-seriguela-medium-training

# Large (774M) - g5.2xlarge
bash scripts/aws/launch_large_training.sh \
  --hf-token <token> \
  --wandb-key <key>
# Cria: augusto-seriguela-large-training

# Avaliação completa - g5.2xlarge
bash scripts/aws/launch_comprehensive_evaluation.sh \
  --hf-token <token> \
  --wandb-key <key>
# Cria: augusto-seriguela-comprehensive-eval
```

#### Monitoramento

```bash
# Monitor training remotely
bash scripts/aws/monitor_training_auto.sh

# SSH para instância (use IP do list_augusto_instances.sh)
ssh -i ~/.ssh/chave-gpu-nova.pem ubuntu@<PUBLIC_IP>

# Verificar progresso do treinamento
tail -f ~/seriguela/training_*.log
```

#### ⚠️ SEGURANÇA

**NUNCA use comandos AWS genéricos sem filtro de nome!** Isso pode afetar instâncias de outras pessoas:

```bash
# ❌ PERIGOSO - Pode listar/parar instâncias de outros
aws ec2 describe-instances
aws ec2 stop-instances --instance-ids i-xxx

# ✅ SEGURO - Usa scripts com filtro "augusto-"
bash scripts/aws/list_augusto_instances.sh
bash scripts/aws/stop_augusto_instances.sh
```

#### Custos e Limpeza

**SEMPRE pare instâncias após uso** para evitar custos desnecessários:

```bash
# Verificar se há instâncias rodando
bash scripts/aws/list_augusto_instances.sh

# Se houver instâncias rodando, pare-as
bash scripts/aws/stop_augusto_instances.sh --confirm
```

Custos típicos (us-east-1):
- **g5.xlarge** (24GB VRAM): ~$1.01/hora
- **g5.2xlarge** (48GB VRAM): ~$1.21/hora

## Architecture

### Key Components

1. **Data Pipeline** (`scripts/data/`)
   - `prepare_experiment_data.py`: Prepares data in JSON (EXP-A) and EOS (EXP-B) formats
   - `prepare_training_data_fixed.py`: Legacy script that adds `<|endofex|>` markers
   - `data_processing.py`: Cleaning, augmentation, format conversion
   - Supports both HuggingFace Hub and local CSV files

2. **Training** (`scripts/train_experiment.py`, `scripts/train.py`)
   - Uses GPT-2 (124M parameters) with LoRA adapters (only 294K trainable)
   - `train_experiment.py`: Supports JSON format (recommended) and EOS format
   - Integrates with Weights & Biases for experiment tracking
   - Validates dataset format before training

3. **Generation** (`scripts/generate.py`)
   - `ExpressionStoppingCriteria`: Custom stopping logic for clean expression boundaries
   - Multi-strategy extraction: marker-based → boundary-based → fallback
   - Validates expressions using SymPy

4. **Expression Parsing** (`classes/expression.py`)
   - `parse_prefix()`: Stack-based prefix notation parser
   - `parse_infix()`: Infix notation parser
   - `validate()`: Syntax and semantic validation
   - `OPERATOR_ARITY`: Maps operators to argument counts

### Data Flow

```
HuggingFace Hub / Local CSV
  ↓ prepare_experiment_data.py (converts to JSON or EOS format)
Train/Val/Test CSV files (in data/experiments/)
  ↓ train_experiment.py (tokenize + LoRA)
output/{experiment_name}/ (checkpoints + final model)
  ↓ evaluate_experiments.py & generate.py (inference + validation)
```

### LoRA Configuration

- **r=8**: LoRA rank
- **alpha=32**: LoRA alpha scaling
- **target_modules**: `c_attn` (attention layers only)
- **dropout=0.05**: LoRA dropout rate
- Results in only 294K trainable parameters vs 124M total

## Important Patterns

### Data Format (JSON - Recommended)

The JSON format (EXP-A) is the recommended approach, providing 80% valid expressions:

**Training data format:**
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "sin(x_1 + C*x_2)"}
```

**Inference prompt:**
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "
```

The model completes the expression and closes with `"}`.

### Why JSON Works Better Than EOS Token

1. **Clear boundaries**: JSON has explicit `{` start and `}` end markers
2. **Structured containment**: Expression is within `"expr": "..."` field
3. **Lower loss**: Easier for model to learn (0.343 vs 0.415)
4. **No repetition**: Unlike EOS approach, model doesn't fall into repetitive patterns

### Expression Validation

```python
from classes.expression import Expression

# Create and validate
expr = Expression.parse_prefix("+", "x", "y")  # Prefix notation
expr = Expression.parse_infix("x + y")         # Infix notation
is_valid = expr.validate()

# Add new operators
# Edit OPERATOR_ARITY and OPERATOR_FUNCS in classes/expression.py
```

### Configuration Files

Training parameters can be set via:
1. CLI arguments (highest priority)
2. JSON config files in `configs/` (e.g., `training_medium.json`)
3. Default values in code

Example config structure:
```json
{
  "model_name_or_path": "gpt2",
  "output_dir": "./output/experiment",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 8,
  "learning_rate": 5e-5,
  "lora_r": 8,
  "lora_alpha": 32
}
```

## Experiment Results Summary

| Metric | EXP-A (JSON) | EXP-B (EOS) |
|--------|--------------|-------------|
| Valid Expressions | **80%** | 0.5% |
| Parseable | 81% | 4.5% |
| Correct Symbols | 76.5% | 11% |
| Train Loss | 0.343 | 0.415 |

Full results in `EXPERIMENT_RESULTS.md`.

## Known Issues & Fixes

### Solved: Expression Generation Stopping

The original problem (models not stopping at expression boundaries) has been solved by using the JSON structured format. The EOS token approach (`<|endoftext|>`) proved ineffective (0.5% valid).

### Historical Bug Fixes

1. **Missing parse_prefix() Method**: Added stack-based implementation in `classes/expression.py:712`.

2. **Wandb Version**: Old version (0.19.9) doesn't support new API key format. Update to `wandb>=0.24.1` in `requirements.txt`.

### GPU Environment

- **Local Windows**: May have PyTorch DLL issues. Recommend testing on AWS.
- **AWS**: Use g5.xlarge instances with NVIDIA A10G GPU (CUDA 12.1).
- Verify GPU: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

### AWS Security Group

Current security group (sg-0deaa73e23482e3f6) restricts SSH access to specific IPs:
- 143.106.58.120/32
- 179.160.37.193/32

To add your IP:
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id sg-0deaa73e23482e3f6 \
  --protocol tcp --port 22 --cidr $MY_IP/32
```

## File Locations

### Critical Files

- `scripts/train_experiment.py` - Recommended training script for experiments
- `scripts/data/prepare_experiment_data.py` - Prepares JSON/EOS format data
- `scripts/evaluate_experiments.py` - Evaluates experiment results
- `classes/expression.py:712+` - Expression parsing and validation core
- `scripts/train.py` - Legacy training entry point
- `scripts/generate.py` - Generation with stopping criteria

### Output Directories

- `output/{experiment_name}/` - Model checkpoints and final models
- `data/experiments/exp_a_json/` - JSON format training data
- `data/experiments/exp_b_eos/` - EOS format training data (not recommended)
- `data/processed/` - Legacy prepared training data
- `wandb/` - Weights & Biases experiment metadata (gitignored)
- `aws/keys/` - SSH keys for EC2 instances (gitignored)

### Configuration

- `configs/training_*.json` - Training hyperparameters for different scales
- `configs/wandb_config.py` - Wandb naming standards and utilities
- `aws/config.json` - AWS instance configurations
- `CREDENTIALS_SETUP.md` - Guide for API tokens and SSH keys setup
- `WANDB_NAMING.md` - Wandb experiment naming conventions
- `.gitignore` - Excludes `*.pem`, `*.key`, `aws/.env`, `output/*`, `data/*`

## Dependencies

### Core ML Stack
- `transformers==4.51.3` - Model loading, tokenizer, Trainer
- `torch==2.5.1+cu121` - Deep learning with CUDA 12.1
- `peft==0.15.1` - LoRA parameter-efficient fine-tuning
- `datasets==3.5.0` - HuggingFace dataset loading
- `accelerate==1.6.0` - Multi-GPU training

### Experiment Tracking
- `wandb>=0.24.1` - Experiment tracking (note: version must be >=0.24.1)
- `tensorboard==2.16.2` - Alternative visualization
- `trl==0.16.1` - Advanced training techniques

### Validation & Analysis
- `sympy==1.13.1` - Symbolic math for expression validation
- `pandas==2.2.1` - Data manipulation
- `scikit-learn==1.6.1` - Metrics and evaluation

## Extensibility Points

### Adding New Operators

Edit `classes/expression.py`:
```python
OPERATOR_ARITY = {
    # Existing operators...
    'new_op': 2,  # binary operator
}

OPERATOR_FUNCS = {
    'new_op': np.new_function,
}
```

### Custom Metrics

Extend `scripts/evaluate.py` with new validation functions:
```python
def custom_metric(expression):
    # Your validation logic
    return score
```

### Different Model Base

Pass `--model_name_or_path` to `train.py`:
```bash
python scripts/train.py --model_name_or_path gpt2-medium  # 355M params
python scripts/train.py --model_name_or_path gpt2-large   # 774M params
```

### New Data Format

Add converter in `scripts/data/data_processing.py` for custom notation or expression formats.

## Best Practices

1. **Always validate data**: Ensure `<|endofex|>` markers exist before training
2. **Use config files**: Store hyperparameters in `configs/` for reproducibility
3. **Track experiments**: Enable W&B logging with standardized naming (see `WANDB_NAMING.md`)
4. **Use standard Wandb names**: Import from `configs/wandb_config.py` for consistent naming
5. **Test locally first**: Run small experiments before AWS deployment
6. **Monitor GPU usage**: Check `nvidia-smi` and training logs for memory issues
7. **Stop AWS instances**: Always stop instances when not in use to avoid charges
8. **Version control**: Commit config files but never commit model weights or keys

## Quick Debugging

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Verify JSON format data
python -c "import pandas as pd; df = pd.read_csv('./data/experiments/exp_a_json/train.csv'); print(df['text'].iloc[0])"

# Test expression parsing
python -c "from classes.expression import Expression; expr = Expression.parse_infix('x + y'); print(expr.validate())"

# Check model output structure
python scripts/generate.py --model_path ./output/exp_a_json --num_generations 1 --validate

# Monitor training on AWS
ssh -i ~/chave-gpu.pem ubuntu@<ip>
tail -f ~/training_exp_a.log
```

## Wandb Naming Standards

**Complete guide**: See `WANDB_NAMING.md` for detailed naming conventions.

### Standard Format

All Wandb runs follow the pattern: `seriguela-{type}-{model}-{dataset}-{timestamp}`

### Quick Usage

```python
from configs.wandb_config import generate_run_name, get_wandb_project_name

# Generate standardized run name
run_name = generate_run_name("ppo", "medium", "nguyen5")
# → seriguela-ppo-medium-nguyen5-20260203-143022

# Initialize wandb
wandb.init(
    project=get_wandb_project_name(),  # → "seriguela"
    name=run_name
)
```

### Common Examples

```python
# Supervised training
generate_run_name("supervised", "base", "700K")
# → seriguela-supervised-base-700k-20260203-143022

# PPO reinforcement learning
generate_run_name("ppo", "medium", "nguyen5")
# → seriguela-ppo-medium-nguyen5-20260203-143022

# GRPO with extra info
generate_run_name("grpo", "large", "nguyen7", "lr5e5")
# → seriguela-grpo-large-nguyen7-lr5e5-20260203-143022
```

**Project name**: Always use `"seriguela"` for production experiments.

---

## Reinforcement Learning for Expression Generation (Feb 2025)

### Problem Discovery: Complexity Gap

After implementing supervised fine-tuning with JSON format (80% valid expressions), RL experiments revealed a critical limitation:

**The base model (GPT-2 124M) generates structurally simple expressions that fail on complex benchmarks.**

#### Evidence from Nguyen-5 Benchmark

Target: `sin(x_1**2)*cos(x_1) - 1`

Analyzing 160 expressions from REINFORCE training:
- **Valid expressions**: 39.4%
- **All valid expressions had R² = -1.0** (terrible fit)
- **Only 15.9%** used power operations (x²)
- **0%** had nested trigonometric functions (sin(x²))
- **Average depth**: 1.40 (target requires 2+)
- **No examples** of function multiplication (sin()*cos())

**Root Cause**: Model learns to generate syntactically valid but structurally trivial expressions. Without proper complexity, all rewards are uniformly bad → no gradient signal → no learning.

### RL Algorithms Implemented

Three algorithms implemented for symbolic regression fine-tuning:

#### 1. REINFORCE (`scripts/reinforce_symbolic.py`)
- Policy gradient with EMA baseline
- Baseline: exponential moving average across epochs
- Advantage: reward - baseline
- Simple but effective for easy problems (Nguyen-1: R²=0.95)

#### 2. GRPO (`scripts/grpo_symbolic.py`)
- Group Relative Policy Optimization (DeepSeek-R1 approach)
- Within-group advantage normalization
- More stable than REINFORCE
- Better for multi-modal reward landscapes

#### 3. PPO (`scripts/ppo_symbolic.py`)
- Proximal Policy Optimization with clipped objective
- Multiple optimization epochs per batch
- KL divergence monitoring with early stopping
- **Problem**: Too conservative when all samples have poor fit
- Failed completely on Nguyen-5 (R²=-1.0)

### Debugging Tools

#### `scripts/debug_reinforce.py`
Captures ALL expressions (valid and invalid) during training for analysis:
- Tracks: expression, R², validity, error_type, error_message
- Saves to `debug_expressions.json`
- Essential for understanding model behavior

Example usage:
```bash
python scripts/debug_reinforce.py \
  --model_path augustocsc/Se124M_700K_infix_v3_json \
  --dataset data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 10
```

#### `scripts/analyze_complexity.py`
Analyzes expression complexity patterns:
- Power operation usage (x²)
- Nested trigonometric functions
- Expression depth (nesting level)
- Operator distribution

Results show base model generates shallow expressions (depth 1.4) with no nesting.

### Solution: Scale Up Model Size

**Hypothesis**: Larger models have more capacity to learn complex compositional patterns.

Implemented training for three model sizes:
- **GPT-2 Base**: 124M parameters (baseline)
- **GPT-2 Medium**: 355M parameters (3x larger)
- **GPT-2 Large**: 774M parameters (6x larger)

### Critical Discovery: Data Format Issue (Feb 2025)

**PROBLEM FOUND**: The HuggingFace dataset (`augustocsc/sintetico_natural`) column `i_prompt_n` is NOT in JSON format!

**Actual format**:
```
vars: x_1, x_2, x_3, x_4, x_5
oper: *, abs, asin, cos, log, sin, tan
cons: C
expr: log(cos(x_4))
```

**Required JSON format** (80% valid):
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "log(cos(x_4))"}
```

This was causing models to learn wrong patterns!

### Solution: `scripts/train_with_json.py`

New training script with critical fixes:

1. **Automatic format conversion** to JSON:
```python
def convert_to_json_format(example):
    """Convert text format to JSON."""
    # Parses "vars: x_1, x_2" → {"vars": ["x_1", "x_2"]}
    # Returns proper JSON string for training
```

2. **Early stopping**:
   - Patience: 3 epochs
   - Monitors validation loss
   - Stops if no improvement → saves time and cost
   - Load best model at end

3. **Train/validation split**:
   - 90% train / 10% validation
   - Evaluation every 500 steps
   - Prevents overfitting

4. **Wandb integration** for monitoring

### Training Larger Models on AWS

#### Launch Scripts

**Medium (355M) - g5.xlarge**:
```bash
bash scripts/aws/launch_medium_training.sh \
  --wandb-key YOUR_KEY \
  --hf-token YOUR_TOKEN
```
- GPU: NVIDIA A10G (24GB VRAM)
- Time: ~2-3 hours
- Cost: ~$2-3 USD

**Large (774M) - g5.2xlarge**:
```bash
bash scripts/aws/launch_large_training.sh \
  --wandb-key YOUR_KEY \
  --hf-token YOUR_TOKEN
```
- GPU: NVIDIA A10G (48GB VRAM)
- Time: ~4-5 hours
- Cost: ~$8-10 USD

#### Monitoring

```bash
# Check training progress
ssh -i ~/.ssh/KEY.pem ubuntu@IP
tail -f /home/ubuntu/training_medium.log  # or training_large.log

# Check completion
ssh ubuntu@IP 'test -f ~/.training_complete && echo "DONE" || echo "Running"'
```

#### Download Trained Models

```bash
# Medium
scp -i ~/.ssh/KEY.pem -r ubuntu@IP:~/seriguela/output/gpt2_medium_700K_json ./

# Large
scp -i ~/.ssh/KEY.pem -r ubuntu@IP:~/seriguela/output/gpt2_large_700K_json ./
```

### Comparing Model Sizes

**`scripts/compare_trained_models.py`**: Compares models on expression complexity

```bash
python scripts/compare_trained_models.py \
  --model_base augustocsc/Se124M_700K_infix_v3_json \
  --model_medium ./gpt2_medium_700K_json \
  --model_large ./gpt2_large_700K_json \
  --dataset data/benchmarks/nguyen/nguyen_5.csv \
  --epochs 10
```

Metrics compared:
- **Valid expression %**: Syntactic correctness
- **Power operation %**: Use of x² (essential for Nguyen-5)
- **Nested trig %**: sin(x²), cos(x²), etc.
- **Average depth**: Compositional complexity
- **Best R²**: Fit quality

**Expected improvements** with larger models:
- Medium (355M): +20-30% power usage, depth 1.8-2.0
- Large (774M): +40-50% power usage, depth 2.0-2.5, possible nested trig

### Key Insights

1. **JSON format is critical**: 80% valid vs much lower with other formats
2. **Model size matters for complexity**: Base (124M) cannot generate nested compositions
3. **RL needs variance in rewards**: PPO fails when all samples are equally bad
4. **Early stopping essential**: Saves compute on AWS
5. **Validation split necessary**: Prevents overfitting on large datasets

### Files Added (Feb 2025)

**Training**:
- `scripts/train_with_json.py` - Correct training with JSON + early stopping
- `scripts/train_medium.py` - Legacy (DO NOT USE - wrong format)

**RL Algorithms**:
- `scripts/reinforce_symbolic.py` - REINFORCE implementation
- `scripts/grpo_symbolic.py` - GRPO implementation
- `scripts/ppo_symbolic.py` - PPO implementation
- `scripts/debug_reinforce.py` - Debug version capturing all expressions

**Analysis**:
- `scripts/analyze_complexity.py` - Expression complexity analysis
- `scripts/compare_trained_models.py` - Multi-model comparison
- `scripts/show_expressions.py` - Display valid/invalid expressions

**AWS Deployment**:
- `scripts/aws/launch_medium_training.sh` - Launch medium training
- `scripts/aws/launch_large_training.sh` - Launch large training
- `scripts/aws/monitor_medium_training.sh` - Monitor progress

**Documentation**:
- `TRAIN_MEDIUM_AWS.md` - Quick guide for AWS training
- `WANDB_NAMING.md` - Wandb experiment naming standards
- `CREDENTIALS_SETUP.md` - API keys and SSH setup guide

### Credentials Location

**Complete setup guide**: See `CREDENTIALS_SETUP.md` for detailed configuration.

**API Tokens**: `C:\Users\madeinweb\.tokens.txt` (gitignored)
```
huggingface = hf_...
wandb = wandb_v1_...
```

**SSH Key (AWS)**: `C:\Users\madeinweb\chave-gpu.pem`
- Used for all AWS EC2 instance access
- Usage: `ssh -i ~/chave-gpu.pem ubuntu@<IP>`

Scripts automatically read tokens from `~/.tokens.txt` when available.

### Next Steps

1. **Wait for training completion** (~5 hours for both)
2. **Compare all three models** on Nguyen benchmarks
3. **Test larger models with RL**: REINFORCE/GRPO on Nguyen-5
4. **Expected result**: Medium/Large should generate complex expressions with proper nesting
5. **If successful**: Deploy to production, create HuggingFace model cards

---

## Model Scaling Study (Feb 2025)

### Overview

Comprehensive experiment training GPT-2 Base (124M), Medium (355M), and Large (774M) on 700K JSON dataset to investigate the impact of model size on symbolic regression capability.

**Status**: ⏳ In Progress

**Complete documentation**: See `EXPERIMENT_MODEL_SCALING.md` for full research report.

### Research Question

**Do larger models generate more complex, valid, and diverse mathematical expressions for symbolic regression?**

### Hypotheses

1. **H1 (Validity)**: Valid expression rate increases with model size (80% → 90%)
2. **H2 (Complexity)**: Expression depth increases (1.4 → 2.5), power operations increase (16% → 50%+)
3. **H3 (Performance)**: R² scores improve on complex benchmarks (Nguyen-5: -1.0 → >0.0)
4. **H4 (Diversity)**: Larger models generate more unique expressions
5. **H5 (RL Interaction)**: RL algorithms benefit more from larger models

### Models Trained

| Model | Parameters | Trainable (LoRA) | Instance | Batch Size | Training Time | Cost |
|-------|-----------|------------------|----------|-----------|---------------|------|
| Base | 124M | 294K | g5.xlarge | 8 | ~2-3h | ~$2-3 |
| Medium | 355M | 294K | g5.xlarge | 4 | ~3-4h | ~$3-4 |
| Large | 774M | 294K | g5.2xlarge | 2 | ~4-5h | ~$5-6 |

**All hyperparameters identical** except batch size (to isolate model size effect).

### Training Scripts

**Parallel training** (all 3 models simultaneously):
```bash
# Launch all models in parallel
bash launch_all_models.sh
```

**Individual training** (if needed):
```bash
# Base (124M)
bash scripts/aws/launch_base_training.sh --wandb-key KEY --hf-token TOKEN

# Medium (355M)
bash scripts/aws/launch_medium_training.sh --wandb-key KEY --hf-token TOKEN

# Large (774M)
bash scripts/aws/launch_large_training.sh --wandb-key KEY --hf-token TOKEN
```

**Critical fix applied**: Removed `cloud-init status --wait` deadlock from all launch scripts.

### Evaluation Pipeline

**Complete Nguyen suite evaluation** (144 experiments):
```bash
# Run full suite: 3 models × 12 benchmarks × 4 algorithms
bash scripts/run_nguyen_suite.sh

# Aggregate results and generate report
python scripts/aggregate_nguyen_results.py --input_dir nguyen_suite_results
```

**Algorithms evaluated**:
1. **Supervised**: Direct generation (no RL)
2. **REINFORCE**: Policy gradient with EMA baseline
3. **GRPO**: Group Relative Policy Optimization
4. **PPO**: Proximal Policy Optimization

### Metrics Tracked

**Quality Metrics**:
- Valid expression rate (%)
- Constraint adherence (uses only allowed vars/ops)
- Diversity rate (unique expressions)

**Complexity Metrics**:
- Power operations usage (x², x**n)
- Nested trigonometric functions (sin(cos(x)))
- Average expression depth
- Operator distribution

**Performance Metrics**:
- Best R² achieved per benchmark
- Mean R² (valid expressions only)
- Convergence rate during RL

### Key Results

*To be filled after evaluation completes*

| Metric | Base (124M) | Medium (355M) | Large (774M) |
|--------|-------------|---------------|--------------|
| Valid Rate (%) | TBD | TBD | TBD |
| Power Ops (%) | TBD | TBD | TBD |
| Avg Depth | TBD | TBD | TBD |
| Nested Trig (%) | TBD | TBD | TBD |
| Best R² (Nguyen-5) | TBD | TBD | TBD |

**Baseline** (from previous work):
- Base model: 39.4% valid on Nguyen-5, R²=-1.0
- Power operations: 15.9%
- Nested trig: 0%
- Average depth: 1.40

### Files Created

**Training scripts**:
- `scripts/aws/launch_base_training.sh` - Launch Base training
- `scripts/aws/launch_medium_training.sh` - Launch Medium training (fixed)
- `scripts/aws/launch_large_training.sh` - Launch Large training (fixed)
- `launch_all_models.sh` - Parallel launch all 3 models

**Evaluation scripts**:
- `scripts/run_nguyen_suite.sh` - Run complete Nguyen 1-12 suite
- `scripts/aggregate_nguyen_results.py` - Aggregate and visualize results

**Documentation**:
- `EXPERIMENT_MODEL_SCALING.md` - Full research report
- `TRAINING_LOG_MODEL_SCALING_2025.md` - Training execution log
- `model_cards/gpt2_base_700K_json_card.md` - Base model card
- `model_cards/gpt2_medium_700K_json_card.md` - Medium model card
- `model_cards/gpt2_large_700K_json_card.md` - Large model card

### Model Locations

**Local paths** (after training):
```
output/
├── gpt2_base_700K_json/       # Base (124M)
├── gpt2_medium_700K_json/     # Medium (355M)
└── gpt2_large_700K_json/      # Large (774M)
```

**HuggingFace** (after publication):
- Base: TBD (to be uploaded)
- Medium: TBD (to be uploaded)
- Large: TBD (to be uploaded)

### Reproduction

**Quick start**:
```bash
# 1. Train all models (requires AWS, ~10h total parallel)
bash launch_all_models.sh

# 2. Stop instances immediately after completion
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=seriguela-*-training" \
  "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].InstanceId" --output text)

# 3. Download models
scp -i ~/.ssh/KEY.pem -r ubuntu@BASE_IP:~/seriguela/output/gpt2_base_700K_json ./output/
scp -i ~/.ssh/KEY.pem -r ubuntu@MEDIUM_IP:~/seriguela/output/gpt2_medium_700K_json ./output/
scp -i ~/.ssh/KEY.pem -r ubuntu@LARGE_IP:~/seriguela/output/gpt2_large_700K_json ./output/

# 4. Evaluate (can run locally or on single AWS instance)
bash scripts/run_nguyen_suite.sh

# 5. Analyze results
python scripts/aggregate_nguyen_results.py --input_dir nguyen_suite_results
```

**Total cost**: ~$10-13 USD for training + optional $8-12 for full suite evaluation (if run on AWS)

### Expected Findings

**If hypotheses confirmed**:
- Larger models produce significantly more complex expressions
- Depth and power operation usage scale with model size
- R² scores improve on complex benchmarks
- Optimal model choice depends on task complexity and budget

**If hypotheses rejected**:
- LoRA may be the limiting factor (fixed rank=8 for all sizes)
- Dataset (700K) may not be large enough to show scaling benefits
- Alternative architectures may be needed

### Implications

**For model selection**:
- **Use Base (124M)** if: Fast inference, simple benchmarks (Nguyen 1-2), limited budget
- **Use Medium (355M)** if: Balanced performance/cost, moderate complexity
- **Use Large (774M)** if: Maximum quality needed, complex benchmarks (Nguyen 5+), budget available

**For future research**:
- Test LoRA rank scaling with model size (r=8/16/32)
- Train on larger datasets (1M, 5M expressions)
- Test other architectures (GPT-Neo, LLaMA)

### References

- **Training log**: `TRAINING_LOG_MODEL_SCALING_2025.md`
- **Research report**: `EXPERIMENT_MODEL_SCALING.md`
- **Model cards**: `model_cards/gpt2_*_700K_json_card.md`
- **Previous RL work**: See "Reinforcement Learning for Expression Generation" section above
