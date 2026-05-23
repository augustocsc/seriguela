# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Seriguela** is an **academic research project** focused on training language models for symbolic regression through fine-tuning and reinforcement learning. The project trains GPT-2 models to generate valid mathematical expressions using LoRA (parameter-efficient fine-tuning) and explores RL algorithms (PPO, GRPO) for optimizing expression quality.

**Research Context**: Graduate-level research exploring the application of large language models to symbolic regression problems, with focus on:
- Parameter-efficient fine-tuning techniques (LoRA)
- Reinforcement learning for expression optimization
- Model scaling effects on compositional complexity
- Benchmark evaluation (Nguyen benchmarks 1-12)

**Status (May 2026)**: 🔬 **RL Experiments in Progress**
- 6 models trained (SFT complete): Base/Medium/Large × Infix/Prefix, all on HuggingFace
- JSON structured format is the standard approach
- Phase A (~9.7K W&B runs) statistically compromised — see `docs/reports/phase_a_post_mortem.md`
- Valid RL data: 36 JSONs in `results/pre_phase__t5_and_t6_merged/` (test5/test6, March 2026)
- Current work: Fase 1 pilot (18 experiments) running on Colab via queue daemon
- Thesis plan and experimental design: `docs/reports/THESIS_PLAN.md`

**⏰ DEADLINE**: Less than 2 days of Colab compute remain (as of 2026-05-23). Every hour counts.
All changes must be pre-validated with `python experiments/test_smoke.py` before pushing.

---

## Models (HuggingFace)

All 6 models are available on HuggingFace:

| Model | Notation | Params | Repository |
|-------|----------|--------|------------|
| Base | Infix | 124M | [augustocsc/gpt2_base_infix_682k](https://huggingface.co/augustocsc/gpt2_base_infix_682k) |
| Base | Prefix | 124M | [augustocsc/gpt2_base_prefix_682k](https://huggingface.co/augustocsc/gpt2_base_prefix_682k) |
| Medium | Infix | 355M | [augustocsc/gpt2_medium_infix_682k](https://huggingface.co/augustocsc/gpt2_medium_infix_682k) |
| Medium | Prefix | 355M | [augustocsc/gpt2_medium_prefix_682k](https://huggingface.co/augustocsc/gpt2_medium_prefix_682k) |
| Large | Infix | 774M | [augustocsc/gpt2_large_infix_682k](https://huggingface.co/augustocsc/gpt2_large_infix_682k) |
| Large | Prefix | 774M | [augustocsc/gpt2_large_prefix_682k](https://huggingface.co/augustocsc/gpt2_large_prefix_682k) |

### How to Load Models

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Choose model (example: Large Infix)
MODEL_REPO = "augustocsc/gpt2_large_infix_682k"
BASE_MODEL = "gpt2-large"  # gpt2, gpt2-medium, or gpt2-large

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, MODEL_REPO)
model.eval()

# For GPU
model = model.to("cuda")
```

### Generate Expressions

**Infix notation** (e.g., `sin(x_1) + C*cos(x_1)`):
```python
prompt = '{"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "'

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Example: {"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "sin(x_1) + C*cos(x_1)"}
```

**Prefix notation** (e.g., `+ sin x_1 * C cos x_1`):
```python
# Same code, but output will be in prefix notation:
# {"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "+ sin x_1 * C cos x_1"}
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Load model and generate (see "How to Load Models" above)
# Or run evaluation on benchmarks:
cd 3_evaluation/benchmarks
python run_all_nguyen_benchmarks.py --model_repo augustocsc/gpt2_large_infix_682k

# 3. Analyze results
cd ../../4_analysis/visualization
python create_visualizations.py
```

---

## Project Structure

The codebase is organized by research phases for systematic experimentation:

```
seriguela/
├── 1_data/                 # FASE 1: Data Preparation
│   ├── benchmarks/         # Nguyen benchmarks
│   ├── processed/          # Processed datasets (682K unified dataset)
│   ├── raw/               # Raw datasets
│   └── README.md          # Data phase documentation
├── 2_training/            # FASE 2: Training & Fine-tuning
│   ├── supervised/        # Supervised training scripts
│   ├── reinforcement/     # RL algorithms (PPO, GRPO enhanced)
│   ├── configs/           # Training configurations
│   └── README.md          # Training phase documentation
├── 3_evaluation/          # FASE 3: Evaluation
│   ├── benchmarks/        # Nguyen benchmark evaluation
│   ├── quality/           # Quality metrics
│   ├── comparison/        # Model comparison
│   └── README.md          # Evaluation phase documentation
├── 4_analysis/            # FASE 4: Analysis & Visualization
│   ├── complexity/        # Expression complexity analysis
│   ├── statistical/       # Statistical tests
│   ├── visualization/     # Plots and charts
│   └── README.md          # Analysis phase documentation
├── docs/                  # Complete documentation
│   ├── guides/            # Technical guides (CREDENTIALS_SETUP, WANDB_NAMING, etc.)
│   ├── reports/           # Scientific reports (SCIENTIFIC_REPORT, EXPERIMENT_*, etc.)
│   ├── model_cards/       # HuggingFace model cards
│   └── archive/           # Historical documentation
├── aws/                   # AWS configurations and keys
├── configs/               # Training configurations
├── src/                   # Source code (package)
├── classes/               # Core classes (Expression parsing, etc.)
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── README.md              # Main README
└── CLAUDE.md             # This file
```

**Each phase directory (1-4) contains a README.md with detailed documentation.**

---

## Credentials & Setup

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 (required for GPU)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Login to experiment tracking (optional)
wandb login
```

### Credentials Location

**API Tokens**: `C:\Users\madeinweb\.tokens.txt` (gitignored)
```
huggingface = hf_...
wandb = wandb_v1_...
```

**SSH Key (AWS)**: `C:\Users\madeinweb\chave-gpu.pem`
- Used for all AWS EC2 instance access
- Usage: `ssh -i ~/chave-gpu.pem ubuntu@<IP>`

Scripts automatically read tokens from `~/.tokens.txt` when available.

**Complete setup guide**: See [docs/guides/CREDENTIALS_SETUP.md](docs/guides/CREDENTIALS_SETUP.md)

---

## Core Commands

### Data

Located in: `1_data/`

**Dataset**: `augustocsc/sintetico_natural_prefix_682k` (already on HuggingFace)
- **Train**: 682,429 examples (90%)
- **Validation**: 75,826 examples (10%)
- **Infix column**: `i_prompt_n`
- **Prefix column**: `p_prompt_n_converted`

Both columns represent the **same expressions** in different notations.

See [docs/guides/TRAINING_CONFIG_REGISTRY.md](docs/guides/TRAINING_CONFIG_REGISTRY.md) for complete dataset details.

### Training

Located in: `2_training/`

**Supervised training** (if you want to retrain):
```bash
cd 2_training/supervised
python train_with_json.py \
  --model_size gpt2-medium \
  --notation infix \
  --output_dir ../../models/gpt2/medium_infix
```

**Reinforcement learning** (PPO/GRPO):
```bash
cd 2_training/reinforcement
python ppo_enhanced.py \
  --model_repo augustocsc/gpt2_base_infix_682k \
  --benchmark nguyen_5

python grpo_enhanced.py \
  --model_repo augustocsc/gpt2_base_infix_682k \
  --benchmark nguyen_5
```

See `2_training/README.md` for detailed training instructions.

### Evaluation

Located in: `3_evaluation/`

**Quality metrics**:
```bash
cd 3_evaluation/quality
python evaluate_quality.py \
  --model_repo augustocsc/gpt2_large_infix_682k
```

**Nguyen benchmarks**:
```bash
cd 3_evaluation/benchmarks
python run_all_nguyen_benchmarks.py \
  --model_repo augustocsc/gpt2_large_infix_682k
```

**Model comparison**:
```bash
cd 3_evaluation/comparison
python compare_models.py \
  --model1 augustocsc/gpt2_base_infix_682k \
  --model2 augustocsc/gpt2_medium_infix_682k
```

See `3_evaluation/README.md` for detailed evaluation instructions.

### Analysis

Located in: `4_analysis/`

**Complexity analysis**:
```bash
cd 4_analysis/complexity
python analyze_complexity.py \
  --model_repo augustocsc/gpt2_large_infix_682k
```

**Statistical analysis**:
```bash
cd 4_analysis/statistical
python statistical_tests.py
```

**Visualizations**:
```bash
cd 4_analysis/visualization
python create_visualizations.py
```

See `4_analysis/README.md` for detailed analysis instructions.

---

## AWS Infrastructure

**⚠️ CONVENÇÃO DE NOMES**: Todas as instâncias AWS criadas por este projeto usam o prefixo **"augusto-"** para evitar conflitos.

### Instance Management

```bash
# ✅ RECOMENDADO: Listar apenas suas instâncias (prefixo "augusto-")
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=augusto-*" \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,Tags[?Key=='Name'].Value|[0]]" \
  --output table

# ✅ RECOMENDADO: Parar TODAS as suas instâncias
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=augusto-*" \
  "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].InstanceId" --output text)
```

### Training on AWS

**Launch training** (example with Medium model):
```bash
# Create instance and start training
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g5.xlarge \
  --key-name chave-gpu-nova \
  --security-group-ids sg-0deaa73e23482e3f6 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=augusto-seriguela-medium-training}]' \
  --user-data file://training_script.sh
```

**Monitor training**:
```bash
# SSH to instance
ssh -i ~/.ssh/chave-gpu-nova.pem ubuntu@<PUBLIC_IP>

# Check training progress
tail -f ~/seriguela/training_*.log
```

**Download trained models**:
```bash
# After training completes
scp -i ~/.ssh/chave-gpu-nova.pem -r ubuntu@<IP>:~/seriguela/models/ ./
```

### Costs and Cleanup

**ALWAYS stop instances after use** to avoid unnecessary costs:

```bash
# Check running instances
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=augusto-*" \
  "Name=instance-state-name,Values=running"

# Stop all your instances
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=augusto-*" \
  "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].InstanceId" --output text)
```

**Typical costs** (us-east-1):
- **g5.xlarge** (24GB VRAM, A10G GPU): ~$1.01/hora
- **g5.2xlarge** (48GB VRAM, A10G GPU): ~$1.21/hora

### Security Group

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

### ⚠️ SEGURANÇA

**NUNCA use comandos AWS genéricos sem filtro de nome!**

```bash
# ❌ PERIGOSO - Pode listar/parar instâncias de outros
aws ec2 describe-instances
aws ec2 stop-instances --instance-ids i-xxx

# ✅ SEGURO - Usa filtro "augusto-"
aws ec2 describe-instances --filters "Name=tag:Name,Values=augusto-*"
```

---

## Important Patterns

### Data Format (JSON - Standard)

**Training data format**:
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "sin(x_1 + C*x_2)"}
```

**Inference prompt**:
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "
```

The model completes the expression and closes with `"}`.

**Why JSON format**:
1. **Clear boundaries**: JSON has explicit `{` start and `}` end markers
2. **Structured containment**: Expression is within `"expr": "..."` field
3. **Better training signal**: Explicit structure helps model learning
4. **Less repetition**: Structure reduces repetitive generation patterns

### LoRA Configuration

All models use LoRA for parameter-efficient fine-tuning:

**Base models** (124M):
- r=8, alpha=32, lr=5e-5, batch=8
- 294K trainable parameters

**Medium/Large models** (355M/774M):
- r=16, alpha=64, lr=3e-5/2e-5, batch=4/2
- Larger rank prevents mode collapse

**Common settings** (all models):
- target_modules: `c_attn` (attention layers only)
- dropout: 0.05
- epochs: 3
- gradient_accumulation_steps: 4

See [docs/guides/TRAINING_CONFIG_REGISTRY.md](docs/guides/TRAINING_CONFIG_REGISTRY.md) for complete configuration details.

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

### Wandb Naming Standards

**Standard format**: `seriguela-{type}-{model}-{dataset}-{timestamp}`

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

See [docs/guides/WANDB_NAMING.md](docs/guides/WANDB_NAMING.md) for complete naming conventions.

---

## Documentation

**For Developers**:
- **This file (CLAUDE.md)**: Commands, architecture, workflows
- **Phase READMEs**: Detailed documentation for each phase
  - [1_data/README.md](1_data/README.md) - Data preparation
  - [2_training/README.md](2_training/README.md) - Training workflows
  - [3_evaluation/README.md](3_evaluation/README.md) - Evaluation pipelines
  - [4_analysis/README.md](4_analysis/README.md) - Analysis methods

**For Academic/Research**:
- [docs/reports/SCIENTIFIC_REPORT_MODEL_SCALING.md](docs/reports/SCIENTIFIC_REPORT_MODEL_SCALING.md) - Complete academic report with statistical analysis
- [docs/model_cards/](docs/model_cards/) - HuggingFace-ready model documentation
- [docs/reports/EXPERIMENT_*.md](docs/reports/) - Detailed experiment documentation

**For Configuration**:
- [docs/guides/TRAINING_CONFIG_REGISTRY.md](docs/guides/TRAINING_CONFIG_REGISTRY.md) - Exact configurations for all 6 models (critical for reproducibility)
- [docs/guides/WANDB_NAMING.md](docs/guides/WANDB_NAMING.md) - Experiment naming conventions
- [docs/guides/CREDENTIALS_SETUP.md](docs/guides/CREDENTIALS_SETUP.md) - API tokens and SSH keys

**Historical**:
- [docs/archive/](docs/archive/) - Old documentation, status files, and experiment logs

---

## Dependencies

### Core ML Stack
- `transformers==4.51.3` - Model loading, tokenizer, Trainer
- `torch==2.5.1+cu121` - Deep learning with CUDA 12.1
- `peft==0.15.1` - LoRA parameter-efficient fine-tuning
- `datasets==3.5.0` - HuggingFace dataset loading
- `accelerate==1.6.0` - Multi-GPU training

### Experiment Tracking
- `wandb>=0.24.1` - Experiment tracking
- `tensorboard==2.16.2` - Alternative visualization
- `trl==0.16.1` - Advanced training techniques (PPO, GRPO)

### Validation & Analysis
- `sympy==1.13.1` - Symbolic math for expression validation
- `pandas==2.2.1` - Data manipulation
- `scikit-learn==1.6.1` - Metrics and evaluation
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Visualization

---

## Local Development Machine

**Hardware** (user's laptop — for local testing only, NOT for training):
- CPU: Intel i5-12450HX (8 cores / 12 threads, 2.4 GHz)
- RAM: 16 GB
- GPU: NVIDIA RTX 3050 6GB Laptop (Compute 8.6) — almost fully used by Windows
- Disk: ~7 GB free (tight — don't download model weights locally)

**Consequence**: All model training and experiment runs happen on **Google Colab** (T4 16GB).
Local machine is only for: code editing, import validation, and running `test_smoke.py`.

---

## Mandatory: Smoke Test Before Pushing

**RULE**: Before pushing any change touching Colab-facing scripts, run:

```bash
python experiments/test_smoke.py
```

This test:
- Validates all imports in `run_experiment.py` (catches path bugs like the sys.path fix)
- Validates `queue.yaml` schema and `--seeds` command building
- Runs a full Best-of-N mini-experiment (2 samples) with a tiny mocked model on CPU
- Checks output JSON structure
- Runs in <60s on any machine, no GPU or HF download required

**Exit 0 = safe to push. Exit 1 = fix before pushing.**

If you add a new module to `run_experiment.py`'s imports, add a corresponding import test to `experiments/test_smoke.py`.

---

## Colab GPU Optimization

**Target GPU**: NVIDIA T4 (free tier) — 16GB VRAM, fp16 native

**Key decisions** (do not change without re-running smoke test):
- **fp16 inference** (`BoNConfig(fp16=True)` — default): halves VRAM, ~2x throughput
  - Base (124M) fp16 ≈ 250MB weights → `batch_size=128` fits comfortably
  - Medium (355M) fp16 ≈ 700MB weights → `batch_size=128` fits comfortably
  - Large (774M) fp16 ≈ 1.5GB weights → `batch_size=64` conservative margin
- **Model cache on Drive** (`HF_HOME=/content/drive/MyDrive/seriguela_models`): avoid
  re-downloading 774MB per session
- **No W&B for Phase 1** (`no_wandb: true`): removes ~5s overhead per run

**Phase 1 time budget** (18 experiments on T4 with fp16):
- 6× Base: ~8 min each = ~48 min
- 6× Medium: ~15 min each = ~90 min
- 6× Large: ~20 min each = ~120 min
- Total Phase 1: **~4.5h** (was ~8h with fp32 + batch_size=64)

**Phase 2 time budget** (after model selection — placeholder):
- 5 algorithms × 4 problems × 5 seeds = 100 runs
- Estimated 30-60 min per run depending on model size and MAX_STEPS

---

## Best Practices

1. **Run smoke test first**: `python experiments/test_smoke.py` before every push touching Colab scripts
2. **Use HuggingFace models**: Don't retrain from scratch - use our 6 pre-trained models
3. **Choose appropriate model size**: Base (124M), Medium (355M), or Large (774M) depending on your needs
4. **Track experiments**: Enable W&B logging with standardized naming
5. **Use config files**: Store hyperparameters in `configs/` for reproducibility
6. **Stop AWS instances**: Always stop instances when not in use to avoid charges
7. **Version control**: Commit config files but never commit model weights or keys
8. **Check GPU**: Verify GPU availability with `nvidia-smi` and `torch.cuda.is_available()`

---

## Quick Debugging

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Test model loading
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('augustocsc/gpt2_large_infix_682k'); print('OK')"

# Test expression parsing
python -c "from classes.expression import Expression; expr = Expression.parse_infix('x + y'); print(expr.validate())"

# Check dataset
python -c "from datasets import load_dataset; ds = load_dataset('augustocsc/sintetico_natural_prefix_682k'); print(len(ds['train']))"
```

---

## Citation

```bibtex
@misc{seriguela2025,
  title={Scaling Laws for Symbolic Regression with LLMs},
  author={Augusto Cesar},
  year={2025},
  note={Parameter-efficient fine-tuning of GPT-2 models for symbolic regression}
}
```

---

**Last Updated**: 2026-02-20
