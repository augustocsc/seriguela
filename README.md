# Seriguela - Symbolic Regression with Large Language Models

A Master's thesis research project exploring reinforcement learning optimization of GPT-2 models fine-tuned for symbolic regression. Compares PPO vs GRPO with elite-buffer augmentation across problem difficulty levels and evaluates against the SRBench benchmark.

## Models (HuggingFace)

Six LoRA-fine-tuned GPT-2 models (682K synthetic expressions, JSON-structured format):

| Model | Parameters | Valid Rate (Nguyen, zero-shot best-of-100) | Avg R² |
|-------|-----------|-------------------------------------------|---------|
| Base | 124M | 62.5% | 0.919 |
| Medium | 355M | 75.2% | 0.981 |
| Large | 774M | 89.0% | 0.985 |

> Note: R² values are from zero-shot best-of-100 sampling. RL fine-tuning experiments are ongoing (see `docs/reports/THESIS_PLAN.md`).

---

## 🤗 Models Available on HuggingFace

All 6 trained models are ready to use:

| Model | Notation | Repository |
|-------|----------|------------|
| Base | Infix | [augustocsc/gpt2_base_infix_682k](https://huggingface.co/augustocsc/gpt2_base_infix_682k) |
| Base | Prefix | [augustocsc/gpt2_base_prefix_682k](https://huggingface.co/augustocsc/gpt2_base_prefix_682k) |
| Medium | Infix | [augustocsc/gpt2_medium_infix_682k](https://huggingface.co/augustocsc/gpt2_medium_infix_682k) |
| Medium | Prefix | [augustocsc/gpt2_medium_prefix_682k](https://huggingface.co/augustocsc/gpt2_medium_prefix_682k) |
| **Large** 🏆 | Infix | [augustocsc/gpt2_large_infix_682k](https://huggingface.co/augustocsc/gpt2_large_infix_682k) |
| **Large** 🏆 | Prefix | [augustocsc/gpt2_large_prefix_682k](https://huggingface.co/augustocsc/gpt2_large_prefix_682k) |

---

## 📂 Project Structure

Organized by research phases for systematic experimentation:

```
seriguela/
├── 1_data/                 # FASE 1: Data Preparation
│   ├── benchmarks/         # Nguyen benchmarks
│   ├── processed/          # Processed datasets
│   └── README.md
├── 2_training/            # FASE 2: Training & Fine-tuning
│   ├── supervised/        # Supervised training
│   ├── reinforcement/     # RL algorithms (PPO, GRPO)
│   └── README.md
├── 3_evaluation/          # FASE 3: Evaluation
│   ├── benchmarks/        # Benchmark evaluation
│   ├── quality/           # Quality metrics
│   └── README.md
├── 4_analysis/            # FASE 4: Analysis & Visualization
│   ├── complexity/        # Complexity analysis
│   ├── statistical/       # Statistical tests
│   ├── visualization/     # Plots and charts
│   └── README.md
├── docs/                  # Complete documentation
│   ├── guides/            # Technical guides (CLAUDE.md, etc.)
│   ├── reports/           # Scientific reports
│   ├── model_cards/       # HuggingFace model cards
│   └── archive/           # Historical documentation
├── src/                   # Source code (package)
├── classes/               # Core classes
├── aws/                   # AWS configurations
└── configs/               # Training configurations
```

**Each phase directory contains a README.md with detailed documentation.**

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Load model from HuggingFace
python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> from peft import PeftModel
>>> tokenizer = AutoTokenizer.from_pretrained("augustocsc/gpt2_large_infix_682k")
>>> base_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
>>> model = PeftModel.from_pretrained(base_model, "augustocsc/gpt2_large_infix_682k")
>>> # Now generate expressions!

# 3. Evaluate on benchmarks
cd 3_evaluation/benchmarks
python run_all_nguyen_benchmarks.py --model_repo augustocsc/gpt2_large_infix_682k

# 4. Analyze results
cd ../../4_analysis/visualization
python create_visualizations.py
```

---

## 📊 Complete Documentation

- **Developer Guide**: [CLAUDE.md](CLAUDE.md) - Commands, architecture, workflows
- **Thesis Plan**: [docs/reports/THESIS_PLAN.md](docs/reports/THESIS_PLAN.md) - Current status, hypotheses, experimental design
- **Phase A Post-Mortem**: [docs/reports/phase_a_post_mortem.md](docs/reports/phase_a_post_mortem.md) - Analysis of Phase A limitations
- **Training Guide**: [docs/guides/TRAINING_CONFIG_REGISTRY.md](docs/guides/TRAINING_CONFIG_REGISTRY.md) - Exact configurations for reproducibility
- **Archive**: [docs/archive/](docs/archive/) - Historical reports from earlier phases

---

## 🎓 Citation

```bibtex
@misc{seriguela2026,
  title={Elite Buffer-Augmented RL for LLM-based Symbolic Regression: PPO vs GRPO},
  author={Augusto Cesar},
  year={2026},
  note={Master's thesis research — GPT-2 models with LoRA fine-tuning and RL optimization}
}
```

---

**Status**: Research in progress | **Last Updated**: 2026-05-18
