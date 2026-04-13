# Seriguela - Symbolic Regression with Large Language Models

A research project exploring the application of GPT-2 and other LLMs to symbolic regression through fine-tuning and reinforcement learning.

## 🏆 Key Results (Model Scaling Study - Feb 2025)

| Model | Parameters | Valid Rate (Quality) | Valid Rate (Nguyen) | Avg R² | Max R² |
|-------|-----------|---------------------|---------------------|---------|--------|
| Base | 124M | 99.4% | 62.5% | 0.9190 | 0.9994 |
| Medium | 355M | 99.2% | 75.2% | 0.9812 | 0.9999 |
| **Large** | **774M** | **100%** 🏆 | **89.0%** 🏆 | **0.9852** 🏆 | **1.0000** 🏆⭐ |

**Breakthrough**: First model to achieve **100% valid rate** and **R²=1.0 perfect fit** on Nguyen-8

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
- **Scientific Report**: [docs/reports/SCIENTIFIC_REPORT_MODEL_SCALING.md](docs/reports/SCIENTIFIC_REPORT_MODEL_SCALING.md) - Complete academic analysis
- **Training Guide**: [docs/guides/TRAINING_CONFIG_REGISTRY.md](docs/guides/TRAINING_CONFIG_REGISTRY.md) - Exact configurations for reproducibility
- **Model Cards**: [docs/model_cards/](docs/model_cards/) - Detailed model documentation

---

## 🎓 Citation

```bibtex
@misc{seriguela2025,
  title={Scaling Laws for Symbolic Regression with LLMs},
  author={Augusto Cesar},
  year={2025},
  note={First 100% valid rate + R²=1.0 achieved with GPT-2 Large}
}
```

---

**Status**: ✅ Production-ready | 📊 Publication-ready | **Last Updated**: 2026-02-20
