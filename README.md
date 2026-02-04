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

## 📂 Project Structure

Organized by research phases for systematic experimentation:

```
seriguela/
├── 1_data/                 # FASE 1: Preparação de Dados
├── 2_training/            # FASE 2: Treinamento e Fine-tuning
├── 3_evaluation/          # FASE 3: Avaliação
├── 4_analysis/            # FASE 4: Análise e Visualização
├── models/                # Modelos treinados
├── results/               # Resultados de experimentos
├── docs/                  # Documentação completa
├── src/                   # Código fonte (package)
└── scripts/               # Scripts auxiliares
```

**Each directory contains a README.md with detailed documentation.**

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train model
cd 2_training/supervised
python train_with_json.py --model_size gpt2-medium

# 3. Evaluate
cd ../../3_evaluation/benchmarks
python run_all_nguyen_benchmarks.py --model_path ../../models/gpt2/medium_700k_json

# 4. Analyze
cd ../../4_analysis/visualization
python create_visualizations.py
```

---

## 📊 Complete Documentation

- **Scientific Report**: [`docs/reports/SCIENTIFIC_REPORT_MODEL_SCALING.md`](docs/reports/SCIENTIFIC_REPORT_MODEL_SCALING.md)
- **Developer Guide**: [`docs/guides/CLAUDE.md`](docs/guides/CLAUDE.md)
- **Model Cards**: [`docs/model_cards/`](docs/model_cards/)

---

## 🎓 Citation

```bibtex
@misc{seriguela2025,
  title={Scaling Laws for Symbolic Regression with LLMs},
  author={Augusto Cesar},
  year={2025},
  note={First 100% valid rate + R²=1.0 achieved}
}
```

---

**Status**: ✅ Production-ready | 📊 Publication-ready | **Last Updated**: 2026-02-04
