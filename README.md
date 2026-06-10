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
├── 1_data/benchmarks/     # nguyen/, strogatz/, feynman metadata (CSVs grandes
│                          #   ficam fora do git — download on-demand do PMLB)
├── 2_training/
│   ├── supervised/        # SFT (concluído — modelos no HuggingFace)
│   └── reinforcement/     # CORE: run_experiment.py, algorithms/, rewards/
├── 3_evaluation/          # cli.py, metrics (symbolic_match), commands/
├── 4_analysis/            # statistical/, complexity/, visualization/
├── experiments/           # queue.yaml + daemon + smoke test + Fase 2 builder
├── runpod/                # orquestrador de compute (ver runpod/README.md)
├── results/               # dados válidos: pre_phase t5/t6, phase_1b, phase_1c
├── classes/               # Expression (parse/validate)
├── dissertation/          # LaTeX da dissertação
├── docs/                  # reports/ (THESIS_PLAN), guides/, proposals/, archive/
└── legacy/                # eras encerradas: colab/, aws/, phase_a/, rl_v1/
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

# 3. Validate the pipeline (CPU, <60s, mandatory before any push)
python experiments/test_smoke.py

# 4. Run an RL experiment (GPU — see runpod/README.md for the cloud workflow)
python 2_training/reinforcement/run_experiment.py \
  --algorithm bon_ppo --problem nguyen_5 --reward sr_ic --penalty gradient \
  --temperature cosine_annealing --batch_size 1024 --max_steps 200 \
  --seeds 42 123 456 789 1011 --no_wandb
```

---

## 📊 Complete Documentation

- **Developer Guide**: [CLAUDE.md](CLAUDE.md) - Commands, architecture, workflows
- **Thesis Plan**: [docs/reports/THESIS_PLAN.md](docs/reports/THESIS_PLAN.md) - Current status, hypotheses, experimental design
- **Repo Consolidation (2026-06-10)**: [docs/reports/REPO_CONSOLIDATION_2026-06-10.md](docs/reports/REPO_CONSOLIDATION_2026-06-10.md) - Deep analysis, cleanup decisions, next steps
- **RunPod Workflow**: [runpod/README.md](runpod/README.md) - Official compute path (pilot → Phase 2)
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

**Status**: Research in progress (Phase 1 done; Phase 2 RL next, on RunPod) | **Last Updated**: 2026-06-10
