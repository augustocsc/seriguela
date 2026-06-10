# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Seriguela** is an **academic research project** (Master's thesis) training small
language models for symbolic regression via fine-tuning and reinforcement learning.
A GPT-2 Base (124M) fine-tuned with LoRA on 682K synthetic expressions is optimized
with RL to solve symbolic regression problems; four algorithmic variants
(Pure-PPO, Pure-GRPO, BoN-PPO, BoN-GRPO) plus a Best-of-N baseline are compared
under a **uniform reward** (`sr_ic`, C=1) on consolidated benchmarks
(Nguyen 1–12 + SRBench Feynman/Strogatz subset).

**Source of truth for the experimental design:** [docs/reports/THESIS_PLAN.md](docs/reports/THESIS_PLAN.md)
**Latest repo consolidation:** [docs/reports/REPO_CONSOLIDATION_2026-06-10.md](docs/reports/REPO_CONSOLIDATION_2026-06-10.md)

**Status (June 2026):**
- ✅ SFT complete: 6 models (Base/Medium/Large × Infix/Prefix) on HuggingFace
- ✅ Valid RL pre-phase data: 36 runs in `results/pre_phase__t5_and_t6_merged/` (reward confound documented)
- ✅ Phase 1b/1c complete: Best-of-N zero-shot, 122 problems × 10 seeds, Base + Large Infix (`results/phase_1b`, `results/phase_1c`)
- ❌ Phase A (9.7K W&B runs) statistically compromised — see [post-mortem](docs/reports/phase_a_post_mortem.md); never use for the defense
- ▶️ Next: RunPod timing pilot (Phase 1.5) → Phase 2 (100 RL seed-runs)

**Compute: RunPod** (community A40 48GB ≈ $0.40–0.47/h; fallbacks A6000/3090/4090).
Colab credits are exhausted and AWS was retired on cost — both live in `legacy/`.
See [runpod/README.md](runpod/README.md) for the full workflow.

---

## Models (HuggingFace)

| Model | Notation | Params | Repository |
|-------|----------|--------|------------|
| Base | Infix | 124M | [augustocsc/gpt2_base_infix_682k](https://huggingface.co/augustocsc/gpt2_base_infix_682k) |
| Base | Prefix | 124M | [augustocsc/gpt2_base_prefix_682k](https://huggingface.co/augustocsc/gpt2_base_prefix_682k) |
| Medium | Infix | 355M | [augustocsc/gpt2_medium_infix_682k](https://huggingface.co/augustocsc/gpt2_medium_infix_682k) |
| Medium | Prefix | 355M | [augustocsc/gpt2_medium_prefix_682k](https://huggingface.co/augustocsc/gpt2_medium_prefix_682k) |
| Large | Infix | 774M | [augustocsc/gpt2_large_infix_682k](https://huggingface.co/augustocsc/gpt2_large_infix_682k) |
| Large | Prefix | 774M | [augustocsc/gpt2_large_prefix_682k](https://huggingface.co/augustocsc/gpt2_large_prefix_682k) |

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_REPO = "augustocsc/gpt2_base_infix_682k"   # thesis default
BASE_MODEL = "gpt2"                               # gpt2 | gpt2-medium | gpt2-large

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE_MODEL), MODEL_REPO).eval()
```

**Prompt/data format (JSON, standard):**
```json
{"vars": ["x_1"], "ops": ["sin", "cos", "+", "*"], "cons": "C", "expr": "
```
The model completes the expression and closes with `"}`. The constant token `C`
is evaluated as **C=1** in all runs (uniform protocol; L-BFGS-B ablation is
champion-only — see THESIS_PLAN).

**Dataset:** `augustocsc/sintetico_natural_prefix_682k` (682,429 train / 75,826 val;
infix column `i_prompt_n`, prefix column `p_prompt_n_converted`).

---

## Project Structure

```
seriguela/
├── 1_data/                # benchmarks (nguyen/, strogatz/, feynman/*.meta.json)
│   └── benchmarks/        # bulky CSVs are NOT in git — PMLB download on demand
├── 2_training/
│   ├── supervised/        # SFT scripts (era concluída; modelos no HF)
│   └── reinforcement/     # CORE: run_experiment.py + algorithms/rewards/buffers
├── 3_evaluation/          # cli.py, core/metrics.py (symbolic_match), commands/
├── 4_analysis/            # statistical/, complexity/, visualization/
├── experiments/           # queue.yaml + queue_processor.py (daemon) + smoke test
│   ├── run_pilot_timing.py    # RunPod pilot (Fase 1.5)
│   └── build_phase2_queue.py  # gera as 20 entradas da Fase 2
├── runpod/                # launch_pilot.py (orquestrador) + README do fluxo
├── results/               # dados válidos: pre_phase__t5*, phase_1b, phase_1c, etapa0
├── classes/               # Expression (parse/validate infix+prefix)
├── configs/               # registro de configs SFT
├── dissertation/          # LaTeX da dissertação
├── docs/                  # reports/ (THESIS_PLAN!), guides/, proposals/, archive/
└── legacy/                # eras encerradas: colab/, aws/, phase_a/, pre_phase*, rl_v1/
```

---

## Core Commands

### RL experiment (single run)
```bash
python 2_training/reinforcement/run_experiment.py \
  --algorithm bon_ppo --model augustocsc/gpt2_base_infix_682k \
  --problem nguyen_5 --reward sr_ic --penalty gradient \
  --temperature cosine_annealing --prompt_type standard \
  --batch_size 1024 --max_steps 200 --seeds 42 123 456 789 1011 --no_wandb
```
Algorithms: `best_of_n | pure_ppo | pure_grpo | bon_ppo | bon_grpo`.
Problems: `nguyen_1..12`, `feynman_*`, `strogatz_*` (loaders download from PMLB).

### Queue daemon (any VM)
```bash
NO_GIT=1 python -c "from experiments.queue_processor import run_queue_loop; run_queue_loop(max_hours=24)"
```

### RunPod (official compute path)
```bash
python runpod/launch_pilot.py            # pod + bootstrap + smoke no pod + piloto
python runpod/launch_pilot.py --status
python runpod/launch_pilot.py --terminate   # SEMPRE ao final — pod não se auto-termina
```

### Phase 2 queue
```bash
python experiments/build_phase2_queue.py --max-steps 200 --write   # após o piloto
python experiments/test_smoke.py
```

### Evaluation / analysis
```bash
python 3_evaluation/cli.py --help
python 4_analysis/statistical/aggregate_nguyen_results.py
```

---

## Mandatory: Smoke Test Before Pushing

```bash
python experiments/test_smoke.py
```
Validates imports of `run_experiment.py`, `queue.yaml` schema, `--seeds` nargs
contract, and a CPU mini Best-of-N with a mocked model. <60s, no GPU, no
downloads. **Exit 0 = safe to push.** The RunPod bootstrap re-runs it on the pod
before burning GPU-hours. If you add an import to `run_experiment.py`, add the
matching import test here.

---

## Credentials

**API tokens:** `~/.tokens.txt` (gitignored, NEVER print or commit):
```
huggingface = hf_...
wandb = ...
runpod = ...
```
Scripts read this file automatically. Full guide: [docs/guides/CREDENTIALS_SETUP.md](docs/guides/CREDENTIALS_SETUP.md).

---

## Important Patterns

### Expression validation
```python
from classes.expression import Expression
expr = Expression.parse_infix("x + y")    # ou parse_prefix("+", "x", "y")
expr.validate()
# novos operadores: OPERATOR_ARITY / OPERATOR_FUNCS em classes/expression.py
```

### LoRA (todos os 6 modelos — detalhes em docs/guides/TRAINING_CONFIG_REGISTRY.md)
- Base: r=8, α=32, lr=5e-5 | Medium/Large: r=16, α=64, lr=3e-5/2e-5
- target_modules `c_attn`, dropout 0.05, 3 epochs

### W&B naming (quando usado)
`seriguela-{type}-{model}-{dataset}-{timestamp}` via `configs/wandb_config.py`.
Fases de produção rodam com `--no_wandb` (resultados em JSON no repo).

### Experiment hygiene (lições da Phase A — ver post-mortem)
1. **Sempre ≥5 seeds** por configuração (single-seed = anomalia, não resultado)
2. **Reward uniforme entre braços comparados** (confound t5/t6)
3. Baseline best_of_n precisa rodar de verdade (o wrapper RewardResult já foi corrigido)
4. Resultados em JSON versionado no repo; W&B é backup, não fonte primária

---

## Local Development Machine

- i5-12450HX, 16GB RAM, RTX 3050 6GB (ocupada pelo Windows), disco apertado
- **Local = edição de código + smoke test. Nunca treinar nem baixar pesos localmente.**
- Toda GPU real é RunPod (ou qualquer VM Linux com CUDA via queue daemon)

## Cloud GPU notes

- fp16 inference é o default do BoN (`BoNConfig(fp16=True)`): Base cabe com
  batch 128 em T4; em A40 48GB use batch 512–1024 e 2–3 runs concorrentes
  (o piloto mede o ganho real de concorrência)
- `HF_HOME` num diretório persistente do pod evita re-download de pesos

---

## Dependencies

Core: `transformers==4.51.3`, `torch==2.5.1+cu121`, `peft==0.15.1`,
`datasets==3.5.0`, `trl==0.16.1`, `sympy==1.13.1`, `wandb`, `pandas`,
`scikit-learn`, `matplotlib`/`seaborn`. Install: `pip install -r requirements.txt`
(+ torch cu121 via index URL — ver runpod/launch_pilot.py BOOTSTRAP).

## Quick Debugging

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from classes.expression import Expression; print(Expression.parse_infix('x + y').validate())"
python experiments/test_smoke.py
```

---

## Citation

```bibtex
@misc{seriguela2026,
  title={Elite Buffer-Augmented Reinforcement Learning for LLM-based Symbolic Regression},
  author={Augusto Cesar},
  year={2026},
  note={PPO vs GRPO with GPT-2 under a uniform reward protocol}
}
```

**Last Updated**: 2026-06-10
