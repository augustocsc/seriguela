# Buffer Re-tokenization Fix — Walkthrough

## Problem

Elite buffer samples in `bon_ppo` and `bon_grpo` were created with `tokens=[]`, causing them to be **silently discarded** during the policy gradient update. The buffer was decorative — it never produced any training signal.

Additionally, in `bon_grpo`, buffer samples were appended at the **end** of the rollout list, isolating them from fresh generations during GRPO group ranking.

## Changes Made

### 1. [base_trainer.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/algorithms/base_trainer.py) — Core Fix

**New method: `retokenize_expression()`** — reconstructs the full text from a buffer expression, tokenizes it, and runs a forward pass to get real `log_probs`:

render_diffs(file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/algorithms/base_trainer.py)

### 2. [bon_grpo.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/algorithms/bon_grpo.py) — Shuffle Fix (applied earlier)

Added `random.shuffle(rollouts)` before GRPO group assignment.

## New Test Scripts

### Quick Smoke Test (~2-3 min on T4)

[test_retokenize_fix.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/test_retokenize_fix.py)

```bash
cd 2_training/reinforcement
python test_retokenize_fix.py
```

### Full Comparison Test (~30 min on T4)

[run_pre_phase_t6.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/run_pre_phase_t6.py)

```bash
cd 2_training/reinforcement
python run_pre_phase_t6.py
```

Results → `results/pre_phase__t6/` — compare directly with `results/pre_phase__t5/`
