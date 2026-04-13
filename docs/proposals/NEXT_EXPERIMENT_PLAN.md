# Next Experiment Plan

**Date**: 2026-02-22
**Goal**: Comprehensive RL ablation study with improved tracking

---

## Changes Since Last Experiment

### Bug Fixes
1. **2-variable bug fixed** - N9-N12 will now work correctly
2. **HuggingFace upload fixed** - Results go to `augustocsc/seriguela-results`

### New Implementations
| Feature | Description |
|---------|-------------|
| **Pure PPO** | PPO without elite buffer |
| **Pure GRPO** | GRPO without elite buffer |
| **Best-of-N Baseline** | Pure sampling, no RL training |
| **Oracle Prompts** | Hints about true operators |
| **Distractor Prompts** | Misleading operator hints |
| **Noise Robustness** | Gaussian noise injection |
| **OOD Evaluation** | Domain and structural OOD |

### Improved Tracking Metrics
| Metric | Purpose |
|--------|---------|
| `fresh_mean_r2` | Mean R² of NEW generations (not buffer) |
| `fresh_valid_rate` | Valid rate of NEW generations |
| `fresh_median_r2` | Median R² (more robust than mean) |
| `fresh_p75_r2` | 75th percentile R² |
| `fresh_p90_r2` | 90th percentile R² |
| `fresh_std_r2` | Standard deviation (variance) |
| `best_step` | When was best expression found |
| `unique_expressions` | Diversity per step |

**Key Question**: Do `fresh_*` metrics improve over time? If yes → RL is learning. If flat → just lucky sampling.

---

## Experiment Categories

### 1. ALGORITHM COMPARISON (Critical!)

**Question**: Is RL actually improving the model, or is sampling enough?

| Algorithm | Has Buffer | Has RL Training | Purpose |
|-----------|------------|-----------------|---------|
| `best_of_n` | No | No | Pure sampling baseline |
| `pure_ppo` | No | Yes | RL without buffer |
| `pure_grpo` | No | Yes | RL without buffer |
| `bon_ppo` | Yes | Yes | Hybrid (current best) |
| `bon_grpo` | Yes | Yes | Hybrid alternative |

**Expected Insight**:
- If `best_of_n` ≈ `bon_ppo` → RL not helping, SFT model already good
- If `pure_ppo` > `best_of_n` → RL genuinely improves policy
- If `bon_ppo` > `pure_ppo` → Buffer replay helps

### 2. REWARD FUNCTION ABLATION

| Reward | Description | Hypothesis |
|--------|-------------|------------|
| `r2_clipped` | Pure R² (clipped to [0,1]) | Simple but may prefer complex expressions |
| `length_penalized` | R² - α×length | Should prefer simpler expressions |
| `sr_ic` | SR-IC complexity penalty | Information-theoretic simplicity |

### 3. PENALTY STRATEGY ABLATION

| Penalty | Invalid Expression Reward | Hypothesis |
|---------|---------------------------|------------|
| `binary` | -1.0 for all invalid | Simple, harsh |
| `gradient` | Differentiated by error type | More informative signal |

### 4. TEMPERATURE SCHEDULE ABLATION

| Schedule | Behavior | Hypothesis |
|----------|----------|------------|
| `fixed_0.7` | Constant low temp | Exploits known good expressions |
| `fixed_0.9` | Constant high temp | More exploration |
| `linear_annealing` | 1.0 → 0.5 | Explore early, exploit late |
| `cosine_annealing` | Smooth annealing | Gradual transition |

### 5. PROMPT ROBUSTNESS

| Prompt Type | Operators Given | Purpose |
|-------------|-----------------|---------|
| `standard` | All operators | Normal condition |
| `oracle` | True operators + extras | Helpful hints |
| `distractor` | Wrong operators | Test robustness to bad hints |

**Expected Insight**: How much does prompt engineering matter?

### 6. NOISE ROBUSTNESS

| Noise Level | Description | Purpose |
|-------------|-------------|---------|
| 0% | Clean data | Baseline |
| 1% | Low noise | Minor perturbation |
| 5% | Medium noise | Realistic noise |
| 10% | High noise | Stress test |

**Expected Insight**: How robust is RL to noisy data?

### 7. MODEL SCALING (All 12 Nguyen Problems)

| Model | Parameters | Notation |
|-------|------------|----------|
| Base Infix | 124M | Infix |
| Base Prefix | 124M | Prefix |
| Medium Infix | 355M | Infix |
| Medium Prefix | 355M | Prefix |
| Large Infix | 774M | Infix |
| Large Prefix | 774M | Prefix |

**Problems**: N1-N12 (including 2-variable N9-N12, now fixed!)

---

## Full Experiment Matrix

### Phase 1: Algorithm Comparison (on Nguyen-5)
```
5 algorithms × 1 problem × 1 seed = 5 runs
```

### Phase 2: Ablations (on Nguyen-5, best algorithm)
```
Reward:      3 configs × 1 seed = 3 runs
Penalty:     2 configs × 1 seed = 2 runs
Temperature: 4 configs × 1 seed = 4 runs
Prompt:      3 configs × 1 seed = 3 runs
Noise:       4 configs × 1 seed = 4 runs
                         Total = 16 runs
```

### Phase 3: Scaling (all problems, best config)
```
6 models × 12 problems × 1 seed = 72 runs
```

### Total: ~93 runs

---

## AWS Instance Allocation

| Instance | Experiment | Est. Runs |
|----------|------------|-----------|
| 1 | algorithm_comparison | 5 |
| 2 | full_ablation_suite | 16 |
| 3 | scaling_base_infix | 12 |
| 4 | scaling_base_prefix | 12 |
| 5 | scaling_medium_infix | 12 |
| 6 | scaling_medium_prefix | 12 |
| 7 | scaling_large_infix | 12 |
| 8 | scaling_large_prefix | 12 |

---

## Key Questions to Answer

### Primary Questions
1. **Is RL improving the model?** → Compare `fresh_mean_r2` over steps
2. **Is the buffer helping?** → Compare `pure_ppo` vs `bon_ppo`
3. **Is RL even needed?** → Compare `best_of_n` vs RL methods

### Secondary Questions
4. **Best reward function?** → Compare final R² across rewards
5. **Best temperature?** → Compare convergence speed
6. **Infix vs Prefix?** → Compare across all models
7. **Model size effect?** → Base vs Medium vs Large

### Robustness Questions
8. **Prompt sensitivity?** → Oracle vs Standard vs Distractor
9. **Noise tolerance?** → Performance degradation with noise
10. **2-variable problems?** → N9-N12 performance (now testable!)

---

## Success Criteria

| Metric | Target |
|--------|--------|
| N1-N4, N7-N8 | R² > 0.999 |
| N5-N6 | R² > 0.995 (improved from 0.993) |
| N9-N12 | R² > 0.99 (new!) |
| `fresh_mean_r2` trend | Increasing over steps |
| Valid rate | > 80% by step 100 |

---

## Commands to Launch

```bash
# Phase 1: Algorithm comparison
python aws/launch_rl_experiment.py --experiment algorithm_comparison

# Phase 2: Full ablation
python aws/launch_rl_experiment.py --experiment full_ablation_suite

# Phase 3: Scaling (6 instances)
python aws/launch_rl_experiment.py --experiment scaling_base_infix
python aws/launch_rl_experiment.py --experiment scaling_base_prefix
python aws/launch_rl_experiment.py --experiment scaling_medium_infix
python aws/launch_rl_experiment.py --experiment scaling_medium_prefix
python aws/launch_rl_experiment.py --experiment scaling_large_infix
python aws/launch_rl_experiment.py --experiment scaling_large_prefix
```

---

## Analysis Plan

After experiments complete:

1. **Download W&B data** for all runs
2. **Plot `fresh_mean_r2` over steps** for each algorithm → Answer "Is RL learning?"
3. **Compare final R²** across algorithms → Answer "Which is best?"
4. **Plot valid_rate over steps** → Answer "Is model generating more valid expressions?"
5. **Analyze N9-N12** → New 2-variable results
6. **Statistical comparison** if clear winner emerges

---

*Plan created: 2026-02-22*
