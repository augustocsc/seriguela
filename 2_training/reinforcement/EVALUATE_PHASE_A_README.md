# Evaluate Phase A Results

Analyze Phase A experimental results to understand what we learned and plan Phase B.

## What This Script Does

The evaluation script:

1. **Downloads Results** from HuggingFace or AWS instances
2. **Analyzes Hyperparameters** - Which combinations work best?
3. **Finds Best Configs** - Top-K performing configurations
4. **Identifies Gaps** - Which combinations are under-tested?
5. **Recommends Phase B** - Which configs to test on all models

## Quick Start

```bash
cd 2_training/reinforcement

# Download from HuggingFace (recommended)
python evaluate_phase_a_results.py

# Or use cached results
python evaluate_phase_a_results.py --use-cache phase_a_results_cache.json
```

## Usage Options

### Download from HuggingFace
```bash
python evaluate_phase_a_results.py --save-cache my_results.json
```

### Download from AWS Instances
```bash
# Requires instances to be running!
python evaluate_phase_a_results.py --download-from-instances
```

### Use Cached Results
```bash
python evaluate_phase_a_results.py --use-cache phase_a_results_cache.json --top-k 20
```

### Test with Limited Data
```bash
python evaluate_phase_a_results.py --limit 500
```

## Output

The script generates:

### 1. Hyperparameter Impact Analysis
Shows mean/median R² for each hyperparameter value:

```
Algorithm:
  Name                 Mean R²    Std        Median R²  Count
  ----------------------------------------------------------------
  bon_ppo             0.9234     0.1234     0.9456     1234
  pure_grpo           0.9123     0.1456     0.9234     1234
  ...

Reward:
  length_penalized    0.9345     0.1123     0.9456     2345
  r2_clipped          0.9234     0.1234     0.9345     2345
  sr_ic               0.9123     0.1345     0.9234     2345
```

### 2. Top-K Best Configurations
Lists the best performing configs with all hyperparameters:

```
1. Test R² = 0.9876 | Gen Gap = 0.0123
   Model: gpt2_base_infix_682k
   Problem: nguyen_5
   Algorithm: bon_ppo
   Reward: length_penalized
   Penalty: gradient
   Temperature: fixed_0.7
   Prompt: oracle
   Expression: sin(x_1) + C*cos(x_1)
```

### 3. Coverage Gaps
Identifies which hyperparameter combinations are missing or under-tested:

```
Combination coverage (algo × reward × penalty):
  best_of_n_length_penalized_binary        1234
  bon_ppo_r2_clipped_gradient                987
  pure_ppo_sr_ic_gradient                     30  ← Under-tested!

Missing combinations: 2
  - pure_grpo_sr_ic_binary
  - best_of_n_sr_ic_binary
```

### 4. Phase B Recommendations
Suggests top-5 configs to test on all models (Base/Medium/Large) and all 12 Nguyen benchmarks:

```
PHASE B RECOMMENDATIONS (Top-5 configs)

Based on Phase A results, test these configurations on:
  - All 6 models (Base/Medium/Large × Infix/Prefix)
  - All 12 Nguyen benchmarks
  - Multiple seeds for statistical significance

Top-5 configurations to test:
  1. algorithm: bon_ppo
     reward: length_penalized
     penalty: gradient
     temperature: fixed_0.7
     prompt: oracle
     ...
```

### 5. JSON Output Files

- **`phase_a_results_cache.json`** - All downloaded results
- **`phase_b_recommended_configs.json`** - Top configs for Phase B

Example `phase_b_recommended_configs.json`:
```json
[
  {
    "algorithm": "bon_ppo",
    "reward_type": "length_penalized",
    "penalty_type": "gradient",
    "temperature_schedule": "fixed_0.7",
    "prompt_type": "oracle",
    "noise_level": 0.0,
    "phase_a_test_r2": 0.9876
  },
  ...
]
```

## Analysis Questions Answered

### 1. Which algorithms work best?
Compare mean R² across algorithms to see if BoN-PPO/GRPO beat pure RL.

### 2. Which reward functions are most effective?
See if `length_penalized` encourages simpler expressions vs `r2_clipped`.

### 3. Does penalty type matter?
Compare `binary` vs `gradient` penalty strategies.

### 4. What's the best temperature schedule?
Fixed vs annealing strategies - which helps exploration?

### 5. How important are prompts?
Does `oracle` (with ground truth ops) beat `distractor` or `standard`?

### 6. Which problems are hardest?
Compare mean R² across Nguyen 1, 5, and 9.

### 7. Infix vs Prefix notation?
Does notation affect performance?

### 8. What should we test in Phase B?
Top-5 configs to scale up to Medium/Large models and full benchmarks.

## Requirements

```bash
pip install numpy huggingface-hub
```

## Notes

- Downloads ~3,300 result files from HuggingFace (~10-15 minutes)
- Results cache speeds up repeated analysis
- Can also download directly from AWS instances (but they must be running)
- Generates recommendations for Phase B automatically

## Example Workflow

```bash
# 1. Download and analyze
python evaluate_phase_a_results.py --save-cache results.json

# 2. Review top configs and gaps

# 3. Use recommendations for Phase B
cat phase_b_recommended_configs.json

# 4. Launch Phase B with top configs
# (Use recommended configs in Phase B experiment launcher)
```

## What's Next (Phase B)

After analyzing Phase A:
1. Take top-5 configs from recommendations
2. Test on all 6 models (Base/Medium/Large × Infix/Prefix)
3. Test on all 12 Nguyen benchmarks (not just 1, 5, 9)
4. Run with multiple seeds for statistical significance
5. Compare model scaling effects

Total Phase B: ~360 experiments (6 models × 12 problems × 5 configs)
