# PPO Symbolic Regression Experiment Plan

## Objective

Test whether PPO (Proximal Policy Optimization) fine-tuning can help a language model find mathematical expressions that fit given datasets.

## Background

### Current State
- **Base model**: GPT-2 + LoRA trained on JSON format data (exp_a_json)
- **Validation rate**: ~80% of generated expressions are syntactically valid
- **Problem**: Model generates valid expressions but doesn't know which one fits a specific dataset

### The Question
Can we use RL (specifically PPO) to fine-tune the model so it learns to generate expressions that maximize R² score on a given dataset?

## Methodology

### Simplifications for This Experiment

To isolate whether PPO works at all, we simplify the problem:

1. **No constants (C)**: Expressions won't contain learnable constants
   - Avoids the complexity of constant optimization during reward computation
   - Ground truth expressions: `x_1 + x_2`, `sin(x_1)`, etc.

2. **Simple operators**: `+`, `-`, `*`, `sin`, `cos`
   - No division (avoids numerical instability)
   - No exponentiation (simplifies search space)

3. **Small datasets**: 500 samples each, 1-2 variables

### Test Datasets

| Dataset | Ground Truth | Difficulty | Variables |
|---------|-------------|------------|-----------|
| add_x1_x2 | x_1 + x_2 | Easy | 2 |
| mul_x1_x2 | x_1 * x_2 | Easy | 2 |
| sub_x1_x2 | x_1 - x_2 | Easy | 2 |
| sin_x1 | sin(x_1) | Medium | 1 |
| cos_x1 | cos(x_1) | Medium | 1 |
| square_x1 | x_1 * x_1 | Medium | 1 |
| sin_x1_plus_x2 | sin(x_1) + x_2 | Hard | 2 |
| x1_mul_sin_x2 | x_1 * sin(x_2) | Hard | 2 |

### Experiment Design

#### Phase 1: Baseline Evaluation
- Generate 200 random expressions per dataset
- Compute R² for each valid expression
- Record best R², mean R², valid rate

#### Phase 2: PPO Training
- 10 epochs of PPO per dataset
- Batch size: 32
- Compare PPO best R² vs baseline best R²

#### Phase 3: Deep PPO (single dataset)
- 20 epochs on `mul_x1_x2` dataset
- Batch size: 64
- Track R² improvement over time

### Success Criteria

| Metric | Success Threshold |
|--------|-------------------|
| PPO finds exact expression | R² > 0.99 |
| PPO improves over baseline | PPO R² > Baseline R² |
| PPO is viable approach | Improvement on >50% of datasets |

## Implementation

### Files Created

```
scripts/
├── ppo_experiment.py           # Main PPO trainer (JSON format)
├── run_ppo_experiments.py      # Multi-dataset experiment runner
└── data/
    └── create_ppo_test_datasets.py  # Test dataset generator

userdata_ppo_experiment.sh      # AWS EC2 launch script
```

### Key Changes from Original trainer.py

1. **JSON format prompts** (matches exp_a_json training)
2. **Max retries** (avoids infinite loop when model generates invalid expressions)
3. **No constant optimization** (C=1 always, or no C in expression)
4. **Proper logging and checkpointing**

### Running the Experiment

```bash
# Local test (if GPU available)
python scripts/data/create_ppo_test_datasets.py
python scripts/run_ppo_experiments.py --model_path ./output/exp_a_json --epochs 5

# AWS (recommended)
# 1. Launch g5.xlarge with userdata_ppo_experiment.sh
# 2. Monitor: tail -f /home/ubuntu/ppo_experiment.log
# 3. Download results when complete
```

## Expected Outcomes

### Optimistic Scenario
- PPO finds exact expressions for easy datasets (R² > 0.99)
- PPO significantly improves R² for medium datasets
- Proves PPO is viable for symbolic regression

### Pessimistic Scenario
- PPO shows no improvement over random sampling
- Model can't learn to maximize R² reward
- Need to investigate: reward shaping, model capacity, exploration

### Diagnostic Scenario
- PPO improves valid rate but not R²
- PPO improves R² but can't find exact expression
- Suggests modifications to approach

## Next Steps (After Experiment)

Based on results:

1. **If PPO works**:
   - Add constant optimization
   - Try harder expressions
   - Scale up to real symbolic regression benchmarks

2. **If PPO partially works**:
   - Try different reward functions
   - Adjust PPO hyperparameters
   - Consider curriculum learning

3. **If PPO doesn't work**:
   - Analyze why model can't learn
   - Consider alternative approaches (beam search, MCTS)
   - May need different base model architecture

## Technical Notes

### Why JSON Format?
The exp_a_json model was trained on JSON format data and achieves 80% valid expressions. Using the same format for PPO ensures consistency.

### Why No Division?
Division can cause numerical instability (division by zero, very large values) which corrupts R² computation and gradient updates.

### Why Max Retries?
The original trainer had a `while reward < 0` loop that could hang forever if the model consistently generates invalid expressions. Max retries ensures progress.

---

*Experiment designed: 2026-02-02*
*Branch: experiment/ppo-symbolic-regression*
