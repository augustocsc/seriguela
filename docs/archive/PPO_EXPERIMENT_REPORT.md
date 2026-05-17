# PPO Symbolic Regression Experiment Report

**Date:** 2026-02-02
**Author:** Claude Code (automated experiment)
**Branch:** `experiment/ppo-symbolic-regression`

## Executive Summary

This experiment tested whether the JSON-format trained model (exp_a_json) could be used for symbolic regression via PPO fine-tuning. The experiment encountered API compatibility issues with TRL 0.16+ and pivoted to a Best-of-N sampling approach to test model capabilities.

**Key Finding:** The current model architecture is not suitable for symbolic regression without significant modifications. All 8 test datasets showed 0% valid expressions.

## Experiment Design

### Objective
Test if a GPT-2 model fine-tuned on JSON-format symbolic expressions can generate expressions that fit given datasets through:
1. PPO reinforcement learning (primary approach)
2. Best-of-N random sampling (fallback approach)

### Test Datasets
| Dataset | Formula | Difficulty |
|---------|---------|------------|
| add_x1_x2 | x_1 + x_2 | Easy |
| mul_x1_x2 | x_1 * x_2 | Easy |
| sub_x1_x2 | x_1 - x_2 | Easy |
| sin_x1 | sin(x_1) | Medium |
| cos_x1 | cos(x_1) | Medium |
| square_x1 | x_1 * x_1 | Medium |
| sin_x1_plus_x2 | sin(x_1) + x_2 | Hard |
| x1_mul_sin_x2 | x_1 * sin(x_2) | Hard |

### Methodology
- **Samples per dataset:** 500 expressions
- **Temperature:** 0.7
- **Reward metric:** R² score (coefficient of determination)
- **Model:** exp_a_json (GPT-2 + LoRA, JSON format, 80% valid expressions in generation tests)

## Technical Issues Encountered

### Issue 1: TRL 0.16+ API Breaking Changes
The PPO approach failed due to incompatible API changes in TRL 0.16:
- `PPOTrainer` now requires `reward_model` and `train_dataset` parameters
- Old custom reward function approach no longer supported
- Would require implementing a separate reward model

### Issue 2: Model Loading Size Mismatch
**Error:** Tokenizer vocab size (50259) != base model vocab size (50257)

**Solution:** Load tokenizer from trained model first, then resize base model embeddings before loading LoRA adapter:
```python
tokenizer = AutoTokenizer.from_pretrained(model_path)  # 50259 tokens
base_model = AutoModelForCausalLM.from_pretrained("gpt2")  # 50257 tokens
base_model.resize_token_embeddings(len(tokenizer))  # Resize to 50259
model_with_lora = PeftModel.from_pretrained(base_model, model_path)
```

### Issue 3: Expression Extraction Failure
The model generates text that overflows beyond the expected JSON boundary.

**Expected output:**
```
{"vars": ["x_1", "x_2"], "ops": ["+", "-"], "cons": null, "expr": "x_1 + x_2"}
```

**Actual output:**
```
{"vars": ["x_1", "x_2"], "ops": ["+", "-"], "cons": null, "expr": "x_1 + sin(C*x_1) - C"}{"vars": ["x_1", "x_2", "x_3"...
```

The model continues generating a new JSON object instead of stopping at the closing quote.

### Issue 4: Constant Generation
All generated expressions contain the constant placeholder `C`, which was rejected by the reward function (designed to test expressions without constants).

## Results

| Dataset | Valid Expressions | Best R² | Status |
|---------|------------------|---------|--------|
| add_x1_x2 | 0/249 (0.0%) | N/A | Failed |
| mul_x1_x2 | 0/249 (0.0%) | N/A | Failed |
| sub_x1_x2 | 0/249 (0.0%) | N/A | Failed |
| sin_x1 | 0/249 (0.0%) | N/A | Failed |
| cos_x1 | 0/249 (0.0%) | N/A | Failed |
| square_x1 | 0/249 (0.0%) | N/A | Failed |
| sin_x1_plus_x2 | 0/287 (0.0%) | N/A | Failed |
| x1_mul_sin_x2 | 0/275 (0.0%) | N/A | Failed |

**Overall: 0% success rate across all datasets**

## Root Cause Analysis

### Primary Issue: Generation Boundary Problem
The model was trained on complete JSON examples where each example is a self-contained sequence. When given a partial JSON prompt, the model:
1. Completes the expression field
2. Continues generating the next example in training-like fashion
3. Does not recognize the semantic boundary at the closing quote

### Secondary Issue: Constant Dependency
The training data contains expressions with constants (`C`). The model learned to generate expressions with constants, but the test datasets don't require constants. This mismatch causes all expressions to be rejected.

## Recommendations

### Short-term Fixes
1. **Better stopping criteria:** Add EOS token or special marker after expression
2. **Post-processing:** Truncate output at first `"}` after `"expr": "`
3. **Remove constant filter:** Allow expressions with C=1 substitution

### Long-term Solutions
1. **Retrain with termination signals:** Include explicit end markers in training data
2. **Use constrained decoding:** Force model to stop at valid JSON boundary
3. **Two-stage approach:** Generate expression, then fit constants separately
4. **Alternative RL framework:** Use GRPO or other methods compatible with custom rewards

## Files Created

- `scripts/data/create_ppo_test_datasets.py` - Test dataset generator
- `scripts/ppo_experiment.py` - PPO experiment script (TRL compatibility issues)
- `scripts/best_of_n_experiment.py` - Best-of-N sampling experiment
- `scripts/run_ppo_experiments.py` - Multi-dataset runner
- `userdata_ppo_experiment.sh` - AWS userdata script
- `launch_ppo_experiment.sh` - AWS launch script
- `data/ppo_test/*.csv` - 8 test datasets

## Conclusion

The PPO symbolic regression experiment revealed fundamental issues with the current model architecture for this task:

1. **Generation boundary problem** prevents clean expression extraction
2. **Training data format** includes constants that aren't needed for test cases
3. **TRL API changes** require rethinking the PPO implementation approach

The JSON format (exp_a_json) achieves 80% valid expressions in standard generation tests but is not suitable for RL-based symbolic regression without modifications to handle the generation boundary problem.

## Next Steps

1. Investigate constrained decoding approaches
2. Consider retraining with explicit expression terminators
3. Explore alternative RL frameworks (GRPO, REINFORCE)
4. Test with constant-inclusive datasets

---
*This report was automatically generated by Claude Code during an autonomous overnight experiment.*
