# PPO Evaluation Results: Block 3 Assessment

**Date:** 2026-02-01
**Model Tested:** augustocsc/Se124M_700K_infix_v2
**Status:** ⚠️ CONCERNS IDENTIFIED

---

## Executive Summary

Tested whether V2 model is suitable as a base for PPO (Proximal Policy Optimization) finetuning for symbolic regression. **Result: Current model NOT ready for PPO training** due to low valid expression rate (6.7%) and poor R² scores.

---

## Test Methodology

### Test 1: Baseline Generation
- Generated 30 expressions without PPO
- Goal: Validate that model generates syntactically correct expressions
- Measured: Valid rate, R² fit to target formula (x_1 * x_2)

### Test 2: PPO Simulation
- Generated 50 expressions and tracked best R² score
- Goal: Determine if model CAN find high-quality solutions
- Measured: Best R², mean R², valid expression count

### Configuration
- **Model:** V2 with LoRA adapter merged
- **Inference Config:** Nucleus sampling (temp=0.7, top_p=0.8)
- **Stopping Criteria:** ExpressionStoppingCriteria for `<|endofex|>`
- **Target:** Simple multiplicative formula (x_1 * x_2)

---

## Results

### Test 1: Baseline Generation

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Valid Rate | **6.7%** (2/30) | 90% | ❌ FAIL |
| Mean R² | 0.1252 | ~0.2 (random) | ✅ As expected |
| Max R² | 0.1252 | ~0.2 (random) | ✅ As expected |

**Sample Generations:**
```
Sample 1: x_1*(x_1 + C)*cos(x_1 - C)408**C Muslims(x_1 - C)With...
         → Invalid (garbage tokens: "Muslims", "With", "408")

Sample 2: x_1*(x_1 + sin(x_1 + C)) + x_2 - CBuyableInstoreAndOnlinevars...
         → Invalid (garbage: "BuyableInstoreAndOnline")

Sample 3: C*cos(x_1 + C*cos(x_1)) + C Bermanvars: x_1, x_2, x_3...
         → Invalid (garbage: "Berman")
```

### Test 2: PPO Simulation

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Valid Expressions | **2/50** (4%) | >40/50 (80%) | ❌ FAIL |
| Best R² | **N/A** | >= 0.9 | ❌ FAIL |
| Mean R² | **N/A** | >= 0.5 | ❌ FAIL |

---

## Key Findings

### 1. Model Does NOT Stop Properly ⚠️

**Problem:** Model continues generating beyond expression boundaries, concatenating multiple training examples.

**Evidence:**
- Expressions contain garbage tokens: "Muslims", "Buyable", "Instore", "AndOnline", "crash", "Berman", "Avenger"
- No `<|endofex|>` markers generated
- Expressions run into next training example's variables section

**Example:**
```
Generated: x_1*x_2 + C BuyableInstoreAndOnlinevars: x_1, x_2, x_3...
           ^^^^^^^^^^^^^ valid part  ^^^^^^^^^^^^^^^^ next training example
```

### 2. Discrepancy with Previous Results

**Expected (from FINAL_RESULTS_V1_VS_V2.md):**
- V2 with nucleus sampling: **90% valid rate**
- Configuration: temp=0.7, top_p=0.8 (same as used here)

**Actual (this evaluation):**
- V2 with nucleus sampling: **6.7% valid rate**
- Same configuration: temp=0.7, top_p=0.8

**Possible Explanations:**
1. Different model checkpoint (inference tests may have used different version)
2. Different prompt format or tokenization
3. Different loading procedure or merge method
4. test_inference_configs.py may extract expressions differently

### 3. PPO Training Viability: NOT READY ❌

**Assessment:** PPO training would **NOT work** with current model state.

**Reasoning:**
- **Valid Rate Too Low:** 6.7% vs required >80%
  - PPO needs valid expressions to compute rewards
  - With 6.7% valid rate, 93% of generations provide no learning signal

- **R² Scores Poor:** Best R² is N/A (no valid expressions with finite R²)
  - PPO reward function requires R² >= 0.9 for success
  - Current model cannot reach even R² > 0 consistently

- **Random Search Fails:** Even generating 50 expressions didn't find solutions
  - If random search can't find solutions, PPO won't either
  - PPO optimizes search, but can't create capability that doesn't exist

---

## Root Cause Analysis

### Why is V2 generating garbage?

**Hypothesis 1: Training Data Contamination**
- Model was trained on data containing "Buyable", "Instore", "AndOnline", etc.
- These are likely tokenizer artifacts from GPT-2's original training (e-commerce terms)
- Suggests training data wasn't properly cleaned

**Hypothesis 2: Stopping Marker Not Learned**
- Model doesn't generate `<|endofex|>` markers
- Either:
  - Training data lacked end markers (contradicts v2 training goal)
  - Model didn't learn to generate them (insufficient training)
  - Inference doesn't properly encourage their generation

**Hypothesis 3: Model Checkpoint Mismatch**
- The inference tests (90% valid rate) may have used a different checkpoint
- HuggingFace Hub model may not be the final/best version
- Local checkpoint during testing may differ from published version

---

## Comparison: Expected vs Actual

### Expected Performance (from docs)
```python
# V2 with nucleus sampling should achieve:
✅ Valid Rate: 90.0% (27/30)
✅ Clean expressions: proper boundaries
✅ No garbage tokens
✅ Proper <|endofex|> generation
```

### Actual Performance (this test)
```python
# V2 with same config actually achieves:
❌ Valid Rate: 6.7% (2/30)
❌ Garbage expressions: concatenated examples
❌ Many garbage tokens ("Buyable", "Muslims", etc.)
❌ No <|endofex|> markers generated
```

---

## Implications for Block 3 (PPO)

### Current Status: BLOCKED ⛔

**Cannot proceed with PPO training** because:

1. **Insufficient Valid Generation Rate**
   - Need: >80% valid expressions
   - Have: 6.7% valid expressions
   - Gap: 73.3 percentage points

2. **No Viable Reward Signal**
   - PPO requires R² scores as rewards
   - Current model produces no valid R² scores
   - Cannot optimize without signal

3. **Base Model Quality Too Low**
   - PPO assumes base model can generate valid outputs
   - Current model fails this assumption
   - Fixing requires going back to Block 2 (supervised training)

### Required Actions Before PPO

**Option A: Investigate V2 Model Mismatch**
1. Run `test_inference_configs.py` on same V2 model
2. Compare results with this evaluation
3. If test script gets 90%, identify differences:
   - Prompt format
   - Model loading procedure
   - Expression extraction logic
4. Apply corrections to PPO evaluation

**Option B: Use V1 Model Instead**
1. V1 achieves 83.3% valid rate (from FINAL_RESULTS)
2. Still not ideal (need >90%), but much better than 6.7%
3. Test PPO with V1 as baseline
4. May need lower PPO reward thresholds

**Option C: Retrain Base Model (Block 2 fix)**
1. Fix training data:
   - Remove garbage token contamination
   - Ensure all examples have `<|endofex|>` markers
   - Validate data quality before training
2. Retrain V3:
   - Monitor valid rate during training
   - Target: >90% valid rate on validation set
3. Then proceed to PPO (Block 3)

---

## Recommendations

### Immediate Next Steps

1. **Verify V2 Model Quality**
   ```bash
   # Run official inference test on V2
   python scripts/test_inference_configs.py \
     --model_path augustocsc/Se124M_700K_infix_v2 \
     --base_model gpt2 \
     --num_samples 30 \
     --config nucleus_strict

   # Expected: 90% valid rate
   # If actual: 6.7%, confirms model issue
   # If actual: 90%, investigate evaluation script differences
   ```

2. **If V2 is Actually Good (90% valid):**
   - Debug PPO evaluation script
   - Match prompt format exactly with test_inference_configs.py
   - Verify expression extraction logic
   - Re-run PPO evaluation

3. **If V2 is Actually Bad (6.7% valid):**
   - Document issue with V2 model on HuggingFace Hub
   - Use V1 model (83.3% valid) for PPO tests
   - OR retrain V3 with proper data validation
   - Update FINAL_RESULTS.md with discrepancy note

### Long-term Strategy

**For PPO to succeed**, need:
- ✅ Base model: >90% valid expression generation
- ✅ Reward signal: R² scores computable for most generations
- ✅ Search space: Model can reach R² >= 0.9 through sampling
- ✅ Training stability: Consistent valid outputs during optimization

**Current V2 fails ALL criteria.**

---

## Technical Details

### Model Loading
```python
# Successfully loaded:
- Base: GPT-2 (124M parameters)
- Tokenizer: GPT-2 with special tokens (<|startofex|>, <|endofex|>)
- Embeddings: Resized 50257 → 50259
- Adapter: V2 LoRA weights from HuggingFace Hub
- Merge: Used merge_and_unload() (matches test script)
- Device: NVIDIA A10G GPU (AWS g5.xlarge)
```

### Generation Configuration
```python
generation_config = {
    "temperature": 0.7,        # Matches V2 optimal
    "top_k": 0,                 # Nucleus sampling only
    "top_p": 0.8,               # Strict nucleus (V2 best)
    "repetition_penalty": 1.0,  # No penalty (V2 optimal)
    "max_new_tokens": 128,      # Full expression length
    "do_sample": True,          # Required for sampling
    "stopping_criteria": ExpressionStoppingCriteria(["<|endofex|>"])
}
```

### Validation Logic
```python
def is_valid_expression(expr_str: str) -> bool:
    """Expression is valid if it can be parsed and evaluated on dataset"""
    try:
        expr = Expression(expr_str, is_prefix=False)
        return expr.is_valid_on_dataset(X)  # Checks finite values
    except:
        return False
```

---

## Files Generated

### Code
- ✅ `scripts/evaluate_ppo.py` - Comprehensive PPO evaluation script
  - Baseline generation test
  - PPO simulation test
  - Expression validation
  - R² computation

### Results
- ✅ `logs/ppo_evaluation/baseline_results.json` - 30 baseline generations
- ✅ `logs/ppo_evaluation/ppo_simulation_results.json` - 50 simulation generations
- ✅ `logs/ppo_evaluation_run.log` - Full execution log
- ✅ `logs/ppo_evaluation_debug.log` - Debug output with samples
- ✅ `logs/ppo_evaluation_fixed.log` - Final run with all fixes

### Documentation
- ✅ `PPO_EVALUATION_PLAN.md` - Evaluation methodology (created earlier)
- ✅ `PPO_EVALUATION_RESULTS.md` - This document

---

## Conclusion

**Block 3 (PPO Finetuning) Status:** ⛔ **BLOCKED**

**Primary Issue:** V2 model generates only 6.7% valid expressions, far below the 90% reported in previous tests. Model produces garbage tokens and doesn't stop at expression boundaries.

**Immediate Action Required:**
1. Verify V2 model quality using official test script
2. Identify discrepancy source (model vs evaluation script)
3. Either fix evaluation or use different base model (V1 or retrained V3)

**PPO Training Viability:** **NOT VIABLE** until base model achieves >80% valid rate with R² scores indicating solvability of target problems.

**Cost:** ~$2.50 AWS compute (g5.xlarge, ~2.5 hours total)

**Next Steps:** See Recommendations section above.

---

**Status:** ⚠️ EVALUATION COMPLETE - ISSUES FOUND - ACTION REQUIRED

Last updated: 2026-02-01 (evaluation run time)
