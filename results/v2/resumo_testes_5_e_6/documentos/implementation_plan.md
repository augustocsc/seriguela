# Buffer Re-tokenization Fix

Fix the elite buffer integration so buffer samples actively participate in policy gradient updates, instead of being silently discarded.

## User Review Required

> [!IMPORTANT]
> **Test 6 will only re-run `bon_grpo` and `bon_ppo`** (since `pure_grpo` and `pure_ppo` are unchanged by this fix). Test 5 results for the pure variants remain valid for comparison.

> [!WARNING]
> Re-tokenization adds ~20% compute overhead per training step (one extra forward pass per buffer sample). On a Colab T4 GPU, this should be negligible given the small batch sizes.

## Proposed Changes

### Core Fix

#### [MODIFY] [base_trainer.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/algorithms/base_trainer.py)

1. **Add `retokenize_expression()` method** (~30 lines)
   - Takes an expression string (e.g. `"x_1**2 + x_1"`)
   - Reconstructs the full text: `prompt + expression + '"`
   - Tokenizes it → gets `tokens` (the generated portion only)
   - Forward pass through current policy → gets per-token `log_probs`
   - Returns a proper `Rollout` with real `tokens` and `log_probs`

2. **Fix `train_step()` buffer integration** (lines 569-590)
   - Replace the current dead-rollout creation with a call to `retokenize_expression()`
   - Add `try/except` so a failed re-tokenization doesn't crash the entire step
   - Add a `from_buffer=True` flag on the rollout for metrics tracking

#### [ALREADY MODIFIED] [bon_grpo.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/algorithms/bon_grpo.py)

- Shuffle already applied in previous fix — no further changes needed

---

### Quick Smoke Test (for Colab)

#### [NEW] [test_retokenize_fix.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/test_retokenize_fix.py)

A minimal test script that:
1. Loads the model and creates a BoNGRPOTrainer and BoNPPOTrainer
2. Runs **3 training steps** on `nguyen_1` with seed 42
3. Asserts that `retokenize_expression()` returns non-empty tokens
4. Asserts that buffer samples appear in gradient updates (by checking update stats)
5. Runs in **~2-3 minutes** on a T4 GPU

---

### Full Comparison Test (Test 6)

#### [NEW] [run_pre_phase_t6.py](file:///c:/Users/madeinweb/seriguela/2_training/reinforcement/run_pre_phase_t6.py)

A script that reproduces Test 5 parameters **only for the fixed algorithms**:

| Parameter | Value |
|-----------|-------|
| Algorithms | `bon_grpo`, `bon_ppo` |
| Benchmarks | `nguyen_1`, `nguyen_5`, `nguyen_9` |
| Seeds | `42`, `123`, `456` |
| Model | `augustocsc/gpt2_base_infix_682k` |
| Temperature | `cosine_annealing` |
| Reward | `sr_ic` |
| Penalty | `gradient` |
| Max Steps | `50` |
| Batch Size | `64` |
| Output | `results/pre_phase__t6/` |

Total: **2 algorithms × 3 benchmarks × 3 seeds = 18 runs**

Results will be saved with the same naming convention as Test 5 (`aggregate_bon_grpo_nguyen_1_seed42.json`, etc.) so they can be compared side-by-side.

## Verification Plan

### Automated Test
```bash
# Run on Google Colab after cloning the repo
cd 2_training/reinforcement
python test_retokenize_fix.py
```
Expected: all assertions pass, prints confirmation that buffer rollouts have tokens.

### Full Comparison Test
```bash
# Run on Google Colab (T4 GPU, ~30 minutes)
cd 2_training/reinforcement
python run_pre_phase_t6.py
```
Expected: results in `results/pre_phase__t6/`. Compare with Test 5:
- `bon_grpo` should now show **different** results from `pure_grpo` (Test 5)
- `bon_ppo` may show improved average R² on harder benchmarks
