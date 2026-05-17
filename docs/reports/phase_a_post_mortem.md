# Phase A Post-Mortem: Integrity and Architectural Failures

**Date:** February 2026
**Context:** Review of the initial Phase A Reinforcement Learning (RL) experiment for Symbolic Regression.

## Executive Summary
An exhaustive post-mortem analysis of the 9,751 Weights & Biases (W&B) runs generated during Phase A reveals severe methodological and technical flaws. The dataset produced by this phase is compromised by a broken baseline, a failure to execute a balanced statistical design, and incomplete data aggregation. 

Consequently, the results from Phase A **cannot be used to statistically defend any hyperparameter choice** in a rigorous academic setting (such as a Master's Thesis). This document details the specific failures that necessitate a controlled, rigorous rerun (Phase A-v2).

## 1. The Baseline Crash Anomaly
The fundamental premise of evaluating an RL algorithm is confirming that it outperforms a non-RL baseline. In this experiment, the chosen baseline was `best_of_n` (pure sampling).

**The Failure:**
All `best_of_n` runs reported a final $R^2$ of `0.000` across all problems. 

**Root Cause:**
This was not due to the inherent difficulty of the problems, but a catastrophic software exception in `algorithms/best_of_n.py` at line 254:
```python
if reward_result.is_valid:
AttributeError: 'float' object has no attribute 'is_valid'
```
The architecture expected the reward function to return a structured `RewardResult` object (containing `is_valid`, `r2`, etc.), but instead it received a primitive `float`. This caused the script to crash internally, failing to evaluate the baseline entirely.

**Impact:**
Without a functioning `best_of_n` baseline, any comparative claim that "PPO improves symbolic sequence generation" is scientifically invalid, as there is no floor performance metric to compare against.

## 2. Collapse of the Factorial Grid (Data Balance)
Phase A was designed as a massive combinatorial grid search comprising 6,912 unique hyperparameter configurations (Models $\times$ Problems $\times$ Algorithms $\times$ Rewards $\times$ Penalties $\times$ Temperatures $\times$ Prompts $\times$ Noise).

**The Failure:**
The execution of the grid was severely fragmented and unbalanced.
1. **Missing Data:** 280 distinct hyperparameter combinations completely failed to execute or were lost, creating holes in the factorial design.
2. **Skewed Marginals:** The distribution of test problems was unequal (e.g., 2,472 runs for `nguyen_5` vs. 2,214 for `nguyen_1`), preventing clean aggregations.

**Impact:**
Because the dataset is not a perfect hypercube, marginal means (e.g., "What is the average performance of PPO across all settings?") are biased heavily toward the configurations that successfully ran the most often, rather than representing true algorithmic capability.

## 3. The "Lucky Seed" Trap (Lack of Replication)
In stochastic optimization (especially RL and LLM sampling), performance variance across random initialization seeds is massive.

**The Failure:**
The experiment design conceptually failed to allocate compute resources for seed replication. 
* Total RL Runs Analyzed: 7,019
* Unique Configurations Evaluated: 6,632
* **Configurations with exactly 1 seed:** 6,245 (89% of the dataset)
* Configurations with exactly 2 seeds: 387

**Impact:**
It is exceptionally common in Symbolic Regression for a model to "get lucky" on a single seed and perfectly solve the equation ($R^2 = 1.0$), even if the underlying algorithm is unstable. Because 89% of setups only ran a single seed, it is statistically impossible to distinguish between a truly robust hyperparameter configuration and statistical anomaly. Reporting a "best configuration" from this dataset is equivalent to reporting an outlier.

## 4. Aggregation Data Loss
The pipeline used to consolidate the W&B runs into a tabular format (`analyze_all_phase_a.py`) contained a logging failure.

**The Failure:**
The `reward` and `penalty` strategy arguments passed via command line were not parsed correctly into the final CSV, leaving thousands of rows with `unknown` values for these critical independent variables.

**Impact:**
While this was patchable post-hoc by writing a custom YAML parser to iterate through all 9,751 raw `config.yaml` files, it highlights structural fragility in the data engineering pipeline that must be hardened before Phase B.

---

## Conclusion and Path Forward
Phase A acts as a valuable "dry run" that surfaced critical bugs in baseline evaluation and data logging, but the resulting dataset is statistically compromised. 

To satisfy the rigorous defense requirements of a Master's Thesis, a **Phase A-v2** must be executed on an isolated, reduced grid. This v2 grid must:
1. Fix the `RewardResult` wrapper bug in the baseline.
2. Reduce the hyperparameter search space to a manageable subset (e.g., locking `sr_ic` and `gradient` penalties based on directional findings).
3. **Mandate a minimum of 5 random seeds per configuration** to allow for proper variance reporting (mean $\pm$ standard deviation) and statistical significance testing between algorithms.
