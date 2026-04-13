# Research Proposal: Comparative Analysis of Reinforcement Learning Algorithms for Symbolic Regression with Large Language Models

**Author:** Augusto Cesar
**Program:** Graduate Research in Machine Learning
**Date:** February 2026
**Status:** Proposal Draft

---

## 1. Introduction

Symbolic regression—the task of discovering mathematical expressions that fit observed data—is a fundamental problem in scientific discovery. Traditional approaches like genetic programming have been the dominant paradigm, but recent advances in Large Language Models (LLMs) offer a promising alternative: treating expression generation as a sequence modeling task.

This research investigates how different **Reinforcement Learning (RL) algorithms** can optimize LLMs for symbolic regression. While supervised fine-tuning can teach models the syntax of mathematical expressions, RL enables optimization toward task-specific objectives such as fitting accuracy (R²), expression simplicity, and generalization.

### 1.1 Research Gap

Despite growing interest in LLMs for symbolic regression, there are **two critical gaps** in the literature:

1. **No comprehensive comparison** of RL algorithms for this specific task. Existing work typically uses a single algorithm (usually PPO) without justification.

2. **No systematic study** of how **hybrid methods** (combining Best-of-N sampling with RL) compare to pure RL approaches. Recent work suggests that iterative refinement with selection may be more effective than pure online RL for discrete generation tasks.

This research aims to fill both gaps by providing:

1. A systematic comparison of pure RL algorithms
2. A comparison of hybrid methods that combine exploration (Best-of-N) with exploitation (RL)
3. Analysis of when each approach is most effective
4. Practical guidelines for practitioners

### 1.2 Research Questions

**Pure RL:**
- **RQ1:** Which pure RL algorithm achieves the best fitting accuracy (R²) on standard benchmarks?
- **RQ2:** How do different algorithms compare in terms of sample efficiency and training stability?

**Hybrid Methods:**
- **RQ3:** Do hybrid methods (Best-of-N + RL) outperform pure RL approaches?
- **RQ4:** What is the optimal balance between exploration (sampling) and exploitation (RL optimization)?
- **RQ5:** Does maintaining a buffer of best expressions improve convergence?

**General:**
- **RQ6:** What is the trade-off between algorithm complexity and performance?
- **RQ7:** How does model scale affect the relative performance of different approaches?

---

## 2. Background

### 2.1 Symbolic Regression with LLMs

Language models can be fine-tuned to generate mathematical expressions given:
- **Input:** Variables available (e.g., x₁, x₂), allowed operators (e.g., +, *, sin), data points
- **Output:** A valid mathematical expression (e.g., `sin(x₁) + x₂²`)

The model is first trained via **Supervised Fine-Tuning (SFT)** on a dataset of valid expressions, then optimized via **Reinforcement Learning** to maximize fitting accuracy on specific problems.

### 2.2 Reinforcement Learning for LLMs

RL fine-tuning has become standard practice for aligning LLMs with desired behaviors. Key algorithms include:

| Algorithm | Year | Key Innovation |
|-----------|------|----------------|
| REINFORCE | 1992 | Basic policy gradient |
| PPO | 2017 | Clipped surrogate objective for stability |
| DPO | 2023 | Direct preference optimization without reward model |
| GRPO | 2024 | Group-relative advantages, no critic needed |
| RLOO | 2024 | Leave-one-out baseline estimation |

### 2.3 Hybrid Methods: Best-of-N + RL

Recent research suggests that combining **sampling-based exploration** with **RL-based optimization** can outperform pure approaches:

| Method | Key Idea | Reference |
|--------|----------|-----------|
| **RAFT** | Reward-ranked fine-tuning with filtered samples | Dong et al. (2023) |
| **ReST-EM** | Iterative: Generate → Filter → Train → Repeat | Gulcehre et al. (2023) |
| **BOND** | Best-of-N distillation into policy | Sessa et al. (2024) |
| **Iterative DPO** | Best-of-N creates preference pairs for DPO | — |

The intuition is that:
- **Best-of-N** provides broad exploration of the expression space
- **RL** provides efficient local optimization around promising solutions
- **Buffer of best expressions** maintains diversity and prevents forgetting

### 2.4 Nguyen Benchmarks

The Nguyen benchmark suite (Nguyen et al., 2011) is the standard evaluation for symbolic regression, containing 12 problems of varying complexity:

| Benchmark | True Expression | Complexity |
|-----------|-----------------|------------|
| Nguyen-1 | x³ + x² + x | Low |
| Nguyen-5 | sin(x²)cos(x) - 1 | Medium |
| Nguyen-8 | √x | Low |
| Nguyen-12 | x⁴ - x³ + ½x² - x | High |

---

## 3. Proposed Methodology

### 3.1 Algorithm Categories

We propose comparing **10 algorithms** across **four categories**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ALGORITHM TAXONOMY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  A. BASELINES (No Training)                                     │
│     └── Best-of-N Sampling                                      │
│                                                                 │
│  B. PURE RL (Online Learning)                                   │
│     ├── REINFORCE                                               │
│     ├── PPO                                                     │
│     ├── GRPO                                                    │
│     └── DPO                                                     │
│                                                                 │
│  C. ITERATIVE METHODS (Offline/Online Alternation)              │
│     ├── Expert Iteration (ReST)                                 │
│     └── Iterative DPO                                           │
│                                                                 │
│  D. HYBRID METHODS (Best-of-N + RL) ← KEY CONTRIBUTION          │
│     ├── BoN-PPO (Best-of-N guided PPO)                          │
│     ├── BoN-GRPO (Best-of-N guided GRPO)                        │
│     └── RAFT (Reward-ranked Fine-Tuning)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Detailed Algorithm Descriptions

#### A. Baseline
1. **Best-of-N Sampling:** Generate N expressions, select best by R². No training—establishes upper bound of base model capability.

#### B. Pure RL Methods

2. **REINFORCE:** Vanilla policy gradient with moving average baseline. Simplest RL approach.

3. **PPO (Proximal Policy Optimization):** Industry standard with clipped objective and value network. Most robust but complex.

4. **GRPO (Group Relative Policy Optimization):** Recent algorithm from DeepSeek. Uses group statistics instead of learned baseline.

5. **DPO (Direct Preference Optimization):** Treats RL as classification over preferences. Requires creating preference pairs from R² comparisons.

#### C. Iterative Methods

6. **Expert Iteration / ReST:**
   - Generate N expressions per benchmark
   - Filter top K% by R²
   - Fine-tune on filtered set
   - Repeat for T iterations

7. **Iterative DPO:**
   - Generate N expressions
   - Create preference pairs: (high R², low R²)
   - Train with DPO
   - Repeat with updated policy

#### D. Hybrid Methods (Key Contribution)

8. **BoN-PPO (Best-of-N Guided PPO):**
   - Maintain buffer B of best expressions found
   - Each epoch: (a) Best-of-N exploration → update buffer, (b) PPO training with buffer-augmented rewards
   - Buffer provides stable targets and diverse starting points

9. **BoN-GRPO (Best-of-N Guided GRPO):**
   - Same as BoN-PPO but using GRPO for optimization
   - Group advantages computed including buffer samples
   - Expected to be more sample-efficient than BoN-PPO

10. **RAFT (Reward-Ranked Fine-Tuning):**
    - Generate large batch of N expressions
    - Rank by R², keep top K%
    - Fine-tune with weighted loss (higher R² = higher weight)
    - Combines Best-of-N selection with reward-weighted training

### 3.3 Hybrid Algorithm Details

#### BoN-RL Framework (Novel Contribution)

```
Algorithm: Best-of-N Guided RL (BoN-RL)
─────────────────────────────────────────

Input: SFT model π₀, benchmark data (X, y), buffer size K

Initialize:
    π ← π₀                          # Policy from SFT
    B ← ∅                           # Expression buffer (priority queue by R²)

For epoch = 1 to max_epochs:

    # ═══════════════════════════════════════════════════════
    # PHASE 1: EXPLORATION (Best-of-N)
    # ═══════════════════════════════════════════════════════

    Sample N expressions from π with temperature τ_explore
    For each expression e:
        r² ← evaluate(e, X, y)
        If r² > min(B) or |B| < K:
            Add (e, r²) to buffer B
            Remove lowest if |B| > K

    # ═══════════════════════════════════════════════════════
    # PHASE 2: EXPLOITATION (RL Update)
    # ═══════════════════════════════════════════════════════

    # Option A: BoN-PPO
    Compute advantages using buffer mean as baseline:
        baseline ← mean(r² for all (e, r²) in B)
        A_i ← r²_i - baseline
    Update π using PPO with advantages A

    # Option B: BoN-GRPO
    Compute group advantages including buffer samples:
        group ← current_batch ∪ sample(B, k)
        A_i ← (r²_i - mean(group)) / std(group)
    Update π using policy gradient

    # ═══════════════════════════════════════════════════════
    # PHASE 3: BUFFER REFINEMENT
    # ═══════════════════════════════════════════════════════

    If epoch % refresh_interval == 0:
        Re-evaluate buffer expressions with current π
        Remove expressions that π can now surpass

    # Early stopping
    If max(B) ≥ target_r²:
        Return best expression from B

Output: Best expression found, trained policy π
```

#### Key Design Choices for Hybrid Methods

| Design Choice | Options | Rationale |
|---------------|---------|-----------|
| **Buffer size K** | 10, 50, 100 | Trade-off: diversity vs. quality |
| **Exploration temperature** | 0.9 - 1.2 | Higher for exploration phase |
| **Exploitation temperature** | 0.5 - 0.7 | Lower for focused optimization |
| **Buffer sampling strategy** | Uniform, Proportional to R² | How to weight buffer in training |
| **Refresh interval** | Every 5-10 epochs | When to re-evaluate buffer |
| **RL algorithm** | PPO, GRPO | Which base RL to use |

### 3.4 Experimental Design

#### Base Models
All experiments use GPT-2 models fine-tuned on 682K synthetic expressions:
- **GPT-2 Base** (124M parameters)
- **GPT-2 Medium** (355M parameters)
- **GPT-2 Large** (774M parameters)

#### Notation Formats
- **Infix notation:** `sin(x) + cos(x)`
- **Prefix notation:** `+ sin x cos x`

#### Evaluation Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **R² Score** | Coefficient of determination | Primary performance metric |
| **Valid Rate** | % syntactically valid expressions | Generation quality |
| **Success Rate** | % achieving R² ≥ 0.99 | Task completion |
| **Sample Efficiency** | Samples to reach target R² | Training cost |
| **Training Stability** | Variance across runs | Reliability |
| **Wall-Clock Time** | Computational cost | Practical efficiency |
| **Buffer Utilization** | How often buffer expressions are used | Hybrid method analysis |

#### Experimental Protocol

```
EXPERIMENT 1: Pure RL Comparison
────────────────────────────────
For each algorithm A in {REINFORCE, PPO, GRPO, DPO}:
    For each model M in {Base, Medium, Large}:
        For each benchmark B in Nguyen 1-12:
            For run R in 1-5:
                Train and evaluate
            Report: mean ± std

EXPERIMENT 2: Iterative Methods
───────────────────────────────
For each algorithm A in {Expert-Iteration, Iterative-DPO}:
    Same protocol as Experiment 1

EXPERIMENT 3: Hybrid Methods (Key Experiment)
─────────────────────────────────────────────
For each algorithm A in {BoN-PPO, BoN-GRPO, RAFT}:
    For each buffer_size K in {10, 50, 100}:
        For each model M in {Base, Medium, Large}:
            For each benchmark B in Nguyen 1-12:
                For run R in 1-5:
                    Train and evaluate
                Report: mean ± std

EXPERIMENT 4: Ablation Study
────────────────────────────
Compare BoN-GRPO with:
    - No buffer (pure GRPO)
    - Buffer but no refresh
    - Different buffer sizes
    - Different exploration temperatures
```

### 3.5 Implementation Details

#### Reward Function
```python
def compute_reward(expression: str, X: np.ndarray, y: np.ndarray) -> float:
    """Compute R² score as reward signal."""
    try:
        y_pred = evaluate_expression(expression, X)
        if not np.all(np.isfinite(y_pred)):
            return -1.0  # Invalid output

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return np.clip(r2, -1.0, 1.0)
    except:
        return -1.0  # Parsing/evaluation error
```

#### Buffer Implementation
```python
class ExpressionBuffer:
    """Priority queue maintaining best expressions found."""

    def __init__(self, max_size: int = 100):
        self.buffer = []  # List of (r2, expression) tuples
        self.max_size = max_size

    def add(self, expression: str, r2: float):
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, (r2, expression))
        elif r2 > self.buffer[0][0]:
            heapq.heapreplace(self.buffer, (r2, expression))

    def sample(self, k: int) -> List[Tuple[str, float]]:
        """Sample k expressions, weighted by R²."""
        ...

    def get_baseline(self) -> float:
        """Return mean R² of buffer for advantage computation."""
        return np.mean([r2 for r2, _ in self.buffer])
```

#### Hyperparameters

| Parameter | Pure RL | Hybrid Methods |
|-----------|---------|----------------|
| Learning rate | 3e-5 | 3e-5 |
| Batch size | 16 | 16 |
| Temperature (exploit) | 0.7 | 0.6 |
| Temperature (explore) | — | 1.0 |
| Buffer size K | — | {10, 50, 100} |
| Best-of-N samples | — | 64 per epoch |
| KL coefficient | 0.01 | 0.01 |
| Max epochs | 100 | 100 |

#### Computational Resources
- **Hardware:** NVIDIA A10G GPU (24GB VRAM)
- **Estimated compute:** ~300 GPU-hours for full experimental suite

---

## 4. Expected Contributions

### 4.1 Scientific Contributions

1. **First comprehensive comparison** of RL algorithms for symbolic regression with LLMs
2. **Novel hybrid framework** (BoN-RL) combining Best-of-N exploration with RL optimization
3. **Empirical analysis** of pure RL vs. hybrid approaches
4. **Ablation study** on buffer design choices
5. **Practical guidelines** for algorithm selection
6. **Open-source implementations** for reproducibility

### 4.2 Hypotheses

| ID | Hypothesis | Rationale |
|----|------------|-----------|
| **H1** | Hybrid methods > Pure RL | Buffer provides stable targets and exploration |
| **H2** | BoN-GRPO > BoN-PPO | GRPO more suited for group-based training |
| **H3** | Larger buffer improves stability | More diverse reference points |
| **H4** | Hybrid methods more sample-efficient | Buffer reuses good expressions |
| **H5** | Pure GRPO > Pure PPO | Simpler, designed for LLMs |
| **H6** | Iterative methods competitive with online RL | Effective for discrete outputs |
| **H7** | Larger models benefit more from hybrid | More capacity to exploit buffer |

### 4.3 Deliverables

1. **Research paper** with comprehensive experimental comparison
2. **Open-source code** for all 10 algorithms
3. **Trained models** on HuggingFace Hub
4. **Benchmark results** as reference for future work
5. **Practical recommendations** for practitioners

---

## 5. Preliminary Results

### 5.1 Completed Work

- ✅ Supervised fine-tuning of 6 models (Base/Medium/Large × Infix/Prefix)
- ✅ Implementation of PPO, GRPO, REINFORCE, Best-of-N
- ✅ Initial evaluation on Nguyen benchmarks
- ✅ Infrastructure for AWS training

### 5.2 Current Model Performance (SFT Only)

| Model | Valid Rate | Mean R² (Nguyen 1-12) |
|-------|------------|----------------------|
| Base Infix | 85.2% | 0.72 |
| Medium Infix | 89.1% | 0.78 |
| Large Infix | 91.3% | 0.82 |
| Base Prefix | 87.4% | 0.74 |
| Medium Prefix | 90.3% | 0.79 |
| Large Prefix | 92.1% | 0.83 |

These establish the **baseline before RL optimization**.

### 5.3 Preliminary RL Results (PPO only)

| Model | SFT R² | PPO R² | Improvement |
|-------|--------|--------|-------------|
| Base Infix | 0.72 | 0.81 | +12.5% |
| Medium Infix | 0.78 | 0.86 | +10.3% |

These preliminary results suggest RL optimization provides meaningful improvements.

---

## 6. Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Phase 1** | Implement DPO, RLOO, Expert Iteration | 2 weeks |
| **Phase 2** | Implement hybrid methods (BoN-PPO, BoN-GRPO, RAFT) | 2 weeks |
| **Phase 3** | Hyperparameter tuning on Nguyen-5 | 1 week |
| **Phase 4** | Experiment 1: Pure RL comparison | 2 weeks |
| **Phase 5** | Experiment 2-3: Iterative and Hybrid methods | 2 weeks |
| **Phase 6** | Experiment 4: Ablation study | 1 week |
| **Phase 7** | Statistical analysis and visualization | 1 week |
| **Phase 8** | Paper writing | 2 weeks |
| **Total** | | **13 weeks** |

---

## 7. Related Work

### 7.1 Symbolic Regression
- **Genetic Programming:** Koza (1992), Schmidt & Lipson (2009)
- **Neural approaches:** Petersen et al. (2021), Biggio et al. (2021)
- **Transformer-based:** Kamienny et al. (2022), Vastl et al. (2022)
- **RL for SR:** Landajuela et al. (2021), Mundhenk et al. (2021)

### 7.2 RL for LLMs
- **RLHF:** Ouyang et al. (2022), Bai et al. (2022)
- **PPO for LLMs:** Schulman et al. (2017), Ziegler et al. (2019)
- **GRPO:** DeepSeek-R1 (2024)
- **DPO:** Rafailov et al. (2023)
- **RLOO:** Ahmadian et al. (2024)

### 7.3 Hybrid and Iterative Methods
- **Expert Iteration:** Silver et al. (2017)
- **STaR:** Zelikman et al. (2022)
- **ReST:** Gulcehre et al. (2023)
- **RAFT:** Dong et al. (2023)
- **BOND:** Sessa et al. (2024)

---

## 8. Conclusion

This research will provide the **first systematic comparison** of reinforcement learning algorithms for symbolic regression with large language models, with special emphasis on **hybrid methods** that combine Best-of-N sampling with RL optimization.

The key innovations are:

1. **Comprehensive comparison** of 10 algorithms across 4 paradigms
2. **Novel BoN-RL framework** that maintains a buffer of best expressions to guide learning
3. **Rigorous experimental design** with ablation studies on buffer design
4. **Practical guidelines** for algorithm selection

By comparing:
- **Pure RL** (REINFORCE, PPO, GRPO, DPO)
- **Iterative methods** (Expert Iteration, Iterative DPO)
- **Hybrid methods** (BoN-PPO, BoN-GRPO, RAFT)

...across multiple model scales (124M-774M) and standard benchmarks (Nguyen 1-12), this research will establish **when and why** hybrid methods outperform pure RL approaches—or whether pure RL remains competitive.

---

## References

1. Nguyen, Q. U., et al. (2011). Semantically-based crossover in genetic programming.
2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
3. Rafailov, R., et al. (2023). Direct Preference Optimization.
4. DeepSeek-AI. (2024). DeepSeek-R1: Incentivizing Reasoning in LLMs via RL.
5. Kamienny, P., et al. (2022). End-to-end symbolic regression with transformers.
6. Zelikman, E., et al. (2022). STaR: Self-Taught Reasoner.
7. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
8. Gulcehre, C., et al. (2023). Reinforced Self-Training (ReST) for Language Modeling.
9. Dong, H., et al. (2023). RAFT: Reward rAnked FineTuning.
10. Ahmadian, A., et al. (2024). Back to Basics: Revisiting REINFORCE Style Optimization.
11. Sessa, P., et al. (2024). BOND: Aligning LLMs with Best-of-N Distillation.

---

## Appendix A: Algorithm Pseudocode

### A.1 PPO (Pure)
```
Initialize policy π_θ, value function V_φ
For each iteration:
    Collect trajectories using π_θ
    Compute advantages A_t = R_t - V_φ(s_t)
    For K epochs:
        Update θ by maximizing clipped objective
        Update φ by minimizing value loss
```

### A.2 GRPO (Pure)
```
Initialize policy π_θ
For each iteration:
    Generate group of N expressions
    Compute rewards r_1, ..., r_N
    Compute group advantage: A_i = (r_i - mean(r)) / std(r)
    Update θ by policy gradient with group advantages
```

### A.3 DPO
```
Given preference pairs (y_win, y_lose) created from R² ranking:
    Compute log-ratio: β * log(π_θ(y_win)/π_ref(y_win)) - β * log(π_θ(y_lose)/π_ref(y_lose))
    Update θ to maximize sigmoid of log-ratio
```

### A.4 Expert Iteration
```
Initialize policy π_θ from SFT
For each iteration:
    Generate N expressions per benchmark
    Filter top K% by R² score
    Fine-tune π_θ on filtered expressions
```

### A.5 BoN-GRPO (Hybrid - Key Algorithm)
```
Initialize policy π_θ from SFT, buffer B ← ∅

For each epoch:
    # Exploration
    Generate N expressions with high temperature
    Add best K to buffer B

    # Exploitation
    Generate M expressions with low temperature
    Include samples from buffer B in group
    Compute group advantages over combined set
    Update π_θ with policy gradient

    # Buffer maintenance
    Periodically re-evaluate and prune buffer
```

### A.6 RAFT
```
Initialize policy π_θ from SFT

For each iteration:
    Generate large batch of N expressions
    Compute R² for each
    Rank and filter top K%
    Compute weights w_i proportional to R²
    Fine-tune π_θ with weighted loss: Σ w_i * log π(e_i)
```

---

## Appendix B: Comparison Matrix

| Algorithm | Category | Needs Value Net | Needs Pairs | Uses Buffer | Complexity |
|-----------|----------|-----------------|-------------|-------------|------------|
| Best-of-N | Baseline | No | No | No | O(N) |
| REINFORCE | Pure RL | No | No | No | Low |
| PPO | Pure RL | Yes | No | No | High |
| GRPO | Pure RL | No | No | No | Medium |
| DPO | Pure RL | No | Yes | No | Medium |
| Expert Iter | Iterative | No | No | Implicit | Low |
| Iter-DPO | Iterative | No | Yes | No | Medium |
| BoN-PPO | Hybrid | Yes | No | Yes | High |
| BoN-GRPO | Hybrid | No | No | Yes | Medium |
| RAFT | Hybrid | No | No | Implicit | Low |

---

*Document prepared for academic review.*
