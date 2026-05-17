# Phase A-v2 Rigorous Experiment Plan

**Context:** Designed for a Master's Thesis to provide statistically significant findings on the efficacy of PPO vs GRPO algorithms for Symbolic Regression capabilities in LLMs. 

## Experimental Aim
1. **Prove Baseline Lift:** Establish that RL genuinely outperforms zero-shot best-of-n sampling on mathematical sequence generation.
2. **Compare PPO vs GRPO:** Measure the absolute performance envelope and the convergence stability of both gradient methods.
3. **Isolate Problem Complexity:** Evaluate across a strictly controlled easy vs. hard problem domain.

## Fixed Constants (Determined from Phase A Analysis)
To maximize signal and minimize computational waste, the following parameters are locked to their optimal settings derived from the deep-dive analysis of Phase A:
- **Reward Algorithm:** `sr_ic` (Symbolic Regression Information Criterion - balances fit vs complexity)
- **Penalty Strategy:** `gradient` (Provides smooth loss landscape compared to discrete binary)
- **Prompt Type:** `standard` (Most robust prompt across all RL variations)
- **Temperature:** `linear_annealing` (Provides maximal exploration fading into greedy exploitation)

## The Rigorous Subset (Dense Grid)
Unlike Phase A which ran 6,000+ configurations without replication, V2 runs a focused subset with **5 Random Seeds** per configuration to ensure absolute statistical significance (allowing for ANOVA testing and $p$-value extraction).

| Variable | Values |
| :--- | :--- |
| **Models** | `augustocsc/gpt2_base_infix_682k`, `augustocsc/gpt2_base_prefix_682k` |
| **Problems** | `nguyen_1` (Easy: $1+x+x^2+x^3$), `nguyen_9` (Hard: $\frac{x^6}{5} + \frac{x^4}{4} - \frac{x^2}{3}$) |
| **Algorithms** | `best_of_n`, `pure_ppo`, `bon_grpo` |
| **Total Grid Size** | 2 Models $\times$ 2 Problems $\times$ 3 Algorithms $\times$ 5 Seeds = **60 Total Runs** |

All runs will be logged to Weights & Biases under the tag `phase_a_v2_rigorous` to isolate them from the corrupted legacy dataset.
