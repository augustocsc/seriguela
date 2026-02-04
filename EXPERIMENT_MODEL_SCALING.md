# Experiment: Model Size Impact on Symbolic Regression

**Research Project**: Seriguela - Language Models for Symbolic Regression

**Date**: February 2025

**Status**: ⏳ In Progress

---

## Abstract

This experiment investigates the impact of model size on the ability of GPT-2 language models to generate valid and complex mathematical expressions for symbolic regression. We train three model variants (Base: 124M, Medium: 355M, Large: 774M parameters) on 700K synthetic expressions using LoRA fine-tuning and evaluate them across multiple dimensions: validity, complexity, diversity, and performance on Nguyen benchmarks with reinforcement learning optimization.

**Hypothesis**: Larger models possess greater capacity to learn compositional patterns, resulting in more complex, valid, and diverse expression generation.

**Key Question**: Is the increased computational cost of larger models justified by improved expression quality and benchmark performance?

---

## 1. Introduction

### 1.1 Motivation

Prior work (see `EXPERIMENT_RESULTS.md`) demonstrated that JSON-formatted training (EXP-A) achieves 80% valid expression rates compared to 0.5% with EOS token approach. However, evaluation on Nguyen-5 benchmark revealed a critical limitation:

**Problem**: The base model (GPT-2 124M) generates structurally simple expressions that fail on complex benchmarks.

**Evidence** (Nguyen-5 analysis):
- Valid expressions: 39.4%
- All valid expressions: R² = -1.0 (terrible fit)
- Power operations (x²): Only 15.9%
- Nested trigonometric functions: 0%
- Average depth: 1.40 (target requires 2+)

**Root Cause**: Model learns syntactically valid but structurally trivial expressions. Without proper complexity, all rewards are uniformly bad → no gradient signal → no RL learning.

### 1.2 Research Questions

1. **RQ1**: Do larger models generate more valid expressions?
2. **RQ2**: Do larger models produce more complex expressions (depth, nesting, power operations)?
3. **RQ3**: Do larger models achieve better R² scores on complex benchmarks?
4. **RQ4**: Do larger models generate more diverse expressions?
5. **RQ5**: What is the optimal model size for symbolic regression considering cost-benefit trade-offs?

### 1.3 Hypotheses

**H1** (Validity): Valid expression rate increases with model size
- Base: 80% → Medium: 82-85% → Large: 85-90%

**H2** (Complexity): Expression complexity increases with model size
- Power operations: Base 15.9% → Medium 35-45% → Large 50-65%
- Average depth: Base 1.40 → Medium 1.8-2.0 → Large 2.0-2.5
- Nested trig: Base 0% → Medium 5-10% → Large 10-20%

**H3** (Performance): Benchmark performance (R²) improves with model size
- Nguyen-5 best R²: Base -1.0 → Medium >-0.5 → Large >0.0

**H4** (Diversity): Expression diversity increases with model size
- Larger models explore broader expression space

**H5** (Algorithm Interaction): RL algorithms work better with larger models
- PPO and GRPO benefit more from increased capacity

---

## 2. Methodology

### 2.1 Models

| Model | Parameters | LoRA Trainable | Instance Type | Batch Size | Cost (est.) |
|-------|-----------|----------------|---------------|-----------|-------------|
| Base | 124M | 294K | g5.xlarge | 8 | $2-3 |
| Medium | 355M | 294K | g5.xlarge | 4 | $3-4 |
| Large | 774M | 294K | g5.2xlarge | 2 | $5-6 |

**Key Design Decision**: Fix all hyperparameters except batch size to isolate model size effect.

### 2.2 Training Configuration

**Dataset**: augustocsc/sintetico_natural (700K subset)

**Format**: JSON (EXP-A)
```json
{"vars": ["x_1"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "sin(x_1 + C)"}
```

**Hyperparameters** (fixed across all models):
- Learning rate: 5e-5
- Epochs: 3 (with early stopping, patience=3)
- Gradient accumulation: 4 steps
- Warmup steps: 500
- Weight decay: 0.01
- FP16: True
- Seed: 42
- LoRA: r=8, alpha=32, target_modules=["c_attn"], dropout=0.05

**Training Split**: 90% train / 10% validation (automatic)

**Infrastructure**: AWS g5.xlarge / g5.2xlarge with NVIDIA A10G GPUs

**Tracking**: Weights & Biases (project: seriguela)

### 2.3 Evaluation Metrics

#### 2.3.1 Quality Metrics

1. **Validity**:
   - Valid expression rate (%): Syntactically correct AND semantically evaluable
   - Parseable rate (%): Syntactically correct only

2. **Constraint Adherence**:
   - Uses allowed variables (%): Only uses vars specified in prompt
   - Uses allowed operators (%): Only uses ops specified in prompt
   - Constraint adherence (%): Both constraints satisfied

3. **Diversity**:
   - Diversity rate (%): Proportion of unique expressions
   - Unique expressions count: Absolute number of different expressions

#### 2.3.2 Complexity Metrics

1. **Power Operations**: Percentage using x², x**n
2. **Nested Trigonometric Functions**: Percentage with sin(cos(x)), etc.
3. **Expression Depth**: Average nesting level
4. **Operator Distribution**: Usage frequencies

#### 2.3.3 Benchmark Performance

**Nguyen Suite** (1-12): Standard symbolic regression benchmarks

**Algorithms**:
1. **Supervised**: Direct generation (no optimization)
2. **REINFORCE**: Policy gradient with EMA baseline
3. **GRPO**: Group Relative Policy Optimization
4. **PPO**: Proximal Policy Optimization

**Metrics**:
- Best R²: Highest R² achieved
- Mean R² (valid expressions): Average fit quality
- Convergence rate: Improvement over epochs
- Valid rate during RL: Maintains validity while optimizing

### 2.4 Experimental Design

**Phase 1**: Supervised Training
- Train all 3 models in parallel
- Monitor loss curves, early stopping
- Save checkpoints

**Phase 2**: Basic Evaluation
- Generate 500 expressions per model
- Compute quality and complexity metrics
- Compare models

**Phase 3**: Nguyen Suite Evaluation
- 3 models × 12 benchmarks × 4 algorithms = **144 experiments**
- 20 epochs, 100 samples per epoch (RL algorithms)
- 200 samples (supervised)

**Phase 4**: Analysis
- Aggregate results
- Statistical significance testing
- Visualization (heatmaps, bar charts)
- Cost-benefit analysis

---

## 3. Results

*To be filled after experiments complete*

### 3.1 Training Results

#### Table 1: Training Metrics

| Model | Final Train Loss | Best Val Loss | Early Stopped | Training Time | Cost |
|-------|------------------|---------------|---------------|---------------|------|
| Base | TBD | TBD | TBD | TBD | TBD |
| Medium | TBD | TBD | TBD | TBD | TBD |
| Large | TBD | TBD | TBD | TBD | TBD |

**Expected**: Lower loss for larger models.

**Actual**: TBD

### 3.2 Quality Metrics

#### Table 2: Supervised Generation Quality

| Metric | Base | Medium | Large | H1 Confirmed? |
|--------|------|--------|-------|---------------|
| Valid Expression Rate (%) | TBD | TBD | TBD | ⏳ |
| Parseable Rate (%) | TBD | TBD | TBD | - |
| Constraint Adherence (%) | TBD | TBD | TBD | - |
| Diversity Rate (%) | TBD | TBD | TBD | ⏳ |
| Unique Expressions | TBD | TBD | TBD | - |

### 3.3 Complexity Metrics

#### Table 3: Expression Complexity

| Metric | Base | Medium | Large | Improvement (B→L) | H2 Confirmed? |
|--------|------|--------|-------|-------------------|---------------|
| Power Operations (%) | TBD | TBD | TBD | TBD | ⏳ |
| Nested Trig (%) | TBD | TBD | TBD | TBD | ⏳ |
| Average Depth | TBD | TBD | TBD | TBD | ⏳ |
| Max Depth | TBD | TBD | TBD | TBD | - |

**Expected** (H2):
- Power ops: Base 15.9% → Large 50-65%
- Depth: Base 1.40 → Large 2.0-2.5
- Nested trig: Base 0% → Large 10-20%

### 3.4 Nguyen Benchmark Performance

#### Table 4: Average R² Across All 12 Benchmarks

| Algorithm | Base | Medium | Large | Best Model | H3 Confirmed? |
|-----------|------|--------|-------|-----------|---------------|
| Supervised | TBD | TBD | TBD | TBD | ⏳ |
| REINFORCE | TBD | TBD | TBD | TBD | ⏳ |
| GRPO | TBD | TBD | TBD | TBD | ⏳ |
| PPO | TBD | TBD | TBD | TBD | ⏳ |

#### Table 5: Nguyen-5 Specific (Complex Benchmark)

| Algorithm | Base | Medium | Large | Improvement |
|-----------|------|--------|-------|-------------|
| Supervised | TBD | TBD | TBD | TBD |
| REINFORCE | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD |
| PPO | TBD | TBD | TBD | TBD |

**Baseline** (from previous work): Base supervised on Nguyen-5 = R² -1.0

**Expected**: Significant improvement with larger models

---

## 4. Visualizations

*To be generated after evaluation completes*

### Figure 1: Model Comparison Overview
- 4 subplots: Valid Rate, R², Power Ops, Depth
- Bar charts comparing Base, Medium, Large

### Figure 2: Algorithm Performance Heatmaps
- One heatmap per algorithm
- Rows: Nguyen benchmarks (1-12)
- Columns: Model sizes
- Color: R² scores

### Figure 3: Complexity Progression
- Line chart showing how complexity metrics scale with model size

### Figure 4: Cost-Benefit Analysis
- Scatter plot: Cost (x-axis) vs Performance (y-axis)
- Shows diminishing returns

---

## 5. Statistical Analysis

*To be completed after results*

### 5.1 Hypothesis Tests

**H1 (Validity)**:
- Test: Chi-square test for valid rate differences
- Significance level: α = 0.05
- Result: TBD
- Conclusion: TBD

**H2 (Complexity)**:
- Test: Mann-Whitney U test for depth differences
- Significance level: α = 0.05
- Result: TBD
- Conclusion: TBD

**H3 (Performance)**:
- Test: Kruskal-Wallis test for R² differences
- Significance level: α = 0.05
- Result: TBD
- Conclusion: TBD

### 5.2 Effect Sizes

- Cohen's d for continuous metrics (depth, R²)
- Cramér's V for categorical metrics (valid rate)

**Results**: TBD

---

## 6. Discussion

*To be written after results*

### 6.1 Key Findings

1. **Finding 1**: TBD
2. **Finding 2**: TBD
3. **Finding 3**: TBD

### 6.2 Interpretation

**RQ1 (Validity)**: TBD

**RQ2 (Complexity)**: TBD

**RQ3 (Performance)**: TBD

**RQ4 (Diversity)**: TBD

**RQ5 (Optimal Size)**: TBD

### 6.3 Comparison with Hypotheses

| Hypothesis | Expected | Actual | Confirmed? |
|-----------|----------|--------|-----------|
| H1 (Validity increases) | 80% → 90% | TBD | ⏳ |
| H2 (Complexity increases) | 1.4 → 2.5 depth | TBD | ⏳ |
| H3 (R² improves) | -1.0 → >0.0 | TBD | ⏳ |
| H4 (Diversity increases) | Higher unique rate | TBD | ⏳ |
| H5 (RL benefits) | Better convergence | TBD | ⏳ |

### 6.4 Unexpected Results

*Document any surprising findings*

1. TBD
2. TBD

### 6.5 Limitations

1. **LoRA fixed parameters**: Using the same LoRA rank (r=8) for all model sizes may not be optimal
   - Larger models might benefit from higher ranks
   - Future: Scale LoRA rank with model size

2. **Single dataset**: Only tested on sintetico_natural 700K
   - Results may not generalize to other expression distributions
   - Future: Test on multiple datasets

3. **Nguyen benchmarks only**: Limited to 12 standard benchmarks
   - May not represent all real-world symbolic regression tasks
   - Future: Test on Feynman equations, real scientific datasets

4. **Batch size variation**: Different batch sizes across models (8→4→2)
   - Effective batch size same (×4 accumulation), but gradient noise differs
   - May affect convergence dynamics

5. **Early stopping**: May have prevented full convergence
   - Trade-off between cost and potential performance
   - Future: Test with longer training

6. **JSON format dependency**: Results specific to JSON-structured prompts
   - May not generalize to other formats
   - Future: Test with multiple prompt formats

### 6.6 Implications

**For Research**:
- TBD

**For Practitioners**:
- TBD

**For Model Selection**:
- When to use Base: TBD
- When to use Medium: TBD
- When to use Large: TBD

---

## 7. Conclusions

*To be written after results*

### 7.1 Summary

This experiment investigated the impact of model size (124M → 355M → 774M) on symbolic regression expression generation across three dimensions: validity, complexity, and benchmark performance.

**Main Result**: TBD

### 7.2 Recommendations

1. **Recommended model size**: TBD (based on cost-benefit)
2. **Best algorithm by model**: TBD
3. **Optimal hyperparameters**: TBD

### 7.3 Future Work

1. **LoRA scaling study**: Vary LoRA rank with model size
   - Test: Base (r=8), Medium (r=16), Large (r=32)
   - Hypothesis: Larger models need higher ranks for full capacity

2. **Dataset scaling**: Train on larger datasets (1M, 5M expressions)
   - Test if larger models benefit more from more data

3. **Architecture variants**: Test other model families
   - GPT-Neo, GPT-J, LLaMA
   - Encoder-decoder models (T5, BART)

4. **Multi-task learning**: Train on multiple benchmarks simultaneously
   - May improve generalization

5. **Interpretability study**: Analyze attention patterns
   - Understand what larger models learn differently

6. **Real-world deployment**: Test on actual scientific datasets
   - Feynman equations
   - Materials science expressions
   - Biological models

---

## 8. Reproducibility

### 8.1 Code and Data

**Repository**: https://github.com/augustocsc/seriguela

**Branch**: experiment/ppo-symbolic-regression

**Commit**: TBD (run `git rev-parse HEAD`)

**Models**: TBD (HuggingFace links)
- Base: https://huggingface.co/USER/gpt2_base_700K_json
- Medium: https://huggingface.co/USER/gpt2_medium_700K_json
- Large: https://huggingface.co/USER/gpt2_large_700K_json

**Dataset**: augustocsc/sintetico_natural (700K subset)

### 8.2 Reproduction Steps

```bash
# 1. Clone repository
git clone https://github.com/augustocsc/seriguela.git
cd seriguela
git checkout experiment/ppo-symbolic-regression

# 2. Install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Train models (requires AWS)
bash launch_all_models.sh

# 4. Download trained models
# (see TRAINING_LOG for specific instance IPs)

# 5. Evaluate
bash scripts/run_nguyen_suite.sh

# 6. Aggregate results
python scripts/aggregate_nguyen_results.py --input_dir nguyen_suite_results
```

### 8.3 Hardware Requirements

**Training**:
- 3× AWS instances (g5.xlarge + g5.2xlarge)
- Total VRAM: 96GB
- Training time: ~10 hours total (parallel)
- Cost: ~$10-13 USD

**Evaluation** (Nguyen suite):
- 1× GPU with 24GB+ VRAM
- Time: ~12-16 hours for full suite (144 experiments)
- Can run on CPU (slower: ~48-72 hours)

### 8.4 Software Versions

See `requirements.txt` for exact versions.

**Key dependencies**:
- Python 3.10+
- PyTorch 2.5.1 (CUDA 12.1)
- Transformers 4.51.3
- PEFT 0.15.1
- Wandb ≥0.24.1

---

## 9. Acknowledgments

*To be filled*

- Dataset: Augusto et al. (sintetico_natural)
- Benchmarks: Nguyen et al.
- Infrastructure: AWS
- Tracking: Weights & Biases

---

## 10. References

*To be filled*

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
2. Nguyen et al. Symbolic Regression Benchmarks
3. Schulman et al. (2017). Proximal Policy Optimization
4. DeepSeek-R1 paper (GRPO algorithm)
5. Previous work: EXPERIMENT_RESULTS.md

---

**Document Version**: 1.0

**Last Updated**: 2025-02-02

**Status**: ⏳ In Progress (Results pending)

**Contact**: [Your contact information]
