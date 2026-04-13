# Experimental Proposal: Systematic Evaluation of Fine-tuned Language Models on Symbolic Regression Benchmarks

**Project**: Seriguela - LLM-based Symbolic Regression
**Author**: Augusto Cesar
**Date**: February 2026
**Status**: Proposed

---

## 1. Executive Summary

This proposal outlines a systematic evaluation of six fine-tuned GPT-2 models on the Nguyen symbolic regression benchmarks. Building upon the comprehensive generation quality evaluation (72 experiments, 7,200 expressions), we now aim to assess the models' ability to solve actual symbolic regression problems—finding mathematical expressions that fit given data.

The key research questions are:
1. Can language models fine-tuned for expression generation solve standard symbolic regression benchmarks?
2. What sampling configuration (temperature, number of samples) maximizes success rate?
3. Does infix notation maintain its advantage over prefix in the symbolic regression task?
4. How does model size affect benchmark performance?

---

## 2. Background and Motivation

### 2.1 Previous Work

The comprehensive evaluation (Phase 1) established:

| Finding | Value |
|---------|-------|
| Infix validity rate | 99.9% |
| Prefix validity rate | 94.6% |
| Infix diversity rate | 78.8% |
| Prefix diversity rate | 62.9% |
| Best model (validity) | Large Infix (100%) |
| Best model (diversity) | Medium Infix (83.5%) |
| Optimal temperature range | 0.7–0.9 |

**Limitation**: These metrics evaluate *generation quality* (syntactic validity, diversity) but not *regression quality* (fitting target data).

### 2.2 Research Gap

The fundamental question remains unanswered: **Can these models find expressions that actually fit data?**

Generating valid expressions is necessary but not sufficient. A model might generate 1,000 valid expressions, none of which approximate the target function. Conversely, a model with lower diversity might repeatedly generate expressions close to the target.

### 2.3 Nguyen Benchmarks

The Nguyen benchmark suite is the standard evaluation framework for symbolic regression, consisting of 12 problems:

| ID | Target Expression | Variables | Complexity |
|----|-------------------|-----------|------------|
| Nguyen-1 | x³ + x² + x | 1 | Low |
| Nguyen-2 | x⁴ + x³ + x² + x | 1 | Low |
| Nguyen-3 | x⁵ + x⁴ + x³ + x² + x | 1 | Medium |
| Nguyen-4 | x⁶ + x⁵ + x⁴ + x³ + x² + x | 1 | Medium |
| Nguyen-5 | sin(x²)cos(x) − 1 | 1 | High |
| Nguyen-6 | sin(x) + sin(x + x²) | 1 | High |
| Nguyen-7 | log(x + 1) + log(x² + 1) | 1 | High |
| Nguyen-8 | √x | 1 | Low |
| Nguyen-9 | sin(x) + sin(y²) | 2 | Medium |
| Nguyen-10 | 2sin(x)cos(y) | 2 | Medium |
| Nguyen-11 | xʸ | 2 | Medium |
| Nguyen-12 | x⁴ − x³ + ½x² − x | 1 | Medium |

Each benchmark provides:
- Training data: 20 (x, y) pairs sampled uniformly
- Test data: 20 different (x, y) pairs for validation
- Target expression: Ground truth for exact match evaluation

---

## 3. Research Questions

### Primary Questions

**RQ1**: What is the success rate of each model on Nguyen benchmarks?
- Metric: Percentage of benchmarks solved (R² > 0.99)

**RQ2**: How does sampling configuration affect performance?
- Variables: Temperature, number of samples, top-p, top-k
- Metric: R² achieved, samples required for success

**RQ3**: Does the infix advantage persist in symbolic regression?
- Comparison: Infix vs Prefix models on identical benchmarks
- Metric: Success rate, average R², exact match rate

### Secondary Questions

**RQ4**: What is the relationship between generation diversity and regression success?
- Analysis: Correlation between diversity rate and benchmark success

**RQ5**: Which benchmarks are easy/hard for language models?
- Analysis: Success rate by benchmark, error analysis for failures

**RQ6**: Is there a scaling law for sample count vs success probability?
- Analysis: Power law fit for P(success) vs N_samples

---

## 4. Experimental Design

### 4.1 Independent Variables

#### Models (6 levels)
| Model | Notation | Parameters | HuggingFace Repository |
|-------|----------|------------|------------------------|
| Base Infix | Infix | 124M | augustocsc/gpt2_base_infix_682k |
| Medium Infix | Infix | 355M | augustocsc/gpt2_medium_infix_682k |
| Large Infix | Infix | 774M | augustocsc/gpt2_large_infix_682k |
| Base Prefix | Prefix | 124M | augustocsc/gpt2_base_prefix_682k |
| Medium Prefix | Prefix | 355M | augustocsc/gpt2_medium_prefix_682k |
| Large Prefix | Prefix | 774M | augustocsc/gpt2_large_prefix_682k |

#### Temperature (4 levels)
- 0.3 (conservative, low diversity)
- 0.5 (moderate)
- 0.7 (balanced - baseline from Phase 1)
- 0.9 (explorative, high diversity)

#### Number of Samples (4 levels)
- 50 (quick evaluation)
- 100 (standard)
- 500 (thorough)
- 1000 (exhaustive)

#### Prompt Configuration (2 levels)

**Restricted operators** (benchmark-specific):
```json
{"vars": ["x_1"], "ops": ["+", "-", "*", "sin", "cos"], "cons": "C", "expr": "
```

**Expanded operators** (all available):
```json
{"vars": ["x_1"], "ops": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"], "cons": "C", "expr": "
```

### 4.2 Dependent Variables

| Metric | Description | Type |
|--------|-------------|------|
| **R² (best)** | Highest R² achieved among all samples | Continuous [0, 1] |
| **R² (mean top-10)** | Average R² of 10 best expressions | Continuous [0, 1] |
| **Success rate** | Proportion with R² > 0.99 | Binary |
| **Exact match** | Found the exact target expression | Binary |
| **Samples to success** | Number of samples until R² > 0.99 | Count |
| **Valid rate** | Proportion of syntactically valid expressions | Continuous [0, 1] |
| **Unique rate** | Proportion of unique expressions | Continuous [0, 1] |

### 4.3 Experimental Matrix

#### Full Factorial Design
```
Models:        6
Benchmarks:   12
Temperatures:  4
Sample sizes:  4
Prompt types:  2
─────────────────
Total:      2,304 experimental conditions
```

#### Reduced Design (Recommended)

To manage computational costs, we propose a two-phase approach:

**Phase 2A: Screening (288 experiments)**
```
Models:        6 (all)
Benchmarks:   12 (all)
Temperatures:  2 (0.5, 0.9 - extremes)
Sample sizes:  1 (100 - standard)
Prompt types:  2 (restricted, expanded)
─────────────────
Total:        288 experiments
Samples:   28,800 expressions
```

**Phase 2B: Deep Dive (144 experiments)**
Based on Phase 2A results, select:
```
Models:        2 (best infix, best prefix)
Benchmarks:   12 (all)
Temperatures:  3 (0.5, 0.7, 0.9)
Sample sizes:  2 (100, 500)
Prompt types:  1 (best from 2A)
─────────────────
Total:        144 experiments
Samples:   43,200 expressions
```

---

## 5. Methodology

### 5.1 Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    For each experiment:                      │
├─────────────────────────────────────────────────────────────┤
│  1. Load model and tokenizer                                │
│  2. Load Nguyen benchmark data (X, y_true)                  │
│  3. Construct prompt with allowed variables/operators        │
│  4. Generate N candidate expressions                         │
│  5. For each expression:                                     │
│     a. Validate syntax                                       │
│     b. If valid, evaluate on X → y_pred                     │
│     c. Compute R² = 1 - SS_res/SS_tot                       │
│  6. Record metrics (best R², success, exact match)          │
│  7. Save results to storage                                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Expression Evaluation

For each candidate expression:

1. **Parse**: Convert string to evaluable function
2. **Substitute**: Replace variables with data points
3. **Compute**: Calculate predicted values
4. **Score**: Compute R² coefficient

```python
def evaluate_expression(expr_str, X, y_true):
    """
    Evaluate expression fitness on benchmark data.

    Args:
        expr_str: Mathematical expression as string
        X: Input data array (n_samples, n_vars)
        y_true: Target values (n_samples,)

    Returns:
        r2: R² score, or -inf if evaluation fails
    """
    try:
        expr = parse_expression(expr_str)
        y_pred = expr.evaluate(X)

        # Handle numerical issues
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('-inf')

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return r2
    except Exception:
        return float('-inf')
```

### 5.3 Constant Optimization

Expressions containing the constant `C` require optimization:

```python
def optimize_constant(expr_str, X, y_true):
    """
    Find optimal value for constant C.

    Uses scipy.optimize.minimize to find C that maximizes R².
    """
    def objective(c_value):
        expr_with_c = expr_str.replace('C', str(c_value[0]))
        r2 = evaluate_expression(expr_with_c, X, y_true)
        return -r2  # Minimize negative R²

    result = minimize(objective, x0=[1.0], method='Nelder-Mead')
    return result.x[0], -result.fun
```

### 5.4 Exact Match Detection

Beyond R², we check for symbolic equivalence:

```python
def check_exact_match(expr_str, target_expr):
    """
    Check if expression is symbolically equivalent to target.

    Uses SymPy simplification to handle algebraic equivalence.
    """
    try:
        expr = sympy.sympify(expr_str)
        target = sympy.sympify(target_expr)
        diff = sympy.simplify(expr - target)
        return diff == 0
    except Exception:
        return False
```

---

## 6. Analysis Plan

### 6.1 Primary Analyses

#### Success Rate by Model
```
┌────────────────┬───────────┬──────────┬─────────────┐
│ Model          │ Successes │ Total    │ Success %   │
├────────────────┼───────────┼──────────┼─────────────┤
│ Large Infix    │ ?         │ 12       │ ?           │
│ Medium Infix   │ ?         │ 12       │ ?           │
│ Base Infix     │ ?         │ 12       │ ?           │
│ Large Prefix   │ ?         │ 12       │ ?           │
│ Medium Prefix  │ ?         │ 12       │ ?           │
│ Base Prefix    │ ?         │ 12       │ ?           │
└────────────────┴───────────┴──────────┴─────────────┘
```

#### Benchmark Difficulty Analysis
```
┌────────────┬─────────────┬───────────┬─────────────┐
│ Benchmark  │ Best R²     │ Success % │ Difficulty  │
├────────────┼─────────────┼───────────┼─────────────┤
│ Nguyen-1   │ ?           │ ?         │ ?           │
│ Nguyen-2   │ ?           │ ?         │ ?           │
│ ...        │ ...         │ ...       │ ...         │
│ Nguyen-12  │ ?           │ ?         │ ?           │
└────────────┴─────────────┴───────────┴─────────────┘
```

### 6.2 Statistical Tests

| Comparison | Test | Hypothesis |
|------------|------|------------|
| Infix vs Prefix | Paired t-test / Wilcoxon | H₁: Infix > Prefix |
| Temperature effect | ANOVA / Kruskal-Wallis | H₁: Temperature affects R² |
| Sample size effect | Regression analysis | H₁: More samples → higher success |
| Model size effect | Trend test | H₁: Larger models perform better |

### 6.3 Visualizations

1. **Heatmap**: Success rate by model × benchmark
2. **Line plot**: R² vs number of samples (scaling curve)
3. **Box plot**: R² distribution by temperature
4. **Bar chart**: Success rate by notation (infix vs prefix)
5. **Scatter plot**: Diversity rate vs benchmark success (correlation)

### 6.4 Error Analysis

For failed benchmarks (R² < 0.99), analyze:
- Common failure modes (wrong operators, missing terms)
- Expression complexity vs target complexity
- Partial matches (correct structure, wrong constants)

---

## 7. Expected Outcomes

### 7.1 Hypotheses

| ID | Hypothesis | Rationale |
|----|------------|-----------|
| H1 | Large Infix achieves highest success rate | Best validity (100%) and good diversity |
| H2 | Temperature 0.7-0.9 optimal for SR | Balance between exploration and validity |
| H3 | 500+ samples needed for hard benchmarks | Complex targets require more exploration |
| H4 | Nguyen 5-7 hardest for LLMs | Require precise transcendental functions |
| H5 | Infix outperforms Prefix in SR | +5pp validity advantage translates to SR |

### 7.2 Potential Findings

**Optimistic scenario**:
- 80%+ success rate on Nguyen 1-4, 8, 12 (polynomial)
- 50%+ success rate on Nguyen 5-7, 9-11 (transcendental)
- Clear optimal configuration identified

**Conservative scenario**:
- 60%+ success on easy benchmarks
- Models struggle with transcendental functions
- Need for RL fine-tuning confirmed

**Pessimistic scenario**:
- <50% success rate overall
- Generation quality ≠ regression quality
- Fundamental limitations identified

---

## 8. Resource Requirements

### 8.1 Computational Resources

| Phase | Experiments | Samples | Est. GPU Hours | Instance |
|-------|-------------|---------|----------------|----------|
| 2A | 288 | 28,800 | ~8h | g5.xlarge |
| 2B | 144 | 43,200 | ~12h | g5.xlarge |
| **Total** | **432** | **72,000** | **~20h** | - |

### 8.2 AWS Cost Estimate

```
Instance: g5.xlarge @ $1.01/hour
GPU hours: 20h
Buffer: 1.5x
─────────────────────────────
Estimated cost: ~$30 USD
```

### 8.3 Storage Requirements

```
Results per experiment: ~50 KB (metrics + samples)
Total experiments: 432
Raw results: ~22 MB
Processed data: ~5 MB
─────────────────────────────
Total storage: ~30 MB
```

---

## 9. Timeline

| Week | Phase | Activities |
|------|-------|------------|
| 1 | Implementation | Develop evaluation pipeline, test on 1 benchmark |
| 1 | Phase 2A | Run screening experiments (288) |
| 2 | Analysis 2A | Analyze results, select configurations for 2B |
| 2 | Phase 2B | Run deep-dive experiments (144) |
| 3 | Analysis | Statistical analysis, visualizations |
| 3 | Writing | Document results, update dissertation |

---

## 10. Deliverables

### 10.1 Code

- `3_evaluation/benchmarks/run_nguyen_evaluation.py` - Main evaluation script
- `3_evaluation/benchmarks/analyze_nguyen_results.py` - Analysis script
- `3_evaluation/core/expression_evaluator.py` - Expression evaluation module

### 10.2 Data

- `results/nguyen/phase_2a/` - Screening results
- `results/nguyen/phase_2b/` - Deep-dive results
- `results/nguyen/summary.json` - Aggregated metrics

### 10.3 Documentation

- LaTeX tables for dissertation
- Visualization figures (PNG/PDF)
- Statistical analysis report

### 10.4 HuggingFace

- Upload results to model repositories
- Update model cards with benchmark performance

---

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Models fail on all benchmarks | Low | High | Start with easy benchmarks, validate pipeline |
| Computational timeout | Medium | Medium | Use checkpointing, resume capability |
| Numerical instability | Medium | Low | Robust error handling, edge case testing |
| AWS instance issues | Low | Medium | Use spot instances with fallback |

---

## 12. Success Criteria

The experiment will be considered successful if:

1. **Completeness**: All 432 experiments complete without critical errors
2. **Actionable insights**: Clear recommendations for model/configuration selection
3. **Statistical rigor**: Results support or refute hypotheses with p < 0.05
4. **Reproducibility**: All code and data available for replication

---

## 13. Next Steps After This Experiment

Based on results, potential follow-up work:

1. **If success rate > 70%**: Focus on hard benchmarks, analyze failure modes
2. **If success rate 30-70%**: Apply RL fine-tuning (PPO/GRPO) to improve
3. **If success rate < 30%**: Investigate fundamental approach, consider architecture changes

---

## Appendix A: Nguyen Benchmark Data Generation

```python
import numpy as np

NGUYEN_BENCHMARKS = {
    "nguyen_1": {
        "expr": "x**3 + x**2 + x",
        "vars": ["x"],
        "x_range": (-1, 1),
        "n_train": 20,
        "n_test": 20,
    },
    "nguyen_2": {
        "expr": "x**4 + x**3 + x**2 + x",
        "vars": ["x"],
        "x_range": (-1, 1),
        "n_train": 20,
        "n_test": 20,
    },
    # ... (complete definitions in implementation)
}

def generate_benchmark_data(benchmark_id):
    """Generate training and test data for a Nguyen benchmark."""
    config = NGUYEN_BENCHMARKS[benchmark_id]

    # Generate random x values
    x_train = np.random.uniform(*config["x_range"], config["n_train"])
    x_test = np.random.uniform(*config["x_range"], config["n_test"])

    # Evaluate target expression
    y_train = eval(config["expr"], {"x": x_train, "np": np})
    y_test = eval(config["expr"], {"x": x_test, "np": np})

    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
        "target": config["expr"],
    }
```

---

## Appendix B: Related Work Comparison

| Method | Type | Nguyen Success Rate | Reference |
|--------|------|---------------------|-----------|
| PySR | Genetic Programming | ~85% | Cranmer 2023 |
| gplearn | Genetic Programming | ~70% | Stephens 2016 |
| DSR | Deep RL | ~75% | Petersen 2021 |
| NeSymReS | Transformer | ~80% | Biggio 2021 |
| **Ours** | Fine-tuned LLM | **TBD** | This work |

---

## Appendix C: Configuration Files

### Experiment Configuration Template

```yaml
# config/nguyen_experiment.yaml
experiment:
  name: nguyen_evaluation_phase_2a
  type: benchmark_evaluation

models:
  - augustocsc/gpt2_base_infix_682k
  - augustocsc/gpt2_medium_infix_682k
  - augustocsc/gpt2_large_infix_682k
  - augustocsc/gpt2_base_prefix_682k
  - augustocsc/gpt2_medium_prefix_682k
  - augustocsc/gpt2_large_prefix_682k

benchmarks:
  - nguyen_1
  - nguyen_2
  - nguyen_3
  - nguyen_4
  - nguyen_5
  - nguyen_6
  - nguyen_7
  - nguyen_8
  - nguyen_9
  - nguyen_10
  - nguyen_11
  - nguyen_12

generation:
  temperatures: [0.5, 0.9]
  num_samples: 100
  top_p: 0.9
  top_k: 50
  max_new_tokens: 100

evaluation:
  r2_threshold: 0.99
  optimize_constants: true
  check_exact_match: true

output:
  dir: results/nguyen/phase_2a
  upload_to_hf: true
```

---

*End of Proposal*
