# 3_evaluation/ - Model Evaluation

Unified CLI for evaluating symbolic regression models.

## Two Evaluation Phases

This CLI supports two distinct evaluation phases:

1. **Quality Evaluation** (Generation Phase)
   - Measures: valid rate, diversity, constraint adherence
   - Used after supervised training to assess generation quality
   - Results in: `results/quality/`

2. **Benchmark Evaluation** (RL Phase)
   - Measures: R² scores on Nguyen benchmarks
   - Used during/after RL training to assess expression fit
   - Results in: `results/benchmark/`

## Quick Start

```bash
# Quality evaluation (generation phase)
python -m 3_evaluation.cli quality --model augustocsc/gpt2_large_infix_682k --num-samples 500

# Benchmark evaluation (RL phase)
python -m 3_evaluation.cli benchmark --model augustocsc/gpt2_large_infix_682k --benchmark nguyen_5

# List available benchmarks
python -m 3_evaluation.cli benchmarks

# List all runs
python -m 3_evaluation.cli list

# Compare runs
python -m 3_evaluation.cli compare --runs run_001 run_002

# Generate report
python -m 3_evaluation.cli report --run run_001 --format markdown
```

## Commands

### `quality` - Generation Quality Evaluation

Evaluates expression generation quality (valid rate, diversity, constraint adherence).

```bash
python -m 3_evaluation.cli quality \
  --model augustocsc/gpt2_large_infix_682k \
  --num-samples 500 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --max-tokens 100 \
  --vars x_1,x_2,x_3 \
  --ops "+,-,*,/,sin,cos" \
  --output-dir results/quality
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | HuggingFace model or local path |
| `--num-samples` | 500 | Number of samples to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Top-p (nucleus sampling) |
| `--top-k` | 50 | Top-k sampling |
| `--max-tokens` | 100 | Max tokens to generate |
| `--vars` | x_1 | Allowed variables (comma-separated) |
| `--ops` | +,-,*,/,sin,cos | Allowed operators |
| `--config` | - | YAML configuration file |
| `--output-dir` | results/quality | Output directory |

### `benchmark` - Nguyen Benchmark Evaluation

Evaluates model on symbolic regression benchmarks (R² scores).

```bash
python -m 3_evaluation.cli benchmark \
  --model augustocsc/gpt2_large_infix_682k \
  --benchmark nguyen_5 \
  --num-samples 100 \
  --temperature 0.7 \
  --output-dir results/benchmark
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | HuggingFace model or local path |
| `--benchmark` | nguyen_5 | Benchmark name (nguyen_1 to nguyen_12) |
| `--csv` | - | Custom benchmark CSV file |
| `--num-samples` | 100 | Number of candidate expressions |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Top-p sampling |
| `--output-dir` | results/benchmark | Output directory |

### `benchmarks` - List Available Benchmarks

```bash
python -m 3_evaluation.cli benchmarks
```

Shows all available Nguyen benchmarks (1-12) with their formulas.

### `list` - List Evaluation Runs

```bash
# List all runs (quality and benchmark)
python -m 3_evaluation.cli list

# List only quality runs
python -m 3_evaluation.cli list --type quality

# List only benchmark runs
python -m 3_evaluation.cli list --type benchmark
```

### `compare` - Compare Runs

```bash
python -m 3_evaluation.cli compare --runs run_001 run_002 --output comparison.md
```

### `report` - Generate Reports

```bash
# Markdown (default)
python -m 3_evaluation.cli report --run run_001

# HTML
python -m 3_evaluation.cli report --run run_001 --format html

# JSON
python -m 3_evaluation.cli report --run run_001 --format json
```

## Output Structure

```
results/
├── quality/                         # Generation phase results
│   └── run_20260220_143022_abc123/
│       ├── config.yaml              # Full configuration
│       ├── samples.jsonl            # Individual samples
│       ├── metrics.json             # Aggregated metrics
│       └── summary.txt              # Human-readable summary
│
└── benchmark/                       # RL phase results
    └── run_20260220_150000_def456/
        ├── config.yaml
        ├── samples.jsonl            # Each expression with R²
        └── metrics.json             # Best R², mean R², etc.
```

### Quality `metrics.json`

```json
{
  "valid_rate": 0.95,
  "parseable_rate": 0.97,
  "diversity_rate": 0.89,
  "total_samples": 500,
  "valid_count": 475,
  "unique_count": 445,
  "constraint_adherence_rate": 0.92,
  "avg_complexity": 7.3,
  "variable_usage": {"x_1": 450, "x_2": 380},
  "operator_usage": {"sin": 320, "+": 280}
}
```

### Benchmark `metrics.json`

```json
{
  "benchmark_name": "nguyen_5",
  "true_formula": "sin(x**2)*cos(x) - 1",
  "num_samples": 100,
  "valid_count": 85,
  "valid_rate": 0.85,
  "num_with_r2": 80,
  "best_r2": 0.9987,
  "mean_r2": 0.7234,
  "median_r2": 0.8012,
  "std_r2": 0.2156,
  "best_expression": "sin(x_1**2)*cos(x_1) - C",
  "best_constants": [0.998]
}
```

## Nguyen Benchmarks

| Benchmark | Formula | Variables | Range |
|-----------|---------|-----------|-------|
| nguyen_1 | x³ + x² + x | x_1 | [-1, 1] |
| nguyen_2 | x⁴ + x³ + x² + x | x_1 | [-1, 1] |
| nguyen_3 | x⁵ + x⁴ + x³ + x² + x | x_1 | [-1, 1] |
| nguyen_4 | x⁶ + x⁵ + x⁴ + x³ + x² + x | x_1 | [-1, 1] |
| nguyen_5 | sin(x²)cos(x) - 1 | x_1 | [-1, 1] |
| nguyen_6 | sin(x) + sin(x + x²) | x_1 | [-1, 1] |
| nguyen_7 | log(x+1) + log(x²+1) | x_1 | [0, 2] |
| nguyen_8 | √x | x_1 | [0, 4] |
| nguyen_9 | sin(x) + sin(y²) | x_1, x_2 | [-1, 1] |
| nguyen_10 | 2sin(x)cos(y) | x_1, x_2 | [-1, 1] |
| nguyen_11 | xʸ | x_1, x_2 | [0, 1] |
| nguyen_12 | x⁴ - x³ + y²/2 - y | x_1, x_2 | [-1, 1] |

## Code Structure

```
3_evaluation/
├── cli.py                  # CLI entry point
├── core/                   # Common library
│   ├── model_loader.py     # LoRA model loading
│   ├── generator.py        # Expression generation
│   ├── extractor.py        # Expression extraction from output
│   ├── validator.py        # SymPy validation
│   ├── metrics.py          # Metrics calculation
│   └── storage.py          # Result persistence
├── commands/               # Subcommands
│   ├── quality.py          # evaluate quality
│   ├── benchmark.py        # evaluate benchmark
│   ├── compare.py          # evaluate compare
│   └── report.py           # evaluate report
└── benchmarks/             # Additional benchmark utilities
```

## Available Models

| Model | Notation | Params | HuggingFace |
|-------|----------|--------|-------------|
| Base | Infix | 124M | `augustocsc/gpt2_base_infix_682k` |
| Base | Prefix | 124M | `augustocsc/gpt2_base_prefix_682k` |
| Medium | Infix | 355M | `augustocsc/gpt2_medium_infix_682k` |
| Medium | Prefix | 355M | `augustocsc/gpt2_medium_prefix_682k` |
| Large | Infix | 774M | `augustocsc/gpt2_large_infix_682k` |
| Large | Prefix | 774M | `augustocsc/gpt2_large_prefix_682k` |

## YAML Configuration

For reproducible experiments, use a configuration file:

```yaml
# config.yaml
model:
  path: augustocsc/gpt2_large_infix_682k

generation:
  temperature: 0.8
  top_p: 0.95
  top_k: 50
  max_new_tokens: 100

prompt:
  vars: [x_1, x_2, x_3]
  ops: ['+', '-', '*', '/', sin, cos]
  cons: C
  format: infix

evaluation:
  num_samples: 1000
  seed: 42
```

```bash
python -m 3_evaluation.cli quality --config config.yaml
```

## Examples

### Evaluate All Models (Quality)

```bash
for model in base medium large; do
  for notation in infix prefix; do
    python -m 3_evaluation.cli quality \
      --model augustocsc/gpt2_${model}_${notation}_682k \
      --num-samples 500
  done
done
```

### Run All Nguyen Benchmarks

```bash
for i in {1..12}; do
  python -m 3_evaluation.cli benchmark \
    --model augustocsc/gpt2_large_infix_682k \
    --benchmark nguyen_$i \
    --num-samples 100
done
```

### Compare Infix vs Prefix

```bash
# Run quality evaluations
python -m 3_evaluation.cli quality --model augustocsc/gpt2_large_infix_682k --num-samples 500
python -m 3_evaluation.cli quality --model augustocsc/gpt2_large_prefix_682k --num-samples 500

# List to get run IDs
python -m 3_evaluation.cli list --type quality

# Compare
python -m 3_evaluation.cli compare --runs run_xxx run_yyy --output comparison.md
```
