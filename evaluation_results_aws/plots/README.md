# Evolution Plots - R² Scores by Epoch

This directory contains scatter plots showing the evolution of R² scores across training epochs for each experiment.

## Directory Structure

```
evolution_by_epoch/
├── base_prefix_nguyen_1_ppo.png
├── base_prefix_nguyen_1_grpo.png
├── ... (72 plots total)
```

## Plot Format

Each plot shows:
- **X-axis**: Epoch (0-19)
- **Y-axis**: R² Score
- **Blue dots**: Valid expressions ✅
- **Red dots**: Invalid expressions ❌
- **Gold star**: Best expression found ⭐
- **Black dashed line**: R² = 0 baseline

## Naming Convention

Files are named as: `{model}_{benchmark}_{algorithm}.png`

Examples:
- `base_prefix_nguyen_1_grpo.png` - Base model on Nguyen-1 with GRPO
- `large_prefix_nguyen_10_ppo.png` - Large model on Nguyen-10 with PPO
- `medium_prefix_nguyen_5_grpo.png` - Medium model on Nguyen-5 with GRPO

## Statistics Box

Each plot includes a statistics box (top-left) showing:
- **Total**: Total expressions generated (usually 640 = 20 epochs × 32 samples)
- **Valid**: Number and percentage of valid expressions
- **Best R²**: Highest R² score achieved
- **Avg R² (valid)**: Average R² considering only valid expressions

## Key Observations

### Success Cases (e.g., base_prefix + nguyen_1 + GRPO)
- Blue dots scattered throughout
- Best R² > 0.5
- Clear improvement over epochs
- Gold star at high R² value

### Mode Collapse Cases (e.g., medium_prefix + nguyen_1 + PPO)
- Mostly red dots (invalid expressions)
- Valid rate < 1%
- Best R² is negative
- Little to no improvement over epochs

### Difficult Benchmarks (e.g., any model + nguyen_5)
- All models struggle
- Most expressions have R² near -1.0
- Very few valid expressions
- No clear best expression

## Usage in Paper

These plots can be used to:
1. Illustrate RL training dynamics
2. Show mode collapse in medium_prefix model
3. Compare PPO vs GRPO behavior
4. Demonstrate benchmark difficulty

## Generation Script

Plots were generated using: `scripts/plot_evolution_by_epoch.py`

```bash
python scripts/plot_evolution_by_epoch.py
```

## Total Files

- **72 plots** (4.7 MB total)
- 3 models × 12 benchmarks × 2 algorithms = 72 combinations
- Resolution: 1800×900 pixels (150 DPI)
