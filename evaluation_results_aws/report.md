# Comprehensive Symbolic Regression Evaluation Report

**Date**: 20260211_143640
**Total Experiments**: 96
**Successful**: 72
**Failed**: 24

## Best Results per Benchmark

| Benchmark | Best R² | Model | Algorithm | Expression | Epoch |
|-----------|---------|-------|-----------|------------|-------|
| nguyen_1 | 0.9709 | base_prefix | grpo | * * -1 C log - * C x_1 C | 4 |
| nguyen_10 | 0.9332 | large_prefix | grpo | tan * C x_1 | 13 |
| nguyen_11 | 0.6773 | large_prefix | grpo | * x_1 exp tan - x_1 C | 11 |
| nguyen_12 | 0.8329 | base_prefix | ppo | - C sin * C exp * -1 x_2 | 18 |
| nguyen_2 | 0.6837 | base_prefix | ppo | - C / C exp * C x_1 | 6 |
| nguyen_3 | 0.9004 | large_prefix | ppo | * x_1 + exp x_1 C | 14 |
| nguyen_4 | 0.7930 | large_prefix | ppo | * x_1 + exp * C x_1 C | 18 |
| nguyen_5 | -0.4994 | medium_prefix | ppo | * -1 cos cos + * C x_1 C | 2 |
| nguyen_6 | 0.8749 | base_prefix | ppo | - C exp * C x_1 | 15 |
| nguyen_7 | 0.8625 | base_prefix | ppo | * C exp + x_1 * -1 C | 4 |
| nguyen_8 | 0.8109 | base_prefix | ppo | * C ** cos x_1 0.5 | 11 |
| nguyen_9 | 0.7277 | base_prefix | grpo | * C cos - x_1 C | 5 |

## Model Comparison


### base_prefix

| Benchmark | Best R² | Algorithm | Expression | Epoch |
|-----------|---------|-----------|------------|-------|
| nguyen_1 | 0.9709 | grpo | * * -1 C log - * C x_1 C | 4 |
| nguyen_10 | 0.9147 | grpo | + * C x_1 sin * C x_1 | 0 |
| nguyen_11 | 0.5277 | grpo | * C ** - * C x_1 C 0.5 | 16 |
| nguyen_12 | 0.8329 | ppo | - C sin * C exp * -1 x_2 | 18 |
| nguyen_2 | 0.6837 | ppo | - C / C exp * C x_1 | 6 |
| nguyen_3 | 0.8536 | ppo | * x_1 exp + x_1 sin x_1 | 0 |
| nguyen_4 | 0.5318 | grpo | - C exp x_1 | 2 |
| nguyen_5 | -1.0000 | ppo | - tan exp * C x_1 C | 0 |
| nguyen_6 | 0.8749 | ppo | - C exp * C x_1 | 15 |
| nguyen_7 | 0.8625 | ppo | * C exp + x_1 * -1 C | 4 |
| nguyen_8 | 0.8109 | ppo | * C ** cos x_1 0.5 | 11 |
| nguyen_9 | 0.7277 | grpo | * C cos - x_1 C | 5 |

### large_prefix

| Benchmark | Best R² | Algorithm | Expression | Epoch |
|-----------|---------|-----------|------------|-------|
| nguyen_1 | 0.6275 | ppo | sin * C x_1 | 14 |
| nguyen_10 | 0.9332 | grpo | tan * C x_1 | 13 |
| nguyen_11 | 0.6773 | grpo | * x_1 exp tan - x_1 C | 11 |
| nguyen_12 | -1.0000 | ppo | * x_1 + + x_1 x_2 C | 0 |
| nguyen_2 | 0.4110 | ppo | * C exp x_1 | 16 |
| nguyen_3 | 0.9004 | ppo | * x_1 + exp x_1 C | 14 |
| nguyen_4 | 0.7930 | ppo | * x_1 + exp * C x_1 C | 18 |
| nguyen_5 | -1.0000 | ppo | sin + x_1 C | 1 |
| nguyen_6 | 0.8380 | ppo | tan * C x_1 | 11 |
| nguyen_7 | 0.8236 | ppo | + x_1 sin x_1 | 10 |
| nguyen_8 | 0.5688 | grpo | log + exp log x_1 C | 1 |
| nguyen_9 | 0.7277 | grpo | cos - x_1 C | 10 |

### medium_prefix

| Benchmark | Best R² | Algorithm | Expression | Epoch |
|-----------|---------|-----------|------------|-------|
| nguyen_1 | -0.3676 | grpo | cos * log + sin cos x_1 C C | 11 |
| nguyen_10 | 0.6412 | grpo | * + x_2 C tan x_1 | 5 |
| nguyen_11 | -0.1281 | ppo | * x_1 + cos sin + x_2 C C | 13 |
| nguyen_12 | -1.0000 | ppo | tan / tan log x_2 C | 2 |
| nguyen_2 | -0.6687 | ppo | sin / - x_1 C * -1 C | 3 |
| nguyen_3 | 0.3157 | ppo | * C sin sin / x_1 C | 0 |
| nguyen_4 | 0.2227 | ppo | sin / x_1 cos cos + x_1 C | 16 |
| nguyen_5 | -0.4994 | ppo | * -1 cos cos + * C x_1 C | 2 |
| nguyen_6 | 0.0035 | grpo | * x_1 + log + x_1 C C | 17 |
| nguyen_7 | -0.0099 | grpo | * x_1 exp sin - x_1 C | 10 |
| nguyen_8 | -0.0217 | grpo | sqrt * C exp cos sin + x_1 C | 16 |
| nguyen_9 | 0.2517 | ppo | * x_1 - x_2 C | 17 |