---
language: en
license: mit
tags:
- symbolic-regression
- gpt2-large
- lora
- expression-generation
- mathematics
datasets:
- augustocsc/sintetico_natural
metrics:
- accuracy
- r2_score
model-index:
- name: GPT-2 Large for Symbolic Regression
  results:
  - task:
      type: symbolic-regression
    dataset:
      name: Sintetico Natural 700K
      type: augustocsc/sintetico_natural
    metrics:
    - name: Valid Expression Rate
      type: accuracy
      value: TBD
    - name: Best R² (Nguyen-5)
      type: r2_score
      value: TBD
---

# GPT-2 Large for Symbolic Regression (JSON Format)

## Model Description

This model is a GPT-2 Large variant (774M parameters) fine-tuned using LoRA for symbolic regression expression generation. It was trained on 700K synthetic mathematical expressions in JSON format.

**Part of research study**: "Impact of Model Size on Symbolic Regression Capability" (Feb 2025)

**Hypothesis**: Large model (774M) should demonstrate the highest capacity for complex compositional patterns, including:
- Highest rates of power operations (x², x**n)
- Greatest expression depth (deeply nested functions)
- Emergence of nested trigonometric functions (sin(cos(x²)), etc.)
- Best performance on complex benchmarks (Nguyen-5, 7, 10+)

## Model Details

- **Base Model**: gpt2-large (774M parameters)
- **Trainable Parameters**: ~294K (LoRA adapters only)
- **Training Data**: 700K expressions from augustocsc/sintetico_natural (700K subset)
- **Format**: JSON (EXP-A format) - achieves 80% valid expression rate vs 0.5% with EOS token approach
- **LoRA Configuration**:
  - r=8 (rank)
  - alpha=32 (scaling)
  - target_modules=["c_attn"] (attention layers only)
  - dropout=0.05
- **Training**: 3 epochs with early stopping (patience=3)
- **Hyperparameters**:
  - Learning rate: 5e-5
  - Batch size: 2 per device (reduced due to large model size)
  - Gradient accumulation: 4 steps
  - Warmup steps: 500
  - Weight decay: 0.01
  - FP16: Yes
  - Seed: 42

## Performance

### Supervised Fine-Tuning (Generation Quality)

Results will be filled after evaluation completes.

| Metric | Value | vs Base (124M) | vs Medium (355M) |
|--------|-------|----------------|------------------|
| Valid Expression Rate | TBD | TBD | TBD |
| Parseable Rate | TBD | TBD | TBD |
| Constraint Adherence | TBD | TBD | TBD |
| Diversity Rate | TBD | TBD | TBD |
| Power Operations Usage | TBD | TBD | TBD |
| Nested Trig Functions | TBD | TBD | TBD |
| Average Expression Depth | TBD | TBD | TBD |

**Expected improvements over Base:**
- +40-50% power operations usage
- Expression depth: 2.0-2.5 (vs 1.4 for Base)
- Nested trig >0% (vs 0% for Base)
- Highest valid expression rate and diversity

### Nguyen Benchmarks (with RL optimization)

Performance across Nguyen 1-12 benchmarks with different RL algorithms. Results will be filled after full suite evaluation.

| Benchmark | Best R² (REINFORCE) | Best R² (GRPO) | Best R² (PPO) |
|-----------|---------------------|----------------|---------------|
| Nguyen-1 | TBD | TBD | TBD |
| Nguyen-2 | TBD | TBD | TBD |
| Nguyen-3 | TBD | TBD | TBD |
| Nguyen-4 | TBD | TBD | TBD |
| Nguyen-5 | TBD | TBD | TBD |
| ... | ... | ... | ... |

**Expected:** Best performance on complex benchmarks (Nguyen-5, 7, 10+) that require nested operations and power functions.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model (774M parameters)
base_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/gpt2_large_700K_json")

# Generate expression
prompt = '{"vars": ["x_1"], "ops": ["*", "+", "sin", "cos", "**"], "cons": "C", "expr": "'
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
expression = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract expression from JSON
import json
result = json.loads(expression)
print(f"Generated expression: {result['expr']}")
```

## Training Details

- **Dataset Split**: 90% train / 10% validation (automatic split)
- **Early Stopping**: Enabled with patience=3 epochs
- **FP16 Precision**: Yes (for efficiency)
- **GPU**: Trained on NVIDIA A10G (48GB VRAM) via AWS g5.2xlarge
- **Training Time**: ~4-5 hours
- **Training Loss**: Final loss TBD (expected lower than Base/Medium)
- **Validation Loss**: Best validation loss TBD
- **Wandb Run**: seriguela-supervised-large-700k-YYYYMMDD-HHMMSS
- **Cost**: ~$5-6 USD for training

## JSON Format Details

The model is trained to complete JSON-structured prompts:

**Input prompt:**
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin", "cos", "**"], "cons": "C", "expr": "
```

**Model completes (example with nested operations):**
```json
sin(cos(x_1**2) + C*x_2)"}
```

**Why JSON format works better:**
1. Clear boundaries: `{` start and `}` end markers
2. Structured containment: Expression within `"expr": "..."` field
3. Lower training loss compared to EOS token approach (0.343 vs 0.415)
4. No repetition issues: Unlike EOS approach, model doesn't fall into repetitive patterns

## Limitations

- **Trained only on infix notation**: Does not support prefix or postfix notation
- **Domain-specific**: Optimized for symbolic regression expressions, not general mathematics
- **May generate invalid expressions**: On complex/unseen operator combinations
- **Division by zero**: May generate expressions that cause evaluation errors
- **LoRA constraints**: Parameter-efficient fine-tuning may limit adaptation vs full fine-tuning
- **Performance varies**: Significantly different results across different Nguyen benchmarks
- **Highest compute requirements**: Requires 48GB VRAM (g5.2xlarge) and longest training time
- **Inference cost**: Slower and more expensive than Base/Medium models

## Model Comparison

This Large model (774M) is part of a 3-model scaling study:
- **Base**: 124M parameters (baseline, fastest inference)
- **Medium**: 355M parameters (+186% size vs Base)
- **Large (this model)**: 774M parameters (+524% size vs Base, best quality expected)

Scaling hypothesis for Large:
- Best at generating complex, deeply nested expressions
- Highest valid expression rate and diversity
- Best R² scores on complex benchmarks
- May handle more operators and variables simultaneously
- Trade-off: Higher computational cost and slower inference

See `EXPERIMENT_MODEL_SCALING.md` for complete comparison results.

## When to Use This Model

**Use Large model when:**
- Maximum expression quality is needed
- Working with complex benchmarks (Nguyen-5+)
- Need for nested operations and power functions
- Computational resources are available (48GB+ VRAM)

**Use Medium/Base instead when:**
- Fast inference is required
- Limited computational resources
- Simple benchmarks (Nguyen-1, 2)
- Cost is a constraint

## Citation

```bibtex
@misc{gpt2_large_symbolic_regression_2025,
  title={GPT-2 Model Scaling for Symbolic Regression: Large Model (774M)},
  author={[Your Name/Research Group]},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/YOUR_USERNAME/gpt2_large_700K_json}}
}
```

## Related Work

- **Supervised Fine-tuning**: Uses LoRA (Hu et al., 2021) for parameter-efficient training
- **Reinforcement Learning**: Implements REINFORCE, GRPO (DeepSeek-R1 style), and PPO algorithms
- **Dataset**: Sintetico Natural (augustocsc/sintetico_natural) - 700K synthetic expressions
- **Benchmarks**: Nguyen et al. symbolic regression benchmarks (1-12)
- **Model Scaling**: Investigates impact of model size on symbolic regression capability

## Contact

For questions or issues:
- Repository: https://github.com/augustocsc/seriguela
- Issues: https://github.com/augustocsc/seriguela/issues

---

**Model Card Authors**: [Your Name]

**Model Card Last Updated**: 2025-02-02
