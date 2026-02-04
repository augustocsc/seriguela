---
language: en
license: mit
tags:
- symbolic-regression
- gpt2-medium
- lora
- expression-generation
- mathematics
datasets:
- augustocsc/sintetico_natural
metrics:
- accuracy
- r2_score
model-index:
- name: GPT-2 Medium for Symbolic Regression
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

# GPT-2 Medium for Symbolic Regression (JSON Format)

## Model Description

This model is a GPT-2 Medium variant (355M parameters) fine-tuned using LoRA for symbolic regression expression generation. It was trained on 700K synthetic mathematical expressions in JSON format.

**Part of research study**: "Impact of Model Size on Symbolic Regression Capability" (Feb 2025)

**Hypothesis**: Medium model (355M) should show improved capacity for complex compositional patterns compared to Base (124M), including:
- Higher rates of power operations (x², x**n)
- Increased expression depth (nested functions)
- Better performance on complex benchmarks

## Model Details

- **Base Model**: gpt2-medium (355M parameters)
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
  - Batch size: 4 per device (reduced due to larger model)
  - Gradient accumulation: 4 steps
  - Warmup steps: 500
  - Weight decay: 0.01
  - FP16: Yes
  - Seed: 42

## Performance

### Supervised Fine-Tuning (Generation Quality)

Results will be filled after evaluation completes.

| Metric | Value | vs Base (124M) |
|--------|-------|----------------|
| Valid Expression Rate | TBD | TBD |
| Parseable Rate | TBD | TBD |
| Constraint Adherence | TBD | TBD |
| Diversity Rate | TBD | TBD |
| Power Operations Usage | TBD | TBD |
| Average Expression Depth | TBD | TBD |

**Expected improvements over Base:**
- +20-30% power operations usage
- Expression depth: 1.8-2.0 (vs 1.4 for Base)
- Possible emergence of nested trigonometric functions

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

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model (355M parameters)
base_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/gpt2_medium_700K_json")

# Generate expression
prompt = '{"vars": ["x_1"], "ops": ["*", "+", "sin", "**"], "cons": "C", "expr": "'
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
- **GPU**: Trained on NVIDIA A10G (24GB VRAM) via AWS g5.xlarge
- **Training Time**: ~3-4 hours
- **Training Loss**: Final loss TBD
- **Validation Loss**: Best validation loss TBD
- **Wandb Run**: seriguela-supervised-medium-700k-YYYYMMDD-HHMMSS

## JSON Format Details

The model is trained to complete JSON-structured prompts:

**Input prompt:**
```json
{"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin", "**"], "cons": "C", "expr": "
```

**Model completes (example with power operation):**
```json
sin(x_1**2 + C*x_2)"}
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
- **Increased compute**: Requires more memory and compute time than Base (124M)

## Model Comparison

This Medium model (355M) is part of a 3-model scaling study:
- **Base**: 124M parameters (baseline)
- **Medium (this model)**: 355M parameters (+186% size vs Base)
- **Large**: 774M parameters (+118% size vs Medium)

Scaling hypothesis for Medium:
- Better than Base at generating complex expressions (power ops, nesting)
- May not reach full complexity of Large model
- Balanced trade-off between performance and computational cost

See `EXPERIMENT_MODEL_SCALING.md` for complete comparison results.

## Citation

```bibtex
@misc{gpt2_medium_symbolic_regression_2025,
  title={GPT-2 Model Scaling for Symbolic Regression: Medium Model (355M)},
  author={[Your Name/Research Group]},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/YOUR_USERNAME/gpt2_medium_700K_json}}
}
```

## Related Work

- **Supervised Fine-tuning**: Uses LoRA (Hu et al., 2021) for parameter-efficient training
- **Reinforcement Learning**: Implements REINFORCE, GRPO (DeepSeek-R1 style), and PPO algorithms
- **Dataset**: Sintetico Natural (augustocsc/sintetico_natural) - 700K synthetic expressions
- **Benchmarks**: Nguyen et al. symbolic regression benchmarks (1-12)

## Contact

For questions or issues:
- Repository: https://github.com/augustocsc/seriguela
- Issues: https://github.com/augustocsc/seriguela/issues

---

**Model Card Authors**: [Your Name]

**Model Card Last Updated**: 2025-02-02
