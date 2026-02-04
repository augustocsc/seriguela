---
language: en
license: mit
tags:
- symbolic-regression
- gpt2
- lora
- expression-generation
- mathematics
datasets:
- augustocsc/sintetico_natural
metrics:
- accuracy
- r2_score
model-index:
- name: GPT-2 Base for Symbolic Regression
  results:
  - task:
      type: symbolic-regression
      name: Expression Generation Quality
    dataset:
      name: Sintetico Natural 700K
      type: augustocsc/sintetico_natural
    metrics:
    - name: Valid Expression Rate
      type: accuracy
      value: 99.4
    - name: Diversity Rate
      type: diversity
      value: 97.8
  - task:
      type: symbolic-regression
      name: Nguyen Benchmark Suite
    dataset:
      name: Nguyen Benchmarks 1-12
      type: nguyen-symbolic-regression
    metrics:
    - name: Average Valid Rate
      type: accuracy
      value: 62.5
    - name: Average Best R²
      type: r2_score
      value: 0.9190
    - name: Maximum R²
      type: r2_score
      value: 0.9994
---

# GPT-2 Base for Symbolic Regression (JSON Format)

## Model Description

This model is a GPT-2 Base (124M parameters) fine-tuned using LoRA for symbolic regression expression generation. It was trained on 700K synthetic mathematical expressions in JSON format, achieving **99.4% valid expression rate** and solid performance on standard symbolic regression benchmarks.

**Part of research study**: "Impact of Model Size on Symbolic Regression Capability in Large Language Models"

This is the baseline model in a comprehensive model scaling study comparing Base (124M), Medium (355M), and Large (774M) parameter models.

## Model Details

### Architecture
- **Base Model**: gpt2 (124M parameters)
- **Trainable Parameters**: ~294K (LoRA adapters only - 0.24% of total)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: 700K expressions from augustocsc/sintetico_natural
- **Data Format**: JSON structured format (EXP-A)
- **Framework**: HuggingFace Transformers + PEFT

### LoRA Configuration
```json
{
  "r": 8,
  "lora_alpha": 32,
  "target_modules": ["c_attn"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

### Training Hyperparameters
```json
{
  "learning_rate": 5e-5,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 32,
  "warmup_steps": 500,
  "weight_decay": 0.01,
  "fp16": true,
  "early_stopping_patience": 3,
  "seed": 42
}
```

### Training Details
- **Training Duration**: ~2-3 hours on NVIDIA A10G (24GB)
- **Instance Type**: AWS g5.xlarge
- **Early Stopping**: Enabled (patience=3, monitored validation loss)
- **Final Training Loss**: [Value from training logs]
- **Dataset Split**: 90% train / 10% validation

## Performance

### Expression Generation Quality (500 samples)

| Metric | Score |
|--------|-------|
| **Valid Expression Rate** | **99.4%** |
| **Diversity Rate** | 97.8% |
| **Unique Expressions** | 489 / 500 |
| **Errors** | 3 / 500 (0.6%) |

**Key Strengths**:
- Near-perfect valid expression generation (99.4%)
- High diversity (97.8% unique expressions)
- Very few errors (only 3 in 500 samples)
- Fast inference (smallest model)
- Most economical option

### Nguyen Benchmark Performance (12 benchmarks, 1,200 expressions)

| Metric | Score |
|--------|-------|
| **Average Valid Rate** | 62.5% |
| **Valid Rate Range** | 46.0% - 93.0% |
| **Average Best R²** | 0.9190 |
| **R² Range** | 0.6735 - 0.9994 |
| **Benchmarks with R² > 0.99** | 4 / 12 |
| **Average Execution Time** | 95.1 seconds per benchmark |

**Per-Benchmark Results** (Best R²):

| Benchmark | Formula | Valid Rate | Best R² |
|-----------|---------|------------|---------|
| Nguyen-1 | x³ + x² + x | 49% | 0.9717 |
| Nguyen-2 | x⁴ + x³ + x² + x | 52% | 0.9975 |
| Nguyen-3 | x⁵ + x⁴ + x³ + x² + x | 46% | 0.9778 |
| Nguyen-4 | x⁶ + x⁵ + x⁴ + x³ + x² + x | 46% | 0.7793 |
| Nguyen-5 | sin(x²)·cos(x) - 1 | 56% | 0.9322 |
| Nguyen-6 | sin(x) + sin(x + x²) | 53% | 0.9982 |
| Nguyen-7 | log(x + 1) + log(x² + 1) | 84% | 0.9983 |
| Nguyen-8 | √x | 82% | 0.9761 |
| Nguyen-9 | sin(x) + sin(y²) | 56% | 0.8038 |
| Nguyen-10 | 2·sin(x)·cos(y) | 50% | **0.9994** ⭐ |
| Nguyen-11 | x^y | 93% | 0.9199 |
| Nguyen-12 | x⁴ - x³ + y²/2 - y | 83% | 0.6735 |

**Best Result**: Nguyen-10 with R² = 0.9994 (near-perfect fit)

**Observations**:
- Excels on simpler benchmarks (Nguyen 7, 10)
- Struggles with complex benchmarks requiring nested operations (Nguyen 4, 12)
- Valid rate varies significantly (46-93%) depending on benchmark complexity
- Fast execution (~95s per benchmark) - fastest among all model sizes

## Usage

### Installation

```bash
pip install transformers peft torch
```

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter (replace with actual HuggingFace repo)
model = PeftModel.from_pretrained(base_model, "USER/gpt2_base_700K_json")
model.eval()
```

### Generating Expressions

```python
import torch

# Define prompt in JSON format
prompt = '{"vars": ["x_1"], "ops": ["*", "+", "sin", "cos"], "cons": "C", "expr": "'

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.encode('"', add_special_tokens=False)[0]  # Stop at closing quote
    )

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract expression (after "expr": ")
expression = generated_text.split('"expr": "')[1].split('"')[0]
print(f"Generated expression: {expression}")
```

### Validation with SymPy

```python
import sympy as sp

def validate_expression(expr_str, variables):
    """Validate expression using SymPy."""
    try:
        # Define symbols
        symbols = {var: sp.Symbol(var) for var in variables}

        # Parse expression
        expr = sp.sympify(expr_str, locals=symbols)

        # Check if valid
        return True, expr
    except Exception as e:
        return False, str(e)

# Example
is_valid, result = validate_expression("sin(x_1**2) + cos(x_1)", ["x_1"])
print(f"Valid: {is_valid}, Expression: {result}")
```

## Intended Use

### Primary Use Cases
- **Symbolic regression research**: Baseline model for symbolic regression experiments
- **Fast prototyping**: Quick expression generation with reasonable quality
- **Educational purposes**: Teaching symbolic regression with LLMs
- **Cost-sensitive applications**: Best performance/cost ratio

### Recommended For
- Simple to moderate complexity benchmarks (Nguyen 1-3, 7-8, 10-11)
- Applications where speed is critical
- Scenarios with limited computational resources
- Baseline comparisons in research

### Not Recommended For
- Complex nested expressions (use Medium or Large models)
- Production systems requiring >95% valid rate on complex benchmarks
- Tasks requiring maximum possible accuracy (consider Large model)

## Limitations

### Known Issues
1. **Complex Expression Generation**: Struggles with deeply nested expressions (avg depth 1.40)
2. **Power Operations**: Limited use of power operations (15.9% on Nguyen-5)
3. **Variable Valid Rates**: Valid rate drops to 46% on complex benchmarks (Nguyen-3, 4)
4. **Benchmark Performance**: Average R² of 0.919 - good but not state-of-the-art
5. **No Nested Trigonometry**: 0% nested trig functions (e.g., sin(cos(x)))

### Model Scaling Insights

Compared to larger models in the study:
- **Medium (355M)**: +12.7% valid rate, +6.8% R² improvement
- **Large (774M)**: +26.5% valid rate, +7.2% R² improvement

**Conclusion**: For maximum quality, consider the Medium or Large models. For speed and cost-effectiveness, this Base model is excellent.

### General Limitations
- Trained only on infix notation
- May generate expressions with division by zero or undefined operations
- LoRA fine-tuning provides less adaptation than full fine-tuning
- No reinforcement learning optimization (supervised learning only)
- Performance depends heavily on prompt format (JSON required)

## Ethical Considerations

- **Bias**: Model inherits biases from GPT-2 pretraining (minimize impact via symbolic math focus)
- **Misuse**: Could generate incorrect formulas - always validate outputs
- **Environmental**: Small carbon footprint (~2-3 hours GPU training)
- **Transparency**: All training details, hyperparameters, and evaluation metrics disclosed

## Model Card Authors

**Research Team**: [Your Name/Institution]

**Contact**: [Email or GitHub]

**Date**: February 2025

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{gpt2_base_symbolic_regression_2025,
  title={GPT-2 Base Model for Symbolic Regression: A Model Scaling Study},
  author={[Your Name]},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/USER/gpt2_base_700K_json}},
  note={Trained on 700K expressions with LoRA fine-tuning}
}
```

## Acknowledgments

- **Dataset**: augustocsc/sintetico_natural (HuggingFace Hub)
- **Framework**: HuggingFace Transformers, PEFT (LoRA)
- **Compute**: AWS g5.xlarge with NVIDIA A10G GPU
- **Experiment Tracking**: Weights & Biases

## Additional Resources

- **Research Report**: See SCIENTIFIC_REPORT_MODEL_SCALING.md
- **Training Details**: See TRAINING_LOG_MODEL_SCALING_2025.md
- **Benchmark Results**: See NGUYEN_RESULTS_FINAL.md
- **Visualizations**: See visualizations/ directory
- **Comparison with Medium/Large**: See FINAL_STATUS.md

---

**Model Version**: 1.0
**Last Updated**: 2026-02-04
**Status**: Production-Ready
