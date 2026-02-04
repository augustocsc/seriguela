#!/usr/bin/env python3
"""
Simple comparison of V1 vs V2 model generation quality
"""

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.expression import Expression


class ExpressionStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(seq, add_special_tokens=False)
                        for seq in stop_sequences]

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_ids:
            if len(stop_ids) > 0 and len(input_ids[0]) >= len(stop_ids):
                if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


def load_model(model_name, model_label):
    print(f"\n{'='*60}")
    print(f"Loading {model_label}: {model_name}")
    print('='*60)

    # Load base GPT-2
    print("Loading base GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
    })

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Load adapter and merge
    print(f"Loading adapter from {model_name}...")
    model = PeftModel.from_pretrained(model, model_name)
    print("Merging adapter...")
    model = model.merge_and_unload()
    model.eval()

    print(f"✓ {model_label} loaded successfully")
    return model, tokenizer


def test_model(model, tokenizer, model_label, n_samples=20):
    print(f"\n{'='*60}")
    print(f"Testing {model_label} - {n_samples} generations")
    print('='*60)

    # Same prompt for both models
    prompt = """vars: x_1, x_2
oper: *, +, -, sin, cos
cons: C
expr:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Stopping criteria
    stopping_criteria = StoppingCriteriaList([
        ExpressionStoppingCriteria(tokenizer, ["<|endofex|>", "\n\nvars:"])
    ])

    # Use OPTIMAL config for each model (from FINAL_RESULTS_V1_VS_V2.md)
    if model_label == "V1":
        # V1 optimal: 83.3% valid rate
        gen_config = {
            "temperature": 0.5,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "max_new_tokens": 100,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        print("Using V1 optimal config: temp=0.5, top_k=40, rep_penalty=1.15")
    else:  # V2
        # V2 optimal: 90% valid rate
        gen_config = {
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
            "max_new_tokens": 128,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        print("Using V2 optimal config: temp=0.7, top_p=0.8 (nucleus sampling)")

    results = {
        "valid_count": 0,
        "correct_symbols_count": 0,
        "expressions": []
    }

    allowed_vars = {"x_1", "x_2", "C"}
    allowed_ops = {"*", "+", "-", "sin", "cos", "(", ")"}

    print(f"\nGenerating {n_samples} expressions...\n")

    for i in range(n_samples):
        output = model.generate(
            **inputs,
            **gen_config,
            stopping_criteria=stopping_criteria
        )
        text = tokenizer.decode(output[0], skip_special_tokens=False)

        # Extract expression
        if "expr:" in text:
            expr_str = text.split("expr:")[-1].strip()
            expr_str = expr_str.split("<|endofex|>")[0].strip()
        else:
            expr_str = text

        # Check if valid (can be parsed and evaluated)
        is_valid = False
        try:
            expr = Expression(expr_str, is_prefix=False)
            X_test = [[1.0, 2.0]]  # Simple test
            result = expr.evaluate(X_test)
            if len(result) > 0 and all(x != float('inf') and x != float('-inf') and x == x for x in result):
                is_valid = True
                results["valid_count"] += 1
        except:
            pass

        # Check if uses only correct symbols
        has_correct_symbols = True
        # Remove spaces and check tokens
        expr_clean = expr_str.replace(" ", "")
        # Check for allowed patterns
        for char in expr_clean:
            if char.isalpha() and char not in "xCsinco_":
                has_correct_symbols = False
                break

        # Check for garbage words
        garbage_words = ["Buyable", "Instore", "Online", "Muslims", "crash", "Berman",
                        "vars:", "oper:", "expressed", "fluent", "Avenger", "repositories"]
        for word in garbage_words:
            if word in expr_str:
                has_correct_symbols = False
                break

        if has_correct_symbols:
            results["correct_symbols_count"] += 1

        results["expressions"].append({
            "index": i + 1,
            "expression": expr_str[:80],  # Limit display length
            "valid": is_valid,
            "correct_symbols": has_correct_symbols
        })

        # Show first 5 samples
        if i < 5:
            status = "✓ Valid" if is_valid else "✗ Invalid"
            symbols = "✓ Clean" if has_correct_symbols else "✗ Garbage"
            print(f"  [{i+1:2d}] {status:10s} {symbols:10s} | {expr_str[:60]}")

    print(f"\n{'-'*60}")
    print(f"RESULTS FOR {model_label}:")
    print(f"  Valid expressions:    {results['valid_count']:2d}/{n_samples} ({results['valid_count']/n_samples*100:.1f}%)")
    print(f"  Correct symbols only: {results['correct_symbols_count']:2d}/{n_samples} ({results['correct_symbols_count']/n_samples*100:.1f}%)")
    print(f"{'-'*60}")

    return results


def main():
    print("\n" + "="*60)
    print("V1 vs V2 MODEL COMPARISON")
    print("="*60)
    print("Testing same prompt on both models")
    print("Measuring: valid expressions + symbol correctness\n")

    # Test V1
    v1_model, v1_tokenizer = load_model("augustocsc/Se124M_700K_infix", "V1")
    v1_results = test_model(v1_model, v1_tokenizer, "V1", n_samples=20)

    # Clean up V1 from memory
    del v1_model
    torch.cuda.empty_cache()

    # Test V2
    v2_model, v2_tokenizer = load_model("augustocsc/Se124M_700K_infix_v2", "V2")
    v2_results = test_model(v2_model, v2_tokenizer, "V2", n_samples=20)

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<30s} {'V1':>10s} {'V2':>10s} {'Winner':>10s}")
    print("-"*60)

    v1_valid = v1_results["valid_count"]
    v2_valid = v2_results["valid_count"]
    valid_winner = "V1" if v1_valid > v2_valid else ("V2" if v2_valid > v1_valid else "TIE")
    print(f"{'Valid Expressions':<30s} {v1_valid:>10d} {v2_valid:>10d} {valid_winner:>10s}")

    v1_clean = v1_results["correct_symbols_count"]
    v2_clean = v2_results["correct_symbols_count"]
    clean_winner = "V1" if v1_clean > v2_clean else ("V2" if v2_clean > v1_clean else "TIE")
    print(f"{'Correct Symbols Only':<30s} {v1_clean:>10d} {v2_clean:>10d} {clean_winner:>10s}")

    print("-"*60)
    print(f"{'Valid Rate':<30s} {v1_valid/20*100:>9.1f}% {v2_valid/20*100:>9.1f}%")
    print(f"{'Clean Symbol Rate':<30s} {v1_clean/20*100:>9.1f}% {v2_clean/20*100:>9.1f}%")
    print("="*60)

    # Conclusion
    print("\nConclusion:")
    if v1_valid > v2_valid and v1_clean > v2_clean:
        print("  → V1 is better on both metrics")
    elif v2_valid > v1_valid and v2_clean > v1_clean:
        print("  → V2 is better on both metrics")
    else:
        print("  → Mixed results - models have different strengths")


if __name__ == "__main__":
    main()
