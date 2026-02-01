#!/usr/bin/env python3
"""
PPO Evaluation Script for Seriguela Block 3
Tests if PPO finetuning can find symbolic regression expressions
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from classes.expression import Expression


class ExpressionStoppingCriteria(StoppingCriteria):
    """Stop generation at natural expression boundaries."""
    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(seq, add_special_tokens=False)
                        for seq in stop_sequences]

    def __call__(self, input_ids, scores, **kwargs):
        # Check if any stop sequence appears in generated text
        for stop_ids in self.stop_ids:
            if len(stop_ids) > 0 and len(input_ids[0]) >= len(stop_ids):
                if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                    return True
        return False

class PPOEvaluator:
    """Evaluates if PPO training works for symbolic regression"""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load V2 model with optimal inference config (90% valid rate)
        print(f"Loading model: {model_name}")

        # Load base model first without adapters
        print("Loading base GPT-2 model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Configure tokenizer with special tokens
        print("Configuring tokenizer with special tokens...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
        })

        # Resize embeddings to match tokenizer
        print(f"Resizing embeddings from {self.model.get_input_embeddings().weight.shape[0]} to {len(self.tokenizer)}...")
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Now load the V2 adapter weights
        print(f"Loading V2 adapter from {model_name}...")
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_name)
            print("V2 adapter loaded successfully (LoRA weights)")
        except Exception as e:
            print(f"Warning: Could not load as PEFT model: {e}")
            print("Attempting to load as full model...")
            # If not a PEFT model, load full weights
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # V2 optimal generation config (from FINAL_RESULTS)
        self.generation_config = {
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
            "max_new_tokens": 128,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        print(f"Model loaded. Using optimal V2 configuration.")

    def create_synthetic_dataset(self, formula: str, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic dataset from a known formula"""
        print(f"Creating dataset for formula: {formula}")

        # Generate random input data
        X = np.random.uniform(-2, 2, (n_samples, 2))

        # Evaluate true formula
        try:
            expr = Expression(formula, is_prefix=False)
            y = expr.evaluate(X)
            return X, y
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise

    def test_baseline_generation(self, n_samples: int = 10) -> Dict:
        """Test baseline: V2 generates valid expressions but not fitted to data"""
        print("\n" + "="*60)
        print("BASELINE TEST: V2 Generation Without PPO")
        print("="*60)

        # Create test dataset (simple formula)
        X, y = self.create_synthetic_dataset("x_1 * x_2", n_samples=50)

        results = {
            "test": "baseline_generation",
            "timestamp": datetime.now().isoformat(),
            "generations": [],
            "summary": {}
        }

        prompt = """vars: x_1, x_2
oper: *, +, -, sin, cos
cons: C
expr:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Create stopping criteria for <|endofex|>
        stopping_criteria = StoppingCriteriaList([
            ExpressionStoppingCriteria(self.tokenizer, ["<|endofex|>", "\n\nvars:"])
        ])

        valid_count = 0
        r2_scores = []

        print(f"\nGenerating {n_samples} expressions...")
        for i in range(n_samples):
            output = self.model.generate(
                **inputs,
                **self.generation_config,
                stopping_criteria=stopping_criteria
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=False)

            # Extract expression
            if "expr:" in text:
                expr_str = text.split("expr:")[-1].strip()
                expr_str = expr_str.split("<|endofex|>")[0].strip()
            else:
                expr_str = text

            # Debug: Show first few generations
            if i < 3:
                print(f"\n  DEBUG Sample {i+1}:")
                print(f"    Raw output: {text[:200]}")
                print(f"    Extracted: {expr_str[:100]}")

            # Validate and compute R²
            is_valid = False
            r2 = -1.0

            try:
                expr = Expression(expr_str, is_prefix=False)
                # Check if expression can be evaluated on dataset
                if expr.is_valid_on_dataset(X):
                    is_valid = True
                    valid_count += 1

                    # Fit constants and compute R²
                    try:
                        r2 = expr.fit_constants(X, y)
                        if np.isfinite(r2):
                            r2_scores.append(r2)
                        else:
                            r2 = -1.0
                    except:
                        r2 = -1.0
            except:
                pass

            results["generations"].append({
                "index": i + 1,
                "expression": expr_str,
                "valid": is_valid,
                "r2_score": float(r2) if r2 != -1.0 else None
            })

            if (i + 1) % 5 == 0:
                print(f"Generated {i + 1}/{n_samples} - Valid: {valid_count}, Avg R²: {np.mean(r2_scores) if r2_scores else 'N/A'}")

        # Summary
        results["summary"] = {
            "total_generations": n_samples,
            "valid_count": valid_count,
            "valid_rate": valid_count / n_samples,
            "r2_scores": r2_scores,
            "mean_r2": float(np.mean(r2_scores)) if r2_scores else None,
            "max_r2": float(np.max(r2_scores)) if r2_scores else None,
            "conclusion": "Baseline generates valid expressions but R² is low (not fitted to target)"
        }

        print("\n" + "-"*60)
        print(f"BASELINE RESULTS:")
        print(f"  Valid Rate: {results['summary']['valid_rate']:.1%} ({valid_count}/{n_samples})")
        print(f"  Mean R²: {results['summary']['mean_r2']:.4f}" if r2_scores else "  Mean R²: N/A")
        print(f"  Max R²: {results['summary']['max_r2']:.4f}" if r2_scores else "  Max R²: N/A")
        print(f"  Interpretation: V2 generates valid expressions (good!), but doesn't fit target data (expected without PPO)")
        print("-"*60)

        # Save results
        output_file = self.output_dir / "baseline_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return results

    def test_ppo_simulation(self, target_formula: str = "x_1 * x_2", n_iterations: int = 10) -> Dict:
        """Simulate PPO: Generate expressions and check if best reward improves"""
        print("\n" + "="*60)
        print("PPO SIMULATION TEST: Check if Reward Can Improve")
        print("="*60)
        print(f"Target formula: {target_formula}")
        print("Note: This simulates PPO by generating multiple expressions")
        print("      and tracking best R² score. Real PPO would optimize")
        print("      the model to generate better expressions over time.")

        # Create target dataset
        X, y = self.create_synthetic_dataset(target_formula, n_samples=100)

        prompt = """vars: x_1, x_2
oper: *, +, -, sin, cos
cons: C
expr:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Create stopping criteria
        stopping_criteria = StoppingCriteriaList([
            ExpressionStoppingCriteria(self.tokenizer, ["<|endofex|>", "\n\nvars:"])
        ])

        results = {
            "test": "ppo_simulation",
            "timestamp": datetime.now().isoformat(),
            "target_formula": target_formula,
            "iterations": [],
            "summary": {}
        }

        print(f"\nGenerating {n_iterations} expressions and tracking best R²...")

        best_r2 = -np.inf
        best_expr = None
        r2_history = []
        valid_count = 0

        for i in range(n_iterations):
            output = self.model.generate(
                **inputs,
                **self.generation_config,
                stopping_criteria=stopping_criteria
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=False)

            # Extract expression
            if "expr:" in text:
                expr_str = text.split("expr:")[-1].strip()
                expr_str = expr_str.split("<|endofex|>")[0].strip()
            else:
                expr_str = text

            # Compute reward (R²)
            is_valid = False
            r2 = -1.0

            try:
                expr = Expression(expr_str, is_prefix=False)
                if expr.is_valid_on_dataset(X):
                    is_valid = True
                    valid_count += 1
                    r2 = expr.fit_constants(X, y)

                    if np.isfinite(r2):
                        r2_history.append(r2)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_expr = expr_str
                    else:
                        r2 = -1.0
            except:
                pass

            results["iterations"].append({
                "iteration": i + 1,
                "expression": expr_str,
                "valid": is_valid,
                "r2": float(r2) if np.isfinite(r2) else None,
                "is_best": (r2 == best_r2) if np.isfinite(r2) else False
            })

            if (i + 1) % 5 == 0:
                print(f"Iteration {i + 1}/{n_iterations} - Valid: {valid_count}, Best R²: {best_r2:.4f}")

        # Summary
        results["summary"] = {
            "total_iterations": n_iterations,
            "valid_count": valid_count,
            "valid_rate": valid_count / n_iterations,
            "best_r2": float(best_r2) if np.isfinite(best_r2) else None,
            "best_expression": best_expr,
            "r2_history": [float(r) for r in r2_history],
            "mean_r2": float(np.mean(r2_history)) if r2_history else None,
            "conclusion": self._analyze_ppo_simulation(best_r2, r2_history)
        }

        print("\n" + "-"*60)
        print("PPO SIMULATION RESULTS:")
        print(f"  Valid expressions: {valid_count}/{n_iterations}")
        print(f"  Best R²: {best_r2:.4f}" if np.isfinite(best_r2) else "  Best R²: N/A")
        print(f"  Mean R²: {results['summary']['mean_r2']:.4f}" if r2_history else "  Mean R²: N/A")
        print(f"  Best expression: {best_expr}")
        print(f"\n  Interpretation:")
        print(f"    - Baseline (Test 1) shows random expressions have low R² (~0.2)")
        print(f"    - PPO should improve this by learning to generate fitted expressions")
        print(f"    - Best R² of {best_r2:.4f} shows what's possible with current model")
        if best_r2 >= 0.9:
            print(f"    ✅ Model CAN find high-quality solutions (R² >= 0.9)")
        elif best_r2 >= 0.5:
            print(f"    ⚠️  Model can find partial solutions (R² >= 0.5)")
        else:
            print(f"    ❌ Model struggles to find good solutions (R² < 0.5)")
        print("-"*60)

        # Save results
        output_file = self.output_dir / "ppo_simulation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return results

    def _analyze_ppo_simulation(self, best_r2: float, r2_history: List[float]) -> str:
        """Analyze PPO simulation results"""
        if not r2_history:
            return "❌ No valid expressions generated"

        if best_r2 >= 0.9:
            return f"✅ EXCELLENT: Found high-quality solution (R² = {best_r2:.4f}). PPO training should work well."
        elif best_r2 >= 0.5:
            return f"⚠️  MODERATE: Found partial solution (R² = {best_r2:.4f}). PPO may help but needs tuning."
        else:
            return f"❌ POOR: Best solution is weak (R² = {best_r2:.4f}). PPO will struggle with current model."

    def _analyze_ppo_results(self, training_results: Dict) -> str:
        """Analyze PPO training results and provide conclusion"""
        if "epoch_rewards" not in training_results:
            return "Unable to analyze: No reward history found"

        rewards = training_results["epoch_rewards"]
        initial = rewards[0]
        final = rewards[-1]
        best = max(rewards)
        improvement = final - initial

        if best >= 0.9:
            return f"✅ EXCELLENT: Found high-quality solution (R² = {best:.4f})"
        elif improvement > 0.2:
            return f"✅ GOOD: Significant improvement ({improvement:+.4f}), PPO is working"
        elif improvement > 0.05:
            return f"⚠️  MODERATE: Some improvement ({improvement:+.4f}), may need more epochs"
        elif improvement > 0:
            return f"⚠️  WEAK: Minimal improvement ({improvement:+.4f}), check hyperparameters"
        else:
            return f"❌ POOR: No improvement or decline ({improvement:+.4f}), PPO not working properly"


def main():
    print("="*60)
    print("SERIGUELA BLOCK 3: PPO EVALUATION")
    print("="*60)
    print("Objective: Test if PPO finetuning works for symbolic regression")
    print("Model: V2 (augustocsc/Se124M_700K_infix_v2)")
    print("="*60)

    # Initialize evaluator
    evaluator = PPOEvaluator(
        model_name="augustocsc/Se124M_700K_infix_v2",
        output_dir="./logs/ppo_evaluation"
    )

    # Test 1: Baseline generation
    print("\n📊 TEST 1: Baseline Generation (V2 without PPO)")
    baseline_results = evaluator.test_baseline_generation(n_samples=30)

    # Test 2: PPO simulation
    print("\n🎯 TEST 2: PPO Simulation (Check if reward CAN improve)")
    ppo_results = evaluator.test_ppo_simulation(target_formula="x_1 * x_2", n_iterations=50)

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults saved to: ./logs/ppo_evaluation/")
    print("\nKey Questions Answered:")
    print("1. Does V2 generate valid expressions? Check baseline_results.json")
    print(f"   Answer: {baseline_results['summary']['valid_rate']:.1%} valid rate")
    print("2. Can model find high R² expressions? Check ppo_simulation_results.json")
    best_r2 = ppo_results['summary'].get('best_r2')
    if best_r2 is None:
        best_r2 = -1.0
    if best_r2 >= 0.9:
        print(f"   Answer: YES! Best R² = {best_r2:.4f} (excellent)")
    elif best_r2 >= 0.5:
        print(f"   Answer: PARTIAL. Best R² = {best_r2:.4f} (moderate)")
    else:
        print(f"   Answer: NO. Best R² = {best_r2:.4f} (poor)")
    print("3. Would PPO training work?")
    if best_r2 >= 0.9:
        print("   Answer: YES - Model can find solutions, PPO should learn to find them consistently")
    elif best_r2 >= 0.5:
        print("   Answer: MAYBE - Model finds partial solutions, PPO may need tuning")
    else:
        print("   Answer: UNLIKELY - Model struggles to find solutions even randomly")
    print("\nNext steps:")
    print("- Review results to understand baseline performance")
    print("- If simulation shows high R², PPO training is worth trying")
    print("- If simulation shows low R², may need to retrain base model")
    print("="*60)


if __name__ == "__main__":
    main()
