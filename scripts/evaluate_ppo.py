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

from transformers import AutoTokenizer, AutoModelForCausalLM
from classes.expression import Expression
from scripts.symbolic_rl.trainer import SymbolicRegressionPPOTrainer
from scripts.symbolic_rl.config import SymbolicRLConfig

class PPOEvaluator:
    """Evaluates if PPO training works for symbolic regression"""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load V2 model with optimal inference config (90% valid rate)
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
        })

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

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

        valid_count = 0
        r2_scores = []

        print(f"\nGenerating {n_samples} expressions...")
        for i in range(n_samples):
            output = self.model.generate(**inputs, **self.generation_config)
            text = self.tokenizer.decode(output[0], skip_special_tokens=False)

            # Extract expression
            if "expr:" in text:
                expr_str = text.split("expr:")[-1].strip()
                expr_str = expr_str.split("<|endofex|>")[0].strip()
            else:
                expr_str = text

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

    def test_ppo_training(self, target_formula: str = "x_1 * x_2", max_epochs: int = 5) -> Dict:
        """Test PPO training: Check if reward improves epoch-by-epoch"""
        print("\n" + "="*60)
        print("PPO TRAINING TEST: Check Reward Improvement")
        print("="*60)
        print(f"Target formula: {target_formula}")

        # Create target dataset
        X, y = self.create_synthetic_dataset(target_formula, n_samples=100)

        # Configure PPO
        config = SymbolicRLConfig()
        config.max_epochs = max_epochs
        config.reward_threshold = 0.9
        config.early_stopping_patience = 3

        # Create trainer
        print("\nInitializing PPO trainer...")
        trainer = SymbolicRegressionPPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=config
        )

        # Train
        print(f"\nStarting PPO training (max {max_epochs} epochs)...")
        print("Expected: Reward should increase epoch-by-epoch if PPO works\n")

        try:
            training_results = trainer.train(
                X=X,
                y=y,
                variables=['x_1', 'x_2'],
                operators=['*', '+', '-', 'sin', 'cos']
            )

            results = {
                "test": "ppo_training",
                "timestamp": datetime.now().isoformat(),
                "target_formula": target_formula,
                "config": {
                    "max_epochs": max_epochs,
                    "reward_threshold": config.reward_threshold
                },
                "training_results": training_results,
                "conclusion": self._analyze_ppo_results(training_results)
            }

            # Print summary
            print("\n" + "-"*60)
            print("PPO TRAINING RESULTS:")
            if "epoch_rewards" in training_results:
                rewards = training_results["epoch_rewards"]
                print(f"  Initial reward: {rewards[0]:.4f}")
                print(f"  Final reward: {rewards[-1]:.4f}")
                print(f"  Improvement: {rewards[-1] - rewards[0]:+.4f}")
                print(f"  Best reward: {max(rewards):.4f}")

                if rewards[-1] > rewards[0] + 0.1:
                    print(f"  ✅ SUCCESS: Reward improved significantly!")
                elif rewards[-1] > rewards[0]:
                    print(f"  ⚠️  PARTIAL: Small improvement, may need more epochs")
                else:
                    print(f"  ❌ FAILURE: No improvement, PPO may not be working")
            print("-"*60)

        except Exception as e:
            print(f"\n❌ Error during PPO training: {e}")
            import traceback
            traceback.print_exc()

            results = {
                "test": "ppo_training",
                "timestamp": datetime.now().isoformat(),
                "target_formula": target_formula,
                "error": str(e),
                "conclusion": "PPO training failed with error"
            }

        # Save results
        output_file = self.output_dir / "ppo_training_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return results

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
    baseline_results = evaluator.test_baseline_generation(n_samples=20)

    # Test 2: PPO training
    print("\n🎯 TEST 2: PPO Training (Check if reward improves)")
    ppo_results = evaluator.test_ppo_training(target_formula="x_1 * x_2", max_epochs=5)

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults saved to: ./logs/ppo_evaluation/")
    print("\nKey Questions Answered:")
    print("1. Does V2 generate valid expressions? Check baseline_results.json")
    print("2. Does PPO improve reward? Check ppo_training_results.json")
    print("3. Can PPO find target formula? Check final R² score")
    print("\nNext steps:")
    print("- If PPO works: Test on more complex Feynman equations")
    print("- If PPO fails: Debug reward function, check hyperparameters")
    print("="*60)


if __name__ == "__main__":
    main()
