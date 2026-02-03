#!/usr/bin/env python3
"""
Debug version of REINFORCE that saves ALL expressions (valid and invalid).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from expression import Expression


class DebugREINFORCE:
    """REINFORCE that logs all expressions."""

    def __init__(self, model_path: str, X: np.ndarray, y: np.ndarray, device: str = None):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            base_model = AutoModelForCausalLM.from_pretrained("gpt2")
            if len(self.tokenizer) != base_model.config.vocab_size:
                base_model.resize_token_embeddings(len(self.tokenizer))
            model_with_lora = PeftModel.from_pretrained(base_model, model_path)
            self.model = model_with_lora.merge_and_unload()
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Add LoRA
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05, bias="none")
        self.model = get_peft_model(self.model, lora_config)
        self.model = self.model.to(self.device)
        self.model.train()

        # Build prompt
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]
        ops_list = ["+", "-", "*", "/", "sin", "cos", "sqrt", "log", "exp", "pow"]
        self.prompt = json.dumps({"vars": vars_list, "ops": ops_list, "cons": "C", "expr": ""})[:-2]
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors="pt")["input_ids"].to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Baseline
        self.baseline = 0.0
        self.baseline_decay = 0.9

        # ALL expressions log
        self.all_expressions = []

    def extract_expression(self, text: str) -> str:
        """Extract expression from generated text."""
        try:
            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                for terminator in ['"}', '"']:
                    if terminator in remaining:
                        return remaining[:remaining.index(terminator)].strip()
        except:
            pass
        return text.strip()

    def compute_r2(self, expression_str: str) -> tuple:
        """Compute R^2 and detailed error info."""
        result = {
            "expression": expression_str,
            "r2": -1.0,
            "is_valid": False,
            "error_type": None,
            "error_message": None,
        }

        if not expression_str or expression_str.isspace():
            result["error_type"] = "empty"
            return result

        test_expr = expression_str.replace('C', '1')

        try:
            expr = Expression(test_expr, is_prefix=False)

            if not expr.is_valid_on_dataset(self.X):
                result["error_type"] = "invalid_on_dataset"
                result["error_message"] = "NaN/Inf on dataset"
                return result

            y_pred = expr.evaluate(self.X)

            if not np.all(np.isfinite(y_pred)):
                result["error_type"] = "non_finite_output"
                return result

            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

            if ss_tot == 0:
                r2 = 0.0
            else:
                r2 = 1 - (ss_res / ss_tot)

            result["r2"] = float(np.clip(r2, -1.0, 1.0))
            result["is_valid"] = True

        except Exception as e:
            result["error_type"] = "parse_error"
            result["error_message"] = str(e)[:100]

        return result

    def generate_batch(self, batch_size: int = 16, max_new_tokens: int = 50):
        """Generate batch and evaluate."""
        results = []

        for _ in range(batch_size):
            generated_ids = self.prompt_ids.clone()
            generated_tokens = []

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(generated_ids)
                    logits = outputs.logits[:, -1, :] / 0.7

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    generated_tokens.append(next_token.item())
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    if '"}' in text[len(self.prompt):]:
                        break

            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)

            # Evaluate with detailed info
            eval_result = self.compute_r2(expr_str)

            # Compute log prob
            if len(generated_tokens) > 0:
                full_ids = torch.cat([self.prompt_ids, torch.tensor([generated_tokens], device=self.device)], dim=1)
                outputs = self.model(full_ids[:, :-1])
                logits = outputs.logits / 0.7
                prompt_len = self.prompt_ids.shape[1]
                gen_logits = logits[:, prompt_len-1:, :]
                log_probs_all = F.log_softmax(gen_logits, dim=-1)
                target_tokens = torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
                selected_log_probs = log_probs_all.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                total_log_prob = selected_log_probs.sum()
            else:
                total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)

            eval_result["log_prob"] = total_log_prob
            results.append(eval_result)

            # Log ALL expressions
            self.all_expressions.append(eval_result.copy())

        return results

    def train_step(self, batch_size: int = 16):
        """One training step."""
        results = self.generate_batch(batch_size)

        # Compute rewards
        rewards = [r["r2"] if r["is_valid"] else -0.1 for r in results]

        # Update baseline
        valid_rewards = [r for r in rewards if r > -0.1]
        if valid_rewards:
            mean_reward = np.mean(valid_rewards)
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_reward

        # Advantages
        advantages = [r - self.baseline for r in rewards]

        # Update
        self.optimizer.zero_grad()
        policy_loss = torch.tensor(0.0, device=self.device)

        for result, advantage in zip(results, advantages):
            if result["is_valid"] or result["error_type"] == "parse_error":
                policy_loss = policy_loss - result["log_prob"] * advantage

        if len(results) > 0:
            policy_loss = policy_loss / len(results)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Stats
        valid_count = sum(1 for r in results if r["is_valid"])
        valid_r2 = [r["r2"] for r in results if r["is_valid"]]

        return {
            "valid_count": valid_count,
            "total_count": len(results),
            "mean_r2": np.mean(valid_r2) if valid_r2 else -1.0,
            "max_r2": max(r["r2"] for r in results),
            "baseline": self.baseline,
        }

    def run(self, epochs: int = 10):
        """Run training."""
        print(f"Running debug REINFORCE for {epochs} epochs...")
        print()

        for epoch in range(1, epochs + 1):
            stats = self.train_step()
            print(f"Epoch {epoch:2d} | Valid: {stats['valid_count']}/{stats['total_count']} | Mean R²: {stats['mean_r2']:.4f} | Max R²: {stats['max_r2']:.4f}")

        # Save ALL expressions
        output_file = "debug_expressions.json"
        with open(output_file, "w") as f:
            json.dump({"all_expressions": self.all_expressions}, f, indent=2, default=str)

        print()
        print(f"Saved {len(self.all_expressions)} expressions to {output_file}")

        # Analyze
        valid = [e for e in self.all_expressions if e["is_valid"]]
        invalid = [e for e in self.all_expressions if not e["is_valid"]]

        print()
        print("SUMMARY:")
        print(f"  Total: {len(self.all_expressions)}")
        print(f"  Valid: {len(valid)} ({100*len(valid)/len(self.all_expressions):.1f}%)")
        print(f"  Invalid: {len(invalid)} ({100*len(invalid)/len(self.all_expressions):.1f}%)")

        if invalid:
            error_types = {}
            for e in invalid:
                et = e.get("error_type", "unknown")
                error_types[et] = error_types.get(et, 0) + 1

            print()
            print("Invalid expression types:")
            for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  {et}: {count} ({100*count/len(invalid):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    import pandas as pd
    df = pd.read_csv(args.dataset)
    x_cols = [c for c in df.columns if c.startswith('x_')]
    X = df[x_cols].values
    y = df['y'].values

    print(f"Dataset: {args.dataset}")
    print(f"  Samples: {len(df)}, Variables: {len(x_cols)}")
    print()

    # Run
    reinforce = DebugREINFORCE(args.model_path, X, y)
    reinforce.run(epochs=args.epochs)


if __name__ == "__main__":
    main()
