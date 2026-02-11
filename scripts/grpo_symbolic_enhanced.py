#!/usr/bin/env python3
"""
Enhanced GRPO for Symbolic Regression with Epoch Tracking
Saves all expressions and metrics per epoch for analysis
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from expression import Expression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class EnhancedGRPO:
    """Enhanced GRPO with comprehensive tracking."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/grpo_enhanced",
        learning_rate: float = 5e-5,
        device: str = None,
        group_size: int = 8,
        kl_coef: float = 0.01,
        clip_range: float = 0.2,
        # Enhanced tracking
        save_all_expressions: bool = True,
        is_prefix: bool = True,
        custom_prompt: str = None,
    ):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range

        # Enhanced tracking
        self.save_all_expressions = save_all_expressions
        self.is_prefix = is_prefix
        self.custom_prompt = custom_prompt
        self.epoch_history = []
        self.best_expression = None
        self.best_r2 = -float('inf')
        self.best_epoch = -1

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model(model_path)

        # Build prompt
        if custom_prompt:
            self.prompt = custom_prompt
        else:
            self.prompt = self._build_prompt()
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors="pt")["input_ids"].to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Temperature
        self.temperature = 0.8

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {model_path}")

        # Check if it's a HuggingFace model ID or local path
        if "/" in model_path and not os.path.exists(model_path):
            # HuggingFace model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
        else:
            # Local model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Check for adapter_config.json (LoRA model)
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                base_model = AutoModelForCausalLM.from_pretrained(
                    "gpt2", torch_dtype=torch.float32
                ).to(self.device)
                self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float32
                ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.train()

    def _build_prompt(self) -> str:
        """Build unified prompt with all necessary operations."""
        # Get all unique operations from Nguyen benchmarks
        all_ops = ["*", "+", "-", "/", "sin", "cos", "tan", "exp", "log", "sqrt", "abs"]

        # Build variable list
        var_list = [f"x_{i+1}" for i in range(self.n_vars)]

        if self.is_prefix:
            # Prefix notation prompt
            prompt = f"vars: {', '.join(var_list)}\noper: {', '.join(all_ops)}\ncons: C\nexpr: "
        else:
            # JSON format for infix models
            prompt = json.dumps({
                "vars": var_list,
                "ops": all_ops,
                "cons": "C",
                "expr": ""
            })[:-2]  # Remove closing "}

        return prompt

    def extract_expression(self, text: str) -> str:
        """Extract expression from generated text."""
        if self.is_prefix:
            # Prefix format
            if "expr:" in text:
                text = text.split("expr:")[-1].strip()
            # Stop at newline or JSON artifacts
            if "\n" in text:
                text = text.split("\n")[0].strip()
            # Remove any trailing JSON artifacts
            for marker in ['"}"', '"}', '"cons"', '"vars"', '"ops"']:
                if marker in text:
                    text = text.split(marker)[0].strip()
            return text.strip()
        else:
            # JSON format
            if '"expr":' in text:
                start = text.find('"expr":') + len('"expr":')
                text = text[start:].strip()
                if text.startswith('"'):
                    text = text[1:]
                end = text.find('"')
                if end > 0:
                    text = text[:end]
            return text.strip()

    def compute_r2(self, expression_str: str) -> Tuple[float, bool, str]:
        """Compute R^2 score with error tracking."""
        error_msg = ""

        if not expression_str or expression_str.isspace():
            return -1.0, False, "Empty expression"

        if 'C' in expression_str:
            expression_str = expression_str.replace('C', '1')

        try:
            expr = Expression(expression_str, is_prefix=self.is_prefix)
            if not expr.is_valid_on_dataset(self.X):
                return -1.0, False, "Invalid on dataset"

            y_pred = expr.evaluate(self.X)
            if not np.all(np.isfinite(y_pred)):
                return -1.0, False, "Non-finite predictions"

            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

            if ss_tot == 0:
                return 0.0, True, ""

            r2 = 1 - (ss_res / ss_tot)
            return float(np.clip(r2, -1.0, 1.0)), True, ""
        except Exception as e:
            return -1.0, False, str(e)

    def generate_sample(self, max_new_tokens: int = 50) -> Dict:
        """Generate a single sample."""
        generated_ids = self.prompt_ids.clone()
        generated_tokens = []
        log_probs_list = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :] / self.temperature

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                token_log_prob = log_probs[0, next_token.item()].item()

                generated_tokens.append(next_token.item())
                log_probs_list.append(token_log_prob)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated_text = text[len(self.prompt):]

                # Stop at newline for prefix, closing brace for infix
                if self.is_prefix and ("\n" in generated_text or "vars:" in generated_text):
                    break
                if not self.is_prefix and '"}' in generated_text:
                    break

        # Decode and evaluate
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        expr_str = self.extract_expression(text)
        r2, is_valid, error_msg = self.compute_r2(expr_str)

        return {
            "text": text,
            "expression": expr_str,
            "r2": r2,
            "is_valid": is_valid,
            "error": error_msg,
            "tokens": generated_tokens,
            "log_probs": log_probs_list,
            "total_log_prob": sum(log_probs_list),
        }

    def train(self, epochs: int = 20, samples_per_group: int = 8, groups_per_epoch: int = 4):
        """Training loop with comprehensive tracking."""
        logger.info("Starting GRPO training")
        logger.info(f"Epochs: {epochs}, Groups per epoch: {groups_per_epoch}, Samples per group: {samples_per_group}")

        for epoch in range(epochs):
            epoch_data = {
                "epoch": epoch,
                "timestamp": datetime.datetime.now().isoformat(),
                "expressions": [],
                "metrics": {}
            }

            all_samples = []
            epoch_loss = 0

            for group_idx in range(groups_per_epoch):
                # Generate a group of samples
                group = []
                self.model.eval()

                for _ in range(samples_per_group):
                    sample = self.generate_sample()
                    group.append(sample)
                    all_samples.append(sample)

                    # Save expression data
                    if self.save_all_expressions:
                        epoch_data["expressions"].append({
                            "expression": sample["expression"],
                            "r2": sample["r2"],
                            "is_valid": sample["is_valid"],
                            "error": sample.get("error", ""),
                            "group": group_idx
                        })

                # Compute group-relative advantages
                group_rewards = [s["r2"] if s["is_valid"] else -1.0 for s in group]
                group_mean = np.mean(group_rewards)
                group_std = np.std(group_rewards) + 1e-8

                advantages = [(r - group_mean) / group_std for r in group_rewards]

                # Update policy using group
                self.model.train()
                group_loss = 0

                for sample, advantage in zip(group, advantages):
                    if not sample["is_valid"]:
                        continue

                    # Reconstruct sequence
                    prompt_len = len(self.prompt_ids[0])
                    token_ids = torch.cat([
                        self.prompt_ids[0],
                        torch.tensor(sample["tokens"], device=self.device)
                    ]).unsqueeze(0)

                    # Forward pass
                    outputs = self.model(token_ids)
                    logits = outputs.logits[:, prompt_len-1:-1, :]

                    # Compute new log probs
                    log_probs = F.log_softmax(logits / self.temperature, dim=-1)
                    token_tensor = torch.tensor(sample["tokens"], device=self.device).unsqueeze(0)
                    new_log_probs = log_probs.gather(2, token_tensor.unsqueeze(-1)).squeeze(-1)

                    # Policy gradient loss
                    loss = -new_log_probs.sum() * advantage

                    # Optional: Add entropy bonus
                    entropy = -(torch.exp(log_probs) * log_probs).sum()
                    loss = loss - 0.01 * entropy

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    group_loss += loss.item()

                epoch_loss += group_loss

            # Scheduler step
            self.scheduler.step()

            # Process epoch metrics
            valid_samples = [s for s in all_samples if s["is_valid"]]
            valid_r2s = [s["r2"] for s in valid_samples]

            # Find best in this epoch
            if valid_r2s:
                best_idx = np.argmax([s["r2"] if s["is_valid"] else -2 for s in all_samples])
                epoch_best = all_samples[best_idx]

                # Update global best
                if epoch_best["r2"] > self.best_r2:
                    self.best_r2 = epoch_best["r2"]
                    self.best_expression = epoch_best["expression"]
                    self.best_epoch = epoch
            else:
                epoch_best = {"r2": -1.0, "expression": ""}

            # Compute metrics
            epoch_data["metrics"] = {
                "valid_rate": len(valid_samples) / len(all_samples) if all_samples else 0,
                "mean_r2": np.mean(valid_r2s) if valid_r2s else -1.0,
                "max_r2": max(valid_r2s) if valid_r2s else -1.0,
                "min_r2": min(valid_r2s) if valid_r2s else -1.0,
                "std_r2": np.std(valid_r2s) if valid_r2s else 0.0,
                "best_expression": epoch_best["expression"],
                "best_r2": epoch_best["r2"],
                "unique_expressions": len(set(s["expression"] for s in all_samples if s["expression"])),
                "total_samples": len(all_samples),
                "loss": epoch_loss / max(len(valid_samples), 1)
            }

            # Store epoch data
            self.epoch_history.append(epoch_data)

            # Log progress
            logger.info(f"Epoch {epoch}: Valid={len(valid_samples)}/{len(all_samples)} "
                       f"Best R²={epoch_best['r2']:.4f} "
                       f"Mean R²={epoch_data['metrics']['mean_r2']:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

        # Final save
        self.save_results()

        return self.epoch_history

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save metrics
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "best_r2": self.best_r2,
                "best_expression": self.best_expression,
                "best_epoch": self.best_epoch
            }, f, indent=2)

    def save_results(self):
        """Save all results."""
        # Save full history
        with open(self.output_dir / "full_history.json", "w") as f:
            json.dump(self.epoch_history, f, indent=2)

        # Save summary
        summary = {
            "best_expression": self.best_expression,
            "best_r2": self.best_r2,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.epoch_history),
            "final_valid_rate": self.epoch_history[-1]["metrics"]["valid_rate"] if self.epoch_history else 0,
            "timestamp": datetime.datetime.now().isoformat()
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Best expression: {self.best_expression} (R²={self.best_r2:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--output_dir", default="./output/grpo_enhanced")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples_per_group", type=int, default=8)
    parser.add_argument("--groups_per_epoch", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--is_prefix", action="store_true", help="Model uses prefix notation")
    parser.add_argument("--custom_prompt", type=str, help="Custom unified prompt")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Initialize and train
    grpo = EnhancedGRPO(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        is_prefix=args.is_prefix,
        custom_prompt=args.custom_prompt,
        save_all_expressions=True
    )

    grpo.train(
        epochs=args.epochs,
        samples_per_group=args.samples_per_group,
        groups_per_epoch=args.groups_per_epoch
    )


if __name__ == "__main__":
    main()