#!/usr/bin/env python3
"""
Enhanced PPO for Symbolic Regression with Epoch Tracking
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
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

from expression import Expression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class EnhancedPPOSymbolic:
    """Enhanced PPO with comprehensive tracking."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/ppo_enhanced",
        learning_rate: float = 3e-5,
        device: str = None,
        batch_size: int = 16,
        # PPO hyperparameters
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        max_kl: float = 0.05,
        # Enhanced tracking
        save_all_expressions: bool = True,
        is_prefix: bool = True,  # Whether model uses prefix notation
        custom_prompt: str = None,  # Allow custom unified prompt
    ):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.max_kl = max_kl

        # Enhanced tracking
        self.save_all_expressions = save_all_expressions
        self.is_prefix = is_prefix
        self.custom_prompt = custom_prompt
        self.epoch_history = []  # Store all epoch data
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
            weight_decay=0.01,
            eps=1e-5,
        )

        # Temperature for sampling
        self.temperature = 0.8

        # Moving average baseline
        self.baseline_ema = 0.0
        self.baseline_alpha = 0.9

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

    def shape_reward(self, r2: float, is_valid: bool) -> float:
        """Shape reward for better learning signal."""
        if not is_valid:
            return -0.1

        if r2 >= 0.99:
            return 2.0
        elif r2 >= 0.9:
            return r2 * 1.5
        elif r2 >= 0.5:
            return r2 * 1.2
        elif r2 >= 0:
            return r2
        else:
            return r2 * 0.5

    def collect_rollouts(self, num_samples: int, max_new_tokens: int = 50) -> List[Dict]:
        """Collect rollouts with comprehensive tracking."""
        rollouts = []

        self.model.eval()

        for _ in range(num_samples):
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
                    if not self.is_prefix and '"}' in text[len(self.prompt):]:
                        break

            # Decode and evaluate
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)
            r2, is_valid, error_msg = self.compute_r2(expr_str)
            reward = self.shape_reward(r2, is_valid)

            # Store rollout with enhanced info
            rollouts.append({
                "text": text,
                "expression": expr_str,
                "r2": r2,
                "is_valid": is_valid,
                "error": error_msg,
                "reward": reward,
                "tokens": generated_tokens,
                "old_log_probs": log_probs_list,
                "total_old_log_prob": sum(log_probs_list),
            })

        return rollouts

    def train(self, epochs: int = 20, samples_per_epoch: int = 32):
        """Training loop with comprehensive tracking."""
        logger.info("Starting PPO training")
        logger.info(f"Epochs: {epochs}, Samples per epoch: {samples_per_epoch}")

        for epoch in range(epochs):
            epoch_data = {
                "epoch": epoch,
                "timestamp": datetime.datetime.now().isoformat(),
                "expressions": [],
                "metrics": {}
            }

            # Collect rollouts
            rollouts = self.collect_rollouts(samples_per_epoch)

            # Save all expressions if requested
            if self.save_all_expressions:
                for r in rollouts:
                    epoch_data["expressions"].append({
                        "expression": r["expression"],
                        "r2": r["r2"],
                        "is_valid": r["is_valid"],
                        "error": r.get("error", ""),
                        "reward": r["reward"]
                    })

            # Process rollouts and compute metrics
            valid_count = sum(1 for r in rollouts if r["is_valid"])
            valid_r2s = [r["r2"] for r in rollouts if r["is_valid"]]

            # Find best in this epoch
            if valid_r2s:
                best_idx = np.argmax([r["r2"] if r["is_valid"] else -2 for r in rollouts])
                epoch_best = rollouts[best_idx]

                # Update global best
                if epoch_best["r2"] > self.best_r2:
                    self.best_r2 = epoch_best["r2"]
                    self.best_expression = epoch_best["expression"]
                    self.best_epoch = epoch
            else:
                epoch_best = {"r2": -1.0, "expression": ""}

            # Compute metrics
            epoch_data["metrics"] = {
                "valid_rate": valid_count / len(rollouts),
                "mean_r2": np.mean(valid_r2s) if valid_r2s else -1.0,
                "max_r2": max(valid_r2s) if valid_r2s else -1.0,
                "min_r2": min(valid_r2s) if valid_r2s else -1.0,
                "std_r2": np.std(valid_r2s) if valid_r2s else 0.0,
                "best_expression": epoch_best["expression"],
                "best_r2": epoch_best["r2"],
                "unique_expressions": len(set(r["expression"] for r in rollouts if r["expression"])),
                "total_samples": len(rollouts)
            }

            # PPO update
            loss = self.ppo_update(rollouts)
            epoch_data["metrics"]["loss"] = float(loss)

            # Store epoch data
            self.epoch_history.append(epoch_data)

            # Log progress
            logger.info(f"Epoch {epoch}: Valid={valid_count}/{len(rollouts)} "
                       f"Best R²={epoch_best['r2']:.4f} "
                       f"Loss={loss:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

        # Final save
        self.save_results()

        return self.epoch_history

    def ppo_update(self, rollouts: List[Dict]) -> float:
        """PPO policy update."""
        # Compute advantages
        rewards = [r["reward"] for r in rollouts]
        self.baseline_ema = self.baseline_alpha * self.baseline_ema + \
                           (1 - self.baseline_alpha) * np.mean(rewards)
        advantages = [r - self.baseline_ema for r in rewards]

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # PPO optimization epochs
        total_loss = 0
        self.model.train()

        for ppo_epoch in range(self.ppo_epochs):
            epoch_loss = 0

            for rollout, advantage in zip(rollouts, advantages):
                # Skip invalid expressions in update
                if not rollout["is_valid"]:
                    continue

                # Reconstruct sequence
                prompt_len = len(self.prompt_ids[0])
                token_ids = torch.cat([
                    self.prompt_ids[0],
                    torch.tensor(rollout["tokens"], device=self.device)
                ]).unsqueeze(0)

                # Forward pass
                outputs = self.model(token_ids)
                logits = outputs.logits[:, prompt_len-1:-1, :]

                # Compute new log probs
                log_probs = F.log_softmax(logits / self.temperature, dim=-1)
                token_tensor = torch.tensor(rollout["tokens"], device=self.device).unsqueeze(0)
                new_log_probs = log_probs.gather(2, token_tensor.unsqueeze(-1)).squeeze(-1)

                # Compute ratio
                old_log_probs = torch.tensor(rollout["old_log_probs"], device=self.device)
                ratio = torch.exp(new_log_probs.sum() - old_log_probs.sum())

                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                loss = -torch.min(surr1, surr2)

                # Add entropy bonus
                entropy = -(torch.exp(log_probs) * log_probs).sum()
                loss = loss - self.entropy_coef * entropy

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            # Check KL divergence for early stopping
            # (simplified - in practice would compute proper KL)
            if ppo_epoch > 0 and epoch_loss > total_loss * 2:
                logger.info(f"Early stopping PPO at epoch {ppo_epoch}")
                break

            total_loss = epoch_loss

        return total_loss / max(len([r for r in rollouts if r["is_valid"]]), 1)

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
    parser.add_argument("--output_dir", default="./output/ppo_enhanced")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples_per_epoch", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--is_prefix", action="store_true", help="Model uses prefix notation")
    parser.add_argument("--custom_prompt", type=str, help="Custom unified prompt")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Initialize and train
    ppo = EnhancedPPOSymbolic(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        is_prefix=args.is_prefix,
        custom_prompt=args.custom_prompt,
        save_all_expressions=True
    )

    ppo.train(epochs=args.epochs, samples_per_epoch=args.samples_per_epoch)


if __name__ == "__main__":
    main()