#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for Symbolic Regression

Based on DeepSeek-R1 approach:
- Generate a group of N samples
- Compute advantages relative to group mean/std
- No external baseline needed

Comparison with REINFORCE:
- REINFORCE: advantage = reward - moving_average_baseline
- GRPO: advantage = (reward - group_mean) / group_std
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from copy import deepcopy

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
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class GRPO:
    """Group Relative Policy Optimization for symbolic regression."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/grpo",
        learning_rate: float = 5e-5,
        device: str = None,
        group_size: int = 8,  # Number of samples per group
        kl_coef: float = 0.01,  # KL penalty coefficient
        clip_range: float = 0.2,  # PPO-style clipping (optional)
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

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model(model_path)

        # Keep reference model for KL penalty
        self.ref_model = None  # Will be set after first update

        # Build prompt
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

        # Tracking
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []
        self.discovered_expressions: Dict[str, float] = {}

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            logger.info("Attempting to load as LoRA adapter...")
            base_model = AutoModelForCausalLM.from_pretrained("gpt2")
            if len(self.tokenizer) != base_model.config.vocab_size:
                base_model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized embeddings to {len(self.tokenizer)}")

            model_with_lora = PeftModel.from_pretrained(base_model, model_path)
            self.model = model_with_lora.merge_and_unload()
            logger.info("LoRA adapter loaded and merged successfully")
        except Exception as e:
            logger.info(f"LoRA load failed ({e}), loading as standalone model...")
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Add LoRA for training
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model = self.model.to(self.device)
        self.model.train()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model loaded with {trainable} trainable params")

    def _build_prompt(self, ops: list = None) -> str:
        """Build JSON format prompt."""
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]

        if ops is None:
            ops_list = ["+", "-", "*", "/", "sin", "cos", "sqrt", "log", "exp", "pow"]
        else:
            ops_list = ops

        prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": "C",
            "expr": ""
        })
        prompt = prompt[:-2]
        return prompt

    def extract_expression(self, text: str) -> str:
        """Extract expression from generated text."""
        try:
            eos_token = "<|endoftext|>"
            if eos_token in text:
                text = text[:text.index(eos_token)]

            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                for terminator in ['"}', '"']:
                    if terminator in remaining:
                        return remaining[:remaining.index(terminator)].strip()
                return remaining.strip()

            if '"expr": ' in text:
                start = text.index('"expr": ') + len('"expr": ')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                return remaining.strip(' "')

        except (ValueError, IndexError):
            pass

        if '"expr"' in text:
            return text.split('"expr"')[-1].strip(' ":{}')
        return text.strip()

    def compute_r2(self, expression_str: str) -> Tuple[float, bool]:
        """Compute R^2 score. Returns (score, is_valid)."""
        if not expression_str or expression_str.isspace():
            return -1.0, False

        if 'C' in expression_str:
            expression_str = expression_str.replace('C', '1')

        try:
            expr = Expression(expression_str, is_prefix=False)
            if not expr.is_valid_on_dataset(self.X):
                return -1.0, False

            y_pred = expr.evaluate(self.X)
            if not np.all(np.isfinite(y_pred)):
                return -1.0, False

            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

            if ss_tot == 0:
                return 0.0, True

            r2 = 1 - (ss_res / ss_tot)
            return float(np.clip(r2, -1.0, 1.0)), True
        except Exception:
            return -1.0, False

    def generate_group(
        self,
        temperature: float = 0.7,
        max_new_tokens: int = 50
    ) -> List[Dict]:
        """Generate a group of expressions."""
        results = []

        for _ in range(self.group_size):
            generated_ids = self.prompt_ids.clone()
            generated_tokens = []

            # Phase 1: Generate tokens without gradients
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(generated_ids)
                    logits = outputs.logits[:, -1, :] / temperature

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens.append(next_token.item())

                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    if '"}' in text[len(self.prompt):]:
                        break

            # Decode and extract expression
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)
            r2, is_valid = self.compute_r2(expr_str)

            # Phase 2: Efficient log prob computation
            if len(generated_tokens) > 0:
                full_ids = torch.cat([
                    self.prompt_ids,
                    torch.tensor([generated_tokens], device=self.device)
                ], dim=1)

                outputs = self.model(full_ids[:, :-1])
                logits = outputs.logits / temperature

                prompt_len = self.prompt_ids.shape[1]
                gen_logits = logits[:, prompt_len-1:, :]

                log_probs_all = F.log_softmax(gen_logits, dim=-1)

                target_tokens = torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
                selected_log_probs = log_probs_all.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                total_log_prob = selected_log_probs.sum()
            else:
                total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)

            results.append({
                "text": text,
                "expression": expr_str,
                "r2": r2,
                "is_valid": is_valid,
                "log_prob": total_log_prob,
                "generated_tokens": generated_tokens,
            })

            # Track best
            if is_valid:
                self.discovered_expressions[expr_str] = max(
                    self.discovered_expressions.get(expr_str, -np.inf), r2
                )

            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_expression = expr_str

            # Clear cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results

    def compute_group_advantages(self, results: List[Dict]) -> List[float]:
        """
        Compute GRPO advantages: (reward - mean) / std

        This is the key difference from REINFORCE:
        - REINFORCE uses external moving average baseline
        - GRPO uses within-group statistics
        """
        # Get rewards (R² values, with penalty for invalid)
        rewards = []
        for r in results:
            if r["is_valid"]:
                rewards.append(r["r2"])
            else:
                rewards.append(-0.1)  # Small penalty for invalid

        rewards = np.array(rewards)

        # Compute group statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Avoid division by zero
        if std_reward < 1e-8:
            std_reward = 1.0

        # Compute normalized advantages
        advantages = (rewards - mean_reward) / std_reward

        return advantages.tolist(), mean_reward, std_reward

    def train_step(self, num_groups: int = 4) -> dict:
        """
        Perform one GRPO training step.

        Args:
            num_groups: Number of groups to sample (effective batch = num_groups * group_size)
        """
        self.model.train()

        all_results = []
        all_advantages = []
        total_loss = 0.0

        self.optimizer.zero_grad()

        # Generate multiple groups
        for _ in range(num_groups):
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Generate a group of samples
            group_results = self.generate_group()
            all_results.extend(group_results)

            # Compute group-relative advantages
            advantages, group_mean, group_std = self.compute_group_advantages(group_results)
            all_advantages.extend(advantages)

            # Compute loss for this group
            group_loss = torch.tensor(0.0, device=self.device)
            valid_count = 0

            for result, advantage in zip(group_results, advantages):
                if result["is_valid"]:
                    # Policy gradient loss with advantage
                    group_loss = group_loss - result["log_prob"] * advantage
                    valid_count += 1

            if valid_count > 0:
                group_loss = group_loss / valid_count
                group_loss = group_loss / num_groups  # Scale for accumulation
                group_loss.backward()
                total_loss += group_loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update
        self.optimizer.step()
        self.scheduler.step()

        # Statistics
        r2_values = [r["r2"] for r in all_results]
        valid_mask = [r["is_valid"] for r in all_results]
        valid_r2 = [r2 for r2, v in zip(r2_values, valid_mask) if v]

        return {
            "valid_count": int(sum(valid_mask)),
            "total_count": len(all_results),
            "valid_rate": sum(valid_mask) / len(all_results),
            "mean_r2": float(np.mean(valid_r2)) if valid_r2 else 0.0,
            "max_r2": float(max(r2_values)),
            "mean_advantage": float(np.mean(all_advantages)),
            "std_advantage": float(np.std(all_advantages)),
            "loss": total_loss,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def run(
        self,
        epochs: int = 50,
        num_groups: int = 4,
        target_r2: float = 0.99,
        patience: int = 20,
    ) -> dict:
        """Run GRPO training."""
        logger.info("=" * 60)
        logger.info("GRPO SYMBOLIC REGRESSION")
        logger.info("=" * 60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Group size: {self.group_size}")
        logger.info(f"Num groups: {num_groups}")
        logger.info(f"Effective batch: {self.group_size * num_groups}")
        logger.info(f"Target R^2: {target_r2}")
        logger.info("=" * 60)

        no_improvement_count = 0
        best_r2_at_start = self.best_r2

        for epoch in range(1, epochs + 1):
            stats = self.train_step(num_groups)
            self.history.append({
                "epoch": epoch,
                **stats,
                "best_r2": self.best_r2,
            })

            logger.info(
                f"Epoch {epoch:3d} | "
                f"Valid: {stats['valid_count']}/{stats['total_count']} | "
                f"Mean R²: {stats['mean_r2']:.4f} | "
                f"Best: {self.best_r2:.4f} | "
                f"Adv μ: {stats['mean_advantage']:.3f} σ: {stats['std_advantage']:.3f} | "
                f"LR: {stats['lr']:.2e}"
            )

            # Check for target
            if self.best_r2 >= target_r2:
                logger.info(f"Target R^2 {target_r2} reached at epoch {epoch}!")
                break

            # Early stopping
            if self.best_r2 > best_r2_at_start:
                best_r2_at_start = self.best_r2
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                logger.info(f"No improvement for {patience} epochs. Early stopping.")
                break

        # Final results
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best R^2: {self.best_r2:.4f}")
        logger.info(f"Best expression: {self.best_expression}")
        logger.info(f"Unique expressions discovered: {len(self.discovered_expressions)}")

        # Top expressions
        top_exprs = sorted(
            self.discovered_expressions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        logger.info("Top 5 expressions:")
        for expr, r2 in top_exprs:
            logger.info(f"  R²={r2:.4f}: {expr}")

        # Save results
        results = {
            "algorithm": "GRPO",
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "history": self.history,
            "discovered_expressions": dict(list(self.discovered_expressions.items())[:100]),
            "config": {
                "group_size": self.group_size,
                "num_groups": num_groups,
                "learning_rate": self.learning_rate,
                "kl_coef": self.kl_coef,
            }
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"results_grpo_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="GRPO for Symbolic Regression")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/grpo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--target_r2", type=float, default=0.99)
    args = parser.parse_args()

    # Load dataset
    import pandas as pd
    df = pd.read_csv(args.dataset)

    x_cols = [c for c in df.columns if c.startswith('x_')]
    X = df[x_cols].values
    y = df['y'].values

    logger.info(f"Loaded dataset: {args.dataset}")
    logger.info(f"  Samples: {len(df)}, Variables: {len(x_cols)}")

    # Create GRPO trainer
    grpo = GRPO(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        group_size=args.group_size,
    )

    # Run training
    results = grpo.run(
        epochs=args.epochs,
        num_groups=args.num_groups,
        target_r2=args.target_r2,
    )


if __name__ == "__main__":
    main()
