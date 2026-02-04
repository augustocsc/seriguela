#!/usr/bin/env python3
"""
Improved GRPO (Group Relative Policy Optimization) for Symbolic Regression

Improvements over basic GRPO:
1. Filter invalid expressions before computing group statistics
2. Reward shaping with softer penalties
3. Hybrid baseline: group stats + exponential moving average
4. Entropy bonus for exploration
5. Advantage clipping to prevent extreme updates
6. Minimum valid ratio check before updates
7. Temperature annealing for better exploration/exploitation
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


class ImprovedGRPO:
    """Improved GRPO for symbolic regression."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/grpo",
        learning_rate: float = 5e-5,
        device: str = None,
        group_size: int = 16,  # Larger groups for better statistics
        entropy_coef: float = 0.01,
        advantage_clip: float = 2.0,  # Clip extreme advantages
        min_valid_ratio: float = 0.2,  # Minimum valid expressions to update
    ):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.entropy_coef = entropy_coef
        self.advantage_clip = advantage_clip
        self.min_valid_ratio = min_valid_ratio

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model(model_path)

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

        # Hybrid baseline: EMA of valid rewards
        self.ema_baseline = 0.0
        self.ema_decay = 0.9
        self.reward_buffer = deque(maxlen=100)

        # Temperature annealing
        self.initial_temp = 0.8
        self.min_temp = 0.5
        self.current_temp = self.initial_temp

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
        """Compute R^2 score."""
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

    def shape_reward(self, r2: float, is_valid: bool) -> float:
        """Shape reward for better learning signal."""
        if not is_valid:
            return -0.1  # Small penalty, not -1.0

        # Bonus for high R²
        if r2 >= 0.99:
            return 2.0  # Big bonus for near-perfect
        elif r2 >= 0.9:
            return r2 * 1.5
        elif r2 >= 0.5:
            return r2 * 1.2
        elif r2 >= 0:
            return r2
        else:
            return r2 * 0.5  # Reduce negative penalty

    def generate_group(self, max_new_tokens: int = 50) -> List[Dict]:
        """Generate a group of expressions."""
        results = []

        for _ in range(self.group_size):
            generated_ids = self.prompt_ids.clone()
            generated_tokens = []

            # Phase 1: Generate tokens
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(generated_ids)
                    logits = outputs.logits[:, -1, :] / self.current_temp

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens.append(next_token.item())

                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    if '"}' in text[len(self.prompt):]:
                        break

            # Decode and evaluate
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)
            r2, is_valid = self.compute_r2(expr_str)
            reward = self.shape_reward(r2, is_valid)

            # Phase 2: Compute log probs with gradients
            if len(generated_tokens) > 0:
                full_ids = torch.cat([
                    self.prompt_ids,
                    torch.tensor([generated_tokens], device=self.device)
                ], dim=1)

                outputs = self.model(full_ids[:, :-1])
                logits = outputs.logits / self.current_temp

                prompt_len = self.prompt_ids.shape[1]
                gen_logits = logits[:, prompt_len-1:, :]

                log_probs_all = F.log_softmax(gen_logits, dim=-1)
                probs_all = F.softmax(gen_logits, dim=-1)

                target_tokens = torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
                selected_log_probs = log_probs_all.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                total_log_prob = selected_log_probs.sum()

                # Entropy for exploration
                entropy_per_pos = -(probs_all * log_probs_all).sum(dim=-1)
                total_entropy = entropy_per_pos.mean()
            else:
                total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)
                total_entropy = torch.tensor(0.0, device=self.device)

            results.append({
                "text": text,
                "expression": expr_str,
                "r2": r2,
                "is_valid": is_valid,
                "reward": reward,
                "log_prob": total_log_prob,
                "entropy": total_entropy,
            })

            # Track best
            if is_valid:
                self.discovered_expressions[expr_str] = max(
                    self.discovered_expressions.get(expr_str, -np.inf), r2
                )
                self.reward_buffer.append(reward)

            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_expression = expr_str

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results

    def compute_advantages(self, results: List[Dict]) -> Tuple[List[float], dict]:
        """
        Compute improved GRPO advantages.

        Key improvement: Only use VALID expressions for group statistics.
        Invalid expressions get a fixed small negative advantage.
        """
        valid_results = [r for r in results if r["is_valid"]]
        valid_rewards = [r["reward"] for r in valid_results]

        stats = {
            "valid_count": len(valid_results),
            "total_count": len(results),
            "valid_ratio": len(valid_results) / len(results),
        }

        # If too few valid expressions, use EMA baseline only
        if len(valid_rewards) < 2:
            advantages = []
            for r in results:
                if r["is_valid"]:
                    adv = r["reward"] - self.ema_baseline
                else:
                    adv = -0.5  # Fixed penalty for invalid
                advantages.append(adv)
            stats["method"] = "ema_only"
            return advantages, stats

        # Compute group statistics from valid expressions only
        group_mean = np.mean(valid_rewards)
        group_std = np.std(valid_rewards)

        # Update EMA baseline
        self.ema_baseline = self.ema_decay * self.ema_baseline + (1 - self.ema_decay) * group_mean

        # Hybrid baseline: combine group mean with EMA
        hybrid_baseline = 0.7 * group_mean + 0.3 * self.ema_baseline

        # Avoid division by zero
        if group_std < 1e-8:
            group_std = 1.0

        # Compute advantages
        advantages = []
        for r in results:
            if r["is_valid"]:
                # Normalized advantage for valid expressions
                adv = (r["reward"] - hybrid_baseline) / group_std
                # Clip to prevent extreme updates
                adv = np.clip(adv, -self.advantage_clip, self.advantage_clip)
            else:
                # Small fixed penalty for invalid (doesn't pollute group stats)
                adv = -0.3
            advantages.append(adv)

        stats["method"] = "hybrid"
        stats["group_mean"] = group_mean
        stats["group_std"] = group_std
        stats["ema_baseline"] = self.ema_baseline

        return advantages, stats

    def train_step(self, num_groups: int = 2) -> dict:
        """Perform one training step."""
        self.model.train()

        all_results = []
        all_advantages = []
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        skipped_groups = 0

        self.optimizer.zero_grad()

        for _ in range(num_groups):
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Generate group
            group_results = self.generate_group()
            all_results.extend(group_results)

            # Compute advantages
            advantages, adv_stats = self.compute_advantages(group_results)
            all_advantages.extend(advantages)

            # Skip update if too few valid expressions
            if adv_stats["valid_ratio"] < self.min_valid_ratio:
                skipped_groups += 1
                continue

            # Compute loss
            policy_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            valid_count = 0

            for result, advantage in zip(group_results, advantages):
                if result["is_valid"] and advantage != 0:
                    policy_loss = policy_loss - result["log_prob"] * advantage
                    entropy_loss = entropy_loss - result["entropy"]
                    valid_count += 1

            if valid_count > 0:
                policy_loss = policy_loss / valid_count
                entropy_loss = entropy_loss / valid_count

                # Combined loss
                loss = policy_loss + self.entropy_coef * entropy_loss
                loss = loss / num_groups
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()

        # Only update if we had valid groups
        if skipped_groups < num_groups:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

        # Statistics
        r2_values = [r["r2"] for r in all_results]
        valid_mask = [r["is_valid"] for r in all_results]
        valid_r2 = [r2 for r2, v in zip(r2_values, valid_mask) if v]

        return {
            "valid_count": int(sum(valid_mask)),
            "total_count": len(all_results),
            "valid_rate": sum(valid_mask) / len(all_results) if all_results else 0,
            "mean_r2": float(np.mean(valid_r2)) if valid_r2 else 0.0,
            "max_r2": float(max(r2_values)) if r2_values else 0.0,
            "mean_advantage": float(np.mean(all_advantages)) if all_advantages else 0.0,
            "ema_baseline": self.ema_baseline,
            "policy_loss": total_policy_loss / max(num_groups - skipped_groups, 1),
            "entropy_loss": total_entropy_loss / max(num_groups - skipped_groups, 1),
            "lr": self.scheduler.get_last_lr()[0],
            "temperature": self.current_temp,
            "skipped_groups": skipped_groups,
        }

    def anneal_temperature(self, epoch: int, total_epochs: int):
        """Anneal temperature from initial to minimum."""
        progress = epoch / total_epochs
        self.current_temp = self.initial_temp - progress * (self.initial_temp - self.min_temp)

    def run(
        self,
        epochs: int = 50,
        num_groups: int = 2,
        target_r2: float = 0.99,
        patience: int = 20,
    ) -> dict:
        """Run improved GRPO training."""
        logger.info("=" * 60)
        logger.info("IMPROVED GRPO SYMBOLIC REGRESSION")
        logger.info("=" * 60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Group size: {self.group_size}")
        logger.info(f"Num groups: {num_groups}")
        logger.info(f"Effective batch: {self.group_size * num_groups}")
        logger.info(f"Entropy coef: {self.entropy_coef}")
        logger.info(f"Advantage clip: {self.advantage_clip}")
        logger.info(f"Min valid ratio: {self.min_valid_ratio}")
        logger.info(f"Target R^2: {target_r2}")
        logger.info("=" * 60)

        no_improvement_count = 0
        best_r2_at_start = self.best_r2

        for epoch in range(1, epochs + 1):
            # Anneal temperature
            self.anneal_temperature(epoch, epochs)

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
                f"EMA: {stats['ema_baseline']:.3f} | "
                f"Temp: {stats['temperature']:.2f} | "
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
            "algorithm": "ImprovedGRPO",
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "history": self.history,
            "discovered_expressions": dict(list(self.discovered_expressions.items())[:100]),
            "config": {
                "group_size": self.group_size,
                "num_groups": num_groups,
                "learning_rate": self.learning_rate,
                "entropy_coef": self.entropy_coef,
                "advantage_clip": self.advantage_clip,
                "min_valid_ratio": self.min_valid_ratio,
            }
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"results_grpo_improved_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Improved GRPO for Symbolic Regression")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/grpo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--num_groups", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--target_r2", type=float, default=0.99)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    args = parser.parse_args()

    # Load dataset
    import pandas as pd
    df = pd.read_csv(args.dataset)

    x_cols = [c for c in df.columns if c.startswith('x_')]
    X = df[x_cols].values
    y = df['y'].values

    logger.info(f"Loaded dataset: {args.dataset}")
    logger.info(f"  Samples: {len(df)}, Variables: {len(x_cols)}")

    # Create trainer
    grpo = ImprovedGRPO(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        group_size=args.group_size,
        entropy_coef=args.entropy_coef,
    )

    # Run training
    results = grpo.run(
        epochs=args.epochs,
        num_groups=args.num_groups,
        target_r2=args.target_r2,
    )


if __name__ == "__main__":
    main()
