#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) for Symbolic Regression

Key features:
1. Clipped surrogate objective to prevent too large policy updates
2. Multiple optimization epochs per batch of samples
3. Advantage estimation with EMA baseline
4. KL divergence monitoring for early stopping
5. Entropy bonus for exploration
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


class PPOSymbolic:
    """PPO for symbolic regression."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/ppo",
        learning_rate: float = 3e-5,
        device: str = None,
        batch_size: int = 16,
        # PPO hyperparameters
        clip_epsilon: float = 0.2,  # Clipping parameter
        ppo_epochs: int = 4,  # Optimization epochs per batch
        entropy_coef: float = 0.01,  # Entropy bonus coefficient
        max_kl: float = 0.05,  # Max KL for early stopping within PPO epochs
        gae_lambda: float = 0.95,  # GAE lambda (not used here, but kept for reference)
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
            weight_decay=0.01,
            eps=1e-5,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # Tracking
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []
        self.discovered_expressions: Dict[str, float] = {}

        # EMA baseline for advantage estimation
        self.baseline = 0.0
        self.baseline_decay = 0.95

        # Temperature for sampling
        self.temperature = 0.7

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
            return -0.1  # Small penalty

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
        """
        Collect rollouts (samples) from current policy.
        Store both the samples and their log probabilities under current policy.
        """
        rollouts = []

        self.model.eval()  # Eval mode for sampling

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
                    if '"}' in text[len(self.prompt):]:
                        break

            # Decode and evaluate
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            expr_str = self.extract_expression(text)
            r2, is_valid = self.compute_r2(expr_str)
            reward = self.shape_reward(r2, is_valid)

            # Store rollout
            rollouts.append({
                "text": text,
                "expression": expr_str,
                "r2": r2,
                "is_valid": is_valid,
                "reward": reward,
                "tokens": generated_tokens,
                "old_log_probs": log_probs_list,  # Store for PPO ratio computation
                "total_old_log_prob": sum(log_probs_list),
            })

            # Track best
            if is_valid:
                self.discovered_expressions[expr_str] = max(
                    self.discovered_expressions.get(expr_str, -np.inf), r2
                )

            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_expression = expr_str

        return rollouts

    def compute_advantages(self, rollouts: List[Dict]) -> List[float]:
        """
        Compute advantages using EMA baseline.
        For simplicity, we use a simple baseline approach instead of GAE.
        """
        valid_rewards = [r["reward"] for r in rollouts if r["is_valid"]]

        if valid_rewards:
            mean_reward = np.mean(valid_rewards)
            # Update EMA baseline
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_reward

        advantages = []
        for r in rollouts:
            if r["is_valid"]:
                adv = r["reward"] - self.baseline
            else:
                adv = -0.3  # Fixed small penalty for invalid
            advantages.append(adv)

        # Normalize advantages
        adv_array = np.array(advantages)
        adv_mean = np.mean(adv_array)
        adv_std = np.std(adv_array)
        if adv_std > 1e-8:
            advantages = ((adv_array - adv_mean) / adv_std).tolist()

        return advantages

    def ppo_update(self, rollouts: List[Dict], advantages: List[float]) -> Dict:
        """
        Perform PPO update with multiple epochs.

        Key PPO components:
        1. Ratio: π_new(a|s) / π_old(a|s)
        2. Clipped objective: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        3. Multiple optimization epochs
        4. Early stopping based on KL divergence
        """
        self.model.train()

        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        early_stopped = False

        # Filter valid rollouts
        valid_indices = [i for i, r in enumerate(rollouts) if r["is_valid"] and len(r["tokens"]) > 0]

        if not valid_indices:
            return {
                "policy_loss": 0.0,
                "entropy_loss": 0.0,
                "kl_divergence": 0.0,
                "early_stopped": False,
                "ppo_epochs_used": 0,
            }

        # PPO optimization epochs
        for ppo_epoch in range(self.ppo_epochs):
            epoch_kl = 0.0
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            valid_count = 0

            self.optimizer.zero_grad()

            for idx in valid_indices:
                rollout = rollouts[idx]
                advantage = advantages[idx]

                tokens = rollout["tokens"]
                old_log_probs = rollout["old_log_probs"]

                if len(tokens) == 0:
                    continue

                # Compute new log probs
                full_ids = torch.cat([
                    self.prompt_ids,
                    torch.tensor([tokens], device=self.device)
                ], dim=1)

                outputs = self.model(full_ids[:, :-1])
                logits = outputs.logits / self.temperature

                prompt_len = self.prompt_ids.shape[1]
                gen_logits = logits[:, prompt_len-1:, :]

                new_log_probs_all = F.log_softmax(gen_logits, dim=-1)
                new_probs_all = F.softmax(gen_logits, dim=-1)

                target_tokens = torch.tensor(tokens, device=self.device).unsqueeze(0)
                new_log_probs_selected = new_log_probs_all.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

                # Compute ratio for each token
                old_log_probs_tensor = torch.tensor(old_log_probs, device=self.device).unsqueeze(0)

                # Log ratio = log(π_new) - log(π_old)
                log_ratio = new_log_probs_selected - old_log_probs_tensor
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence
                kl = (ratio - 1 - log_ratio).mean()
                epoch_kl += kl.item()

                # PPO clipped objective (per token, then averaged)
                advantage_tensor = torch.tensor(advantage, device=self.device)

                # Unclipped objective
                surr1 = ratio * advantage_tensor

                # Clipped objective
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                surr2 = clipped_ratio * advantage_tensor

                # PPO loss: negative because we want to maximize
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_per_token = -(new_probs_all * new_log_probs_all).sum(dim=-1)
                entropy_loss = -entropy_per_token.mean()

                # Combined loss
                loss = policy_loss + self.entropy_coef * entropy_loss
                loss = loss / len(valid_indices)  # Normalize by batch size
                loss.backward()

                epoch_policy_loss += policy_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                valid_count += 1

            if valid_count > 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                avg_kl = epoch_kl / valid_count
                total_kl += avg_kl
                total_policy_loss += epoch_policy_loss / valid_count
                total_entropy_loss += epoch_entropy_loss / valid_count
                num_updates += 1

                # Early stopping if KL is too large
                if avg_kl > self.max_kl:
                    early_stopped = True
                    break

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "entropy_loss": total_entropy_loss / max(num_updates, 1),
            "kl_divergence": total_kl / max(num_updates, 1),
            "early_stopped": early_stopped,
            "ppo_epochs_used": num_updates,
        }

    def train_step(self) -> dict:
        """Perform one training step: collect rollouts + PPO update."""
        # Collect rollouts
        rollouts = self.collect_rollouts(self.batch_size)

        # Compute advantages
        advantages = self.compute_advantages(rollouts)

        # PPO update
        ppo_stats = self.ppo_update(rollouts, advantages)

        # Update learning rate
        self.scheduler.step()

        # Statistics
        r2_values = [r["r2"] for r in rollouts]
        valid_mask = [r["is_valid"] for r in rollouts]
        valid_r2 = [r2 for r2, v in zip(r2_values, valid_mask) if v]

        return {
            "valid_count": int(sum(valid_mask)),
            "total_count": len(rollouts),
            "valid_rate": sum(valid_mask) / len(rollouts) if rollouts else 0,
            "mean_r2": float(np.mean(valid_r2)) if valid_r2 else 0.0,
            "max_r2": float(max(r2_values)) if r2_values else 0.0,
            "baseline": self.baseline,
            "lr": self.scheduler.get_last_lr()[0],
            **ppo_stats,
        }

    def run(
        self,
        epochs: int = 50,
        target_r2: float = 0.99,
        patience: int = 20,
    ) -> dict:
        """Run PPO training."""
        logger.info("=" * 60)
        logger.info("PPO SYMBOLIC REGRESSION")
        logger.info("=" * 60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"PPO epochs per batch: {self.ppo_epochs}")
        logger.info(f"Clip epsilon: {self.clip_epsilon}")
        logger.info(f"Entropy coef: {self.entropy_coef}")
        logger.info(f"Max KL: {self.max_kl}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Target R^2: {target_r2}")
        logger.info("=" * 60)

        no_improvement_count = 0
        best_r2_at_start = self.best_r2

        for epoch in range(1, epochs + 1):
            stats = self.train_step()
            self.history.append({
                "epoch": epoch,
                **stats,
                "best_r2": self.best_r2,
            })

            kl_str = f"KL: {stats['kl_divergence']:.4f}" if stats['kl_divergence'] > 0 else "KL: N/A"
            es_str = " (ES)" if stats['early_stopped'] else ""

            logger.info(
                f"Epoch {epoch:3d} | "
                f"Valid: {stats['valid_count']}/{stats['total_count']} | "
                f"Mean R²: {stats['mean_r2']:.4f} | "
                f"Best: {self.best_r2:.4f} | "
                f"{kl_str}{es_str} | "
                f"PPO: {stats['ppo_epochs_used']} | "
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
            "algorithm": "PPO",
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "history": self.history,
            "discovered_expressions": dict(list(self.discovered_expressions.items())[:100]),
            "config": {
                "batch_size": self.batch_size,
                "ppo_epochs": self.ppo_epochs,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coef": self.entropy_coef,
                "max_kl": self.max_kl,
                "learning_rate": self.learning_rate,
            }
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"results_ppo_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="PPO for Symbolic Regression")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/ppo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_kl", type=float, default=0.05)
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

    # Create trainer
    ppo = PPOSymbolic(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        max_kl=args.max_kl,
    )

    # Run training
    results = ppo.run(
        epochs=args.epochs,
        target_r2=args.target_r2,
    )


if __name__ == "__main__":
    main()
