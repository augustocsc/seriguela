#!/usr/bin/env python3
"""
Improved REINFORCE for Symbolic Regression

Improvements over basic REINFORCE:
1. Larger batch size with gradient accumulation
2. Entropy bonus for exploration
3. Better baseline (exponential moving average with warmup)
4. Reward shaping (softer penalty for invalid expressions)
5. Best-of-N sampling to find good expressions faster
6. Learning rate scheduling
7. Gradient clipping
8. Detailed logging per epoch
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Tuple, Dict
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


class ImprovedREINFORCE:
    """Improved REINFORCE algorithm for symbolic regression."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/reinforce",
        learning_rate: float = 5e-5,
        device: str = None,
        entropy_coef: float = 0.01,
        baseline_decay: float = 0.95,
    ):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.baseline_decay = baseline_decay

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

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Tracking
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []

        # Improved baseline: use recent rewards buffer
        self.reward_buffer = deque(maxlen=50)
        self.baseline = 0.0

        # Track all discovered expressions
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

        # Add LoRA for training (reduced for memory efficiency)
        lora_config = LoraConfig(
            r=8,  # Reduced for memory
            lora_alpha=16,
            target_modules=["c_attn"],  # Only attention
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

        # Default operators - includes all operators from training data
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
        prompt = prompt[:-2]  # Remove closing "}
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

    def shape_reward(self, r2: float, is_valid: bool) -> float:
        """Shape reward to encourage exploration."""
        if not is_valid:
            return -0.1  # Small penalty instead of -1.0

        # Transform R^2 to encourage improvement
        if r2 < 0:
            return r2 * 0.5  # Reduce negative penalty
        elif r2 < 0.5:
            return r2
        elif r2 < 0.9:
            return r2 * 1.5  # Bonus for good expressions
        else:
            return r2 * 2.0  # Big bonus for great expressions

    def generate_batch(
        self,
        batch_size: int,
        temperature: float = 0.7,
        max_new_tokens: int = 50
    ) -> List[Dict]:
        """Generate a batch of expressions with log probabilities."""
        results = []

        for _ in range(batch_size):
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
            reward = self.shape_reward(r2, is_valid)

            # Phase 2: Efficient log prob computation using full sequence
            if len(generated_tokens) > 0:
                # Build target sequence
                full_ids = torch.cat([
                    self.prompt_ids,
                    torch.tensor([generated_tokens], device=self.device)
                ], dim=1)

                # Single forward pass for all positions
                outputs = self.model(full_ids[:, :-1])  # Input all but last
                logits = outputs.logits / temperature

                # Get log probs for generated portion
                prompt_len = self.prompt_ids.shape[1]
                gen_logits = logits[:, prompt_len-1:, :]  # Logits predicting generated tokens

                log_probs_all = F.log_softmax(gen_logits, dim=-1)
                probs_all = F.softmax(gen_logits, dim=-1)

                # Gather log probs of selected tokens
                target_tokens = torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
                selected_log_probs = log_probs_all.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                total_log_prob = selected_log_probs.sum()

                # Compute mean entropy
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

            # Track best and discovered expressions
            if is_valid:
                self.discovered_expressions[expr_str] = max(
                    self.discovered_expressions.get(expr_str, -np.inf), r2
                )

            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_expression = expr_str

        return results

    def update_baseline(self, rewards: List[float]):
        """Update baseline using reward buffer."""
        valid_rewards = [r for r in rewards if r > -0.5]
        self.reward_buffer.extend(valid_rewards)

        if len(self.reward_buffer) > 0:
            # Use median for robustness
            self.baseline = self.baseline_decay * self.baseline + \
                           (1 - self.baseline_decay) * np.median(list(self.reward_buffer))

    def train_step(self, batch_size: int = 8, grad_accum_steps: int = 4) -> dict:
        """Perform one training step with gradient accumulation."""
        self.model.train()

        all_results = []
        total_policy_loss = 0.0
        total_entropy_loss = 0.0

        self.optimizer.zero_grad()

        effective_batch = batch_size * grad_accum_steps

        for accum_step in range(grad_accum_steps):

            results = self.generate_batch(batch_size)
            all_results.extend(results)

            # Compute losses for this mini-batch
            policy_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            valid_count = 0

            for r in results:
                if r["is_valid"]:
                    advantage = r["reward"] - self.baseline
                    policy_loss = policy_loss - r["log_prob"] * advantage
                    entropy_loss = entropy_loss - r["entropy"]
                    valid_count += 1

            if valid_count > 0:
                policy_loss = policy_loss / valid_count
                entropy_loss = entropy_loss / valid_count

                # Combined loss
                loss = policy_loss + self.entropy_coef * entropy_loss
                loss = loss / grad_accum_steps  # Scale for accumulation

                loss.backward()

                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update
        self.optimizer.step()
        self.scheduler.step()

        # Update baseline
        rewards = [r["reward"] for r in all_results]
        self.update_baseline(rewards)

        # Statistics
        r2_values = [r["r2"] for r in all_results]
        valid_mask = [r["is_valid"] for r in all_results]
        valid_r2 = [r2 for r2, v in zip(r2_values, valid_mask) if v]

        return {
            "valid_count": sum(valid_mask),
            "total_count": len(all_results),
            "valid_rate": sum(valid_mask) / len(all_results),
            "mean_r2": np.mean(valid_r2) if valid_r2 else 0.0,
            "max_r2": max(r2_values),
            "baseline": self.baseline,
            "policy_loss": total_policy_loss / grad_accum_steps,
            "entropy_loss": total_entropy_loss / grad_accum_steps,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def run(
        self,
        n_epochs: int = 50,
        batch_size: int = 16,
        grad_accum_steps: int = 2,
        target_r2: float = 0.99,
        patience: int = 20,
    ):
        """Run training with early stopping."""
        logger.info("=" * 60)
        logger.info("IMPROVED REINFORCE SYMBOLIC REGRESSION")
        logger.info("=" * 60)
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Batch size: {batch_size} x {grad_accum_steps} = {batch_size * grad_accum_steps}")
        logger.info(f"Entropy coef: {self.entropy_coef}")
        logger.info(f"Target R^2: {target_r2}")
        logger.info("=" * 60)

        no_improvement = 0
        prev_best = -np.inf

        for epoch in range(n_epochs):
            stats = self.train_step(batch_size, grad_accum_steps)

            self.history.append({
                "epoch": epoch + 1,
                **stats,
                "best_r2": self.best_r2,
            })

            # Check for improvement
            if self.best_r2 > prev_best + 0.001:
                no_improvement = 0
                prev_best = self.best_r2
            else:
                no_improvement += 1

            # Log every epoch for visibility
            logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Valid: {stats['valid_count']}/{stats['total_count']} | "
                f"Mean R²: {stats['mean_r2']:.4f} | "
                f"Best: {self.best_r2:.4f} | "
                f"Baseline: {self.baseline:.4f} | "
                f"LR: {stats['lr']:.2e}"
            )

            # Early stopping conditions
            if self.best_r2 >= target_r2:
                logger.info(f"Target R^2 {target_r2} reached at epoch {epoch+1}!")
                break

            if no_improvement >= patience:
                logger.info(f"No improvement for {patience} epochs. Early stopping.")
                break

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best R^2: {self.best_r2:.4f}")
        logger.info(f"Best expression: {self.best_expression}")
        logger.info(f"Unique expressions discovered: {len(self.discovered_expressions)}")

        # Show top 5 expressions
        top_exprs = sorted(self.discovered_expressions.items(), key=lambda x: -x[1])[:5]
        logger.info("Top 5 expressions:")
        for expr, r2 in top_exprs:
            logger.info(f"  R²={r2:.4f}: {expr}")

        return {
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "history": self.history,
            "discovered_expressions": self.discovered_expressions,
        }


def main():
    parser = argparse.ArgumentParser(description="Improved REINFORCE Symbolic Regression")
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/sin_x1.csv")
    parser.add_argument("--output_dir", type=str, default="./output/reinforce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    reg = RegressionDataset(str(dataset_path.parent), dataset_path.name)
    X, y = reg.get_numpy()

    # Run experiment
    experiment = ImprovedREINFORCE(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        device="cpu" if args.cpu else None,
        entropy_coef=args.entropy_coef,
    )

    results = experiment.run(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        patience=args.patience,
    )

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.output_dir) / f"results_improved_{timestamp}.json"

    # Convert for JSON serialization
    results_json = {
        "best_r2": float(results["best_r2"]),
        "best_expression": results["best_expression"],
        "history": results["history"],
        "discovered_expressions": {k: float(v) for k, v in results["discovered_expressions"].items()},
    }

    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
