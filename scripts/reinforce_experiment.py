#!/usr/bin/env python3
"""
REINFORCE for Symbolic Regression

Simple policy gradient (REINFORCE) with R^2 as reward.
This is a more natural fit for symbolic regression than GRPO/PPO
since we have absolute rewards, not relative preferences.

Algorithm:
1. Generate expressions using current policy (model)
2. Compute R^2 reward for each expression
3. Compute policy gradient: grad = reward * grad(log_prob)
4. Update model parameters

With baseline subtraction for variance reduction:
    grad = (reward - baseline) * grad(log_prob)
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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


class REINFORCESymbolicRegression:
    """REINFORCE algorithm for symbolic regression."""

    def __init__(
        self,
        model_path: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = "./output/reinforce",
        learning_rate: float = 1e-5,
        device: str = None,
    ):
        self.X = X
        self.y = y
        self.n_vars = X.shape[1]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Tracking
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []
        self.baseline = 0.0  # Moving average baseline for variance reduction

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try to load as LoRA adapter first (works for both local and HuggingFace paths)
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
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model = self.model.to(self.device)
        self.model.train()

        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable params")

    def _build_prompt(self) -> str:
        """Build JSON format prompt compatible with trained model."""
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]
        ops_list = ["+", "-", "*", "sin", "cos"]

        # Build prompt that ends with "expr": " (with opening quote)
        # The model will generate the expression and end with <|endoftext|>
        prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": "C",  # Use "C" string, not None
            "expr": ""
        })
        # Remove closing "} to get: {..., "expr": "
        prompt = prompt[:-2]

        return prompt

    def extract_expression(self, text: str) -> str:
        """Extract expression from generated text.

        The trained model generates: {"vars": [...], "expr": "sin(x_1) + C<|endoftext|>
        So we need to extract text after "expr": " and before EOS or end of string.
        """
        try:
            # Remove EOS token if present
            eos_token = "<|endoftext|>"
            if eos_token in text:
                text = text[:text.index(eos_token)]

            # Find expression after "expr": "
            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                # Expression might end with ", "}, or just end of string
                for terminator in ['"}', '"']:
                    if terminator in remaining:
                        return remaining[:remaining.index(terminator)].strip()
                # No terminator found - return everything (EOS already removed)
                return remaining.strip()

            if '"expr": ' in text:
                start = text.index('"expr": ') + len('"expr": ')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                return remaining.strip(' "')

        except (ValueError, IndexError):
            pass

        # Fallback: take everything after "expr"
        if '"expr"' in text:
            return text.split('"expr"')[-1].strip(' ":{}')
        return text.strip()

    def compute_r2(self, expression_str: str) -> float:
        """Compute R^2 score."""
        if not expression_str or expression_str.isspace():
            return -1.0

        if 'C' in expression_str:
            expression_str = expression_str.replace('C', '1')

        try:
            expr = Expression(expression_str, is_prefix=False)
            if not expr.is_valid_on_dataset(self.X):
                return -1.0

            y_pred = expr.evaluate(self.X)
            if not np.all(np.isfinite(y_pred)):
                return -1.0

            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

            if ss_tot == 0:
                return 0.0

            r2 = 1 - (ss_res / ss_tot)
            return float(np.clip(r2, -1.0, 1.0))
        except Exception:
            return -1.0

    def generate_with_logprobs(self, max_new_tokens: int = 50, temperature: float = 0.7) -> Tuple[str, torch.Tensor]:
        """Generate a sequence and return log probabilities with gradients."""
        generated_ids = self.prompt_ids.clone()
        generated_tokens = []

        # Phase 1: Generate tokens without gradients (for efficiency)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :] / temperature

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token.item())

                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Check for JSON end (for models that use JSON format)
                text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                if '"}' in text[len(self.prompt):]:
                    break

        # Phase 2: Recompute log probabilities WITH gradients
        # This is the key fix - we need gradients for REINFORCE
        if len(generated_tokens) == 0:
            # No tokens generated - return with zero log prob
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return text, torch.tensor(0.0, device=self.device, requires_grad=True)

        # Re-run forward pass with gradients
        log_probs = []
        current_ids = self.prompt_ids.clone()

        for token_id in generated_tokens:
            outputs = self.model(current_ids)
            logits = outputs.logits[:, -1, :] / temperature
            log_prob = F.log_softmax(logits, dim=-1)

            # Get log prob of the token we sampled
            token_tensor = torch.tensor([[token_id]], device=self.device)
            selected_log_prob = log_prob.gather(1, token_tensor)
            log_probs.append(selected_log_prob)

            # Append token for next iteration
            current_ids = torch.cat([current_ids, token_tensor], dim=1)

        total_log_prob = torch.cat(log_probs, dim=1).sum()
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return text, total_log_prob

    def train_step(self, batch_size: int = 8) -> dict:
        """Perform one REINFORCE training step."""
        self.model.train()

        texts = []
        expressions = []
        rewards = []
        log_probs = []

        # Generate batch
        for _ in range(batch_size):
            text, log_prob = self.generate_with_logprobs()
            expr_str = self.extract_expression(text)
            r2 = self.compute_r2(expr_str)

            texts.append(text)
            expressions.append(expr_str)
            rewards.append(r2)
            log_probs.append(log_prob)

            # Track best
            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_expression = expr_str

        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Update baseline (exponential moving average)
        valid_rewards = rewards_tensor[rewards_tensor > -1.0]
        if len(valid_rewards) > 0:
            self.baseline = 0.9 * self.baseline + 0.1 * valid_rewards.mean().item()

        # Compute policy gradient with baseline
        # Loss = -sum(log_prob * (reward - baseline))
        policy_loss = 0.0
        valid_count = 0

        for log_prob, reward in zip(log_probs, rewards):
            if reward > -1.0:  # Only use valid expressions
                advantage = reward - self.baseline
                policy_loss -= log_prob * advantage
                valid_count += 1

        if valid_count > 0:
            policy_loss = policy_loss / valid_count

            # Backward pass
            self.optimizer.zero_grad()
            policy_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        # Statistics (convert numpy types to Python native for JSON serialization)
        rewards_array = np.array(rewards)
        valid_mask = rewards_array > -1.0

        return {
            "valid_count": int(valid_mask.sum()),
            "valid_rate": float(valid_mask.mean()),
            "mean_reward": float(rewards_array[valid_mask].mean()) if valid_mask.any() else 0.0,
            "max_reward": float(rewards_array.max()),
            "baseline": float(self.baseline),
            "policy_loss": policy_loss.item() if isinstance(policy_loss, torch.Tensor) else 0.0,
        }

    def run(
        self,
        n_epochs: int = 50,
        batch_size: int = 8,
        target_r2: float = 0.99,
    ):
        """Run REINFORCE training."""
        logger.info("=" * 60)
        logger.info("REINFORCE SYMBOLIC REGRESSION")
        logger.info("=" * 60)
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Target R^2: {target_r2}")
        logger.info("=" * 60)

        for epoch in range(n_epochs):
            stats = self.train_step(batch_size)

            self.history.append({
                "epoch": epoch + 1,
                **stats,
                "best_r2": self.best_r2,
            })

            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d} | "
                    f"Valid: {stats['valid_count']}/{batch_size} | "
                    f"Mean R^2: {stats['mean_reward']:.4f} | "
                    f"Best: {self.best_r2:.4f} | "
                    f"Baseline: {self.baseline:.4f}"
                )

            # Early stop
            if self.best_r2 >= target_r2:
                logger.info(f"Target R^2 {target_r2} reached at epoch {epoch+1}!")
                break

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best R^2: {self.best_r2:.4f}")
        logger.info(f"Best expression: {self.best_expression}")

        return {
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "history": self.history,
        }


def main():
    parser = argparse.ArgumentParser(description="REINFORCE Symbolic Regression")
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/sin_x1.csv")
    parser.add_argument("--output_dir", type=str, default="./output/reinforce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    experiment = REINFORCESymbolicRegression(
        model_path=args.model_path,
        X=X,
        y=y,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        device="cpu" if args.cpu else None,
    )

    results = experiment.run(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.output_dir) / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
