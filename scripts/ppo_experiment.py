#!/usr/bin/env python3
"""
PPO Experiment for Symbolic Regression using JSON Format Model

This script tests whether PPO fine-tuning can help find better expressions
for symbolic regression tasks. It uses the JSON format model (exp_a_json)
which achieves 80% valid expressions.

Key Design Decisions:
1. JSON format prompts (matches training format)
2. No constants (C) - simplified to avoid optimization complexity
3. Max retries to avoid infinite loops
4. Proper logging and checkpointing
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import PeftModel
from datasets import Dataset

from expression import Expression
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "output" / "ppo_experiment.log")
    ]
)
logger = logging.getLogger(__name__)


class PPOSymbolicRegression:
    """PPO-based symbolic regression using JSON format model."""

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str = "./output/ppo_results",
        batch_size: int = 64,
        learning_rate: float = 1e-5,
        max_retries: int = 10,
        device: str = None,
    ):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_retries = max_retries

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load dataset
        self._load_dataset()

        # Load model
        self._load_model()

        # Build JSON prompt
        self._build_prompt()

        # Setup PPO trainer
        self._setup_ppo()

        # Results tracking
        self.results = {
            "config": {
                "model_path": model_path,
                "dataset_path": str(dataset_path),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "n_vars": self.n_vars,
                "prompt": self.prompt,
            },
            "epochs": [],
            "best_expression": None,
            "best_r2": -np.inf,
        }

    def _load_dataset(self):
        """Load regression dataset."""
        logger.info(f"Loading dataset from {self.dataset_path}")

        # Load CSV
        reg = RegressionDataset(
            path=str(self.dataset_path.parent),
            file_name=self.dataset_path.name,
            delimiter=',',
        )
        self.X, self.y = reg.get_numpy()
        self.n_vars = self.X.shape[1]

        logger.info(f"Dataset loaded: {self.X.shape[0]} samples, {self.n_vars} variables")
        logger.info(f"y range: [{self.y.min():.3f}, {self.y.max():.3f}]")

    def _load_model(self):
        """Load the JSON format model with LoRA adapters."""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer from trained model (has special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer loaded with vocab size: {len(self.tokenizer)}")

        # Load base GPT-2
        base_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,  # PPO needs float32
        )

        # Resize embeddings to match tokenizer (handles special tokens)
        if len(self.tokenizer) != base_model.config.vocab_size:
            logger.info(f"Resizing embeddings: {base_model.config.vocab_size} -> {len(self.tokenizer)}")
            base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        try:
            model_with_lora = PeftModel.from_pretrained(base_model, self.model_path)
            merged_model = model_with_lora.merge_and_unload()
            logger.info("LoRA adapter loaded and merged")
        except Exception as e:
            logger.warning(f"Could not load as PEFT model: {e}")
            logger.info("Loading as full model...")
            merged_model = AutoModelForCausalLM.from_pretrained(self.model_path)

        # Wrap with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)

        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)

        logger.info("Model loaded successfully")

    def _build_prompt(self):
        """Build JSON format prompt matching training data."""
        # Variables based on dataset dimensions
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]

        # Operators (no division to avoid numerical issues)
        ops_list = ["+", "-", "*", "sin", "cos"]

        # Build JSON prompt (truncated for model to complete)
        self.prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": None,  # No constants for this experiment
            "expr": ""
        })[:-3]  # Remove trailing '"}' so model completes it

        logger.info(f"Prompt template: {self.prompt}...")

    def _setup_ppo(self):
        """Setup PPO trainer."""
        logger.info("Setting up PPO trainer...")

        self.ppo_config = PPOConfig(
            model_name=None,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=min(16, self.batch_size),
            gradient_accumulation_steps=1,
            ppo_epochs=4,
            log_with=None,  # or "wandb" for logging
            optimize_cuda_cache=True,
        )

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        logger.info("PPO trainer ready")

    def extract_expression(self, generated_text: str) -> str:
        """Extract expression from JSON format output."""
        try:
            # Find the expression part
            if '"expr": "' in generated_text:
                expr_start = generated_text.index('"expr": "') + len('"expr": "')
                expr_end = generated_text.index('"', expr_start)
                return generated_text[expr_start:expr_end].strip()
            elif '"expr":"' in generated_text:
                expr_start = generated_text.index('"expr":"') + len('"expr":"')
                expr_end = generated_text.index('"', expr_start)
                return generated_text[expr_start:expr_end].strip()
        except (ValueError, IndexError):
            pass

        # Fallback: return everything after prompt
        return generated_text.split('"expr"')[-1].strip(' ":}')

    def compute_reward(self, expression_str: str) -> float:
        """
        Compute reward (R^2 score) for an expression.
        No constant optimization - expressions should not contain C.
        """
        if not expression_str or expression_str.isspace():
            return -1.0

        # Reject expressions with constants (we don't want them)
        if 'C' in expression_str:
            return -0.5  # Penalty but not as harsh as invalid

        try:
            expr = Expression(expression_str, is_prefix=False)

            # Check if valid on dataset
            if not expr.is_valid_on_dataset(self.X):
                return -1.0

            # Compute R^2 (no constant fitting)
            y_pred = expr.evaluate(self.X)

            if not np.all(np.isfinite(y_pred)):
                return -1.0

            # R^2 score
            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

            if ss_tot == 0:
                return 0.0

            r2 = 1 - (ss_res / ss_tot)

            # Clip to reasonable range
            return float(np.clip(r2, -1.0, 1.0))

        except Exception as e:
            return -1.0

    def generate_batch(self):
        """Generate a batch of expressions."""
        # Tokenize prompt
        inputs = self.tokenizer(
            [self.prompt] * self.batch_size,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        queries = [inputs["input_ids"][i] for i in range(self.batch_size)]

        responses = []
        expressions = []
        rewards = []
        retries_used = []

        for i in tqdm(range(self.batch_size), desc="Generating", leave=False):
            # Try to generate valid expression (with retry limit)
            best_reward = -np.inf
            best_response = None
            best_expr = None

            for retry in range(self.max_retries):
                output = self.model.generate(
                    input_ids=inputs["input_ids"][i:i+1],
                    attention_mask=inputs["attention_mask"][i:i+1],
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Get response tokens only
                response_ids = output[0][inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Extract expression
                full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                expr_str = self.extract_expression(full_text)

                # Compute reward
                reward = self.compute_reward(expr_str)

                if reward > best_reward:
                    best_reward = reward
                    best_response = response_ids
                    best_expr = expr_str

                # If we found a valid expression, stop retrying
                if reward > 0:
                    break

            responses.append(best_response if best_response is not None else response_ids)
            expressions.append(best_expr if best_expr is not None else expr_str)
            rewards.append(best_reward)
            retries_used.append(retry + 1)

        return queries, responses, expressions, rewards, retries_used

    def train_epoch(self, epoch: int):
        """Run one epoch of PPO training."""
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}")
        logger.info(f"{'='*60}")

        # Generate batch
        queries, responses, expressions, rewards, retries = self.generate_batch()

        # Convert rewards to tensors
        reward_tensors = [torch.tensor(r, dtype=torch.float32, device=self.device) for r in rewards]

        # Ensure responses are tensors on correct device
        response_tensors = [r.to(self.device) if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device) for r in responses]

        # PPO step
        try:
            stats = self.ppo_trainer.step(queries, response_tensors, reward_tensors)
        except Exception as e:
            logger.error(f"PPO step failed: {e}")
            stats = {}

        # Analyze results
        valid_count = sum(1 for r in rewards if r > 0)
        invalid_count = sum(1 for r in rewards if r <= -1.0)

        rewards_array = np.array(rewards)
        valid_rewards = rewards_array[rewards_array > 0]

        epoch_results = {
            "epoch": epoch + 1,
            "valid_count": valid_count,
            "valid_rate": valid_count / len(rewards),
            "invalid_count": invalid_count,
            "mean_reward": float(np.mean(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "mean_valid_reward": float(np.mean(valid_rewards)) if len(valid_rewards) > 0 else None,
            "mean_retries": float(np.mean(retries)),
            "top_expressions": [],
        }

        # Find best expressions
        sorted_idx = np.argsort(rewards)[::-1]
        for i in sorted_idx[:5]:
            if rewards[i] > -1.0:
                epoch_results["top_expressions"].append({
                    "expression": expressions[i],
                    "r2": rewards[i],
                })

                # Update global best
                if rewards[i] > self.results["best_r2"]:
                    self.results["best_r2"] = rewards[i]
                    self.results["best_expression"] = expressions[i]

        self.results["epochs"].append(epoch_results)

        # Log results
        logger.info(f"Valid expressions: {valid_count}/{len(rewards)} ({epoch_results['valid_rate']:.1%})")
        logger.info(f"Mean reward: {epoch_results['mean_reward']:.4f}")
        logger.info(f"Max reward: {epoch_results['max_reward']:.4f}")
        logger.info(f"Mean retries: {epoch_results['mean_retries']:.1f}")

        if epoch_results["top_expressions"]:
            logger.info("Top expressions:")
            for i, expr_info in enumerate(epoch_results["top_expressions"][:3]):
                logger.info(f"  {i+1}. {expr_info['expression']} (R²={expr_info['r2']:.4f})")

        return epoch_results

    def run(self, n_epochs: int = 10, early_stop_r2: float = 0.95):
        """Run full PPO training."""
        logger.info("=" * 60)
        logger.info("PPO SYMBOLIC REGRESSION EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Early stop R²: {early_stop_r2}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for epoch in range(n_epochs):
            epoch_results = self.train_epoch(epoch)

            # Save checkpoint
            checkpoint_file = self.output_dir / f"checkpoint_epoch_{epoch+1}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            # Early stopping
            if self.results["best_r2"] >= early_stop_r2:
                logger.info(f"\nEarly stopping: R² >= {early_stop_r2}")
                break

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best expression: {self.results['best_expression']}")
        logger.info(f"Best R²: {self.results['best_r2']:.4f}")

        # Save final results
        final_file = self.output_dir / f"final_results_{timestamp}.json"
        with open(final_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to: {final_file}")

        return self.results


def main():
    parser = argparse.ArgumentParser(description="PPO Symbolic Regression Experiment")
    parser.add_argument("--model_path", type=str, default="./output/exp_a_json",
                        help="Path to trained model (JSON format)")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/mul_x1_x2.csv",
                        help="Path to test dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./output/ppo_results",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for PPO")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of PPO epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--early_stop_r2", type=float, default=0.95,
                        help="Early stop when R² reaches this value")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiment
    experiment = PPOSymbolicRegression(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    results = experiment.run(n_epochs=args.epochs, early_stop_r2=args.early_stop_r2)

    return results


if __name__ == "__main__":
    main()
