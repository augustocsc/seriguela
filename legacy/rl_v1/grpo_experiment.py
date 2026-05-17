#!/usr/bin/env python3
"""
GRPO Experiment for Symbolic Regression

GRPO (Group Relative Policy Optimization) supports custom reward functions
via the reward_funcs parameter, making it ideal for symbolic regression
where we compute R^2 scores as rewards.

This is the recommended approach for TRL 0.27+ since PPO experimental
has compatibility issues.

Usage:
    python scripts/grpo_experiment.py --dataset ./data/ppo_test/sin_x1.csv
"""

import os
os.environ['TRL_EXPERIMENTAL_SILENCE'] = '1'

import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from peft import PeftModel

from expression import Expression
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class SymbolicRegressionReward:
    """
    Reward function for symbolic regression.
    Computes R^2 score for generated expressions.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.n_vars = X.shape[1]
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []

    def extract_expression(self, text: str) -> str:
        """Extract expression from JSON format output."""
        try:
            # Case 1: Standard JSON with quotes
            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                if '"' in remaining:
                    return remaining[:remaining.index('"')].strip()
                return remaining.strip()

            # Case 2: Model output without quotes
            if '"expr": ' in text:
                start = text.index('"expr": ') + len('"expr": ')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                return remaining.strip()

        except (ValueError, IndexError):
            pass

        return text.split('"expr"')[-1].strip(' ":}')

    def compute_r2(self, expression_str: str) -> float:
        """Compute R^2 score for an expression."""
        if not expression_str or expression_str.isspace():
            return -1.0

        # Substitute C with 1
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

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: List of generated completion strings

        Returns:
            List of R^2 scores
        """
        rewards = []

        for completion in completions:
            # Extract expression from completion
            expr_str = self.extract_expression(completion)

            # Compute R^2
            r2 = self.compute_r2(expr_str)
            rewards.append(r2)

            # Track best
            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_expression = expr_str
                logger.info(f"New best R^2: {r2:.4f} - {expr_str}")

        # Log batch statistics
        valid_rewards = [r for r in rewards if r > -1.0]
        if valid_rewards:
            self.history.append({
                "mean_r2": np.mean(valid_rewards),
                "max_r2": max(valid_rewards),
                "valid_rate": len(valid_rewards) / len(rewards),
            })

        return rewards


def build_prompt(n_vars: int) -> str:
    """Build JSON format prompt matching training data."""
    vars_list = [f"x_{i+1}" for i in range(n_vars)]
    ops_list = ["+", "-", "*", "sin", "cos"]

    prompt = json.dumps({
        "vars": vars_list,
        "ops": ops_list,
        "cons": None,
        "expr": ""
    })[:-3]  # Remove trailing '"}' for model to complete

    return prompt


def run_grpo_experiment(
    model_path: str,
    dataset_path: str,
    output_dir: str = "./output/grpo_results",
    num_episodes: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    use_cpu: bool = False,
):
    """Run GRPO experiment with custom R^2 reward function."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset_path = Path(dataset_path)
    reg = RegressionDataset(str(dataset_path.parent), dataset_path.name)
    X, y = reg.get_numpy()
    n_vars = X.shape[1]
    logger.info(f"Dataset: {X.shape[0]} samples, {n_vars} variables")

    # Load tokenizer and model
    logger.info(f"Loading model from {model_path}")

    # Check if model_path is a local path or HuggingFace model
    if Path(model_path).exists():
        # Load tokenizer from trained model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        # Load base model and LoRA
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        if len(tokenizer) != base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(tokenizer))

        try:
            model_with_lora = PeftModel.from_pretrained(base_model, model_path)
            model = model_with_lora.merge_and_unload()
            logger.info("LoRA adapter loaded and merged")
        except Exception as e:
            logger.warning(f"Could not load LoRA: {e}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        # Load from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)

    logger.info("Model loaded successfully")

    # Build prompt and create dataset
    prompt = build_prompt(n_vars)
    logger.info(f"Prompt: {prompt}...")

    train_dataset = Dataset.from_dict({"prompt": [prompt] * num_episodes})

    # Create reward function
    reward_func = SymbolicRegressionReward(X, y, tokenizer)

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_generations=batch_size,  # Generate batch_size samples per prompt
        max_completion_length=50,
        num_train_epochs=1,
        report_to=[],
        use_cpu=use_cpu or device == "cpu",
        bf16=False if use_cpu or device == "cpu" else True,
        logging_steps=10,
        save_strategy="epoch",
    )

    # Create trainer
    logger.info("Creating GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_func,
    )

    # Train
    logger.info("="*60)
    logger.info("GRPO SYMBOLIC REGRESSION EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info("="*60)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        trainer.train()
        logger.info("Training completed!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    # Results
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Best R^2: {reward_func.best_r2:.4f}")
    logger.info(f"Best expression: {reward_func.best_expression}")

    # Save results
    results = {
        "timestamp": timestamp,
        "model_path": model_path,
        "dataset_path": str(dataset_path),
        "best_r2": reward_func.best_r2,
        "best_expression": reward_func.best_expression,
        "history": reward_func.history,
    }

    results_file = output_dir / f"grpo_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Save model
    trainer.save_model(str(output_dir / "final_model"))

    return results


def main():
    parser = argparse.ArgumentParser(description="GRPO Symbolic Regression")
    parser.add_argument("--model_path", type=str, default="gpt2",
                        help="Path to model (local or HuggingFace)")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/sin_x1.csv",
                        help="Path to test dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./output/grpo_results",
                        help="Output directory")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")

    args = parser.parse_args()

    run_grpo_experiment(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
