#!/usr/bin/env python3
"""
PPO Experiment V2 for Symbolic Regression using TRL 0.16+ API

This script implements PPO with a custom RewardModel that computes R² scores
for symbolic expressions. The key insight is that TRL's reward_model parameter
accepts any torch.nn.Module that returns scores.

Key Design:
1. CustomRewardModel wraps R² computation as a neural network module
2. Uses the experimental PPO API from TRL 0.16+
3. JSON format prompts (matches training format)
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Import from experimental PPO (TRL 0.16+)
from trl.experimental.ppo import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import PeftModel

from expression import Expression
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class SequenceClassifierOutput:
    """Mimics transformers.modeling_outputs.SequenceClassifierOutput"""
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class SymbolicRegressionRewardModel(nn.Module):
    """
    Custom reward model that computes R² scores for symbolic expressions.

    This wraps the R² computation as a torch.nn.Module that mimics
    AutoModelForSequenceClassification output format, so it can be used
    with TRL's PPOTrainer which expects a reward_model parameter.

    The model doesn't have trainable parameters - it just decodes sequences
    and computes R² scores based on how well the expression fits the data.
    """

    def __init__(self, tokenizer, X: np.ndarray, y: np.ndarray, device: torch.device):
        super().__init__()
        self.tokenizer = tokenizer
        self.X = X
        self.y = y
        self.device = device
        self.n_vars = X.shape[1]

        # Add config attribute to mimic HuggingFace model
        self.config = type('Config', (), {'pad_token_id': tokenizer.pad_token_id})()

        # Dummy parameter so PyTorch recognizes this as a module
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        logger.info(f"RewardModel initialized with {len(X)} samples, {self.n_vars} variables")

    def extract_expression(self, generated_text: str) -> str:
        """Extract expression from JSON format output."""
        try:
            # Case 1: Standard JSON with quotes
            if '"expr": "' in generated_text:
                expr_start = generated_text.index('"expr": "') + len('"expr": "')
                remaining = generated_text[expr_start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                if '"' in remaining:
                    return remaining[:remaining.index('"')].strip()
                return remaining.strip()

            # Case 2: Model output without quotes: "expr": value"}
            if '"expr": ' in generated_text:
                expr_start = generated_text.index('"expr": ') + len('"expr": ')
                remaining = generated_text[expr_start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                if '"{' in remaining:
                    return remaining[:remaining.index('"{')].strip().rstrip('}')
                return remaining.strip()

            # Case 3: Compact JSON
            if '"expr":"' in generated_text:
                expr_start = generated_text.index('"expr":"') + len('"expr":"')
                remaining = generated_text[expr_start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                if '"' in remaining:
                    return remaining[:remaining.index('"')].strip()
                return remaining.strip()

        except (ValueError, IndexError):
            pass

        # Fallback
        fallback = generated_text.split('"expr"')[-1].strip(' ":}')
        if '"}' in fallback:
            fallback = fallback[:fallback.index('"}')]
        return fallback.strip()

    def compute_r2(self, expression_str: str) -> float:
        """Compute R² score for an expression."""
        if not expression_str or expression_str.isspace():
            return -1.0

        # Replace constant placeholder C with 1
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Compute rewards for a batch of sequences.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            attention_mask: Optional attention mask

        Returns:
            SequenceClassifierOutput with logits of shape (batch_size, 1)
        """
        batch_size = input_ids.shape[0]
        rewards = []

        for i in range(batch_size):
            # Decode the sequence
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)

            # Extract expression
            expr_str = self.extract_expression(text)

            # Compute R² score
            r2 = self.compute_r2(expr_str)
            rewards.append(r2)

        # Return in format expected by TRL (SequenceClassifierOutput with logits)
        logits = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return SequenceClassifierOutput(logits=logits)


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


def create_ppo_dataset(prompt: str, num_samples: int = 1000) -> Dataset:
    """Create a dataset of prompts for PPO training."""
    return Dataset.from_dict({
        "query": [prompt] * num_samples,
    })


def run_ppo_experiment(
    model_path: str,
    dataset_path: str,
    output_dir: str = "./output/ppo_v2",
    num_episodes: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
):
    """Run PPO experiment with custom R² reward model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset_path = Path(dataset_path)
    reg = RegressionDataset(str(dataset_path.parent), dataset_path.name)
    X, y = reg.get_numpy()
    n_vars = X.shape[1]
    logger.info(f"Dataset: {X.shape[0]} samples, {n_vars} variables")

    # Load tokenizer from trained model
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model and resize embeddings
    logger.info("Loading base GPT-2 model")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)

    if len(tokenizer) != base_model.config.vocab_size:
        logger.info(f"Resizing embeddings: {base_model.config.vocab_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter
    try:
        model_with_lora = PeftModel.from_pretrained(base_model, model_path)
        merged_model = model_with_lora.merge_and_unload()
        logger.info("LoRA adapter loaded and merged")
    except Exception as e:
        logger.warning(f"Could not load as PEFT model: {e}")
        merged_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Wrap with value head for PPO
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)

    # Create custom reward model
    reward_model = SymbolicRegressionRewardModel(tokenizer, X, y, device)

    # Build prompt and dataset
    prompt = build_prompt(n_vars)
    logger.info(f"Prompt template: {prompt}...")

    train_dataset = create_ppo_dataset(prompt, num_episodes)

    # PPO Config
    ppo_config = PPOConfig(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        total_episodes=num_episodes,
        num_ppo_epochs=4,
        gradient_accumulation_steps=1,
        response_length=50,
        temperature=0.7,
        kl_coef=0.05,
        missing_eos_penalty=0.0,  # Don't penalize, expressions don't need EOS
        report_to=None,
    )

    # Create PPO Trainer
    logger.info("Initializing PPO Trainer...")

    try:
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=policy_model,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
        )

        logger.info("PPO Trainer initialized successfully!")

        # Run training
        logger.info("Starting PPO training...")
        ppo_trainer.train()

        # Save results
        logger.info(f"Saving model to {output_dir}")
        ppo_trainer.save_model(str(output_dir / "final_model"))

        return {"status": "success", "output_dir": str(output_dir)}

    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="PPO Symbolic Regression V2")
    parser.add_argument("--model_path", type=str, default="./output/exp_a_json",
                        help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/sin_x1.csv",
                        help="Path to test dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./output/ppo_v2",
                        help="Output directory")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")

    args = parser.parse_args()

    run_ppo_experiment(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
