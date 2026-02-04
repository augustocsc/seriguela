#!/usr/bin/env python3
"""
PPO Experiment using Legacy TRL API (v0.11.0 or earlier)

This script uses the old PPOTrainer.step() API which accepts custom rewards
directly. This is the fallback approach if the modern TRL API doesn't work.

REQUIRES: pip install trl==0.11.0

Usage:
    pip install trl==0.11.0  # Downgrade TRL first
    python scripts/ppo_experiment_legacy.py --dataset ./data/ppo_test/sin_x1.csv
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
from peft import PeftModel

from expression import Expression
from dataset import RegressionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def check_trl_version():
    """Check if TRL version supports legacy API."""
    import trl
    version = trl.__version__
    major, minor = map(int, version.split('.')[:2])

    if major > 0 or minor >= 12:
        logger.warning(f"TRL version {version} may not support legacy step() API")
        logger.warning("Consider: pip install trl==0.11.0")
        return False
    return True


class LegacyPPOSymbolicRegression:
    """PPO-based symbolic regression using legacy TRL API."""

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str = "./output/ppo_legacy",
        batch_size: int = 16,
        learning_rate: float = 1e-5,
    ):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Device setup
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
        self.best_r2 = -np.inf
        self.best_expression = None
        self.history = []

    def _load_dataset(self):
        """Load regression dataset."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        reg = RegressionDataset(str(self.dataset_path.parent), self.dataset_path.name)
        self.X, self.y = reg.get_numpy()
        self.n_vars = self.X.shape[1]
        logger.info(f"Dataset: {self.X.shape[0]} samples, {self.n_vars} variables")

    def _load_model(self):
        """Load the JSON format model with LoRA adapters."""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer from trained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base GPT-2
        base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)

        # Resize embeddings
        if len(self.tokenizer) != base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        try:
            model_with_lora = PeftModel.from_pretrained(base_model, self.model_path)
            merged_model = model_with_lora.merge_and_unload()
            logger.info("LoRA adapter loaded and merged")
        except Exception as e:
            logger.warning(f"Could not load as PEFT model: {e}")
            merged_model = AutoModelForCausalLM.from_pretrained(self.model_path)

        # Import legacy PPO (TRL 0.11.0)
        try:
            from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
            self.ppo_modules = {
                'PPOConfig': PPOConfig,
                'PPOTrainer': PPOTrainer,
                'AutoModelForCausalLMWithValueHead': AutoModelForCausalLMWithValueHead,
            }
        except ImportError:
            logger.error("Could not import legacy TRL modules")
            logger.error("Try: pip install trl==0.11.0")
            raise

        # Wrap with value head
        ValueHeadModel = self.ppo_modules['AutoModelForCausalLMWithValueHead']
        self.model = ValueHeadModel.from_pretrained(merged_model)
        self.ref_model = ValueHeadModel.from_pretrained(merged_model)

        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)

        logger.info("Model loaded successfully")

    def _build_prompt(self):
        """Build JSON format prompt."""
        vars_list = [f"x_{i+1}" for i in range(self.n_vars)]
        ops_list = ["+", "-", "*", "sin", "cos"]

        self.prompt = json.dumps({
            "vars": vars_list,
            "ops": ops_list,
            "cons": None,
            "expr": ""
        })[:-3]

        logger.info(f"Prompt: {self.prompt}...")

    def _setup_ppo(self):
        """Setup legacy PPO trainer."""
        PPOConfig = self.ppo_modules['PPOConfig']
        PPOTrainer = self.ppo_modules['PPOTrainer']

        self.ppo_config = PPOConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=min(4, self.batch_size),
            ppo_epochs=4,
            log_with=None,
        )

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        logger.info("Legacy PPO trainer ready")

    def extract_expression(self, text: str) -> str:
        """Extract expression from JSON output."""
        try:
            if '"expr": "' in text:
                start = text.index('"expr": "') + len('"expr": "')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                if '"' in remaining:
                    return remaining[:remaining.index('"')].strip()
                return remaining.strip()

            if '"expr": ' in text:
                start = text.index('"expr": ') + len('"expr": ')
                remaining = text[start:]
                if '"}' in remaining:
                    return remaining[:remaining.index('"}')].strip()
                return remaining.strip()

        except (ValueError, IndexError):
            pass

        return text.split('"expr"')[-1].strip(' ":}')

    def compute_reward(self, expression_str: str) -> float:
        """Compute R² reward for an expression."""
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

    def train_epoch(self, epoch: int):
        """Run one epoch of PPO training using legacy step() API."""
        logger.info(f"\n{'='*60}\nEPOCH {epoch + 1}\n{'='*60}")

        # Tokenize prompt
        inputs = self.tokenizer(
            [self.prompt] * self.batch_size,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        queries = [inputs["input_ids"][i] for i in range(self.batch_size)]

        # Generate responses
        responses = []
        expressions = []
        rewards = []

        for i in tqdm(range(self.batch_size), desc="Generating"):
            output = self.model.generate(
                input_ids=inputs["input_ids"][i:i+1],
                attention_mask=inputs["attention_mask"][i:i+1],
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            response_ids = output[0][inputs["input_ids"].shape[1]:]
            full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            expr_str = self.extract_expression(full_text)
            reward = self.compute_reward(expr_str)

            responses.append(response_ids)
            expressions.append(expr_str)
            rewards.append(reward)

        # Convert to tensors
        reward_tensors = [torch.tensor(r, dtype=torch.float32, device=self.device) for r in rewards]

        # PPO step with custom rewards (legacy API)
        try:
            stats = self.ppo_trainer.step(queries, responses, reward_tensors)
            logger.info(f"PPO step completed")
        except Exception as e:
            logger.error(f"PPO step failed: {e}")
            stats = {}

        # Analyze results
        valid_count = sum(1 for r in rewards if r > 0)
        rewards_array = np.array(rewards)

        epoch_result = {
            "epoch": epoch + 1,
            "valid_count": valid_count,
            "valid_rate": valid_count / len(rewards),
            "mean_reward": float(np.mean(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "top_expressions": [],
        }

        # Find best expressions
        sorted_idx = np.argsort(rewards)[::-1]
        for i in sorted_idx[:5]:
            if rewards[i] > -1.0:
                epoch_result["top_expressions"].append({
                    "expression": expressions[i],
                    "r2": rewards[i],
                })

                if rewards[i] > self.best_r2:
                    self.best_r2 = rewards[i]
                    self.best_expression = expressions[i]

        self.history.append(epoch_result)

        # Log results
        logger.info(f"Valid: {valid_count}/{len(rewards)} ({epoch_result['valid_rate']:.1%})")
        logger.info(f"Mean R²: {epoch_result['mean_reward']:.4f}")
        logger.info(f"Max R²: {epoch_result['max_reward']:.4f}")

        if epoch_result["top_expressions"]:
            logger.info("Top expressions:")
            for i, expr in enumerate(epoch_result["top_expressions"][:3]):
                logger.info(f"  {i+1}. {expr['expression']} (R²={expr['r2']:.4f})")

        return epoch_result

    def run(self, n_epochs: int = 10):
        """Run PPO training."""
        logger.info("="*60)
        logger.info("LEGACY PPO SYMBOLIC REGRESSION")
        logger.info("="*60)
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Epochs: {n_epochs}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for epoch in range(n_epochs):
            self.train_epoch(epoch)

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "best_r2": self.best_r2,
                "best_expression": self.best_expression,
                "history": self.history,
            }

            with open(self.output_dir / f"checkpoint_{epoch+1}.json", 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Early stopping
            if self.best_r2 > 0.99:
                logger.info(f"Early stopping: R² > 0.99")
                break

        # Final results
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best R²: {self.best_r2:.4f}")
        logger.info(f"Best expression: {self.best_expression}")

        # Save final results
        final_file = self.output_dir / f"final_results_{timestamp}.json"
        with open(final_file, 'w') as f:
            json.dump({
                "best_r2": self.best_r2,
                "best_expression": self.best_expression,
                "history": self.history,
            }, f, indent=2)

        logger.info(f"Results saved to: {final_file}")

        return self.history


def main():
    parser = argparse.ArgumentParser(description="Legacy PPO Symbolic Regression")
    parser.add_argument("--model_path", type=str, default="./output/exp_a_json")
    parser.add_argument("--dataset", type=str, default="./data/ppo_test/sin_x1.csv")
    parser.add_argument("--output_dir", type=str, default="./output/ppo_legacy")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    # Check TRL version
    check_trl_version()

    experiment = LegacyPPOSymbolicRegression(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    experiment.run(n_epochs=args.epochs)


if __name__ == "__main__":
    main()
