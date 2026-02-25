"""
Best-of-N Baseline: Pure sampling without RL training.

This is a baseline that simply:
1. Generates N samples from the frozen model
2. Evaluates all samples
3. Returns the best one by reward/R2

No policy updates are performed - this measures the model's
inherent capability without any RL optimization.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

import numpy as np
import torch

import sys
from pathlib import Path

# Add paths for imports
REINFORCEMENT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = REINFORCEMENT_ROOT.parent.parent
sys.path.insert(0, str(REINFORCEMENT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from rewards import BaseReward, PenaltyHandler, RewardResult
from schedulers import TemperatureScheduler
from expression import Expression

logger = logging.getLogger(__name__)


@dataclass
class BoNConfig:
    """Configuration for Best-of-N baseline."""
    model_path: str
    base_model: str = "gpt2"
    n_samples: int = 1000
    batch_size: int = 64
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    output_dir: str = "./results"
    use_wandb: bool = False
    wandb_project: str = "seriguela"
    wandb_run_name: Optional[str] = None


class BestOfNBaseline:
    """
    Best-of-N baseline for symbolic regression.

    This baseline evaluates the model's inherent capability
    by generating many samples and selecting the best one,
    without any RL training.
    """

    def __init__(
        self,
        config: BoNConfig,
        x: np.ndarray,
        y: np.ndarray,
        reward_fn: BaseReward,
        penalty_handler: PenaltyHandler,
        is_prefix: bool = False,
        valid_variables: Optional[Set[str]] = None,
        ground_truth: Optional[str] = None,
    ):
        self.config = config
        self.x = x
        self.y = y
        self.reward_fn = reward_fn
        self.penalty_handler = penalty_handler
        self.is_prefix = is_prefix
        self.valid_variables = valid_variables or {"x_1"}
        self.ground_truth = ground_truth

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self._load_model()

        # Build prompt
        self._build_prompt()

        # Best tracking
        self.best_r2 = -float("inf")
        self.best_expression = None
        self.all_results: List[Dict] = []

        logger.info(f"Best-of-N baseline initialized with n_samples={config.n_samples}")

    def _load_model(self):
        """Load the model from HuggingFace."""
        logger.info(f"Loading model from {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
        self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()  # Never train

        logger.info(f"Model loaded on {self.device}")

    def _build_prompt(self):
        """Build the generation prompt."""
        ops = ["sin", "cos", "exp", "log", "sqrt", "+", "-", "*", "/", "**"]
        vars_list = sorted(list(self.valid_variables))

        prompt_dict = {
            "vars": vars_list,
            "ops": ops,
            "cons": "C",
            "expr": ""
        }

        import json
        self.prompt = json.dumps(prompt_dict)[:-2]  # Remove trailing "}
        self.prompt_ids = self.tokenizer.encode(
            self.prompt, return_tensors="pt"
        ).to(self.device)

        logger.info(f"Prompt: {self.prompt[:80]}...")

    def _extract_expression(self, text: str) -> Optional[str]:
        """Extract expression from generated text."""
        import json
        import re

        # Find the expression field
        expr_match = re.search(r'"expr":\s*"([^"]*)"', text)
        if expr_match:
            expr = expr_match.group(1)
            if expr.strip():
                return expr.strip()

        # Try to find expression after the prompt
        if '"expr": "' in text:
            parts = text.split('"expr": "')
            if len(parts) > 1:
                expr = parts[1].split('"')[0]
                if expr.strip():
                    return expr.strip()

        return None

    def compute_reward(self, expression: str) -> RewardResult:
        """Compute reward for an expression."""
        try:
            # Parse expression
            if self.is_prefix:
                expr_obj = Expression.from_prefix(expression)
            else:
                expr_obj = Expression.from_infix(expression)

            if expr_obj is None:
                penalty = self.penalty_handler.get_penalty(ErrorType.PARSING)
                return RewardResult(reward=penalty, is_valid=False, error_type=ErrorType.PARSING)

            # Check variables
            expr_vars = expr_obj.get_variables()
            if not expr_vars.issubset(self.valid_variables):
                penalty = self.penalty_handler.get_penalty(ErrorType.VARIABLES)
                return RewardResult(reward=penalty, is_valid=False, error_type=ErrorType.VARIABLES)

            # Evaluate
            y_pred = expr_obj.evaluate(self.x)
            if y_pred is None or not np.isfinite(y_pred).all():
                penalty = self.penalty_handler.get_penalty(ErrorType.NAN_INF)
                return RewardResult(reward=penalty, is_valid=False, error_type=ErrorType.NAN_INF)

            # Compute reward
            return self.reward_fn.compute(self.y, y_pred, expression)

        except Exception as e:
            return self.penalty_handler.get_penalty(ErrorType.EVALUATION)

    @torch.no_grad()
    def generate_batch(self, batch_size: int) -> List[str]:
        """Generate a batch of expressions."""
        outputs = self.model.generate(
            self.prompt_ids.expand(batch_size, -1),
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        expressions = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            expr = self._extract_expression(text)
            expressions.append(expr)

        return expressions

    def run(self) -> Dict:
        """
        Run Best-of-N sampling.

        Returns results dictionary with best expression and metrics.
        """
        logger.info(f"Starting Best-of-N sampling (n={self.config.n_samples})")

        # Initialize wandb if requested
        if self.config.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"bon_baseline",
                config={
                    "algorithm": "best_of_n",
                    "n_samples": self.config.n_samples,
                    "temperature": self.config.temperature,
                    "model": self.config.model_path,
                }
            )

        all_expressions = []
        all_rewards = []
        valid_count = 0
        total_generated = 0

        # Generate in batches
        n_batches = (self.config.n_samples + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(n_batches):
            current_batch_size = min(
                self.config.batch_size,
                self.config.n_samples - total_generated
            )

            if current_batch_size <= 0:
                break

            expressions = self.generate_batch(current_batch_size)

            for expr in expressions:
                if expr:
                    reward_result = self.compute_reward(expr)

                    all_expressions.append(expr)
                    all_rewards.append(reward_result)

                    if reward_result.is_valid:
                        valid_count += 1

                        if reward_result.r2 is not None and reward_result.r2 > self.best_r2:
                            self.best_r2 = reward_result.r2
                            self.best_expression = expr

            total_generated += current_batch_size

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {total_generated}/{self.config.n_samples} "
                    f"(valid: {valid_count}, best R2: {self.best_r2:.4f})"
                )

                if self.config.use_wandb:
                    import wandb
                    wandb.log({
                        "samples_generated": total_generated,
                        "valid_count": valid_count,
                        "valid_rate": valid_count / total_generated,
                        "best_r2": self.best_r2,
                    })

        # Final metrics
        valid_rewards = [r for r in all_rewards if r.is_valid]
        valid_r2_values = [r.r2 for r in valid_rewards if r.r2 is not None]

        results = {
            "algorithm": "best_of_n",
            "n_samples": self.config.n_samples,
            "n_valid": valid_count,
            "valid_rate": valid_count / max(total_generated, 1),
            "best_r2": self.best_r2,
            "best_expression": self.best_expression,
            "ground_truth": self.ground_truth,
            "model": self.config.model_path,
            "temperature": self.config.temperature,
        }

        if valid_r2_values:
            results["mean_r2"] = np.mean(valid_r2_values)
            results["std_r2"] = np.std(valid_r2_values)
            results["median_r2"] = np.median(valid_r2_values)

        logger.info(f"\n{'='*60}")
        logger.info("BEST-OF-N RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {total_generated}")
        logger.info(f"Valid samples: {valid_count} ({results['valid_rate']*100:.1f}%)")
        logger.info(f"Best R²: {self.best_r2:.6f}")
        logger.info(f"Best expression: {self.best_expression}")
        logger.info(f"Ground truth: {self.ground_truth}")

        if self.config.use_wandb:
            import wandb
            wandb.log(results)
            wandb.finish()

        return results


def run_best_of_n_baseline(
    model_path: str,
    base_model: str,
    x: np.ndarray,
    y: np.ndarray,
    reward_fn: BaseReward,
    penalty_handler: PenaltyHandler,
    n_samples: int = 1000,
    is_prefix: bool = False,
    valid_variables: Optional[Set[str]] = None,
    ground_truth: Optional[str] = None,
    temperature: float = 0.7,
    use_wandb: bool = False,
) -> Dict:
    """
    Convenience function to run Best-of-N baseline.

    Args:
        model_path: HuggingFace model path
        base_model: Base model name (gpt2, gpt2-medium, gpt2-large)
        x: Input data
        y: Target data
        reward_fn: Reward function
        penalty_handler: Penalty handler
        n_samples: Number of samples to generate
        is_prefix: Whether to use prefix notation
        valid_variables: Set of valid variable names
        ground_truth: Ground truth expression
        temperature: Generation temperature
        use_wandb: Whether to log to wandb

    Returns:
        Results dictionary
    """
    config = BoNConfig(
        model_path=model_path,
        base_model=base_model,
        n_samples=n_samples,
        temperature=temperature,
        use_wandb=use_wandb,
    )

    baseline = BestOfNBaseline(
        config=config,
        x=x,
        y=y,
        reward_fn=reward_fn,
        penalty_handler=penalty_handler,
        is_prefix=is_prefix,
        valid_variables=valid_variables,
        ground_truth=ground_truth,
    )

    return baseline.run()
