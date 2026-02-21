"""
Structured logging for RL experiments.

Provides:
- CSV logging of metrics
- JSONL logging of expressions
- Checkpoint management
- WandB integration
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    valid_count: int
    total_count: int
    valid_rate: float
    mean_reward: float
    mean_r2: float
    max_r2: float
    best_r2: float
    temperature: float
    policy_loss: float = 0.0
    entropy_loss: float = 0.0
    kl_divergence: float = 0.0
    learning_rate: float = 0.0


@dataclass
class ExpressionLog:
    """Log entry for a generated expression."""
    step: int
    expression: str
    r2: float
    reward: float
    is_valid: bool
    complexity: int
    log_prob: float = 0.0


class ExperimentLogger:
    """
    Structured logger for RL experiments.

    Logs:
    - Metrics per step (CSV)
    - Generated expressions (JSONL)
    - Checkpoints (model files)
    - Wandb (if enabled)
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        wandb_project: str = "seriguela",
    ):
        """
        Initialize experiment logger.

        Args:
            output_dir: Base output directory
            experiment_name: Name for this experiment
            use_wandb: Whether to log to Weights & Biases
            wandb_project: WandB project name
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # File paths
        self.metrics_file = self.output_dir / "metrics.csv"
        self.expressions_file = self.output_dir / "expressions.jsonl"
        self.config_file = self.output_dir / "config.json"

        # Initialize CSV
        self._init_csv()

        # Initialize WandB
        self.wandb_run = None
        if use_wandb:
            self._init_wandb()

        # Tracking
        self.step_count = 0
        self.expression_count = 0

    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "valid_count", "total_count", "valid_rate",
                    "mean_reward", "mean_r2", "max_r2", "best_r2",
                    "temperature", "policy_loss", "entropy_loss",
                    "kl_divergence", "learning_rate", "timestamp"
                ])

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                reinit=True,
            )
            logger.info(f"WandB initialized: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.use_wandb = False

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.wandb_run:
            import wandb
            wandb.config.update(config)

    def log_step(self, metrics: StepMetrics):
        """
        Log metrics for a training step.

        Args:
            metrics: StepMetrics dataclass with step metrics
        """
        self.step_count += 1

        # Log to CSV
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step,
                metrics.valid_count,
                metrics.total_count,
                metrics.valid_rate,
                metrics.mean_reward,
                metrics.mean_r2,
                metrics.max_r2,
                metrics.best_r2,
                metrics.temperature,
                metrics.policy_loss,
                metrics.entropy_loss,
                metrics.kl_divergence,
                metrics.learning_rate,
                datetime.now().isoformat(),
            ])

        # Log to WandB
        if self.wandb_run:
            import wandb
            wandb.log(asdict(metrics), step=metrics.step)

    def log_expression(self, expr_log: ExpressionLog):
        """
        Log a generated expression.

        Args:
            expr_log: ExpressionLog dataclass with expression info
        """
        self.expression_count += 1

        with open(self.expressions_file, "a") as f:
            json.dump(asdict(expr_log), f)
            f.write("\n")

    def log_expressions_batch(self, expressions: List[ExpressionLog]):
        """Log a batch of expressions."""
        with open(self.expressions_file, "a") as f:
            for expr_log in expressions:
                json.dump(asdict(expr_log), f)
                f.write("\n")

    def log_checkpoint(self, model, step: int) -> Path:
        """
        Save a model checkpoint.

        Args:
            model: Model to save (must have save_pretrained method)
            step: Current training step

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

        if self.wandb_run:
            import wandb
            wandb.save(str(checkpoint_dir / "*"))

        return checkpoint_dir

    def log_final_results(self, results: Dict[str, Any]):
        """
        Log final experiment results.

        Args:
            results: Dictionary with final results
        """
        results_file = self.output_dir / "final_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self.wandb_run:
            import wandb
            wandb.log({"final": results})
            wandb.finish()

        logger.info(f"Final results saved: {results_file}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged data."""
        return {
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "steps_logged": self.step_count,
            "expressions_logged": self.expression_count,
            "metrics_file": str(self.metrics_file),
            "expressions_file": str(self.expressions_file),
        }

    def close(self):
        """Close logger and finalize."""
        if self.wandb_run:
            import wandb
            wandb.finish()

        logger.info(f"Logger closed. Summary: {self.get_summary()}")
