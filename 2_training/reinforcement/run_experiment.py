#!/usr/bin/env python3
"""
Unified experiment runner for symbolic regression RL.

Usage:
    python run_experiment.py \
        --algorithm bon_ppo \
        --model augustocsc/gpt2_base_infix_682k \
        --reward length_penalized \
        --penalty gradient \
        --temperature cosine_annealing \
        --problem nguyen_5 \
        --seeds 42 123 456 789 1337

    python run_experiment.py --config experiment_config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import logging
import datetime
import random
from pathlib import Path
from typing import Optional, List, Set

import numpy as np
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classes"))

from algorithms import BoNPPOTrainer, BoNGRPOTrainer, TrainerConfig
from rewards import (
    R2ClippedReward, LengthPenalizedReward, SRICReward,
    PenaltyStrategy, PenaltyHandler, create_reward_with_penalty
)
from schedulers import create_temperature_scheduler
from callbacks import EarlyStoppingCallback, EarlyStoppingConfig
from buffers import EliteBuffer
from utils.hf_upload import HuggingFaceUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Nguyen benchmark definitions
NGUYEN_BENCHMARKS = {
    "nguyen_1": {
        "equation": "x**3 + x**2 + x",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_2": {
        "equation": "x**4 + x**3 + x**2 + x",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_3": {
        "equation": "x**5 + x**4 + x**3 + x**2 + x",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_4": {
        "equation": "x**6 + x**5 + x**4 + x**3 + x**2 + x",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_5": {
        "equation": "sin(x**2) * cos(x) - 1",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_6": {
        "equation": "sin(x) + sin(x + x**2)",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_7": {
        "equation": "log(x + 1) + log(x**2 + 1)",
        "vars": ["x_1"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_8": {
        "equation": "sqrt(x)",
        "vars": ["x_1"],
        "domain": (0, 4),
        "n_samples": 100,
    },
    "nguyen_9": {
        "equation": "sin(x) + sin(y**2)",
        "vars": ["x_1", "x_2"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_10": {
        "equation": "2*sin(x)*cos(y)",
        "vars": ["x_1", "x_2"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_11": {
        "equation": "x**y",
        "vars": ["x_1", "x_2"],
        "domain": (0, 2),
        "n_samples": 100,
    },
    "nguyen_12": {
        "equation": "x**4 - x**3 + y**2/2 - y",
        "vars": ["x_1", "x_2"],
        "domain": (0, 2),
        "n_samples": 100,
    },
}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_nguyen_data(problem: str) -> tuple:
    """Generate data for a Nguyen benchmark problem."""
    if problem not in NGUYEN_BENCHMARKS:
        raise ValueError(f"Unknown problem: {problem}. Available: {list(NGUYEN_BENCHMARKS.keys())}")

    benchmark = NGUYEN_BENCHMARKS[problem]
    n_vars = len(benchmark["vars"])
    n_samples = benchmark["n_samples"]
    domain = benchmark["domain"]

    # Generate X
    x = np.random.uniform(domain[0], domain[1], (n_samples, n_vars))

    # Compute y using the equation
    # Map variables
    local_vars = {}
    for i, var_name in enumerate(benchmark["vars"]):
        local_vars[var_name.replace("_", "")] = x[:, i]  # x_1 -> x1
        local_vars["x"] = x[:, 0] if n_vars == 1 else None
        local_vars["y"] = x[:, 1] if n_vars >= 2 else None

    # Add safe functions
    safe_funcs = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sqrt": np.sqrt,
        "log": np.log,
        "exp": np.exp,
    }
    local_vars.update(safe_funcs)

    y = eval(benchmark["equation"], {"__builtins__": None}, local_vars)

    return x, y, benchmark["equation"], set(benchmark["vars"])


def create_trainer(
    algorithm: str,
    config: TrainerConfig,
    x: np.ndarray,
    y: np.ndarray,
    reward_fn,
    penalty_handler,
    temp_scheduler,
    early_stopping,
    elite_buffer,
    is_prefix: bool,
    valid_variables: Set[str],
    ground_truth: str,
):
    """Create the appropriate trainer based on algorithm."""
    trainer_classes = {
        "bon_ppo": BoNPPOTrainer,
        "bon_grpo": BoNGRPOTrainer,
    }

    if algorithm not in trainer_classes:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(trainer_classes.keys())}")

    return trainer_classes[algorithm](
        config=config,
        x=x,
        y=y,
        reward_fn=reward_fn,
        penalty_handler=penalty_handler,
        temp_scheduler=temp_scheduler,
        early_stopping=early_stopping,
        elite_buffer=elite_buffer,
        is_prefix=is_prefix,
        valid_variables=valid_variables,
        ground_truth=ground_truth,
    )


def run_single_experiment(args, seed: int) -> dict:
    """Run a single experiment with specified seed."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment with seed {seed}")
    logger.info(f"{'='*60}")

    # Set seed
    set_seed(seed)

    # Generate data
    x, y, ground_truth, valid_variables = generate_nguyen_data(args.problem)
    logger.info(f"Generated data for {args.problem}: X shape {x.shape}")

    # Determine notation from model name
    is_prefix = "prefix" in args.model.lower()
    logger.info(f"Using {'prefix' if is_prefix else 'infix'} notation")

    # Create reward function
    reward_kwargs = {}
    if args.reward == "length_penalized":
        reward_kwargs["alpha"] = args.reward_alpha
    elif args.reward == "sr_ic":
        reward_kwargs["lambda_complexity"] = args.reward_lambda

    reward_fn, penalty_handler = create_reward_with_penalty(
        reward_type=args.reward,
        penalty_strategy=args.penalty,
        **reward_kwargs
    )

    # Create temperature scheduler
    temp_scheduler = create_temperature_scheduler(args.temperature)

    # Create early stopping
    es_config = EarlyStoppingConfig(
        patience=args.patience,
        delta=args.delta,
        r2_threshold=0.999,
        max_steps=args.max_steps,
    )
    early_stopping = EarlyStoppingCallback(es_config, ground_truth=ground_truth)

    # Create elite buffer
    elite_buffer = EliteBuffer(
        max_size=args.buffer_size,
        sample_ratio=args.buffer_ratio,
    )

    # Create trainer config
    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) / f"{args.algorithm}_{model_name}" / args.problem / f"seed_{seed}"

    # Detect base model size
    base_model = "gpt2"
    if "medium" in args.model.lower():
        base_model = "gpt2-medium"
    elif "large" in args.model.lower():
        base_model = "gpt2-large"

    config = TrainerConfig(
        model_path=args.model,
        base_model=base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
        max_kl=args.max_kl,
        group_size=args.group_size,
        buffer_size=args.buffer_size,
        buffer_sample_ratio=args.buffer_ratio,
        patience=args.patience,
        delta=args.delta,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=str(output_dir),
        use_wandb=args.use_wandb,
        wandb_project="seriguela",
        wandb_run_name=f"seriguela-{args.algorithm}-{model_name}-{args.problem}-seed{seed}",
    )

    # Create trainer
    trainer = create_trainer(
        algorithm=args.algorithm,
        config=config,
        x=x,
        y=y,
        reward_fn=reward_fn,
        penalty_handler=penalty_handler,
        temp_scheduler=temp_scheduler,
        early_stopping=early_stopping,
        elite_buffer=elite_buffer,
        is_prefix=is_prefix,
        valid_variables=valid_variables,
        ground_truth=ground_truth,
    )

    # Run training
    results = trainer.run()

    # Add seed to results
    results["seed"] = seed
    results["problem"] = args.problem
    results["model"] = args.model

    # Upload to HuggingFace if requested
    if args.upload_hf:
        try:
            uploader = HuggingFaceUploader()
            repo_name = f"seriguela-{args.algorithm}-{model_name}-{args.problem}-seed{seed}"
            url = uploader.upload_model(output_dir / "checkpoints" / "final", repo_name)
            logger.info(f"Model uploaded to: {url}")
            results["hf_url"] = url
        except Exception as e:
            logger.warning(f"Failed to upload to HuggingFace: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run symbolic regression RL experiment")

    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Algorithm
    parser.add_argument("--algorithm", choices=["bon_ppo", "bon_grpo"],
                        default="bon_ppo", help="RL algorithm to use")

    # Model
    parser.add_argument("--model", type=str, default="augustocsc/gpt2_base_infix_682k",
                        help="HuggingFace model repository")

    # Problem
    parser.add_argument("--problem", type=str, default="nguyen_5",
                        help="Nguyen benchmark problem")

    # Reward
    parser.add_argument("--reward", choices=["r2_clipped", "length_penalized", "sr_ic"],
                        default="r2_clipped", help="Reward function")
    parser.add_argument("--reward_alpha", type=float, default=0.01,
                        help="Alpha for length penalty")
    parser.add_argument("--reward_lambda", type=float, default=0.1,
                        help="Lambda for SR-IC complexity")

    # Penalty
    parser.add_argument("--penalty", choices=["binary", "gradient"],
                        default="binary", help="Penalty strategy for invalid expressions")

    # Temperature
    parser.add_argument("--temperature", choices=[
        "fixed_0.7", "fixed_0.9", "linear_annealing", "cosine_annealing"
    ], default="fixed_0.7", help="Temperature scheduler")

    # Training
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")

    # PPO/GRPO
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per batch")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max_kl", type=float, default=0.1, help="Max KL divergence")
    parser.add_argument("--group_size", type=int, default=8, help="GRPO group size")

    # Buffer
    parser.add_argument("--buffer_size", type=int, default=1000, help="Elite buffer size")
    parser.add_argument("--buffer_ratio", type=float, default=0.2,
                        help="Ratio of batch from buffer")

    # Early stopping
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--delta", type=float, default=0.01, help="Minimum improvement")

    # Logging
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases")

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds to run")

    # Upload
    parser.add_argument("--upload_hf", action="store_true",
                        help="Upload results to HuggingFace")

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Handle wandb flags
    if args.no_wandb:
        args.use_wandb = False

    # Run experiments for all seeds
    all_results = []
    for seed in args.seeds:
        try:
            results = run_single_experiment(args, seed)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Experiment with seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    if all_results:
        best_r2_values = [r["best_r2"] for r in all_results]
        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATE RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Seeds: {args.seeds}")
        logger.info(f"Mean Best R²: {np.mean(best_r2_values):.4f}")
        logger.info(f"Std Best R²: {np.std(best_r2_values):.4f}")
        logger.info(f"Max Best R²: {np.max(best_r2_values):.4f}")
        logger.info(f"Min Best R²: {np.min(best_r2_values):.4f}")

        # Save aggregate results
        aggregate = {
            "seeds": args.seeds,
            "problem": args.problem,
            "algorithm": args.algorithm,
            "model": args.model,
            "mean_best_r2": float(np.mean(best_r2_values)),
            "std_best_r2": float(np.std(best_r2_values)),
            "max_best_r2": float(np.max(best_r2_values)),
            "min_best_r2": float(np.min(best_r2_values)),
            "individual_results": all_results,
        }

        output_path = Path(args.output_dir) / f"aggregate_{args.algorithm}_{args.problem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aggregate, f, indent=2, default=str)
        logger.info(f"Aggregate results saved to: {output_path}")


if __name__ == "__main__":
    main()
