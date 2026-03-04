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

from algorithms import (
    BoNPPOTrainer, BoNGRPOTrainer, PurePPOTrainer, PureGRPOTrainer,
    TrainerConfig, BestOfNBaseline, BoNConfig, run_best_of_n_baseline
)
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


def generate_nguyen_data(problem: str, seed: Optional[int] = None) -> tuple:
    """Generate data for a Nguyen benchmark problem.

    Args:
        problem: Name of the Nguyen benchmark
        seed: Random seed for reproducibility (if None, uses current RNG state)

    Returns:
        Tuple of (X, y, equation, valid_variables)
    """
    if problem not in NGUYEN_BENCHMARKS:
        raise ValueError(f"Unknown problem: {problem}. Available: {list(NGUYEN_BENCHMARKS.keys())}")

    benchmark = NGUYEN_BENCHMARKS[problem]
    n_vars = len(benchmark["vars"])
    n_samples = benchmark["n_samples"]
    domain = benchmark["domain"]

    # Use specific seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate X
    x = np.random.uniform(domain[0], domain[1], (n_samples, n_vars))

    # Compute y using the equation
    # Map variables - FIX: Always set x, conditionally set y
    local_vars = {}
    for i, var_name in enumerate(benchmark["vars"]):
        local_vars[var_name.replace("_", "")] = x[:, i]  # x_1 -> x1

    # Set shorthand variables for equations (FIX: x is always column 0)
    local_vars["x"] = x[:, 0]  # Always available
    local_vars["y"] = x[:, 1] if n_vars >= 2 else None  # Only for 2+ variables

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


def generate_train_test_data(problem: str, seed: int) -> dict:
    """Generate separate train and test data for a Nguyen benchmark.

    Uses different random seeds for train (seed) and test (seed + 10000)
    to ensure no data leakage.

    Args:
        problem: Name of the Nguyen benchmark
        seed: Base random seed

    Returns:
        Dictionary with train/test data:
        {
            "train": {"x": ..., "y": ...},
            "test": {"x": ..., "y": ...},
            "equation": ...,
            "valid_variables": ...
        }
    """
    # Generate training data with base seed
    x_train, y_train, equation, valid_vars = generate_nguyen_data(problem, seed=seed)

    # Generate test data with offset seed (ensures different random points)
    x_test, y_test, _, _ = generate_nguyen_data(problem, seed=seed + 10000)

    return {
        "train": {"x": x_train, "y": y_train},
        "test": {"x": x_test, "y": y_test},
        "equation": equation,
        "valid_variables": valid_vars,
    }


def evaluate_on_test_set(expression_str: str, x_test: np.ndarray, y_test: np.ndarray,
                         is_prefix: bool = False) -> dict:
    """Evaluate an expression on the test set.

    Args:
        expression_str: The expression string to evaluate
        x_test: Test input data
        y_test: Test target values
        is_prefix: Whether expression is in prefix notation

    Returns:
        Dictionary with test metrics:
        {
            "test_r2": float or None,
            "test_mse": float or None,
            "test_valid": bool,
            "test_error": str or None
        }
    """
    from sklearn.metrics import r2_score, mean_squared_error

    try:
        # Import Expression class
        from expression import Expression

        # Parse expression
        expr = Expression(expression_str, is_prefix=is_prefix)

        # Evaluate with C=1 (fit_constants with optimize=False)
        y_pred = expr.evaluate(x_test)

        # Check for valid predictions
        if y_pred is None or not np.all(np.isfinite(y_pred)):
            return {
                "test_r2": None,
                "test_mse": None,
                "test_valid": False,
                "test_error": "Non-finite predictions"
            }

        # Compute metrics
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)

        return {
            "test_r2": float(test_r2),
            "test_mse": float(test_mse),
            "test_valid": True,
            "test_error": None
        }

    except Exception as e:
        return {
            "test_r2": None,
            "test_mse": None,
            "test_valid": False,
            "test_error": str(e)
        }


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
        "pure_ppo": PurePPOTrainer,
        "pure_grpo": PureGRPOTrainer,
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

    # Generate train/test data with separate random seeds
    data = generate_train_test_data(args.problem, seed)
    x = data["train"]["x"]
    y = data["train"]["y"]
    x_test = data["test"]["x"]
    y_test = data["test"]["y"]
    ground_truth = data["equation"]
    valid_variables = data["valid_variables"]

    logger.info(f"Generated data for {args.problem}: train {x.shape}, test {x_test.shape}")

    # Add noise if requested
    if args.noise_type != "none" and args.noise_level > 0:
        from utils.noise_generator import add_noise, create_noise_config
        noise_config = create_noise_config(
            noise_type=args.noise_type,
            noise_level=args.noise_level,
        )
        y_original = y.copy()
        y = add_noise(y, noise_config, seed=seed)
        logger.info(f"Added {args.noise_type} noise (level={args.noise_level})")

    # Determine notation from model name
    is_prefix = "prefix" in args.model.lower()
    logger.info(f"Using {'prefix' if is_prefix else 'infix'} notation")

    # Detect base model size
    base_model = "gpt2"
    if "medium" in args.model.lower():
        base_model = "gpt2-medium"
    elif "large" in args.model.lower():
        base_model = "gpt2-large"

    model_name = args.model.split("/")[-1]

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

    # Handle Best-of-N baseline separately (no RL training)
    if args.algorithm == "best_of_n":
        results = run_best_of_n_baseline(
            model_path=args.model,
            base_model=base_model,
            x=x,
            y=y,
            reward_fn=reward_fn,
            penalty_handler=penalty_handler,
            n_samples=args.max_steps * args.batch_size,  # Total samples similar to RL
            is_prefix=is_prefix,
            valid_variables=valid_variables,
            ground_truth=ground_truth,
            temperature=0.7 if args.temperature.startswith("fixed") else 0.7,
            use_wandb=args.use_wandb,
        )

        # Evaluate best expression on TEST SET
        if results.get("best_expression"):
            logger.info(f"\n--- Evaluating on TEST SET ---")
            test_metrics = evaluate_on_test_set(
                expression_str=results["best_expression"],
                x_test=x_test,
                y_test=y_test,
                is_prefix=is_prefix,
            )
            results.update(test_metrics)
            if test_metrics["test_valid"]:
                logger.info(f"Test R²: {test_metrics['test_r2']:.6f}")

        results["seed"] = seed
        results["problem"] = args.problem
        results["model"] = args.model
        return results

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
    output_dir = Path(args.output_dir) / f"{args.algorithm}_{model_name}" / args.problem / args.temperature / f"seed_{seed}"

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
        prompt_type=args.prompt_type,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=str(output_dir),
        use_wandb=args.use_wandb,
        wandb_project="seriguela",
        wandb_run_name=f"seriguela-{args.algorithm}-{model_name}-{args.problem}-seed{seed}",
        resume=not args.no_resume,
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

    # Evaluate best expression on TEST SET (separate from training data)
    if results.get("best_expression"):
        logger.info(f"\n--- Evaluating on TEST SET ---")
        test_metrics = evaluate_on_test_set(
            expression_str=results["best_expression"],
            x_test=x_test,
            y_test=y_test,
            is_prefix=is_prefix,
        )
        results.update(test_metrics)

        # Log test results
        if test_metrics["test_valid"]:
            logger.info(f"Test R²: {test_metrics['test_r2']:.6f}")
            logger.info(f"Test MSE: {test_metrics['test_mse']:.6f}")
            logger.info(f"Train R² (best_r2): {results.get('best_r2', 'N/A')}")

            # Compute generalization gap
            train_r2 = results.get("best_r2", 0)
            test_r2 = test_metrics["test_r2"]
            if train_r2 > 0 and test_r2 is not None:
                gen_gap = train_r2 - test_r2
                results["generalization_gap"] = float(gen_gap)
                logger.info(f"Generalization gap: {gen_gap:.6f}")
        else:
            logger.warning(f"Test evaluation failed: {test_metrics['test_error']}")
    else:
        logger.warning("No best expression found - skipping test evaluation")
        results["test_r2"] = None
        results["test_mse"] = None
        results["test_valid"] = False
        results["test_error"] = "No best expression"

    # Add seed to results
    results["seed"] = seed
    results["problem"] = args.problem
    results["model"] = args.model

    # Upload to HuggingFace if requested
    if args.upload_hf:
        try:
            from utils.hf_upload import upload_results
            url = upload_results(
                results=results,
                algorithm=args.algorithm,
                model_name=model_name,
                problem=args.problem,
                seed=seed,
                experiment_type="benchmark",
            )
            if url:
                logger.info(f"Results uploaded to: {url}")
                results["hf_url"] = url
        except Exception as e:
            logger.warning(f"Failed to upload to HuggingFace: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run symbolic regression RL experiment")

    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Algorithm
    parser.add_argument("--algorithm", choices=["bon_ppo", "bon_grpo", "pure_ppo", "pure_grpo", "best_of_n"],
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

    # Prompt type (for robustness testing)
    parser.add_argument("--prompt_type", choices=["standard", "oracle", "distractor"],
                        default="standard", help="Prompt type for robustness testing")

    # Noise (for robustness testing)
    parser.add_argument("--noise_type", choices=["none", "gaussian", "uniform"],
                        default="none", help="Type of noise to add to data")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise level (fraction of signal std)")

    # OOD evaluation
    parser.add_argument("--ood_test", type=str, default=None,
                        help="OOD test to run (e.g., 'near_ood', 'far_ood_right', 'structural_rational')")

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
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch even if checkpoints exist")

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
        best_r2_values = [r["best_r2"] for r in all_results if r.get("best_r2") is not None]
        test_r2_values = [r["test_r2"] for r in all_results if r.get("test_r2") is not None]

        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATE RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Seeds: {args.seeds}")

        # Training R² (on training data)
        if best_r2_values:
            logger.info(f"Train R² - Mean: {np.mean(best_r2_values):.4f}, Std: {np.std(best_r2_values):.4f}")
            logger.info(f"Train R² - Max: {np.max(best_r2_values):.4f}, Min: {np.min(best_r2_values):.4f}")

        # Test R² (on held-out test data)
        if test_r2_values:
            logger.info(f"Test R² - Mean: {np.mean(test_r2_values):.4f}, Std: {np.std(test_r2_values):.4f}")
            logger.info(f"Test R² - Max: {np.max(test_r2_values):.4f}, Min: {np.min(test_r2_values):.4f}")

            # Generalization gap
            if best_r2_values and len(best_r2_values) == len(test_r2_values):
                gen_gaps = [b - t for b, t in zip(best_r2_values, test_r2_values)]
                logger.info(f"Generalization Gap - Mean: {np.mean(gen_gaps):.4f}")

        # Save aggregate results
        aggregate = {
            "seeds": args.seeds,
            "problem": args.problem,
            "algorithm": args.algorithm,
            "model": args.model,
            # Training metrics
            "mean_train_r2": float(np.mean(best_r2_values)) if best_r2_values else None,
            "std_train_r2": float(np.std(best_r2_values)) if best_r2_values else None,
            "max_train_r2": float(np.max(best_r2_values)) if best_r2_values else None,
            "min_train_r2": float(np.min(best_r2_values)) if best_r2_values else None,
            # Test metrics
            "mean_test_r2": float(np.mean(test_r2_values)) if test_r2_values else None,
            "std_test_r2": float(np.std(test_r2_values)) if test_r2_values else None,
            "max_test_r2": float(np.max(test_r2_values)) if test_r2_values else None,
            "min_test_r2": float(np.min(test_r2_values)) if test_r2_values else None,
            # Legacy names for compatibility
            "mean_best_r2": float(np.mean(best_r2_values)) if best_r2_values else None,
            "std_best_r2": float(np.std(best_r2_values)) if best_r2_values else None,
            "individual_results": all_results,
        }

        output_path = Path(args.output_dir) / f"aggregate_{args.algorithm}_{args.problem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aggregate, f, indent=2, default=str)
        logger.info(f"Aggregate results saved to: {output_path}")


if __name__ == "__main__":
    main()
