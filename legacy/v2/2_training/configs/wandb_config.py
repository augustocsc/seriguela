"""
Wandb Configuration and Naming Standards for Seriguela Project

This module provides standardized naming conventions for Wandb experiment tracking.
"""

import os
from datetime import datetime
from typing import Optional


# Default Wandb project name
DEFAULT_PROJECT = "seriguela"

# Alternative project name for experiments
EXPERIMENTS_PROJECT = "seriguela-experiments"


def get_wandb_project_name(use_experiments: bool = False) -> str:
    """
    Get the standard Wandb project name.

    Args:
        use_experiments: If True, use experiments project name

    Returns:
        Project name string
    """
    return EXPERIMENTS_PROJECT if use_experiments else DEFAULT_PROJECT


def generate_run_name(
    experiment_type: str,
    model_size: str = "base",
    dataset: Optional[str] = None,
    extra_info: Optional[str] = None,
    include_timestamp: bool = True
) -> str:
    """
    Generate a standardized Wandb run name.

    Naming Convention: seriguela-{type}-{model}-{dataset}-{extra}-{timestamp}

    Args:
        experiment_type: Type of experiment (supervised, ppo, grpo, reinforce, iterative-sft)
        model_size: Model size (base, medium, large) or full name (gpt2, gpt2-medium)
        dataset: Dataset identifier (700K, nguyen5, nguyen7, etc)
        extra_info: Additional information (optional)
        include_timestamp: Whether to include timestamp suffix

    Returns:
        Formatted run name

    Examples:
        >>> generate_run_name("supervised", "medium", "700K")
        'seriguela-supervised-medium-700K-20260203-143022'

        >>> generate_run_name("ppo", "base", "nguyen5", "lr3e5")
        'seriguela-ppo-base-nguyen5-lr3e5-20260203-143022'

        >>> generate_run_name("grpo", "large", "nguyen7", include_timestamp=False)
        'seriguela-grpo-large-nguyen7'
    """
    # Normalize model size
    model_map = {
        "gpt2": "base",
        "gpt2-base": "base",
        "124m": "base",
        "gpt2-medium": "medium",
        "355m": "medium",
        "gpt2-large": "large",
        "774m": "large"
    }
    model_size = model_map.get(model_size.lower(), model_size.lower())

    # Build run name parts
    parts = ["seriguela", experiment_type.lower()]

    # Add model size
    parts.append(model_size)

    # Add dataset if provided
    if dataset:
        parts.append(dataset.lower().replace("_", "").replace("-", ""))

    # Add extra info if provided
    if extra_info:
        parts.append(extra_info.lower().replace("_", ""))

    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts.append(timestamp)

    return "-".join(parts)


def get_run_tags(
    experiment_type: str,
    model_size: str,
    dataset: Optional[str] = None,
    success: Optional[bool] = None
) -> list:
    """
    Generate standardized tags for Wandb runs.

    Args:
        experiment_type: Type of experiment
        model_size: Model size
        dataset: Dataset name
        success: Whether experiment was successful (optional)

    Returns:
        List of tags

    Examples:
        >>> get_run_tags("ppo", "medium", "nguyen5", True)
        ['ppo', 'gpt2-medium', 'nguyen5', 'rl', 'success']
    """
    tags = [experiment_type.lower()]

    # Add model size
    if model_size.lower() in ["base", "124m", "gpt2"]:
        tags.append("gpt2-base")
    elif model_size.lower() in ["medium", "355m", "gpt2-medium"]:
        tags.append("gpt2-medium")
    elif model_size.lower() in ["large", "774m", "gpt2-large"]:
        tags.append("gpt2-large")
    else:
        tags.append(model_size.lower())

    # Add dataset
    if dataset:
        tags.append(dataset.lower())

    # Add category based on experiment type
    if experiment_type.lower() in ["ppo", "grpo", "reinforce"]:
        tags.append("rl")
    elif experiment_type.lower() in ["supervised", "sft"]:
        tags.append("supervised")
    elif experiment_type.lower() == "iterative-sft":
        tags.append("iterative")

    # Add success tag if provided
    if success is not None:
        tags.append("success" if success else "failed")

    return tags


# Common experiment types
EXPERIMENT_TYPES = {
    "SUPERVISED": "supervised",
    "SFT": "sft",
    "PPO": "ppo",
    "GRPO": "grpo",
    "REINFORCE": "reinforce",
    "ITERATIVE_SFT": "iterative-sft",
    "BEST_OF_N": "best-of-n",
    "EVALUATION": "eval"
}

# Common datasets
DATASETS = {
    "MAIN_700K": "700K",
    "NGUYEN_1": "nguyen1",
    "NGUYEN_5": "nguyen5",
    "NGUYEN_7": "nguyen7",
    "NGUYEN_10": "nguyen10",
    "CUSTOM": "custom"
}


def setup_wandb_env():
    """
    Setup Wandb environment from credentials file.
    Reads from ~/.tokens.txt if available.
    """
    tokens_file = os.path.expanduser("~/.tokens.txt")
    if os.path.exists(tokens_file):
        with open(tokens_file) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key.lower() == "wandb":
                        os.environ["WANDB_API_KEY"] = value
                        print(f"[OK] Wandb API key loaded from {tokens_file}")
                        return True

    # Check if already in environment
    if "WANDB_API_KEY" in os.environ:
        print("[OK] Wandb API key found in environment")
        return True

    print("[WARN] Wandb API key not found. Run 'wandb login' or add to ~/.tokens.txt")
    return False


if __name__ == "__main__":
    # Example usage
    print("Wandb Configuration Examples:\n")

    print("1. Supervised training on 700K dataset:")
    print(f"   {generate_run_name('supervised', 'medium', '700K')}\n")

    print("2. PPO on Nguyen-5 benchmark:")
    print(f"   {generate_run_name('ppo', 'base', 'nguyen5')}\n")

    print("3. GRPO with custom learning rate:")
    print(f"   {generate_run_name('grpo', 'large', 'nguyen7', 'lr5e5')}\n")

    print("4. Evaluation run (no timestamp):")
    print(f"   {generate_run_name('eval', 'medium', 'nguyen5', include_timestamp=False)}\n")

    print("5. Tags example:")
    print(f"   {get_run_tags('ppo', 'medium', 'nguyen5', True)}\n")

    print("6. Setup Wandb environment:")
    setup_wandb_env()
