"""
Training Configuration and Best Practices for Seriguela Project
================================================================

This module provides standardized training configurations, hyperparameters,
and best practices for training GPT-2 models on symbolic regression tasks.

All configurations are documented for reproducibility and academic reporting.

Author: Seriguela Research Team
Last Updated: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


# ==============================================================================
# STANDARD HYPERPARAMETERS
# ==============================================================================

@dataclass
class StandardTrainingConfig:
    """
    Standard training configuration used across all experiments.

    These hyperparameters have been validated across multiple experiments
    and provide consistent, reproducible results.
    """

    # Model architecture
    model_family: str = "gpt2"  # gpt2, gpt2-medium, gpt2-large

    # LoRA (Parameter-Efficient Fine-Tuning)
    lora_r: int = 8  # LoRA rank - controls adapter capacity
    lora_alpha: int = 32  # LoRA scaling factor (typically 2-4x rank)
    lora_dropout: float = 0.05  # Dropout for LoRA layers
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn"])

    # Training hyperparameters
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4

    # Batch sizes (model-specific, see get_model_config)
    per_device_train_batch_size: int = 8  # Default for Base model
    per_device_eval_batch_size: int = 8

    # Mixed precision training
    fp16: bool = True  # Enable for GPU efficiency
    fp16_opt_level: str = "O1"  # Conservative mixed precision

    # Evaluation and checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100

    # Early stopping
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # Data handling
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Reproducibility
    seed: int = 42

    # Experiment tracking
    report_to: str = "wandb"
    logging_first_step: bool = True


# ==============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ==============================================================================

MODEL_CONFIGS = {
    "gpt2": {
        "parameters": "124M",
        "trainable_params_lora": "294K",
        "recommended_instance": "g5.xlarge",
        "gpu_memory": "24GB",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
        "expected_training_time_hours": "2-3",
        "cost_estimate_usd": "2-3",
    },
    "gpt2-medium": {
        "parameters": "355M",
        "trainable_params_lora": "294K",
        "recommended_instance": "g5.xlarge",
        "gpu_memory": "24GB",
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 16,
        "expected_training_time_hours": "3-4",
        "cost_estimate_usd": "3-4",
    },
    "gpt2-large": {
        "parameters": "774M",
        "trainable_params_lora": "294K",
        "recommended_instance": "g5.2xlarge",
        "gpu_memory": "48GB",
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "expected_training_time_hours": "4-5",
        "cost_estimate_usd": "5-6",
    },
}


def get_model_config(model_size: str) -> Dict:
    """
    Get recommended configuration for specific model size.

    Args:
        model_size: One of "gpt2", "gpt2-medium", "gpt2-large"

    Returns:
        Dictionary with model-specific configuration

    Example:
        >>> config = get_model_config("gpt2-medium")
        >>> print(f"Batch size: {config['per_device_train_batch_size']}")
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. "
                        f"Choose from: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_size]


# ==============================================================================
# DATASET BEST PRACTICES
# ==============================================================================

@dataclass
class DatasetConfig:
    """
    Best practices for dataset loading and splitting.

    CRITICAL: Always use pre-existing train/validation splits from datasets.
    Do NOT perform additional splitting on top of existing splits.
    """

    # Dataset source
    dataset_repo: str = "augustocsc/sintetico_natural_prefix_682k"

    # Split configuration - USE EXISTING SPLITS
    use_existing_splits: bool = True  # ALWAYS True for production
    expected_train_size: int = 682_429
    expected_validation_size: int = 75_826

    # Only used if dataset doesn't have pre-existing splits
    train_val_split_ratio: float = 0.9
    split_seed: int = 42

    # Data column
    text_column: str = "p_prompt_n_converted"  # For prefix notation

    # Tokenization
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"


def validate_dataset_splits(dataset) -> Tuple[int, int]:
    """
    Validate that dataset has correct splits and sizes.

    Args:
        dataset: HuggingFace dataset object

    Returns:
        Tuple of (train_size, validation_size)

    Raises:
        ValueError: If splits are missing or sizes don't match expected

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("augustocsc/sintetico_natural_prefix_682k")
        >>> train_size, val_size = validate_dataset_splits(dataset)
        >>> print(f"Train: {train_size}, Validation: {val_size}")
    """
    expected_train = 682_429
    expected_val = 75_826

    if "train" not in dataset:
        raise ValueError("Dataset missing 'train' split!")
    if "validation" not in dataset:
        raise ValueError("Dataset missing 'validation' split!")

    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])

    if train_size != expected_train:
        raise ValueError(
            f"Train split size mismatch: got {train_size}, "
            f"expected {expected_train}"
        )

    if val_size != expected_val:
        raise ValueError(
            f"Validation split size mismatch: got {val_size}, "
            f"expected {expected_val}"
        )

    return train_size, val_size


# ==============================================================================
# TRAINING MONITORING AND LOGGING
# ==============================================================================

@dataclass
class MonitoringConfig:
    """
    Configuration for experiment tracking and monitoring.
    """

    # Wandb configuration
    wandb_project: str = "seriguela"
    wandb_entity: Optional[str] = None

    # Run naming convention: seriguela-{type}-{model}-{dataset}-{timestamp}
    # Examples:
    #   - seriguela-supervised-base-700k-20260203-143022
    #   - seriguela-ppo-medium-nguyen5-20260203-143022

    # Metrics to track
    primary_metrics: List[str] = field(default_factory=lambda: [
        "train/loss",
        "eval/loss",
        "train/learning_rate",
        "train/epoch",
    ])

    # GPU monitoring
    log_gpu_memory: bool = True
    log_gpu_utilization: bool = True

    # Training speed metrics
    expected_steps_per_second: Dict[str, float] = field(default_factory=lambda: {
        "gpt2": 0.4,  # ~2.5s per step
        "gpt2-medium": 0.35,  # ~3s per step
        "gpt2-large": 0.25,  # ~4s per step
    })

    def get_expected_speed(self, model_size: str) -> float:
        """Get expected training speed (steps/sec) for model size."""
        return self.expected_steps_per_second.get(model_size, 0.3)


def check_training_speed(steps_per_second: float, model_size: str) -> str:
    """
    Check if training speed is within expected range.

    Args:
        steps_per_second: Measured training speed
        model_size: Model identifier (gpt2, gpt2-medium, gpt2-large)

    Returns:
        Status message: "OK", "SLOW", or "VERY_SLOW"

    Example:
        >>> speed = 0.06  # 17 seconds per step
        >>> status = check_training_speed(speed, "gpt2")
        >>> print(status)  # "VERY_SLOW"
    """
    config = MonitoringConfig()
    expected = config.get_expected_speed(model_size)

    if steps_per_second >= expected * 0.8:
        return "OK"
    elif steps_per_second >= expected * 0.3:
        return "SLOW"
    else:
        return "VERY_SLOW"


# ==============================================================================
# AWS INFRASTRUCTURE CONFIGURATION
# ==============================================================================

@dataclass
class AWSConfig:
    """
    Standard AWS configuration for training instances.
    """

    # AMI and region
    ami_id: str = "ami-0c7217cdde317cfec"  # Ubuntu Deep Learning AMI
    region: str = "us-east-1"

    # Security
    key_name: str = "chave-gpu-nova"
    security_group: str = "sg-0deaa73e23482e3f6"

    # Instance types
    instance_types: Dict[str, str] = field(default_factory=lambda: {
        "gpt2": "g5.xlarge",
        "gpt2-medium": "g5.xlarge",
        "gpt2-large": "g5.2xlarge",
    })

    # Storage
    volume_sizes: Dict[str, int] = field(default_factory=lambda: {
        "gpt2": 100,
        "gpt2-medium": 100,
        "gpt2-large": 120,
    })
    volume_type: str = "gp3"

    # Git repository
    github_repo: str = "https://github.com/augustocsc/seriguela.git"
    branch: str = "experiment/ppo-symbolic-regression"


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_training_summary(model_size: str, dataset_size: str) -> None:
    """
    Print comprehensive training configuration summary.

    Args:
        model_size: Model identifier (gpt2, gpt2-medium, gpt2-large)
        dataset_size: Dataset identifier (700K, nguyen5, etc.)
    """
    config = get_model_config(model_size)
    training_config = StandardTrainingConfig()

    print("=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nModel: {model_size} ({config['parameters']} parameters)")
    print(f"Trainable (LoRA): {config['trainable_params_lora']}")
    print(f"Dataset: {dataset_size}")
    print(f"\nAWS Instance: {config['recommended_instance']} ({config['gpu_memory']} VRAM)")
    print(f"Expected Time: {config['expected_training_time_hours']} hours")
    print(f"Estimated Cost: ${config['cost_estimate_usd']} USD")

    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  Batch Size (per device): {config['per_device_train_batch_size']}")
    print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective Batch Size: {config['effective_batch_size']}")
    print(f"  Warmup Steps: {training_config.warmup_steps}")
    print(f"  Weight Decay: {training_config.weight_decay}")

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {training_config.lora_r}")
    print(f"  Alpha: {training_config.lora_alpha}")
    print(f"  Dropout: {training_config.lora_dropout}")
    print(f"  Target Modules: {training_config.lora_target_modules}")

    print(f"\nEarly Stopping:")
    print(f"  Patience: {training_config.early_stopping_patience} epochs")
    print(f"  Metric: {training_config.metric_for_best_model}")

    print(f"\nMixed Precision:")
    print(f"  FP16: {training_config.fp16}")

    print(f"\nReproducibility:")
    print(f"  Seed: {training_config.seed}")

    print("=" * 80)


def get_training_args_dict(model_size: str, output_dir: str) -> Dict:
    """
    Get complete training arguments dictionary for Trainer API.

    Args:
        model_size: Model identifier (gpt2, gpt2-medium, gpt2-large)
        output_dir: Output directory for checkpoints

    Returns:
        Dictionary compatible with transformers.TrainingArguments

    Example:
        >>> args = get_training_args_dict("gpt2-medium", "./output/medium")
        >>> from transformers import TrainingArguments
        >>> training_args = TrainingArguments(**args)
    """
    config = get_model_config(model_size)
    standard = StandardTrainingConfig()

    return {
        "output_dir": output_dir,
        "num_train_epochs": standard.num_train_epochs,
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": standard.learning_rate,
        "weight_decay": standard.weight_decay,
        "warmup_steps": standard.warmup_steps,
        "fp16": standard.fp16,
        "logging_steps": standard.logging_steps,
        "eval_steps": standard.eval_steps,
        "save_steps": standard.save_steps,
        "save_total_limit": standard.save_total_limit,
        "eval_strategy": standard.eval_strategy,
        "load_best_model_at_end": standard.load_best_model_at_end,
        "metric_for_best_model": standard.metric_for_best_model,
        "greater_is_better": standard.greater_is_better,
        "report_to": standard.report_to,
        "logging_first_step": standard.logging_first_step,
        "dataloader_num_workers": standard.dataloader_num_workers,
        "dataloader_pin_memory": standard.dataloader_pin_memory,
        "seed": standard.seed,
    }


def get_lora_config_dict() -> Dict:
    """
    Get LoRA configuration dictionary for PEFT library.

    Returns:
        Dictionary compatible with peft.LoraConfig

    Example:
        >>> from peft import LoraConfig
        >>> lora_config = LoraConfig(**get_lora_config_dict())
    """
    standard = StandardTrainingConfig()

    return {
        "r": standard.lora_r,
        "lora_alpha": standard.lora_alpha,
        "lora_dropout": standard.lora_dropout,
        "target_modules": standard.lora_target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


# ==============================================================================
# ACADEMIC REPORTING HELPERS
# ==============================================================================

def generate_methods_section() -> str:
    """
    Generate Methods section text for academic paper.

    Returns:
        Formatted text describing training methodology
    """
    standard = StandardTrainingConfig()

    return f"""
Training Methodology

Models were fine-tuned using LoRA (Low-Rank Adaptation) with the following
hyperparameters: rank r={standard.lora_r}, scaling factor α={standard.lora_alpha},
and dropout rate {standard.lora_dropout}. LoRA adapters were applied to attention
layers only ({standard.lora_target_modules}), resulting in approximately 294K
trainable parameters while keeping the base model frozen.

Training was performed for {standard.num_train_epochs} epochs with a learning rate
of {standard.learning_rate}, weight decay of {standard.weight_decay}, and
{standard.warmup_steps} warmup steps. We used mixed-precision training (FP16) for
computational efficiency. Early stopping with patience of {standard.early_stopping_patience}
epochs was employed to prevent overfitting.

Batch sizes were adjusted for each model size to optimize GPU memory utilization:
- GPT-2 Base (124M): batch size 8 with gradient accumulation of 4 steps (effective batch size 32)
- GPT-2 Medium (355M): batch size 4 with gradient accumulation of 4 steps (effective batch size 16)
- GPT-2 Large (774M): batch size 2 with gradient accumulation of 4 steps (effective batch size 8)

All experiments used seed {standard.seed} for reproducibility. Training was conducted
on AWS EC2 instances with NVIDIA A10G GPUs.
"""


def generate_hyperparameters_table() -> str:
    """
    Generate LaTeX table of hyperparameters for paper.

    Returns:
        LaTeX table code
    """
    standard = StandardTrainingConfig()

    return f"""
\\begin{{table}}[ht]
\\centering
\\caption{{Training Hyperparameters}}
\\label{{tab:hyperparameters}}
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Value}} \\\\
\\midrule
Learning Rate & {standard.learning_rate} \\\\
Number of Epochs & {standard.num_train_epochs} \\\\
Warmup Steps & {standard.warmup_steps} \\\\
Weight Decay & {standard.weight_decay} \\\\
LoRA Rank (r) & {standard.lora_r} \\\\
LoRA Alpha (α) & {standard.lora_alpha} \\\\
LoRA Dropout & {standard.lora_dropout} \\\\
Early Stopping Patience & {standard.early_stopping_patience} \\\\
Mixed Precision & FP16 \\\\
Random Seed & {standard.seed} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# ==============================================================================
# VALIDATION AND ERROR CHECKING
# ==============================================================================

def validate_training_environment() -> List[str]:
    """
    Validate training environment and return list of warnings/errors.

    Returns:
        List of warning/error messages (empty if all OK)
    """
    issues = []

    # Check CUDA availability
    if not torch.cuda.is_available():
        issues.append("CRITICAL: CUDA not available. GPU training required.")

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem_gb < 20:
            issues.append(f"WARNING: GPU memory ({gpu_mem_gb:.1f}GB) may be insufficient for training.")

    return issues


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Print configuration for Medium model
    print_training_summary("gpt2-medium", "prefix_682k")

    # Validate environment
    issues = validate_training_environment()
    if issues:
        print("\nEnvironment Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Training environment validated successfully")

    # Show methods section for paper
    print("\n" + "=" * 80)
    print("METHODS SECTION FOR PAPER")
    print("=" * 80)
    print(generate_methods_section())
