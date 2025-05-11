# finetuning/arguments.py

import argparse
from . import config  # Import constants from config.py
from .utils import logger # Import logger

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 model using PEFT (LoRA) on an equation dataset."
    )

    # Model & Data Args
    parser.add_argument("--model_name_or_path", type=str, default=config.DEFAULT_MODEL_NAME,
                        help="Pretrained model name or path (e.g., 'gpt2', 'gpt2-medium').")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="Hugging Face Hub repository ID for the dataset (e.g., 'username/my-equation-dataset').")
    parser.add_argument("--data_dir", type=str, default=config.DEFAULT_DATA_DIR,
                        help="Directory containing the dataset files within the repo (optional).")
    parser.add_argument("--source_data_column", type=str, default=config.DEFAULT_SOURCE_DATA_COLUMN,
                        help="Column name in the *source* dataset to use for training (will be renamed to 'text').")
    parser.add_argument("--block_size", type=int, default=config.DEFAULT_BLOCK_SIZE,
                        help="Block size for tokenizing and chunking.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=config.DEFAULT_BATCH_SIZE,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=config.DEFAULT_BATCH_SIZE,
                        help="Batch size per device during evaluation.")
    parser.add_argument("--learning_rate", type=float, default=config.DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default=config.DEFAULT_LR_SCHEDULER_TYPE,
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.DEFAULT_GRAD_ACCUM_STEPS,
                        help="Steps for gradient accumulation.")
    parser.add_argument("--warmup_steps", type=int, default=config.DEFAULT_WARMUP_STEPS, help="Learning rate scheduler warmup steps.")

    # LoRA / PEFT Parameters
    parser.add_argument("--lora_r", type=int, default=config.DEFAULT_LORA_R, help="LoRA rank (dimension).")
    parser.add_argument("--lora_alpha", type=int, default=config.DEFAULT_LORA_ALPHA, help="LoRA alpha (scaling factor).")
    parser.add_argument("--lora_dropout", type=float, default=config.DEFAULT_LORA_DROPOUT, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", nargs='+', default=config.DEFAULT_LORA_TARGET_MODULES,
                        help="Module names to apply LoRA to (e.g., 'c_attn' for GPT-2 query/key/value).")
    parser.add_argument("--lora_bias", type=str, default=config.DEFAULT_LORA_BIAS, choices=["none", "all", "lora_only"],
                        help="Bias type for LoRA.")

    # Logging, Saving & Evaluation Args
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model, checkpoints, and logs.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory if it exists.")
    parser.add_argument("--logging_steps", type=int, default=config.DEFAULT_LOGGING_STEPS, help="Log training metrics every N steps.")
    parser.add_argument("--eval_steps", type=int, default=config.DEFAULT_SAVE_EVAL_STEPS,
                        help="Evaluate every N steps (if eval_strategy='steps').")
    parser.add_argument("--save_steps", type=int, default=config.DEFAULT_SAVE_EVAL_STEPS,
                        help="Save checkpoint every N steps (if save_strategy='steps').")
    parser.add_argument("--eval_strategy", type=str, default=config.DEFAULT_EVAL_STRATEGY, choices=["steps", "epoch", "no"], help="Evaluation strategy.")
    parser.add_argument("--save_strategy", type=str, default=config.DEFAULT_SAVE_STRATEGY, choices=["steps", "epoch", "no"],
                        help="Checkpoint saving strategy.")
    parser.add_argument("--save_total_limit", type=int, default=config.DEFAULT_SAVE_TOTAL_LIMIT,
                        help="Limit the total number of checkpoints saved.")
    parser.add_argument("--load_best_model_at_end", action='store_true',
                        help="Load the best model (based on evaluation loss) at the end.")
    parser.add_argument("--early_stopping_patience", type=int, default=config.DEFAULT_EARLY_STOPPING_PATIENCE,
                        help="Number of evaluations with no improvement to trigger early stopping. Requires load_best_model_at_end.")

    # Technical Args
    parser.add_argument("--fp16", action='store_true', help="Use mixed precision training (FP16).")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--report_to", type=str, default=config.DEFAULT_REPORT_TO, choices=["tensorboard", "wandb", "none"],
                        help="Where to report metrics.")
    parser.add_argument("--run_name", type=str, default=config.DEFAULT_RUN_NAME,
                        help="Name of the run for logging purposes.")

    # Hugging Face Hub Args
    parser.add_argument("--push_to_hub", action='store_true', help="Push the final model to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Repository ID for pushing (e.g., 'username/gpt2-finetuned-equations'). Required if --push_to_hub.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.push_to_hub and not args.hub_model_id:
        logger.error("--hub_model_id is required when --push_to_hub is set.")
        raise ValueError("--hub_model_id is required when --push_to_hub is set.")
    if args.early_stopping_patience is not None and args.early_stopping_patience > 0 and not args.load_best_model_at_end:
        logger.warning("--early_stopping_patience is set, but --load_best_model_at_end is False. Early stopping requires loading the best model.")
    if args.eval_strategy == "no" and (args.load_best_model_at_end or (args.early_stopping_patience is not None and args.early_stopping_patience > 0)):
        logger.error("Cannot use --load_best_model_at_end or --early_stopping_patience without evaluation (set --eval_strategy to 'steps' or 'epoch').")
        raise ValueError("Cannot use --load_best_model_at_end or --early_stopping_patience without evaluation.")

    return args