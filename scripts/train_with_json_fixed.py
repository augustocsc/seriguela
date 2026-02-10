#!/usr/bin/env python3
"""
Train GPT-2 variants with JSON format and early stopping.
FIXED VERSION - Uses pre-existing train/validation splits (NO double-split).

CRITICAL FIX: This version correctly uses the dataset's existing validation split
instead of creating a new one, avoiding the double-split problem that causes
extremely slow training (17s/step instead of 2-3s/step).

Dataset structure:
  - Train: 682,429 examples
  - Validation: 75,826 examples
  - Total: 758,255 examples

Author: Seriguela Research Team
Last Updated: February 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Import standardized configurations
try:
    from scripts.training_config import (
        get_model_config,
        get_training_args_dict,
        get_lora_config_dict,
        validate_dataset_splits,
        print_training_summary,
    )
    USE_STANDARD_CONFIG = True
except ImportError:
    print("Warning: training_config.py not found. Using inline defaults.")
    USE_STANDARD_CONFIG = False


def convert_to_json_format(example, text_column='i_prompt_n'):
    """
    Convert dataset format to JSON format.

    Args:
        example: Dataset example
        text_column: Name of the text column to convert

    Returns:
        Dictionary with 'text' key containing JSON string

    Example input (text format):
        vars: x_1, x_2
        oper: *, +, sin
        cons: C
        expr: sin(x_1 + C*x_2)

    Example output (JSON):
        {"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "sin(x_1 + C*x_2)"}
    """
    text = example[text_column]

    # Parse the text format
    lines = text.strip().split('\n')
    data = {}

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if key == 'vars':
                # Convert "x_1, x_2, x_3" to ["x_1", "x_2", "x_3"]
                data['vars'] = [v.strip() for v in value.split(',')]
            elif key == 'oper':
                # Convert "*, +, sin" to ["*", "+", "sin"]
                data['ops'] = [o.strip() for o in value.split(',')]
            elif key == 'cons':
                data['cons'] = value
            elif key == 'expr':
                data['expr'] = value

    # Convert to JSON string
    json_str = json.dumps(data, ensure_ascii=False)

    return {'text': json_str}


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT-2 models with proper train/validation splits"
    )
    parser.add_argument("--model_size", type=str, default="gpt2-medium",
                        choices=["gpt2", "gpt2-medium", "gpt2-large"],
                        help="Model size to train")
    parser.add_argument("--dataset_repo", type=str,
                        default="augustocsc/sintetico_natural_prefix_682k",
                        help="HuggingFace dataset repository")
    parser.add_argument("--text_column", type=str, default="p_prompt_n_converted",
                        help="Name of text column in dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                        help="Batch size (auto-set based on model size if not specified)")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    args = parser.parse_args()

    # Auto-configure batch size based on model size
    if args.per_device_train_batch_size is None:
        if USE_STANDARD_CONFIG:
            model_config = get_model_config(args.model_size)
            args.per_device_train_batch_size = model_config["per_device_train_batch_size"]
        else:
            # Fallback defaults
            batch_sizes = {"gpt2": 8, "gpt2-medium": 4, "gpt2-large": 2}
            args.per_device_train_batch_size = batch_sizes[args.model_size]

    # Set output dir
    if args.output_dir is None:
        model_name = args.model_size.replace("-", "_")
        dataset_name = args.dataset_repo.split('/')[-1]
        args.output_dir = f"./output/{model_name}_{dataset_name}"

    print("=" * 80)
    print(f"Training {args.model_size} with JSON format + Early Stopping")
    print("FIXED VERSION - Using pre-existing train/validation splits")
    print("=" * 80)
    print(f"Dataset: {args.dataset_repo}")
    print(f"Text column: {args.text_column}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print()

    # Print standardized configuration summary if available
    if USE_STANDARD_CONFIG:
        dataset_name = args.dataset_repo.split('/')[-1]
        print_training_summary(args.model_size, dataset_name)
        print()

    # Load tokenizer
    print(f"Loading tokenizer for {args.model_size}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_size)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading {args.model_size}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_size)

    # Add LoRA
    if USE_STANDARD_CONFIG:
        lora_config_dict = get_lora_config_dict()
        lora_config = LoraConfig(**lora_config_dict)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    print()

    # Load dataset - CRITICAL: Load entire dataset with all splits
    print(f"Loading dataset: {args.dataset_repo}")
    try:
        # Try loading without data_dir first (for new datasets)
        dataset = load_dataset(args.dataset_repo)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load with data_dir parameter...")
        # Fallback to data_dir if needed
        dataset = load_dataset(args.dataset_repo, data_dir="700K")

    # CRITICAL: Validate that dataset has correct splits
    print("\nDataset structure:")
    print(f"  Available splits: {list(dataset.keys())}")

    if "train" not in dataset or "validation" not in dataset:
        raise ValueError(
            f"Dataset must have 'train' and 'validation' splits! "
            f"Found: {list(dataset.keys())}"
        )

    print(f"  Train size: {len(dataset['train']):,}")
    print(f"  Validation size: {len(dataset['validation']):,}")
    print(f"  Total: {len(dataset['train']) + len(dataset['validation']):,}")

    # Validate split sizes if using standard config
    if USE_STANDARD_CONFIG:
        try:
            validate_dataset_splits(dataset)
            print("✓ Dataset splits validated successfully")
        except ValueError as e:
            print(f"WARNING: {e}")
            print("Continuing anyway...")

    # Check original format
    print(f"\nOriginal format sample (column '{args.text_column}'):")
    print(dataset["train"][0][args.text_column][:150])
    print()

    # CRITICAL FIX: Convert BOTH splits to JSON format
    # Do NOT create a new split - use existing validation split
    print("Converting train split to JSON format...")
    train_dataset = dataset["train"].map(
        lambda x: convert_to_json_format(x, args.text_column),
        remove_columns=[args.text_column]
    )

    print("Converting validation split to JSON format...")
    eval_dataset = dataset["validation"].map(
        lambda x: convert_to_json_format(x, args.text_column),
        remove_columns=[args.text_column]
    )

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Validation: {len(eval_dataset):,}")
    print()

    print("JSON format sample:")
    print(train_dataset[0]['text'][:150])
    print()

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False,
        )

    print("Tokenizing train dataset...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    print("Tokenizing validation dataset...")
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    print(f"Tokenization complete.")
    print(f"  Train: {len(train_tokenized):,} examples")
    print(f"  Validation: {len(eval_tokenized):,} examples")
    print()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments with early stopping
    if USE_STANDARD_CONFIG:
        training_args_dict = get_training_args_dict(args.model_size, args.output_dir)
        # Override with command-line args if provided
        training_args_dict["num_train_epochs"] = args.num_train_epochs
        training_args_dict["learning_rate"] = args.learning_rate
        training_args = TrainingArguments(**training_args_dict)
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=True,
            report_to="wandb",
        )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.001,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    # Train
    print("Starting training with early stopping...")
    print(f"Expected speed: ~2-3 seconds per step")
    print(f"If training is slower than 5s/step, something is wrong!")
    print()
    trainer.train()

    # Save final model
    print(f"\nSaving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Model saved to: {args.output_dir}")
    print(f"Format: JSON (80% valid expressions expected)")
    print(f"Train size: {len(train_dataset):,}")
    print(f"Validation size: {len(eval_dataset):,}")


if __name__ == "__main__":
    main()
