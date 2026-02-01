#!/usr/bin/env python3
"""
Training script for expression generation experiments.

Supports two formats:
- EXP-A (JSON): Uses custom <|endofex|> token
- EXP-B (EOS): Uses native GPT-2 <|endoftext|> token

Usage:
    # EXP-A (JSON format)
    python scripts/train_experiment.py \
        --experiment_name exp_a_json \
        --train_file ./data/experiments/exp_a_json/train.csv \
        --output_dir ./output/exp_a_json \
        --end_marker "<|endofex|>"

    # EXP-B (EOS format)
    python scripts/train_experiment.py \
        --experiment_name exp_b_eos \
        --train_file ./data/experiments/exp_b_eos/train.csv \
        --output_dir ./output/exp_b_eos \
        --end_marker "<|endoftext|>" \
        --use_native_eos
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tokenize_function(examples, tokenizer):
    """Tokenize the text field."""
    return tokenizer(examples["text"])


def group_texts(examples, block_size):
    """Group texts into blocks of block_size."""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        logger.warning(f"Total length ({total_length}) < block_size ({block_size})")

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def validate_data_format(dataset, tokenizer, end_marker, num_samples=10, is_json_format=False):
    """Validate that training data is in the expected format."""
    import json as json_module

    if is_json_format:
        logger.info("Validating JSON format data...")
        marker_to_check = '"expr":'  # JSON format has expr field
    else:
        logger.info(f"Validating data contains '{end_marker}'...")
        marker_to_check = end_marker

    sample_indices = random.sample(
        range(len(dataset)),
        min(num_samples, len(dataset))
    )

    valid_count = 0
    for idx in sample_indices:
        text = dataset[idx]["text"]
        if is_json_format:
            # For JSON format, validate it's valid JSON with expr field
            try:
                obj = json_module.loads(text)
                if "expr" in obj and "vars" in obj:
                    valid_count += 1
            except:
                pass
        else:
            # For EOS format, check marker presence
            if marker_to_check in text:
                valid_count += 1

    rate = valid_count / len(sample_indices) * 100
    logger.info(f"Validation: {valid_count}/{len(sample_indices)} ({rate:.1f}%) valid")

    if valid_count == 0:
        logger.error("No valid samples found! Data not properly prepared.")
        sys.exit(1)

    return rate


def main():
    parser = argparse.ArgumentParser(
        description="Train expression generation model"
    )

    # Required arguments
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name (e.g., 'exp_a_json', 'exp_b_eos')")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model")

    # Format options
    parser.add_argument("--end_marker", type=str, default="<|endofex|>",
                        help="End marker token (e.g., '<|endofex|>' or '<|endoftext|>')")
    parser.add_argument("--use_native_eos", action="store_true",
                        help="Use native GPT-2 EOS token instead of custom token")
    parser.add_argument("--json_format", action="store_true",
                        help="Data is in JSON format (for EXP-A)")

    # Optional data arguments
    parser.add_argument("--validation_file", type=str, default=None,
                        help="Path to validation CSV file")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test CSV file")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Base model name")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for tokenization")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="seriguela_experiments",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hub model ID for pushing")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Set seed
    set_seed(args.seed)

    # Configure wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key)

    wandb_run_name = args.wandb_run_name or args.experiment_name
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config=vars(args)
    )

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {args.experiment_name}")
    logger.info("=" * 60)
    logger.info(f"End marker: {args.end_marker}")
    logger.info(f"Use native EOS: {args.use_native_eos}")
    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    if args.test_file:
        data_files["test"] = args.test_file

    raw_datasets = load_dataset("csv", data_files=data_files)
    logger.info(f"Loaded dataset: {raw_datasets}")

    # Validate data format
    validate_data_format(
        raw_datasets["train"],
        tokenizer=None,
        end_marker=args.end_marker,
        is_json_format=args.json_format
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens based on experiment type
    if args.use_native_eos:
        # EXP-B: Use native EOS token, no special tokens needed
        logger.info("Using native GPT-2 EOS token (<|endoftext|>)")
        end_token_id = tokenizer.eos_token_id
        logger.info(f"EOS token ID: {end_token_id}")
    else:
        # EXP-A: Add custom <|endofex|> token
        logger.info("Adding custom special tokens")
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|startofex|>", "<|endofex|>"]
        })
        end_token_id = tokenizer.convert_tokens_to_ids("<|endofex|>")
        logger.info(f"Custom end token ID: {end_token_id}")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # Group into blocks
    logger.info(f"Grouping texts into blocks of {args.block_size}...")
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, args.block_size),
        batched=True
    )

    logger.info(f"Processed dataset: {lm_datasets}")

    # Validate processed data has end markers
    logger.info("Validating processed data...")
    sample_indices = random.sample(
        range(len(lm_datasets["train"])),
        min(10, len(lm_datasets["train"]))
    )

    valid_count = 0
    for idx in sample_indices:
        sample = lm_datasets["train"][idx]
        decoded = tokenizer.decode(sample["input_ids"])
        if args.end_marker in decoded:
            valid_count += 1

    logger.info(f"Processed data validation: {valid_count}/{len(sample_indices)} contain end marker")

    if valid_count == 0:
        logger.error("No processed samples contain end marker! Check data format.")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Resize embeddings if using custom tokens
    if not args.use_native_eos:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings to {len(tokenizer)}")

    # Configure EOS token for generation
    model.config.eos_token_id = end_token_id
    logger.info(f"Model EOS token ID: {model.config.eos_token_id}")

    # Apply LoRA
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # Training arguments
    logger.info("Configuring training...")

    has_validation = "validation" in lm_datasets and len(lm_datasets["validation"]) > 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        fp16=args.fp16,
        report_to="wandb",
        run_name=wandb_run_name,
        seed=args.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    logger.info("Initializing Trainer...")
    callbacks = []
    if has_validation:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    try:
        train_result = trainer.train()

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

        # Save experiment info
        import json
        exp_info = {
            "experiment_name": args.experiment_name,
            "end_marker": args.end_marker,
            "use_native_eos": args.use_native_eos,
            "train_file": args.train_file,
            "end_token_id": end_token_id,
            "final_loss": metrics.get("train_loss", None),
        }
        with open(os.path.join(args.output_dir, "experiment_info.json"), "w") as f:
            json.dump(exp_info, f, indent=2)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
        logger.info(f"Model saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        wandb.finish()

    # Push to Hub if requested
    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing to Hub: {args.hub_model_id}")
        try:
            trainer.push_to_hub(commit_message=f"Training complete: {args.experiment_name}")
            logger.info("Push successful!")
        except Exception as e:
            logger.error(f"Push failed: {e}")


if __name__ == "__main__":
    main()
