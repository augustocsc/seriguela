#!/usr/bin/env python3
"""
Train GPT-2 variants with JSON format and early stopping.
FIXED VERSION with proper data format conversion.
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


def convert_to_json_format(example):
    """Convert dataset format to JSON format."""
    text = example['i_prompt_n']

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="gpt2-medium",
                        choices=["gpt2", "gpt2-medium", "gpt2-large"],
                        help="Model size to train")
    parser.add_argument("--dataset_repo", type=str, default="augustocsc/sintetico_natural")
    parser.add_argument("--data_dir", type=str, default="700K")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    args = parser.parse_args()

    # Set output dir
    if args.output_dir is None:
        model_name = args.model_size.replace("-", "_")
        args.output_dir = f"./output/{model_name}_700K_json"

    print("="*80)
    print(f"Training {args.model_size} with JSON format + Early Stopping")
    print("="*80)
    print(f"Output dir: {args.output_dir}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_size)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading {args.model_size}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_size)

    # Add LoRA
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
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    print()

    # Load dataset
    print(f"Loading dataset: {args.dataset_repo}/{args.data_dir}")
    dataset = load_dataset(args.dataset_repo, data_dir=args.data_dir)

    # Check original format
    print("Original format sample:")
    print(dataset["train"][0]['i_prompt_n'][:150])
    print()

    # Convert to JSON format
    print("Converting to JSON format...")
    train_dataset = dataset["train"].map(convert_to_json_format, remove_columns=['i_prompt_n'])

    # Split for validation (10%)
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"Train size: {len(train_dataset):,}")
    print(f"Eval size: {len(eval_dataset):,}")
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

    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments with early stopping
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
    print()
    trainer.train()

    # Save final model
    print(f"\nSaving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Model saved to: {args.output_dir}")
    print(f"Format: JSON (80% valid expressions expected)")


if __name__ == "__main__":
    main()
