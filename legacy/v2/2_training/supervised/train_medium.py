#!/usr/bin/env python3
"""
Train GPT-2 Medium (355M) on expression dataset to compare with base GPT-2 (124M).
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
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="gpt2-medium",
                        choices=["gpt2", "gpt2-medium", "gpt2-large"],
                        help="Model size to train")
    parser.add_argument("--dataset_repo", type=str, default="augustocsc/sintetico_natural")
    parser.add_argument("--data_dir", type=str, default="700K")
    parser.add_argument("--data_column", type=str, default="i_prompt_n")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    # Set output dir based on model size
    if args.output_dir is None:
        model_name = args.model_size.replace("-", "_")
        args.output_dir = f"./output/{model_name}_700K_json"

    print("="*80)
    print(f"Training {args.model_size} on expression dataset")
    print("="*80)
    print(f"Output dir: {args.output_dir}")
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

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    print()

    # Load dataset
    print(f"Loading dataset: {args.dataset_repo}/{args.data_dir}")
    dataset = load_dataset(args.dataset_repo, data_dir=args.data_dir)
    train_dataset = dataset["train"]

    print(f"Dataset size: {len(train_dataset)} examples")
    print(f"Sample: {train_dataset[0][args.data_column][:100]}...")
    print()

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples[args.data_column],
            truncation=True,
            max_length=512,
            padding=False,
        )

    print("Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining completed!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
