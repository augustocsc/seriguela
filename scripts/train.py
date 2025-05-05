# train_gpt2_equations.py
# Script to fine-tune a GPT-2 model on a dataset of equations from the Hugging Face Hub.
# Author: Your Name
# Date: April 17, 2025

import argparse
import os
import logging
from dotenv import load_dotenv
import sys
from transformers import EarlyStoppingCallback
import numpy as np


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)


from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Preprocessing Functions ---

def tokenize_function(examples, tokenizer):
    """Applies the tokenizer to the 'text' field of the dataset examples."""
    return tokenizer(examples["text"])

def group_texts(examples, block_size):
    """Groups texts into chunks of block_size."""
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
         # Handle case where total length is less than block size (might happen with very small datasets/splits)
         # You might want to pad here, or simply return empty if Trainer handles it
         logger.warning(f"Total length ({total_length}) is smaller than block_size ({block_size}). Chunking might result in empty data for small splits.")
         # Returning empty might cause issues later, consider padding or adjusting block_size
         # For now, let's proceed but be aware.
         pass # Let the slicing below handle it, might result in empty lists

    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # For Causal LM, labels are usually the input_ids shifted, Trainer handles this if labels aren't provided
    # or we can create them explicitly like this:
    result["labels"] = result["input_ids"].copy()
    return result

# --- Main Training Function ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model on an equation dataset from Hugging Face Hub.")

    # --- Arguments ---
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Pretrained model name or path (e.g., 'gpt2', 'gpt2-medium').")
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face Hub repository ID for the dataset (e.g., 'username/my-equation-dataset').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model and checkpoints.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files.")
    parser.add_argument("--data_column", type=str, default="text", help="Column name in the dataset to be used for training.")
    parser.add_argument("--approach", default="infix_expr", type=str, required=True, help="Approach to be used for training (e.g., 'infix_expr', 'prefix_expr').")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for tokenizing and chunking the dataset.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating weights.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training metrics every N steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate on the validation set every N steps. Ignored if eval_strategy='epoch'.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps. Ignored if save_strategy='epoch'.")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"], help="Evaluation strategy ('steps', 'epoch', 'no').")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"], help="Checkpoint saving strategy ('steps', 'epoch', 'no').")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints saved.")
    parser.add_argument("--load_best_model_at_end", action='store_true', help="Load the best model (based on evaluation loss) at the end of training.")
    parser.add_argument("--fp16", action='store_true', help="Use mixed precision training (FP16). Requires CUDA.")
    parser.add_argument("--push_to_hub", action='store_true', help="Push the final model to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository ID for pushing the model (e.g., 'username/gpt2-finetuned-equations'). Required if --push_to_hub is set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Carrega as variáveis do .env
    load_dotenv()

    # Acessa o token
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Token da Hugging Face não encontrado no .env.")

    # Set seed for reproducibility
    set_seed(args.seed)

    logger.info(f"Starting fine-tuning with parameters: {args}")

    # 1. Load Dataset from Hub
    logger.info(f"Loading dataset from Hub: {args.dataset_repo_id}")
    try:
        raw_datasets = load_dataset(args.dataset_repo_id, data_dir=args.data_dir)

        # Keep only the 'arg.approach' column and rename it to 'text'
        ds = ds.map(lambda x: {"text": x["i_simple"]}, remove_columns=ds["train"].column_names)

        logger.info(f"Dataset loaded: {raw_datasets}")
        # Basic validation: Check for train/validation splits
        if "train" not in raw_datasets:
             raise ValueError("Dataset missing 'train' split.")
        if args.eval_strategy != "no" and "validation" not in raw_datasets:
             raise ValueError("Dataset missing 'validation' split, required for evaluation.")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer for model: {args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) #, use_fast=True) # Consider use_fast=True

        # Handle GPT-2 specific padding token if necessary
        if tokenizer.pad_token is None and "gpt2" in args.model_name_or_path.lower():
            logger.warning("GPT-2 tokenizer does not have a default pad token. Setting pad_token = eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        # Adding special tokens
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|startofex|>", "<|endofex|>"]})

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # 3. Preprocess Dataset (Tokenize & Chunk)
    logger.info("Tokenizing dataset...")
    # Need functools.partial or lambda if tokenize_function needs tokenizer arg with map
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        # num_proc=4, # Optional: Use multiple processes for faster tokenization
        remove_columns=raw_datasets["train"].column_names # Remove all original columns
    )
    logger.info("Tokenization complete.")

    logger.info(f"Grouping texts into blocks of size: {args.block_size}")
    # Need functools.partial or lambda if group_texts needs block_size arg with map
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, args.block_size),
        batched=True,
        # num_proc=4 # Optional: Use multiple processes
    )
    logger.info("Grouping complete.")
    logger.info(f"Processed dataset structure: {lm_datasets}")

    # Ensure datasets aren't empty after processing
    if not lm_datasets["train"]:
        logger.error("Training dataset is empty after processing. Check block_size and original data.")
        sys.exit(1)
    if args.eval_strategy != "no" and not lm_datasets["validation"]:
        logger.warning("Validation dataset is empty after processing. Evaluation might fail or be skipped.")


    # 4. Load Model
    logger.info(f"Loading pretrained model: {args.model_name_or_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

        # Update with tokenizer special tokens
        base_model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # Specify task type
        r=8,                          # LoRA rank (dimension of adapter matrices, e.g., 8, 16, 32)
        lora_alpha=32,                # LoRA alpha (scaling factor, often 2*r)
        target_modules=["c_attn"],    # Modules to apply LoRA to in GPT-2. 'c_attn' often covers query/key/value projections. May need adjustment based on exact model variant.
        lora_dropout=0.05,            # Dropout probability for LoRA layers
        bias="none"                   # Usually set to 'none' or 'all'
        # ... other LoraConfig parameters
    )

    # Apply PEFT
    logger.info("Applying PEFT (LoRA) configuration to the model...")
    model = get_peft_model(base_model, lora_config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Param will be trained: {name} | requires_grad={param.requires_grad}")

    model.train()

    requires_grad_params = [p for p in model.parameters() if p.requires_grad]
    if not requires_grad_params:
        logger.error("Nenhum parâmetro com requires_grad=True. O modelo está congelado e não pode ser treinado.")
        sys.exit(1)

    model.print_trainable_parameters() # This will show how few parameters are actually trainable!
    #model.gradient_checkpointing_enable()

    # 5. Configure Training Arguments
    logger.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True, # Be careful with this in production
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'), # Log Tensorboard data within output_dir
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if "validation" in lm_datasets and lm_datasets["validation"] else "no", # Check if validation set exists and is not empty
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else 500, # Default save_steps if strategy is steps but value not provided
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end if "validation" in lm_datasets and lm_datasets["validation"] else False, # Requires eval_dataset
        metric_for_best_model="loss" if args.load_best_model_at_end else None, # Use loss for best model selection if evaluating
        greater_is_better=False if args.load_best_model_at_end else None,
        fp16=args.fp16,
        report_to="wandb", #"tensorboard", # Or "wandb", "none"
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_token=token if args.push_to_hub else None, # Use the obtained token
        seed=args.seed,
        # Add deepspeed config path if using deepspeed
        # deepspeed=args.deepspeed_config_path
    )

    # Data collator - for CLM, pads inputs dynamically.
    # With chunking, sequences should already be block_size, but this handles potential variations/labels.
    # `mlm=False` specifies Causal LM.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets.get("validation"), # Use .get() to handle missing validation split gracefully if eval_strategy is 'no'
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics, # Optional: Define a function for custom eval metrics besides loss/perplexity
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # Optional: Early stopping callback
    )

    # 7. Start Training
    logger.info("*** Starting Training ***")
    try:
        train_result = trainer.train()
        logger.info("Training finished.")

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save final model and tokenizer
        logger.info(f"Saving final model to {args.output_dir}")
        trainer.save_model() # Saves model, tokenizer, config, training args
        # No need to call trainer.save_state() explicitly here unless needed outside Trainer's saves
        tokenizer.save_pretrained(args.output_dir)

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        sys.exit(1)


    # 8. Evaluate (Optional, but good practice if validation set exists)
    if training_args.do_eval and lm_datasets.get("validation"): # Check if evaluation was configured AND validation set exists
        logger.info("*** Evaluating Final Model ***")
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    # 9. Push to Hub (if requested)
    if args.push_to_hub:
        if not args.hub_model_id:
            logger.error("Cannot push to hub: --hub_model_id is required when --push_to_hub is set.")
        else:
            logger.info(f"Pushing final model to Hub repository: {args.hub_model_id}")
            try:
                # This pushes the content saved by save_model()
                trainer.push_to_hub(commit_message="End of training")
                logger.info("Model pushed successfully.")
            except Exception as e:
                logger.error(f"Failed to push model to Hub: {e}")

    logger.info("--- Script Finished ---")


if __name__ == "__main__":
    main()