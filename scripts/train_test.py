# train_gpt2_equations.py
# Script to fine-tune a GPT-2 model using PEFT (LoRA) on a dataset of equations.

# Author: Your Name
# Date: April 17, 2025 # Updated dynamically if needed, or keep original date

import argparse
import logging
import os
import sys
from datetime import datetime # For dynamic dating if preferred
from typing import Dict, Any, Optional, List, Union # For type hinting
import json # For loading training args from JSON

# Environment variable loading
from dotenv import load_dotenv

# Third-party libraries
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel # Import PeftModel for type hint

# --- Constants ---
SPECIAL_TOKENS = ["<startofex>", "<endofex>"]
DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LOGGING_STEPS = 100
DEFAULT_SAVE_EVAL_STEPS = 500
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_SEED = 42
DEFAULT_EVAL_STRATEGY = "epoch"
DEFAULT_SAVE_STRATEGY = "epoch"
DEFAULT_DATA_COLUMN = "text" # Default target column after processing

# --- Logging Configuration ---
# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_hf_token() -> str:
    """Loads Hugging Face token from .env file."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("Hugging Face token (HF_TOKEN) not found in .env file.")
        raise ValueError("Hugging Face token not found in .env.")
    logger.info("Hugging Face token loaded successfully.")
    return token

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 model using PEFT (LoRA) on an equation dataset."
    )

    # Model & Data Args
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME,
                        help="Pretrained model name or path (e.g., 'gpt2', 'gpt2-medium').")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="Hugging Face Hub repository ID for the dataset (e.g., 'username/my-equation-dataset').")
    parser.add_argument("--data_dir", type=str, default="10k", 
                        help="Directory containing the dataset files within the repo (optional).")
    parser.add_argument("--source_data_column", type=str, default="i_simple", # Changed from args.approach based on usage
                        help="Column name in the *source* dataset to use for training (will be renamed to 'text').")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="Block size for tokenizing and chunking.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size per device during evaluation.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS,
                        help="Steps for gradient accumulation.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Learning rate scheduler warmup steps.")

    # LoRA / PEFT Parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (dimension).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor).")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["c_attn"],
                    help="Module names to apply LoRA to (e.g., 'c_attn' for GPT-2 query/key/value).")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
                        help="Bias type for LoRA.")

    # Logging, Saving & Evaluation Args
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model, checkpoints, and logs.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory if it exists.")
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Log training metrics every N steps.")
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_SAVE_EVAL_STEPS,
                        help="Evaluate every N steps (if eval_strategy='steps').")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_EVAL_STEPS,
                        help="Save checkpoint every N steps (if save_strategy='steps').")
    parser.add_argument("--eval_strategy", type=str, default=DEFAULT_EVAL_STRATEGY, choices=["steps", "epoch", "no"], help="Evaluation strategy.")
    parser.add_argument("--save_strategy", type=str, default=DEFAULT_SAVE_STRATEGY, choices=["steps", "epoch", "no"],
                        help="Checkpoint saving strategy.")
    parser.add_argument("--save_total_limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT,
                        help="Limit the total number of checkpoints saved.")
    parser.add_argument("--load_best_model_at_end", action='store_true',
                        help="Load the best model (based on evaluation loss) at the end.")
    parser.add_argument("--early_stopping_patience", type=int, default=2, # Default to no early stopping
                        help="Number of evaluations with no improvement to trigger early stopping. Requires load_best_model_at_end.")

    # Technical Args
    parser.add_argument("--fp16", action='store_true', help="Use mixed precision training (FP16).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"],
                        help="Where to report metrics.")
    parser.add_argument("--run_name", type=str, default="train_gpt2_equations",
                        help="Name of the run for logging purposes.")
    

    # Hugging Face Hub Args
    parser.add_argument("--push_to_hub", action='store_true', help="Push the final model to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Repository ID for pushing (e.g., 'username/gpt2-finetuned-equations'). Required if --push_to_hub.")


    args = parser.parse_args()

    # --- Argument Validation ---
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub_model_id is required when --push_to_hub is set.")
    if args.early_stopping_patience is not None and not args.load_best_model_at_end:
        logger.warning("--early_stopping_patience is set, but --load_best_model_at_end is False. Early stopping requires loading the best model.")
        # Or raise ValueError if strictness is needed.
    if args.eval_strategy == "no" and (args.load_best_model_at_end or args.early_stopping_patience is not None):
        raise ValueError("Cannot use --load_best_model_at_end or --early_stopping_patience without evaluation (set --eval_strategy to 'steps' or 'epoch').")

    return args

# --- Dataset Loading and Preprocessing ---

def load_and_prepare_dataset(
    dataset_repo_id: str,
    data_dir: Optional[str],
    source_column: str,
    target_column: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    eval_strategy: str
) -> DatasetDict:
    
    """Loads dataset, renames column, tokenizes, and groups texts."""
    logger.info(f"Loading dataset from Hub: {dataset_repo_id} (data_dir: {data_dir})")
    try:
        raw_datasets = load_dataset(dataset_repo_id, data_dir=data_dir)
        logger.info(f"Dataset loaded: {raw_datasets}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        sys.exit(1)

    # --- Preprocessing Steps ---
    # 1. Rename source column to target column (e.g., 'text')
    logger.info(f"Renaming column '{source_column}' to '{target_column}' and removing others.")
    try:
        # Define the mapping function robustly
        def rename_and_keep_column(example: Dict[str, Any]) -> Dict[str, Any]:
            if source_column not in example:
                raise KeyError(f"Source column '{source_column}' not found in example: {list(example.keys())}")
            return {target_column: example[source_column]}

        # Get all column names *before* mapping to correctly remove them
        column_names_to_remove = {}
        for split in raw_datasets.keys():
             column_names_to_remove[split] = raw_datasets[split].column_names
        
        processed_datasets = DatasetDict()
        for split, names in column_names_to_remove.items():
            processed_datasets[split] = raw_datasets[split].map(
                rename_and_keep_column,
                batched=False, # Process example by example for renaming usually safer
                remove_columns=names # Remove all original columns
            )
        logger.info(f"Dataset after column renaming: {processed_datasets}")

    except KeyError as e:
        logger.error(f"Error during column renaming: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during column renaming/cleanup: {e}", exc_info=True)
        sys.exit(1)


    # 2. Tokenize
    logger.info("Tokenizing dataset...")
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        """Applies the tokenizer to the target text column."""
  
        return tokenizer(examples[target_column], truncation=True, padding=False)

    tokenized_datasets = processed_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=processed_datasets["train"].column_names,
        # num_proc=os.cpu_count(), # Optional: Use multiple processes for speed
        desc="Running tokenizer on dataset", # Progress bar description
    )
    logger.info("Tokenization complete.")

   
    # 3. Group texts into blocks
    logger.info(f"Grouping texts into blocks of size: {block_size}")

    def group_texts(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Concatenates all input sequences and splits them into blocks of `block_size`,
        with attention_mask and labels aligned for causal LM training.
        """

        # 🔄 Step 1: Concatenate all examples together for each field
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # 📏 Step 2: Trim to nearest multiple of block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            logger.warning(
                f"Total length ({total_length}) < block_size ({block_size}), might return empty batches."
            )

        # 🧱 Step 3: Split into chunks
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

        # 🎯 Step 4: Create labels (deep copy of input_ids)
        result["labels"] = [list(x) for x in result["input_ids"]]

        return result



    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        #num_proc=os.cpu_count(), 
        desc=f"Grouping texts into chunks of {block_size}", # Progress bar description
    )
    logger.info("Grouping complete.")
    logger.info(f"Processed dataset structure: {lm_datasets}")

    return lm_datasets


# --- Tokenizer and Model Loading ---

def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Loads the tokenizer and adds special tokens."""
    logger.info(f"Loading tokenizer for model: {model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        # Defina seus tokens especiais de forma clara
        SPECIAL_TOKENS = {
            "eos_token": "<endofex>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<startofex>"]
        }

        # Adiciona os tokens especiais
        num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)

        # Reforça as definições (importante para compatibilidade com Trainer)
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<endofex>"

        logger.info(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        sys.exit(1)

def load_model(model_name_or_path: str, tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace) -> PeftModel:
    """Loads the base model, resizes embeddings, and applies PEFT (LoRA)."""
    logger.info(f"Loading pretrained model: {model_name_or_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        # Resize token embeddings to match tokenizer vocabulary size (including added tokens)
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model token embeddings to: {len(tokenizer)}")

    except Exception as e:
        logger.error(f"Failed to load base model: {e}", exc_info=True)
        sys.exit(1)

    # --- PEFT (LoRA) Configuration ---
    logger.info("Configuring PEFT (LoRA)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        # modules_to_save = ["lm_head"], # Optional: If you want to train the lm_head as well
    )
    logger.info(f"LoRA Config: {lora_config}")

    # Apply PEFT to the base model
    try:
        model = get_peft_model(base_model, lora_config)
        logger.info("Applied PEFT (LoRA) configuration to the model.")
        model.print_trainable_parameters() # Shows trainable vs total parameters

        # Basic check for trainable parameters
        if not any(p.requires_grad for p in model.parameters()):
             logger.error("No parameters marked as trainable after applying LoRA. Check LoRA config and target modules.")
             sys.exit(1)
        # model.gradient_checkpointing_enable() # Consider enabling if memory is an issue

        return model

    except Exception as e:
        logger.error(f"Failed to apply PEFT (LoRA) to the model: {e}", exc_info=True)
        sys.exit(1)

# --- Trainer Initialization ---

def initialize_trainer(
    model: PeftModel,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    tokenizer: PreTrainedTokenizerBase,
    early_stopping_patience: Optional[int]
) -> Trainer:
    """Initializes and returns the Hugging Face Trainer."""
    logger.info("Initializing Trainer...")

    # Data collator for Causal LM (handles padding and labels)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Callbacks
    callbacks: List[TrainerCallback] = []
    if args.load_best_model_at_end and early_stopping_patience is not None and early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled with patience: {early_stopping_patience}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Trainer handles None eval_dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
        # compute_metrics=compute_metrics, # Add if custom metrics are needed
    )
    logger.info("Trainer initialized.")
    return trainer

# --- Main Execution ---

def main():
    """Main function to orchestrate the fine-tuning process."""
    start_time = datetime.now()
    logger.info(f"--- Starting Fine-Tuning Script at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Parse Arguments
    args = parse_arguments()
    logger.info(f"Running with arguments: {vars(args)}")

    # 2. Load HF Token (only if needed)
    hf_token = None
    if args.push_to_hub:
        hf_token = load_hf_token()

    # 3. Set Seed for Reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # 4. Load Tokenizer
    tokenizer = load_tokenizer(args.model_name_or_path)

    # 5. Load and Prepare Dataset
    lm_datasets = load_and_prepare_dataset(
        dataset_repo_id=args.dataset_repo_id,
        data_dir=args.data_dir,
        source_column=args.source_data_column,
        target_column=DEFAULT_DATA_COLUMN, # Use the constant target column name
        tokenizer=tokenizer,
        block_size=args.block_size,
        eval_strategy=args.eval_strategy # Pass eval strategy to handle warnings correctly
    )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation") # Returns None if 'validation' doesn't exist
    has_validation = eval_dataset is not None and len(eval_dataset) > 0
    if not has_validation:
        logger.warning("No validation dataset found. Skipping evaluation during training.")
        eval_dataset = None

    
    # 6. Load Model and Apply PEFT
    model = load_model(args.model_name_or_path, tokenizer, args)
    
    # 7. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        seed=args.seed,
        eval_strategy=args.eval_strategy,
        metric_for_best_model="eval_loss", # Or make this an arg
        greater_is_better=False,         # Or make this an arg
        load_best_model_at_end=args.load_best_model_at_end,
        save_strategy=args.save_strategy, # Ensure this matches eval_strategy for early stopping
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(args.output_dir, "logs"), # Example
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=hf_token if args.push_to_hub else None, # Assuming hf_token is loaded
        overwrite_output_dir=args.overwrite_output_dir,
        # Add any other relevant arguments from your parse_arguments function
    )

    # 8. Initialize Trainer
    trainer = initialize_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        early_stopping_patience=args.early_stopping_patience
    )

    # 9. Start Training
    logger.info("*** Starting Training ***")
    try:
        train_result = trainer.train()
        logger.info("Training finished.")

        # Log and save final training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save the final model, tokenizer, and config
        logger.info(f"Saving final model and tokenizer to {training_args.output_dir}")
        trainer.save_model() # Saves PEFT adapter, base model config, tokenizer, etc.
        # Tokenizer is usually saved by save_model, but explicit save is harmless
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("Model and tokenizer saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)

    # 10. Evaluate (if configured and possible)
    if training_args.do_eval: # Checks if eval_strategy is not 'no'
        if eval_dataset:
            logger.info("*** Evaluating Final Model ***")
            try:
                eval_metrics = trainer.evaluate()
                # Modify metrics for perplexity if desired
                try:
                    perplexity = np.exp(eval_metrics["eval_loss"])
                    eval_metrics["perplexity"] = perplexity
                    logger.info(f"Perplexity: {perplexity:.4f}")
                except OverflowError:
                     eval_metrics["perplexity"] = float("inf")
                     logger.warning("Could not compute perplexity due to overflow in exp(eval_loss).")
                
                logger.info(f"Evaluation metrics: {eval_metrics}")
                trainer.log_metrics("eval", eval_metrics)
                trainer.save_metrics("eval", eval_metrics)
            except Exception as e:
                logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
        else:
            logger.warning("Evaluation was configured but no valid evaluation dataset was found/processed. Skipping final evaluation.")

    # 11. Push to Hub (if requested)
    if training_args.push_to_hub:
        logger.info(f"Pushing final model artifacts to Hub repository: {training_args.hub_model_id}")
        try:
            # This pushes the content saved by save_model() (adapter, configs, tokenizer)
            trainer.push_to_hub(commit_message="End of fine-tuning training")
            logger.info("Model pushed successfully to the Hub.")
        except Exception as e:
            logger.error(f"Failed to push model to Hub: {e}", exc_info=True)
            # Don't exit, training still completed locally

    end_time = datetime.now()
    logger.info(f"--- Script Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    logger.info(f"Total execution time: {end_time - start_time}")


if __name__ == "__main__":
    main()