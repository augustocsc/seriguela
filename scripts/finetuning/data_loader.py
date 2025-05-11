# finetuning/data_loader.py

import sys
from typing import Dict, Any, Optional, List
from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase
from .utils import logger # Import logger

# Note: The original script had a commented-out section for group_texts.
# I've kept it commented out here as well, returning tokenized_datasets directly.
# If text grouping is needed, uncomment the relevant parts.

def load_and_prepare_dataset(
    dataset_repo_id: str,
    data_dir: Optional[str],
    source_column: str,
    target_column: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    eval_strategy: str # Keep for potential future use or warnings
) -> DatasetDict:
    """Loads dataset, renames column, tokenizes, and optionally groups texts."""
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
        def rename_and_keep_column(example: Dict[str, Any]) -> Dict[str, Any]:
            if source_column not in example:
                raise KeyError(f"Source column '{source_column}' not found in example: {list(example.keys())}")
            return {target_column: example[source_column]}

        column_names_to_remove = {}
        for split in raw_datasets.keys():
            column_names_to_remove[split] = [name for name in raw_datasets[split].column_names if name != source_column]
            # Ensure target_column is not accidentally removed if it's the same as source_column initially
            if source_column in column_names_to_remove[split]: # Should not happen if logic is correct
                column_names_to_remove[split].remove(source_column)


        processed_datasets = DatasetDict()
        for split, original_cols in raw_datasets.items():
            cols_to_remove = [col for col in original_cols.column_names if col != source_column]
            processed_datasets[split] = raw_datasets[split].map(
                rename_and_keep_column,
                batched=False,
                remove_columns=cols_to_remove
            )
        logger.info(f"Dataset after column renaming: {processed_datasets}")

    except KeyError as e:
        logger.error(f"Error during column renaming: {e}. Ensure '{source_column}' exists.", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during column renaming/cleanup: {e}", exc_info=True)
        sys.exit(1)

    # 2. Tokenize
    logger.info("Tokenizing dataset...")
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        # Ensure tokenizer handles truncation as per original intention
        return tokenizer(examples[target_column], truncation=True, max_length=block_size if block_size else None)


    try:
        tokenized_datasets = processed_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=processed_datasets["train"].column_names, # Removes the 'text' column
            desc="Running tokenizer on dataset",
        )
        logger.info("Tokenization complete.")
    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        sys.exit(1)


    # 3. Group texts into blocks (Currently commented out in original script logic)
    # logger.info(f"Grouping texts into blocks of size: {block_size}")
    # def group_texts(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    #     concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated["input_ids"])
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     else:
    #         logger.warning(
    #             f"Total length ({total_length}) < block_size ({block_size}), might return empty batches."
    #         )
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated.items()
    #     }
    #     result["labels"] = [list(x) for x in result["input_ids"]] # Deep copy for labels
    #     return result

    # lm_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     desc=f"Grouping texts into chunks of {block_size}",
    # )
    # logger.info("Grouping complete.")
    # logger.info(f"Processed dataset structure after grouping: {lm_datasets}")
    # return lm_datasets

    logger.info(f"Processed dataset structure (tokenized only): {tokenized_datasets}")
    return tokenized_datasets