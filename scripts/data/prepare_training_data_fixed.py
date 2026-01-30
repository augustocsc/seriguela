"""
Data preparation script that adds proper <|endofex|> markers to training data.

This script processes the existing dataset and wraps expressions with end-of-expression
markers so the model learns to stop generation correctly.

Usage:
    python scripts/data/prepare_training_data_fixed.py \
        --dataset_repo_id augustocsc/sintetico_natural \
        --data_dir 700K \
        --data_column i_prompt_n \
        --output_dir ./data/processed/700K_fixed \
        --validate
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_end_markers(example: Dict) -> Dict:
    """
    Add end-of-expression markers to training data.

    This function:
    1. Locates the expression in the text (after 'expr:')
    2. Finds the natural end boundary (before 'vars:', newlines, etc.)
    3. Inserts <|endofex|> marker at the end
    4. Preserves any remaining content after the marker

    Args:
        example: Dictionary containing 'text' field with training data

    Returns:
        Dictionary with modified 'text' field containing end markers
    """
    text = example['text']

    # Check if expression part exists
    if 'expr:' not in text:
        logger.warning(f"No 'expr:' found in text: {text[:100]}...")
        return {'text': text}

    # Split at expr: and add marker after expression
    parts = text.split('expr:', 1)
    if len(parts) != 2:
        logger.warning(f"Unexpected format in text: {text[:100]}...")
        return {'text': text}

    prefix = parts[0]
    expression_part = parts[1]

    # Check if marker already exists
    if '<|endofex|>' in expression_part:
        logger.debug("Marker already present, skipping")
        return {'text': text}

    # Find natural end of expression (before vars:, newline, etc)
    end_idx = len(expression_part)
    boundaries = ['\nvars:', '\nVariables:', '\n\n', '\nvar:', '\nVariable:']

    for boundary in boundaries:
        idx = expression_part.find(boundary)
        if idx != -1 and idx < end_idx:
            end_idx = idx

    # Insert marker
    clean_expr = expression_part[:end_idx].strip()
    remaining = expression_part[end_idx:]

    # Reconstruct text with marker
    new_text = f"{prefix}expr: {clean_expr}<|endofex|>{remaining}"

    return {'text': new_text}


def validate_markers(example: Dict) -> Dict:
    """
    Validate that markers are properly present in the text.

    Args:
        example: Dictionary containing 'text' field

    Returns:
        Dictionary with validation metadata
    """
    text = example['text']
    start_count = text.count('<|startofex|>')
    end_count = text.count('<|endofex|>')

    # Valid if we have at least one end marker
    # (start marker is optional depending on format)
    valid = end_count > 0

    return {
        'valid': valid,
        'start_count': start_count,
        'end_count': end_count,
        'text': text
    }


def process_dataset(
    dataset_repo_id: str,
    data_dir: str,
    data_column: str,
    output_dir: Path,
    validate: bool = True
) -> Tuple[DatasetDict, Dict]:
    """
    Process the dataset by adding end markers to all splits.

    Args:
        dataset_repo_id: HuggingFace dataset repository ID
        data_dir: Subdirectory within the dataset (e.g., '700K')
        data_column: Column to use for training data
        output_dir: Directory to save processed dataset
        validate: Whether to run validation after processing

    Returns:
        Tuple of (processed_dataset, statistics)
    """
    logger.info(f"Loading dataset from {dataset_repo_id}/{data_dir}...")

    try:
        # Load dataset from HuggingFace Hub
        dataset = load_dataset(
            dataset_repo_id,
            data_dir=data_dir,
            split=None  # Load all splits
        )

        if not isinstance(dataset, dict):
            # If single split, convert to dict
            dataset = {'train': dataset}

        logger.info(f"Loaded {len(dataset)} split(s): {list(dataset.keys())}")

        # Show sample before processing
        if 'train' in dataset and len(dataset['train']) > 0:
            logger.info(f"\nSample BEFORE processing:")
            logger.info(f"{dataset['train'][0][data_column][:200]}...")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Process each split
    processed_dataset = {}
    statistics = {
        'total_examples': 0,
        'processed_examples': 0,
        'already_marked': 0,
        'splits': {}
    }

    for split_name, split_data in dataset.items():
        logger.info(f"\nProcessing {split_name} split ({len(split_data)} examples)...")

        # Rename column to 'text' if needed
        if data_column != 'text':
            split_data = split_data.rename_column(data_column, 'text')

        # Count examples that already have markers
        already_marked = sum(1 for ex in split_data if '<|endofex|>' in ex['text'])
        statistics['already_marked'] += already_marked

        if already_marked > 0:
            logger.info(f"Found {already_marked} examples already with markers")

        # Apply marker addition
        processed_split = split_data.map(
            add_end_markers,
            desc=f"Adding markers to {split_name}"
        )

        processed_dataset[split_name] = processed_split

        # Update statistics
        split_stats = {
            'total': len(split_data),
            'processed': len(processed_split),
            'already_marked': already_marked
        }
        statistics['splits'][split_name] = split_stats
        statistics['total_examples'] += len(split_data)
        statistics['processed_examples'] += len(processed_split)

        # Show sample after processing
        if len(processed_split) > 0:
            logger.info(f"\nSample AFTER processing:")
            logger.info(f"{processed_split[0]['text'][:200]}...")

    # Validate if requested
    if validate:
        logger.info("\n" + "="*60)
        logger.info("VALIDATION")
        logger.info("="*60)

        for split_name, split_data in processed_dataset.items():
            logger.info(f"\nValidating {split_name} split...")

            # Apply validation
            validated = split_data.map(validate_markers)

            # Count valid examples
            valid_count = sum(validated['valid'])
            invalid_count = len(validated) - valid_count

            valid_rate = valid_count / len(validated) * 100

            logger.info(f"Valid examples: {valid_count}/{len(validated)} ({valid_rate:.1f}%)")

            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid examples!")

                # Show first few invalid examples
                invalid_examples = [
                    ex for ex in validated if not ex['valid']
                ][:3]

                for i, ex in enumerate(invalid_examples):
                    logger.warning(f"\nInvalid example {i+1}:")
                    logger.warning(f"Start markers: {ex['start_count']}")
                    logger.warning(f"End markers: {ex['end_count']}")
                    logger.warning(f"Text: {ex['text'][:200]}...")

            # Update statistics
            statistics['splits'][split_name]['valid'] = valid_count
            statistics['splits'][split_name]['invalid'] = invalid_count
            statistics['splits'][split_name]['valid_rate'] = valid_rate

    # Convert back to DatasetDict
    processed_dataset = DatasetDict(processed_dataset)

    return processed_dataset, statistics


def save_dataset(dataset: DatasetDict, output_dir: Path, data_dir: str):
    """
    Save processed dataset to local directory.

    Args:
        dataset: Processed dataset to save
        output_dir: Directory to save to
        data_dir: Original data directory name (for filename)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving processed dataset to {output_dir}...")

    for split_name, split_data in dataset.items():
        # Save as CSV
        output_file = output_dir / f"{split_name}_{data_dir}.csv"

        # Convert to pandas and save
        df = split_data.to_pandas()
        df.to_csv(output_file, index=False)

        logger.info(f"Saved {split_name} split: {output_file} ({len(df)} examples)")

    logger.info("Dataset saved successfully!")


def print_statistics(statistics: Dict):
    """
    Print processing statistics in a formatted table.

    Args:
        statistics: Dictionary containing processing statistics
    """
    logger.info("\n" + "="*60)
    logger.info("PROCESSING STATISTICS")
    logger.info("="*60)

    logger.info(f"\nTotal examples: {statistics['total_examples']}")
    logger.info(f"Processed examples: {statistics['processed_examples']}")
    logger.info(f"Already marked: {statistics['already_marked']}")

    logger.info("\nPer-split statistics:")
    logger.info("-"*60)

    for split_name, split_stats in statistics['splits'].items():
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Total: {split_stats['total']}")
        logger.info(f"  Processed: {split_stats['processed']}")
        logger.info(f"  Already marked: {split_stats.get('already_marked', 0)}")

        if 'valid' in split_stats:
            logger.info(f"  Valid: {split_stats['valid']}")
            logger.info(f"  Invalid: {split_stats['invalid']}")
            logger.info(f"  Valid rate: {split_stats['valid_rate']:.1f}%")

    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data with proper end-of-expression markers"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Subdirectory within the dataset (e.g., '700K')"
    )
    parser.add_argument(
        "--data_column",
        type=str,
        required=True,
        help="Column to use for training data (e.g., 'i_prompt_n')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed dataset"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after processing"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push processed dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID for pushing (if --push_to_hub)"
    )

    args = parser.parse_args()

    # Convert output_dir to Path
    output_dir = Path(args.output_dir)

    # Process dataset
    try:
        processed_dataset, statistics = process_dataset(
            dataset_repo_id=args.dataset_repo_id,
            data_dir=args.data_dir,
            data_column=args.data_column,
            output_dir=output_dir,
            validate=args.validate
        )

        # Print statistics
        print_statistics(statistics)

        # Save to local directory
        save_dataset(processed_dataset, output_dir, args.data_dir)

        # Push to Hub if requested
        if args.push_to_hub:
            if not args.hub_repo_id:
                logger.error("--hub_repo_id required when using --push_to_hub")
                sys.exit(1)

            logger.info(f"\nPushing to HuggingFace Hub: {args.hub_repo_id}")
            processed_dataset.push_to_hub(args.hub_repo_id)
            logger.info("Successfully pushed to Hub!")

        # Check if any validation failed
        if args.validate:
            all_valid = all(
                split_stats.get('invalid', 0) == 0
                for split_stats in statistics['splits'].values()
            )

            if not all_valid:
                logger.error("\n⚠️ Some examples failed validation!")
                sys.exit(1)
            else:
                logger.info("\n✅ All examples validated successfully!")

        logger.info("\n✅ Data preparation complete!")

    except Exception as e:
        logger.error(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
