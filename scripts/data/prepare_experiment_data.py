#!/usr/bin/env python3
"""
Data preparation script for training experiments.

Prepares data in two formats:
- EXP-A: JSON structured format
- EXP-B: EOS token format (GPT-2's <|endoftext|>)

Usage:
    python scripts/data/prepare_experiment_data.py \
        --dataset_repo_id augustocsc/sintetico_natural \
        --data_dir 700K \
        --data_column i_prompt_n \
        --output_base_dir ./data/experiments
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_original_format(text: str) -> Optional[Dict]:
    """
    Parse the original format into components.

    Original format:
        vars: x_1, x_2
        oper: *, +, sin
        cons: C
        expr: C*sin(x_1) + x_2

    Returns:
        Dictionary with vars, ops, cons, expr or None if parsing fails
    """
    result = {
        'vars': [],
        'ops': [],
        'cons': None,
        'expr': None,
        'raw_text': text
    }

    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('vars:') or line.startswith('Variables:'):
            # Extract variables
            var_part = line.split(':', 1)[1].strip()
            vars_list = [v.strip() for v in var_part.split(',') if v.strip()]
            result['vars'] = vars_list

        elif line.startswith('oper:') or line.startswith('Operators:'):
            # Extract operators
            op_part = line.split(':', 1)[1].strip()
            ops_list = [o.strip() for o in op_part.split(',') if o.strip()]
            result['ops'] = ops_list

        elif line.startswith('cons:') or line.startswith('Constants:'):
            # Extract constants
            cons_part = line.split(':', 1)[1].strip()
            result['cons'] = cons_part if cons_part else None

        elif line.startswith('expr:'):
            # Extract expression - everything after 'expr:'
            expr_part = line.split(':', 1)[1].strip()
            # Clean expression: remove any markers or trailing content
            expr_part = expr_part.split('<|')[0].strip()  # Remove any existing markers
            expr_part = expr_part.split('\n')[0].strip()  # Remove newlines
            result['expr'] = expr_part

    # Validate we got the essential parts
    if not result['expr']:
        return None

    return result


def convert_to_json_format(parsed: Dict) -> str:
    """
    Convert parsed data to JSON format (EXP-A).

    Output format:
        {"vars": ["x_1", "x_2"], "ops": ["*", "+", "sin"], "cons": "C", "expr": "C*sin(x_1) + x_2"}
    """
    json_obj = {
        'vars': parsed['vars'],
        'ops': parsed['ops'],
    }

    if parsed['cons']:
        json_obj['cons'] = parsed['cons']

    json_obj['expr'] = parsed['expr']

    return json.dumps(json_obj, ensure_ascii=False)


def convert_to_eos_format(parsed: Dict) -> str:
    """
    Convert parsed data to EOS token format (EXP-B).

    Output format:
        vars: x_1, x_2
        oper: *, +, sin
        cons: C
        expr: C*sin(x_1) + x_2<|endoftext|>
    """
    lines = []

    if parsed['vars']:
        lines.append(f"vars: {', '.join(parsed['vars'])}")

    if parsed['ops']:
        lines.append(f"oper: {', '.join(parsed['ops'])}")

    if parsed['cons']:
        lines.append(f"cons: {parsed['cons']}")

    # Add expression with EOS token
    lines.append(f"expr: {parsed['expr']}<|endoftext|>")

    return '\n'.join(lines)


def process_example_json(example: Dict) -> Dict:
    """Process a single example into JSON format."""
    text = example['text']
    parsed = parse_original_format(text)

    if parsed is None:
        logger.warning(f"Failed to parse: {text[:100]}...")
        return {'text': '', 'valid': False}

    json_text = convert_to_json_format(parsed)
    return {'text': json_text, 'valid': True}


def process_example_eos(example: Dict) -> Dict:
    """Process a single example into EOS format."""
    text = example['text']
    parsed = parse_original_format(text)

    if parsed is None:
        logger.warning(f"Failed to parse: {text[:100]}...")
        return {'text': '', 'valid': False}

    eos_text = convert_to_eos_format(parsed)
    return {'text': eos_text, 'valid': True}


def validate_json_format(text: str) -> bool:
    """Validate JSON format is correct."""
    try:
        obj = json.loads(text)
        return 'expr' in obj and 'vars' in obj and 'ops' in obj
    except:
        return False


def validate_eos_format(text: str) -> bool:
    """Validate EOS format is correct."""
    return '<|endoftext|>' in text and 'expr:' in text


def process_dataset(
    dataset_repo_id: str,
    data_dir: str,
    data_column: str,
    output_base_dir: Path,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Process the dataset into both formats.

    Args:
        dataset_repo_id: HuggingFace dataset repository ID
        data_dir: Subdirectory within the dataset
        data_column: Column containing the text data
        output_base_dir: Base directory for output
        max_samples: Optional limit on number of samples (for testing)

    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Loading dataset from {dataset_repo_id}/{data_dir}...")

    # Load dataset
    dataset = load_dataset(
        dataset_repo_id,
        data_dir=data_dir,
        split=None
    )

    if not isinstance(dataset, dict):
        dataset = {'train': dataset}

    logger.info(f"Loaded {len(dataset)} split(s): {list(dataset.keys())}")

    # Show sample
    if 'train' in dataset:
        sample = dataset['train'][0][data_column]
        logger.info(f"\nSample ORIGINAL format:\n{sample}\n")

    # Create output directories
    output_json = output_base_dir / 'exp_a_json'
    output_eos = output_base_dir / 'exp_b_eos'
    output_json.mkdir(parents=True, exist_ok=True)
    output_eos.mkdir(parents=True, exist_ok=True)

    statistics = {
        'total': 0,
        'json_valid': 0,
        'eos_valid': 0,
        'json_invalid': 0,
        'eos_invalid': 0,
        'splits': {}
    }

    for split_name, split_data in dataset.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split_name} split ({len(split_data)} examples)")
        logger.info('='*60)

        # Rename column if needed
        if data_column != 'text':
            split_data = split_data.rename_column(data_column, 'text')

        # Limit samples if specified
        if max_samples and len(split_data) > max_samples:
            logger.info(f"Limiting to {max_samples} samples for testing")
            split_data = split_data.select(range(max_samples))

        statistics['total'] += len(split_data)

        # Process to JSON format
        logger.info("\nConverting to JSON format (EXP-A)...")
        json_data = split_data.map(
            process_example_json,
            desc=f"JSON format ({split_name})"
        )

        # Filter valid examples
        json_valid = json_data.filter(lambda x: x['valid'])
        json_invalid_count = len(json_data) - len(json_valid)

        logger.info(f"JSON format: {len(json_valid)}/{len(json_data)} valid")

        if len(json_valid) > 0:
            logger.info(f"\nSample JSON format:\n{json_valid[0]['text']}\n")

        # Process to EOS format
        logger.info("\nConverting to EOS format (EXP-B)...")
        eos_data = split_data.map(
            process_example_eos,
            desc=f"EOS format ({split_name})"
        )

        # Filter valid examples
        eos_valid = eos_data.filter(lambda x: x['valid'])
        eos_invalid_count = len(eos_data) - len(eos_valid)

        logger.info(f"EOS format: {len(eos_valid)}/{len(eos_data)} valid")

        if len(eos_valid) > 0:
            logger.info(f"\nSample EOS format:\n{eos_valid[0]['text']}\n")

        # Update statistics
        statistics['json_valid'] += len(json_valid)
        statistics['json_invalid'] += json_invalid_count
        statistics['eos_valid'] += len(eos_valid)
        statistics['eos_invalid'] += eos_invalid_count
        statistics['splits'][split_name] = {
            'total': len(split_data),
            'json_valid': len(json_valid),
            'eos_valid': len(eos_valid)
        }

        # Save JSON format
        json_df = pd.DataFrame({'text': [ex['text'] for ex in json_valid]})
        json_file = output_json / f'{split_name}.csv'
        json_df.to_csv(json_file, index=False)
        logger.info(f"Saved JSON: {json_file} ({len(json_df)} examples)")

        # Save EOS format
        eos_df = pd.DataFrame({'text': [ex['text'] for ex in eos_valid]})
        eos_file = output_eos / f'{split_name}.csv'
        eos_df.to_csv(eos_file, index=False)
        logger.info(f"Saved EOS: {eos_file} ({len(eos_df)} examples)")

    return statistics


def validate_output_files(output_base_dir: Path) -> Dict:
    """
    Validate the generated output files.

    Returns:
        Validation results dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("VALIDATION OF OUTPUT FILES")
    logger.info("="*60)

    results = {
        'exp_a_json': {'valid': True, 'issues': []},
        'exp_b_eos': {'valid': True, 'issues': []}
    }

    # Validate JSON format (EXP-A)
    json_dir = output_base_dir / 'exp_a_json'
    for csv_file in json_dir.glob('*.csv'):
        logger.info(f"\nValidating {csv_file.name}...")
        df = pd.read_csv(csv_file)

        valid_count = 0
        invalid_samples = []

        for idx, row in df.iterrows():
            text = row['text']
            if validate_json_format(text):
                valid_count += 1
            else:
                if len(invalid_samples) < 3:
                    invalid_samples.append(text[:100])

        rate = valid_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  Valid: {valid_count}/{len(df)} ({rate:.1f}%)")

        if invalid_samples:
            results['exp_a_json']['valid'] = False
            results['exp_a_json']['issues'].extend(invalid_samples)

    # Validate EOS format (EXP-B)
    eos_dir = output_base_dir / 'exp_b_eos'
    for csv_file in eos_dir.glob('*.csv'):
        logger.info(f"\nValidating {csv_file.name}...")
        df = pd.read_csv(csv_file)

        valid_count = 0
        invalid_samples = []

        for idx, row in df.iterrows():
            text = row['text']
            if validate_eos_format(text):
                valid_count += 1
            else:
                if len(invalid_samples) < 3:
                    invalid_samples.append(text[:100])

        rate = valid_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  Valid: {valid_count}/{len(df)} ({rate:.1f}%)")

        if invalid_samples:
            results['exp_b_eos']['valid'] = False
            results['exp_b_eos']['issues'].extend(invalid_samples)

    return results


def print_final_report(statistics: Dict, validation: Dict):
    """Print final processing report."""
    logger.info("\n" + "="*60)
    logger.info("FINAL REPORT")
    logger.info("="*60)

    logger.info(f"\nTotal examples processed: {statistics['total']}")

    logger.info("\nEXP-A (JSON Format):")
    logger.info(f"  Valid: {statistics['json_valid']}")
    logger.info(f"  Invalid: {statistics['json_invalid']}")
    json_rate = statistics['json_valid'] / statistics['total'] * 100 if statistics['total'] > 0 else 0
    logger.info(f"  Success rate: {json_rate:.1f}%")
    logger.info(f"  Validation: {'PASS' if validation['exp_a_json']['valid'] else 'FAIL'}")

    logger.info("\nEXP-B (EOS Format):")
    logger.info(f"  Valid: {statistics['eos_valid']}")
    logger.info(f"  Invalid: {statistics['eos_invalid']}")
    eos_rate = statistics['eos_valid'] / statistics['total'] * 100 if statistics['total'] > 0 else 0
    logger.info(f"  Success rate: {eos_rate:.1f}%")
    logger.info(f"  Validation: {'PASS' if validation['exp_b_eos']['valid'] else 'FAIL'}")

    logger.info("\nPer-split breakdown:")
    for split_name, split_stats in statistics['splits'].items():
        logger.info(f"\n  {split_name.upper()}:")
        logger.info(f"    Total: {split_stats['total']}")
        logger.info(f"    JSON valid: {split_stats['json_valid']}")
        logger.info(f"    EOS valid: {split_stats['eos_valid']}")

    logger.info("\n" + "="*60)

    all_valid = validation['exp_a_json']['valid'] and validation['exp_b_eos']['valid']
    if all_valid:
        logger.info("STATUS: ALL VALIDATIONS PASSED")
    else:
        logger.info("STATUS: SOME VALIDATIONS FAILED")

    logger.info("="*60)

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Prepare experiment data in JSON and EOS formats"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="augustocsc/sintetico_natural",
        help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="700K",
        help="Subdirectory within the dataset"
    )
    parser.add_argument(
        "--data_column",
        type=str,
        default="i_prompt_n",
        help="Column containing text data"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./data/experiments",
        help="Base directory for output"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split (for testing)"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip output file validation"
    )

    args = parser.parse_args()

    output_base_dir = Path(args.output_base_dir)

    logger.info("="*60)
    logger.info("EXPERIMENT DATA PREPARATION")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset_repo_id}/{args.data_dir}")
    logger.info(f"Column: {args.data_column}")
    logger.info(f"Output: {output_base_dir}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    logger.info("="*60)

    try:
        # Process dataset
        statistics = process_dataset(
            dataset_repo_id=args.dataset_repo_id,
            data_dir=args.data_dir,
            data_column=args.data_column,
            output_base_dir=output_base_dir,
            max_samples=args.max_samples
        )

        # Validate output
        if not args.skip_validation:
            validation = validate_output_files(output_base_dir)
        else:
            validation = {
                'exp_a_json': {'valid': True, 'issues': []},
                'exp_b_eos': {'valid': True, 'issues': []}
            }

        # Print report
        all_valid = print_final_report(statistics, validation)

        if all_valid:
            logger.info("\nData preparation completed successfully!")
            logger.info(f"\nOutput directories:")
            logger.info(f"  EXP-A (JSON): {output_base_dir / 'exp_a_json'}")
            logger.info(f"  EXP-B (EOS):  {output_base_dir / 'exp_b_eos'}")
            sys.exit(0)
        else:
            logger.error("\nData preparation completed with validation errors!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"\nFailed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
