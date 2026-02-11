#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert infix expressions to prefix notation.

This script reads the HuggingFace dataset with infix notation and creates
a new column with the same expressions in prefix notation, maintaining
the same variables and operators from the original prompt.
"""

import sys
import re
import argparse
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import sympy
from tqdm import tqdm
import os

sys.path.append('.')
sys.path.append('..')


def sympy_to_prefix(expr):
    """
    Convert a SymPy expression to prefix notation (Polish notation).

    Args:
        expr: SymPy expression

    Returns:
        str: Expression in prefix notation

    Examples:
        x_1 + x_2 -> + x_1 x_2
        x_1 * (x_2 + C) -> * x_1 + x_2 C
        sin(x_1**2) -> sin ** x_1 2
    """
    if isinstance(expr, sympy.Symbol):
        return str(expr)

    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        val = float(expr)
        # Clean up floats: 2.0 -> 2, but keep 2.5 -> 2.5
        if val == int(val):
            return str(int(val))
        return str(val)

    # Handle negative numbers
    if isinstance(expr, sympy.Mul):
        # Check if it's a negative multiplication (e.g., -1 * x)
        if len(expr.args) == 2:
            if expr.args[0] == -1:
                # Keep as multiplication for consistency
                arg = sympy_to_prefix(expr.args[1])
                return f"* -1 {arg}"
            elif expr.args[1] == -1:
                arg = sympy_to_prefix(expr.args[0])
                return f"* -1 {arg}"

        # Check for division (x * y**-1 pattern)
        numer = []
        denom = []
        for arg in expr.args:
            if isinstance(arg, sympy.Pow) and arg.args[1] == -1:
                denom.append(arg.args[0])
            else:
                numer.append(arg)

        if len(denom) > 0:
            # This is a division
            if len(numer) == 0:
                numer_expr = sympy.Integer(1)
            elif len(numer) == 1:
                numer_expr = numer[0]
            else:
                numer_expr = sympy.Mul(*numer)

            if len(denom) == 1:
                denom_expr = denom[0]
            else:
                denom_expr = sympy.Mul(*denom)

            numer_str = sympy_to_prefix(numer_expr)
            denom_str = sympy_to_prefix(denom_expr)
            return f"/ {numer_str} {denom_str}"

        # Regular multiplication
        args = [sympy_to_prefix(arg) for arg in expr.args]
        if len(args) == 2:
            return f"* {args[0]} {args[1]}"
        else:
            result = args[0]
            for arg in args[1:]:
                result = f"* {result} {arg}"
            return result

    # Handle function calls (sin, cos, exp, etc.)
    if isinstance(expr, sympy.Function):
        func_name = expr.func.__name__.lower()
        args = [sympy_to_prefix(arg) for arg in expr.args]
        return f"{func_name} {' '.join(args)}"

    # Handle power operator
    if isinstance(expr, sympy.Pow):
        base = sympy_to_prefix(expr.args[0])
        exp_val = sympy_to_prefix(expr.args[1])
        return f"** {base} {exp_val}"

    # Handle addition with special case for subtraction
    if isinstance(expr, sympy.Add):
        # Check if any term is negative (subtraction)
        positive_terms = []
        negative_terms = []

        for arg in expr.args:
            if isinstance(arg, sympy.Mul) and len(arg.args) >= 1:
                if arg.args[0] == -1:
                    # This is a negative term
                    if len(arg.args) == 2:
                        negative_terms.append(arg.args[1])
                    else:
                        negative_terms.append(sympy.Mul(*arg.args[1:]))
                else:
                    positive_terms.append(arg)
            else:
                positive_terms.append(arg)

        # If we have exactly 1 positive and 1 negative, it's a subtraction
        if len(positive_terms) == 1 and len(negative_terms) == 1:
            left = sympy_to_prefix(positive_terms[0])
            right = sympy_to_prefix(negative_terms[0])
            return f"- {left} {right}"

        # Otherwise, treat as addition
        args = [sympy_to_prefix(arg) for arg in expr.args]
        if len(args) == 2:
            return f"+ {args[0]} {args[1]}"
        else:
            result = args[0]
            for arg in args[1:]:
                result = f"+ {result} {arg}"
            return result

    # Fallback: try to handle as generic expression
    if hasattr(expr, 'func') and hasattr(expr, 'args') and expr.args:
        func_name = str(expr.func).split('.')[-1].lower()
        args = [sympy_to_prefix(arg) for arg in expr.args]
        return f"{func_name} {' '.join(args)}"

    # Last resort: return string representation
    return str(expr)


def parse_infix_prompt(prompt_text):
    """
    Parse an infix prompt to extract vars, operators, constants, and expression.

    Args:
        prompt_text: String in format:
            vars: x_1, x_2, ...
            oper: +, -, *, ...
            cons: C
            expr: x_1 + x_2

    Returns:
        dict with keys: vars, oper, cons, expr
    """
    lines = prompt_text.strip().split('\n')
    result = {}

    for line in lines:
        if line.startswith('vars:'):
            vars_str = line.replace('vars:', '').strip()
            result['vars'] = [v.strip() for v in vars_str.split(',')]
        elif line.startswith('oper:'):
            oper_str = line.replace('oper:', '').strip()
            result['oper'] = [o.strip() for o in oper_str.split(',')]
        elif line.startswith('cons:'):
            result['cons'] = line.replace('cons:', '').strip()
        elif line.startswith('expr:'):
            expr_text = line.replace('expr:', '').strip()
            # Remove <|endofex|> marker if present
            expr_text = expr_text.replace('<|endofex|>', '').strip()
            result['expr'] = expr_text

    return result


def convert_infix_to_prefix_prompt(infix_prompt):
    """
    Convert an infix prompt to prefix format.

    Args:
        infix_prompt: String with infix notation prompt

    Returns:
        str: Prompt in prefix notation with same vars/operators
    """
    # Parse infix prompt
    parsed = parse_infix_prompt(infix_prompt)

    # Parse the expression
    try:
        expr_str = parsed['expr']

        # Handle special case: C needs to be treated as a symbol
        expr_str_sympy = expr_str.replace('C', 'C_const')

        # Parse expression
        sympy_expr = sympy.sympify(expr_str_sympy, evaluate=False)

        # Convert to prefix
        prefix_expr = sympy_to_prefix(sympy_expr)

        # Restore C
        prefix_expr = prefix_expr.replace('C_const', 'C')

        # Build prefix prompt
        prefix_prompt = f"vars: {', '.join(parsed['vars'])}\n"
        prefix_prompt += f"oper: {', '.join(parsed['oper'])}\n"
        prefix_prompt += f"cons: {parsed['cons']}\n"
        prefix_prompt += f"expr: {prefix_expr}"

        return prefix_prompt

    except Exception as e:
        print(f"Error converting expression: {parsed['expr']}")
        print(f"Error: {e}")
        return None


def process_dataset(dataset_name='augustocsc/sintetico_natural',
                   split='test',
                   output_path='./data/processed/700K_prefix_converted'):
    """
    Process the entire dataset, converting infix to prefix.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to process
        output_path: Where to save the converted dataset

    Returns:
        Dataset with new column 'p_prompt_n_converted'
    """
    print(f"Loading dataset {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split)

    print(f"Dataset loaded: {len(ds)} examples")
    print(f"Columns: {ds.column_names}")

    # Check if i_prompt_n exists
    if 'i_prompt_n' not in ds.column_names:
        raise ValueError("Column 'i_prompt_n' not found in dataset!")

    # Convert all examples
    converted_prompts = []
    conversion_success = []

    print("\nConverting infix to prefix...")
    for i, example in enumerate(tqdm(ds)):
        infix_prompt = example['i_prompt_n']
        prefix_prompt = convert_infix_to_prefix_prompt(infix_prompt)

        if prefix_prompt is not None:
            converted_prompts.append(prefix_prompt)
            conversion_success.append(True)
        else:
            # Keep original if conversion failed
            converted_prompts.append(infix_prompt)
            conversion_success.append(False)

    # Add new column to dataset
    ds = ds.add_column('p_prompt_n_converted', converted_prompts)
    ds = ds.add_column('conversion_success', conversion_success)

    success_rate = sum(conversion_success) / len(conversion_success) * 100
    print(f"\nConversion success rate: {success_rate:.2f}% ({sum(conversion_success)}/{len(conversion_success)})")

    # Save locally
    print(f"\nSaving dataset to {output_path}...")
    ds.save_to_disk(output_path)

    print("\n[OK] Dataset saved successfully!")

    return ds


def process_hf_dataset_with_split(dataset_name='augustocsc/sintetico_natural',
                                   data_dir='700K',
                                   output_path='./1_data/processed/700K_prefix_682k',
                                   test_size=0.1,
                                   seed=42):
    """
    Process HuggingFace dataset with the same train/val split used in training.

    This matches the exact split used in train_with_json.py:
    - Loads train split from HF (758K)
    - Splits into 90% train / 10% validation (682K / 76K)
    - Converts both to prefix notation

    Args:
        dataset_name: HuggingFace dataset name
        data_dir: Data directory within dataset
        output_path: Where to save converted dataset
        test_size: Validation split size (0.1 = 10%)
        seed: Random seed for reproducibility (42 matches training)
    """
    print(f"Loading dataset {dataset_name} (data_dir={data_dir})...")
    ds = load_dataset(dataset_name, data_dir=data_dir, split='train')

    print(f"Loaded {len(ds):,} examples from train split")
    print(f"Splitting: {int((1-test_size)*100)}% train / {int(test_size*100)}% validation (seed={seed})")

    # Apply same split as training script
    split_ds = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds = split_ds['train']
    val_ds = split_ds['test']

    print(f"\nTrain: {len(train_ds):,} examples")
    print(f"Validation: {len(val_ds):,} examples")

    # Convert train
    print("\n" + "="*60)
    print("Converting TRAIN split")
    print("="*60)

    train_converted = []
    train_success = []

    for example in tqdm(train_ds, desc="Converting train"):
        infix_prompt = example['i_prompt_n']
        prefix_prompt = convert_infix_to_prefix_prompt(infix_prompt)

        if prefix_prompt is not None:
            train_converted.append(prefix_prompt)
            train_success.append(True)
        else:
            train_converted.append(infix_prompt)
            train_success.append(False)

    train_ds = train_ds.add_column('p_prompt_n_converted', train_converted)
    train_ds = train_ds.add_column('conversion_success', train_success)

    train_success_rate = sum(train_success) / len(train_success) * 100
    print(f"\nTrain conversion: {sum(train_success):,}/{len(train_success):,} ({train_success_rate:.1f}%)")

    # Convert validation
    print("\n" + "="*60)
    print("Converting VALIDATION split")
    print("="*60)

    val_converted = []
    val_success = []

    for example in tqdm(val_ds, desc="Converting validation"):
        infix_prompt = example['i_prompt_n']
        prefix_prompt = convert_infix_to_prefix_prompt(infix_prompt)

        if prefix_prompt is not None:
            val_converted.append(prefix_prompt)
            val_success.append(True)
        else:
            val_converted.append(infix_prompt)
            val_success.append(False)

    val_ds = val_ds.add_column('p_prompt_n_converted', val_converted)
    val_ds = val_ds.add_column('conversion_success', val_success)

    val_success_rate = sum(val_success) / len(val_success) * 100
    print(f"\nValidation conversion: {sum(val_success):,}/{len(val_success):,} ({val_success_rate:.1f}%)")

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })

    # Save
    print(f"\nSaving dataset to {output_path}...")
    dataset_dict.save_to_disk(output_path)

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total converted: {sum(train_success) + sum(val_success):,}")
    print(f"Overall success rate: {(sum(train_success) + sum(val_success)) / (len(train_success) + len(val_success)) * 100:.1f}%")

    return dataset_dict


def process_csv_files(input_dir, output_dir, chunksize=10000):
    """
    Process local CSV files (train, validation, test) and convert infix to prefix.

    Args:
        input_dir: Directory containing train_700K.csv, validation_700K.csv, test_700K.csv
        output_dir: Directory to save converted CSV files
        chunksize: Number of rows to process at once (for memory efficiency)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    files_to_process = {
        'train': 'train_700K.csv',
        'validation': 'validation_700K.csv',
        'test': 'test_700K.csv'
    }

    for split_name, filename in files_to_process.items():
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(input_path):
            print(f"\n[SKIP] {filename} not found at {input_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split_name}: {filename}")
        print(f"{'='*60}")

        # Count total rows for progress bar
        print("Counting rows...")
        total_rows = sum(1 for _ in open(input_path, encoding='utf-8')) - 1  # -1 for header
        print(f"Total rows: {total_rows:,}")

        # Process in chunks
        converted_count = 0
        failed_count = 0

        first_chunk = True

        with tqdm(total=total_rows, desc=f"Converting {split_name}") as pbar:
            for chunk in pd.read_csv(input_path, chunksize=chunksize):
                converted_prompts = []
                conversion_success = []

                for idx, row in chunk.iterrows():
                    # Get the infix prompt from 'text' column
                    infix_prompt = row['text']

                    # Convert to prefix
                    prefix_prompt = convert_infix_to_prefix_prompt(infix_prompt)

                    if prefix_prompt is not None:
                        converted_prompts.append(prefix_prompt)
                        conversion_success.append(True)
                        converted_count += 1
                    else:
                        # Keep original if conversion fails
                        converted_prompts.append(infix_prompt)
                        conversion_success.append(False)
                        failed_count += 1

                    pbar.update(1)

                # Add new columns
                chunk['p_prompt_n_converted'] = converted_prompts
                chunk['conversion_success'] = conversion_success

                # Save chunk
                if first_chunk:
                    chunk.to_csv(output_path, index=False, mode='w', encoding='utf-8')
                    first_chunk = False
                else:
                    chunk.to_csv(output_path, index=False, mode='a', header=False, encoding='utf-8')

        success_rate = (converted_count / total_rows * 100) if total_rows > 0 else 0

        print(f"\n[OK] {split_name} completed:")
        print(f"  Converted: {converted_count:,} ({success_rate:.1f}%)")
        print(f"  Failed: {failed_count:,}")
        print(f"  Saved to: {output_path}")


def upload_to_hub(dataset, repo_id, token=None):
    """
    Upload the converted dataset to HuggingFace Hub.

    Args:
        dataset: Dataset object to upload
        repo_id: Repository ID (e.g., 'username/dataset-name')
        token: HuggingFace API token (optional, uses cached if not provided)
    """
    print(f"\nUploading dataset to {repo_id}...")

    try:
        dataset.push_to_hub(repo_id, token=token)
        print(f"[OK] Dataset uploaded successfully to {repo_id}")
        print(f"  View at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[FAIL] Failed to upload dataset: {e}")
        print("  Make sure you have write permissions to the repository")
        print("  You may need to run: huggingface-cli login")


def main():
    parser = argparse.ArgumentParser(
        description="Convert infix expressions to prefix notation"
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='augustocsc/sintetico_natural',
        help='HuggingFace dataset name'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to process'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='./1_data/processed/700K_prefix_converted',
        help='Path to save converted dataset'
    )

    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload converted dataset to HuggingFace Hub'
    )

    parser.add_argument(
        '--repo_id',
        type=str,
        default=None,
        help='Repository ID for upload (e.g., username/dataset-name)'
    )

    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Test conversion on first 10 examples only'
    )

    parser.add_argument(
        '--process_csv',
        action='store_true',
        help='Process local CSV files (train, validation, test)'
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        default='./1_data/processed/700K_fixed',
        help='Directory containing train_700K.csv, validation_700K.csv, test_700K.csv'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./1_data/processed/700K_prefix_full',
        help='Directory to save converted CSV files'
    )

    parser.add_argument(
        '--chunksize',
        type=int,
        default=10000,
        help='Number of rows to process at once (for memory efficiency)'
    )

    parser.add_argument(
        '--use_training_split',
        action='store_true',
        help='Use same 90/10 train/val split as training (682K train + 76K val)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='700K',
        help='Data directory within HuggingFace dataset'
    )

    args = parser.parse_args()

    # Test mode
    if args.test_only:
        print("=" * 60)
        print("TEST MODE: Converting first 10 examples")
        print("=" * 60)

        ds = load_dataset(args.dataset_name, split='test[:10]')

        for i, example in enumerate(ds):
            print(f"\n{'='*60}")
            print(f"Example {i+1}")
            print(f"{'='*60}")
            print("\nINFIX:")
            print(example['i_prompt_n'])

            prefix_prompt = convert_infix_to_prefix_prompt(example['i_prompt_n'])

            if prefix_prompt:
                print("\nCONVERTED PREFIX:")
                print(prefix_prompt)
                print("\n[OK] Conversion successful")
            else:
                print("\n[FAIL] Conversion failed")

        return

    # Training split mode (682K train + 76K val)
    if args.use_training_split:
        print("=" * 60)
        print("TRAINING SPLIT MODE: 90% train / 10% validation")
        print("=" * 60)
        print("This matches the exact split used in train_with_json.py")
        print(f"Dataset: {args.dataset_name} (data_dir={args.data_dir})")
        print(f"Output: {args.output_path}")
        print("=" * 60)

        dataset_dict = process_hf_dataset_with_split(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            output_path=args.output_path,
            test_size=0.1,
            seed=42
        )

        # Show examples
        print("\n" + "=" * 60)
        print("SAMPLE CONVERSIONS")
        print("=" * 60)

        print("\nTRAIN example:")
        print("INFIX:")
        print(dataset_dict['train'][0]['i_prompt_n'])
        print("\nPREFIX:")
        print(dataset_dict['train'][0]['p_prompt_n_converted'])

        print("\nVALIDATION example:")
        print("INFIX:")
        print(dataset_dict['validation'][0]['i_prompt_n'])
        print("\nPREFIX:")
        print(dataset_dict['validation'][0]['p_prompt_n_converted'])

        # Upload if requested
        if args.upload:
            if args.repo_id is None:
                print("\n[ERROR] --repo_id required for upload")
                print("  Example: --repo_id augustocsc/sintetico_natural_prefix_682k")
            else:
                print(f"\n{'='*60}")
                print(f"Uploading to HuggingFace Hub: {args.repo_id}")
                print("="*60)

                try:
                    dataset_dict.push_to_hub(args.repo_id)
                    print(f"[OK] Dataset uploaded successfully!")
                    print(f"  View at: https://huggingface.co/datasets/{args.repo_id}")
                except Exception as e:
                    print(f"[FAIL] Failed to upload: {e}")
                    print("  Make sure you have write permissions")
                    print("  Run: huggingface-cli login")
        else:
            print("\n" + "=" * 60)
            print("To upload to HuggingFace Hub, run:")
            print(f"  python {__file__} --use_training_split --upload --repo_id augustocsc/sintetico_natural_prefix_682k")
            print("=" * 60)

        return

    # CSV mode
    if args.process_csv:
        print("=" * 60)
        print("CSV MODE: Processing local CSV files")
        print("=" * 60)
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Chunk size: {args.chunksize:,} rows")
        print("=" * 60)

        process_csv_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chunksize=args.chunksize
        )

        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"Converted files saved to: {args.output_dir}")
        print("\nNext steps:")
        print("1. Verify converted files:")
        print(f"   head -3 {args.output_dir}/train_700K.csv")
        print("2. Upload to HuggingFace (optional):")
        print("   # TODO: Add upload functionality for CSV files")
        print("=" * 60)

        return

    # Full conversion (HuggingFace dataset mode)
    dataset = process_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        output_path=args.output_path
    )

    # Show examples
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSIONS (first 3 examples)")
    print("=" * 60)

    for i in range(min(3, len(dataset))):
        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"{'='*60}")
        print("\nORIGINAL INFIX:")
        print(dataset[i]['i_prompt_n'])
        print("\nCONVERTED PREFIX:")
        print(dataset[i]['p_prompt_n_converted'])

        if 'p_prompt_n' in dataset.column_names:
            print("\nORIGINAL PREFIX (from dataset):")
            print(dataset[i]['p_prompt_n'])

    # Upload if requested
    if args.upload:
        if args.repo_id is None:
            print("\n[ERROR] --repo_id required for upload")
            print("  Example: --repo_id username/sintetico_natural_prefix_converted")
        else:
            upload_to_hub(dataset, args.repo_id)
    else:
        print("\n" + "=" * 60)
        print("To upload to HuggingFace Hub, run:")
        print(f"  python {__file__} --upload --repo_id username/dataset-name")
        print("=" * 60)


if __name__ == '__main__':
    main()
