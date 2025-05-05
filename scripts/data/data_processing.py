import os
import sys
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# Adjust import paths for custom modules
def setup_paths():
    for folder in ["../scripts", "../classes"]:
        path = os.path.abspath(os.path.join(folder))
        if path not in sys.path:
            sys.path.append(path)

setup_paths()

# Local imports after path setup
import scripts.data.data_cleaning as dc
from expression import Expression
from data.parallel_utils import augment_dataframe_parallel

def parallel_apply(series, func, n_jobs=None):
    """Apply a function to a pandas Series in parallel."""
    def apply_chunk(chunk, func):
        return chunk.apply(func)

    n_jobs = mp.cpu_count() if n_jobs is None else n_jobs
    chunks = np.array_split(series, n_jobs)
    with mp.Pool(n_jobs) as pool:
        results = pool.starmap(apply_chunk, [(chunk, func) for chunk in chunks])
    return pd.concat(results)

def process_chunk(chunk):
    """Clean and transform a single data chunk."""
    chunk = chunk[['eq']]
    chunk = chunk[~chunk['eq'].str.contains('ERROR_simplify')]
    chunk['eq'] = parallel_apply(chunk['eq'], dc.augment_expression)
    chunk.rename(columns={'eq': 'infix_expr'}, inplace=True)
    chunk['prefix_expr'] = parallel_apply(chunk['infix_expr'], Expression.infix_to_prefix)
    return chunk

def process_file(file_path, chunk_size=100000):
    """Process the CSV file in chunks."""
    processed_chunks = []
    total_rows = sum(1 for _ in open(file_path)) - 1
    total_chunks = (total_rows // chunk_size) + 1

    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            processed_chunk = process_chunk(chunk)
            processed_chunks.append(processed_chunk)
            pbar.update(1)

    return pd.concat(processed_chunks, ignore_index=True)

def augment_df(df):
    """Apply augmentation to both infix and prefix expressions."""
    df = augment_dataframe_parallel(df, expression_col="infix_expr", n_jobs=4)
    df.rename(columns={
        'simple': 'i_simple',
        'key_value': 'i_key_value',
        'delimiter': 'i_delimiter',
        'minimalist': 'i_minimalist'
    }, inplace=True)

    df = augment_dataframe_parallel(df, expression_col="prefix_expr", n_jobs=4)
    df.rename(columns={
        'simple': 'p_simple',
        'key_value': 'p_key_value',
        'delimiter': 'p_delimiter',
        'minimalist': 'p_minimalist'
    }, inplace=True)

    return df

def split_and_save(df, base_file_path):
    """Split into train/val/test and save them."""
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    file = os.path.basename(base_file_path)
    base_dir = f'../data/processed/{file.replace(".csv", "")}'
    os.makedirs(base_dir, exist_ok=True)

    train_df.to_csv(os.path.join(base_dir, f"train_{file}"), index=False)
    val_df.to_csv(os.path.join(base_dir, f"val_{file}"), index=False)
    test_df.to_csv(os.path.join(base_dir, f"test_{file}"), index=False)
    df.to_csv(os.path.join(base_dir, file), index=False)

def main():
    parser = argparse.ArgumentParser(description="Process a raw equation CSV file.")
    parser.add_argument("file_path", type=str, help="Path to the raw CSV file to process.", default="../data/raw/13k.csv")
    args = parser.parse_args()

    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    df_processed = process_file(file_path)
    df_processed.drop_duplicates(subset=['infix_expr'], inplace=True)
    df_augmented = augment_df(df_processed)
    split_and_save(df_augmented, file_path)

if __name__ == '__main__':
    main()
