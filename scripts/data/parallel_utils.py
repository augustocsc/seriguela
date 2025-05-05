# parallel_utils.py

from joblib import Parallel, delayed
import pandas as pd
from .data_augmentation import generate_expression_instructions

def augment_dataframe_parallel(df, expression_col="expression", n_jobs=-1):
    """
    Parallelized augmentation of a DataFrame with math expressions.

    Args:
        df (pd.DataFrame): DataFrame with a column of expressions.
        expression_col (str): Name of the column with expressions.
        n_jobs (int): Number of parallel workers (-1 = all cores).

    Returns:
        pd.DataFrame: Original DataFrame with new instruction columns.
    """
    expressions = df[expression_col].tolist()
    augmented_data = Parallel(n_jobs=n_jobs)(
        delayed(generate_expression_instructions)(expr) for expr in expressions
    )

    df_aug = df.copy()
    df_aug["simple"] = [item["Simple_Instruct"] for item in augmented_data]
    df_aug["key_value"] = [item["Key_Value"] for item in augmented_data]
    df_aug["delimiter"] = [item["Delimiter_Based"] for item in augmented_data]
    df_aug["minimalist"] = [item["Minimalist"] for item in augmented_data]

    return df_aug
