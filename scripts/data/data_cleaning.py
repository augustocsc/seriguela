import re
import pandas as pd
import numpy as np
from sympy import sympify, Eq
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from sympy import simplify, sympify
from sympy.core.sympify import SympifyError
import swifter
import random

from joblib import Parallel, delayed


from tqdm.auto import tqdm

def apply_chunk(chunk, func):
    """Helper function to apply a function to a chunk of data."""
    return chunk.apply(func)

def parallel_apply(series, func, n_jobs=None):
    n_jobs = mp.cpu_count() if n_jobs is None else n_jobs
    # Split into roughly equal chunks
    chunks = np.array_split(series, n_jobs)
    with mp.Pool(n_jobs) as pool:
        # Use the helper function instead of a lambda
        results = pool.starmap(apply_chunk, [(chunk, func) for chunk in chunks])
    # Concatenate the resulting Series
    return pd.concat(results)

def canonicalize_expr(expr, canonicalizer=simplify):
    canon = canonicalizer(expr)
    return (hash(canon), canon, expr)

def replace_constants(equation):
    # Match positive/negative floats and integers not part of variable names
    pattern = r'(?<![\w.])(?:[-+]?\d*\.\d+|\d+)(?![\w.])'
    return re.sub(pattern, 'C', equation)


def augment_expression(equation, var_prefix='x', max_index=10, p=0.5):
    """
    1. Replace all standalone numeric constants (including scientific notation) with 'C'.
    2. For each occurrence of a variable (e.g., x_1), with probability p replace it
       by a randomly chosen new variable x_1…x_max_index; otherwise leave as is.
    """
    # Step 1: Replace constants (including scientific notation)
    const_pattern = r'(?<![\w.])(?:[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)(?![\w.])'
    equation = re.sub(const_pattern, 'C', equation)
    
    # Step 2: Replace variables with probability p
    var_pattern = rf'\b{var_prefix}_\d+\b'
    def repl(match):
        if random.random() < p:
            new_idx = random.randint(1, max_index)
            return f"{var_prefix}_{new_idx}"
        return match.group(0)
    
    return re.sub(var_pattern, repl, equation)



def is_valid_equation(equation_str):
    """Verifica se uma string representa uma expressão matemática válida para o SymPy."""
    if not isinstance(equation_str, str):
        return False
    if pd.isna(equation_str) or equation_str.strip() == '':
        return False
    
    try:
        # Tenta analisar a expressão
        expr = parse_expr(equation_str.strip())
        return True
    except (SympifyError, SyntaxError, ValueError, TypeError, AttributeError):
        print(f"Erro ao analisar a equação: {equation_str}")
        
        return False

def canonical_form(expr_str):
    """
    Recebe uma expressão como string e retorna sua forma canônica (simplificada).
    """
    try:
        #expr_str = sympify(expr_str)
        canonica = simplify(expr_str).expand()
        return str(canonica)
    except SympifyError as e:
        return f"Erro ao interpretar a expressão: {expr_str}"