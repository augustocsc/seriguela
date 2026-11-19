"""
Feynman and Strogatz benchmark loader for run_experiment.py.

Reads pre-downloaded CSV files from 1_data/benchmarks/{feynman,strogatz}/,
renames physical variable names to x_1, x_2, ..., and splits 75/25 train/test.

Usage:
    from utils.feynman_loader import load_benchmark_data

    data = load_benchmark_data("feynman_I_14_3", seed=42)
    # data = {"train": {"x": ..., "y": ...}, "test": ..., "equation": ..., "valid_variables": ...}
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Root of the seriguela repo (2 levels up from this file)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_FEYNMAN_DIR = _REPO_ROOT / "1_data" / "benchmarks" / "feynman"
_STROGATZ_DIR = _REPO_ROOT / "1_data" / "benchmarks" / "strogatz"


def load_benchmark_data(problem: str, seed: int, test_fraction: float = 0.25,
                        n_samples: Optional[int] = None) -> dict:
    """Load a Feynman or Strogatz benchmark problem.

    Variable names are remapped to x_1, x_2, ... so the model receives
    the same format it was trained on.

    Args:
        problem:       Problem name, e.g. "feynman_I_14_3" or "strogatz_bacres1"
        seed:          Random seed for reproducible train/test split
        test_fraction: Fraction of data used for test set (default 0.25, SRBench protocol)
        n_samples:     If given, subsample this many rows from the full dataset
                       (100K rows). None = use all rows.

    Returns:
        {
            "train": {"x": np.ndarray, "y": np.ndarray},
            "test":  {"x": np.ndarray, "y": np.ndarray},
            "equation": str,           # ground-truth formula (if available)
            "valid_variables": set,    # {"x_1", "x_2", ...}
            "original_vars": list,     # physical names ["m", "g", "z"]
            "var_map": dict,           # {"m": "x_1", "g": "x_2", "z": "x_3"}
        }
    """
    if problem.startswith("feynman"):
        data_dir = _FEYNMAN_DIR
    elif problem.startswith("strogatz"):
        data_dir = _STROGATZ_DIR
    else:
        raise ValueError(f"Unknown benchmark type for problem '{problem}'. "
                         f"Expected 'feynman_*' or 'strogatz_*'.")

    csv_path = data_dir / f"{problem}.csv"
    meta_path = data_dir / f"{problem}.meta.json"

    if not csv_path.exists() or _is_lfs_pointer(csv_path):
        _auto_download(problem, csv_path, meta_path)

    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    # Identify feature columns (everything except 'target')
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]
    n_vars = len(feature_cols)

    # Subsample if requested (default: use all, up to 100K rows)
    rng = np.random.RandomState(seed)
    if n_samples is not None and n_samples < len(df):
        idx = rng.choice(len(df), size=n_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    # --- Variable remapping: physical names → x_1, x_2, ... ---
    var_map = {phys: f"x_{i+1}" for i, phys in enumerate(feature_cols)}
    original_vars = feature_cols

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # --- Train/test split (75/25, SRBench protocol) ---
    n_total = len(df)
    n_test = max(1, int(n_total * test_fraction))
    idx_all = rng.permutation(n_total)
    idx_test = idx_all[:n_test]
    idx_train = idx_all[n_test:]

    x_train = X[idx_train]
    y_train = y[idx_train]
    x_test = X[idx_test]
    y_test = y[idx_test]

    # --- Ground-truth equation (remapped to x_1/x_2/...) ---
    equation = _get_equation(problem, meta_path, var_map)

    valid_variables = {f"x_{i+1}" for i in range(n_vars)}

    return {
        "train": {"x": x_train, "y": y_train},
        "test":  {"x": x_test,  "y": y_test},
        "equation": equation,
        "valid_variables": valid_variables,
        "original_vars": original_vars,
        "var_map": var_map,
        "n_vars": n_vars,
    }


def _is_lfs_pointer(path: Path) -> bool:
    """Return True if the file is a git-lfs pointer (not real data)."""
    try:
        with open(path, "r", errors="ignore") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs") or "oid sha256:" in first_line
    except Exception:
        return False


def _auto_download(problem: str, csv_path: Path, meta_path: Path):
    """Download CSV from PMLB if not present locally."""
    import urllib.request
    import gzip
    import shutil

    # Build PMLB URL from problem name
    base_url = f"https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{problem}/{problem}.tsv.gz"

    print(f"[feynman_loader] Downloading {problem} from PMLB...", flush=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    gz_path = csv_path.with_suffix(".tsv.gz")
    tsv_path = csv_path.with_suffix(".tsv")

    try:
        urllib.request.urlretrieve(base_url, gz_path)

        # Decompress gz → tsv
        with gzip.open(gz_path, 'rb') as f_in, open(tsv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()

        # Convert tsv → csv (rename 'target' col if needed)
        import pandas as pd
        df = pd.read_csv(tsv_path, sep='\t')
        # PMLB uses 'target' as the last column — rename if it's the output
        if 'target' not in df.columns:
            df = df.rename(columns={df.columns[-1]: 'target'})
        df.to_csv(csv_path, index=False)
        tsv_path.unlink()

        print(f"[feynman_loader] Downloaded {problem}: {len(df)} rows, {len(df.columns)-1} features", flush=True)

    except Exception as e:
        # Cleanup partial files
        for p in [gz_path, tsv_path]:
            if p.exists():
                p.unlink()
        raise RuntimeError(
            f"Failed to download {problem} from PMLB: {e}\n"
            f"URL tried: {base_url}\n"
            f"Run manually: 1_data/benchmarks/download_all_benchmarks.py"
        ) from e


def _get_equation(problem: str, meta_path: Path, var_map: dict) -> str:
    """Extract and remap ground-truth equation from feynman_equations.py."""
    try:
        import importlib.util
        equations_py = _FEYNMAN_DIR.parent / "feynman_equations.py"
        if equations_py.exists():
            spec = importlib.util.spec_from_file_location("feynman_eq", equations_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            eqs = getattr(mod, "FEYNMAN_EQUATIONS", {})
            if problem in eqs:
                expr = eqs[problem]["equation"]
                # Remap physical variable names to x_1, x_2, ...
                # Sort by length descending to avoid partial replacements
                for phys, xi in sorted(var_map.items(), key=lambda kv: -len(kv[0])):
                    expr = expr.replace(phys, xi)
                return expr
    except Exception:
        pass

    # Fallback: return problem name as placeholder
    return f"unknown ({problem})"


def is_feynman_or_strogatz(problem: str) -> bool:
    """Return True if the problem name belongs to Feynman/Strogatz benchmarks."""
    return problem.startswith("feynman") or problem.startswith("strogatz")
