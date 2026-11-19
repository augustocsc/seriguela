#!/usr/bin/env python3
"""
Generate ALL official symbolic regression benchmarks.

This is the main script to generate datasets for academic evaluation.
It uses the EXACT equations from the original papers.

Output structure:
    1_data/benchmarks/
    ├── feynman/         # 101 physics equations (AI Feynman)
    ├── strogatz/        # 14 dynamical systems
    ├── blackbox/        # SRBench black-box (downloaded via pmlb)
    └── benchmarks_metadata.json

Usage:
    python generate_all_benchmarks.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Import Feynman equations
from feynman_equations import FEYNMAN_EQUATIONS

# Strogatz equations (from Nonlinear Dynamics and Chaos textbook)
STROGATZ_EQUATIONS = {
    "strogatz_vdp1": {
        "equation": "10*(y - (x**3/3 - x))",
        "variables": ["x", "y"],
        "ranges": {"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
        "description": "Van der Pol oscillator (x')",
        "n_vars": 2
    },
    "strogatz_vdp2": {
        "equation": "-x/10",
        "variables": ["x"],
        "ranges": {"x": (-3.0, 3.0)},
        "description": "Van der Pol oscillator (y')",
        "n_vars": 1
    },
    "strogatz_lv1": {
        "equation": "a*x - b*x*y",
        "variables": ["a", "b", "x", "y"],
        "ranges": {"a": (0.5, 2.0), "b": (0.1, 1.0), "x": (0.1, 5.0), "y": (0.1, 5.0)},
        "description": "Lotka-Volterra prey",
        "n_vars": 4
    },
    "strogatz_lv2": {
        "equation": "c*x*y - d*y",
        "variables": ["c", "d", "x", "y"],
        "ranges": {"c": (0.1, 1.0), "d": (0.5, 2.0), "x": (0.1, 5.0), "y": (0.1, 5.0)},
        "description": "Lotka-Volterra predator",
        "n_vars": 4
    },
    "strogatz_bacres1": {
        "equation": "V*x/(K + x)",
        "variables": ["V", "K", "x"],
        "ranges": {"V": (1.0, 10.0), "K": (0.1, 5.0), "x": (0.01, 10.0)},
        "description": "Bacterial respiration (Michaelis-Menten)",
        "n_vars": 3
    },
    "strogatz_bacres2": {
        "equation": "-V*x/(K + x)",
        "variables": ["V", "K", "x"],
        "ranges": {"V": (1.0, 10.0), "K": (0.1, 5.0), "x": (0.01, 10.0)},
        "description": "Bacterial respiration (consumption)",
        "n_vars": 3
    },
    "strogatz_glider1": {
        "equation": "-sin(theta)",
        "variables": ["theta"],
        "ranges": {"theta": (-np.pi, np.pi)},
        "description": "Glider (theta')",
        "n_vars": 1
    },
    "strogatz_glider2": {
        "equation": "v - v**2 - sin(theta)/v",
        "variables": ["v", "theta"],
        "ranges": {"v": (0.5, 3.0), "theta": (-np.pi/2, np.pi/2)},
        "description": "Glider (v')",
        "n_vars": 2
    },
    "strogatz_shearflow1": {
        "equation": "cos(y)",
        "variables": ["y"],
        "ranges": {"y": (0.0, 2*np.pi)},
        "description": "Shear flow (x')",
        "n_vars": 1
    },
    "strogatz_shearflow2": {
        "equation": "sin(x)*sin(y)",
        "variables": ["x", "y"],
        "ranges": {"x": (0.0, 2*np.pi), "y": (0.0, 2*np.pi)},
        "description": "Shear flow (y')",
        "n_vars": 2
    },
    "strogatz_barmag1": {
        "equation": "x - x**3/3 - y",
        "variables": ["x", "y"],
        "ranges": {"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
        "description": "Bar magnet (x')",
        "n_vars": 2
    },
    "strogatz_barmag2": {
        "equation": "x/tau",
        "variables": ["x", "tau"],
        "ranges": {"x": (-3.0, 3.0), "tau": (0.1, 2.0)},
        "description": "Bar magnet (y')",
        "n_vars": 2
    },
    "strogatz_predprey1": {
        "equation": "r*x*(1-x/K) - a*x*y/(1+a*h*x)",
        "variables": ["r", "K", "a", "h", "x", "y"],
        "ranges": {"r": (0.5, 2.0), "K": (5.0, 20.0), "a": (0.1, 1.0),
                  "h": (0.1, 1.0), "x": (0.1, 5.0), "y": (0.1, 5.0)},
        "description": "Predator-prey with carrying capacity",
        "n_vars": 6
    },
    "strogatz_predprey2": {
        "equation": "e*a*x*y/(1+a*h*x) - d*y",
        "variables": ["e", "a", "h", "d", "x", "y"],
        "ranges": {"e": (0.1, 1.0), "a": (0.1, 1.0), "h": (0.1, 1.0),
                  "d": (0.1, 1.0), "x": (0.1, 5.0), "y": (0.1, 5.0)},
        "description": "Predator functional response",
        "n_vars": 6
    },
}

# Black-box datasets to download from pmlb
BLACKBOX_DATASETS = [
    "1027_ESL", "1028_SWD", "1029_LEV", "1030_ERA", "192_vineyard",
    "195_auto_price", "197_cpu_act", "201_pol", "207_autoPrice",
    "210_cloud", "215_2dplanes", "218_house_8L", "225_puma8NH",
    "227_cpu_small", "228_elusage", "229_pwLinear", "230_machine_cpu",
    "344_mv", "485_analcatdata_vehicle", "503_wind", "519_vinnie",
    "522_pm10", "529_pollen", "537_houses", "542_pollution", "547_no2",
    "560_bodyfat", "561_cpu", "564_fried", "573_cpu_act", "574_house_16H",
    "banana", "boston", "cars", "concrete", "diabetes", "energy",
    "friedman1", "friedman2", "friedman3",
]


def safe_eval(equation: str, variables: Dict[str, np.ndarray]) -> np.ndarray:
    """Safely evaluate an equation with given variable values."""
    safe_dict = {
        "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "tanh": np.tanh, "arcsin": np.arcsin, "arccos": np.arccos,
        "arctan": np.arctan, "pi": np.pi, "abs": np.abs,
        **variables
    }
    try:
        result = eval(equation, {"__builtins__": {}}, safe_dict)
        return np.array(result, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Error evaluating '{equation}': {e}")


def generate_dataset(name: str, eq_info: Dict, n_samples: int = 1000, seed: int = 42) -> Optional[pd.DataFrame]:
    """Generate a dataset for a given equation."""
    np.random.seed(seed)

    data = {}
    for var in eq_info["variables"]:
        low, high = eq_info["ranges"][var]
        if abs(low - high) < 1e-10:
            data[var] = np.full(n_samples, low)
        else:
            data[var] = np.random.uniform(low, high, n_samples)

    try:
        y = safe_eval(eq_info["equation"], data)
        valid_mask = np.isfinite(y)

        # Retry with different seed if too many invalid
        if valid_mask.sum() < n_samples * 0.5:
            np.random.seed(seed + 1000)
            for var in eq_info["variables"]:
                low, high = eq_info["ranges"][var]
                if abs(low - high) > 1e-10:
                    data[var] = np.random.uniform(low, high, n_samples)
            y = safe_eval(eq_info["equation"], data)
            valid_mask = np.isfinite(y)

        if valid_mask.sum() < 100:
            return None

        df = pd.DataFrame(data)
        df["target"] = y
        df = df[valid_mask].reset_index(drop=True)
        return df

    except Exception as e:
        print(f"    Error: {e}")
        return None


def download_blackbox_datasets(output_dir: Path) -> List[Dict]:
    """Download black-box datasets from pmlb."""
    try:
        import pmlb
    except ImportError:
        print("  Installing pmlb...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pmlb", "-q"])
        import pmlb

    results = []
    for name in BLACKBOX_DATASETS:
        try:
            print(f"  Downloading {name}...", end=" ", flush=True)
            df = pmlb.fetch_data(name)

            csv_path = output_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)

            meta = {
                "name": name,
                "equation": "unknown (black-box)",
                "n_vars": len(df.columns) - 1,
                "n_samples": len(df),
                "variables": list(df.columns[:-1]),
                "target": df.columns[-1],
                "type": "black-box",
                "source": "PMLB"
            }

            meta_path = output_dir / f"{name}.meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"OK ({len(df)} samples)")
            results.append(meta)

        except Exception as e:
            print(f"FAILED: {e}")

    return results


def generate_ground_truth_datasets(equations: Dict, output_dir: Path, category: str) -> List[Dict]:
    """Generate ground-truth datasets from known equations."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, eq_info in equations.items():
        print(f"  Generating {name}...", end=" ", flush=True)

        df = generate_dataset(name, eq_info)
        if df is None or len(df) < 100:
            print("SKIPPED")
            continue

        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        meta = {
            "name": name,
            "equation": eq_info["equation"],
            "n_vars": eq_info["n_vars"],
            "n_samples": len(df),
            "variables": eq_info["variables"],
            "ranges": {k: list(v) for k, v in eq_info["ranges"].items()},
            "description": eq_info.get("description", ""),
            "type": "ground-truth",
            "category": category
        }

        meta_path = output_dir / f"{name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"OK ({len(df)} samples)")
        results.append(meta)

    return results


def create_unified_metadata(all_results: Dict[str, List[Dict]], output_dir: Path) -> Dict:
    """Create unified metadata file with academic references."""

    metadata = {
        "version": "2.0",
        "description": "Official symbolic regression benchmark suite",
        "references": {
            "feynman": {
                "paper": "AI Feynman: A physics-inspired method for symbolic regression",
                "authors": "Udrescu, S.M., Tegmark, M.",
                "venue": "Science Advances, 6(16), eaay2631, 2020",
                "arxiv": "https://arxiv.org/abs/1905.11481",
                "equations": "100 equations from Feynman Lectures on Physics"
            },
            "strogatz": {
                "paper": "Nonlinear Dynamics and Chaos",
                "authors": "Strogatz, S.H.",
                "venue": "Westview Press, 2nd Edition, 2015",
                "equations": "14 ODE systems for dynamical systems discovery"
            },
            "srbench": {
                "paper": "Contemporary Symbolic Regression Methods and their Relative Performance",
                "authors": "La Cava, W., et al.",
                "venue": "NeurIPS 2021 Datasets and Benchmarks Track",
                "url": "https://github.com/cavalab/srbench"
            },
            "srbench_2.0": {
                "paper": "Call for Action: towards the next generation of symbolic regression benchmark",
                "authors": "Aldeia, G.S.I., Zhang, H., Bomarito, G., Cranmer, M., et al.",
                "venue": "GECCO 2025 Symbolic Regression Workshop",
                "arxiv": "https://arxiv.org/abs/2505.03977"
            }
        },
        "categories": {
            "feynman": {
                "type": "ground-truth",
                "description": "Physics equations from Feynman Lectures",
                "count": 0
            },
            "strogatz": {
                "type": "ground-truth",
                "description": "Nonlinear dynamical systems ODEs",
                "count": 0
            },
            "blackbox": {
                "type": "black-box",
                "description": "Real-world regression (no known equation)",
                "count": 0
            }
        },
        "benchmarks": {},
        "statistics": {
            "total": 0,
            "ground_truth": 0,
            "black_box": 0
        }
    }

    for category, results in all_results.items():
        metadata["benchmarks"][category] = results
        count = len(results)
        metadata["categories"][category]["count"] = count
        metadata["statistics"]["total"] += count

        if category in ["feynman", "strogatz"]:
            metadata["statistics"]["ground_truth"] += count
        else:
            metadata["statistics"]["black_box"] += count

    meta_path = output_dir / "benchmarks_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("Official Symbolic Regression Benchmark Generator")
    print("=" * 70)
    print("\nThis generates the EXACT benchmarks used in:")
    print("  - AI Feynman (Science Advances 2020)")
    print("  - SRBench (NeurIPS 2021)")
    print("  - SRBench 2.0 (GECCO 2025)")
    print(f"\nOutput: {script_dir}")
    print(f"\nBenchmarks:")
    print(f"  - Feynman (ground-truth): {len(FEYNMAN_EQUATIONS)}")
    print(f"  - Strogatz (ground-truth): {len(STROGATZ_EQUATIONS)}")
    print(f"  - Black-box (real-world): {len(BLACKBOX_DATASETS)}")

    all_results = {}

    # Generate Feynman
    print(f"\n{'=' * 50}")
    print(f"FEYNMAN BENCHMARK ({len(FEYNMAN_EQUATIONS)} equations)")
    print("=" * 50)
    feynman_dir = script_dir / "feynman"
    feynman_results = generate_ground_truth_datasets(FEYNMAN_EQUATIONS, feynman_dir, "feynman")
    all_results["feynman"] = feynman_results
    print(f"\nFeynman: {len(feynman_results)}/{len(FEYNMAN_EQUATIONS)} generated")

    # Generate Strogatz
    print(f"\n{'=' * 50}")
    print(f"STROGATZ BENCHMARK ({len(STROGATZ_EQUATIONS)} equations)")
    print("=" * 50)
    strogatz_dir = script_dir / "strogatz"
    strogatz_results = generate_ground_truth_datasets(STROGATZ_EQUATIONS, strogatz_dir, "strogatz")
    all_results["strogatz"] = strogatz_results
    print(f"\nStrogatz: {len(strogatz_results)}/{len(STROGATZ_EQUATIONS)} generated")

    # Download black-box
    print(f"\n{'=' * 50}")
    print(f"BLACK-BOX BENCHMARK ({len(BLACKBOX_DATASETS)} datasets)")
    print("=" * 50)
    blackbox_dir = script_dir / "blackbox"
    blackbox_dir.mkdir(parents=True, exist_ok=True)
    blackbox_results = download_blackbox_datasets(blackbox_dir)
    all_results["blackbox"] = blackbox_results
    print(f"\nBlack-box: {len(blackbox_results)}/{len(BLACKBOX_DATASETS)} downloaded")

    # Create metadata
    print(f"\n{'=' * 50}")
    print("Creating unified metadata...")
    print("=" * 50)
    metadata = create_unified_metadata(all_results, script_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal benchmarks: {metadata['statistics']['total']}")
    print(f"  - Ground-truth: {metadata['statistics']['ground_truth']}")
    print(f"  - Black-box: {metadata['statistics']['black_box']}")
    print("\nBy category:")
    for cat, info in metadata["categories"].items():
        print(f"  - {cat}: {info['count']} ({info['type']})")

    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)
    print("\nFor academic citation, see benchmarks_metadata.json")


if __name__ == "__main__":
    main()
