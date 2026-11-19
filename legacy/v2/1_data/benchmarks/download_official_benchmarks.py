#!/usr/bin/env python3
"""
Download official SRBench datasets from PMLB (Penn Machine Learning Benchmarks).

This script downloads the EXACT datasets used in:
- SRBench (NeurIPS 2021): La Cava et al. "Contemporary Symbolic Regression Methods
  and their Relative Performance"
- SRBench 2.0 (GECCO 2025): "Call for Action: towards the next generation of
  symbolic regression benchmark"

References:
- PMLB: https://github.com/EpistasisLab/pmlb
- SRBench: https://github.com/cavalab/srbench
- AI Feynman: https://arxiv.org/abs/1905.11481

Dataset Categories:
1. Ground-truth (symbolic): Feynman + Strogatz - equations are KNOWN
2. Black-box: Real-world regression problems - equations are UNKNOWN

Usage:
    python download_official_benchmarks.py

Requirements:
    - Git with LFS support installed
    - ~2GB disk space for full download
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import tempfile


# Official dataset lists from SRBench
# Source: https://github.com/cavalab/srbench/blob/master/experiment/analyze.py

FEYNMAN_DATASETS = [
    # Feynman I - Mechanics (50 equations)
    "feynman_I_6_2", "feynman_I_6_2a", "feynman_I_6_2b",
    "feynman_I_8_14", "feynman_I_9_18", "feynman_I_10_7",
    "feynman_I_11_19", "feynman_I_12_1", "feynman_I_12_2",
    "feynman_I_12_4", "feynman_I_12_5", "feynman_I_12_11",
    "feynman_I_13_4", "feynman_I_13_12", "feynman_I_14_3",
    "feynman_I_14_4", "feynman_I_15_3t", "feynman_I_15_3x",
    "feynman_I_15_10", "feynman_I_16_6", "feynman_I_18_4",
    "feynman_I_18_12", "feynman_I_18_14", "feynman_I_24_6",
    "feynman_I_25_13", "feynman_I_26_2", "feynman_I_27_6",
    "feynman_I_29_4", "feynman_I_29_16", "feynman_I_30_3",
    "feynman_I_30_5", "feynman_I_32_5", "feynman_I_32_17",
    "feynman_I_34_1", "feynman_I_34_8", "feynman_I_34_10",
    "feynman_I_34_14", "feynman_I_34_27", "feynman_I_37_4",
    "feynman_I_38_12", "feynman_I_39_1", "feynman_I_39_11",
    "feynman_I_39_22", "feynman_I_40_1", "feynman_I_41_16",
    "feynman_I_43_16", "feynman_I_43_31", "feynman_I_43_43",
    "feynman_I_44_4", "feynman_I_47_23", "feynman_I_48_2",
    "feynman_I_50_26",
    # Feynman II - Electromagnetism (38 equations)
    "feynman_II_2_42", "feynman_II_3_24", "feynman_II_4_23",
    "feynman_II_6_11", "feynman_II_6_15a", "feynman_II_6_15b",
    "feynman_II_8_7", "feynman_II_8_31", "feynman_II_10_9",
    "feynman_II_11_3", "feynman_II_11_17", "feynman_II_11_20",
    "feynman_II_11_27", "feynman_II_11_28", "feynman_II_13_17",
    "feynman_II_13_23", "feynman_II_13_34", "feynman_II_15_4",
    "feynman_II_15_5", "feynman_II_21_32", "feynman_II_24_17",
    "feynman_II_27_16", "feynman_II_27_18", "feynman_II_34_2",
    "feynman_II_34_2a", "feynman_II_34_11", "feynman_II_34_29a",
    "feynman_II_34_29b", "feynman_II_35_18", "feynman_II_35_21",
    "feynman_II_36_38", "feynman_II_37_1", "feynman_II_38_3",
    "feynman_II_38_14",
    # Feynman III - Quantum Mechanics (15 equations)
    "feynman_III_4_32", "feynman_III_4_33", "feynman_III_7_38",
    "feynman_III_8_54", "feynman_III_9_52", "feynman_III_10_19",
    "feynman_III_12_43", "feynman_III_13_18", "feynman_III_14_14",
    "feynman_III_15_12", "feynman_III_15_14", "feynman_III_15_27",
    "feynman_III_17_37", "feynman_III_19_51", "feynman_III_21_20",
]

STROGATZ_DATASETS = [
    "strogatz_bacres1", "strogatz_bacres2",
    "strogatz_barmag1", "strogatz_barmag2",
    "strogatz_glider1", "strogatz_glider2",
    "strogatz_lv1", "strogatz_lv2",
    "strogatz_predprey1", "strogatz_predprey2",
    "strogatz_shearflow1", "strogatz_shearflow2",
    "strogatz_vdp1", "strogatz_vdp2",
]

# Black-box datasets from SRBench (real-world regression, no known equation)
# These are the 122 regression datasets from PMLB used in SRBench
BLACKBOX_DATASETS = [
    "1027_ESL", "1028_SWD", "1029_LEV", "1030_ERA",
    "1089_USCrime", "1096_FacultySalaries", "192_vineyard",
    "195_auto_price", "197_cpu_act", "201_pol", "207_autoPrice",
    "210_cloud", "215_2dplanes", "218_house_8L", "225_puma8NH",
    "227_cpu_small", "228_elusage", "229_pwLinear", "230_machine_cpu",
    "294_satellite_image", "344_mv", "4544_GeographicalOriginalofMusic",
    "485_analcatdata_vehicle", "503_wind", "505_tecator", "519_vinnie",
    "522_pm10", "523_analcatdata_neavote", "527_analcatdata_election2000",
    "529_pollen", "537_houses", "542_pollution", "547_no2",
    "556_analcatdata_apnea2", "557_analcatdata_apnea1", "560_bodyfat",
    "561_cpu", "564_fried", "573_cpu_act", "574_house_16H",
    "579_fri_c0_250_5", "581_fri_c3_500_25", "582_fri_c1_500_25",
    "583_fri_c1_1000_50", "584_fri_c4_500_25", "586_fri_c3_1000_25",
    "588_fri_c4_1000_100", "589_fri_c2_1000_25", "590_fri_c0_1000_50",
    "591_fri_c1_1000_25", "592_fri_c4_1000_25", "593_fri_c1_1000_10",
    "594_fri_c2_100_5", "595_fri_c0_1000_10", "596_fri_c2_250_5",
    "597_fri_c2_500_5", "598_fri_c0_1000_25", "599_fri_c2_1000_5",
    "601_fri_c1_250_5", "602_fri_c3_250_10", "603_fri_c0_250_50",
    "604_fri_c4_500_10", "605_fri_c2_250_25", "606_fri_c2_1000_10",
    "607_fri_c4_1000_50", "608_fri_c3_1000_10", "609_fri_c0_500_5",
    "611_fri_c3_100_5", "612_fri_c1_1000_5", "613_fri_c3_250_5",
    "615_fri_c4_250_10", "616_fri_c4_500_50", "617_fri_c3_500_5",
    "618_fri_c3_1000_50", "620_fri_c1_500_5", "621_fri_c0_100_10",
    "622_fri_c2_1000_50", "623_fri_c4_1000_10", "624_fri_c0_100_5",
    "626_fri_c2_500_50", "627_fri_c2_500_10", "628_fri_c1_250_10",
    "631_fri_c1_500_50", "633_fri_c0_500_25", "634_fri_c2_100_10",
    "635_fri_c0_250_10", "637_fri_c1_500_10", "644_fri_c4_250_25",
    "645_fri_c3_500_50", "646_fri_c3_500_10", "647_fri_c1_250_50",
    "648_fri_c1_250_25", "649_fri_c0_500_10", "650_fri_c0_500_50",
    "651_fri_c0_100_25", "653_fri_c0_250_25", "654_fri_c3_250_25",
    "656_fri_c1_100_5", "657_fri_c2_250_10", "658_fri_c3_250_50",
    "659_sleuth_ex1714", "663_rabe_266", "665_sleuth_case2002",
    "666_rmftsa_ladata", "678_visualizing_environmental",
    "687_sleuth_ex1605", "690_visualizing_galaxy", "695_chatfield_4",
    "706_sleuth_case1202", "712_chscase_geyser1",
    "banana", "boston", "cars", "concrete",
    "diabetes", "energy", "friedman1", "friedman2", "friedman3",
]


def run_command(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"  Running: {' '.join(cmd[:5])}{'...' if len(cmd) > 5 else ''}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}")
    return result


def clone_pmlb_sparse(output_dir: Path, datasets: List[str]) -> bool:
    """Clone PMLB repository with sparse checkout for specific datasets."""

    pmlb_dir = output_dir / "pmlb_repo"

    if pmlb_dir.exists():
        print(f"  PMLB repo already exists at {pmlb_dir}")
        return True

    print("\n  Cloning PMLB repository (sparse checkout)...")

    # Initialize sparse clone
    pmlb_dir.mkdir(parents=True, exist_ok=True)

    run_command(["git", "init"], cwd=str(pmlb_dir))
    run_command(["git", "remote", "add", "origin", "https://github.com/EpistasisLab/pmlb.git"], cwd=str(pmlb_dir))
    run_command(["git", "config", "core.sparseCheckout", "true"], cwd=str(pmlb_dir))

    # Configure sparse checkout paths
    sparse_file = pmlb_dir / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)

    with open(sparse_file, 'w') as f:
        for dataset in datasets:
            f.write(f"datasets/{dataset}/\n")

    # Fetch and checkout
    print("  Fetching repository structure...")
    run_command(["git", "fetch", "--depth=1", "origin", "master"], cwd=str(pmlb_dir))
    run_command(["git", "checkout", "master"], cwd=str(pmlb_dir))

    # Pull LFS files
    print("  Pulling LFS files (this may take a while)...")
    run_command(["git", "lfs", "pull"], cwd=str(pmlb_dir))

    return True


def download_via_pmlb_python(output_dir: Path, datasets: List[str], category: str) -> Dict:
    """Download datasets using pmlb Python package as fallback."""
    import pandas as pd

    try:
        import pmlb as pmlb_lib
    except ImportError:
        print("  Installing pmlb package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pmlb", "-q"])
        import pmlb as pmlb_lib

    results = []
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        try:
            print(f"  Downloading {name}...", end=" ", flush=True)
            df = pmlb_lib.fetch_data(name)

            # Save CSV
            csv_path = cat_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)

            # Get metadata
            try:
                metadata = pmlb_lib.get_metadata(name)
            except:
                metadata = {}

            # Save metadata
            meta = {
                "name": name,
                "n_samples": len(df),
                "n_features": len(df.columns) - 1,
                "features": list(df.columns[:-1]),
                "target": df.columns[-1],
                "category": category,
                "source": "PMLB",
                "pmlb_metadata": metadata
            }

            meta_path = cat_dir / f"{name}.meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"OK ({len(df)} samples)")
            results.append(meta)

        except Exception as e:
            print(f"FAILED: {e}")

    return results


def extract_from_pmlb_repo(pmlb_dir: Path, output_dir: Path, datasets: List[str], category: str) -> List[Dict]:
    """Extract datasets from cloned PMLB repository."""
    import gzip
    import pandas as pd
    import yaml

    results = []
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        try:
            print(f"  Extracting {name}...", end=" ", flush=True)

            dataset_dir = pmlb_dir / "datasets" / name
            if not dataset_dir.exists():
                print(f"SKIPPED (not found)")
                continue

            # Find and read data file
            tsv_gz = dataset_dir / f"{name}.tsv.gz"
            if tsv_gz.exists():
                with gzip.open(tsv_gz, 'rt') as f:
                    df = pd.read_csv(f, sep='\t')
            else:
                print(f"SKIPPED (no data file)")
                continue

            # Save as CSV
            csv_path = cat_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)

            # Read metadata
            meta_yaml = dataset_dir / "metadata.yaml"
            pmlb_meta = {}
            if meta_yaml.exists():
                with open(meta_yaml) as f:
                    pmlb_meta = yaml.safe_load(f)

            # Create our metadata
            meta = {
                "name": name,
                "n_samples": len(df),
                "n_features": len(df.columns) - 1,
                "features": list(df.columns[:-1]),
                "target": df.columns[-1],
                "category": category,
                "source": "PMLB",
                "pmlb_metadata": pmlb_meta
            }

            meta_path = cat_dir / f"{name}.meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"OK ({len(df)} samples)")
            results.append(meta)

        except Exception as e:
            print(f"FAILED: {e}")

    return results


def create_unified_metadata(all_results: Dict[str, List[Dict]], output_dir: Path) -> Dict:
    """Create unified metadata with academic references."""

    metadata = {
        "version": "2.0",
        "description": "Official SRBench benchmark suite for symbolic regression",
        "references": {
            "srbench": {
                "title": "Contemporary Symbolic Regression Methods and their Relative Performance",
                "authors": "La Cava, W., Orzechowski, P., Burlacu, B., de França, F.O., Virgolin, M., Jin, Y., Kommenda, M., Moore, J.H.",
                "venue": "NeurIPS 2021 Datasets and Benchmarks Track",
                "url": "https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c0c7c76d30bd3dcaefc96f40275bdc0a-Abstract-round1.html"
            },
            "srbench_2.0": {
                "title": "Call for Action: towards the next generation of symbolic regression benchmark",
                "authors": "Aldeia, G.S.I., Zhang, H., Bomarito, G., Cranmer, M., et al.",
                "venue": "GECCO 2025 Symbolic Regression Workshop",
                "arxiv": "https://arxiv.org/abs/2505.03977"
            },
            "pmlb": {
                "title": "PMLB v1.0: an open source dataset collection for benchmarking machine learning methods",
                "authors": "Romano, J.D., Le, T.T., La Cava, W., Moore, J.H., Olson, R.S.",
                "venue": "Bioinformatics 2022",
                "url": "https://github.com/EpistasisLab/pmlb"
            },
            "ai_feynman": {
                "title": "AI Feynman: A physics-inspired method for symbolic regression",
                "authors": "Udrescu, S.M., Tegmark, M.",
                "venue": "Science Advances 2020",
                "arxiv": "https://arxiv.org/abs/1905.11481"
            },
            "strogatz": {
                "title": "Nonlinear Dynamics and Chaos",
                "authors": "Strogatz, S.H.",
                "venue": "Westview Press, 2nd Edition",
                "description": "14 ODE systems for dynamical systems discovery"
            }
        },
        "dataset_categories": {
            "feynman": {
                "description": "Physics equations from Feynman Lectures (ground-truth known)",
                "type": "ground-truth",
                "source": "AI Feynman benchmark",
                "count": 0
            },
            "strogatz": {
                "description": "Nonlinear dynamical systems (ground-truth known)",
                "type": "ground-truth",
                "source": "Strogatz textbook ODEs",
                "count": 0
            },
            "blackbox": {
                "description": "Real-world regression problems (no known equation)",
                "type": "black-box",
                "source": "PMLB regression datasets",
                "count": 0
            }
        },
        "benchmarks": {},
        "statistics": {"total": 0, "by_category": {}, "by_type": {"ground-truth": 0, "black-box": 0}}
    }

    for category, results in all_results.items():
        metadata["benchmarks"][category] = results
        count = len(results)
        metadata["dataset_categories"][category]["count"] = count
        metadata["statistics"]["by_category"][category] = count
        metadata["statistics"]["total"] += count

        if category in ["feynman", "strogatz"]:
            metadata["statistics"]["by_type"]["ground-truth"] += count
        else:
            metadata["statistics"]["by_type"]["black-box"] += count

    # Save
    meta_path = output_dir / "benchmarks_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nUnified metadata saved to: {meta_path}")
    return metadata


def main():
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("Official SRBench Benchmark Downloader")
    print("=" * 70)
    print("\nThis downloads the EXACT datasets used in:")
    print("  - SRBench (NeurIPS 2021)")
    print("  - SRBench 2.0 (GECCO 2025)")
    print(f"\nOutput directory: {script_dir}")
    print(f"\nDatasets to download:")
    print(f"  - Feynman (ground-truth): {len(FEYNMAN_DATASETS)} problems")
    print(f"  - Strogatz (ground-truth): {len(STROGATZ_DATASETS)} problems")
    print(f"  - Black-box (real-world): {len(BLACKBOX_DATASETS)} problems")
    print(f"  - Total: {len(FEYNMAN_DATASETS) + len(STROGATZ_DATASETS) + len(BLACKBOX_DATASETS)} problems")

    all_results = {}

    # Download Feynman
    print(f"\n{'=' * 50}")
    print(f"Downloading FEYNMAN datasets ({len(FEYNMAN_DATASETS)})")
    print("=" * 50)
    feynman_results = download_via_pmlb_python(script_dir, FEYNMAN_DATASETS, "feynman")
    all_results["feynman"] = feynman_results
    print(f"\nFeynman: {len(feynman_results)}/{len(FEYNMAN_DATASETS)} downloaded")

    # Download Strogatz
    print(f"\n{'=' * 50}")
    print(f"Downloading STROGATZ datasets ({len(STROGATZ_DATASETS)})")
    print("=" * 50)
    strogatz_results = download_via_pmlb_python(script_dir, STROGATZ_DATASETS, "strogatz")
    all_results["strogatz"] = strogatz_results
    print(f"\nStrogatz: {len(strogatz_results)}/{len(STROGATZ_DATASETS)} downloaded")

    # Download Black-box
    print(f"\n{'=' * 50}")
    print(f"Downloading BLACK-BOX datasets ({len(BLACKBOX_DATASETS)})")
    print("=" * 50)
    blackbox_results = download_via_pmlb_python(script_dir, BLACKBOX_DATASETS, "blackbox")
    all_results["blackbox"] = blackbox_results
    print(f"\nBlack-box: {len(blackbox_results)}/{len(BLACKBOX_DATASETS)} downloaded")

    # Create unified metadata
    print("\n" + "=" * 50)
    print("Creating unified metadata with academic references...")
    print("=" * 50)
    metadata = create_unified_metadata(all_results, script_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"\nTotal benchmarks: {metadata['statistics']['total']}")
    print("\nBy category:")
    for cat, count in metadata['statistics']['by_category'].items():
        print(f"  - {cat}: {count}")
    print("\nBy type:")
    for t, count in metadata['statistics']['by_type'].items():
        print(f"  - {t}: {count}")

    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)
    print("\nFor academic citation, see benchmarks_metadata.json")


if __name__ == "__main__":
    main()
