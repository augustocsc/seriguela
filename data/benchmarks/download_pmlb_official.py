#!/usr/bin/env python3
"""
Download official PMLB datasets (Feynman, Strogatz) from the original repository.

This script downloads the EXACT data files used in:
- AI Feynman (Science Advances 2020)
- SRBench (NeurIPS 2021)
- SRBench 2.0 (GECCO 2025)

The data is downloaded from the official PMLB repository via media.githubusercontent.com
which serves Git LFS files directly.

Usage:
    python download_pmlb_official.py
"""

import os
import sys
import json
import gzip
import requests
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Official PMLB URL (Git LFS files served via media.githubusercontent.com)
PMLB_DATA_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets"
PMLB_META_URL = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets"

# Complete list of Feynman datasets from PMLB
FEYNMAN_DATASETS = [
    # Volume I - Mechanics
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
    # Volume II - Electromagnetism
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
    # Volume III - Quantum Mechanics
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


def download_dataset(name: str, output_dir: Path, timeout: int = 60) -> Optional[Dict]:
    """Download a single dataset from PMLB."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download data (gzipped TSV)
        data_url = f"{PMLB_DATA_URL}/{name}/{name}.tsv.gz"
        response = requests.get(data_url, timeout=timeout)
        response.raise_for_status()

        # Decompress and parse
        import io
        data = gzip.decompress(response.content).decode('utf-8')
        df = pd.read_csv(io.StringIO(data), sep='\t')

        # Save as CSV
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        # Download metadata
        meta_url = f"{PMLB_META_URL}/{name}/metadata.yaml"
        pmlb_meta = {}
        try:
            meta_response = requests.get(meta_url, timeout=10)
            if meta_response.status_code == 200:
                pmlb_meta = yaml.safe_load(meta_response.text)
        except:
            pass

        # Create our metadata
        target_col = df.columns[-1]
        feature_cols = list(df.columns[:-1])

        meta = {
            "name": name,
            "n_samples": len(df),
            "n_vars": len(feature_cols),
            "variables": feature_cols,
            "target": target_col,
            "source": "PMLB (official)",
            "url": data_url,
            "pmlb_metadata": pmlb_meta
        }

        # Save metadata
        meta_path = output_dir / f"{name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        return {"name": name, "error": str(e)}


def download_category(datasets: List[str], output_dir: Path, category: str, max_workers: int = 5) -> List[Dict]:
    """Download a category of datasets in parallel."""
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_dataset, name, output_dir): name for name in datasets}

        for future in as_completed(futures):
            name = futures[future]
            result = future.result()

            if result and "error" not in result:
                print(f"  [OK] {name} ({result['n_samples']} samples)")
                results.append(result)
            else:
                error = result.get("error", "Unknown error") if result else "Unknown error"
                print(f"  [FAIL] {name}: {error}")
                failed.append(name)

    return results, failed


def create_unified_metadata(all_results: Dict[str, List[Dict]], output_dir: Path) -> Dict:
    """Create unified metadata file with academic references."""

    metadata = {
        "version": "2.0",
        "description": "Official PMLB symbolic regression benchmarks",
        "data_source": {
            "repository": "https://github.com/EpistasisLab/pmlb",
            "download_url": PMLB_DATA_URL,
            "accessed": "Downloaded from official PMLB repository"
        },
        "references": {
            "pmlb": {
                "title": "PMLB v1.0: an open source dataset collection for benchmarking machine learning methods",
                "authors": "Romano, J.D., Le, T.T., La Cava, W., Moore, J.H., Olson, R.S.",
                "venue": "Bioinformatics, 2022",
                "doi": "10.1093/bioinformatics/btab727",
                "url": "https://github.com/EpistasisLab/pmlb"
            },
            "feynman": {
                "title": "AI Feynman: A physics-inspired method for symbolic regression",
                "authors": "Udrescu, S.M., Tegmark, M.",
                "venue": "Science Advances, 6(16), eaay2631, 2020",
                "arxiv": "https://arxiv.org/abs/1905.11481"
            },
            "strogatz": {
                "title": "Nonlinear Dynamics and Chaos",
                "authors": "Strogatz, S.H.",
                "venue": "Westview Press, 2nd Edition, 2015"
            },
            "srbench": {
                "title": "Contemporary Symbolic Regression Methods and their Relative Performance",
                "authors": "La Cava, W., et al.",
                "venue": "NeurIPS 2021 Datasets and Benchmarks Track",
                "url": "https://github.com/cavalab/srbench"
            },
            "srbench_2.0": {
                "title": "Call for Action: towards the next generation of symbolic regression benchmark",
                "authors": "Aldeia, G.S.I., Zhang, H., Bomarito, G., Cranmer, M., et al.",
                "venue": "GECCO 2025 Symbolic Regression Workshop",
                "arxiv": "https://arxiv.org/abs/2505.03977"
            }
        },
        "categories": {},
        "benchmarks": {},
        "statistics": {"total": 0, "ground_truth": 0}
    }

    for category, results in all_results.items():
        metadata["benchmarks"][category] = results
        count = len(results)
        metadata["categories"][category] = {
            "type": "ground-truth",
            "count": count,
            "description": f"Official {category} datasets from PMLB"
        }
        metadata["statistics"]["total"] += count
        metadata["statistics"]["ground_truth"] += count

    meta_path = output_dir / "benchmarks_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("Official PMLB Benchmark Downloader")
    print("=" * 70)
    print(f"\nData source: {PMLB_DATA_URL}")
    print(f"Output: {script_dir}")
    print(f"\nDatasets to download:")
    print(f"  - Feynman: {len(FEYNMAN_DATASETS)} (physics equations)")
    print(f"  - Strogatz: {len(STROGATZ_DATASETS)} (dynamical systems)")

    all_results = {}

    # Download Feynman
    print(f"\n{'=' * 50}")
    print(f"FEYNMAN ({len(FEYNMAN_DATASETS)} datasets)")
    print("=" * 50)
    feynman_dir = script_dir / "feynman"
    feynman_results, feynman_failed = download_category(FEYNMAN_DATASETS, feynman_dir, "feynman")
    all_results["feynman"] = feynman_results
    print(f"\nFeynman: {len(feynman_results)}/{len(FEYNMAN_DATASETS)} downloaded")
    if feynman_failed:
        print(f"  Failed: {feynman_failed}")

    # Download Strogatz
    print(f"\n{'=' * 50}")
    print(f"STROGATZ ({len(STROGATZ_DATASETS)} datasets)")
    print("=" * 50)
    strogatz_dir = script_dir / "strogatz"
    strogatz_results, strogatz_failed = download_category(STROGATZ_DATASETS, strogatz_dir, "strogatz")
    all_results["strogatz"] = strogatz_results
    print(f"\nStrogatz: {len(strogatz_results)}/{len(STROGATZ_DATASETS)} downloaded")
    if strogatz_failed:
        print(f"  Failed: {strogatz_failed}")

    # Create metadata
    print(f"\n{'=' * 50}")
    print("Creating unified metadata...")
    print("=" * 50)
    metadata = create_unified_metadata(all_results, script_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"\nTotal downloaded: {metadata['statistics']['total']}")
    print(f"  - Feynman: {metadata['categories']['feynman']['count']}")
    print(f"  - Strogatz: {metadata['categories']['strogatz']['count']}")

    total_expected = len(FEYNMAN_DATASETS) + len(STROGATZ_DATASETS)
    if metadata['statistics']['total'] == total_expected:
        print("\n[OK] All datasets downloaded successfully!")
    else:
        print(f"\n[WARNING] Some datasets failed ({total_expected - metadata['statistics']['total']} missing)")

    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
