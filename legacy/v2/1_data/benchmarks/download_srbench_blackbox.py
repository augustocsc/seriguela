#!/usr/bin/env python3
"""
Download SRBench black-box datasets from PMLB.

These are real-world regression datasets without known ground-truth equations,
used in SRBench (NeurIPS 2021) for evaluating symbolic regression methods.

Usage:
    python download_srbench_blackbox.py
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

# Official PMLB URL
PMLB_DATA_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets"
PMLB_META_URL = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets"

# Black-box regression datasets from SRBench (excluding Feynman, Strogatz, test sets)
BLACKBOX_DATASETS = [
    # Standard regression benchmarks
    "1027_ESL", "1028_SWD", "1029_LEV", "1030_ERA",
    "1089_USCrime", "1096_FacultySalaries",
    # BNG datasets
    "1191_BNG_pbc", "1193_BNG_lowbwt", "1196_BNG_pharynx",
    "1199_BNG_echoMonths", "1201_BNG_breastTumor", "1203_BNG_pwLinear",
    # Other benchmarks
    "192_vineyard", "197_cpu_act", "201_pol", "210_cloud",
    "215_2dplanes", "218_house_8L", "225_puma8NH", "227_cpu_small",
    "228_elusage", "229_pwLinear", "230_machine_cpu",
    "294_satellite_image", "344_mv",
    "4544_GeographicalOriginalofMusic",
    "485_analcatdata_vehicle", "503_wind", "505_tecator",
    "519_vinnie", "522_pm10", "523_analcatdata_neavote",
    "527_analcatdata_election2000", "529_pollen", "537_houses",
    "542_pollution", "547_no2", "556_analcatdata_apnea2",
    "557_analcatdata_apnea1", "560_bodyfat", "561_cpu",
    "562_cpu_small", "564_fried", "573_cpu_act", "574_house_16H",
    # Friedman datasets (fri_*)
    "579_fri_c0_250_5", "581_fri_c3_500_25", "582_fri_c1_500_25",
    "583_fri_c1_1000_50", "584_fri_c4_500_25", "586_fri_c3_1000_25",
    "588_fri_c4_1000_100", "589_fri_c2_1000_25", "590_fri_c0_1000_50",
    "591_fri_c1_100_10", "592_fri_c4_1000_25", "593_fri_c1_1000_10",
    "594_fri_c2_100_5", "595_fri_c0_1000_10", "596_fri_c2_250_5",
    "597_fri_c2_500_5", "598_fri_c0_1000_25", "599_fri_c2_1000_5",
    "601_fri_c1_250_5", "602_fri_c3_250_10", "603_fri_c0_250_50",
    "604_fri_c4_500_10", "605_fri_c2_250_25", "606_fri_c2_1000_10",
    "607_fri_c4_1000_50", "608_fri_c3_1000_10", "609_fri_c0_1000_5",
    "611_fri_c3_100_5", "612_fri_c1_1000_5", "613_fri_c3_250_5",
    "615_fri_c4_250_10", "616_fri_c4_500_50", "617_fri_c3_500_5",
    "618_fri_c3_1000_50", "620_fri_c1_1000_25", "621_fri_c0_100_10",
    "622_fri_c2_1000_50", "623_fri_c4_1000_10", "624_fri_c0_100_5",
    "626_fri_c2_500_50", "627_fri_c2_500_10", "628_fri_c3_1000_5",
    "631_fri_c1_500_5", "633_fri_c0_500_25", "634_fri_c2_100_10",
    "635_fri_c0_250_10", "637_fri_c1_500_50", "641_fri_c1_500_10",
    "643_fri_c2_500_25", "644_fri_c4_250_25", "645_fri_c3_500_50",
    "646_fri_c3_500_10", "647_fri_c1_250_10", "648_fri_c1_250_50",
    "649_fri_c0_500_5", "650_fri_c0_500_50", "651_fri_c0_100_25",
    "653_fri_c0_250_25", "654_fri_c0_500_10", "656_fri_c1_100_5",
    "657_fri_c2_250_10", "658_fri_c3_250_25",
    # Sleuth datasets
    "659_sleuth_ex1714", "663_rabe_266", "665_sleuth_case2002",
    "666_rmftsa_ladata", "678_visualizing_environmental",
    "687_sleuth_ex1605", "690_visualizing_galaxy",
    "695_chatfield_4", "706_sleuth_case1202", "712_chscase_geyser1",
    # First principles (ground-truth but different from Feynman)
    "first_principles_absorption", "first_principles_bode",
    "first_principles_hubble", "first_principles_ideal_gas",
    "first_principles_kepler", "first_principles_leavitt",
    "first_principles_newton", "first_principles_planck",
    "first_principles_rydberg", "first_principles_schechter",
    "first_principles_supernovae_zg", "first_principles_supernovae_zr",
    "first_principles_tully_fisher",
    # Nikuradse (fluid dynamics)
    "nikuradse_1", "nikuradse_2",
    # Other
    "solar_flare",
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


def download_category(datasets: List[str], output_dir: Path, max_workers: int = 5) -> tuple:
    """Download datasets in parallel."""
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_dataset, name, output_dir): name for name in datasets}

        for future in as_completed(futures):
            name = futures[future]
            result = future.result()

            if result and "error" not in result:
                print(f"  [OK] {name} ({result['n_samples']} samples, {result['n_vars']} vars)")
                results.append(result)
            else:
                error = result.get("error", "Unknown error") if result else "Unknown error"
                print(f"  [FAIL] {name}: {error}")
                failed.append(name)

    return results, failed


def update_unified_metadata(blackbox_results: List[Dict], output_dir: Path) -> Dict:
    """Update the unified metadata file with black-box datasets."""

    meta_path = output_dir / "benchmarks_metadata.json"

    # Load existing metadata
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "version": "2.0",
            "description": "Official PMLB symbolic regression benchmarks",
            "data_source": {
                "repository": "https://github.com/EpistasisLab/pmlb",
                "download_url": PMLB_DATA_URL,
                "accessed": "Downloaded from official PMLB repository"
            },
            "references": {},
            "categories": {},
            "benchmarks": {},
            "statistics": {"total": 0, "ground_truth": 0, "black_box": 0}
        }

    # Add black-box category
    metadata["benchmarks"]["blackbox"] = blackbox_results
    metadata["categories"]["blackbox"] = {
        "type": "black-box",
        "count": len(blackbox_results),
        "description": "Real-world regression datasets without known ground-truth equations (SRBench)"
    }

    # Update statistics
    metadata["statistics"]["black_box"] = len(blackbox_results)
    metadata["statistics"]["total"] = (
        metadata["categories"].get("feynman", {}).get("count", 0) +
        metadata["categories"].get("strogatz", {}).get("count", 0) +
        len(blackbox_results)
    )

    # Save updated metadata
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("SRBench Black-box Dataset Downloader")
    print("=" * 70)
    print(f"\nData source: {PMLB_DATA_URL}")
    print(f"Output: {script_dir / 'srbench'}")
    print(f"\nDatasets to download: {len(BLACKBOX_DATASETS)} black-box regression datasets")

    # Download black-box datasets
    print(f"\n{'=' * 50}")
    print(f"BLACK-BOX DATASETS ({len(BLACKBOX_DATASETS)} datasets)")
    print("=" * 50)

    srbench_dir = script_dir / "srbench"
    results, failed = download_category(BLACKBOX_DATASETS, srbench_dir)

    print(f"\nBlack-box: {len(results)}/{len(BLACKBOX_DATASETS)} downloaded")
    if failed:
        print(f"  Failed: {failed}")

    # Update unified metadata
    print(f"\n{'=' * 50}")
    print("Updating unified metadata...")
    print("=" * 50)
    metadata = update_unified_metadata(results, script_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"\nBlack-box downloaded: {len(results)}")
    print(f"\nTotal benchmarks now available:")
    for cat, info in metadata["categories"].items():
        print(f"  - {cat.capitalize()}: {info['count']} ({info['type']})")
    print(f"\nGrand total: {metadata['statistics']['total']} datasets")

    if len(results) == len(BLACKBOX_DATASETS):
        print("\n[OK] All black-box datasets downloaded successfully!")
    else:
        print(f"\n[WARNING] Some datasets failed ({len(BLACKBOX_DATASETS) - len(results)} missing)")

    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
