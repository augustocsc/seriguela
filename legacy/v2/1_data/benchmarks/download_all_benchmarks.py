#!/usr/bin/env python3
"""
Download all symbolic regression benchmarks using pmlb library.

Benchmarks included:
- Feynman (100+ physics equations from Feynman Lectures)
- Strogatz (14 nonlinear dynamical systems)
- SRBench datasets (additional regression problems)

Usage:
    python download_all_benchmarks.py

Output structure:
    1_data/benchmarks/
    ├── nguyen/          # Already exists (Nguyen 1-12)
    ├── feynman/         # Feynman physics equations
    ├── strogatz/        # Dynamical systems
    └── srbench/         # Additional SRBench datasets
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import pmlb
import pandas as pd

# Known Feynman equations with their formulas
FEYNMAN_EQUATIONS = {
    # Feynman I (Mechanics)
    "feynman_I_6_2": "exp(-theta**2/(2*sigma**2))/(sqrt(2*pi)*sigma)",
    "feynman_I_6_2a": "exp(-theta**2/(2*sigma**2))",
    "feynman_I_6_2b": "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
    "feynman_I_8_14": "sqrt((x2-x1)**2 + (y2-y1)**2)",
    "feynman_I_9_18": "G*m1*m2/((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)",
    "feynman_I_10_7": "m_0/sqrt(1-v**2/c**2)",
    "feynman_I_11_19": "x1*y1 + x2*y2 + x3*y3",
    "feynman_I_12_1": "mu*Nn",
    "feynman_I_12_2": "q1*q2*r/(4*pi*epsilon*r**3)",
    "feynman_I_12_4": "q1*r/(4*pi*epsilon*r**3)",
    "feynman_I_12_5": "q2*Ef",
    "feynman_I_12_11": "q*(Ef + B*v*sin(theta))",
    "feynman_I_13_4": "0.5*m*(v**2 + u**2 + w**2)",
    "feynman_I_13_12": "G*m1*m2*(1/r2 - 1/r1)",
    "feynman_I_14_3": "m*g*z",
    "feynman_I_14_4": "0.5*k_spring*x**2",
    "feynman_I_15_3t": "(t-u*x/c**2)/sqrt(1-u**2/c**2)",
    "feynman_I_15_3x": "(x-u*t)/sqrt(1-u**2/c**2)",
    "feynman_I_15_10": "m_0*v/sqrt(1-v**2/c**2)",
    "feynman_I_16_6": "(u+v)/(1+u*v/c**2)",
    "feynman_I_18_4": "(m1*r1 + m2*r2)/(m1+m2)",
    "feynman_I_18_12": "r*F*sin(theta)",
    "feynman_I_18_14": "m*r*v*sin(theta)",
    "feynman_I_24_6": "0.25*m*(omega**2 + omega_0**2)*x**2",
    "feynman_I_25_13": "q/C",
    "feynman_I_26_2": "arcsin(n*sin(theta2))",
    "feynman_I_27_6": "1/(1/d1 + n/d2)",
    "feynman_I_29_4": "omega/c",
    "feynman_I_29_16": "sqrt(x1**2 + x2**2 - 2*x1*x2*cos(theta1-theta2))",
    "feynman_I_30_3": "Int_0*sin(n*theta/2)**2/sin(theta/2)**2",
    "feynman_I_30_5": "arcsin(lambd/(n*d))",
    "feynman_I_32_5": "q**2*a**2/(6*pi*epsilon*c**3)",
    "feynman_I_32_17": "(0.5*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
    "feynman_I_34_8": "q*v*B/p",
    "feynman_I_34_10": "omega_0/(1-v/c)",
    "feynman_I_34_14": "(1+v/c)/sqrt(1-v**2/c**2)*omega_0",
    "feynman_I_34_27": "(h/(2*pi))*omega",
    "feynman_I_37_4": "I1 + I2 + 2*sqrt(I1*I2)*cos(delta)",
    "feynman_I_38_12": "4*pi*epsilon*(h/(2*pi))**2/(m*q**2)",
    "feynman_I_39_1": "3/2*pr*V",
    "feynman_I_39_11": "1/(gamma-1)*pr*V",
    "feynman_I_39_22": "n*kb*T/V",
    "feynman_I_40_1": "n_0*exp(-m*g*x/(kb*T))",
    "feynman_I_41_16": "(h/(2*pi))*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))",
    "feynman_I_43_16": "mu_drift*q*Ef*u/(kb*T)",
    "feynman_I_43_31": "mu_drift*kb*T",
    "feynman_I_43_43": "kb*v/(2*pi*d**2)",
    "feynman_I_44_4": "n*kb*T*ln(V2/V1)",
    "feynman_I_47_23": "sqrt(gamma*pr/rho)",
    "feynman_I_48_2": "m*c**2/sqrt(1-v**2/c**2)",
    "feynman_I_50_26": "x1*(cos(omega*t) + alpha*cos(omega*t)**2)",

    # Feynman II (Electromagnetism)
    "feynman_II_2_42": "kappa*(T2-T1)*A/d",
    "feynman_II_3_24": "Pwr/(4*pi*r**2)",
    "feynman_II_4_23": "q/(4*pi*epsilon*r)",
    "feynman_II_6_11": "1/(4*pi*epsilon)*p_d*cos(theta)/r**2",
    "feynman_II_6_15a": "p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)",
    "feynman_II_6_15b": "p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
    "feynman_II_8_7": "3/5*q**2/(4*pi*epsilon*d)",
    "feynman_II_8_31": "epsilon*Ef**2/2",
    "feynman_II_10_9": "sigma_den/epsilon*1/(1+chi)",
    "feynman_II_11_3": "q*Ef/(m*(omega_0**2-omega**2))",
    "feynman_II_11_17": "n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
    "feynman_II_11_20": "n_rho*p_d**2*Ef/(3*kb*T)",
    "feynman_II_11_27": "n*alpha/(1-(n*alpha/3))*epsilon*Ef",
    "feynman_II_11_28": "1+n*alpha/(1-(n*alpha/3))",
    "feynman_II_13_17": "1/(4*pi*epsilon*c**2)*2*I/r",
    "feynman_II_13_23": "rho_c_0/sqrt(1-v**2/c**2)",
    "feynman_II_13_34": "rho_c_0*v/sqrt(1-v**2/c**2)",
    "feynman_II_15_4": "-mom*B*cos(theta)",
    "feynman_II_15_5": "-p_d*Ef*cos(theta)",
    "feynman_II_21_32": "q/(4*pi*epsilon*r*(1-v/c))",
    "feynman_II_24_17": "sqrt(omega**2/c**2-pi**2/d**2)",
    "feynman_II_27_16": "epsilon*c*Ef**2",
    "feynman_II_27_18": "epsilon*Ef**2",
    "feynman_II_34_2": "q*v/(2*pi*r)",
    "feynman_II_34_2a": "q*v*r/2",
    "feynman_II_34_11": "g_*q*B/(2*m)",
    "feynman_II_34_29a": "q*h/(4*pi*m)",
    "feynman_II_34_29b": "g_*mom*B*Jz/(h/(2*pi))",
    "feynman_II_35_18": "n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
    "feynman_II_35_21": "n_rho*mom*tanh(mom*B/(kb*T))",
    "feynman_II_36_38": "mom*H/(kb*T) + mom*alpha*M/(kb*T)",
    "feynman_II_37_1": "mom*(1+chi)*B",
    "feynman_II_38_3": "Y*A*x/d",
    "feynman_II_38_14": "Y/(2*(1+sigma))",

    # Feynman III (Quantum)
    "feynman_III_4_32": "1/(exp((h/(2*pi))*omega/(kb*T))-1)",
    "feynman_III_4_33": "(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
    "feynman_III_7_38": "2*mom*sqrt(Bx**2+By**2+Bz**2)",
    "feynman_III_8_54": "sin(E_n*t/(h/(2*pi)))**2",
    "feynman_III_9_52": "(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2",
    "feynman_III_10_19": "mom*sqrt(Bx**2+By**2+Bz**2)",
    "feynman_III_12_43": "n*(h/(2*pi))",
    "feynman_III_13_18": "2*E_n*d**2*k/(h/(2*pi))",
    "feynman_III_14_14": "I_0*(exp(q*Volt/(kb*T))-1)",
    "feynman_III_15_12": "2*U*(1-cos(k*d))",
    "feynman_III_15_14": "(h/(2*pi))**2/(2*E_n*d**2)",
    "feynman_III_15_27": "2*pi*alpha/(n*d)",
    "feynman_III_17_37": "beta*(1+alpha*cos(theta))",
    "feynman_III_19_51": "-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
    "feynman_III_21_20": "-rho_c_0*q*A_vec/m",
}

# Strogatz equations
STROGATZ_EQUATIONS = {
    "strogatz_bacres1": "V*x/(K + x)",  # Bacterial respiration
    "strogatz_bacres2": "-V*x/(K + x)",
    "strogatz_barmag1": "x - x**3/3 - y",  # Bar magnet
    "strogatz_barmag2": "x/tau",
    "strogatz_glider1": "-sin(theta)",  # Glider
    "strogatz_glider2": "v - v**2 - sin(theta)/v",
    "strogatz_lv1": "a*x - b*x*y",  # Lotka-Volterra
    "strogatz_lv2": "c*x*y - d*y",
    "strogatz_predprey1": "r*x*(1-x/K) - a*x*y/(1+a*h*x)",  # Predator-prey
    "strogatz_predprey2": "e*a*x*y/(1+a*h*x) - d*y",
    "strogatz_shearflow1": "cos(y)",  # Shear flow
    "strogatz_shearflow2": "sin(x)*sin(y)",
    "strogatz_vdp1": "10*(y - (x**3/3 - x))",  # Van der Pol
    "strogatz_vdp2": "-x/10",
}

def get_feynman_datasets() -> List[str]:
    """Get list of Feynman datasets from pmlb."""
    all_datasets = pmlb.dataset_names
    return [d for d in all_datasets if d.startswith('feynman')]

def get_strogatz_datasets() -> List[str]:
    """Get list of Strogatz datasets from pmlb."""
    all_datasets = pmlb.dataset_names
    return [d for d in all_datasets if d.startswith('strogatz')]

def classify_difficulty(name: str, n_vars: int) -> str:
    """Classify problem difficulty."""
    if name.startswith("strogatz"):
        return "hard"
    if "III" in name:
        return "very_hard"
    if "II" in name:
        return "hard"
    if "test" in name:
        return "hard"
    if n_vars >= 6:
        return "very_hard"
    elif n_vars >= 4:
        return "hard"
    elif n_vars >= 2:
        return "medium"
    return "easy"

def download_dataset(name: str, output_dir: Path, equations: Dict[str, str]) -> Optional[Dict]:
    """Download a dataset using pmlb."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Downloading {name}...", end=" ", flush=True)

        # Fetch dataset using pmlb
        df = pmlb.fetch_data(name)

        # Save as CSV
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        # Get target column (last column)
        target_col = df.columns[-1]
        feature_cols = [c for c in df.columns if c != target_col]
        n_vars = len(feature_cols)

        # Get equation if known
        equation = equations.get(name, "unknown")

        # Create metadata
        metadata = {
            "name": name,
            "equation": equation,
            "n_vars": n_vars,
            "n_samples": len(df),
            "features": feature_cols,
            "target": target_col,
            "difficulty": classify_difficulty(name, n_vars),
        }

        # Save metadata
        meta_path = output_dir / f"{name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"OK ({len(df)} samples, {n_vars} vars)")
        return metadata

    except Exception as e:
        print(f"FAILED: {e}")
        return None

def create_unified_metadata(all_results: Dict[str, List[Dict]], output_dir: Path):
    """Create unified metadata file."""
    metadata = {
        "version": "1.0",
        "description": "Unified benchmark suite for symbolic regression",
        "sources": {
            "feynman": "AI Feynman - Feynman Lectures on Physics (100+ equations)",
            "strogatz": "Nonlinear Dynamics and Chaos by Steven Strogatz (14 ODE systems)",
            "nguyen": "Nguyen Symbolic Regression Benchmarks (12 equations)"
        },
        "benchmarks": {},
        "statistics": {"total": 0, "by_category": {}, "by_difficulty": {}}
    }

    for category, results in all_results.items():
        valid_results = [r for r in results if r is not None]
        metadata["benchmarks"][category] = valid_results
        metadata["statistics"]["by_category"][category] = len(valid_results)
        metadata["statistics"]["total"] += len(valid_results)

        for r in valid_results:
            diff = r.get("difficulty", "unknown")
            metadata["statistics"]["by_difficulty"][diff] = \
                metadata["statistics"]["by_difficulty"].get(diff, 0) + 1

    # Save unified metadata
    meta_path = output_dir / "benchmarks_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nUnified metadata saved to: {meta_path}")
    return metadata

def main():
    script_dir = Path(__file__).parent

    print("=" * 60)
    print("Symbolic Regression Benchmark Downloader (using pmlb)")
    print("=" * 60)
    print(f"\nOutput directory: {script_dir}")

    # Get datasets
    feynman_datasets = get_feynman_datasets()
    strogatz_datasets = get_strogatz_datasets()

    print(f"\nFound {len(feynman_datasets)} Feynman datasets")
    print(f"Found {len(strogatz_datasets)} Strogatz datasets")

    all_results = {}

    # Download Feynman
    print(f"\n{'=' * 40}")
    print(f"Downloading FEYNMAN ({len(feynman_datasets)} datasets)")
    print("=" * 40)

    feynman_dir = script_dir / "feynman"
    feynman_results = []
    for name in sorted(feynman_datasets):
        result = download_dataset(name, feynman_dir, FEYNMAN_EQUATIONS)
        if result:
            feynman_results.append(result)

    all_results["feynman"] = feynman_results
    print(f"\nFeynman: {len(feynman_results)}/{len(feynman_datasets)} downloaded")

    # Download Strogatz
    print(f"\n{'=' * 40}")
    print(f"Downloading STROGATZ ({len(strogatz_datasets)} datasets)")
    print("=" * 40)

    strogatz_dir = script_dir / "strogatz"
    strogatz_results = []
    for name in sorted(strogatz_datasets):
        result = download_dataset(name, strogatz_dir, STROGATZ_EQUATIONS)
        if result:
            strogatz_results.append(result)

    all_results["strogatz"] = strogatz_results
    print(f"\nStrogatz: {len(strogatz_results)}/{len(strogatz_datasets)} downloaded")

    # Create unified metadata
    print("\n" + "=" * 40)
    print("Creating unified metadata...")
    print("=" * 40)

    metadata = create_unified_metadata(all_results, script_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"\nTotal benchmarks: {metadata['statistics']['total']}")
    print("\nBy category:")
    for cat, count in metadata['statistics']['by_category'].items():
        print(f"  - {cat}: {count}")
    print("\nBy difficulty:")
    for diff, count in sorted(metadata['statistics']['by_difficulty'].items()):
        print(f"  - {diff}: {count}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
