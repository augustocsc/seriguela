#!/usr/bin/env python3
"""
Run Nguyen benchmark subset with multiple algorithms.
Supports supervised generation (no RL) for faster evaluation.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_supervised_evaluation(model_path, dataset_path, output_file, num_samples=200):
    """Run supervised evaluation (generation without RL)"""
    cmd = [
        sys.executable,
        "scripts/evaluate_quality_simple.py",
        "--model_path", model_path,
        "--num_samples", str(num_samples),
        "--output_dir", os.path.dirname(output_file)
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_bench", type=int, required=True, help="Start benchmark number (1-12)")
    parser.add_argument("--end_bench", type=int, required=True, help="End benchmark number (1-12)")
    parser.add_argument("--models", nargs="+", default=["base", "medium", "large"])
    parser.add_argument("--output_dir", default="./results/nguyen")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks = range(args.start_bench, args.end_bench + 1)
    total = len(args.models) * len(benchmarks)
    completed = 0
    failed = 0

    logger.info(f"Running {total} experiments: {len(args.models)} models × {len(benchmarks)} benchmarks")

    for model_name in args.models:
        model_path = f"./output/gpt2_{model_name}_700K_json"

        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            failed += len(benchmarks)
            continue

        for bench in benchmarks:
            output_file = f"{args.output_dir}/{model_name}_nguyen{bench}_supervised.json"

            logger.info(f"[{completed+1}/{total}] {model_name} + Nguyen-{bench}")

            start_time = time.time()

            success = run_supervised_evaluation(
                model_path,
                f"./data/benchmarks/nguyen/nguyen_{bench}.csv",
                output_file,
                args.num_samples
            )

            duration = time.time() - start_time

            if success:
                completed += 1
                logger.info(f"✓ Completed in {duration:.1f}s")
            else:
                failed += 1
                logger.error(f"✗ Failed after {duration:.1f}s")

    logger.info(f"Done! Completed: {completed}/{total}, Failed: {failed}")

    summary = {
        "total_experiments": total,
        "completed": completed,
        "failed": failed,
        "benchmarks": list(benchmarks),
        "models": args.models
    }

    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
