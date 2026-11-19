#!/usr/bin/env python3
"""
Run complete Nguyen benchmark suite on all models.
Executes 3 models × 12 benchmarks = 36 experiments.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["base", "medium", "large"],
                        help="Model names to evaluate")
    parser.add_argument("--benchmarks", nargs="+", type=int,
                        default=list(range(1, 13)),
                        help="Benchmark numbers (1-12)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of candidate expressions per benchmark")
    parser.add_argument("--output_dir", default="./results_nguyen_benchmarks",
                        help="Output directory for results")
    parser.add_argument("--models_dir", default="./output",
                        help="Directory containing trained models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_experiments = len(args.models) * len(args.benchmarks)
    completed = 0
    failed = 0
    results_summary = []

    logger.info("="*60)
    logger.info("NGUYEN BENCHMARK SUITE EVALUATION")
    logger.info("="*60)
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Benchmarks: Nguyen {min(args.benchmarks)} - {max(args.benchmarks)}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Samples per experiment: {args.num_samples}")
    logger.info("="*60)

    start_time_total = time.time()

    for model_name in args.models:
        model_path = os.path.join(args.models_dir, f"gpt2_{model_name}_700K_json")

        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            failed += len(args.benchmarks)
            continue

        for bench_num in args.benchmarks:
            benchmark_csv = f"./data/benchmarks/nguyen/nguyen_{bench_num}.csv"

            if not os.path.exists(benchmark_csv):
                logger.warning(f"Benchmark not found: {benchmark_csv}")
                failed += 1
                continue

            output_file = os.path.join(args.output_dir, f"{model_name}_nguyen{bench_num}.json")

            logger.info(f"\n[{completed+1}/{total_experiments}] {model_name.upper()} + Nguyen-{bench_num}")
            logger.info("-" * 60)

            start_time = time.time()

            cmd = [
                sys.executable,
                "scripts/evaluate_nguyen_benchmarks.py",
                "--model_path", model_path,
                "--benchmark_csv", benchmark_csv,
                "--num_samples", str(args.num_samples),
                "--output_file", output_file
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout per experiment
                )

                duration = time.time() - start_time

                if result.returncode == 0:
                    completed += 1

                    # Load results to get summary
                    try:
                        with open(output_file) as f:
                            data = json.load(f)
                            summary = data.get("summary", {})

                            logger.info(f"✓ Completed in {duration:.1f}s")
                            logger.info(f"  Valid: {summary.get('valid_count', 0)}/{summary.get('num_samples', 0)} ({summary.get('valid_rate', 0)*100:.1f}%)")
                            logger.info(f"  Best R²: {summary.get('best_r2', 'N/A')}")

                            results_summary.append({
                                "model": model_name,
                                "benchmark": f"nguyen_{bench_num}",
                                "valid_rate": summary.get('valid_rate', 0),
                                "best_r2": summary.get('best_r2'),
                                "mean_r2": summary.get('mean_r2'),
                                "num_with_r2": summary.get('num_with_r2', 0),
                                "duration": duration
                            })
                    except Exception as e:
                        logger.warning(f"Could not load results: {e}")

                else:
                    failed += 1
                    logger.error(f"✗ Failed in {duration:.1f}s")
                    logger.error(f"  Error: {result.stderr[:200]}")

            except subprocess.TimeoutExpired:
                failed += 1
                logger.error(f"✗ Timeout after 10 minutes")
            except Exception as e:
                failed += 1
                logger.error(f"✗ Error: {str(e)[:200]}")

    total_duration = time.time() - start_time_total

    logger.info("\n" + "="*60)
    logger.info("SUITE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_duration/60:.1f} minutes")
    logger.info("="*60)

    # Save summary
    summary_file = os.path.join(args.output_dir, "summary.json")
    summary_data = {
        "total_experiments": total_experiments,
        "completed": completed,
        "failed": failed,
        "total_duration_seconds": total_duration,
        "models": args.models,
        "benchmarks": args.benchmarks,
        "num_samples_per_experiment": args.num_samples,
        "results": results_summary
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_file}")

    # Create results table
    logger.info("\n" + "="*60)
    logger.info("RESULTS TABLE")
    logger.info("="*60)

    # Group by model
    for model_name in args.models:
        model_results = [r for r in results_summary if r["model"] == model_name]

        if model_results:
            logger.info(f"\n{model_name.upper()} Model:")
            logger.info(f"  Benchmarks completed: {len(model_results)}")

            valid_rates = [r["valid_rate"] for r in model_results]
            r2_scores = [r["best_r2"] for r in model_results if r["best_r2"] is not None]

            logger.info(f"  Avg valid rate: {sum(valid_rates)/len(valid_rates)*100:.1f}%")
            if r2_scores:
                logger.info(f"  Avg best R²: {sum(r2_scores)/len(r2_scores):.4f}")
                logger.info(f"  Max R²: {max(r2_scores):.4f}")
            else:
                logger.info(f"  No valid R² scores")

    return completed == total_experiments


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
