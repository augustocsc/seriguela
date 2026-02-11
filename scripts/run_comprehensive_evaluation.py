#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Symbolic Regression Models
Evaluates all models on Nguyen benchmarks with PPO and GRPO algorithms
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Orchestrates evaluation of all models on all benchmarks."""

    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define models
        self.models = {
            "base_prefix": {
                "path": "./output/gpt2_base_prefix_682k",
                "is_prefix": True,
                "size": "124M"
            },
            "medium_prefix": {
                "path": "./output/gpt2_medium_prefix_682k",
                "is_prefix": True,
                "size": "355M"
            },
            "large_prefix": {
                "path": "./output/gpt2_large_prefix_682k",
                "is_prefix": True,
                "size": "774M"
            },
            "base_infix": {
                "path": "augustocsc/Se124M_700K_infix_v3_json",
                "is_prefix": False,
                "size": "124M"
            }
        }

        # Define benchmarks
        self.benchmarks = {}
        for i in range(1, 13):
            self.benchmarks[f"nguyen_{i}"] = f"./data/benchmarks/nguyen/nguyen_{i}.csv"

        # Algorithms to evaluate
        self.algorithms = ["ppo", "grpo"]

        # Unified prompt for all models
        self.unified_prompt = self._build_unified_prompt()

    def _build_unified_prompt(self) -> Dict[str, str]:
        """Build unified prompts for prefix and infix models."""
        all_ops = ["*", "+", "-", "/", "sin", "cos", "tan", "exp", "log", "sqrt", "abs"]

        # Prefix prompt (max 5 variables to cover all benchmarks)
        prefix_prompt = f"vars: x_1, x_2, x_3, x_4, x_5\noper: {', '.join(all_ops)}\ncons: C\nexpr: "

        # JSON prompt for infix
        json_prompt = json.dumps({
            "vars": ["x_1", "x_2", "x_3", "x_4", "x_5"],
            "ops": all_ops,
            "cons": "C",
            "expr": ""
        })[:-2]  # Remove closing "}

        return {
            "prefix": prefix_prompt,
            "infix": json_prompt
        }

    def run_experiment(
        self,
        model_name: str,
        model_info: dict,
        benchmark_name: str,
        benchmark_path: str,
        algorithm: str,
        epochs: int = 20,
        samples_per_epoch: int = 32
    ) -> Dict:
        """Run a single experiment."""
        logger.info(f"Running: {model_name} + {benchmark_name} + {algorithm.upper()}")

        # Determine output directory
        exp_output_dir = self.output_dir / f"{self.timestamp}" / model_name / benchmark_name / algorithm
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        # Select appropriate prompt
        prompt = self.unified_prompt["prefix"] if model_info["is_prefix"] else self.unified_prompt["infix"]

        # Build command
        if algorithm == "ppo":
            script = "scripts/ppo_symbolic_enhanced.py"
            cmd = [
                sys.executable, script,
                "--model_path", model_info["path"],
                "--dataset", benchmark_path,
                "--output_dir", str(exp_output_dir),
                "--epochs", str(epochs),
                "--samples_per_epoch", str(samples_per_epoch),
                "--custom_prompt", prompt,
                "--learning_rate", "3e-5"
            ]
        else:  # grpo
            script = "scripts/grpo_symbolic_enhanced.py"
            cmd = [
                sys.executable, script,
                "--model_path", model_info["path"],
                "--dataset", benchmark_path,
                "--output_dir", str(exp_output_dir),
                "--epochs", str(epochs),
                "--samples_per_group", "8",
                "--groups_per_epoch", str(samples_per_epoch // 8),
                "--custom_prompt", prompt,
                "--learning_rate", "5e-5"
            ]

        # Add prefix flag if needed
        if model_info["is_prefix"]:
            cmd.append("--is_prefix")

        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            duration = time.time() - start_time

            # Load results
            summary_path = exp_output_dir / "summary.json"
            history_path = exp_output_dir / "full_history.json"

            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                summary = {
                    "error": "No summary file found",
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:]
                }

            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
            else:
                history = []

            return {
                "success": result.returncode == 0,
                "duration": duration,
                "summary": summary,
                "history": history,
                "model": model_name,
                "benchmark": benchmark_name,
                "algorithm": algorithm
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout for {model_name} + {benchmark_name} + {algorithm}")
            return {
                "success": False,
                "error": "Timeout",
                "duration": time.time() - start_time,
                "model": model_name,
                "benchmark": benchmark_name,
                "algorithm": algorithm
            }
        except Exception as e:
            logger.error(f"Error in {model_name} + {benchmark_name} + {algorithm}: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time,
                "model": model_name,
                "benchmark": benchmark_name,
                "algorithm": algorithm
            }

    def run_full_suite(
        self,
        models_to_run: List[str] = None,
        benchmarks_to_run: List[str] = None,
        algorithms_to_run: List[str] = None,
        epochs: int = 20,
        samples_per_epoch: int = 32
    ):
        """Run complete evaluation suite."""
        # Use all if not specified
        if models_to_run is None:
            models_to_run = list(self.models.keys())
        if benchmarks_to_run is None:
            benchmarks_to_run = list(self.benchmarks.keys())
        if algorithms_to_run is None:
            algorithms_to_run = self.algorithms

        total_experiments = len(models_to_run) * len(benchmarks_to_run) * len(algorithms_to_run)
        logger.info(f"Starting evaluation suite: {total_experiments} experiments")
        logger.info(f"Models: {models_to_run}")
        logger.info(f"Benchmarks: {benchmarks_to_run}")
        logger.info(f"Algorithms: {algorithms_to_run}")

        results = []
        completed = 0

        for model_name in models_to_run:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found, skipping")
                continue

            model_info = self.models[model_name]

            # Check if model exists
            if not model_info["path"].startswith("augustocsc/"):  # Local model
                if not Path(model_info["path"]).exists():
                    logger.warning(f"Model path {model_info['path']} not found, skipping")
                    continue

            for benchmark_name in benchmarks_to_run:
                if benchmark_name not in self.benchmarks:
                    logger.warning(f"Benchmark {benchmark_name} not found, skipping")
                    continue

                benchmark_path = self.benchmarks[benchmark_name]

                for algorithm in algorithms_to_run:
                    completed += 1
                    logger.info(f"\n[{completed}/{total_experiments}] "
                               f"{model_name} + {benchmark_name} + {algorithm}")
                    logger.info("-" * 60)

                    result = self.run_experiment(
                        model_name=model_name,
                        model_info=model_info,
                        benchmark_name=benchmark_name,
                        benchmark_path=benchmark_path,
                        algorithm=algorithm,
                        epochs=epochs,
                        samples_per_epoch=samples_per_epoch
                    )

                    results.append(result)

                    # Save intermediate results
                    self.save_results(results)

        # Generate final report
        self.generate_report(results)

        return results

    def save_results(self, results: List[Dict]):
        """Save intermediate results."""
        output_path = self.output_dir / f"{self.timestamp}" / "raw_results.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def generate_report(self, results: List[Dict]):
        """Generate comprehensive report."""
        report = {
            "timestamp": self.timestamp,
            "total_experiments": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "summary_table": [],
            "best_per_benchmark": {},
            "best_per_model": {}
        }

        # Build summary table
        for result in results:
            if result.get("success") and "summary" in result:
                summary = result["summary"]
                report["summary_table"].append({
                    "model": result["model"],
                    "benchmark": result["benchmark"],
                    "algorithm": result["algorithm"],
                    "best_r2": summary.get("best_r2", -1),
                    "best_expression": summary.get("best_expression", ""),
                    "best_epoch": summary.get("best_epoch", -1),
                    "final_valid_rate": summary.get("final_valid_rate", 0),
                    "duration": result.get("duration", 0)
                })

                # Track best per benchmark
                bench = result["benchmark"]
                if bench not in report["best_per_benchmark"] or \
                   summary.get("best_r2", -1) > report["best_per_benchmark"][bench]["r2"]:
                    report["best_per_benchmark"][bench] = {
                        "r2": summary.get("best_r2", -1),
                        "expression": summary.get("best_expression", ""),
                        "model": result["model"],
                        "algorithm": result["algorithm"],
                        "epoch": summary.get("best_epoch", -1)
                    }

                # Track best per model
                model = result["model"]
                if model not in report["best_per_model"]:
                    report["best_per_model"][model] = {}

                if bench not in report["best_per_model"][model] or \
                   summary.get("best_r2", -1) > report["best_per_model"][model][bench]["r2"]:
                    report["best_per_model"][model][bench] = {
                        "r2": summary.get("best_r2", -1),
                        "expression": summary.get("best_expression", ""),
                        "algorithm": result["algorithm"],
                        "epoch": summary.get("best_epoch", -1)
                    }

        # Save report
        report_path = self.output_dir / f"{self.timestamp}" / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        self.generate_markdown_report(report)

        logger.info(f"Report saved to {report_path}")

    def generate_markdown_report(self, report: Dict):
        """Generate human-readable markdown report."""
        md_lines = []
        md_lines.append("# Comprehensive Symbolic Regression Evaluation Report")
        md_lines.append(f"\n**Date**: {report['timestamp']}")
        md_lines.append(f"**Total Experiments**: {report['total_experiments']}")
        md_lines.append(f"**Successful**: {report['successful']}")
        md_lines.append(f"**Failed**: {report['failed']}")

        # Best per benchmark table
        md_lines.append("\n## Best Results per Benchmark\n")
        md_lines.append("| Benchmark | Best R² | Model | Algorithm | Expression | Epoch |")
        md_lines.append("|-----------|---------|-------|-----------|------------|-------|")

        for bench in sorted(report["best_per_benchmark"].keys()):
            data = report["best_per_benchmark"][bench]
            expr = data["expression"][:50] + "..." if len(data["expression"]) > 50 else data["expression"]
            md_lines.append(f"| {bench} | {data['r2']:.4f} | {data['model']} | "
                          f"{data['algorithm']} | {expr} | {data['epoch']} |")

        # Model comparison table
        md_lines.append("\n## Model Comparison\n")

        for model in sorted(report["best_per_model"].keys()):
            md_lines.append(f"\n### {model}\n")
            md_lines.append("| Benchmark | Best R² | Algorithm | Expression | Epoch |")
            md_lines.append("|-----------|---------|-----------|------------|-------|")

            for bench in sorted(report["best_per_model"][model].keys()):
                data = report["best_per_model"][model][bench]
                expr = data["expression"][:50] + "..." if len(data["expression"]) > 50 else data["expression"]
                md_lines.append(f"| {bench} | {data['r2']:.4f} | {data['algorithm']} | {expr} | {data['epoch']} |")

        # Save markdown
        md_path = self.output_dir / f"{self.timestamp}" / "report.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown report saved to {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./evaluation_results")
    parser.add_argument("--models", nargs="+", help="Models to evaluate")
    parser.add_argument("--benchmarks", nargs="+", help="Benchmarks to run")
    parser.add_argument("--algorithms", nargs="+", default=["ppo", "grpo"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples_per_epoch", type=int, default=32)
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with subset")
    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(output_dir=args.output_dir)

    if args.quick_test:
        # Quick test with subset
        results = evaluator.run_full_suite(
            models_to_run=["base_prefix"],
            benchmarks_to_run=["nguyen_1"],
            algorithms_to_run=["ppo"],
            epochs=2,
            samples_per_epoch=8
        )
    else:
        results = evaluator.run_full_suite(
            models_to_run=args.models,
            benchmarks_to_run=args.benchmarks,
            algorithms_to_run=args.algorithms,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch
        )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()