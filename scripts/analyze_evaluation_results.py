#!/usr/bin/env python3
"""
Analyze and visualize comprehensive evaluation results
Generates academic-quality tables and insights
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class EvaluationAnalyzer:
    """Analyze evaluation results with academic rigor."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = self._load_all_results()

    def _load_all_results(self) -> Dict:
        """Load all results from directory."""
        # Find latest timestamp directory
        timestamp_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            raise ValueError(f"No results found in {self.results_dir}")

        latest_dir = max(timestamp_dirs, key=lambda x: x.name)
        print(f"Loading results from: {latest_dir}")

        # Load raw results
        raw_results_path = latest_dir / "raw_results.json"
        if not raw_results_path.exists():
            raise ValueError(f"No raw_results.json found in {latest_dir}")

        with open(raw_results_path) as f:
            raw_results = json.load(f)

        # Load report
        report_path = latest_dir / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
        else:
            report = {}

        # Load all individual histories
        histories = {}
        for result in raw_results:
            if result.get("success") and result.get("history"):
                key = f"{result['model']}_{result['benchmark']}_{result['algorithm']}"
                histories[key] = result["history"]

        return {
            "raw": raw_results,
            "report": report,
            "histories": histories,
            "dir": latest_dir
        }

    def analyze_convergence(self) -> pd.DataFrame:
        """Analyze convergence patterns across epochs."""
        convergence_data = []

        for key, history in self.data["histories"].items():
            model, benchmark, algorithm = key.split("_", 2)

            for epoch_data in history:
                epoch = epoch_data["epoch"]
                metrics = epoch_data.get("metrics", {})

                convergence_data.append({
                    "model": model,
                    "benchmark": benchmark,
                    "algorithm": algorithm,
                    "epoch": epoch,
                    "best_r2": metrics.get("best_r2", -1),
                    "mean_r2": metrics.get("mean_r2", -1),
                    "valid_rate": metrics.get("valid_rate", 0),
                    "unique_expressions": metrics.get("unique_expressions", 0)
                })

        return pd.DataFrame(convergence_data)

    def analyze_expression_complexity(self) -> pd.DataFrame:
        """Analyze expression complexity patterns."""
        complexity_data = []

        for key, history in self.data["histories"].items():
            model, benchmark, algorithm = key.split("_", 2)

            # Get all expressions from final epoch
            if history:
                final_epoch = history[-1]
                expressions = final_epoch.get("expressions", [])

                for expr_data in expressions:
                    if expr_data.get("is_valid"):
                        expr = expr_data["expression"]
                        complexity_data.append({
                            "model": model,
                            "benchmark": benchmark,
                            "algorithm": algorithm,
                            "expression": expr,
                            "r2": expr_data["r2"],
                            "length": len(expr),
                            "has_power": "**" in expr or "^" in expr,
                            "has_trig": any(op in expr for op in ["sin", "cos", "tan"]),
                            "has_exp": "exp" in expr,
                            "has_log": "log" in expr,
                            "depth": self._estimate_depth(expr)
                        })

        return pd.DataFrame(complexity_data)

    def _estimate_depth(self, expr: str) -> int:
        """Estimate expression depth (nesting level)."""
        max_depth = 0
        current_depth = 0

        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        return max_depth

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate comprehensive summary table."""
        summary_data = []

        for result in self.data["raw"]:
            if result.get("success") and "summary" in result:
                summary = result["summary"]
                summary_data.append({
                    "Model": result["model"],
                    "Size": self._get_model_size(result["model"]),
                    "Notation": "Prefix" if "prefix" in result["model"] else "Infix",
                    "Benchmark": result["benchmark"].replace("nguyen_", "N-"),
                    "Algorithm": result["algorithm"].upper(),
                    "Best R²": round(summary.get("best_r2", -1), 4),
                    "Best Expression": summary.get("best_expression", "")[:40],
                    "Epoch": summary.get("best_epoch", -1),
                    "Valid Rate": f"{summary.get('final_valid_rate', 0)*100:.1f}%"
                })

        df = pd.DataFrame(summary_data)

        # Sort by benchmark then model
        if not df.empty:
            df = df.sort_values(["Benchmark", "Model", "Algorithm"])

        return df

    def _get_model_size(self, model_name: str) -> str:
        """Get model size from name."""
        if "base" in model_name:
            return "124M"
        elif "medium" in model_name:
            return "355M"
        elif "large" in model_name:
            return "774M"
        return "Unknown"

    def plot_convergence_curves(self, output_dir: Path = None):
        """Plot convergence curves for all experiments."""
        if output_dir is None:
            output_dir = self.data["dir"] / "plots"
        output_dir.mkdir(exist_ok=True)

        convergence_df = self.analyze_convergence()

        # Group by benchmark
        benchmarks = convergence_df["benchmark"].unique()

        for benchmark in benchmarks:
            bench_data = convergence_df[convergence_df["benchmark"] == benchmark]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Convergence Analysis - {benchmark}", fontsize=14, fontweight='bold')

            # Plot 1: Best R² over epochs
            ax = axes[0, 0]
            for (model, algo), group in bench_data.groupby(["model", "algorithm"]):
                label = f"{model}-{algo}"
                ax.plot(group["epoch"], group["best_r2"], marker='o', label=label, alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Best R²")
            ax.set_title("Best R² Convergence")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot 2: Mean R² over epochs
            ax = axes[0, 1]
            for (model, algo), group in bench_data.groupby(["model", "algorithm"]):
                label = f"{model}-{algo}"
                ax.plot(group["epoch"], group["mean_r2"], marker='s', label=label, alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean R²")
            ax.set_title("Mean R² Convergence")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot 3: Valid rate over epochs
            ax = axes[1, 0]
            for (model, algo), group in bench_data.groupby(["model", "algorithm"]):
                label = f"{model}-{algo}"
                ax.plot(group["epoch"], group["valid_rate"], marker='^', label=label, alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Valid Expression Rate")
            ax.set_title("Validity Rate Evolution")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot 4: Unique expressions over epochs
            ax = axes[1, 1]
            for (model, algo), group in bench_data.groupby(["model", "algorithm"]):
                label = f"{model}-{algo}"
                ax.plot(group["epoch"], group["unique_expressions"], marker='d', label=label, alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Unique Expressions")
            ax.set_title("Expression Diversity")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f"convergence_{benchmark}.png", dpi=150, bbox_inches='tight')
            plt.close()

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for academic paper."""
        df = self.generate_summary_table()

        if df.empty:
            return "% No data available"

        # Pivot table for better presentation
        pivot = df.pivot_table(
            index=["Benchmark"],
            columns=["Model", "Algorithm"],
            values="Best R²",
            aggfunc="first"
        )

        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Symbolic Regression Performance on Nguyen Benchmarks}")
        latex.append("\\label{tab:results}")
        latex.append("\\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
        latex.append("\\toprule")

        # Header
        header = "Benchmark"
        for model, algo in pivot.columns:
            header += f" & {model.replace('_', ' ').title()} {algo}"
        latex.append(header + " \\\\")
        latex.append("\\midrule")

        # Data rows
        for benchmark in pivot.index:
            row = benchmark
            for col in pivot.columns:
                val = pivot.loc[benchmark, col]
                if pd.isna(val):
                    row += " & -"
                else:
                    row += f" & {val:.3f}"
            latex.append(row + " \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def generate_insights(self) -> Dict:
        """Generate key insights from the evaluation."""
        insights = {
            "best_overall": None,
            "best_per_model": {},
            "model_ranking": [],
            "algorithm_comparison": {},
            "complexity_analysis": {}
        }

        # Find best overall result
        best_r2 = -float('inf')
        for result in self.data["raw"]:
            if result.get("success") and "summary" in result:
                r2 = result["summary"].get("best_r2", -1)
                if r2 > best_r2:
                    best_r2 = r2
                    insights["best_overall"] = {
                        "model": result["model"],
                        "benchmark": result["benchmark"],
                        "algorithm": result["algorithm"],
                        "r2": r2,
                        "expression": result["summary"].get("best_expression", "")
                    }

        # Model ranking
        model_scores = defaultdict(list)
        for result in self.data["raw"]:
            if result.get("success") and "summary" in result:
                model = result["model"]
                r2 = result["summary"].get("best_r2", -1)
                model_scores[model].append(r2)

        for model, scores in model_scores.items():
            insights["model_ranking"].append({
                "model": model,
                "mean_r2": np.mean(scores),
                "max_r2": max(scores),
                "min_r2": min(scores),
                "std_r2": np.std(scores)
            })

        insights["model_ranking"].sort(key=lambda x: x["mean_r2"], reverse=True)

        # Algorithm comparison
        algo_scores = defaultdict(list)
        for result in self.data["raw"]:
            if result.get("success") and "summary" in result:
                algo = result["algorithm"]
                r2 = result["summary"].get("best_r2", -1)
                algo_scores[algo].append(r2)

        for algo, scores in algo_scores.items():
            insights["algorithm_comparison"][algo] = {
                "mean_r2": np.mean(scores),
                "wins": sum(1 for s in scores if s > 0.9),
                "total": len(scores)
            }

        # Complexity analysis
        complexity_df = self.analyze_expression_complexity()
        if not complexity_df.empty:
            for model in complexity_df["model"].unique():
                model_data = complexity_df[complexity_df["model"] == model]
                insights["complexity_analysis"][model] = {
                    "avg_length": model_data["length"].mean(),
                    "power_usage": (model_data["has_power"].sum() / len(model_data)) * 100,
                    "trig_usage": (model_data["has_trig"].sum() / len(model_data)) * 100,
                    "avg_depth": model_data["depth"].mean()
                }

        return insights

    def save_all_results(self):
        """Save all analysis results."""
        output_dir = self.data["dir"] / "analysis"
        output_dir.mkdir(exist_ok=True)

        # Save summary table
        summary_df = self.generate_summary_table()
        summary_df.to_csv(output_dir / "summary_table.csv", index=False)
        summary_df.to_excel(output_dir / "summary_table.xlsx", index=False)

        # Save convergence data
        convergence_df = self.analyze_convergence()
        convergence_df.to_csv(output_dir / "convergence_data.csv", index=False)

        # Save complexity analysis
        complexity_df = self.analyze_expression_complexity()
        complexity_df.to_csv(output_dir / "complexity_analysis.csv", index=False)

        # Save LaTeX table
        latex_table = self.generate_latex_table()
        with open(output_dir / "results_table.tex", "w") as f:
            f.write(latex_table)

        # Save insights
        insights = self.generate_insights()
        with open(output_dir / "insights.json", "w") as f:
            json.dump(insights, f, indent=2)

        # Generate plots
        self.plot_convergence_curves(output_dir / "plots")

        print(f"All analysis results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./evaluation_results",
                       help="Directory containing evaluation results")
    args = parser.parse_args()

    analyzer = EvaluationAnalyzer(args.results_dir)

    # Generate all analyses
    analyzer.save_all_results()

    # Print key insights
    insights = analyzer.generate_insights()

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    if insights["best_overall"]:
        print(f"\nBest Overall Result:")
        print(f"  Model: {insights['best_overall']['model']}")
        print(f"  Benchmark: {insights['best_overall']['benchmark']}")
        print(f"  Algorithm: {insights['best_overall']['algorithm']}")
        print(f"  R²: {insights['best_overall']['r2']:.4f}")
        print(f"  Expression: {insights['best_overall']['expression']}")

    print(f"\nModel Ranking (by mean R²):")
    for i, model in enumerate(insights["model_ranking"][:3], 1):
        print(f"  {i}. {model['model']}: {model['mean_r2']:.4f} (±{model['std_r2']:.4f})")

    print(f"\nAlgorithm Comparison:")
    for algo, stats in insights["algorithm_comparison"].items():
        print(f"  {algo.upper()}: mean R²={stats['mean_r2']:.4f}, "
              f"wins={stats['wins']}/{stats['total']}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()