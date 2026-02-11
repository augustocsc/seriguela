#!/usr/bin/env python3
"""
Generate academic report from raw_results.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def load_results(json_file):
    """Load raw results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def analyze_results(data):
    """Analyze results and generate statistics"""

    # Group by model
    by_model = defaultdict(list)
    by_benchmark = defaultdict(list)
    by_algorithm = defaultdict(list)

    for exp in data:
        model = exp['model']
        benchmark = exp['benchmark']
        algorithm = exp['algorithm']
        best_r2 = exp.get('best_r2', -1.0)

        by_model[model].append({'benchmark': benchmark, 'algorithm': algorithm, 'r2': best_r2})
        by_benchmark[benchmark].append({'model': model, 'algorithm': algorithm, 'r2': best_r2})
        by_algorithm[algorithm].append({'model': model, 'benchmark': benchmark, 'r2': best_r2})

    return by_model, by_benchmark, by_algorithm

def generate_latex_table(by_model, by_benchmark):
    """Generate LaTeX table"""

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Best R² scores by model and benchmark}")
    latex.append("\\label{tab:results}")
    latex.append("\\begin{tabular}{l" + "r" * len(by_model) + "}")
    latex.append("\\toprule")

    # Header
    models = sorted(by_model.keys())
    header = "Benchmark & " + " & ".join(models) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Rows
    benchmarks = sorted(by_benchmark.keys())
    for benchmark in benchmarks:
        row = [benchmark.replace('_', '\\_')]

        for model in models:
            # Find best R² for this model+benchmark
            best_r2 = -999
            for exp in by_model[model]:
                if exp['benchmark'] == benchmark and exp['r2'] > best_r2:
                    best_r2 = exp['r2']

            if best_r2 > -999:
                row.append(f"{best_r2:.4f}")
            else:
                row.append("---")

        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)

def generate_markdown_report(data, by_model, by_benchmark, by_algorithm):
    """Generate comprehensive markdown report"""

    md = []
    md.append("# Comprehensive Evaluation Report - Academic Analysis")
    md.append("")
    md.append(f"**Total Experiments**: {len(data)}")
    md.append("")

    # Summary statistics
    md.append("## Summary Statistics")
    md.append("")

    for model, experiments in sorted(by_model.items()):
        r2_values = [exp['r2'] for exp in experiments if exp['r2'] > -1.0]
        if r2_values:
            avg_r2 = statistics.mean(r2_values)
            max_r2 = max(r2_values)
            min_r2 = min(r2_values)
            md.append(f"### {model}")
            md.append(f"- Experiments: {len(experiments)}")
            md.append(f"- Valid experiments: {len(r2_values)}")
            md.append(f"- Average R²: {avg_r2:.4f}")
            md.append(f"- Best R²: {max_r2:.4f}")
            md.append(f"- Worst R²: {min_r2:.4f}")
            md.append("")

    # Best results per benchmark
    md.append("## Best Results per Benchmark")
    md.append("")
    md.append("| Benchmark | Model | Algorithm | R² | Expression |")
    md.append("|-----------|-------|-----------|----|-----------  |")

    for benchmark in sorted(by_benchmark.keys()):
        exps = by_benchmark[benchmark]
        best_exp = max(exps, key=lambda x: x['r2'])

        # Find full experiment data
        full_exp = None
        for exp in data:
            if (exp['model'] == best_exp['model'] and
                exp['benchmark'] == benchmark and
                exp['algorithm'] == best_exp['algorithm']):
                full_exp = exp
                break

        expr = full_exp.get('best_expression', 'N/A')[:50] if full_exp else 'N/A'
        md.append(f"| {benchmark} | {best_exp['model']} | {best_exp['algorithm']} | {best_exp['r2']:.4f} | {expr} |")

    md.append("")

    # PPO vs GRPO comparison
    md.append("## Algorithm Comparison (PPO vs GRPO)")
    md.append("")

    ppo_r2 = [exp['r2'] for exp in by_algorithm.get('ppo', []) if exp['r2'] > -1.0]
    grpo_r2 = [exp['r2'] for exp in by_algorithm.get('grpo', []) if exp['r2'] > -1.0]

    md.append(f"**PPO**:")
    md.append(f"- Experiments: {len(by_algorithm.get('ppo', []))}")
    if ppo_r2:
        md.append(f"- Average R²: {statistics.mean(ppo_r2):.4f}")
        md.append(f"- Best R²: {max(ppo_r2):.4f}")
    md.append("")

    md.append(f"**GRPO**:")
    md.append(f"- Experiments: {len(by_algorithm.get('grpo', []))}")
    if grpo_r2:
        md.append(f"- Average R²: {statistics.mean(grpo_r2):.4f}")
        md.append(f"- Best R²: {max(grpo_r2):.4f}")
    md.append("")

    # Model comparison table
    md.append("## Model Comparison Matrix")
    md.append("")
    md.append("| Model | Benchmarks Won | Avg R² | Best R² | Success Rate |")
    md.append("|-------|----------------|--------|---------|--------------|")

    for model in sorted(by_model.keys()):
        exps = by_model[model]
        r2_values = [exp['r2'] for exp in exps if exp['r2'] > -1.0]

        # Count benchmarks won
        benchmarks_won = 0
        for benchmark in by_benchmark.keys():
            best_in_benchmark = max(by_benchmark[benchmark], key=lambda x: x['r2'])
            if best_in_benchmark['model'] == model:
                benchmarks_won += 1

        if r2_values:
            avg_r2 = statistics.mean(r2_values)
            max_r2 = max(r2_values)
            success_rate = len(r2_values) / len(exps) * 100
            md.append(f"| {model} | {benchmarks_won}/12 | {avg_r2:.4f} | {max_r2:.4f} | {success_rate:.1f}% |")
        else:
            md.append(f"| {model} | 0/12 | N/A | N/A | 0.0% |")

    md.append("")

    return "\n".join(md)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_academic_report.py <raw_results.json>")
        sys.exit(1)

    json_file = Path(sys.argv[1])
    if not json_file.exists():
        print(f"Error: {json_file} not found")
        sys.exit(1)

    print(f"Loading results from {json_file}...")
    data = load_results(json_file)
    print(f"Loaded {len(data)} experiments")

    print("Analyzing results...")
    by_model, by_benchmark, by_algorithm = analyze_results(data)

    # Generate reports
    output_dir = json_file.parent / "analysis_results"
    output_dir.mkdir(exist_ok=True)

    # Markdown report
    print("Generating markdown report...")
    md_report = generate_markdown_report(data, by_model, by_benchmark, by_algorithm)
    md_file = output_dir / "academic_report.md"
    md_file.write_text(md_report, encoding='utf-8')
    print(f"[OK] Markdown report: {md_file}")

    # LaTeX table
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(by_model, by_benchmark)
    latex_file = output_dir / "results_table.tex"
    latex_file.write_text(latex_table, encoding='utf-8')
    print(f"[OK] LaTeX table: {latex_file}")

    # Summary JSON
    print("Generating summary JSON...")
    summary = {
        "total_experiments": len(data),
        "models": {model: {
            "experiments": len(exps),
            "valid_experiments": len([e for e in exps if e['r2'] > -1.0]),
            "avg_r2": statistics.mean([e['r2'] for e in exps if e['r2'] > -1.0]) if [e for e in exps if e['r2'] > -1.0] else None,
            "best_r2": max([e['r2'] for e in exps if e['r2'] > -1.0]) if [e for e in exps if e['r2'] > -1.0] else None,
        } for model, exps in by_model.items()},
        "algorithms": {algo: {
            "experiments": len(exps),
            "avg_r2": statistics.mean([e['r2'] for e in exps if e['r2'] > -1.0]) if [e for e in exps if e['r2'] > -1.0] else None,
        } for algo, exps in by_algorithm.items()}
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"[OK] Summary JSON: {summary_file}")

    print(f"\n[DONE] Analysis complete! Reports saved to: {output_dir}")

if __name__ == "__main__":
    main()
