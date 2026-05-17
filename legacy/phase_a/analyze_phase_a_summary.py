"""
Phase A Results Analysis - Summary Report
==========================================
Analyzes the Phase A factorial experiment results to identify
best hyperparameter configurations for Phase B.

Reads from: phase_a_all_results.csv
"""
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_data(csv_path):
    """Load and parse the CSV results."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse numeric fields
            try:
                row['best_r2'] = float(row['best_r2']) if row['best_r2'] else 0.0
                row['final_r2'] = float(row['final_r2']) if row['final_r2'] else 0.0
                row['final_loss'] = float(row['final_loss']) if row['final_loss'] else 0.0
                row['valid_rate'] = float(row['valid_rate']) if row['valid_rate'] else 0.0
                row['noise'] = float(row['noise']) if row['noise'] else 0.0
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def filter_rl_runs(rows):
    """Filter to only RL algorithm runs (exclude best_of_n baseline)."""
    return [r for r in rows if r['algorithm'] != 'best_of_n']


def filter_bon_runs(rows):
    """Filter to only best_of_n runs."""
    return [r for r in rows if r['algorithm'] == 'best_of_n']


def overall_stats(rows):
    """Compute overall statistics."""
    r2_values = [r['best_r2'] for r in rows if r['best_r2'] > 0]
    perfect = [r for r in rows if r['best_r2'] >= 0.999]
    high = [r for r in rows if r['best_r2'] >= 0.99]
    good = [r for r in rows if r['best_r2'] >= 0.95]

    print("=" * 70)
    print("PHASE A RESULTS - OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total runs:            {len(rows)}")
    print(f"Runs with R² > 0:      {len(r2_values)}")
    print(f"Perfect fits (≥0.999): {len(perfect)} ({100*len(perfect)/len(rows):.1f}%)")
    print(f"High fits (≥0.99):     {len(high)} ({100*len(high)/len(rows):.1f}%)")
    print(f"Good fits (≥0.95):     {len(good)} ({100*len(good)/len(rows):.1f}%)")
    if r2_values:
        print(f"Mean best R² (>0):     {sum(r2_values)/len(r2_values):.4f}")
        print(f"Max best R²:           {max(r2_values):.6f}")
        print(f"Median best R²:        {sorted(r2_values)[len(r2_values)//2]:.4f}")
    print()


def by_algorithm(rows):
    """Performance breakdown by algorithm."""
    algos = defaultdict(list)
    for r in rows:
        algos[r['algorithm']].append(r['best_r2'])

    print("=" * 70)
    print("PERFORMANCE BY ALGORITHM")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Count':>6} {'Mean R²':>10} {'Max R²':>10} {'≥0.99':>8} {'≥0.95':>8}")
    print("-" * 70)
    for algo in sorted(algos.keys()):
        vals = algos[algo]
        positive = [v for v in vals if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        max_r2 = max(vals) if vals else 0
        gte99 = sum(1 for v in vals if v >= 0.99)
        gte95 = sum(1 for v in vals if v >= 0.95)
        print(f"{algo:<15} {len(vals):>6} {mean_r2:>10.4f} {max_r2:>10.6f} {gte99:>8} {gte95:>8}")
    print()


def by_problem(rows):
    """Performance breakdown by Nguyen problem."""
    problems = defaultdict(list)
    for r in rows:
        problems[r['problem']].append(r['best_r2'])

    print("=" * 70)
    print("PERFORMANCE BY PROBLEM")
    print("=" * 70)
    print(f"{'Problem':<15} {'Count':>6} {'Mean R²':>10} {'Max R²':>10} {'≥0.99':>8} {'≥0.95':>8}")
    print("-" * 70)
    for prob in sorted(problems.keys()):
        vals = problems[prob]
        positive = [v for v in vals if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        max_r2 = max(vals) if vals else 0
        gte99 = sum(1 for v in vals if v >= 0.99)
        gte95 = sum(1 for v in vals if v >= 0.95)
        print(f"{prob:<15} {len(vals):>6} {mean_r2:>10.4f} {max_r2:>10.6f} {gte99:>8} {gte95:>8}")
    print()


def by_notation(rows):
    """Performance breakdown by notation (infix vs prefix)."""
    notations = defaultdict(list)
    for r in rows:
        if 'infix' in r.get('model', '').lower() or 'infix' in r.get('config', '').lower():
            notations['infix'].append(r['best_r2'])
        elif 'prefix' in r.get('model', '').lower() or 'prefix' in r.get('config', '').lower():
            notations['prefix'].append(r['best_r2'])

    print("=" * 70)
    print("PERFORMANCE BY NOTATION")
    print("=" * 70)
    print(f"{'Notation':<15} {'Count':>6} {'Mean R²':>10} {'Max R²':>10} {'≥0.99':>8} {'≥0.95':>8}")
    print("-" * 70)
    for notation in sorted(notations.keys()):
        vals = notations[notation]
        positive = [v for v in vals if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        max_r2 = max(vals) if vals else 0
        gte99 = sum(1 for v in vals if v >= 0.99)
        gte95 = sum(1 for v in vals if v >= 0.95)
        print(f"{notation:<15} {len(vals):>6} {mean_r2:>10.4f} {max_r2:>10.6f} {gte99:>8} {gte95:>8}")
    print()


def by_temperature(rows):
    """Performance breakdown by temperature strategy."""
    temps = defaultdict(list)
    for r in rows:
        t = r.get('temperature', 'unknown')
        if t:
            temps[t].append(r['best_r2'])

    print("=" * 70)
    print("PERFORMANCE BY TEMPERATURE STRATEGY")
    print("=" * 70)
    print(f"{'Temperature':<20} {'Count':>6} {'Mean R²':>10} {'Max R²':>10} {'≥0.99':>8} {'≥0.95':>8}")
    print("-" * 70)
    for temp in sorted(temps.keys()):
        vals = temps[temp]
        positive = [v for v in vals if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        max_r2 = max(vals) if vals else 0
        gte99 = sum(1 for v in vals if v >= 0.99)
        gte95 = sum(1 for v in vals if v >= 0.95)
        print(f"{temp:<20} {len(vals):>6} {mean_r2:>10.4f} {max_r2:>10.6f} {gte99:>8} {gte95:>8}")
    print()


def by_prompt(rows):
    """Performance breakdown by prompt type."""
    prompts = defaultdict(list)
    for r in rows:
        p = r.get('prompt', 'unknown')
        if p:
            prompts[p].append(r['best_r2'])

    print("=" * 70)
    print("PERFORMANCE BY PROMPT TYPE")
    print("=" * 70)
    print(f"{'Prompt':<15} {'Count':>6} {'Mean R²':>10} {'Max R²':>10} {'≥0.99':>8} {'≥0.95':>8}")
    print("-" * 70)
    for prompt in sorted(prompts.keys()):
        vals = prompts[prompt]
        positive = [v for v in vals if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        max_r2 = max(vals) if vals else 0
        gte99 = sum(1 for v in vals if v >= 0.99)
        gte95 = sum(1 for v in vals if v >= 0.95)
        print(f"{prompt:<15} {len(vals):>6} {mean_r2:>10.4f} {max_r2:>10.6f} {gte99:>8} {gte95:>8}")
    print()


def by_noise(rows):
    """Performance breakdown by noise level."""
    noise_levels = defaultdict(list)
    for r in rows:
        noise_levels[r['noise']].append(r['best_r2'])

    print("=" * 70)
    print("PERFORMANCE BY NOISE LEVEL")
    print("=" * 70)
    print(f"{'Noise':<10} {'Count':>6} {'Mean R²':>10} {'Max R²':>10} {'≥0.99':>8} {'≥0.95':>8}")
    print("-" * 70)
    for noise in sorted(noise_levels.keys()):
        vals = noise_levels[noise]
        positive = [v for v in vals if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        max_r2 = max(vals) if vals else 0
        gte99 = sum(1 for v in vals if v >= 0.99)
        gte95 = sum(1 for v in vals if v >= 0.95)
        print(f"{noise:<10.2f} {len(vals):>6} {mean_r2:>10.4f} {max_r2:>10.6f} {gte99:>8} {gte95:>8}")
    print()


def top_configurations(rows, n=20):
    """Find the top N best configurations by best_r2."""
    # Group by (algorithm, problem, temperature, prompt, noise) and pick best run
    sorted_rows = sorted(rows, key=lambda r: r['best_r2'], reverse=True)

    print("=" * 70)
    print(f"TOP {n} INDIVIDUAL RUNS BY BEST R²")
    print("=" * 70)
    print(f"{'#':>3} {'Algorithm':<12} {'Problem':<12} {'Temp':<18} {'Prompt':<10} {'Noise':>6} {'Config':<18} {'R²':>10}")
    print("-" * 100)
    seen = set()
    count = 0
    for r in sorted_rows:
        key = (r['algorithm'], r['problem'], r['temperature'], r['prompt'], r['noise'], r['config'])
        if key in seen:
            continue
        seen.add(key)
        count += 1
        if count > n:
            break
        print(f"{count:>3} {r['algorithm']:<12} {r['problem']:<12} {r['temperature']:<18} {r['prompt']:<10} {r['noise']:>6.2f} {r['config']:<18} {r['best_r2']:>10.6f}")
    print()


def best_configs_for_phase_b(rows):
    """Identify best configurations to carry forward to Phase B."""
    # For RL runs only, group by (algorithm, temperature, prompt) and compute mean R² across problems
    rl_runs = filter_rl_runs(rows)

    config_scores = defaultdict(lambda: {'r2_values': [], 'perfect': 0, 'count': 0})

    for r in rl_runs:
        key = (r['algorithm'], r['temperature'], r['prompt'])
        config_scores[key]['r2_values'].append(r['best_r2'])
        config_scores[key]['count'] += 1
        if r['best_r2'] >= 0.999:
            config_scores[key]['perfect'] += 1

    print("=" * 70)
    print("BEST RL CONFIGURATIONS FOR PHASE B")
    print("(Ranked by mean R² across all problems/noise levels)")
    print("=" * 70)
    print(f"{'#':>3} {'Algorithm':<12} {'Temperature':<18} {'Prompt':<10} {'Count':>6} {'Mean R²':>10} {'Perfect':>8}")
    print("-" * 80)

    ranked = []
    for key, stats in config_scores.items():
        positive = [v for v in stats['r2_values'] if v > 0]
        mean_r2 = sum(positive) / len(positive) if positive else 0
        ranked.append((key, mean_r2, stats['perfect'], stats['count']))

    ranked.sort(key=lambda x: x[1], reverse=True)

    for i, (key, mean_r2, perfect, count) in enumerate(ranked[:15], 1):
        algo, temp, prompt = key
        print(f"{i:>3} {algo:<12} {temp:<18} {prompt:<10} {count:>6} {mean_r2:>10.4f} {perfect:>8}")
    print()


def algorithm_vs_problem_heatmap(rows):
    """Cross-table: algorithm x problem showing mean best R²."""
    rl_runs = [r for r in rows if r['best_r2'] > 0]
    combo = defaultdict(list)
    for r in rl_runs:
        combo[(r['algorithm'], r['problem'])].append(r['best_r2'])

    algos = sorted(set(r['algorithm'] for r in rl_runs))
    problems = sorted(set(r['problem'] for r in rl_runs))

    print("=" * 70)
    print("ALGORITHM × PROBLEM CROSS-TABLE (Mean Best R²)")
    print("=" * 70)

    # Header
    header = f"{'Algorithm':<15}" + "".join(f"{p:>12}" for p in problems)
    print(header)
    print("-" * len(header))

    for algo in algos:
        line = f"{algo:<15}"
        for prob in problems:
            vals = combo.get((algo, prob), [])
            if vals:
                mean_r2 = sum(vals) / len(vals)
                line += f"{mean_r2:>12.4f}"
            else:
                line += f"{'N/A':>12}"
        print(line)
    print()


def bon_vs_rl_comparison(rows):
    """Compare best_of_n baseline results vs RL algorithms."""
    bon = filter_bon_runs(rows)
    rl = filter_rl_runs(rows)

    print("=" * 70)
    print("BEST-OF-N (BASELINE) vs RL ALGORITHMS")
    print("=" * 70)

    bon_positive = [r['best_r2'] for r in bon if r['best_r2'] > 0]
    rl_positive = [r['best_r2'] for r in rl if r['best_r2'] > 0]

    print(f"{'Metric':<30} {'Best-of-N':>15} {'RL':>15}")
    print("-" * 60)
    print(f"{'Total runs':<30} {len(bon):>15} {len(rl):>15}")
    print(f"{'Runs with R²>0':<30} {len(bon_positive):>15} {len(rl_positive):>15}")
    if bon_positive:
        print(f"{'Mean R² (>0)':<30} {sum(bon_positive)/len(bon_positive):>15.4f} {sum(rl_positive)/len(rl_positive):>15.4f}")
        print(f"{'Max R²':<30} {max(bon_positive) if bon_positive else 0:>15.6f} {max(rl_positive) if rl_positive else 0:>15.6f}")
        bon99 = sum(1 for v in bon_positive if v >= 0.99)
        rl99 = sum(1 for v in rl_positive if v >= 0.99)
        print(f"{'R² ≥ 0.99 count':<30} {bon99:>15} {rl99:>15}")
        bon95 = sum(1 for v in bon_positive if v >= 0.95)
        rl95 = sum(1 for v in rl_positive if v >= 0.95)
        print(f"{'R² ≥ 0.95 count':<30} {bon95:>15} {rl95:>15}")
    print()


def save_summary_json(rows, output_path):
    """Save a structured summary as JSON for further analysis."""
    summary = {
        'total_runs': len(rows),
        'by_algorithm': {},
        'by_problem': {},
        'by_notation': {},
        'by_temperature': {},
        'by_prompt': {},
        'by_noise': {},
        'top_configs': []
    }

    # By algorithm
    algos = defaultdict(list)
    for r in rows:
        algos[r['algorithm']].append(r['best_r2'])
    for algo, vals in algos.items():
        positive = [v for v in vals if v > 0]
        summary['by_algorithm'][algo] = {
            'count': len(vals),
            'mean_r2': round(sum(positive) / len(positive), 4) if positive else 0,
            'max_r2': round(max(vals), 6) if vals else 0,
            'gte_099': sum(1 for v in vals if v >= 0.99),
            'gte_095': sum(1 for v in vals if v >= 0.95),
        }

    # By problem
    problems = defaultdict(list)
    for r in rows:
        problems[r['problem']].append(r['best_r2'])
    for prob, vals in problems.items():
        positive = [v for v in vals if v > 0]
        summary['by_problem'][prob] = {
            'count': len(vals),
            'mean_r2': round(sum(positive) / len(positive), 4) if positive else 0,
            'max_r2': round(max(vals), 6) if vals else 0,
            'gte_099': sum(1 for v in vals if v >= 0.99),
            'gte_095': sum(1 for v in vals if v >= 0.95),
        }

    # Top 10 configs
    sorted_rows = sorted(rows, key=lambda r: r['best_r2'], reverse=True)
    seen = set()
    for r in sorted_rows[:50]:
        key = (r['algorithm'], r['problem'], r['temperature'], r['prompt'])
        if key not in seen:
            seen.add(key)
            summary['top_configs'].append({
                'algorithm': r['algorithm'],
                'problem': r['problem'],
                'temperature': r['temperature'],
                'prompt': r['prompt'],
                'noise': r['noise'],
                'config': r['config'],
                'best_r2': round(r['best_r2'], 6),
            })
        if len(summary['top_configs']) >= 10:
            break

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to: {output_path}")


def main():
    csv_path = Path(__file__).parent / 'phase_a_all_results.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found!")
        return

    print(f"Loading data from {csv_path}...")
    rows = load_data(csv_path)
    print(f"Loaded {len(rows)} rows.\n")

    overall_stats(rows)
    by_algorithm(rows)
    by_problem(rows)
    by_notation(rows)
    by_temperature(rows)
    by_prompt(rows)
    by_noise(rows)
    bon_vs_rl_comparison(rows)
    algorithm_vs_problem_heatmap(rows)
    top_configurations(rows)
    best_configs_for_phase_b(rows)

    # Save summary JSON
    output_path = Path(__file__).parent / 'phase_a_summary.json'
    save_summary_json(rows, output_path)


if __name__ == '__main__':
    main()
