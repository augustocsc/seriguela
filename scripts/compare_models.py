"""
Compare two models: band-aided vs properly trained.
Evaluates both on same test set and reports metrics.

Usage:
    python scripts/compare_models.py \
        --model1 ./output/Se124M_700K_infix \
        --model2 ./output/Se124M_700K_infix_v2 \
        --num_samples 500
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Import evaluate_model from evaluate.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_model


def format_metric(value, metric_type):
    """Format metric value for display."""
    if metric_type == "rate":
        return f"{value * 100:5.1f}%"
    elif metric_type == "float":
        return f"{value:7.2f}"
    elif metric_type == "int":
        return f"{int(value):7d}"
    else:
        return f"{value:7}"


def print_comparison_table(metrics1, metrics2, model1_name, model2_name):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Metric':<35} {model1_name:>20} {model2_name:>20}")
    print("-" * 80)

    # Define metrics to compare
    comparison_metrics = [
        ("valid_rate", "Valid Rate", "rate"),
        ("parseable_rate", "Parseable Rate", "rate"),
        ("constraints_met_rate", "Constraints Met", "rate"),
        ("diversity_rate", "Diversity", "rate"),
        ("avg_expression_length", "Avg Expression Length", "float"),
        ("total_samples", "Total Samples", "int"),
        ("total_valid", "Total Valid", "int"),
    ]

    improvements = []

    for key, label, metric_type in comparison_metrics:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)

        formatted_val1 = format_metric(val1, metric_type)
        formatted_val2 = format_metric(val2, metric_type)

        print(f"{label:<35} {formatted_val1:>20} {formatted_val2:>20}")

        # Calculate improvement for rate metrics
        if metric_type == "rate" and val1 > 0:
            improvement = ((val2 - val1) / val1) * 100
            improvements.append((label, improvement, val2 - val1))

    print("=" * 80)

    # Show improvements
    print("\nIMPROVEMENTS (Model 2 vs Model 1):")
    print("-" * 80)

    for label, improvement, absolute_diff in improvements:
        sign = "+" if improvement > 0 else ""
        abs_sign = "+" if absolute_diff > 0 else ""
        print(f"{label:<35} {sign}{improvement:>6.1f}%  ({abs_sign}{absolute_diff * 100:>5.1f} pp)")

    print("-" * 80)

    # Determine winner
    valid_rate_improvement = metrics2.get("valid_rate", 0) - metrics1.get("valid_rate", 0)

    print("\n" + "=" * 80)
    if valid_rate_improvement > 0.20:  # >20% improvement
        print(f"🎯 SIGNIFICANT IMPROVEMENT: Model 2 wins by {valid_rate_improvement * 100:.1f} percentage points")
        print("   The properly trained model significantly outperforms the band-aided version!")
    elif valid_rate_improvement > 0.05:  # >5% improvement
        print(f"✅ IMPROVEMENT: Model 2 wins by {valid_rate_improvement * 100:.1f} percentage points")
        print("   The properly trained model shows clear improvement.")
    elif valid_rate_improvement > 0:  # Any improvement
        print(f"📈 SLIGHT IMPROVEMENT: Model 2 wins by {valid_rate_improvement * 100:.1f} percentage points")
        print("   The properly trained model shows modest improvement.")
    elif valid_rate_improvement == 0:
        print("⚖️  TIE: Both models perform equally")
        print("   No significant difference between models.")
    else:
        print(f"⚠️  REGRESSION: Model 1 wins by {-valid_rate_improvement * 100:.1f} percentage points")
        print("   The band-aided model performs better - retraining may need adjustment.")

    print("=" * 80)


def save_comparison_report(metrics1, metrics2, model1_name, model2_name, output_dir):
    """Save detailed comparison report to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"comparison_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "model1": {
            "name": model1_name,
            "metrics": metrics1
        },
        "model2": {
            "name": model2_name,
            "metrics": metrics2
        },
        "comparison": {
            "valid_rate_diff": metrics2.get("valid_rate", 0) - metrics1.get("valid_rate", 0),
            "parseable_rate_diff": metrics2.get("parseable_rate", 0) - metrics1.get("parseable_rate", 0),
            "constraints_met_diff": metrics2.get("constraints_met_rate", 0) - metrics1.get("constraints_met_rate", 0),
            "diversity_diff": metrics2.get("diversity_rate", 0) - metrics1.get("diversity_rate", 0),
        }
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed comparison report saved to: {report_file}")
    return report_file


def compare_models(model1_path, model2_path, model1_name, model2_name,
                   num_samples=500, dataset_repo_id="augustocsc/sintetico_natural",
                   data_dir="700K", data_column="i_prompt_n", output_dir="./evaluation_results/comparison"):
    """Compare two models on same test set."""

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"Model 1 ({model1_name}): {model1_path}")
    print(f"Model 2 ({model2_name}): {model2_path}")
    print(f"Samples: {num_samples}")
    print(f"Dataset: {dataset_repo_id}/{data_dir}")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate Model 1 (band-aided)
    print(f"\n[1/2] Evaluating Model 1: {model1_name}")
    print("-" * 80)

    args1 = argparse.Namespace(
        model_path=model1_path,
        base_model=None,
        dataset_repo_id=dataset_repo_id,
        data_dir=data_dir,
        data_column=data_column,
        num_samples=num_samples,
        num_generations=1,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        output_dir=os.path.join(output_dir, "model1"),
        seed=42,
        device="auto"
    )

    try:
        metrics1 = evaluate_model(args1)
    except Exception as e:
        print(f"\n❌ Error evaluating Model 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Evaluate Model 2 (properly trained)
    print(f"\n[2/2] Evaluating Model 2: {model2_name}")
    print("-" * 80)

    args2 = argparse.Namespace(
        model_path=model2_path,
        base_model=None,
        dataset_repo_id=dataset_repo_id,
        data_dir=data_dir,
        data_column=data_column,
        num_samples=num_samples,
        num_generations=1,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        output_dir=os.path.join(output_dir, "model2"),
        seed=42,
        device="auto"
    )

    try:
        metrics2 = evaluate_model(args2)
    except Exception as e:
        print(f"\n❌ Error evaluating Model 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print comparison
    print_comparison_table(metrics1, metrics2, model1_name, model2_name)

    # Save report
    save_comparison_report(metrics1, metrics2, model1_name, model2_name, output_dir)

    return metrics1, metrics2


def main():
    parser = argparse.ArgumentParser(
        description="Compare two models on the same test set"
    )
    parser.add_argument("--model1", type=str, required=True,
                        help="Path to first model (band-aided)")
    parser.add_argument("--model2", type=str, required=True,
                        help="Path to second model (properly trained)")
    parser.add_argument("--model1_name", type=str, default="Band-Aided",
                        help="Display name for model 1")
    parser.add_argument("--model2_name", type=str, default="Proper",
                        help="Display name for model 2")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--dataset_repo_id", type=str, default="augustocsc/sintetico_natural",
                        help="HuggingFace dataset repository")
    parser.add_argument("--data_dir", type=str, default="700K",
                        help="Data directory within dataset")
    parser.add_argument("--data_column", type=str, default="i_prompt_n",
                        help="Column name for prompts")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results/comparison",
                        help="Directory to save comparison results")

    args = parser.parse_args()

    # Run comparison
    try:
        compare_models(
            model1_path=args.model1,
            model2_path=args.model2,
            model1_name=args.model1_name,
            model2_name=args.model2_name,
            num_samples=args.num_samples,
            dataset_repo_id=args.dataset_repo_id,
            data_dir=args.data_dir,
            data_column=args.data_column,
            output_dir=args.output_dir
        )

        print("\n✅ Comparison complete!")

    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
