"""
Parse Exp13 Phase A results and generate LaTeX tables for paper.

Usage:
    python parse_exp13_results.py exp13_phase_a_scaling_20260722_174401.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def parse_results(filename):
    """Load and parse exp13 Phase A results."""
    with open(filename, 'r') as f:
        results = json.load(f)

    # Group by dataset
    by_dataset = defaultdict(list)
    for r in results:
        if r['status'] == 'OK':
            by_dataset[r['dataset']].append(r)

    return by_dataset

def generate_latex_table(by_dataset):
    """Generate LaTeX table with saturation analysis."""

    print("\n% LaTeX Table: Extended Scaling Study")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Extended Scaling Study: Accuracy Saturation and Context Efficiency}")
    print(r"\label{tab:scaling_results}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Dataset & Train Rows & AUC & Fit (s) & Predict (s/sample) \\")
    print(r"\midrule")

    for dataset_name in sorted(by_dataset.keys()):
        results = sorted(by_dataset[dataset_name], key=lambda x: x['train_rows'])

        # Print dataset header
        for i, r in enumerate(results):
            if i == 0:
                prefix = f"{dataset_name} "
            else:
                prefix = ""

            auc_str = f"{r['auc']:.4f}"
            fit_str = f"{r['fit_sec']:.2f}"
            pred_str = f"{r['predict_sec'] / r['test_rows']:.4f}"

            print(f"{prefix:20s} & {r['train_rows']:>6,d} & {auc_str} & {fit_str} & {pred_str} \\\\")

        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Generate summary statistics
    print("\n% Summary Statistics")
    print("%")
    for dataset_name in sorted(by_dataset.keys()):
        results = sorted(by_dataset[dataset_name], key=lambda x: x['train_rows'])
        if len(results) > 1:
            auc_min = min(r['auc'] for r in results)
            auc_max = max(r['auc'] for r in results)
            auc_gain = (auc_max - auc_min) / auc_min * 100 if auc_min > 0 else 0

            n_min = results[0]['train_rows']
            n_max = results[-1]['train_rows']

            print(f"% {dataset_name}: n={n_min:,}-{n_max:,}, AUC={auc_min:.4f}→{auc_max:.4f} (+{auc_gain:.2f}%)")

def find_saturation_point(by_dataset, threshold=0.01):
    """Identify saturation points (where AUC gain < threshold)."""
    print("\n% Saturation Analysis (threshold: <{:.2f}% AUC gain)")
    print("%")

    for dataset_name in sorted(by_dataset.keys()):
        results = sorted(by_dataset[dataset_name], key=lambda x: x['train_rows'])

        if len(results) < 2:
            continue

        saturated_at = None
        for i in range(1, len(results)):
            gain = (results[i]['auc'] - results[i-1]['auc']) / results[i-1]['auc'] * 100

            if abs(gain) < threshold and saturated_at is None:
                saturated_at = results[i-1]['train_rows']
                print(f"% {dataset_name}: saturates at n ≥ {saturated_at:,} (gain < {threshold:.2f}%)")
                break

        if saturated_at is None:
            # No saturation observed in range
            n_max = results[-1]['train_rows']
            print(f"% {dataset_name}: no saturation by n={n_max:,}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_exp13_results.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: {results_file} not found")
        sys.exit(1)

    print(f"Parsing: {results_file}")
    by_dataset = parse_results(results_file)

    print(f"Found {len(by_dataset)} datasets with {sum(len(v) for v in by_dataset.values())} measurements\n")

    generate_latex_table(by_dataset)
    find_saturation_point(by_dataset)
