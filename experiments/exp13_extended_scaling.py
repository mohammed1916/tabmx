"""
Experiment 13: Extended Scaling Study

Objective: Determine practical limits of full TabPFN across larger datasets
and more diverse sources.

Datasets: bank-marketing (40K), adult (43K), electricity (40K) at multiple
subsample sizes up to natural size, plus larger public datasets if available.

Focus: Where does TabPFN break? Memory, accuracy, latency.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SEED = 42

# Extended dataset list with different sizes
DATASETS = [
    # Small (control)
    (37, "diabetes", 691),
    (31, "credit-g", 1000),

    # Medium (original)
    (1461, "bank-marketing", 40711),
    (1590, "adult", 43957),
    (151, "electricity", 40780),

    # Large (new) - try if available
    # (310, "forest-type", 580000),  # Covertype
    # (344, "higgs-small", 1000000),  # Higgs (large)
]

# Subsample sizes to test (in addition to full dataset)
SUBSAMPLE_SIZES = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]


def run_extended_scaling():
    """Extended scaling study across multiple datasets and sizes."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp13_extended_scaling_{stamp}.json"
    results = []

    print("=" * 80)
    print("EXPERIMENT 13: Extended Scaling Study")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print()

    for openml_id, name, expected_size in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {name} (expected ~{expected_size:,} rows)")
        print(f"{'='*80}")

        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:
            print(f"FAILED to load {name}: {e}")
            continue

        actual_size = X.shape[0]
        n_classes = len(np.unique(y))
        print(f"Loaded: {actual_size:,} rows, {X.shape[1]} features, {n_classes} classes")

        # Test at full size and subsamples
        test_sizes = [s for s in SUBSAMPLE_SIZES if s < actual_size] + [actual_size]
        test_sizes = sorted(set(test_sizes))

        for n_test in test_sizes:
            print(f"\n  Testing n={n_test:,}...", end=" ", flush=True)

            try:
                # Subsample if needed
                if n_test < actual_size:
                    rng = np.random.RandomState(SEED)
                    idx = rng.choice(actual_size, size=n_test, replace=False)
                    X_sub, y_sub = X[idx], y[idx]
                else:
                    X_sub, y_sub = X, y

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sub, y_sub, test_size=0.1, random_state=SEED, stratify=y_sub
                )

                # Time and run TabPFN
                start_time = time.time()
                clf = TabPFNClassifier(
                    device=DEVICE,
                    random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf.fit(X_train, y_train)
                proba = clf.predict_proba(X_test)

                # Calculate AUC
                if n_classes == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")

                elapsed = time.time() - start_time

                # Get GPU memory usage (if CUDA)
                gpu_mem = None
                if DEVICE == "cuda":
                    import torch
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9  # GB

                result = {
                    "dataset": name,
                    "n_rows": X_train.shape[0],
                    "n_features": X_train.shape[1],
                    "n_classes": n_classes,
                    "auc": float(auc),
                    "elapsed_sec": float(elapsed),
                    "gpu_memory_gb": float(gpu_mem) if gpu_mem else None,
                    "status": "OK",
                }
                results.append(result)
                print(f"AUC={auc:.4f}, time={elapsed:.1f}s", end="")
                if gpu_mem:
                    print(f", mem={gpu_mem:.1f}GB", end="")
                print()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at n={n_test:,}")
                    result = {
                        "dataset": name,
                        "n_rows": n_test,
                        "status": "OOM",
                        "error": "GPU out of memory",
                    }
                    results.append(result)
                    break  # Stop testing larger sizes for this dataset
                else:
                    print(f"ERROR: {e}")
                    result = {
                        "dataset": name,
                        "n_rows": n_test,
                        "status": "ERROR",
                        "error": str(e),
                    }
                    results.append(result)
                    break

            except Exception as e:
                print(f"FAILED: {e}")
                result = {
                    "dataset": name,
                    "n_rows": n_test,
                    "status": "FAILED",
                    "error": str(e),
                }
                results.append(result)
                break

        # Save intermediate results
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*80}")
    print(f"Results saved to: {out_path}")
    print(f"Total results: {len(results)}")

    # Summary
    successful = [r for r in results if r.get("status") == "OK"]
    oom = [r for r in results if r.get("status") == "OOM"]

    print(f"\nSummary:")
    print(f"  Successful runs: {len(successful)}")
    print(f"  OOM errors: {len(oom)}")

    if oom:
        print(f"\nOOM occurred at:")
        for r in oom:
            print(f"  {r['dataset']}: n={r['n_rows']:,}")


if __name__ == "__main__":
    run_extended_scaling()
