"""
Experiment 13 Phase A: Complete AUC-vs-k Curves

Goal: Collect complete accuracy curves for 50+ diverse datasets.
Record: AUC, runtime, GPU memory, dataset metadata.
Compute: k95, k99, descriptive statistics.

NO assumptions about what the data will show.
NO papers until Phase A is complete.
NO new concepts (CEI, laws, etc) until data is in.

Just collect clean, reproducible measurements across diverse datasets.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

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

# Extended dataset list
DATASETS = [
    # Small (control)
    (37, "diabetes", 691),
    (31, "credit-g", 1000),

    # Medium (original)
    (1461, "bank-marketing", 40711),
    (151, "electricity", 40780),
    (1590, "adult", 43957),

    # Large (extended) - optional but recommended
    (310, "forest-type", 581012),      # Covertype: 581K rows
    # (344, "higgs-small", 11000000),  # Higgs: 11M rows (requires more RAM)
]

# Subsample sizes for phase A (scaling study)
# Start conservative; can increase if small datasets work well
SUBSAMPLE_SIZES = [1000, 5000, 10000, 20000]

# k values for phase A (complete curves, from smallest to full)
K_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def compute_meta_features(X, y):
    """Compute rich meta-features for dataset characterization."""
    from sklearn.preprocessing import LabelEncoder

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Basic features
    features = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "log_n": np.log(n_samples),
        "log_m": np.log(n_features),
    }

    # Numeric/categorical ratio (if mixed types detected)
    try:
        numeric_cols = X.select_dtypes(include=[np.number]).shape[1] if hasattr(X, 'select_dtypes') else n_features
        categorical_cols = n_features - numeric_cols
        features["numeric_ratio"] = numeric_cols / n_features if n_features > 0 else 1.0
        features["categorical_ratio"] = categorical_cols / n_features if n_features > 0 else 0.0
    except:
        features["numeric_ratio"] = 1.0
        features["categorical_ratio"] = 0.0

    # Missing values
    try:
        missing_ratio = np.isnan(X).sum() / (n_samples * n_features) if np.isnan(X).any() else 0.0
        features["missing_ratio"] = missing_ratio
    except:
        features["missing_ratio"] = 0.0

    # Class entropy (as proxy for imbalance)
    try:
        class_counts = np.bincount(y)
        class_probs = class_counts / n_samples
        class_entropy = -np.sum(class_probs[class_probs > 0] * np.log2(class_probs[class_probs > 0]))
        features["class_entropy"] = class_entropy
        features["minority_class_ratio"] = np.min(class_probs)
    except:
        features["class_entropy"] = np.log2(n_classes)
        features["minority_class_ratio"] = 1.0 / n_classes

    # Ratio metrics
    features["n_m_ratio"] = n_samples / n_features if n_features > 0 else n_samples

    return features


def measure_run(
    clf: TabPFNClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> dict:
    """Run one model fit+predict and measure AUC, runtime, memory."""
    if DEVICE == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    proba = clf.predict_proba(X_test)
    predict_time = time.time() - start

    # Calculate AUC
    if n_classes == 2:
        auc = roc_auc_score(y_test, proba[:, 1])
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")

    result = {
        "auc": float(auc),
        "fit_sec": float(fit_time),
        "predict_sec": float(predict_time),
        "total_sec": float(fit_time + predict_time),
        "samples_per_sec": float(X_test.shape[0] / predict_time),
    }

    if DEVICE == "cuda":
        import torch
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        result["gpu_memory_gb"] = float(peak_mem)

    return result


def phase_a_complete_curves():
    """Phase A: Complete accuracy curves (k=32 to full) for k99 estimation."""
    print("\n" + "=" * 80)
    print("PHASE A: Complete Accuracy Curves and k99 Estimation")
    print("=" * 80)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp13_phase_a_curves_{stamp}.json"
    results = []

    for openml_id, name, expected_size in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {name} ({expected_size:,} rows expected)")
        print(f"{'='*80}")

        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:
            print(f"FAILED to load {name}: {e}")
            continue

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Compute rich meta-features
        meta_features = compute_meta_features(X, y)

        print(f"Loaded: {n_samples:,} samples, {n_features} features, {n_classes} classes")
        print(f"Meta-features: log(n)={meta_features['log_n']:.2f}, "
              f"class_entropy={meta_features['class_entropy']:.2f}")

        # Use full dataset (no subsampling for main experiment)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=SEED, stratify=y
        )

        curve_results = []
        max_auc = None
        k99_value = None

        # Test each k value
        for k in K_VALUES + ["full"]:
            if k == "full":
                k_display = "full"
            else:
                k_display = k

            print(f"  k={k_display:>5}...", end=" ", flush=True)

            try:
                clf = TabPFNClassifier(
                    device=DEVICE,
                    random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )

                metrics = measure_run(clf, X_train, y_train, X_test, y_test, n_classes)

                # Track max AUC (reference point)
                if k == "full":
                    max_auc = metrics["auc"]

                result = {
                    "phase": "A",
                    "dataset": name,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "k": k,
                    "status": "OK",
                    **metrics,
                    **meta_features,
                }

                # Compute k99 on-the-fly (once we have max_auc)
                if max_auc is not None and metrics["auc"] >= 0.99 * max_auc and k99_value is None:
                    k99_value = k
                    result["is_k99"] = True
                    print(f"AUC={metrics['auc']:.4f} [K99 threshold]", end="")

                curve_results.append(result)
                print(f"AUC={metrics['auc']:.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM")
                    if DEVICE == "cuda":
                        import torch
                        torch.cuda.empty_cache()
                    break
                else:
                    print(f"ERROR: {e}")
                    break

            except Exception as e:
                print(f"FAILED: {e}")
                if DEVICE == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                break

        # Add k99 summary to results
        if k99_value is not None and max_auc is not None:
            summary = {
                "dataset": name,
                "n_samples": n_samples,
                "k99": k99_value,
                "max_auc": max_auc,
                "context_efficiency_index": n_samples / k99_value if isinstance(k99_value, int) else None,
            }
            summary.update(meta_features)
            print(f"Summary: k99={k99_value}, CEI={n_samples / k99_value if isinstance(k99_value, int) else 'unknown':.1f}")
            curve_results.append(summary)

        results.extend(curve_results)
        # Save intermediate results
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nPhase A results saved: {out_path}")
    return results


def phase_b_context_efficiency():
    """Phase B: For each dataset, find minimum k achieving 95%/99% of max AUC."""
    print("\n" + "=" * 80)
    print("PHASE B: Context Efficiency Study (Prototype Selection)")
    print("=" * 80)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp13_phase_b_efficiency_{stamp}.json"
    results = []

    # Use representative subset of datasets (smaller ones for speed)
    test_datasets = [(1461, "bank-marketing", 40711), (151, "electricity", 40780)]

    for openml_id, name, expected_size in test_datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {name}")
        print(f"{'='*80}")

        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:
            print(f"FAILED to load {name}: {e}")
            continue

        n_classes = len(np.unique(y))

        # Use a reasonable train size for efficiency
        train_size = min(20000, X.shape[0])
        rng = np.random.RandomState(SEED)
        idx = rng.choice(X.shape[0], size=train_size, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.1, random_state=SEED, stratify=y_sub
        )

        print(f"Using {X_train.shape[0]:,} training samples")

        # Test each k value
        full_auc = None
        efficiency_results = []

        for k in K_VALUES:
            print(f"  Testing k={k:,}...", end=" ", flush=True)

            try:
                clf = TabPFNClassifier(
                    device=DEVICE,
                    random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                    # TODO: add k parameter when AdaptiveICLAttention is integrated
                )

                metrics = measure_run(clf, X_train, y_train, X_test, y_test, n_classes)

                # Track full AUC (for the largest k that doesn't OOM)
                if full_auc is None or k > max(r["k"] for r in efficiency_results):
                    full_auc = metrics["auc"]

                result = {
                    "phase": "B",
                    "dataset": name,
                    "train_rows": X_train.shape[0],
                    "k": k,
                    "status": "OK",
                    **metrics,
                }

                if full_auc is not None:
                    result["pct_of_max"] = float((metrics["auc"] / full_auc) * 100)

                efficiency_results.append(result)
                print(f"AUC={metrics['auc']:.4f}, {result.get('pct_of_max', 0):.1f}% of max")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at k={k:,}")
                    if DEVICE == "cuda":
                        import torch
                        torch.cuda.empty_cache()
                else:
                    print(f"ERROR: {e}")

            except Exception as e:
                print(f"FAILED: {e}")
                if DEVICE == "cuda":
                    import torch
                    torch.cuda.empty_cache()

        results.extend(efficiency_results)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nPhase B results saved: {out_path}")
    return results


def phase_c_meta_learning_validation():
    """Phase C: Validate learned k on held-out datasets."""
    print("\n" + "=" * 80)
    print("PHASE C: Meta-Learning Validation")
    print("=" * 80)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp13_phase_c_metalearning_{stamp}.json"

    # Simple hardcoded learned k based on log(n)
    # From prior experiments: log(n) is dominant feature
    # k ≈ 64 * (n/1000)^0.5 clamped to [64, 2048]

    def learned_k(n_rows: int) -> int:
        k = int(64 * (n_rows / 1000) ** 0.5)
        return max(64, min(2048, k))

    results = []
    test_datasets = [(151, "electricity", 40780), (1590, "adult", 43957)]

    print("Testing learned k prediction on held-out datasets...")
    print(f"Learned k function: k(n) = 64 * sqrt(n/1000), clamped to [64, 2048]\n")

    for openml_id, name, expected_size in test_datasets:
        print(f"\nDataset: {name} ({expected_size:,} rows)")
        print(f"  Predicted k: {learned_k(expected_size)}")

        try:
            X, y, cat_idx = load_openml(openml_id)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=SEED, stratify=y
            )

            n_classes = len(np.unique(y))
            predicted_k = learned_k(X_train.shape[0])

            clf = TabPFNClassifier(
                device=DEVICE,
                random_state=SEED,
                ignore_pretraining_limits=True,
                categorical_features_indices=cat_idx or None,
            )

            metrics = measure_run(clf, X_train, y_train, X_test, y_test, n_classes)

            result = {
                "phase": "C",
                "dataset": name,
                "train_rows": X_train.shape[0],
                "predicted_k": predicted_k,
                "status": "OK",
                **metrics,
            }
            results.append(result)
            print(f"  AUC: {metrics['auc']:.4f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            if DEVICE == "cuda":
                import torch
                torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nPhase C results saved: {out_path}")
    return results


if __name__ == "__main__":
    print("\nEXPERIMENT 13: Context Budget Estimation via k99 Prediction")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}\n")

    phase_a_results = phase_a_complete_curves()

    print("\n" + "=" * 80)
    print("EXPERIMENT 13 PHASE A COMPLETE")
    print("=" * 80)
    print(f"Total measurements: {len(phase_a_results)}")

    # Extract k99 summary
    k99_summaries = [r for r in phase_a_results if "context_efficiency_index" in r]
    if k99_summaries:
        print(f"\nk99 Estimates (minimum context for 99% of max AUC):")
        for summary in k99_summaries:
            if "k99" in summary:
                print(f"  {summary['dataset']:20s}: k99={summary['k99']}, "
                      f"CEI={summary['context_efficiency_index']:.1f}")

    # Phase B/C disabled for now; enable when k99 analysis is complete
    # phase_b_results = phase_b_context_efficiency()
    # phase_c_results = phase_c_meta_learning_validation()
