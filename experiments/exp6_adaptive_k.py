"""
Experiment 6: Adaptive k via Learned Dataset Properties

Objective: Learn to predict optimal prototype count k from dataset features,
rather than using fixed k=512 for all datasets.

Phase 1: Measure optimal k on training datasets (Exp 1 + Exp 3 datasets).
Phase 2: Fit a regressor: dataset_features -> optimal_k.
Phase 3: Test on held-out datasets; compare learned_k vs fixed_k=512.

Hypothesis: Learned k beats fixed k=512 by 3-5% on average.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_CANDIDATES = [64, 128, 256, 512, 1024, 2048]
SEED = 42

# Datasets to train the adaptive-k model (use Exp 1 + some from Exp 3).
TRAIN_DATASETS = [
    (37, "diabetes", 691),         # small
    (54, "vehicle", 761),          # small
    (31, "credit-g", 1000),        # small
    (1461, "bank-marketing", 45211),  # large
    (1590, "adult", 48842),        # large
]

# Held-out datasets for testing (new OpenML datasets, not in train set).
TEST_DATASETS = [
    (151, "electricity"),
    (293, "adult"),
]


def compute_dataset_features(X, y, n_train, n_test):
    """Compute predictive features from a dataset."""
    n, m = X.shape
    n_classes = len(np.unique(y))
    sparsity = np.isnan(X).sum() / (n * m) if X.size > 0 else 0.0

    return {
        "log_n": np.log1p(n_train),
        "log_m": np.log1p(m),
        "n_classes": n_classes,
        "class_density": n_classes / n_train,
        "n_m_ratio": n_train / m if m > 0 else 1.0,
        "sparsity": float(sparsity),
        "n_test": n_test,
    }


def find_optimal_k(X_train, X_val, y_train, y_val, cat_idx, n_classes):
    """Grid search over k; return k with highest val AUC."""
    best_k = 512
    best_auc = 0.0

    for k in K_CANDIDATES:
        try:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(X_train.shape[0], size=min(k, X_train.shape[0]), replace=False)
            Xk, yk = X_train[idx], y_train[idx]

            if len(np.unique(yk)) < 2:
                continue

            clf = TabPFNClassifier(
                device=DEVICE, random_state=SEED,
                ignore_pretraining_limits=True,
                categorical_features_indices=cat_idx or None,
            )
            clf.fit(Xk, yk)
            proba = clf.predict_proba(X_val)

            if n_classes == 2:
                auc = roc_auc_score(y_val, proba[:, 1])
            else:
                auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")

            if auc > best_auc:
                best_auc = auc
                best_k = k
        except Exception:  # noqa: BLE001
            pass

    return best_k, best_auc


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp6_adaptive_k_{stamp}.json"
    results = []

    print("Experiment 6: Adaptive k via Learned Dataset Properties")
    print(f"Device: {DEVICE}")
    print("-" * 70)

    # Phase 1: Train the adaptive-k model
    print("\nPhase 1: Learning optimal k per training dataset")
    print("-" * 70)

    X_features = []
    y_optimal_k = []
    dataset_names = []

    for openml_id, name, _ in TRAIN_DATASETS:
        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue

        n_classes = len(np.unique(y))
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, test_size=0.20, random_state=0, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.50, random_state=0, stratify=y_val_test
        )

        # Find optimal k on this dataset
        optimal_k, opt_auc = find_optimal_k(X_train, X_val, y_train, y_val, cat_idx, n_classes)
        print(f"[{name:16s}] optimal_k={optimal_k:>4} (AUC={opt_auc:.4f})")

        # Compute dataset features
        features = compute_dataset_features(X_train, y_train, X_train.shape[0], X_test.shape[0])
        X_features.append(list(features.values()))
        y_optimal_k.append(optimal_k)
        dataset_names.append(name)

        # Store for later reference
        results.append({
            "phase": "train",
            "dataset": name,
            "optimal_k": int(optimal_k),
            "val_auc": float(opt_auc),
            "features": features,
        })
        out_path.write_text(json.dumps(results, indent=2))

    if len(X_features) < 2:
        print("Not enough training datasets; aborting.")
        return

    # Phase 1b: Fit the regressor
    print("\nPhase 1b: Fitting adaptive-k regressor")
    X_feat_arr = np.array(X_features)
    y_k_arr = np.array(y_optimal_k)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat_arr)

    regressor = LinearRegression()
    regressor.fit(X_scaled, y_k_arr)

    print(f"Regressor R²: {regressor.score(X_scaled, y_k_arr):.4f}")
    print(f"Coefficients: {dict(zip(list(compute_dataset_features(np.array([[0]]), np.array([0]), 1, 1).keys()), regressor.coef_))}")

    # Phase 2: Test on held-out datasets
    print("\nPhase 2: Testing on held-out datasets")
    print("-" * 70)

    for openml_id, name in TEST_DATASETS:
        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue

        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=0, stratify=y
        )

        # Compute dataset features
        features = compute_dataset_features(X_train, y_train, X_train.shape[0], X_test.shape[0])
        X_feat_scaled = scaler.transform([list(features.values())])

        # Predict k
        predicted_k_float = float(regressor.predict(X_feat_scaled)[0])
        predicted_k = int(np.clip(np.round(predicted_k_float), 64, 2048))

        # Evaluate three k values: predicted, fixed=512, and a grid for comparison
        results_per_k = {}
        for test_k in [predicted_k, 512, 256, 1024]:
            try:
                rng = np.random.RandomState(SEED)
                idx = rng.choice(X_train.shape[0], size=min(test_k, X_train.shape[0]), replace=False)
                Xk, yk = X_train[idx], y_train[idx]

                if len(np.unique(yk)) < 2:
                    continue

                clf = TabPFNClassifier(
                    device=DEVICE, random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf.fit(Xk, yk)
                proba = clf.predict_proba(X_test)

                if n_classes == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")

                results_per_k[test_k] = auc
            except Exception:  # noqa: BLE001
                results_per_k[test_k] = None

        # Report
        pred_auc = results_per_k.get(predicted_k)
        fixed_auc = results_per_k.get(512)
        pred_vs_fixed = (pred_auc - fixed_auc) / fixed_auc * 100 if fixed_auc and pred_auc else 0

        print(
            f"[{name:16s}] predicted_k={predicted_k:>4} AUC={pred_auc:.4f} | "
            f"fixed_k=512 AUC={fixed_auc:.4f} | delta={pred_vs_fixed:+.1f}%"
        )

        results.append({
            "phase": "test",
            "dataset": name,
            "n_train": int(X_train.shape[0]),
            "predicted_k": int(predicted_k),
            "fixed_k": 512,
            "predicted_auc": float(pred_auc) if pred_auc else None,
            "fixed_auc": float(fixed_auc) if fixed_auc else None,
            "improvement": float(pred_vs_fixed),
            "features": features,
            "k_sweep": {int(k): float(v) if v else None for k, v in results_per_k.items()},
        })
        out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
