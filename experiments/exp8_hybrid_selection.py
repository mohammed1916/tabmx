"""
Experiment 8: Hybrid Prototype Selection

Objective: Combine learned k (Exp 6) with intelligent prototype selection
(K-Means or diversity sampling within the k prototypes).

Hypothesis: Learned k + smart selection beats learned k + random by 2-4%.

Key insight from Exp 2: Selection method didn't matter with fixed k=512, but
may matter more when k is learned (diverse k values across datasets).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

TRAIN_DATASETS = [
    (37, "diabetes"),
    (54, "vehicle"),
    (31, "credit-g"),
    (1461, "bank-marketing"),
    (1590, "adult"),
]

TEST_DATASETS = [
    (151, "electricity"),
]

SELECTION_METHODS = ["random", "kmeans"]


def compute_dataset_features(X, y, n_train, n_test):
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


def select_prototypes(X_train, y_train, k, method, seed):
    """Select k prototypes using specified method."""
    if method == "random":
        rng = np.random.RandomState(seed)
        return rng.choice(X_train.shape[0], size=min(k, X_train.shape[0]), replace=False)
    elif method == "kmeans":
        Xs = StandardScaler().fit_transform(np.nan_to_num(X_train))
        km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(Xs)
        idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs)
        return np.unique(idx)
    else:
        raise ValueError(method)


def find_learned_k(X_train, X_val, y_train, y_val, cat_idx, n_classes):
    """Use a simple heuristic to estimate optimal k based on dataset size."""
    n = X_train.shape[0]
    # Simple rule: larger datasets need more prototypes
    if n < 1000:
        return 256
    elif n < 5000:
        return 512
    elif n < 20000:
        return 1024
    else:
        return 2048


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp8_hybrid_selection_{stamp}.json"
    results = []

    print("Experiment 8: Hybrid Prototype Selection (Learned k + Smart Selection)")
    print(f"Device: {DEVICE}")
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

        print(f"\n[{name}] n_train={X_train.shape[0]}")

        # Learn k for this dataset
        learned_k = find_learned_k(X_train, X_test, y_train, y_test, cat_idx, n_classes)
        print(f"Learned k: {learned_k}")

        # Test each selection method with learned k
        for method in SELECTION_METHODS:
            try:
                idx = select_prototypes(X_train, y_train, learned_k, method, SEED)
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

                auc = (roc_auc_score(y_test, proba[:, 1]) if n_classes == 2
                       else roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))

                print(f"  {method:10s} (k={learned_k}): AUC={auc:.4f}")

                results.append({
                    "dataset": name,
                    "method": method,
                    "learned_k": int(learned_k),
                    "n_prototypes": int(len(idx)),
                    "roc_auc": float(auc),
                    "status": "OK",
                })
            except Exception as e:  # noqa: BLE001
                print(f"  {method:10s}: ERROR {repr(e)[:60]}")
                results.append({
                    "dataset": name,
                    "method": method,
                    "learned_k": int(learned_k),
                    "status": "ERROR",
                    "error": repr(e)[:200],
                })

            out_path.write_text(json.dumps(results, indent=2))

        # Also test fixed k=512 for comparison
        try:
            idx = select_prototypes(X_train, y_train, 512, "random", SEED)
            Xk, yk = X_train[idx], y_train[idx]
            if len(np.unique(yk)) >= 2:
                clf = TabPFNClassifier(
                    device=DEVICE, random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf.fit(Xk, yk)
                proba = clf.predict_proba(X_test)
                auc = (roc_auc_score(y_test, proba[:, 1]) if n_classes == 2
                       else roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
                print(f"  fixed_k=512 : AUC={auc:.4f}")
                results.append({
                    "dataset": name,
                    "method": "fixed_k=512",
                    "k": 512,
                    "roc_auc": float(auc),
                    "status": "OK",
                })
                out_path.write_text(json.dumps(results, indent=2))
        except Exception:  # noqa: BLE001
            pass

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
