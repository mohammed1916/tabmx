"""
Experiment 11: Feature Importance-Based Prototype Selection

Objective: Select prototypes that maximize feature diversity (samples with diverse
feature values) rather than outcome-space uncertainty or simple random sampling.

Hypothesis: Prototypes with high feature variance beat random by 2-3% by providing
better coverage of the feature space.

Method: Compute feature importance via variance; select k samples with highest
weighted feature variance score.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
K = 512

TEST_DATASETS = [
    (151, "electricity"),
    (1461, "bank-marketing"),
]


def select_by_feature_variance(X_train, k, seed):
    """Select k samples that maximize feature diversity."""
    # Standardize features
    X_scaled = StandardScaler().fit_transform(np.nan_to_num(X_train))

    # Compute per-sample "importance" as sum of squared feature deviations from mean
    # High-variance samples (far from centroid across multiple features) score high
    centroid = X_scaled.mean(axis=0)
    sample_importance = np.sqrt(np.sum((X_scaled - centroid) ** 2, axis=1))

    # Select top-k high-importance samples
    top_k_idx = np.argsort(-sample_importance)[:k]
    return top_k_idx


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp11_importance_prototypes_{stamp}.json"
    results = []

    print("Experiment 11: Feature Importance-Based Prototype Selection")
    print(f"Device: {DEVICE} | k={K}")
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

        # Baseline: random prototype selection
        rng = np.random.RandomState(SEED)
        random_idx = rng.choice(X_train.shape[0], size=min(K, X_train.shape[0]), replace=False)
        Xk_random, yk_random = X_train[random_idx], y_train[random_idx]

        try:
            clf_random = TabPFNClassifier(
                device=DEVICE, random_state=SEED,
                ignore_pretraining_limits=True,
                categorical_features_indices=cat_idx or None,
            )
            clf_random.fit(Xk_random, yk_random)
            proba_random = clf_random.predict_proba(X_test)

            auc_random = (roc_auc_score(y_test, proba_random[:, 1]) if n_classes == 2
                          else roc_auc_score(y_test, proba_random, multi_class="ovr", average="macro"))
            print(f"  Random k={K}            : AUC={auc_random:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  Random selection FAILED: {repr(e)[:60]}")
            auc_random = None

        # Feature importance-based selection
        try:
            importance_idx = select_by_feature_variance(X_train, K, SEED)
            Xk_importance, yk_importance = X_train[importance_idx], y_train[importance_idx]

            if len(np.unique(yk_importance)) >= 2:
                clf_importance = TabPFNClassifier(
                    device=DEVICE, random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf_importance.fit(Xk_importance, yk_importance)
                proba_importance = clf_importance.predict_proba(X_test)

                auc_importance = (roc_auc_score(y_test, proba_importance[:, 1]) if n_classes == 2
                                 else roc_auc_score(y_test, proba_importance, multi_class="ovr", average="macro"))

                improvement = (auc_importance - auc_random) / auc_random * 100 if auc_random else 0
                print(f"  Feature Importance k={K}: AUC={auc_importance:.4f} | delta={improvement:+.1f}%")

                results.append({
                    "dataset": name,
                    "method_random": "random",
                    "method_importance": "feature_variance",
                    "k": K,
                    "auc_random": float(auc_random) if auc_random else None,
                    "auc_importance": float(auc_importance),
                    "improvement": float(improvement),
                    "status": "OK",
                })
            else:
                print(f"  Feature importance selection: not enough classes")
                results.append({
                    "dataset": name,
                    "status": "ERROR",
                    "error": "insufficient classes in importance-selected subset",
                })
        except Exception as e:  # noqa: BLE001
            print(f"  Feature importance selection FAILED: {repr(e)[:60]}")
            results.append({
                "dataset": name,
                "status": "ERROR",
                "error": repr(e)[:200],
            })

        out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
