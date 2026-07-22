"""
Experiment 10: Uncertainty-Weighted Prototype Selection

Objective: Prioritize hard/ambiguous samples (high uncertainty) in prototype
selection, rather than random or diversity-based.

Hypothesis: Prototypes selected by uncertainty (hard examples) beat random
selection by 2-4%, as the model learns from edge cases.

Method: Two-pass approach:
  1. First TabPFN fit on random k prototypes (baseline)
  2. Identify highest-entropy predictions on training set
  3. Select top-k uncertain samples as new prototypes
  4. Re-fit TabPFN on uncertainty-selected prototypes
  5. Compare new AUC vs baseline
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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


def select_by_entropy(X_train, y_train, proba, k, seed):
    """Select k samples with highest entropy (highest uncertainty)."""
    # Entropy of predicted probabilities
    entropies = np.array([entropy(p) for p in proba])
    top_k_idx = np.argsort(-entropies)[:k]
    return top_k_idx


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp10_uncertainty_prototypes_{stamp}.json"
    results = []

    print("Experiment 10: Uncertainty-Weighted Prototype Selection")
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
            print(f"  Random k={K}        : AUC={auc_random:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  Random selection FAILED: {repr(e)[:60]}")
            auc_random = None

        # Second pass: use uncertainty to refine prototypes
        try:
            # Get predictions on full training set (this is expensive, use subsample)
            subsample_size = min(5000, X_train.shape[0])
            subsample_idx = rng.choice(X_train.shape[0], size=subsample_size, replace=False)
            X_subsample, y_subsample = X_train[subsample_idx], y_train[subsample_idx]

            # Re-use random clf to score subsample
            proba_subsample = clf_random.predict_proba(X_subsample)

            # Select high-entropy samples from subsample
            uncertainty_idx_subsample = select_by_entropy(X_subsample, y_subsample, proba_subsample, K, SEED)
            uncertainty_idx = subsample_idx[uncertainty_idx_subsample]

            Xk_uncertain, yk_uncertain = X_train[uncertainty_idx], y_train[uncertainty_idx]

            if len(np.unique(yk_uncertain)) >= 2:
                clf_uncertain = TabPFNClassifier(
                    device=DEVICE, random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf_uncertain.fit(Xk_uncertain, yk_uncertain)
                proba_uncertain = clf_uncertain.predict_proba(X_test)

                auc_uncertain = (roc_auc_score(y_test, proba_uncertain[:, 1]) if n_classes == 2
                                else roc_auc_score(y_test, proba_uncertain, multi_class="ovr", average="macro"))

                improvement = (auc_uncertain - auc_random) / auc_random * 100 if auc_random else 0
                print(f"  Uncertainty k={K}   : AUC={auc_uncertain:.4f} | delta={improvement:+.1f}%")

                results.append({
                    "dataset": name,
                    "method_random": "random",
                    "method_uncertainty": "entropy",
                    "k": K,
                    "auc_random": float(auc_random) if auc_random else None,
                    "auc_uncertain": float(auc_uncertain),
                    "improvement": float(improvement),
                    "status": "OK",
                })
            else:
                print(f"  Uncertainty selection: not enough classes in subsample")
                results.append({
                    "dataset": name,
                    "method_uncertainty": "entropy",
                    "status": "ERROR",
                    "error": "insufficient classes in uncertainty subsample",
                })
        except Exception as e:  # noqa: BLE001
            print(f"  Uncertainty selection FAILED: {repr(e)[:60]}")
            results.append({
                "dataset": name,
                "method_uncertainty": "entropy",
                "status": "ERROR",
                "error": repr(e)[:200],
            })

        out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
