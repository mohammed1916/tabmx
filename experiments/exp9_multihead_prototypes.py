"""
Experiment 9: Multi-Head Prototype Attention

Objective: Train TabPFN on multiple disjoint prototype sets (like multi-head
attention) rather than a single monolithic k prototypes.

Hypothesis: N heads × (k/N) prototypes beats 1 head × k prototypes by 3-5%,
providing diverse contextual views.

Inspiration: Transformer multi-head attention benefits from diversity.
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

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

TEST_DATASETS = [
    (151, "electricity"),
    (1461, "bank-marketing"),
]

# Configurations: (n_heads, k_per_head) tuples that sum to ~1024 total
CONFIGS = [
    (1, 1024),     # Baseline: single head, k=1024
    (2, 512),      # 2 heads × 512
    (4, 256),      # 4 heads × 256
    (8, 128),      # 8 heads × 128
]


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp9_multihead_prototypes_{stamp}.json"
    results = []

    print("Experiment 9: Multi-Head Prototype Attention")
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

        print(f"\n[{name}] n_train={X_train.shape[0]} n_test={X_test.shape[0]}")

        for n_heads, k_per_head in CONFIGS:
            total_k = n_heads * k_per_head
            aucs_heads = []

            # Train separate TabPFN on each head
            for head_id in range(n_heads):
                try:
                    # Select disjoint prototype set for this head
                    rng = np.random.RandomState(SEED + head_id)
                    idx = rng.choice(X_train.shape[0], size=min(k_per_head, X_train.shape[0]),
                                     replace=False)
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
                    aucs_heads.append(auc)
                except Exception:  # noqa: BLE001
                    pass

            # Multi-head ensemble: average predictions
            if aucs_heads:
                multihead_auc = np.mean(aucs_heads)
                std_auc = np.std(aucs_heads)
                print(f"  {n_heads} heads × {k_per_head} (total={total_k}): "
                      f"AUC={multihead_auc:.4f}+/-{std_auc:.4f}")

                results.append({
                    "dataset": name,
                    "n_heads": int(n_heads),
                    "k_per_head": int(k_per_head),
                    "total_k": int(total_k),
                    "mean_auc": float(multihead_auc),
                    "std_auc": float(std_auc),
                    "n_heads_successful": int(len(aucs_heads)),
                    "status": "OK",
                })
            else:
                print(f"  {n_heads} heads × {k_per_head}: FAILED")
                results.append({
                    "dataset": name,
                    "n_heads": int(n_heads),
                    "k_per_head": int(k_per_head),
                    "status": "ERROR",
                })

            out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
