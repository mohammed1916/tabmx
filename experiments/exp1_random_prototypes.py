"""
Experiment 1: Random Prototype Sampling

Objective: Determine whether TabPFN actually needs all training samples, or
whether a randomly sampled subset of k prototypes preserves accuracy.

Method (the whole idea): subsample the training set to k rows, fit TabPFN on
those k rows only, predict the full test set. Compare to fitting on all rows.

One variable changes: the number of prototypes k (plus seed for variance).
Records: dataset, k, seed, ROC AUC, runtime, GPU memory. Append-only JSON.

Model: v3 (TABPFN_MODEL_VERSION=v3). License accepted; token cached.
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

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]  # 5 seeds per spec
K_VALUES = [64, 128, 256, 512, 1024, 2048, None]  # None = full training set
TEST_SIZE = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Datasets: mix of sizes/classes, all <=10 classes. Larger ones let us test
# k << n. (id, name).
DATASETS = [
    (31, "credit-g"),       # 1000 x 20, binary
    (37, "diabetes"),       # 768 x 8, binary
    (54, "vehicle"),        # 846 x 18, 4-class
    (1461, "bank-marketing"),  # ~45K x 16, binary  -> big, k << n
    (1590, "adult"),        # ~48K x 14, binary  -> big, k << n
]


def gpu_mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0


def reset_gpu_mem() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def evaluate(openml_id, name, k, seed, X_train, X_test, y_train, y_test,
             n_classes, cat_idx):
    rec = {
        "experiment": "exp1_random_prototypes",
        "dataset": name,
        "openml_id": openml_id,
        "k": k if k is not None else "full",
        "seed": seed,
        "device": DEVICE,
        "model_version": os.environ.get("TABPFN_MODEL_VERSION", "v3"),
        "n_train_full": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(n_classes),
    }
    try:
        # Select prototypes (random subsample of the training set).
        if k is None or k >= X_train.shape[0]:
            Xk, yk = X_train, y_train
            rec["k_effective"] = int(X_train.shape[0])
        else:
            rng = np.random.RandomState(seed)
            idx = rng.choice(X_train.shape[0], size=k, replace=False)
            Xk, yk = X_train[idx], y_train[idx]
            rec["k_effective"] = int(k)

        # Guard: need >=2 classes present in the prototype set.
        if len(np.unique(yk)) < 2:
            rec["status"] = "SKIP"
            rec["error"] = "prototype set has <2 classes"
            return rec

        reset_gpu_mem()
        clf = TabPFNClassifier(
            device=DEVICE,
            random_state=seed,
            ignore_pretraining_limits=True,  # allow full-set fit on big datasets
            categorical_features_indices=cat_idx if cat_idx else None,
        )
        t0 = time.perf_counter()
        clf.fit(Xk, yk)
        proba = clf.predict_proba(X_test)
        runtime = time.perf_counter() - t0

        if n_classes == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")

        rec["roc_auc"] = float(auc)
        rec["runtime_sec"] = float(runtime)
        rec["gpu_mem_mb"] = float(gpu_mem_mb())
        rec["status"] = "OK"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "ERROR"
        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()
    return rec


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp1_random_prototypes_{stamp}.json"
    results: list[dict] = []

    print("Experiment 1: Random Prototype Sampling")
    print(f"Device: {DEVICE} | Model: {os.environ.get('TABPFN_MODEL_VERSION')}")
    print(f"K values: {K_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {[d[1] for d in DATASETS]}")
    print("-" * 70)

    for openml_id, name in DATASETS:
        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            results.append(
                {
                    "experiment": "exp1_random_prototypes",
                    "dataset": name,
                    "openml_id": openml_id,
                    "status": "LOAD_ERROR",
                    "error": repr(e),
                }
            )
            out_path.write_text(json.dumps(results, indent=2))
            continue

        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=0, stratify=y
        )
        print(
            f"\n[{name}] n_train={X_train.shape[0]} n_test={X_test.shape[0]} "
            f"feats={X_train.shape[1]} classes={n_classes} cat={len(cat_idx)}"
        )

        for k in K_VALUES:
            # For k=full on very large datasets we still ATTEMPT the fit once
            # (seed 42 only) to capture full-fit feasibility/accuracy — the core
            # scalability evidence. OOM/errors are recorded, not fatal.
            seeds = SEEDS
            if k is None and X_train.shape[0] > 15000:
                seeds = [42]
                print(f"  k=full : attempting single full-fit "
                      f"(n_train={X_train.shape[0]}, seed=42 only)")

            aucs = []
            for seed in seeds:
                rec = evaluate(
                    openml_id, name, k, seed,
                    X_train, X_test, y_train, y_test, n_classes, cat_idx,
                )
                results.append(rec)
                out_path.write_text(json.dumps(results, indent=2))
                if rec["status"] == "OK":
                    aucs.append(rec["roc_auc"])
                elif rec["status"] == "ERROR" and k is None:
                    print(f"    full-fit ERROR: {rec['error'][:80]}")
            k_label = "full" if k is None else str(k)
            if aucs:
                print(
                    f"  k={k_label:>5}: AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f} "
                    f"(n={len(aucs)} seeds)"
                )
            else:
                print(f"  k={k_label:>5}: no successful runs")

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
