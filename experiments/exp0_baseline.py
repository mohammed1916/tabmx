"""
Experiment 0: Baseline Verification

Objective: Verify TabPFN reproduces reasonable published performance.

Records: accuracy (ROC AUC), runtime, GPU memory, dataset, seed.
Writes results as JSON to results/exp0_baseline.json (append-only via timestamped file).

Do NOT modify multiple variables. This experiment only measures default TabPFN.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

# Use the non-gated V2 model (the Nature-paper TabPFN v2). V2.5/V2.6/V3 are
# gated and require interactive browser license acceptance, which cannot be
# completed in a non-interactive subprocess. Must be set before importing tabpfn.
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Fixed parameters recorded for reproducibility
SEEDS = [42, 123, 456]
TEST_SIZE = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small, well-known OpenML datasets with <=10 classes, <=10K samples.
# (openml_id, name)
DATASETS = [
    (31, "credit-g"),        # 1000 samples, binary
    (37, "diabetes"),        # 768 samples, binary
    (54, "vehicle"),         # 846 samples, 4 classes
]


def gpu_mem_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_mem() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def evaluate_one(openml_id: int, name: str, seed: int) -> dict:
    record: dict = {
        "experiment": "exp0_baseline",
        "dataset": name,
        "openml_id": openml_id,
        "seed": seed,
        "device": DEVICE,
        "method": "TabPFNClassifier(default)",
        "model_version": os.environ.get("TABPFN_MODEL_VERSION", "v2"),
    }
    try:
        X, y = fetch_openml(
            data_id=openml_id, as_frame=False, return_X_y=True, parser="auto"
        )
        # Encode labels to integers
        y = LabelEncoder().fit_transform(y)
        n_classes = len(np.unique(y))
        record["n_samples"] = int(X.shape[0])
        record["n_features"] = int(X.shape[1])
        record["n_classes"] = int(n_classes)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        record["n_train"] = int(X_train.shape[0])
        record["n_test"] = int(X_test.shape[0])

        reset_gpu_mem()
        clf = TabPFNClassifier(device=DEVICE, random_state=seed)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        runtime = time.perf_counter() - t0

        if n_classes == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro"
            )

        record["roc_auc"] = float(auc)
        record["runtime_sec"] = float(runtime)
        record["gpu_mem_mb"] = float(gpu_mem_mb())
        record["status"] = "OK"
    except Exception as e:  # noqa: BLE001 - record failures carefully
        record["status"] = "ERROR"
        record["error"] = repr(e)
        record["traceback"] = traceback.format_exc()
    return record


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp0_baseline_{stamp}.json"
    results: list[dict] = []

    print(f"Experiment 0: Baseline Verification")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {[d[1] for d in DATASETS]}")
    print("-" * 60)

    for openml_id, name in DATASETS:
        for seed in SEEDS:
            print(f"Running {name} (id={openml_id}), seed={seed} ... ", end="", flush=True)
            rec = evaluate_one(openml_id, name, seed)
            results.append(rec)
            if rec["status"] == "OK":
                print(
                    f"AUC={rec['roc_auc']:.4f}  "
                    f"time={rec['runtime_sec']:.2f}s  "
                    f"mem={rec['gpu_mem_mb']:.0f}MB"
                )
            else:
                print(f"ERROR: {rec['error']}")
            # Save incrementally (append-only, never lose data)
            out_path.write_text(json.dumps(results, indent=2))

    print("-" * 60)
    print(f"Results written to: {out_path}")

    # Summary per dataset (mean +/- std across seeds)
    print("\nSummary (mean +/- std across seeds):")
    for _, name in DATASETS:
        aucs = [r["roc_auc"] for r in results if r["dataset"] == name and r["status"] == "OK"]
        times = [r["runtime_sec"] for r in results if r["dataset"] == name and r["status"] == "OK"]
        if aucs:
            print(
                f"  {name:12s}: AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  "
                f"time={np.mean(times):.2f}s"
            )
        else:
            print(f"  {name:12s}: ALL FAILED")


if __name__ == "__main__":
    main()
