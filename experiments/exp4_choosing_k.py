"""
Experiment 4: Choosing k (Pareto frontier)

Objective: Characterize the accuracy / runtime / memory trade-off as a function
of prototype count k, to recommend an operating point.

This reuses the best selection method from Experiment 2 (set via BEST_METHOD).
One variable changes: k. Records accuracy, inference time, GPU memory per k.
Append-only JSON.

Run only after Exp 2 assigns a winning method. Edit BEST_METHOD accordingly.
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
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import pairwise_distances_argmin_min, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tabpfn import TabPFNClassifier

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# EDIT after Exp 2 verdict. Default to random if selection showed no benefit.
BEST_METHOD = os.environ.get("EXP4_METHOD", "random")

SEEDS = [42, 123, 456]
K_VALUES = [64, 128, 256, 512, 1024, 2048]
TEST_SIZE = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = [
    (1461, "bank-marketing"),
    (1590, "adult"),
    (54, "vehicle"),
]


def load_dataset(openml_id):
    X, y = fetch_openml(data_id=openml_id, as_frame=False, return_X_y=True, parser="auto")
    y = LabelEncoder().fit_transform(y)
    return np.asarray(X, dtype=np.float64), y


def select(method, X, y, k, seed):
    if method == "random":
        rng = np.random.RandomState(seed)
        return rng.choice(X.shape[0], size=min(k, X.shape[0]), replace=False)
    if method == "kmeans":
        Xs = StandardScaler().fit_transform(np.nan_to_num(X))
        km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(Xs)
        idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs)
        return np.unique(idx)
    raise ValueError(method)


def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp4_choosing_k_{stamp}.json"
    results = []
    print(f"Experiment 4: Choosing k | method={BEST_METHOD} | device={DEVICE}")
    print("-" * 70)

    for openml_id, name in DATASETS:
        try:
            X, y = load_dataset(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue
        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=0, stratify=y
        )
        print(f"\n[{name}] n_train={X_train.shape[0]} classes={n_classes}")
        for k in K_VALUES:
            if k > X_train.shape[0]:
                continue
            aucs, times, mems = [], [], []
            for seed in SEEDS:
                rec = {
                    "experiment": "exp4_choosing_k", "dataset": name,
                    "openml_id": openml_id, "method": BEST_METHOD, "k": k,
                    "seed": seed, "device": DEVICE,
                    "model_version": os.environ.get("TABPFN_MODEL_VERSION", "v3"),
                    "n_classes": int(n_classes),
                }
                try:
                    idx = select(BEST_METHOD, X_train, y_train, k, seed)
                    Xk, yk = X_train[idx], y_train[idx]
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); continue
                    clf = TabPFNClassifier(device=DEVICE, random_state=seed,
                                           ignore_pretraining_limits=True)
                    t0 = time.perf_counter()
                    clf.fit(Xk, yk)
                    proba = clf.predict_proba(X_test)
                    rt = time.perf_counter() - t0
                    auc = (roc_auc_score(y_test, proba[:, 1]) if n_classes == 2
                           else roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
                    rec.update(roc_auc=float(auc), inference_time_sec=float(rt),
                               gpu_mem_mb=float(gpu_mem_mb()),
                               latency_ms_per_1k=float(rt / max(1, len(y_test)) * 1000 * 1000),
                               status="OK")
                    aucs.append(auc); times.append(rt); mems.append(gpu_mem_mb())
                except Exception as e:  # noqa: BLE001
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                    rec["traceback"] = traceback.format_exc()
                results.append(rec)
                out_path.write_text(json.dumps(results, indent=2))
            if aucs:
                print(f"  k={k:>5}: AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f} "
                      f"time={np.mean(times):.2f}s mem={np.mean(mems):.0f}MB")
    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
