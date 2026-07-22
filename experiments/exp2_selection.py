"""
Experiment 2: Prototype Selection Baselines

Objective: Determine whether an intelligent selection of k prototypes beats
random selection at the same k.

Methods compared (k fixed = 512):
  - random            : uniform subsample (Exp 1 baseline)
  - kmeans            : nearest training point to each KMeans centroid
  - fps               : farthest point sampling (max diversity / coverage)
  - kmeans_balanced   : KMeans within each class, proportional to class size

One variable changes: the selection method. Everything else fixed.
Records: dataset, method, seed, ROC AUC, runtime (selection + inference).
Append-only JSON.
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

SEEDS = [42, 123, 456]
K = 512
TEST_SIZE = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["random", "kmeans", "fps", "kmeans_balanced"]

DATASETS = [
    (31, "credit-g"),
    (37, "diabetes"),
    (54, "vehicle"),
    (1461, "bank-marketing"),
    (1590, "adult"),
]


def load_dataset(openml_id):
    X, y = fetch_openml(
        data_id=openml_id, as_frame=False, return_X_y=True, parser="auto"
    )
    y = LabelEncoder().fit_transform(y)
    X = np.asarray(X, dtype=np.float64)
    return X, y


def select_random(X, y, k, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(X.shape[0], size=k, replace=False)


def select_kmeans(X, y, k, seed):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(Xs)
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs)
    return np.unique(idx)


def select_fps(X, y, k, seed):
    # Farthest point sampling on standardized features.
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    n = Xs.shape[0]
    rng = np.random.RandomState(seed)
    first = rng.randint(n)
    selected = [first]
    dist = np.linalg.norm(Xs - Xs[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(Xs - Xs[nxt], axis=1))
    return np.array(selected)


def select_kmeans_balanced(X, y, k, seed):
    # Allocate prototypes per class proportional to class frequency, then
    # KMeans within each class. Ensures minority-class coverage.
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    classes, counts = np.unique(y, return_counts=True)
    alloc = np.maximum(1, np.round(k * counts / counts.sum()).astype(int))
    idx_all = []
    for c, a in zip(classes, alloc):
        c_idx = np.where(y == c)[0]
        a = min(a, len(c_idx))
        if a <= 1 or len(c_idx) <= a:
            idx_all.extend(c_idx[:a].tolist())
            continue
        km = KMeans(n_clusters=a, random_state=seed, n_init=3).fit(Xs[c_idx])
        sub, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs[c_idx])
        idx_all.extend(c_idx[np.unique(sub)].tolist())
    return np.array(idx_all)


SELECTORS = {
    "random": select_random,
    "kmeans": select_kmeans,
    "fps": select_fps,
    "kmeans_balanced": select_kmeans_balanced,
}


def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp2_selection_{stamp}.json"
    results = []

    print("Experiment 2: Prototype Selection Baselines")
    print(f"Device: {DEVICE} | Model: {os.environ.get('TABPFN_MODEL_VERSION')} | k={K}")
    print(f"Methods: {METHODS} | Seeds: {SEEDS}")
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
        if X_train.shape[0] <= K:
            print(f"[{name}] SKIP: n_train={X_train.shape[0]} <= k={K}")
            continue
        print(f"\n[{name}] n_train={X_train.shape[0]} classes={n_classes}")

        for method in METHODS:
            aucs = []
            for seed in SEEDS:
                rec = {
                    "experiment": "exp2_selection",
                    "dataset": name,
                    "openml_id": openml_id,
                    "method": method,
                    "k": K,
                    "seed": seed,
                    "device": DEVICE,
                    "model_version": os.environ.get("TABPFN_MODEL_VERSION", "v3"),
                    "n_train_full": int(X_train.shape[0]),
                    "n_classes": int(n_classes),
                }
                try:
                    t_sel0 = time.perf_counter()
                    idx = SELECTORS[method](X_train, y_train, K, seed)
                    sel_time = time.perf_counter() - t_sel0
                    Xk, yk = X_train[idx], y_train[idx]
                    rec["k_effective"] = int(len(idx))
                    rec["selection_time_sec"] = float(sel_time)

                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"
                        rec["error"] = "prototype set <2 classes"
                        results.append(rec)
                        out_path.write_text(json.dumps(results, indent=2))
                        continue

                    clf = TabPFNClassifier(
                        device=DEVICE, random_state=seed,
                        ignore_pretraining_limits=True,
                    )
                    t0 = time.perf_counter()
                    clf.fit(Xk, yk)
                    proba = clf.predict_proba(X_test)
                    rec["inference_time_sec"] = float(time.perf_counter() - t0)

                    if n_classes == 2:
                        auc = roc_auc_score(y_test, proba[:, 1])
                    else:
                        auc = roc_auc_score(
                            y_test, proba, multi_class="ovr", average="macro"
                        )
                    rec["roc_auc"] = float(auc)
                    rec["gpu_mem_mb"] = float(gpu_mem_mb())
                    rec["status"] = "OK"
                    aucs.append(auc)
                except Exception as e:  # noqa: BLE001
                    rec["status"] = "ERROR"
                    rec["error"] = repr(e)
                    rec["traceback"] = traceback.format_exc()
                results.append(rec)
                out_path.write_text(json.dumps(results, indent=2))
            if aucs:
                print(f"  {method:16s}: AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")
            else:
                print(f"  {method:16s}: no successful runs")

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
