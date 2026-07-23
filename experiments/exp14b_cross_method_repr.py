"""
Experiment 14b: Cross-method representativeness.

Purpose: strengthen H2. exp14 logged representativeness (class_tv, feat_shift)
only for random draws. Here we log the SAME metrics for random, kmeans, and fps
selections at a fixed k, so we can test whether representativeness predicts AUC
ACROSS selection methods -- i.e. "representative contexts win regardless of the
algorithm that produced them."

For each dataset, at k=128, we generate many contexts per method (multiple
seeds), and for each context record: method, seed, AUC, class_tv, feat_shift.
Then analysis pools all methods together and correlates representativeness->AUC.

Uses the verified subsampling path (fit on X[idx]). CPU-friendly, small data.
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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
TEST_SIZE = 0.2
K = 128
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044]  # 8 seeds per method

DATASETS = [
    (37, "diabetes"), (31, "credit-g"), (54, "vehicle"),
    (1510, "wdbc"), (15, "breast-w"), (1464, "blood-transfusion"),
]


def class_tv(y_full, y_sample, n_classes):
    pf = np.bincount(y_full, minlength=n_classes) / len(y_full)
    ps = np.bincount(y_sample, minlength=n_classes) / len(y_sample)
    return 0.5 * float(np.abs(pf - ps).sum())


def feat_shift(X_full_std, idx):
    return float(np.abs(X_full_std[idx].mean(axis=0) - X_full_std.mean(axis=0)).mean())


def sel_random(X, k, seed):
    return np.random.RandomState(seed).choice(X.shape[0], size=k, replace=False)


def sel_kmeans(X, k, seed):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(Xs)
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs)
    return np.unique(idx)


def sel_fps(X, k, seed):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    n = Xs.shape[0]
    rng = np.random.RandomState(seed)
    first = rng.randint(n)
    sel = [first]
    dist = np.linalg.norm(Xs - Xs[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        sel.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(Xs - Xs[nxt], axis=1))
    return np.array(sel)


SELECTORS = {"random": sel_random, "kmeans": sel_kmeans, "fps": sel_fps}


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp14b_cross_method_{stamp}.json"
    results = []

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    print(f"Experiment 14b: Cross-method representativeness | device={DEVICE} | k={K}")
    print("=" * 70)

    for openml_id, name in DATASETS:
        try:
            X, y, _ = load_openml(openml_id)
        except Exception as e:
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue
        n_classes = len(np.unique(y))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=0, stratify=y
        )
        X_full_std = StandardScaler().fit_transform(np.nan_to_num(Xtr))
        k = min(K, Xtr.shape[0] - 1)
        print(f"\n[{name}] n_train={Xtr.shape[0]} classes={n_classes} k={k}")

        for method in ("random", "kmeans", "fps"):
            for seed in SEEDS:
                rec = {"exp": "exp14b", "dataset": name, "method": method,
                       "k": int(k), "seed": seed, "n_classes": int(n_classes),
                       "device": DEVICE}
                try:
                    idx = SELECTORS[method](Xtr, k, seed)
                    Xk, yk = Xtr[idx], ytr[idx]
                    rec["k_effective"] = int(len(idx))
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); save(); continue
                    clf = TabPFNClassifier(device=DEVICE, random_state=seed,
                                           ignore_pretraining_limits=True)
                    clf.fit(Xk, yk)
                    proba = clf.predict_proba(Xte)
                    if n_classes == 2:
                        auc = roc_auc_score(yte, proba[:, 1])
                    else:
                        auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
                    rec["auc"] = float(auc)
                    rec["class_tv"] = class_tv(ytr, yk, n_classes)
                    rec["feat_shift"] = feat_shift(X_full_std, idx)
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                    rec["traceback"] = traceback.format_exc()
                results.append(rec); save()
            oks = [r for r in results if r["dataset"] == name
                   and r["method"] == method and r.get("status") == "OK"]
            if oks:
                au = np.array([r["auc"] for r in oks])
                print(f"  {method:8s} AUC={au.mean():.4f}+/-{au.std():.4f} (n={len(oks)})")

    print("\n" + "=" * 70)
    print(f"Results written to: {out_path}")
    return out_path


if __name__ == "__main__":
    run()
