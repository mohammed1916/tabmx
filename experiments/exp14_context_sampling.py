"""
Experiment 14: Context Sampling on Small Tabular Datasets (Paper 2)

Central question: how stable and representative does the context need to be for
reliable TabPFN inference on SMALL datasets?

Three linked hypotheses:
  H1 Stability        - different random prototype samples give different AUC;
                        variance shrinks as k grows.
  H2 Representativeness- distribution-preserving prototype sets do better;
                        AUC correlates with how well the sample matches the
                        full training distribution.
  H3 Selection cost   - expensive selection (kmeans/fps) gives little over
                        random once the context is representative.

Supporting:
  Redundancy - duplicate a fraction of prototypes (no new info): does AUC move?
  Robustness - drop a fraction of the context: how fast does AUC degrade?

All on small datasets (n_train small enough to run many repeats on CPU).
Every configuration uses the verified subsampling path: fit on X_train[idx].
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

# Small datasets only (kept small so we can run many repeats on CPU).
DATASETS = [
    (37, "diabetes"),     # 768 rows,  8 feat, 2 cls
    (31, "credit-g"),     # 1000 rows, 20 feat, 2 cls
    (54, "vehicle"),      # 846 rows,  18 feat, 4 cls
    (1510, "wdbc"),       # 569 rows,  30 feat, 2 cls
    (15, "breast-w"),     # 699 rows,  9 feat, 2 cls
    (1464, "blood-transfusion"),  # 748 rows, 4 feat, 2 cls
    (59, "ionosphere"),   # 351 rows,  34 feat, 2 cls
    (40, "sonar"),        # 208 rows,  60 feat, 2 cls
    (41, "glass"),        # 214 rows,  9 feat, 6 cls
    (40981, "australian"),# 690 rows,  14 feat, 2 cls
    (53, "heart-statlog"),# 270 rows,  13 feat, 2 cls
    (40984, "segment"),   # 2310 rows, 18 feat, 7 cls
    (60, "waveform-5000"),# 5000 rows, 40 feat, 3 cls
    (44, "spambase"),     # 4601 rows, 57 feat, 2 cls
    (1067, "kc1"),        # 2109 rows, 21 feat, 2 cls
]

# k values for stability/representativeness sweeps (small-data range).
K_VALUES = [16, 32, 64, 128, 256]
N_REPEATS = 20          # random selections per k (for stability + representativeness)
SELECT_METHODS = ["random", "kmeans", "fps"]
K_SELECT = 128          # fixed k for the selection-method comparison
REDUNDANCY_FRACS = [0.0, 0.1, 0.3, 0.5, 0.7]   # fraction of prototypes duplicated
ROBUSTNESS_FRACS = [0.0, 0.1, 0.2, 0.3, 0.5]   # fraction of context dropped


def auc_score(clf, X_test, y_test, n_classes):
    proba = clf.predict_proba(X_test)
    if n_classes == 2:
        return roc_auc_score(y_test, proba[:, 1])
    return roc_auc_score(y_test, proba, multi_class="ovr", average="macro")


def fit_predict_auc(Xk, yk, X_test, y_test, n_classes, seed):
    clf = TabPFNClassifier(device=DEVICE, random_state=seed,
                           ignore_pretraining_limits=True)
    clf.fit(Xk, yk)
    return auc_score(clf, X_test, y_test, n_classes)


# --- representativeness metrics ------------------------------------------------
def class_tv_distance(y_full, y_sample, n_classes):
    """Total-variation distance between class distributions (0=identical)."""
    pf = np.bincount(y_full, minlength=n_classes) / len(y_full)
    ps = np.bincount(y_sample, minlength=n_classes) / len(y_sample)
    return 0.5 * np.abs(pf - ps).sum()


def feature_mean_shift(X_full_std, idx):
    """Mean absolute standardized feature-mean shift of the sample vs full."""
    return float(np.abs(X_full_std[idx].mean(axis=0) - X_full_std.mean(axis=0)).mean())


# --- selectors (same definitions as exp2) -------------------------------------
def sel_random(X, y, k, seed):
    return np.random.RandomState(seed).choice(X.shape[0], size=k, replace=False)


def sel_kmeans(X, y, k, seed):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(Xs)
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Xs)
    return np.unique(idx)


def sel_fps(X, y, k, seed):
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
    out_path = RESULTS_DIR / f"exp14_context_sampling_{stamp}.json"
    results = []

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    print(f"Experiment 14: Context Sampling on Small Datasets | device={DEVICE}")
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
        n_train = Xtr.shape[0]
        X_full_std = StandardScaler().fit_transform(np.nan_to_num(Xtr))
        print(f"\n[{name}] n_train={n_train} classes={n_classes}")

        # full-context reference AUC
        try:
            full_auc = fit_predict_auc(Xtr, ytr, Xte, yte, n_classes, 0)
        except Exception as e:
            print(f"  full-context FAILED: {e!r}")
            full_auc = None
        print(f"  full-context AUC={full_auc}")

        # ---- H1 Stability + H2 Representativeness (shared loop) -------------
        for k in K_VALUES:
            if k >= n_train:
                continue
            for rep in range(N_REPEATS):
                seed = 1000 + rep
                rec = {"exp": "exp14", "block": "stability_repr", "dataset": name,
                       "n_train": int(n_train), "n_classes": int(n_classes),
                       "k": int(k), "rep": rep, "seed": seed, "device": DEVICE,
                       "full_auc": full_auc}
                try:
                    idx = sel_random(Xtr, ytr, k, seed)
                    Xk, yk = Xtr[idx], ytr[idx]
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); save(); continue
                    rec["auc"] = float(fit_predict_auc(Xk, yk, Xte, yte, n_classes, seed))
                    rec["class_tv"] = float(class_tv_distance(ytr, yk, n_classes))
                    rec["feat_shift"] = feature_mean_shift(X_full_std, idx)
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                results.append(rec); save()
            oks = [r["auc"] for r in results
                   if r.get("block") == "stability_repr" and r["dataset"] == name
                   and r["k"] == k and r.get("status") == "OK"]
            if oks:
                print(f"  k={k:4d}  AUC mean={np.mean(oks):.4f} std={np.std(oks):.4f} (n={len(oks)})")

        # ---- H3 Selection method comparison at fixed k ----------------------
        ks = min(K_SELECT, n_train - 1)
        for method in SELECT_METHODS:
            for seed in [42, 123, 456]:
                rec = {"exp": "exp14", "block": "selection", "dataset": name,
                       "n_train": int(n_train), "n_classes": int(n_classes),
                       "k": int(ks), "method": method, "seed": seed, "device": DEVICE}
                try:
                    t0 = time.perf_counter()
                    idx = SELECTORS[method](Xtr, ytr, ks, seed)
                    rec["select_sec"] = float(time.perf_counter() - t0)
                    Xk, yk = Xtr[idx], ytr[idx]
                    rec["k_effective"] = int(len(idx))
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); save(); continue
                    rec["auc"] = float(fit_predict_auc(Xk, yk, Xte, yte, n_classes, seed))
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                results.append(rec); save()
            oks = [r["auc"] for r in results
                   if r.get("block") == "selection" and r["dataset"] == name
                   and r["method"] == method and r.get("status") == "OK"]
            ts = [r["select_sec"] for r in results
                  if r.get("block") == "selection" and r["dataset"] == name
                  and r["method"] == method and "select_sec" in r]
            if oks:
                print(f"  [sel] {method:8s} AUC={np.mean(oks):.4f} sel_time={np.mean(ts):.4f}s")

        # ---- Redundancy: duplicate a fraction of a base k=128 context -------
        base_k = min(128, n_train - 1)
        for frac in REDUNDANCY_FRACS:
            for seed in [42, 123, 456]:
                rec = {"exp": "exp14", "block": "redundancy", "dataset": name,
                       "base_k": int(base_k), "dup_frac": frac, "seed": seed,
                       "device": DEVICE}
                try:
                    idx = sel_random(Xtr, ytr, base_k, seed)
                    n_dup = int(round(frac * base_k))
                    dup_idx = np.concatenate([idx, idx[:n_dup]]) if n_dup else idx
                    Xk, yk = Xtr[dup_idx], ytr[dup_idx]
                    rec["total_rows"] = int(len(dup_idx))
                    rec["auc"] = float(fit_predict_auc(Xk, yk, Xte, yte, n_classes, seed))
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                results.append(rec); save()
            oks = [r["auc"] for r in results
                   if r.get("block") == "redundancy" and r["dataset"] == name
                   and r["dup_frac"] == frac and r.get("status") == "OK"]
            if oks:
                print(f"  [redund] dup={frac:.0%} AUC={np.mean(oks):.4f}")

        # ---- Robustness: drop a fraction of a base k=128 context ------------
        for frac in ROBUSTNESS_FRACS:
            for seed in [42, 123, 456]:
                rec = {"exp": "exp14", "block": "robustness", "dataset": name,
                       "base_k": int(base_k), "drop_frac": frac, "seed": seed,
                       "device": DEVICE}
                try:
                    idx = sel_random(Xtr, ytr, base_k, seed)
                    n_keep = max(2, int(round((1 - frac) * base_k)))
                    keep = idx[:n_keep]
                    Xk, yk = Xtr[keep], ytr[keep]
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); save(); continue
                    rec["kept_rows"] = int(len(keep))
                    rec["auc"] = float(fit_predict_auc(Xk, yk, Xte, yte, n_classes, seed))
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                results.append(rec); save()
            oks = [r["auc"] for r in results
                   if r.get("block") == "robustness" and r["dataset"] == name
                   and r["drop_frac"] == frac and r.get("status") == "OK"]
            if oks:
                print(f"  [robust] drop={frac:.0%} AUC={np.mean(oks):.4f}")

    print("\n" + "=" * 70)
    print(f"Results written to: {out_path}")
    return out_path


if __name__ == "__main__":
    run()
