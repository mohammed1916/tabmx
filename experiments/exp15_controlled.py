"""
Experiment 15: Controlled / extended context-sampling experiments (Paper 2 v2).

Covers GPT's "one week" priority set:

  Block A - Controlled representativeness (causal-flavored):
    Construct contexts at target representativeness levels instead of observing
    random ones. For each level (high / medium / low representativeness) we
    build many contexts and measure AUC. If forcing lower feat_shift raises AUC,
    that is stronger than a correlation.
      - "high"  : greedily pick rows that keep the running feature-mean close to
                  the full-data mean (low feat_shift).
      - "low"   : greedily pick rows far from the full-data mean (high feat_shift).
      - "random": uniform baseline.
    (Class balance is held at the full-data ratio for all three so we isolate
     feature representativeness, not class shift.)

  Block B - KMeans/FPS across ALL k (16,32,64,128,256), not just k=128.

  Block C - Prediction agreement across random contexts: for a fixed test set,
    how often do independent random contexts assign the same label to the same
    test row? (mean per-sample modal-label agreement).

  Block D - Oracle subset: draw 200 random contexts at k=128, report best vs
    mean AUC (how much room a perfect selector could gain).

All CPU-friendly on small datasets. Verified subsampling path (fit on X[idx]).
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

DATASETS = [
    (37, "diabetes"), (31, "credit-g"), (54, "vehicle"),
    (1510, "wdbc"), (15, "breast-w"), (1464, "blood-transfusion"),
    (59, "ionosphere"), (40, "sonar"), (41, "glass"), (40981, "australian"),
    (53, "heart-statlog"), (40984, "segment"), (60, "waveform-5000"),
    (44, "spambase"), (1067, "kc1"),
]

K_ALL = [16, 32, 64, 128, 256]
K_FIX = 128
N_CONTROLLED = 15     # contexts per representativeness level
N_AGREEMENT = 20      # random contexts for prediction-agreement
N_ORACLE = 200        # random contexts for oracle-subset


def feat_shift(X_full_std, idx):
    return float(np.abs(X_full_std[idx].mean(axis=0) - X_full_std.mean(axis=0)).mean())


def auc_of(proba, y_test, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_test, proba[:, 1])
    return roc_auc_score(y_test, proba, multi_class="ovr", average="macro")


def fit_proba(Xk, yk, X_test, seed):
    clf = TabPFNClassifier(device=DEVICE, random_state=seed,
                           ignore_pretraining_limits=True)
    clf.fit(Xk, yk)
    return clf.predict_proba(X_test), clf.predict(X_test)


# ---- controlled selection: bias toward / away from the full-data centroid ----
def _greedy_min_shift(X_std, cand_idx, a, center, rng):
    """Greedily build a size-a subset of cand_idx whose running feature-mean
    stays close to `center` (global mean) -> low feat_shift ('high' repr)."""
    # start from a random seed row, then repeatedly add the candidate that most
    # reduces the distance of the running sum-mean to the target center.
    remaining = list(cand_idx)
    rng.shuffle(remaining)
    chosen = [remaining.pop()]
    running = X_std[chosen[0]].astype(float).copy()
    while len(chosen) < a and remaining:
        m = len(chosen)
        # candidate new mean = (running + X[j]) / (m+1); want closest to center
        cand = np.array(remaining)
        new_means = (running + X_std[cand]) / (m + 1)
        d = np.linalg.norm(new_means - center, axis=1)
        best = int(np.argmin(d))
        j = cand[best]
        chosen.append(int(j))
        running = running + X_std[j]
        remaining.pop(best)
    return chosen


def controlled_idx(X_std, y, k, mode, seed):
    """Pick k rows with class ratio matched to full data. Feature bias by mode:
      'high'   = greedily minimize feature-mean shift (most representative),
      'low'    = pick rows far from the centroid (least representative),
      'random' = uniform within class."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    alloc = np.maximum(1, np.round(k * counts / counts.sum()).astype(int))
    center = X_std.mean(axis=0)
    dist = np.linalg.norm(X_std - center, axis=1)
    chosen = []
    for c, a in zip(classes, alloc):
        c_idx = np.where(y == c)[0]
        a = min(a, len(c_idx))
        if mode == "random":
            pick = rng.choice(c_idx, size=a, replace=False).tolist()
        elif mode == "high":
            pick = _greedy_min_shift(X_std, c_idx, a, center, rng)
        else:  # low: farthest from centroid, with mild jitter for variety
            order = np.argsort(dist[c_idx])[::-1]
            noise = rng.rand(len(order)) * 0.15
            key = np.argsort(np.argsort(order) + noise * len(order))
            pick = c_idx[key[:a]].tolist()
        chosen.extend(pick)
    return np.array(chosen)


def sel_kmeans(X_std, k, seed):
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X_std)
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X_std)
    return np.unique(idx)


def sel_fps(X_std, k, seed):
    n = X_std.shape[0]
    rng = np.random.RandomState(seed)
    first = rng.randint(n)
    sel = [first]
    d = np.linalg.norm(X_std - X_std[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        sel.append(nxt)
        d = np.minimum(d, np.linalg.norm(X_std - X_std[nxt], axis=1))
    return np.array(sel)


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp15_controlled_{stamp}.json"
    results = []

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    print(f"Experiment 15: controlled + extended | device={DEVICE}")
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
        X_std = StandardScaler().fit_transform(np.nan_to_num(Xtr))
        n_train = Xtr.shape[0]
        print(f"\n[{name}] n_train={n_train} classes={n_classes}")

        # ---- Block A: controlled representativeness at k=128 ----
        kfix = min(K_FIX, n_train - 1)
        for mode in ("high", "random", "low"):
            for rep in range(N_CONTROLLED):
                seed = 500 + rep
                rec = {"exp": "exp15", "block": "controlled", "dataset": name,
                       "n_classes": int(n_classes), "k": int(kfix),
                       "mode": mode, "seed": seed, "device": DEVICE}
                try:
                    idx = controlled_idx(X_std, ytr, kfix, mode, seed)
                    Xk, yk = Xtr[idx], ytr[idx]
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); save(); continue
                    proba, _ = fit_proba(Xk, yk, Xte, seed)
                    rec["auc"] = float(auc_of(proba, yte, n_classes))
                    rec["feat_shift"] = feat_shift(X_std, idx)
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                    rec["traceback"] = traceback.format_exc()
                results.append(rec); save()
            oks = [r for r in results if r.get("block") == "controlled"
                   and r["dataset"] == name and r["mode"] == mode and r.get("status") == "OK"]
            if oks:
                au = np.array([r["auc"] for r in oks]); fs = np.array([r["feat_shift"] for r in oks])
                print(f"  [ctrl] {mode:6s} AUC={au.mean():.4f}+/-{au.std():.4f} feat_shift={fs.mean():.3f}")

        # ---- Block B: kmeans/fps across all k ----
        for method in ("kmeans", "fps"):
            for k in K_ALL:
                k = min(k, n_train - 1)
                for seed in (42, 123, 456):
                    rec = {"exp": "exp15", "block": "sel_vs_k", "dataset": name,
                           "n_classes": int(n_classes), "k": int(k),
                           "method": method, "seed": seed, "device": DEVICE}
                    try:
                        idx = sel_kmeans(X_std, k, seed) if method == "kmeans" else sel_fps(X_std, k, seed)
                        Xk, yk = Xtr[idx], ytr[idx]
                        rec["k_effective"] = int(len(idx))
                        if len(np.unique(yk)) < 2:
                            rec["status"] = "SKIP"; results.append(rec); save(); continue
                        proba, _ = fit_proba(Xk, yk, Xte, seed)
                        rec["auc"] = float(auc_of(proba, yte, n_classes))
                        rec["status"] = "OK"
                    except Exception as e:
                        rec["status"] = "ERROR"; rec["error"] = repr(e)
                    results.append(rec); save()
            got = [r for r in results if r.get("block") == "sel_vs_k"
                   and r["dataset"] == name and r["method"] == method and r.get("status") == "OK"]
            if got:
                print(f"  [selk] {method}: " + " ".join(
                    f"k{k}={np.mean([r['auc'] for r in got if r['k']==min(k,n_train-1)]):.3f}" for k in K_ALL))

        # ---- Block C: prediction agreement across random contexts (k=128) ----
        preds = []
        for rep in range(N_AGREEMENT):
            seed = 700 + rep
            try:
                rng = np.random.RandomState(seed)
                idx = rng.choice(n_train, size=kfix, replace=False)
                Xk, yk = Xtr[idx], ytr[idx]
                if len(np.unique(yk)) < 2:
                    continue
                _, yhat = fit_proba(Xk, yk, Xte, seed)
                preds.append(yhat)
            except Exception as e:
                print(f"  [agree] rep {rep} failed: {e!r}")
        if len(preds) >= 2:
            P = np.stack(preds)  # (n_contexts, n_test)
            # per-test-sample agreement = fraction voting for the modal label
            agree = []
            for j in range(P.shape[1]):
                vals, cnts = np.unique(P[:, j], return_counts=True)
                agree.append(cnts.max() / P.shape[0])
            rec = {"exp": "exp15", "block": "agreement", "dataset": name,
                   "k": int(kfix), "n_contexts": len(preds),
                   "mean_agreement": float(np.mean(agree)),
                   "min_agreement": float(np.min(agree)),
                   "frac_unanimous": float(np.mean(np.array(agree) == 1.0)),
                   "status": "OK"}
            results.append(rec); save()
            print(f"  [agree] mean={rec['mean_agreement']:.3f} "
                  f"unanimous={rec['frac_unanimous']:.3f} (n_ctx={len(preds)})")

        # ---- Block D: oracle subset (200 random contexts at k=128) ----
        aucs = []
        for rep in range(N_ORACLE):
            seed = 10000 + rep
            try:
                rng = np.random.RandomState(seed)
                idx = rng.choice(n_train, size=kfix, replace=False)
                Xk, yk = Xtr[idx], ytr[idx]
                if len(np.unique(yk)) < 2:
                    continue
                proba, _ = fit_proba(Xk, yk, Xte, seed)
                aucs.append(auc_of(proba, yte, n_classes))
            except Exception:
                pass
        if aucs:
            aucs = np.array(aucs)
            rec = {"exp": "exp15", "block": "oracle", "dataset": name,
                   "k": int(kfix), "n_contexts": len(aucs),
                   "mean_auc": float(aucs.mean()), "best_auc": float(aucs.max()),
                   "worst_auc": float(aucs.min()), "std_auc": float(aucs.std(ddof=1)),
                   "oracle_gain": float(aucs.max() - aucs.mean()), "status": "OK"}
            results.append(rec); save()
            print(f"  [oracle] mean={rec['mean_auc']:.4f} best={rec['best_auc']:.4f} "
                  f"gain={rec['oracle_gain']:.4f} (n={len(aucs)})")

    print("\n" + "=" * 70)
    print(f"Results written to: {out_path}")
    return out_path


if __name__ == "__main__":
    run()
