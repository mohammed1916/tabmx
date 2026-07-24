"""
Experiment 16: Disentangle mean-shift from diversity/coverage.

Block A of exp15 confounded two things: the greedy min-mean-shift construction
also collapsed diversity. So "low shift -> low AUC" could be caused by low
diversity, not low shift. Here we vary the two factors as independently as we
can and measure which one actually predicts AUC.

For each context we record THREE geometric descriptors (all on standardized
features of the full training set):
  - mean_shift : mean |context feature-mean - full feature-mean|   (representativeness)
  - diversity  : mean pairwise Euclidean distance among context rows (spread)
  - logdet_cov : log-det of (context covariance + eps I)             (volume spanned)
...and the resulting test AUC.

Construction methods, chosen to populate the (shift, diversity) plane:
  random            : uniform (medium shift, medium diversity)
  meanmatch_tight   : greedy min mean-shift            -> LOW shift, LOW diversity
  meanmatch_diverse : min mean-shift via antipodal pairs-> LOW shift, HIGH diversity
  cluster_offcenter : tight cluster around an off-center seed -> HIGH shift, LOW diversity
  fps               : farthest-point sampling          -> HIGH shift, HIGH diversity
  kmeans            : centroid nearest points          -> MED shift, HIGH diversity

Then analysis (analyze_exp16.py) regresses AUC on the three descriptors and
computes partial correlations: if diversity stays significant while mean_shift
drops out, the causal factor is diversity/coverage, not distribution matching.

k=128, class ratio matched where applicable, multiple seeds. GPU. Append-only.
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
N_SEEDS = 12

DATASETS = [
    (37, "diabetes"), (31, "credit-g"), (54, "vehicle"),
    (1510, "wdbc"), (15, "breast-w"), (1464, "blood-transfusion"),
    (59, "ionosphere"), (40, "sonar"), (41, "glass"), (40981, "australian"),
    (53, "heart-statlog"), (40984, "segment"), (60, "waveform-5000"),
    (44, "spambase"), (1067, "kc1"),
]


# ---------- geometric descriptors ----------
def descriptors(X_std, idx):
    S = X_std[idx]
    center = X_std.mean(axis=0)
    mean_shift = float(np.abs(S.mean(axis=0) - center).mean())
    # mean pairwise distance (subsample pairs if large; k=128 is fine to do fully)
    n = len(S)
    if n > 1:
        # efficient mean pairwise dist via random pairs to bound cost
        rng = np.random.RandomState(0)
        a = rng.randint(0, n, size=2000)
        b = rng.randint(0, n, size=2000)
        mask = a != b
        diversity = float(np.linalg.norm(S[a[mask]] - S[b[mask]], axis=1).mean())
    else:
        diversity = 0.0
    cov = np.cov(S, rowvar=False) + 1e-3 * np.eye(S.shape[1])
    sign, logdet = np.linalg.slogdet(cov)
    logdet_cov = float(logdet) if sign > 0 else float("nan")
    return mean_shift, diversity, logdet_cov


# ---------- selection / construction methods ----------
def m_random(X_std, y, k, seed):
    return np.random.RandomState(seed).choice(X_std.shape[0], size=k, replace=False)


def _alloc(y, k):
    classes, counts = np.unique(y, return_counts=True)
    return classes, np.maximum(1, np.round(k * counts / counts.sum()).astype(int))


def m_meanmatch_tight(X_std, y, k, seed):
    """Greedy: add the row that keeps running mean closest to center. Low shift,
    low diversity (clumps near mean)."""
    rng = np.random.RandomState(seed)
    center = X_std.mean(axis=0)
    classes, alloc = _alloc(y, k)
    chosen = []
    for c, a in zip(classes, alloc):
        c_idx = list(np.where(y == c)[0]); rng.shuffle(c_idx)
        a = min(a, len(c_idx))
        picks = [c_idx.pop()]
        running = X_std[picks[0]].astype(float).copy()
        while len(picks) < a and c_idx:
            cand = np.array(c_idx)
            nm = (running + X_std[cand]) / (len(picks) + 1)
            j = int(np.argmin(np.linalg.norm(nm - center, axis=1)))
            picks.append(int(cand[j])); running += X_std[cand[j]]; c_idx.pop(j)
        chosen.extend(picks)
    return np.array(chosen)


def m_meanmatch_diverse(X_std, y, k, seed):
    """Low shift but HIGH diversity: pair each far-from-center point with a
    roughly antipodal far point, so extremes cancel in the mean but spread stays."""
    rng = np.random.RandomState(seed)
    center = X_std.mean(axis=0)
    classes, alloc = _alloc(y, k)
    chosen = []
    for c, a in zip(classes, alloc):
        c_idx = np.where(y == c)[0]
        V = X_std[c_idx] - center
        dist = np.linalg.norm(V, axis=1)
        order = np.argsort(dist)[::-1]  # farthest first
        used = np.zeros(len(c_idx), dtype=bool)
        picks = []
        for i in order:
            if used[i] or len(picks) >= a:
                continue
            # find the unused point most antipodal to V[i]
            cos = (V @ V[i]) / (dist * dist[i] + 1e-9)
            cand = np.where(~used)[0]
            cand = cand[cand != i]
            if len(cand) == 0:
                picks.append(int(c_idx[i])); used[i] = True; continue
            j = cand[np.argmin(cos[cand])]  # most opposite direction
            picks.append(int(c_idx[i])); picks.append(int(c_idx[j]))
            used[i] = True; used[j] = True
        picks = picks[:a]
        # jitter order across seeds
        rng.shuffle(picks)
        chosen.extend(picks)
    return np.array(chosen)


def m_cluster_offcenter(X_std, y, k, seed):
    """HIGH shift, LOW diversity: pick a random seed, take its k nearest
    neighbors within class (tight off-center clump)."""
    rng = np.random.RandomState(seed)
    classes, alloc = _alloc(y, k)
    chosen = []
    for c, a in zip(classes, alloc):
        c_idx = np.where(y == c)[0]
        a = min(a, len(c_idx))
        seed_row = c_idx[rng.randint(len(c_idx))]
        d = np.linalg.norm(X_std[c_idx] - X_std[seed_row], axis=1)
        near = c_idx[np.argsort(d)[:a]]
        chosen.extend(near.tolist())
    return np.array(chosen)


def m_fps(X_std, y, k, seed):
    n = X_std.shape[0]
    rng = np.random.RandomState(seed)
    first = rng.randint(n)
    sel = [first]
    d = np.linalg.norm(X_std - X_std[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d)); sel.append(nxt)
        d = np.minimum(d, np.linalg.norm(X_std - X_std[nxt], axis=1))
    return np.array(sel)


def m_kmeans(X_std, y, k, seed):
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X_std)
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X_std)
    return np.unique(idx)


METHODS = {
    "random": m_random,
    "meanmatch_tight": m_meanmatch_tight,
    "meanmatch_diverse": m_meanmatch_diverse,
    "cluster_offcenter": m_cluster_offcenter,
    "fps": m_fps,
    "kmeans": m_kmeans,
}


def auc_of(proba, y_test, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_test, proba[:, 1])
    return roc_auc_score(y_test, proba, multi_class="ovr", average="macro")


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp16_disentangle_{stamp}.json"
    results = []

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    print(f"Experiment 16: disentangle shift vs diversity | device={DEVICE}")
    print("=" * 74)

    for openml_id, name in DATASETS:
        try:
            X, y, _ = load_openml(openml_id)
        except Exception as e:
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue
        n_classes = len(np.unique(y))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=0, stratify=y)
        X_std = StandardScaler().fit_transform(np.nan_to_num(Xtr))
        n_train = Xtr.shape[0]
        k = min(K, n_train - 1)
        print(f"\n[{name}] n_train={n_train} classes={n_classes} k={k}")

        for method, fn in METHODS.items():
            seeds = range(N_SEEDS) if method != "kmeans" else range(3)  # kmeans is slow/det.
            for s in seeds:
                seed = 300 + s
                rec = {"exp": "exp16", "dataset": name, "n_classes": int(n_classes),
                       "k": int(k), "method": method, "seed": seed, "device": DEVICE}
                try:
                    idx = fn(X_std, ytr, k, seed)
                    idx = np.unique(idx)
                    Xk, yk = Xtr[idx], ytr[idx]
                    rec["k_effective"] = int(len(idx))
                    if len(np.unique(yk)) < 2:
                        rec["status"] = "SKIP"; results.append(rec); save(); continue
                    ms, dv, ld = descriptors(X_std, idx)
                    rec["mean_shift"] = ms; rec["diversity"] = dv; rec["logdet_cov"] = ld
                    clf = TabPFNClassifier(device=DEVICE, random_state=seed,
                                           ignore_pretraining_limits=True)
                    clf.fit(Xk, yk)
                    rec["auc"] = float(auc_of(clf.predict_proba(Xte), yte, n_classes))
                    rec["status"] = "OK"
                except Exception as e:
                    rec["status"] = "ERROR"; rec["error"] = repr(e)
                    rec["traceback"] = traceback.format_exc()
                results.append(rec); save()
            oks = [r for r in results if r["dataset"] == name and r["method"] == method
                   and r.get("status") == "OK"]
            if oks:
                au = np.mean([r["auc"] for r in oks])
                ms = np.mean([r["mean_shift"] for r in oks])
                dv = np.mean([r["diversity"] for r in oks])
                print(f"  {method:18s} AUC={au:.4f}  shift={ms:.3f}  diversity={dv:.3f}")

    print("\n" + "=" * 74)
    print(f"Results written to: {out_path}")
    return out_path


if __name__ == "__main__":
    run()
