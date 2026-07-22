"""
Experiment 5: Distribution Shift Robustness

Objective: Does prototype selection (k=512) make TabPFN more or less robust to
covariate shift than full-fit TabPFN?

For each dataset: fit proto (k=512 random) and full TabPFN on clean training
data, then evaluate on test sets perturbed by covariate shift of increasing
severity. Report accuracy drop relative to the unshifted test set.

Shifts (applied to numeric feature columns only):
  - scale:  multiply features by factor in {1,2,3,5}
  - noise:  add Gaussian noise sigma in {0.0, 0.25, 0.5, 1.0} (on standardized)
One variable changes per row: shift type + magnitude + method. Append-only JSON.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_PROTO = 512
SEED = 42

DATASETS = [
    (37, "diabetes"),
    (54, "vehicle"),
    (1461, "bank-marketing"),
    (1590, "adult"),
]

SCALE_FACTORS = [1.0, 2.0, 3.0, 5.0]
NOISE_SIGMAS = [0.0, 0.25, 0.5, 1.0]


def auc_of(y_test, proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_test, proba[:, 1])
    return roc_auc_score(y_test, proba, multi_class="ovr", average="macro")


def apply_scale(X, num_cols, factor):
    Xs = X.copy()
    Xs[:, num_cols] = Xs[:, num_cols] * factor
    return Xs


def apply_noise(X, num_cols, sigma, seed):
    rng = np.random.RandomState(seed)
    Xs = X.copy()
    col = Xs[:, num_cols]
    std = np.nanstd(col, axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xs[:, num_cols] = col + rng.normal(0, sigma, size=col.shape) * std
    return Xs


def fit_model(X, y, cat_idx):
    clf = TabPFNClassifier(device=DEVICE, random_state=SEED,
                           ignore_pretraining_limits=True,
                           categorical_features_indices=cat_idx or None)
    clf.fit(X, y)
    return clf


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp5_shift_{stamp}.json"
    results = []
    print("Experiment 5: Distribution Shift Robustness")
    print(f"Device: {DEVICE} | proto k={K_PROTO}")
    print("-" * 70)

    for openml_id, name in DATASETS:
        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue
        n_classes = len(np.unique(y))
        num_cols = [j for j in range(X.shape[1]) if j not in cat_idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=0, stratify=y
        )
        # Cap full-fit training pool for very large datasets so full model is
        # trainable on 8GB; record the cap. Proto uses k=512 regardless.
        if X_train.shape[0] > 10000:
            rng = np.random.RandomState(SEED)
            cap = rng.choice(X_train.shape[0], size=10000, replace=False)
            X_full, y_full = X_train[cap], y_train[cap]
            full_note = "full-fit pool capped at 10000"
        else:
            X_full, y_full = X_train, y_train
            full_note = "full pool"
        rng = np.random.RandomState(SEED)
        pidx = rng.choice(X_train.shape[0], size=min(K_PROTO, X_train.shape[0]),
                          replace=False)
        X_proto, y_proto = X_train[pidx], y_train[pidx]

        print(f"\n[{name}] proto={X_proto.shape[0]} full={X_full.shape[0]} "
              f"num_cols={len(num_cols)} ({full_note})")

        models = {}
        try:
            models["proto_k512"] = fit_model(X_proto, y_proto, cat_idx)
            models["full"] = fit_model(X_full, y_full, cat_idx)
        except Exception as e:  # noqa: BLE001
            print(f"  FIT FAILED: {e!r}")
            continue

        # Baseline (unshifted) AUC per model, to compute drops.
        base = {}
        for mname, clf in models.items():
            base[mname] = auc_of(y_test, clf.predict_proba(X_test), n_classes)
        print(f"  baseline AUC: proto={base['proto_k512']:.4f} "
              f"full={base['full']:.4f}")

        # Scale shifts
        for factor in SCALE_FACTORS:
            Xsh = apply_scale(X_test, num_cols, factor)
            for mname, clf in models.items():
                try:
                    auc = auc_of(y_test, clf.predict_proba(Xsh), n_classes)
                    drop = (base[mname] - auc) / base[mname] * 100
                    results.append({
                        "experiment": "exp5_shift", "dataset": name,
                        "method": mname, "shift": "scale", "magnitude": factor,
                        "roc_auc": float(auc), "baseline_auc": float(base[mname]),
                        "pct_drop": float(drop), "n_classes": int(n_classes),
                        "status": "OK"})
                except Exception as e:  # noqa: BLE001
                    results.append({"experiment": "exp5_shift", "dataset": name,
                                    "method": mname, "shift": "scale",
                                    "magnitude": factor, "status": "ERROR",
                                    "error": repr(e)[:150]})
            out_path.write_text(json.dumps(results, indent=2))

        # Noise shifts
        for sigma in NOISE_SIGMAS:
            Xsh = apply_noise(X_test, num_cols, sigma, SEED)
            for mname, clf in models.items():
                try:
                    auc = auc_of(y_test, clf.predict_proba(Xsh), n_classes)
                    drop = (base[mname] - auc) / base[mname] * 100
                    results.append({
                        "experiment": "exp5_shift", "dataset": name,
                        "method": mname, "shift": "noise", "magnitude": sigma,
                        "roc_auc": float(auc), "baseline_auc": float(base[mname]),
                        "pct_drop": float(drop), "n_classes": int(n_classes),
                        "status": "OK"})
                except Exception as e:  # noqa: BLE001
                    results.append({"experiment": "exp5_shift", "dataset": name,
                                    "method": mname, "shift": "noise",
                                    "magnitude": sigma, "status": "ERROR",
                                    "error": repr(e)[:150]})
            out_path.write_text(json.dumps(results, indent=2))

        # Print compact summary for this dataset
        for shift, mags in [("scale", SCALE_FACTORS), ("noise", NOISE_SIGMAS)]:
            for m in ["proto_k512", "full"]:
                drops = [r["pct_drop"] for r in results
                         if r.get("dataset") == name and r.get("shift") == shift
                         and r.get("method") == m and r.get("status") == "OK"]
                if drops:
                    print(f"  {shift:5s} {m:10s}: max drop={max(drops):.1f}%")

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
