"""
Experiment 3: Scaling Study

Objective: Directly measure how prototype TabPFN (fixed k) compares to full-fit
TabPFN and gradient-boosted baselines as the training-set size n grows.

For each dataset with a large training pool, subsample the TRAINING POOL to
n in {1k, 2k, 5k, 10k, 20k, full}. For each n:
  - Proto-TabPFN (k=512 random): fit on 512 random rows from the n-pool
  - Full-TabPFN: fit on all n rows (records OOM/time as n grows)
  - CatBoost, LightGBM, XGBoost: fit on all n rows (default settings)
Test set is fixed (held out once). One variable changes per row: n (and method).

Records accuracy, fit+predict time, GPU/CPU memory, OOM status. Append-only JSON.
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
N_SIZES = [1000, 2000, 5000, 10000, 20000, None]  # None = full pool
SEED = 42

# Large datasets only (need big training pools).
DATASETS = [
    (1461, "bank-marketing"),  # ~45K
    (1590, "adult"),           # ~49K
]


def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0


def reset_gpu():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def try_import_boosters():
    boosters = {}
    try:
        from catboost import CatBoostClassifier
        boosters["catboost"] = lambda: CatBoostClassifier(
            verbose=0, random_seed=SEED, thread_count=8
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from lightgbm import LGBMClassifier
        boosters["lightgbm"] = lambda: LGBMClassifier(
            random_state=SEED, n_jobs=8, verbose=-1
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from xgboost import XGBClassifier
        boosters["xgboost"] = lambda: XGBClassifier(
            random_state=SEED, n_jobs=8, verbosity=0, tree_method="hist"
        )
    except Exception:  # noqa: BLE001
        pass
    return boosters


def auc_of(y_test, proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_test, proba[:, 1])
    return roc_auc_score(y_test, proba, multi_class="ovr", average="macro")


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp3_scaling_{stamp}.json"
    results = []
    boosters = try_import_boosters()
    print("Experiment 3: Scaling Study")
    print(f"Device: {DEVICE} | proto k={K_PROTO} | boosters={list(boosters)}")
    print(f"N sizes: {N_SIZES}")
    print("-" * 70)

    for openml_id, name in DATASETS:
        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue
        n_classes = len(np.unique(y))
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.10, random_state=0, stratify=y
        )
        print(f"\n[{name}] pool={X_pool.shape[0]} test={X_test.shape[0]} "
              f"classes={n_classes} cat={len(cat_idx)}")

        for n in N_SIZES:
            if n is not None and n > X_pool.shape[0]:
                continue
            # Subsample training pool to size n (stratified via simple shuffle).
            if n is None:
                Xn, yn = X_pool, y_pool
                n_eff = X_pool.shape[0]
            else:
                rng = np.random.RandomState(SEED)
                idx = rng.choice(X_pool.shape[0], size=n, replace=False)
                Xn, yn = X_pool[idx], y_pool[idx]
                n_eff = n
            print(f"  n={n_eff}:")

            # --- Proto-TabPFN (k=512 random from the n-pool) ---
            rec = {"experiment": "exp3_scaling", "dataset": name, "n": n_eff,
                   "method": "proto_tabpfn_k512", "seed": SEED, "device": DEVICE,
                   "n_classes": int(n_classes)}
            try:
                rng = np.random.RandomState(SEED)
                pidx = rng.choice(Xn.shape[0], size=min(K_PROTO, Xn.shape[0]),
                                  replace=False)
                reset_gpu()
                clf = TabPFNClassifier(device=DEVICE, random_state=SEED,
                                       ignore_pretraining_limits=True,
                                       categorical_features_indices=cat_idx or None)
                t0 = time.perf_counter()
                clf.fit(Xn[pidx], yn[pidx])
                proba = clf.predict_proba(X_test)
                rec.update(roc_auc=float(auc_of(y_test, proba, n_classes)),
                           time_sec=float(time.perf_counter() - t0),
                           gpu_mem_mb=float(gpu_mem_mb()), status="OK")
                print(f"    proto_k512 : AUC={rec['roc_auc']:.4f} "
                      f"t={rec['time_sec']:.2f}s mem={rec['gpu_mem_mb']:.0f}MB")
            except Exception as e:  # noqa: BLE001
                rec.update(status="ERROR", error=repr(e))
                print(f"    proto_k512 : ERROR {rec['error'][:60]}")
            results.append(rec); out_path.write_text(json.dumps(results, indent=2))

            # --- Full-TabPFN (all n rows) ---
            rec = {"experiment": "exp3_scaling", "dataset": name, "n": n_eff,
                   "method": "full_tabpfn", "seed": SEED, "device": DEVICE,
                   "n_classes": int(n_classes)}
            try:
                reset_gpu()
                clf = TabPFNClassifier(device=DEVICE, random_state=SEED,
                                       ignore_pretraining_limits=True,
                                       categorical_features_indices=cat_idx or None)
                t0 = time.perf_counter()
                clf.fit(Xn, yn)
                proba = clf.predict_proba(X_test)
                rec.update(roc_auc=float(auc_of(y_test, proba, n_classes)),
                           time_sec=float(time.perf_counter() - t0),
                           gpu_mem_mb=float(gpu_mem_mb()), status="OK")
                print(f"    full_tabpfn: AUC={rec['roc_auc']:.4f} "
                      f"t={rec['time_sec']:.2f}s mem={rec['gpu_mem_mb']:.0f}MB")
            except Exception as e:  # noqa: BLE001
                rec.update(status="ERROR", error=repr(e)[:200])
                is_oom = "out of memory" in repr(e).lower() or "OutOfMemory" in repr(e)
                rec["oom"] = bool(is_oom)
                print(f"    full_tabpfn: {'OOM' if is_oom else 'ERROR'} "
                      f"{repr(e)[:60]}")
            results.append(rec); out_path.write_text(json.dumps(results, indent=2))

            # --- Gradient-boosted baselines (all n rows) ---
            for bname, ctor in boosters.items():
                rec = {"experiment": "exp3_scaling", "dataset": name, "n": n_eff,
                       "method": bname, "seed": SEED, "n_classes": int(n_classes)}
                try:
                    model = ctor()
                    t0 = time.perf_counter()
                    Xn_f = np.nan_to_num(Xn)
                    Xt_f = np.nan_to_num(X_test)
                    model.fit(Xn_f, yn)
                    proba = model.predict_proba(Xt_f)
                    rec.update(roc_auc=float(auc_of(y_test, proba, n_classes)),
                               time_sec=float(time.perf_counter() - t0), status="OK")
                    print(f"    {bname:10s}: AUC={rec['roc_auc']:.4f} "
                          f"t={rec['time_sec']:.2f}s")
                except Exception as e:  # noqa: BLE001
                    rec.update(status="ERROR", error=repr(e)[:200])
                    print(f"    {bname:10s}: ERROR {repr(e)[:50]}")
                results.append(rec); out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
