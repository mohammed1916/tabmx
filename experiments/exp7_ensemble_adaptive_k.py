"""
Experiment 7: Ensemble Adaptive k

Objective: Improve adaptive-k robustness via ensemble of regressors,
each trained on different subsets of training datasets.

Hypothesis: Ensemble of N adaptive-k models beats single model by 2-3%.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from data_utils import load_openml

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
K_CANDIDATES = [64, 128, 256, 512, 1024, 2048]
SEED = 42
N_ENSEMBLE = 3  # 3-member ensemble

TRAIN_DATASETS = [
    (37, "diabetes"),
    (54, "vehicle"),
    (31, "credit-g"),
    (1461, "bank-marketing"),
    (1590, "adult"),
]

TEST_DATASETS = [
    (151, "electricity"),
]


def compute_dataset_features(X, y, n_train, n_test):
    n, m = X.shape
    n_classes = len(np.unique(y))
    sparsity = np.isnan(X).sum() / (n * m) if X.size > 0 else 0.0
    return {
        "log_n": np.log1p(n_train),
        "log_m": np.log1p(m),
        "n_classes": n_classes,
        "class_density": n_classes / n_train,
        "n_m_ratio": n_train / m if m > 0 else 1.0,
        "sparsity": float(sparsity),
        "n_test": n_test,
    }


def find_optimal_k(X_train, X_val, y_train, y_val, cat_idx, n_classes):
    best_k = 512
    best_auc = 0.0
    for k in K_CANDIDATES:
        try:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(X_train.shape[0], size=min(k, X_train.shape[0]), replace=False)
            Xk, yk = X_train[idx], y_train[idx]
            if len(np.unique(yk)) < 2:
                continue
            clf = TabPFNClassifier(device=DEVICE, random_state=SEED,
                                   ignore_pretraining_limits=True,
                                   categorical_features_indices=cat_idx or None)
            clf.fit(Xk, yk)
            proba = clf.predict_proba(X_val)
            auc = (roc_auc_score(y_val, proba[:, 1]) if n_classes == 2
                   else roc_auc_score(y_val, proba, multi_class="ovr", average="macro"))
            if auc > best_auc:
                best_auc, best_k = auc, k
        except Exception:  # noqa: BLE001
            pass
    return best_k, best_auc


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp7_ensemble_adaptive_k_{stamp}.json"
    results = []

    print("Experiment 7: Ensemble Adaptive k")
    print(f"Device: {DEVICE} | N_ensemble={N_ENSEMBLE}")
    print("-" * 70)

    # Phase 1: Train N ensemble members (each on random subset of train datasets)
    print("\nPhase 1: Training ensemble members")
    print("-" * 70)

    ensemble = []
    for member_id in range(N_ENSEMBLE):
        # Bootstrap subset of training datasets
        rng = np.random.RandomState(SEED + member_id)
        subset_idx = rng.choice(len(TRAIN_DATASETS), size=len(TRAIN_DATASETS), replace=True)
        subset = [TRAIN_DATASETS[i] for i in subset_idx]

        X_features, y_optimal_k = [], []
        for openml_id, name in subset:
            try:
                X, y, cat_idx = load_openml(openml_id)
                n_classes = len(np.unique(y))
                X_train, X_val_test, y_train, y_val_test = train_test_split(
                    X, y, test_size=0.20, random_state=0, stratify=y
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_val_test, y_val_test, test_size=0.50, random_state=0, stratify=y_val_test
                )
                optimal_k, _ = find_optimal_k(X_train, X_val, y_train, y_val, cat_idx, n_classes)
                features = compute_dataset_features(X_train, y_train, X_train.shape[0], X_test.shape[0])
                X_features.append(list(features.values()))
                y_optimal_k.append(optimal_k)
            except Exception:  # noqa: BLE001
                pass

        if len(X_features) < 2:
            continue

        X_feat_arr = np.array(X_features)
        y_k_arr = np.array(y_optimal_k)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feat_arr)
        regressor = LinearRegression()
        regressor.fit(X_scaled, y_k_arr)
        r2 = regressor.score(X_scaled, y_k_arr)

        ensemble.append({"regressor": regressor, "scaler": scaler, "r2": r2})
        print(f"Member {member_id}: R²={r2:.4f} (trained on {len(X_features)} datasets)")

    # Phase 2: Test ensemble on held-out datasets
    print("\nPhase 2: Testing ensemble on held-out datasets")
    print("-" * 70)

    for openml_id, name in TEST_DATASETS:
        try:
            X, y, cat_idx = load_openml(openml_id)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] LOAD FAILED: {e!r}")
            continue

        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=0, stratify=y
        )

        features = compute_dataset_features(X_train, y_train, X_train.shape[0], X_test.shape[0])

        # Ensemble prediction: mean of all member predictions
        k_predictions = []
        for member in ensemble:
            X_feat_scaled = member["scaler"].transform([list(features.values())])
            pred_k = float(member["regressor"].predict(X_feat_scaled)[0])
            k_predictions.append(int(np.clip(np.round(pred_k), 64, 2048)))

        ensemble_k = int(np.median(k_predictions))
        print(f"[{name}] ensemble predictions: {k_predictions}; median={ensemble_k}")

        # Evaluate ensemble_k vs fixed_k=512
        auc_ensemble, auc_fixed = None, None
        for test_k, auc_var in [(ensemble_k, "auc_ensemble"), (512, "auc_fixed")]:
            try:
                rng = np.random.RandomState(SEED)
                idx = rng.choice(X_train.shape[0], size=min(test_k, X_train.shape[0]), replace=False)
                Xk, yk = X_train[idx], y_train[idx]
                if len(np.unique(yk)) < 2:
                    continue
                clf = TabPFNClassifier(device=DEVICE, random_state=SEED,
                                       ignore_pretraining_limits=True,
                                       categorical_features_indices=cat_idx or None)
                clf.fit(Xk, yk)
                proba = clf.predict_proba(X_test)
                auc = (roc_auc_score(y_test, proba[:, 1]) if n_classes == 2
                       else roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
                if auc_var == "auc_ensemble":
                    auc_ensemble = auc
                else:
                    auc_fixed = auc
            except Exception:  # noqa: BLE001
                pass

        improvement = (auc_ensemble - auc_fixed) / auc_fixed * 100 if auc_fixed and auc_ensemble else 0
        print(f"  ensemble_k={ensemble_k} AUC={auc_ensemble:.4f} | fixed_k=512 AUC={auc_fixed:.4f} | delta={improvement:+.1f}%\n")

        results.append({
            "dataset": name,
            "ensemble_k": int(ensemble_k),
            "k_predictions": k_predictions,
            "auc_ensemble": float(auc_ensemble) if auc_ensemble else None,
            "auc_fixed": float(auc_fixed) if auc_fixed else None,
            "improvement": float(improvement),
        })
        out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
