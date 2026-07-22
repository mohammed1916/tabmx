"""
Experiment 12: Calibration Preservation Under Prototype Selection

Objective: Measure whether prototype-based TabPFN maintains calibration
(reliability of confidence estimates) compared to full TabPFN.

Hypothesis: Prototype selection (k=512) preserves calibration within 1-2% ECE
degradation; Full TabPFN and proto-TabPFN are equivalently calibrated.

Metric: Expected Calibration Error (ECE) — measure of confidence reliability.
"""

from __future__ import annotations

import json
import os
import time
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
SEED = 42

TEST_DATASETS = [
    (151, "electricity"),
    (1461, "bank-marketing"),
]


def compute_ece(y_true, proba, n_bins=10):
    """Compute Expected Calibration Error."""
    n_classes = proba.shape[1] if len(proba.shape) > 1 else 2
    if n_classes == 2:
        conf = np.max(proba, axis=1)
        pred = np.argmax(proba, axis=1)
    else:
        conf = np.max(proba, axis=1)
        pred = np.argmax(proba, axis=1)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_count = 0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_count += 1
        bin_acc = (pred[mask] == y_true[mask]).mean()
        bin_conf = conf[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(y_true)
    return ece


def run():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp12_calibration_{stamp}.json"
    results = []

    print("Experiment 12: Calibration Preservation")
    print(f"Device: {DEVICE}")
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

        print(f"\n[{name}] n_train={X_train.shape[0]} | n_classes={n_classes}")

        # Full TabPFN (if feasible)
        full_auc, full_ece = None, None
        try:
            if X_train.shape[0] <= 20000:  # Only if feasible
                clf_full = TabPFNClassifier(
                    device=DEVICE, random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf_full.fit(X_train, y_train)
                proba_full = clf_full.predict_proba(X_test)

                full_auc = (roc_auc_score(y_test, proba_full[:, 1]) if n_classes == 2
                           else roc_auc_score(y_test, proba_full, multi_class="ovr", average="macro"))
                full_ece = compute_ece(y_test, proba_full)

                print(f"  Full TabPFN (n={X_train.shape[0]}): AUC={full_auc:.4f}, ECE={full_ece:.4f}")
            else:
                print(f"  Full TabPFN: skipped (n={X_train.shape[0]} > 20K)")
        except Exception as e:  # noqa: BLE001
            print(f"  Full TabPFN FAILED: {repr(e)[:60]}")

        # Prototype-based (k=512)
        proto_auc, proto_ece = None, None
        try:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(X_train.shape[0], size=min(512, X_train.shape[0]), replace=False)
            Xk, yk = X_train[idx], y_train[idx]

            if len(np.unique(yk)) >= 2:
                clf_proto = TabPFNClassifier(
                    device=DEVICE, random_state=SEED,
                    ignore_pretraining_limits=True,
                    categorical_features_indices=cat_idx or None,
                )
                clf_proto.fit(Xk, yk)
                proba_proto = clf_proto.predict_proba(X_test)

                proto_auc = (roc_auc_score(y_test, proba_proto[:, 1]) if n_classes == 2
                            else roc_auc_score(y_test, proba_proto, multi_class="ovr", average="macro"))
                proto_ece = compute_ece(y_test, proba_proto)

                print(f"  Proto k=512 : AUC={proto_auc:.4f}, ECE={proto_ece:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  Prototype FAILED: {repr(e)[:60]}")

        # Compute deltas
        auc_delta = (proto_auc - full_auc) if proto_auc and full_auc else None
        ece_delta = (proto_ece - full_ece) if proto_ece and full_ece else None

        if auc_delta is not None:
            print(f"  Delta: AUC {auc_delta:+.4f}, ECE {ece_delta:+.4f}")

        results.append({
            "dataset": name,
            "n_train": X_train.shape[0],
            "full_auc": float(full_auc) if full_auc else None,
            "full_ece": float(full_ece) if full_ece else None,
            "proto_auc": float(proto_auc) if proto_auc else None,
            "proto_ece": float(proto_ece) if proto_ece else None,
            "auc_delta": float(auc_delta) if auc_delta else None,
            "ece_delta": float(ece_delta) if ece_delta else None,
            "status": "OK" if proto_auc else "PARTIAL",
        })

        out_path.write_text(json.dumps(results, indent=2))

    print("-" * 70)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    run()
