"""Quick check: does fitting on k random rows actually change AUC across k?
This verifies the exp2-style subsampling path (fit on X_train[idx]) works,
unlike exp13 which fit on the full set regardless of k."""
import os
os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from data_utils import load_openml

X, y, cat = load_openml(37)  # diabetes, n=768
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(f"n_train={Xtr.shape[0]}")
for k in [16, 32, 64, 128, 256, Xtr.shape[0]]:
    k = min(k, Xtr.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(Xtr.shape[0], size=k, replace=False)
    clf = TabPFNClassifier(device="cpu", random_state=42, ignore_pretraining_limits=True)
    clf.fit(Xtr[idx], ytr[idx])
    p = clf.predict_proba(Xte)[:, 1]
    print(f"k={k:4d}  AUC={roc_auc_score(yte, p):.4f}")
