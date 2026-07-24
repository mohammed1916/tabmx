"""
Analyze exp15 results with effect sizes and bootstrap confidence intervals.

Blocks:
  A controlled : mean AUC + 95% bootstrap CI per representativeness level;
                 Cohen's d and Cliff's delta for high-vs-low; the causal test.
  B sel_vs_k   : kmeans/fps AUC vs k (compared to exp14 random at same k).
  C agreement  : mean prediction agreement across random contexts.
  D oracle     : mean / best AUC and the oracle gain (best - mean).

Writes exp15_analysis_summary.json and prints tables.
Usage: python analyze_exp15.py [path]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def newest():
    fs = sorted(RESULTS_DIR.glob("exp15_controlled_*.json"), key=lambda p: p.stat().st_mtime)
    if not fs:
        sys.exit("No exp15_controlled_*.json found.")
    return fs[-1]


def boot_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return float((gt - lt) / (len(a) * len(b))) if len(a) and len(b) else float("nan")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest()
    recs = json.loads(Path(path).read_text())
    print(f"Loaded {len(recs)} records from {path.name}\n")
    summary = {"source": path.name}
    datasets = sorted({r["dataset"] for r in recs if "dataset" in r})

    def ok(block, **f):
        return [r for r in recs if r.get("block") == block and r.get("status") == "OK"
                and all(r.get(k) == v for k, v in f.items())]

    # ---- A controlled ----
    print("=" * 74)
    print("A. CONTROLLED REPRESENTATIVENESS (k=128): does forcing repr. change AUC?")
    print("=" * 74)
    print(f"{'dataset':18s} {'high AUC':>16s} {'random':>16s} {'low AUC':>16s} {'d(hi-lo)':>9s} {'cliff':>7s}")
    A = {}
    for ds in datasets:
        row = {}
        vals = {}
        for mode in ("high", "random", "low"):
            au = [r["auc"] for r in ok("controlled", dataset=ds, mode=mode)]
            fs = [r["feat_shift"] for r in ok("controlled", dataset=ds, mode=mode)]
            if not au:
                continue
            lo, hi = boot_ci(au, seed=1)
            vals[mode] = au
            row[mode] = {"mean": float(np.mean(au)), "ci": [lo, hi],
                         "feat_shift": float(np.mean(fs)), "n": len(au)}
        if {"high", "low"} <= vals.keys():
            row["cohens_d_high_low"] = cohens_d(vals["high"], vals["low"])
            row["cliffs_delta_high_low"] = cliffs_delta(vals["high"], vals["low"])
        A[ds] = row
        if row.get("high") and row.get("low"):
            h, r_, l = row["high"], row.get("random", {"mean": float('nan'), "ci":[float('nan')]*2}), row["low"]
            print(f"{ds:18s} {h['mean']:.3f}[{h['ci'][0]:.3f},{h['ci'][1]:.3f}] "
                  f"{r_['mean']:.3f}[{r_['ci'][0]:.3f},{r_['ci'][1]:.3f}] "
                  f"{l['mean']:.3f}[{l['ci'][0]:.3f},{l['ci'][1]:.3f}] "
                  f"{row['cohens_d_high_low']:>9.2f} {row['cliffs_delta_high_low']:>7.2f}")
    summary["A_controlled"] = A

    # ---- B sel_vs_k ----
    print("\n" + "=" * 74)
    print("B. KMEANS/FPS ACROSS k")
    print("=" * 74)
    B = {}
    for ds in datasets:
        B[ds] = {}
        for method in ("kmeans", "fps"):
            ks = sorted({r["k"] for r in ok("sel_vs_k", dataset=ds, method=method)})
            B[ds][method] = {}
            for k in ks:
                au = [r["auc"] for r in ok("sel_vs_k", dataset=ds, method=method, k=k)]
                B[ds][method][k] = {"mean": float(np.mean(au)), "n": len(au)}
            if ks:
                print(f"{ds:18s} {method:7s} " +
                      " ".join(f"k{k}={B[ds][method][k]['mean']:.3f}" for k in ks))
    summary["B_sel_vs_k"] = B

    # ---- C agreement ----
    print("\n" + "=" * 74)
    print("C. PREDICTION AGREEMENT across random contexts (k=128)")
    print("=" * 74)
    C = {}
    for ds in datasets:
        rs = ok("agreement", dataset=ds)
        if rs:
            r = rs[0]
            C[ds] = {"mean_agreement": r["mean_agreement"],
                     "frac_unanimous": r["frac_unanimous"],
                     "min_agreement": r["min_agreement"], "n_contexts": r["n_contexts"]}
            print(f"{ds:18s} mean_agree={r['mean_agreement']:.3f} "
                  f"unanimous={r['frac_unanimous']:.3f} min={r['min_agreement']:.3f}")
    summary["C_agreement"] = C

    # ---- D oracle ----
    print("\n" + "=" * 74)
    print("D. ORACLE SUBSET (200 random contexts, k=128): best vs mean")
    print("=" * 74)
    D = {}
    for ds in datasets:
        rs = ok("oracle", dataset=ds)
        if rs:
            r = rs[0]
            D[ds] = {"mean_auc": r["mean_auc"], "best_auc": r["best_auc"],
                     "worst_auc": r["worst_auc"], "oracle_gain": r["oracle_gain"],
                     "n": r["n_contexts"]}
            print(f"{ds:18s} mean={r['mean_auc']:.4f} best={r['best_auc']:.4f} "
                  f"worst={r['worst_auc']:.4f} gain(best-mean)={r['oracle_gain']:.4f}")
    summary["D_oracle"] = D

    out = RESULTS_DIR / "exp15_analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
