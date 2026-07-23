"""
Analyze exp14 context-sampling results into the numbers Paper 2 needs.

Produces (printed + written to a small JSON summary):
  H1 Stability:
    per dataset x k -> mean, std, coefficient of variation (CV = std/mean),
    min, max of AUC across the 20 random contexts.
  H2 Representativeness:
    Pearson r, Spearman rho, and linear-regression R^2 for
      class_tv   -> AUC
      feat_shift -> AUC
    computed per dataset AND pooled (z-scored within dataset so datasets with
    different AUC levels can be pooled fairly). Also a "most vs least
    representative" contrast: mean AUC of the most-representative third of draws
    vs the least-representative third.
  H3 Selection cost:
    per dataset x method -> mean AUC, mean selection time.
  Redundancy / Robustness:
    per dataset x fraction -> mean AUC.

Usage: python analyze_exp14.py [path_to_exp14_json]
If no path given, uses the newest exp14_context_sampling_*.json in results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = Path(__file__).parent / "results"


def newest_results():
    files = sorted(RESULTS_DIR.glob("exp14_context_sampling_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("No exp14_context_sampling_*.json found.")
    return files[-1]


def r2_linear(x, y):
    """R^2 of the least-squares line y ~ a*x + b."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_results()
    records = json.loads(Path(path).read_text())
    print(f"Loaded {len(records)} records from {path.name}\n")

    summary = {"source": path.name}

    def rows(block, **filt):
        out = []
        for r in records:
            if r.get("block") != block or r.get("status") != "OK":
                continue
            if all(r.get(k) == v for k, v in filt.items()):
                out.append(r)
        return out

    datasets = sorted({r["dataset"] for r in records if "dataset" in r})

    # ---------------- H1 Stability ----------------
    print("=" * 70)
    print("H1 STABILITY: AUC across 20 random contexts per k")
    print("=" * 70)
    h1 = {}
    for ds in datasets:
        ks = sorted({r["k"] for r in rows("stability_repr", dataset=ds)})
        h1[ds] = {}
        for k in ks:
            aucs = np.array([r["auc"] for r in rows("stability_repr", dataset=ds, k=k)])
            if len(aucs) == 0:
                continue
            m, s = float(aucs.mean()), float(aucs.std())
            cv = float(s / m) if m else float("nan")
            h1[ds][k] = {"n": len(aucs), "mean": m, "std": s, "cv": cv,
                         "min": float(aucs.min()), "max": float(aucs.max())}
        if h1[ds]:
            print(f"\n{ds}")
            print(f"  {'k':>5} {'mean':>8} {'std':>8} {'CV%':>7} {'range':>16}")
            for k, v in sorted(h1[ds].items()):
                print(f"  {k:>5} {v['mean']:>8.4f} {v['std']:>8.4f} "
                      f"{100*v['cv']:>6.2f}% [{v['min']:.4f},{v['max']:.4f}]")
    summary["H1_stability"] = h1

    # ---------------- H2 Representativeness ----------------
    print("\n" + "=" * 70)
    print("H2 REPRESENTATIVENESS: distance-to-full vs AUC")
    print("=" * 70)
    h2 = {"per_dataset": {}}
    pooled_tv, pooled_fs, pooled_auc = [], [], []
    for ds in datasets:
        rs = rows("stability_repr", dataset=ds)
        if len(rs) < 5:
            continue
        tv = np.array([r["class_tv"] for r in rs])
        fs = np.array([r["feat_shift"] for r in rs])
        au = np.array([r["auc"] for r in rs])
        d = {}
        for name, x in [("class_tv", tv), ("feat_shift", fs)]:
            if np.std(x) == 0:
                d[name] = {"pearson": None, "spearman": None, "r2": None,
                           "note": "no variance"}
                continue
            pr = float(pearsonr(x, au)[0])
            sr = float(spearmanr(x, au)[0])
            d[name] = {"pearson": pr, "spearman": sr, "r2": r2_linear(x, au)}
        h2["per_dataset"][ds] = d
        # pool with within-dataset z-scoring so AUC levels are comparable
        if np.std(au) > 0:
            pooled_auc.extend(((au - au.mean()) / au.std()).tolist())
            pooled_tv.extend((((tv - tv.mean()) / tv.std()) if np.std(tv) else np.zeros_like(tv)).tolist())
            pooled_fs.extend((((fs - fs.mean()) / fs.std()) if np.std(fs) else np.zeros_like(fs)).tolist())
        print(f"\n{ds}")
        for name in ("class_tv", "feat_shift"):
            v = d[name]
            if v.get("pearson") is None:
                print(f"  {name:11s}: (no variance)")
            else:
                print(f"  {name:11s}: Pearson={v['pearson']:+.3f}  "
                      f"Spearman={v['spearman']:+.3f}  R2={v['r2']:.3f}")

    if pooled_auc:
        pa = np.array(pooled_auc)
        pooled = {}
        for name, x in [("class_tv", np.array(pooled_tv)), ("feat_shift", np.array(pooled_fs))]:
            if np.std(x) == 0:
                pooled[name] = {"pearson": None}
                continue
            pooled[name] = {"pearson": float(pearsonr(x, pa)[0]),
                            "spearman": float(spearmanr(x, pa)[0]),
                            "r2": r2_linear(x, pa)}
        h2["pooled_zscored"] = pooled
        print("\nPOOLED (within-dataset z-scored)")
        for name in ("class_tv", "feat_shift"):
            v = pooled[name]
            if v.get("pearson") is None:
                print(f"  {name:11s}: (no variance)")
            else:
                print(f"  {name:11s}: Pearson={v['pearson']:+.3f}  "
                      f"Spearman={v['spearman']:+.3f}  R2={v['r2']:.3f}")

    # most vs least representative third (by class_tv, lower = more representative)
    print("\nMost- vs least-representative third (by class_tv):")
    contrast = {}
    for ds in datasets:
        rs = rows("stability_repr", dataset=ds)
        if len(rs) < 6:
            continue
        rs_sorted = sorted(rs, key=lambda r: r["class_tv"])
        third = max(1, len(rs_sorted) // 3)
        best = np.mean([r["auc"] for r in rs_sorted[:third]])       # most representative
        worst = np.mean([r["auc"] for r in rs_sorted[-third:]])     # least representative
        contrast[ds] = {"most_repr_auc": float(best), "least_repr_auc": float(worst),
                        "delta": float(best - worst)}
        print(f"  {ds:18s} most={best:.4f}  least={worst:.4f}  delta={best-worst:+.4f}")
    h2["representativeness_contrast"] = contrast
    summary["H2_representativeness"] = h2

    # ---------------- H3 Selection cost ----------------
    print("\n" + "=" * 70)
    print("H3 SELECTION COST: random vs kmeans vs fps at fixed k")
    print("=" * 70)
    h3 = {}
    for ds in datasets:
        methods = sorted({r["method"] for r in rows("selection", dataset=ds)})
        if not methods:
            continue
        h3[ds] = {}
        print(f"\n{ds}")
        for m in methods:
            rs = rows("selection", dataset=ds, method=m)
            au = np.array([r["auc"] for r in rs])
            ts = np.array([r.get("select_sec", np.nan) for r in rs])
            h3[ds][m] = {"mean_auc": float(au.mean()), "std_auc": float(au.std()),
                         "mean_select_sec": float(np.nanmean(ts))}
            print(f"  {m:8s} AUC={au.mean():.4f}+/-{au.std():.4f}  "
                  f"sel_time={np.nanmean(ts):.4f}s")
    summary["H3_selection"] = h3

    # ---------------- Redundancy / Robustness ----------------
    print("\n" + "=" * 70)
    print("REDUNDANCY (duplicate fraction) and ROBUSTNESS (drop fraction)")
    print("=" * 70)
    red = {}; rob = {}
    for ds in datasets:
        rr = rows("redundancy", dataset=ds)
        if rr:
            red[ds] = {}
            fr = sorted({r["dup_frac"] for r in rr})
            print(f"\n{ds} [redundancy]")
            for f in fr:
                au = np.array([r["auc"] for r in rows("redundancy", dataset=ds, dup_frac=f)])
                red[ds][f] = float(au.mean())
                print(f"  dup={f:.0%}: AUC={au.mean():.4f}")
        rb = rows("robustness", dataset=ds)
        if rb:
            rob[ds] = {}
            fr = sorted({r["drop_frac"] for r in rb})
            print(f"{ds} [robustness]")
            for f in fr:
                au = np.array([r["auc"] for r in rows("robustness", dataset=ds, drop_frac=f)])
                rob[ds][f] = float(au.mean())
                print(f"  drop={f:.0%}: AUC={au.mean():.4f}")
    summary["redundancy"] = red
    summary["robustness"] = rob

    out = RESULTS_DIR / "exp14_analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
