"""
Analyze exp16: does mean_shift or diversity/coverage drive AUC?

Key outputs, per dataset AND pooled (z-scored within dataset):
  - simple Pearson r of each descriptor (mean_shift, diversity, logdet_cov) with AUC
  - PARTIAL correlation of mean_shift with AUC controlling for diversity, and
    of diversity with AUC controlling for mean_shift
  - standardized multiple-regression coefficients (AUC ~ shift + diversity)
  - per-method mean AUC/shift/diversity table

Decision rule (stated, not hidden):
  If diversity's partial correlation stays strong while mean_shift's collapses
  toward 0 (or flips sign) once diversity is controlled, the causal factor is
  diversity/coverage, not distribution matching.

Usage: python analyze_exp16.py [path]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).parent / "results"


def newest():
    fs = sorted(RESULTS_DIR.glob("exp16_disentangle_*.json"), key=lambda p: p.stat().st_mtime)
    if not fs:
        sys.exit("No exp16_disentangle_*.json found.")
    return fs[-1]


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z (all 1D)."""
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    if np.std(z) == 0:
        return pearsonr(x, y)[0]
    rx, ry = resid(x, z), resid(y, z)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(pearsonr(rx, ry)[0])


def std_reg(y, X):
    """Standardized regression coefficients for y ~ X (columns already named)."""
    y = (y - y.mean()) / (y.std() + 1e-12)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xz1 = np.c_[np.ones(len(y)), Xz]
    coef, *_ = np.linalg.lstsq(Xz1, y, rcond=None)
    return coef[1:]  # drop intercept


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest()
    recs = [r for r in json.loads(Path(path).read_text()) if r.get("status") == "OK"]
    print(f"Loaded {len(recs)} OK records from {path.name}\n")
    datasets = sorted({r["dataset"] for r in recs})
    summary = {"source": path.name, "per_dataset": {}, "per_method": {}}

    print("=" * 82)
    print("PER-METHOD means (AUC, shift, diversity)")
    print("=" * 82)
    methods = ["meanmatch_tight", "cluster_offcenter", "random", "kmeans",
               "meanmatch_diverse", "fps"]
    for ds in datasets:
        print(f"\n{ds}")
        print(f"  {'method':18s} {'AUC':>7s} {'shift':>7s} {'diversity':>9s}")
        summary["per_method"].setdefault(ds, {})
        for m in methods:
            rs = [r for r in recs if r["dataset"] == ds and r["method"] == m]
            if not rs:
                continue
            au = np.mean([r["auc"] for r in rs]); ms = np.mean([r["mean_shift"] for r in rs])
            dv = np.mean([r["diversity"] for r in rs])
            summary["per_method"][ds][m] = {"auc": float(au), "shift": float(ms),
                                            "diversity": float(dv), "n": len(rs)}
            print(f"  {m:18s} {au:>7.3f} {ms:>7.3f} {dv:>9.3f}")

    print("\n" + "=" * 82)
    print("CORRELATIONS and PARTIAL CORRELATIONS with AUC (across all methods)")
    print("=" * 82)
    print(f"{'dataset':18s} {'r(shift)':>9s} {'r(div)':>8s} {'r(logdet)':>10s} "
          f"{'shift|div':>10s} {'div|shift':>10s} {'beta_sh':>8s} {'beta_dv':>8s}")
    pool = {"shift": [], "div": [], "ld": [], "auc": []}
    for ds in datasets:
        rs = [r for r in recs if r["dataset"] == ds]
        au = np.array([r["auc"] for r in rs])
        sh = np.array([r["mean_shift"] for r in rs])
        dv = np.array([r["diversity"] for r in rs])
        ld = np.array([r.get("logdet_cov", np.nan) for r in rs])
        ldok = np.isfinite(ld)
        r_sh = pearsonr(sh, au)[0] if np.std(sh) else float("nan")
        r_dv = pearsonr(dv, au)[0] if np.std(dv) else float("nan")
        r_ld = pearsonr(ld[ldok], au[ldok])[0] if ldok.sum() > 2 and np.std(ld[ldok]) else float("nan")
        pc_sh = partial_corr(sh, au, dv)   # shift controlling for diversity
        pc_dv = partial_corr(dv, au, sh)   # diversity controlling for shift
        betas = std_reg(au, np.c_[sh, dv])
        summary["per_dataset"][ds] = {
            "r_shift": float(r_sh), "r_diversity": float(r_dv), "r_logdet": float(r_ld),
            "partial_shift_given_div": float(pc_sh),
            "partial_div_given_shift": float(pc_dv),
            "beta_shift": float(betas[0]), "beta_diversity": float(betas[1])}
        print(f"{ds:18s} {r_sh:>9.2f} {r_dv:>8.2f} {r_ld:>10.2f} "
              f"{pc_sh:>10.2f} {pc_dv:>10.2f} {betas[0]:>8.2f} {betas[1]:>8.2f}")
        # pooled z-scored
        if np.std(au) > 0:
            z = lambda a: (a - a.mean()) / a.std()
            pool["auc"].extend(z(au)); pool["shift"].extend(z(sh) if np.std(sh) else np.zeros_like(sh))
            pool["div"].extend(z(dv) if np.std(dv) else np.zeros_like(dv))

    ps, pd_, pa = map(np.array, (pool["shift"], pool["div"], pool["auc"]))
    pc_sh = partial_corr(ps, pa, pd_); pc_dv = partial_corr(pd_, pa, ps)
    betas = std_reg(pa, np.c_[ps, pd_])
    summary["pooled"] = {"r_shift": float(pearsonr(ps, pa)[0]),
                         "r_diversity": float(pearsonr(pd_, pa)[0]),
                         "partial_shift_given_div": float(pc_sh),
                         "partial_div_given_shift": float(pc_dv),
                         "beta_shift": float(betas[0]), "beta_diversity": float(betas[1])}
    print("-" * 82)
    print(f"{'POOLED(z)':18s} {pearsonr(ps,pa)[0]:>9.2f} {pearsonr(pd_,pa)[0]:>8.2f} "
          f"{'--':>10s} {pc_sh:>10.2f} {pc_dv:>10.2f} {betas[0]:>8.2f} {betas[1]:>8.2f}")
    print("\nLegend: r(shift)=raw corr of mean_shift with AUC; div|shift=partial corr of")
    print("diversity with AUC controlling for shift; beta_*=standardized regression coef.")

    out = RESULTS_DIR / "exp16_analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
