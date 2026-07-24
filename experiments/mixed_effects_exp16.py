"""
Mixed-effects inference for the H2 disentangling (exp16).

Fits AUC ~ diversity + shift + (1|dataset) and, if it converges,
AUC ~ diversity + shift + (1 + diversity | dataset), accounting for
dataset-level variation. Reports coefficient, SE, z, p-value, and 95% CI for
each fixed effect. Predictors are z-scored WITHIN dataset so the fixed effects
describe the within-dataset relationship on a common scale.

Usage: python mixed_effects_exp16.py [path_to_exp16_json]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RESULTS_DIR = Path(__file__).parent / "results"


def newest():
    fs = sorted(RESULTS_DIR.glob("exp16_disentangle_*.json"), key=lambda p: p.stat().st_mtime)
    if not fs:
        sys.exit("No exp16_disentangle_*.json found.")
    return fs[-1]


def zwithin(df, col):
    return df.groupby("dataset")[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0))


def report(res, name):
    print(f"\n--- {name} ---")
    print(f"  n_obs={int(res.nobs)}  groups={res.model.n_groups}")
    ci = res.conf_int()
    for term in ["diversity", "shift"]:
        if term in res.params.index:
            coef = res.params[term]; se = res.bse[term]; p = res.pvalues[term]
            lo, hi = ci.loc[term]
            print(f"  {term:10s} beta={coef:+.4f}  SE={se:.4f}  z={coef/se:+.2f}  "
                  f"p={p:.3g}  95% CI=[{lo:+.4f}, {hi:+.4f}]")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest()
    recs = [r for r in json.loads(Path(path).read_text()) if r.get("status") == "OK"]
    df = pd.DataFrame(recs)[["dataset", "auc", "diversity", "mean_shift"]].dropna()
    df = df.rename(columns={"mean_shift": "shift"})
    df["diversity"] = zwithin(df, "diversity")
    df["shift"] = zwithin(df, "shift")
    # z-score AUC within dataset too, so the outcome scale is comparable
    df["auc_z"] = df.groupby("dataset")["auc"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0))
    print(f"Loaded {len(df)} OK contexts across {df['dataset'].nunique()} datasets from {path.name}")

    # Random-intercept model
    m1 = smf.mixedlm("auc_z ~ diversity + shift", df, groups=df["dataset"])
    r1 = m1.fit(reml=True, method="lbfgs")
    report(r1, "AUC ~ diversity + shift + (1 | dataset)")

    # Random-slope model (diversity slope varies by dataset)
    try:
        m2 = smf.mixedlm("auc_z ~ diversity + shift", df, groups=df["dataset"],
                         re_formula="~diversity")
        r2 = m2.fit(reml=True, method="lbfgs")
        report(r2, "AUC ~ diversity + shift + (1 + diversity | dataset)")
    except Exception as e:
        print(f"\nRandom-slope model did not converge: {e!r}")

    # Save a compact summary
    ci = r1.conf_int()
    out = {
        "source": path.name, "n_obs": int(r1.nobs), "n_groups": int(r1.model.n_groups),
        "model": "auc_z ~ diversity + shift + (1|dataset)",
        "diversity": {"beta": float(r1.params["diversity"]), "se": float(r1.bse["diversity"]),
                      "p": float(r1.pvalues["diversity"]),
                      "ci": [float(ci.loc["diversity"][0]), float(ci.loc["diversity"][1])]},
        "shift": {"beta": float(r1.params["shift"]), "se": float(r1.bse["shift"]),
                  "p": float(r1.pvalues["shift"]),
                  "ci": [float(ci.loc["shift"][0]), float(ci.loc["shift"][1])]},
    }
    (RESULTS_DIR / "exp16_mixedeffects_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nSummary written to: {RESULTS_DIR / 'exp16_mixedeffects_summary.json'}")


if __name__ == "__main__":
    main()
