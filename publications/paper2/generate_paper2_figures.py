"""
Generate Paper 2 figures from the exp14 analysis summary (measured data only).
Reads experiments/results/exp14_analysis_summary.json.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SUMMARY = ROOT.parent.parent / "experiments" / "results" / "exp14_analysis_summary.json"

S = json.loads(SUMMARY.read_text())
H1 = S["H1_stability"]
datasets = sorted(H1.keys())
K = [16, 32, 64, 128, 256]
colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

# --- Figure 1: stability -- mean AUC (left) and CV% (right) vs k -------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ds, c in zip(datasets, colors):
    ks = sorted(int(k) for k in H1[ds].keys())
    mean = [H1[ds][str(k)]["mean"] for k in ks]
    ci = [H1[ds][str(k)].get("ci95", 0.0) for k in ks]
    cv = [100 * H1[ds][str(k)]["cv"] for k in ks]
    ax1.errorbar(ks, mean, yerr=ci, fmt='o-', color=c, label=ds, capsize=3, elinewidth=1)
    ax2.plot(ks, cv, 'o-', color=c, label=ds)
ax1.set_xscale('log', base=2); ax2.set_xscale('log', base=2)
ax1.set_xticks(K); ax1.set_xticklabels(K); ax2.set_xticks(K); ax2.set_xticklabels(K)
ax1.set_xlabel('context size k', fontweight='bold'); ax1.set_ylabel('mean ROC AUC', fontweight='bold')
ax2.set_xlabel('context size k', fontweight='bold'); ax2.set_ylabel('coefficient of variation (%)', fontweight='bold')
ax1.set_title('Accuracy rises with k', fontweight='bold')
ax2.set_title('Variability falls with k', fontweight='bold')
ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
fig.suptitle('H1: Larger contexts are more accurate and more stable', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / "p2_fig1_stability.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "p2_fig1_stability.png", dpi=300, bbox_inches='tight')
print("Saved p2_fig1_stability")

# --- Figure 2: representativeness -- feat_shift correlation + contrast -------
H2 = S["H2_representativeness"]
per = H2["per_dataset"]
contrast = H2["representativeness_contrast"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
# left: |Pearson r| of feat_shift vs AUC, per dataset
ds_sorted = sorted(per.keys(), key=lambda d: abs(per[d]["feat_shift"]["pearson"] or 0), reverse=True)
rvals = [per[d]["feat_shift"]["pearson"] for d in ds_sorted]
ax1.barh(ds_sorted, rvals, color='#1f77b4')
ax1.set_xlabel('Pearson r (feature-mean shift vs AUC)', fontweight='bold')
ax1.set_title('More feature shift -> lower AUC\n(negative = as hypothesized)', fontweight='bold')
ax1.axvline(0, color='k', lw=0.8)
ax1.grid(True, alpha=0.3, axis='x')
# right: most vs least representative third
ds2 = sorted(contrast.keys())
most = [contrast[d]["most_repr_auc"] for d in ds2]
least = [contrast[d]["least_repr_auc"] for d in ds2]
x = np.arange(len(ds2)); w = 0.38
ax2.bar(x - w/2, most, w, label='most representative third', color='#2ca02c')
ax2.bar(x + w/2, least, w, label='least representative third', color='#d62728')
ax2.set_xticks(x); ax2.set_xticklabels(ds2, rotation=30, ha='right', fontsize=8)
ax2.set_ylabel('mean ROC AUC', fontweight='bold')
ax2.set_title('Representative contexts do better (all 6)', fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
fig.suptitle('H2: Accuracy tracks how well the context matches the full data', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / "p2_fig2_representativeness.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "p2_fig2_representativeness.png", dpi=300, bbox_inches='tight')
print("Saved p2_fig2_representativeness")

# --- Figure 3: selection method -- AUC nearly equal, time very different -----
H3 = S["H3_selection"]
ds3 = sorted(H3.keys())
methods = ["random", "kmeans", "fps"]
mcolors = {"random": "#1f77b4", "kmeans": "#ff7f0e", "fps": "#2ca02c"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(ds3)); w = 0.26
for i, m in enumerate(methods):
    aucs = [H3[d][m]["mean_auc"] for d in ds3]
    cis = [H3[d][m].get("ci95", 0.0) for d in ds3]
    ax1.bar(x + (i - 1) * w, aucs, w, label=m, color=mcolors[m],
            yerr=cis, capsize=2, error_kw={"elinewidth": 1})
ax1.set_xticks(x); ax1.set_xticklabels(ds3, rotation=30, ha='right', fontsize=8)
ax1.set_ylabel('mean ROC AUC', fontweight='bold'); ax1.set_ylim([0.6, 1.0])
ax1.set_title('Accuracy: methods nearly tied', fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')
# selection time (mean across datasets), log
mean_time = {m: np.mean([H3[d][m]["mean_select_sec"] for d in ds3]) for m in methods}
ax2.bar(range(len(methods)), [mean_time[m] for m in methods],
        color=[mcolors[m] for m in methods])
ax2.set_yscale('log'); ax2.set_xticks(range(len(methods))); ax2.set_xticklabels(methods)
ax2.set_ylabel('mean selection time (s, log)', fontweight='bold')
ax2.set_title('Cost: random is orders of magnitude cheaper', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i, m in enumerate(methods):
    ax2.annotate(f'{mean_time[m]:.4f}s', xy=(i, mean_time[m]), xytext=(0, 5),
                 textcoords='offset points', ha='center', fontsize=9)
fig.suptitle('H3: Expensive selection adds little over random', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / "p2_fig3_selection.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "p2_fig3_selection.png", dpi=300, bbox_inches='tight')
print("Saved p2_fig3_selection")

print("Done.")
