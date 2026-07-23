"""
Generate publication figures from MEASURED data only.

All numbers below are copied directly from experiments/results/*.json:
- exp3_scaling_20260722_102707.json    (Obs 1: accuracy/time vs n, CPU)
- exp6_adaptive_k_20260722_124640.json (Obs 2: AUC vs k on electricity)
- exp2 / exp8 / exp10 / exp11           (Obs 3: selection methods)
- exp9_multihead_prototypes            (Obs 4: budget split across heads)

No GPU memory, no OOM, no Nature-baseline framing: none of that was measured.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Full-context AUC vs training size (Observation 1). CPU, seed 42.
# ---------------------------------------------------------------------------
bank_n     = np.array([1000, 2000, 5000, 10000, 20000, 40689])
bank_full  = np.array([0.9159, 0.9299, 0.9377, 0.9439, 0.9439, 0.9465])
bank_proto = np.array([0.9098, 0.8879, 0.8856, 0.9048, 0.8958, 0.9023])
bank_lgbm  = np.array([0.8743, 0.8933, 0.9115, 0.9252, 0.9326, 0.9383])

adult_n    = np.array([1000, 2000, 5000, 10000, 20000, 43957])
adult_full = np.array([0.8952, 0.9050, 0.9099, 0.9115, 0.9146, 0.9174])
adult_lgbm = np.array([0.8843, 0.8965, 0.9121, 0.9196, 0.9244, 0.9277])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name, n, full, lgbm, proto in [
    (axes[0], "bank-marketing", bank_n, bank_full, bank_lgbm, bank_proto),
    (axes[1], "adult", adult_n, adult_full, adult_lgbm, None),
]:
    ax.plot(n, full, 'o-', lw=2.2, ms=7, label='full TabPFN', color='#1f77b4')
    if proto is not None:
        ax.plot(n, proto, 's--', lw=1.8, ms=6, label='TabPFN k=512', color='#ff7f0e')
    ax.plot(n, lgbm, '^--', lw=1.8, ms=6, label='LightGBM (tuned)', color='#2ca02c')
    ax.set_xscale('log')
    ax.set_xlabel('training samples (n)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
    ax.set_title(f'{name} (CPU, seed 42)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
fig.suptitle('Observation 1: Full-context AUC vs training size (diminishing returns)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_scaling.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig1_scaling.png", dpi=300, bbox_inches='tight')
print("Saved: fig1_scaling")

# ---------------------------------------------------------------------------
# Figure 2: AUC vs k on electricity (Observation 2).
# ---------------------------------------------------------------------------
k_vals = np.array([256, 512, 1024, 2048])
k_auc  = np.array([0.8483, 0.8606, 0.8887, 0.9181])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_vals, k_auc, 'o-', lw=2.4, ms=9, color='#1f77b4')
for x, y in zip(k_vals, k_auc):
    ax.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks(k_vals)
ax.set_xticklabels([str(k) for k in k_vals])
ax.set_xlabel('prototype count k', fontsize=11, fontweight='bold')
ax.set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
ax.set_title('Observation 2: AUC vs k on electricity (no plateau by k=2048)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_k_sweep.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig2_k_sweep.png", dpi=300, bbox_inches='tight')
print("Saved: fig2_k_sweep")

# ---------------------------------------------------------------------------
# Figure 3: Selection methods - accuracy similar, random far cheaper (Obs 3).
# ---------------------------------------------------------------------------
methods = ['random', 'FPS', 'K-Means', 'K-Means\n(bal.)']
diabetes_auc = [0.9081, 0.9143, 0.9165, 0.9116]
vehicle_auc  = [0.9740, 0.9781, 0.9762, 0.9778]
# representative selection times (s), log scale
sel_time     = [0.0002, 0.01, 0.20, 0.22]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(methods))
w = 0.38
ax1.bar(x - w/2, diabetes_auc, w, label='diabetes', color='#1f77b4')
ax1.bar(x + w/2, vehicle_auc, w, label='vehicle', color='#ff7f0e')
ax1.set_xticks(x); ax1.set_xticklabels(methods)
ax1.set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
ax1.set_ylim([0.85, 1.0])
ax1.set_title('Accuracy (k=512, 3 seeds): differences small', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(x, sel_time, color='#2ca02c')
ax2.set_yscale('log')
ax2.set_xticks(x); ax2.set_xticklabels(methods)
ax2.set_ylabel('selection time (s, log)', fontsize=11, fontweight='bold')
ax2.set_title('Selection cost: random ~1000x cheaper', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for xi, t in zip(x, sel_time):
    ax2.annotate(f'{t:g}s', xy=(xi, t), xytext=(0, 5),
                 textcoords='offset points', ha='center', fontsize=9)
fig.suptitle('Observation 3: Prototype selection method', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_selection.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig3_selection.png", dpi=300, bbox_inches='tight')
print("Saved: fig3_selection")

# ---------------------------------------------------------------------------
# Figure 4: Splitting a fixed budget across heads hurts (Observation 4).
# ---------------------------------------------------------------------------
heads = np.array([1, 2, 4, 8])
elec_h = np.array([0.8887, 0.8661, 0.8530, 0.8320])
bank_h = np.array([0.9193, 0.8982, 0.8664, 0.8439])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(heads, elec_h, 'o-', lw=2.2, ms=8, label='electricity', color='#1f77b4')
ax.plot(heads, bank_h, 's-', lw=2.2, ms=8, label='bank-marketing', color='#ff7f0e')
ax.set_xscale('log', base=2)
ax.set_xticks(heads); ax.set_xticklabels([str(h) for h in heads])
ax.set_xlabel('number of heads (total budget k=1024 fixed)', fontsize=11, fontweight='bold')
ax.set_ylabel('mean ROC AUC', fontsize=11, fontweight='bold')
ax.set_title('Observation 4: Splitting the budget across heads reduces accuracy',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_multihead.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig4_multihead.png", dpi=300, bbox_inches='tight')
print("Saved: fig4_multihead")

print(f"\nAll figures written to {FIGURES_DIR}")
