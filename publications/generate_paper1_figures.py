"""
Generate publication-ready figures for Paper 1: Scaling TabPFN Beyond 10K Samples

Figures:
1. Scaling curve: AUC vs dataset size (n_rows)
2. GPU memory usage vs dataset size
3. Baseline comparison: TabPFN vs LightGBM accuracy
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
FIGURES_DIR = Path(__file__).parent / "paper_figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Data from experiments (TRACKER.md Part 17)
n_rows = np.array([1000, 2000, 5000, 10000, 20000, 40689])
tabpfn_full_auc = np.array([0.9159, 0.9299, 0.9377, 0.9439, 0.9439, 0.9465])
tabpfn_proto_auc = np.array([0.9098, 0.8879, 0.8856, 0.9048, 0.8958, 0.9023])
lightgbm_auc = np.array([0.8743, 0.8933, 0.9115, 0.9252, 0.9326, 0.9383])

gpu_memory_gb = np.array([1.2, 3.8, 7.6])  # for n=1K, 10K, 40K
gpu_memory_n = np.array([1000, 10000, 40689])

# Figure 1: Scaling Curve
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(n_rows, tabpfn_full_auc, 'o-', linewidth=2.5, markersize=8,
        label='Full TabPFN', color='#1f77b4')
ax.plot(n_rows, tabpfn_proto_auc, 's--', linewidth=2, markersize=7,
        label='TabPFN (k=512)', color='#ff7f0e')
ax.plot(n_rows, lightgbm_auc, '^--', linewidth=2, markersize=7,
        label='LightGBM (tuned)', color='#2ca02c')

# Nature paper baseline (constant line at 0.939)
ax.axhline(y=0.939, color='#d62728', linestyle='--', linewidth=2,
           label='Nature Paper (0.939)', alpha=0.8)

ax.set_xlabel('Dataset Size (number of training samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Scaling Behavior: TabPFN Accuracy Beyond 10K Samples', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0.85, 0.96])

# Annotate key point
ax.annotate('40K rows: 0.9465\n(exceeds Nature 0.939)',
            xy=(40689, 0.9465), xytext=(20000, 0.945),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold', ha='left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper1_fig1_scaling_curve.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper1_fig1_scaling_curve.png", dpi=300, bbox_inches='tight')
print("Saved: paper1_fig1_scaling_curve.pdf / .png")

# Figure 2: GPU Memory Usage
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(gpu_memory_n, gpu_memory_gb, 'o-', linewidth=2.5, markersize=10,
        label='Full TabPFN (RTX 4060)', color='#1f77b4')
ax.axhline(y=8.0, color='#d62728', linestyle='--', linewidth=2,
           label='RTX 4060 Total (8 GB)', alpha=0.8)

ax.fill_between(gpu_memory_n, 0, gpu_memory_gb, alpha=0.2, color='#1f77b4', label='GPU Memory Used')
ax.fill_between(gpu_memory_n, gpu_memory_gb, 8.0, alpha=0.1, color='#d62728', label='Available Headroom')

ax.set_xlabel('Dataset Size (number of training samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
ax.set_title('GPU Memory Scaling: Full TabPFN on RTX 4060', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim([0, 9])

# Annotate utilization percentages
for n, mem in zip(gpu_memory_n, gpu_memory_gb):
    pct = (mem / 8.0) * 100
    ax.annotate(f'{pct:.0f}%', xy=(n, mem), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper1_fig2_gpu_memory.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper1_fig2_gpu_memory.png", dpi=300, bbox_inches='tight')
print("Saved: paper1_fig2_gpu_memory.pdf / .png")

# Figure 3: Baseline Comparison (Bar chart)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(n_rows))
width = 0.25

bars1 = ax.bar(x - width, tabpfn_full_auc, width, label='Full TabPFN', color='#1f77b4')
bars2 = ax.bar(x, tabpfn_proto_auc, width, label='TabPFN (k=512)', color='#ff7f0e')
bars3 = ax.bar(x + width, lightgbm_auc, width, label='LightGBM', color='#2ca02c')

ax.axhline(y=0.939, color='#d62728', linestyle='--', linewidth=2, alpha=0.7, label='Nature Paper (0.939)')

ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Method Comparison at Different Dataset Sizes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{n//1000:.0f}K' for n in n_rows])
ax.legend(fontsize=11)
ax.set_ylim([0.85, 0.96])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper1_fig3_baseline_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper1_fig3_baseline_comparison.png", dpi=300, bbox_inches='tight')
print("Saved: paper1_fig3_baseline_comparison.pdf / .png")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("Total: 6 figure files (3 sets of PDF + PNG)")
