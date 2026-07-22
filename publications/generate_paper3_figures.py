"""
Generate publication-ready figures for Paper 3: Efficient Prototype Selection

Figures:
1. Accuracy comparison: All methods on electricity (bar chart)
2. Computation time comparison (bar chart, log scale)
3. Accuracy vs computation trade-off (scatter + efficiency frontier)
4. Stability across runs (error bars)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
FIGURES_DIR = Path(__file__).parent / "paper_figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Data from experiments (TRACKER.md Part 21 + exp2, exp8 results)
methods = ['Fixed k=512', 'Importance', 'FPS', 'K-Means', 'Random']
auc_electricity = np.array([0.8606, 0.8924, 0.9056, 0.9078, 0.9181])
auc_bankmarketing = np.array([0.8579, 0.9008, 0.9134, 0.9187, 0.9198])
compute_time = np.array([0.02, 5.67, 89.34, 12.45, 0.02])  # seconds
std_dev = np.array([0.10, 0.58, 0.15, 0.12, 0.08])  # percentage

# Figure 1: Accuracy Comparison (electricity)
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#d62728', '#ff7f0e', '#9467bd', '#17becf', '#1f77b4']
bars = ax.bar(methods, auc_electricity, color=colors, edgecolor='black', linewidth=1.2)

# Add baseline line
ax.axhline(y=0.8606, color='#d62728', linestyle='--', linewidth=2, alpha=0.5,
           label='Fixed k=512 baseline')

# Highlight winner
bars[-1].set_edgecolor('green')
bars[-1].set_linewidth(3)

ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Selection Method Accuracy Comparison (electricity, k=2048)', fontsize=14, fontweight='bold')
ax.set_ylim([0.85, 0.93])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, auc in zip(bars, auc_electricity):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.002, f'{auc:.4f}',
            ha='center', fontsize=10, fontweight='bold')

# Add improvement % vs baseline
baseline_auc = auc_electricity[0]
for bar, auc in zip(bars, auc_electricity):
    if auc > baseline_auc:
        improvement = ((auc - baseline_auc) / baseline_auc) * 100
        ax.text(bar.get_x() + bar.get_width()/2, baseline_auc - 0.003, f'+{improvement:.1f}%',
                ha='center', fontsize=9, fontweight='bold', color='green')

ax.text(0.98, 0.02, 'Winner: Random selection\n(0.9181 AUC, 0x preprocessing)',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper3_fig1_accuracy_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper3_fig1_accuracy_comparison.png", dpi=300, bbox_inches='tight')
print("Saved: paper3_fig1_accuracy_comparison.pdf / .png")

# Figure 2: Computation Time Comparison (log scale)
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(methods, compute_time, color=colors, edgecolor='black', linewidth=1.2)

# Highlight random (fastest)
bars[-1].set_edgecolor('green')
bars[-1].set_linewidth(3)

ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12, fontweight='bold')
ax.set_title('Prototype Selection Time Comparison (k=2048, n=40K)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y', which='both')

# Add value labels
for bar, time in zip(bars, compute_time):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height * 1.2, f'{time:.2f}s',
            ha='center', fontsize=10, fontweight='bold')

# Add speedup factors
random_time = compute_time[-1]
for i, (bar, time) in enumerate(zip(bars[:-1], compute_time[:-1])):
    speedup = time / random_time
    ax.text(bar.get_x() + bar.get_width()/2, time * 0.7, f'{speedup:.0f}x slower',
            ha='center', fontsize=8, color='red', fontweight='bold')

ax.text(0.98, 0.02, 'Random is 623x faster than K-Means\n0.02 sec vs 12.45 sec',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper3_fig2_computation_time.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper3_fig2_computation_time.png", dpi=300, bbox_inches='tight')
print("Saved: paper3_fig2_computation_time.pdf / .png")

# Figure 3: Accuracy vs Computation (Efficiency Frontier)
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(compute_time, auc_electricity, s=200, alpha=0.7, c=range(len(methods)),
                     cmap='viridis', edgecolors='black', linewidth=1.5)

# Annotate each point
for i, (time, auc, method) in enumerate(zip(compute_time, auc_electricity, methods)):
    ax.annotate(method, xy=(time, auc), xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Draw efficiency frontier (Pareto curve)
sorted_idx = np.argsort(compute_time)
ax.plot(compute_time[sorted_idx], auc_electricity[sorted_idx], 'r--', linewidth=2, alpha=0.5,
        label='Efficiency frontier')

ax.set_xlabel('Computation Time (seconds, log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Accuracy-Efficiency Trade-off: Prototype Selection Methods', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim([0.85, 0.93])

# Highlight efficient frontier point (Random)
ax.scatter([compute_time[-1]], [auc_electricity[-1]], s=300, marker='*', color='green',
           edgecolors='darkgreen', linewidth=2, zorder=5, label='Optimal (Random)')
ax.legend(fontsize=11, loc='lower right')

ax.text(0.02, 0.98, 'Random dominates: highest AUC, lowest cost',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper3_fig3_efficiency_frontier.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper3_fig3_efficiency_frontier.png", dpi=300, bbox_inches='tight')
print("Saved: paper3_fig3_efficiency_frontier.pdf / .png")

# Figure 4: Accuracy Across Both Datasets (dual bar chart)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Electricity
bars1 = ax1.bar(methods, auc_electricity, color=colors, edgecolor='black', linewidth=1.2)
ax1.axhline(y=0.8606, color='#d62728', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
ax1.set_title('(a) Electricity (40K rows, k=2048)', fontsize=12, fontweight='bold')
ax1.set_ylim([0.85, 0.93])
ax1.grid(True, alpha=0.3, axis='y')
for bar, auc in zip(bars1, auc_electricity):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.002, f'{auc:.4f}',
             ha='center', fontsize=9, fontweight='bold')

# Bank-marketing
bars2 = ax2.bar(methods, auc_bankmarketing, color=colors, edgecolor='black', linewidth=1.2)
ax2.axhline(y=0.8579, color='#d62728', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
ax2.set_title('(b) Bank-Marketing (40K rows, k=2048)', fontsize=12, fontweight='bold')
ax2.set_ylim([0.85, 0.93])
ax2.grid(True, alpha=0.3, axis='y')
for bar, auc in zip(bars2, auc_bankmarketing):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.002, f'{auc:.4f}',
             ha='center', fontsize=9, fontweight='bold')

# Highlight random in both
bars1[-1].set_edgecolor('green')
bars1[-1].set_linewidth(3)
bars2[-1].set_edgecolor('green')
bars2[-1].set_linewidth(3)

plt.suptitle('Robustness Across Datasets: Random Wins Consistently', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper3_fig4_dual_datasets.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper3_fig4_dual_datasets.png", dpi=300, bbox_inches='tight')
print("Saved: paper3_fig4_dual_datasets.pdf / .png")

# Figure 5: Stability (Error bars across runs)
fig, ax = plt.subplots(figsize=(10, 6))

methods_stability = ['Random', 'K-Means', 'Importance']
auc_vals = [0.9181, 0.9078, 0.8924]
std_vals = [0.08, 0.12, 0.58]  # percentage

# Convert std percentages to absolute AUC std
abs_std = [auc_vals[i] * std_vals[i] / 100 for i in range(len(methods_stability))]

bars = ax.bar(methods_stability, auc_vals, yerr=abs_std, color=['#1f77b4', '#17becf', '#ff7f0e'],
              capsize=10, edgecolor='black', linewidth=1.2, error_kw={'linewidth': 2})

# Highlight random (smallest error bar)
bars[0].set_edgecolor('green')
bars[0].set_linewidth(3)

ax.set_ylabel('AUC Score with Standard Deviation', fontsize=12, fontweight='bold')
ax.set_title('Stability Across 10 Random Seeds (electricity, k=2048)', fontsize=14, fontweight='bold')
ax.set_ylim([0.87, 0.93])
ax.grid(True, alpha=0.3, axis='y')

# Add std labels
for i, (bar, std) in enumerate(zip(bars, std_vals)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + abs_std[i] + 0.002, f'Std: {std:.2f}%',
            ha='center', fontsize=10, fontweight='bold')

ax.text(0.98, 0.02, 'Random is most stable\n(0.08% std, simplest method)',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "paper3_fig5_stability.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "paper3_fig5_stability.png", dpi=300, bbox_inches='tight')
print("Saved: paper3_fig5_stability.pdf / .png")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("Total: 10 figure files (5 sets of PDF + PNG)")
