# src/symmetry/visualization/contour_model_plot.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.symmetry.utils import get_r2_contour

# ── Load data ─────────────────────────────────────────────────────────────────
path = os.environ.get("RESULTS_DIR", "results/symmetry/nested_model")
file = f'{path}/mir_accumulation_73000img_all_reduced.txt'
R2_mean_contour = get_r2_contour(file, 'mean')
LAYER_NAMES = [
    '1-1', '1-2',
    '2-1', '2-2',
    '3-1', '3-2', '3-3',
    '4-1', '4-2', '4-3',
    '5-1', '5-2', '5-3',
]
def fmt(x, _):
    return f'{x:.4f}'
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
x = np.arange(len(LAYER_NAMES))
ax.plot(x, R2_mean_contour,
        marker='s', linewidth=2.0, markersize=7,
        color='#1524f8', markerfacecolor='white',
        markeredgecolor='#1524f8', markeredgewidth=1.5,
        linestyle='--', label='R² Reduced (mean)')
ax.fill_between(x, R2_mean_contour, alpha=0.12, color='#1524f8')
ax.set_xticks(x)
ax.set_xticklabels(LAYER_NAMES, rotation=45, ha='right', fontsize=10)
ax.set_xlabel('VGG16 Convolutional Layer', fontsize=11)
ax.set_ylabel('Mean R²', fontsize=11)
ax.set_title('Mean R² Reduced Across VGG16 Layers for 73,000 Images', fontsize=13, fontweight='bold', color='#1a1a2e')
ax.tick_params(axis='y', labelsize=10)
ax.yaxis.set_major_formatter(FuncFormatter(fmt))
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#1524f8")
plt.tight_layout()

plots_dir = os.environ.get("PLOTS_DIR", "results/symmetry/plots")
out = os.path.join(plots_dir, 'r2_reduced_mean.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print(f"Saved: {out}")
