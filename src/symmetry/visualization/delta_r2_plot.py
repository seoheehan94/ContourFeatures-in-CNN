# src/symmetry/visualization/delta_r2_plot.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

from src.symmetry.utils import read_values, get_r2_contour

# ── Data extracted directly from the provided output ─────────────────────────

LAYER_NAMES = [
    '1-1', '1-2',
    '2-1', '2-2',
    '3-1', '3-2', '3-3',
    '4-1', '4-2', '4-3',
    '5-1', '5-2', '5-3',
]

# ── Load values ─────────────────────────
mir_max_delta = read_values('Delta_R2',  sym_type='mir', mode='max')
par_max_delta = read_values('Delta_R2',  sym_type='par', mode='max')
tap_max_delta = read_values('Delta_R2',  sym_type='tap', mode='max')

mir_mean_delta = read_values('Delta_R2',  sym_type='mir', mode='mean')
par_mean_delta = read_values('Delta_R2',  sym_type='par', mode='mean')
tap_mean_delta = read_values('Delta_R2',  sym_type='tap', mode='mean')

mir_mean_R2 = read_values('R2_full',  sym_type='mir', mode='mean')
par_mean_R2 = read_values('R2_full',  sym_type='par', mode='mean')
tap_mean_R2 = read_values('R2_full',  sym_type='tap', mode='mean')

path = os.environ.get("RESULTS_DIR", "results/symmetry/nested_model")
file = f'{path}/mir_accumulation_73000img_all_reduced.txt'
R2_mean_contour = get_r2_contour(file, 'mean')


# ── Mirror (mir) data ─────────────────────────────────────────────────────────
mir_reduced_mean = R2_mean_contour

mir_data = {
    'contour': {
        'mean_dr2': mir_mean_delta['contour'],
        'max_dr2':  mir_max_delta['contour'],
        'r2_full':  mir_mean_R2['contour'],
    },
    'medial': {
        'mean_dr2': mir_mean_delta['medial'],
        'max_dr2':  mir_max_delta['medial'],
        'r2_full':  mir_mean_R2['medial'],
    },
    'area': {
        'mean_dr2': mir_mean_delta['area'],
        'max_dr2':  mir_max_delta['area'],
        'r2_full':  mir_mean_R2['area'],
    },
}

par_reduced_mean = R2_mean_contour

par_data = {
    'contour': {
        'mean_dr2': par_mean_delta['contour'],
        'max_dr2':  par_max_delta['contour'],
        'r2_full':  par_mean_R2['contour'],
    },
    'medial': {
        'mean_dr2': par_mean_delta['medial'],
        'max_dr2':  par_max_delta['medial'],
        'r2_full':  par_mean_R2['medial'],
    },
    'area': {
        'mean_dr2': par_mean_delta['area'],
        'max_dr2':  par_max_delta['area'],
        'r2_full':  par_mean_R2['area'],
    },
}

tap_reduced_mean = R2_mean_contour

tap_data = {
    'contour': {
        'mean_dr2': tap_mean_delta['contour'],
        'max_dr2':  tap_max_delta['contour'],
        'r2_full':  tap_mean_R2['contour'],
    },
    'medial': {
        'mean_dr2': tap_mean_delta['medial'],
        'max_dr2':  tap_max_delta['medial'],
        'r2_full':  tap_mean_R2['medial'],
    },
    'area': {
        'mean_dr2': tap_mean_delta['area'],
        'max_dr2':  tap_max_delta['area'],
        'r2_full':  tap_mean_R2['area'],
    },
}

all_sym_data = {
    'Mirror':   mir_data,
    'Parallel': par_data,
    'Taper':    tap_data,
}

reduced_data = {
    'Mirror':   mir_reduced_mean,
    'Parallel': par_reduced_mean,
    'Taper':    tap_reduced_mean,
}

# ── Compute global y-axis limits ──────────────────────────────────────────────
def compute_ylims():
    mean_vals, max_vals, r2_vals = [], [], []
    for sym_data in all_sym_data.values():
        for t_data in sym_data.values():
            mean_vals.extend(t_data['mean_dr2'])
            max_vals.extend(t_data['max_dr2'])
            r2_vals.extend(t_data['r2_full'])
    mean_ylim = (0.0, max(mean_vals) + 0.0005)
    max_ylim  = (0.0, max(max_vals)  + 0.002)
    r2_ylim   = (0.0, max(r2_vals)   + 0.002)
    return mean_ylim, max_ylim, r2_ylim

mean_ylim, max_ylim, r2_ylim = compute_ylims()

# ── Styling ───────────────────────────────────────────────────────────────────
SYM_COLORS = {
    'Mirror':   {'line': '#1565C0', 'fill': '#1565C0', 'marker': '#E3F2FD'},
    'Parallel': {'line': '#2E7D32', 'fill': '#2E7D32', 'marker': '#E8F5E9'},
    'Taper':    {'line': '#C62828', 'fill': '#C62828', 'marker': '#FFEBEE'},
}

def fmt(x, _):
    return f'{x:.4f}'

# ── plot_panel: removed title parameter (replaced by row/col headers) ─────────
def plot_panel(ax, layer_names, values, color, fill_color, marker_color,
               ylabel, ylim, show_reduced=None):
    x = np.arange(len(layer_names))
    ax.plot(x, values, marker='o', linewidth=2.2, markersize=7,
            color=color, markerfacecolor=marker_color,
            markeredgecolor=color, markeredgewidth=1.8, zorder=3)
    ax.fill_between(x, values, alpha=0.12, color=fill_color, zorder=1)

    if show_reduced is not None:
        ax.plot(x, show_reduced, marker='s', linewidth=1.5, markersize=5,
                color='#666666', markerfacecolor='white',
                markeredgecolor='#666666', markeredgewidth=1.2,
                linestyle='--', alpha=0.7, zorder=2)

    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── MAP type display labels ───────────────────────────────────────────────────
MAP_TYPES   = ['contour', 'medial', 'area']
TYPE_LABELS = {'contour': 'Contour Map', 'medial': 'Medial Axis Map', 'area': 'Area Map'}

plots_dir = os.environ.get("PLOTS_DIR", "results/symmetry/plots")

# ── Generate one figure per symmetry type ────────────────────────────────────
for sym_label, sym_data in all_sym_data.items():

    color_cfg    = SYM_COLORS[sym_label]
    reduced_vals = reduced_data[sym_label]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#FFFFFF')

    # Plots now stretch fully across; legend floats inside top-right corner
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.30,
                           top=0.88, bottom=0.08,
                           left=0.09, right=0.97)

    for col, mt in enumerate(MAP_TYPES):
        t_data = sym_data[mt]

        # Row 0: R² Full & R² Reduced
        ax0 = fig.add_subplot(gs[0, col])
        plot_panel(ax0, LAYER_NAMES,
                   t_data['r2_full'],
                   color_cfg['line'], color_cfg['fill'], color_cfg['marker'],
                   'Mean R²', r2_ylim,
                   show_reduced=reduced_vals)
        
        # Row 1: Mean ΔR²
        ax1 = fig.add_subplot(gs[1, col])
        plot_panel(ax1, LAYER_NAMES,
                   t_data['mean_dr2'],
                   color_cfg['line'], color_cfg['fill'], color_cfg['marker'],
                   'Mean ΔR²', mean_ylim)

        # Row 2: Max ΔR²
        ax2 = fig.add_subplot(gs[2, col])
        plot_panel(ax2, LAYER_NAMES,
                   t_data['max_dr2'],
                   color_cfg['line'], color_cfg['fill'], color_cfg['marker'],
                   'Max ΔR²', max_ylim)

    # ── Column headers ────────────────────────────────────────────────────────
    # gs left=0.09, right=0.97 → total width=0.88, each col = 0.88/3 ≈ 0.293
    col_centers = [0.09 + 0.293 * (i + 0.5) for i in range(3)]
    for col_idx, col_label in enumerate(['Contour Map', 'Medial Axis Map', 'Area Map']):
        fig.text(col_centers[col_idx], 0.905, col_label,
                 ha='center', va='bottom', fontsize=15,
                 fontweight='bold', color='#1a1a2e')

    # ── Row labels ────────────────────────────────────────────────────────────
    # gs top=0.88, bottom=0.08 → total height=0.80, each row = 0.80/3 ≈ 0.267
    row_centers = [0.88 - 0.267 * (i + 0.5) for i in range(3)]
    for row_idx, row_label in enumerate(['R² Full & R² Reduced', 'Max ΔR²', 'Mean ΔR²']):
        fig.text(0.01, row_centers[row_idx], row_label,
                 ha='left', va='center', fontsize=15,
                 fontweight='bold', color='#1a1a2e', rotation=90)

    # ── Shared x-axis label ───────────────────────────────────────────────────
    fig.text(0.53, 0.01, 'VGG16 Convolutional Layer',
             ha='center', va='bottom', fontsize=11, color='#444444')

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f'{sym_label} Symmetry — ΔR² Across VGG16 Layers for 73,000 Images',
        fontsize=18, fontweight='bold', y=0.965, color='#1a1a2e'
    )

    # ── Legend (right margin) — R² Full + R² Reduced ─────────────────────────
    legend_handles = [
        Line2D([0], [0],
               color=color_cfg['line'], linewidth=2.2,
               marker='o', markersize=7,
               markerfacecolor=color_cfg['marker'],
               markeredgecolor=color_cfg['line'], markeredgewidth=1.8,
               label='R² Full'),
        Line2D([0], [0],
               color='#666666', linewidth=1.5,
               marker='s', markersize=5,
               markerfacecolor='white',
               markeredgecolor='#666666', markeredgewidth=1.2,
               linestyle='--', alpha=0.7,
               label='R² Reduced'),
    ]
    fig.legend(handles=legend_handles,
               loc='upper right',
               bbox_to_anchor=(0.97, 0.97),   # top-right corner of the figure
               fontsize=11,
               frameon=True,
               framealpha=0.9,
               edgecolor='#cccccc',
               title='Legend',
               title_fontsize=11)

    out_path = os.path.join(plots_dir, f'{sym_label}(mean)_73000img.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FFFFFF')
    plt.show()
    plt.close()
    print(f"Saved: {out_path}")

# ── Combined comparison figure (2×3) ─────────────────────────────────────────
fig2 = plt.figure(figsize=(18, 9))
fig2.patch.set_facecolor('#FAFAFA')
 
gs2 = gridspec.GridSpec(2, 3, figure=fig2,
                        hspace=0.45, wspace=0.30,
                        top=0.88, bottom=0.10,
                        left=0.07, right=0.97)
 
fig2.suptitle('ΔR² Comparison: Mirror vs Parallel vs Taper',
              fontsize=18, fontweight='bold', y=0.97, color='#1a1a2e')
 
for col, mt in enumerate(MAP_TYPES):
    ax_mean = fig2.add_subplot(gs2[0, col])
    ax_max  = fig2.add_subplot(gs2[1, col], sharex=ax_mean)
 
    for sym_label, sym_data in all_sym_data.items():
        c = SYM_COLORS[sym_label]
        x = np.arange(len(LAYER_NAMES))
 
        # Row 0: Mean ΔR²
        vals_mean = sym_data[mt]['mean_dr2']
        ax_mean.plot(x, vals_mean, marker='o', linewidth=2.2, markersize=6,
                     color=c['line'], markerfacecolor=c['marker'],
                     markeredgecolor=c['line'], markeredgewidth=1.5,
                     label=sym_label, zorder=3)
        ax_mean.fill_between(x, vals_mean, alpha=0.08, color=c['fill'])
 
        # Row 1: Max ΔR²
        vals_max = sym_data[mt]['max_dr2']
        ax_max.plot(x, vals_max, marker='o', linewidth=2.2, markersize=6,
                    color=c['line'], markerfacecolor=c['marker'],
                    markeredgecolor=c['line'], markeredgewidth=1.5,
                    label=sym_label, zorder=3)
        ax_max.fill_between(x, vals_max, alpha=0.08, color=c['fill'])
 
    for ax, ylim, ylabel in [
        (ax_mean, mean_ylim, 'Mean ΔR²'),
        (ax_max,  max_ylim,  'Max ΔR²'),
    ]:
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
    # x-axis ticks only on bottom row, hide on top row
    ax_mean.tick_params(labelbottom=False)
    ax_max.set_xticks(np.arange(len(LAYER_NAMES)))
    ax_max.set_xticklabels(LAYER_NAMES, rotation=45, ha='right', fontsize=9)
 
# ── Column headers ────────────────────────────────────────────────────────────
# gs2 left=0.07, right=0.97 → width=0.90, each col=0.30
col_centers2 = [0.07 + 0.30 * (i + 0.5) for i in range(3)]
for col_idx, col_label in enumerate(['Contour Map', 'Medial Axis Map', 'Area Map']):
    fig2.text(col_centers2[col_idx], 0.905, col_label,
              ha='center', va='bottom', fontsize=15,
              fontweight='bold', color='#1a1a2e')
 
# ── Row labels ────────────────────────────────────────────────────────────────
# gs2 top=0.88, bottom=0.10 → height=0.78, each row=0.39
row_centers2 = [0.88 - 0.39 * (i + 0.5) for i in range(2)]
for row_idx, row_label in enumerate(['Mean ΔR²', 'Max ΔR²']):
    fig2.text(0.01, row_centers2[row_idx], row_label,
              ha='left', va='center', fontsize=15,
              fontweight='bold', color='#1a1a2e', rotation=90)
 
# ── Shared x-axis label ───────────────────────────────────────────────────────
fig2.text(0.52, 0.02, 'VGG16 Convolutional Layer',
          ha='center', va='bottom', fontsize=11, color='#444444')
 
# ── Legend — top-right corner ─────────────────────────────────────────────────
legend_handles2 = [
    Line2D([0], [0],
           color=SYM_COLORS[s]['line'], linewidth=2.2,
           marker='o', markersize=6,
           markerfacecolor=SYM_COLORS[s]['marker'],
           markeredgecolor=SYM_COLORS[s]['line'], markeredgewidth=1.5,
           label=s)
    for s in all_sym_data
]
fig2.legend(handles=legend_handles2,
            loc='upper right',
            bbox_to_anchor=(0.97, 0.97),
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            edgecolor='#cccccc',
            title='Symmetry Type',
            title_fontsize=10)
 
out2 = os.path.join(plots_dir, 'combined_mean_73000img.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#FFFFFF')
plt.close()
print(f"Saved: {out2}")
print("All plots generated!")
