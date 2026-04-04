# src/symmetry/modeling/statistics_testing.py

import numpy as np
import ast
from statsmodels.stats.multitest import multipletests

def parse_full_model_txt(filepath, sym_type):
    """
    Parses the full model result .txt file based on the exact structure:
    
    Lines   1–145  : contour map  (13 layers × 11 fields each = 143 lines + blank/header)
    Lines 146–290  : medial map
    Lines 291–end  : area map
    
    Each line format: "LayerN, FieldName, [values...]"
    
    Returns: dict — mt -> list[13] of tuples
    (b0, b1, b2, t1, p1, t2, p2, r2f, dr2, f_stat, p_val_F)
    """

    def parse_section(lines, start, end):
        """Parse a slice of lines into {layer_idx: {field: np.array}}"""
        data = {}
        for line in lines[start:end]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',', 2)
            if len(parts) < 3:
                continue
            layer_str, label, values_str = parts
            layer_idx = int(layer_str.strip().replace('Layer', ''))
            label     = label.strip()
            values    = np.array(ast.literal_eval(values_str.strip()))
            if layer_idx not in data:
                data[layer_idx] = {}
            data[layer_idx][label] = values
        return data

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # ── Exact same slicing as your read_values function ──────────────────────
    sections = {
        'contour': parse_section(lines, 1,   146),
        'medial':  parse_section(lines, 146, 291),
        'area':    parse_section(lines, 291, len(lines)),
    }

    # ── Field name mapping from txt labels to tuple positions ────────────────
    field_map = {
        'Beta0'      : 'b0',
        'Beta1'      : 'b1',
        'Beta2'      : 'b2',
        't_contour'  : 't1',
        'p_contour'  : 'p1',
        't_symmetry' : 't2',
        'p_symmetry' : 'p2',
        'R2_full'    : 'r2f',
        'Delta_R2'   : 'dr2',
        'F_stat'     : 'f_stat',
        'p_val_F'    : 'p_val_F',
    }
    field_order = ['b0','b1','b2','t1','p1','t2','p2','r2f','dr2','f_stat','p_val_F']

    # ── Convert each section to list[13] of tuples ───────────────────────────
    full_results_dict = {}
    for mt, sec_data in sections.items():
        layer_list = []
        for l in range(13):
            layer_data = sec_data[l]
            # Remap field names
            remapped = {field_map[k]: v for k, v in layer_data.items() if k in field_map}
            tuple_data = tuple(remapped[f] for f in field_order)
            layer_list.append(tuple_data)
        full_results_dict[mt] = layer_list

    return full_results_dict


def apply_bh_correction(full_results_dict, map_types, alpha=0.05):
    """
    Applies Benjamini-Hochberg FDR correction per layer per map type.
    Compares raw p_val_F against the BH threshold to determine surviving channels.
    """
    bh_results = {}

    for mt in map_types:
        bh_results[mt] = []

        for l in range(13):
            b0, b1, b2, t1, p1, t2, p2, r2f, dr2, f_stat, p_val_F = \
                full_results_dict[mt][l]

            p_val_F = np.array(p_val_F)
            dr2     = np.array(dr2)
            r2f     = np.array(r2f)

            # ── BH correction ─────────────────────────────────────────────
            reject, p_corrected, _, bh_threshold = multipletests(
                p_val_F, alpha=alpha, method='fdr_bh')

            n_total       = len(p_val_F)
            n_surviving   = int(reject.sum())
            survival_rate = n_surviving / n_total * 100

            surviving_delta_r2 = dr2[reject]
            surviving_r2_full  = r2f[reject]

            bh_results[mt].append({
                'layer'              : l,
                'n_total'            : n_total,
                'n_surviving'        : n_surviving,
                'survival_rate_pct'  : survival_rate,
                'bh_threshold'       : bh_threshold,
                'reject'             : reject,
                'p_val_F'            : p_val_F,
                'p_corrected'        : p_corrected,
                'delta_r2_all'       : dr2,
                'delta_r2_surviving' : surviving_delta_r2,
                'mean_dr2_all'       : float(np.mean(dr2)),
                'mean_dr2_surviving' : float(np.mean(surviving_delta_r2))
                                       if n_surviving > 0 else 0.0,
                'r2_full_surviving'  : surviving_r2_full,
            })

            mean_dr2_surv = float(np.mean(surviving_delta_r2)) \
                            if n_surviving > 0 else 0.0
            print(f"  [{mt}] Layer {l:2d} | "
                  f"Surviving: {n_surviving:3d}/{n_total} ({survival_rate:5.1f}%) | "
                  f"BH threshold: {bh_threshold:.2e} | "
                  f"Mean ΔR² (surviving): {mean_dr2_surv:.6f}")

    return bh_results


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    BASE_PATH = os.environ.get(
        "RESULTS_DIR",
        "results/symmetry/nested_model"
    )
    MAP_TYPES = ['contour', 'medial', 'area']

    # One call per symmetry type (mir, par, tap)
    for sym_type in ['mir', 'par', 'tap']:
        filepath = f"{BASE_PATH}/{sym_type}_accumulation_73000img_all_full.txt"

        print(f"\n{'='*60}")
        print(f"Processing: {sym_type.upper()} symmetry")
        print(f"{'='*60}")

        full_results_dict = parse_full_model_txt(filepath, sym_type)
        bh_results        = apply_bh_correction(full_results_dict, MAP_TYPES, alpha=0.05)
