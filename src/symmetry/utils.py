# src/symmetry/utils.py

import ast
import pickle
import numpy as np
import torch


def accumulate_5batches(sym_type: str):
    # ── Define your 5 file paths ──
    import os
    path = os.environ.get(
        "RESULTS_DIR",
        "results/symmetry/nested_model"
    )
    file_paths = [f"{path}/{sym_type}_accumulation_14600img_0{i}.pkl" for i in range(1,6)]

    # ── Helper: accumulate one SuffStatsGPU object into another in-place ──
    def accumulate_ss(target, source):
        """Add all numeric attributes of source into target."""
        for attr in vars(source):
            val_src = getattr(source, attr)
            val_tgt = getattr(target, attr)
            if isinstance(val_src, torch.Tensor):
                val_tgt += val_src
            elif isinstance(val_src, (int, float)):
                setattr(target, attr, val_tgt + val_src)
            # skip non-numeric attributes (strings, etc.)

    # ── Load first file as the base accumulator ──
    with open(file_paths[0], "rb") as f:
        accumulated = pickle.load(f)

    # ── Loop over remaining files and accumulate ──
    for path in file_paths[1:]:
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Accumulate 'reduced' list (13 layers)
        for i, ss in enumerate(data['reduced']):
            accumulate_ss(accumulated['reduced'][i], ss)

        # Accumulate 'full' dict: contour / medial / area (each 13 layers)
        for key in accumulated['full']:          # 'contour', 'medial', 'area'
            for i, ss in enumerate(data['full'][key]):
                accumulate_ss(accumulated['full'][key][i], ss)

    print("Done! All 5 files accumulated.")

    # ── Verify: check total n for reduced layer 0 ──
    print("Total n (reduced, layer 0):", accumulated['reduced'][0].n)

    # ── Save accumulated result ──
    import os
    output_dir = os.environ.get(
        "RESULTS_DIR",
        "results/symmetry/nested_model"
    )
    output_path = f"{output_dir}/{sym_type}_accumulation_73000img_all.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(accumulated, f)

    print(f"Saved to: {output_path}")


def read_values(value: str, sym_type: str, mode: str) -> dict:
    """Read specified values that we want to pick up on the accumulation_73000img_all_full.txt"""
    import os
    path = os.environ.get(
        "RESULTS_DIR",
        "results/symmetry/nested_model"
    )
    
    def parse_section(lines, start, end):
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
            label = label.strip()
            values = ast.literal_eval(values_str.strip())
            if layer_idx not in data:
                data[layer_idx] = {}
            data[layer_idx][label] = np.array(values)
        return data

    with open(f'{path}/{sym_type}_accumulation_73000img_all_full.txt', 'r') as f:
        lines = f.readlines()

    sections = {
        'contour': parse_section(lines, 1, 146),
        'medial':  parse_section(lines, 146, 291),
        'area':    parse_section(lines, 291, len(lines)),
    }

    result = {}
    for sec_name, sec_data in sections.items():
        vals = []
        for li in range(13):
            p_arr = sec_data[li][value]
            if mode == 'max':
                vals.append(float(np.max(p_arr)))
            elif mode == 'mean':
                vals.append(float(np.mean(p_arr)))
        result[sec_name] = vals

    return result


def get_r2_contour(filepath: str, mode: str):
    """
    Reads a reduced model result file and returns a list of 13 max R² values,
    one per layer.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    max_r2 = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',', 2)
        if len(parts) < 3:
            continue
        layer_str, label, values_str = parts
        if label.strip() == 'R2':
            values = np.array(ast.literal_eval(values_str.strip()))
            if mode == 'max':
                max_r2.append(float(np.max(values)))
            elif mode == 'mean':
                max_r2.append(float(np.mean(values)))
    
    return max_r2  # list of 13 floats
