# src/symmetry/modeling/nested_model.py

"""
Hierarchical (Nested) Regression Analysis for Symmetry Maps vs VGG16 Features
==============================================================================
Pipeline:
  Pass 1  – Reduced model  : Y ~ β0 + β1·X_contour
  Pass 2  – Full model     : Y ~ β0 + β1·X_contour + β2·X_symmetry
  ΔR² = R²_full − R²_reduced

Outputs (all in txt_dir):
  {sym_type}_reduced_model_{n}img_0{batch}.txt
      └─ Per layer: Beta0, Beta1, t, p, R2 (per channel) + TOTAL mean R2
  {sym_type}_full_model_all_maps_{n}img_0{batch}.txt
      └─ Per map type, per layer: Beta0, Beta1, Beta2, t_contour, p_contour,
         t_symmetry, p_symmetry, R2_full, Delta_R2 (per channel)
  {sym_type}_full_delta_R2_{mt}_{n}img_0{batch}.txt  (one per map type)
      └─ Per layer: Delta_R2 (per channel) + TOTAL mean/min/max Delta_R2
         + TOTAL mean/min/max R2_full
  {sym_type}_checkpoint_{n}img_0{batch}.pt
      └─ Preserved after completion for suff-stats retrieval
  {sym_type}_accumulation_{n}img_0{batch}.txt  +  .pkl
      └─ Human-readable + pickle dump of all SuffStatsGPU objects

Optimized Pipeline:
  • Single Pass Architecture: Processes Reduced and Full models simultaneously.
  • GPU Acceleration: Feature maps and map statistics are accumulated on the GPU
    using PyTorch tensors to eliminate CPU/NumPy bottlenecks.
  • Solves OLS and calculates p-values on CPU at the end of the accumulation.
  • Checkpoint is KEPT after completion for future retrieval of sufficient stats.
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from scipy import stats

from src.symmetry.modeling.stuff_stats import SuffStatsGPU

# ─── VGG helpers ─────────────────────────────────────────────────────────────

def load_model(device):
    model = models.vgg16(weights='DEFAULT')
    model.eval()
    model.to(device)
    return model


def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)


def register_hooks(model):
    outputs = []
    handles = []
    def hook_fn(module, inp, out):
        outputs.append(out)
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(hook_fn))
    return outputs, handles


def get_feature_maps_gpu(hook_outputs, original_size=(425, 425)):
    """Return list of (C,H,W) tensors on the GPU, one per conv layer."""
    result = []
    for out in hook_outputs:
        resized = F.interpolate(out, size=original_size,
                                mode='bilinear', align_corners=False)
        result.append(resized.squeeze(0))  # keep on GPU, squeeze batch dim
    return result


# ─── I/O helpers ─────────────────────────────────────────────────────────────

LAYER_NAMES = ['conv1-1', 'conv1-2', 'conv2-1', 'conv2-2',
               'conv3-1', 'conv3-2', 'conv3-3',
               'conv4-1', 'conv4-2', 'conv4-3',
               'conv5-1', 'conv5-2', 'conv5-3']
VGG_CHANNELS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]


def write_reduced_results(path, results_per_layer):
    """
    Save reduced model (contour only) results.

    Per layer writes: Beta0, Beta1, t, p, R2 (all per-channel lists).
    Footer writes:    TOTAL mean R2 across all 13 layers.
    """
    mean_r2_per_layer = []
    with open(path, 'w') as f:
        for l, (b0, b1, t, p, r2) in enumerate(results_per_layer):
            f.write(f"Layer{l}, Beta0, {np.array(b0).tolist()}\n")
            f.write(f"Layer{l}, Beta1, {np.array(b1).tolist()}\n")
            f.write(f"Layer{l}, t, {np.array(t).tolist()}\n")
            f.write(f"Layer{l}, p, {np.array(p).tolist()}\n")
            f.write(f"Layer{l}, R2, {np.array(r2).tolist()}\n")
            mean_r2_per_layer.append(float(np.mean(r2)))

        f.write(f"TOTAL, Mean_R2, {mean_r2_per_layer}\n")
    print(f"  Saved: {path}")


def write_full_model_all_maps(path, full_results_dict, map_types):
    """
    Save full model results for all map types into one file.

    For each map type and layer writes:
        Beta0, Beta1, Beta2, t_contour, p_contour,
        t_symmetry, p_symmetry, R2_full, Delta_R2  (all per-channel lists).
    """
    with open(path, 'w') as f:
        for mt in map_types:
            f.write(f"\n# === {mt.upper()} SYMMETRY MAP ===\n")
            for l, res in enumerate(full_results_dict[mt]):
                b0, b1, b2, t1, p1, t2, p2, r2f, dr2, f_stat, p_f = res
                f.write(f"Layer{l}, Beta0, {np.array(b0).tolist()}\n")
                f.write(f"Layer{l}, Beta1, {np.array(b1).tolist()}\n")
                f.write(f"Layer{l}, Beta2, {np.array(b2).tolist()}\n")
                f.write(f"Layer{l}, t_contour, {np.array(t1).tolist()}\n")
                f.write(f"Layer{l}, p_contour, {np.array(p1).tolist()}\n")
                f.write(f"Layer{l}, t_symmetry, {np.array(t2).tolist()}\n")
                f.write(f"Layer{l}, p_symmetry, {np.array(p2).tolist()}\n")
                f.write(f"Layer{l}, R2_full, {np.array(r2f).tolist()}\n")
                f.write(f"Layer{l}, Delta_R2, {np.array(dr2).tolist()}\n")
                f.write(f"Layer{l}, F_stat, {np.array(f_stat).tolist()}\n")
                f.write(f"Layer{l}, p_val_F, {np.array(p_f).tolist()}\n")
    print(f"  Saved: {path}")


def write_delta_r2_summary(path, full_results_per_layer):
    """
    Save per-map-type Delta R2 summary file.

    Per layer writes: Delta_R2 (per channel), R2_full (per channel).
    Footer writes:
        TOTAL, Delta_R2   — mean Delta_R2 across layers
        TOTAL, Min_delta  — min  Delta_R2 across layers
        TOTAL, Max_delta  — max  Delta_R2 across layers
        TOTAL, Mean_R2_full — mean R2_full across layers
        TOTAL, Min_R2_full  — min  R2_full across layers
        TOTAL, Max_R2_full  — max  R2_full across layers
    """
    mean_dr2_per_layer   = []
    min_dr2_per_layer    = []
    max_dr2_per_layer    = []
    mean_r2f_per_layer   = []
    min_r2f_per_layer    = []
    max_r2f_per_layer    = []

    with open(path, 'w') as f:
        for l, res in enumerate(full_results_per_layer):
            b0, b1, b2, t1, p1, t2, p2, r2f, dr2, f_stat, p_f = res
            dr2_np = np.array(dr2)
            r2f_np = np.array(r2f)

            f.write(f"Layer{l}, Delta_R2, {dr2_np.tolist()}\n")
            f.write(f"Layer{l}, R2_full,  {r2f_np.tolist()}\n")

            mean_dr2_per_layer.append(float(np.mean(dr2_np)))
            min_dr2_per_layer.append(float(np.min(dr2_np)))
            max_dr2_per_layer.append(float(np.max(dr2_np)))
            mean_r2f_per_layer.append(float(np.mean(r2f_np)))
            min_r2f_per_layer.append(float(np.min(r2f_np)))
            max_r2f_per_layer.append(float(np.max(r2f_np)))

        # Summary rows — primary ones match the original format exactly
        f.write(f"TOTAL, Delta_R2,    {mean_dr2_per_layer}\n")
        f.write(f"TOTAL, Min_delta,   {min_dr2_per_layer}\n")
        f.write(f"TOTAL, Max_delta,   {max_dr2_per_layer}\n")
        f.write(f"TOTAL, Mean_R2_full,{mean_r2f_per_layer}\n")
        f.write(f"TOTAL, Min_R2_full, {min_r2f_per_layer}\n")
        f.write(f"TOTAL, Max_R2_full, {max_r2f_per_layer}\n")

    print(f"  Saved: {path}")


def write_accumulation(path, reduced_stats, full_stats):
    """
    Save human-readable summary (.txt) and pickle dump (.pkl) of all
    SuffStatsGPU objects so sufficient statistics can be reloaded later.
    """
    pkl_path = path.replace('.txt', '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'reduced': reduced_stats, 'full': full_stats}, f)

    with open(path, 'w') as f:
        f.write("# Accumulated sufficient statistics (SuffStatsGPU objects)\n")
        f.write(f"# Pickle saved to: {pkl_path}\n\n")
        for l in range(13):
            ss = reduced_stats[l]
            f.write(f"[REDUCED] Layer{l}\n")
            f.write(f"  n={ss.n}\n")
            f.write(f"  Sx1={ss.Sx1.item():.6f}\n")
            f.write(f"  Sx1x1={ss.Sx1x1.item():.6f}\n")
            f.write(f"  mean_Sy={float(torch.mean(ss.Sy).item()):.6f}\n")
            f.write(f"  mean_Sx1y={float(torch.mean(ss.Sx1y).item()):.6f}\n")
            f.write(f"  mean_Syy={float(torch.mean(ss.Syy).item()):.6f}\n\n")
        for mt, stats_list in full_stats.items():
            for l in range(13):
                ss = stats_list[l]
                f.write(f"[FULL-{mt}] Layer{l}\n")
                f.write(f"  n={ss.n}\n")
                f.write(f"  Sx1={ss.Sx1.item():.6f}, Sx2={ss.Sx2.item():.6f}\n")
                f.write(f"  Sx1x1={ss.Sx1x1.item():.6f}, "
                        f"Sx2x2={ss.Sx2x2.item():.6f}, "
                        f"Sx1x2={ss.Sx1x2.item():.6f}\n")
                f.write(f"  mean_Sy={float(torch.mean(ss.Sy).item()):.6f}\n")
                f.write(f"  mean_Sx1y={float(torch.mean(ss.Sx1y).item()):.6f}\n")
                f.write(f"  mean_Sx2y={float(torch.mean(ss.Sx2y).item()):.6f}\n")
                f.write(f"  mean_Syy={float(torch.mean(ss.Syy).item()):.6f}\n\n")

    print(f"  Saved accumulation: {path}")
    print(f"  Saved accumulation pickle: {pkl_path}")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(batch_no, image_dir, contour_dir_pt, symmetry_dir_pt,
                      num_images, txt_dir, sym_type, save_interval=500):
    """
    Parameters
    ----------
    batch_no         : batch number (1-indexed), used to compute start image index
    image_dir        : path to image files (img{N}.png)
    contour_dir_pt   : path to pre-processed contour .pt tensors
    symmetry_dir_pt  : path to pre-processed symmetry .pt tensors
    num_images       : how many images to process in this batch
    txt_dir          : where to save all result files
    sym_type         : tag for filenames, e.g. 'mir', 'par', 'tap'
    save_interval    : checkpoint every N images
    """
    start_no = (batch_no - 1) * 14600
    os.makedirs(txt_dir, exist_ok=True)

    pixels     = 425 * 425
    image_size = (425, 425)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model    = load_model(device)
    MAP_TYPES = ['contour', 'medial', 'area']

    # ─── Checkpoint Loading ───────────────────────────────────────────────────
    ckpt_path = os.path.join(
        txt_dir, f"{sym_type}_checkpoint_{num_images}img_0{batch_no}.pt")
    start_img_idx = 1

    if os.path.exists(ckpt_path):
        print(f"\n[INFO] Found existing checkpoint: {ckpt_path}")
        cp_state      = torch.load(ckpt_path, map_location=device, weights_only=False)
        reduced_stats = cp_state['reduced_stats']
        full_stats    = cp_state['full_stats']
        start_img_idx = cp_state['next_img_idx']
        print(f"Resuming at image {start_img_idx} / {num_images}...")
    else:
        reduced_stats = [SuffStatsGPU(c, device) for c in VGG_CHANNELS]
        full_stats    = {mt: [SuffStatsGPU(c, device) for c in VGG_CHANNELS]
                         for mt in MAP_TYPES}

    # ─── Single-Pass Accumulation ─────────────────────────────────────────────
    print("\n=== Accumulating Reduced and Full Models (single pass) ===")
    hook_out, handles = register_hooks(model)
    start_time = time.time()
    images_processed_this_run = 0

    for img_idx in range(start_img_idx, num_images + 1):
        img_no = start_no + img_idx

        # 1. Load pre-processed contour tensor → GPU
        c_pt_path = os.path.join(contour_dir_pt, f'contour_{img_no}.pt')
        if not os.path.exists(c_pt_path):
            continue
        x_contour = (torch.load(c_pt_path, map_location=device, weights_only=True)
                     .to(torch.float64).flatten())

        # 2. Load pre-processed symmetry tensors → GPU
        sym_maps = {}
        for mt in MAP_TYPES:
            file_tag  = 'medialAxis' if mt == 'medial' else mt
            s_pt_path = os.path.join(symmetry_dir_pt, f'{file_tag}_{img_no}.pt')
            if os.path.exists(s_pt_path):
                sym_maps[mt] = (torch.load(s_pt_path, map_location=device,
                                           weights_only=True)
                                .to(torch.float64).flatten())

        # 3. Forward pass through VGG
        img_path = os.path.join(image_dir, f'img{img_no}.png')
        if not os.path.exists(img_path):
            continue

        img   = Image.open(img_path).convert('RGB')
        batch = torch.unsqueeze(preprocess(img), 0).to(device)

        hook_out.clear()
        with torch.no_grad():
            _ = model(batch)
        feature_maps = get_feature_maps_gpu(hook_out, image_size)

        # 4. Update sufficient statistics for all 13 layers
        for l in range(13):
            # y: (pixels, C) float64 on GPU
            y = feature_maps[l].permute(1, 2, 0).reshape(pixels, -1).to(torch.float64)
            # Reduced model: contour only
            reduced_stats[l].update1(x_contour, y)
            # Full model: contour + each symmetry map type
            for mt, x2 in sym_maps.items():
                full_stats[mt][l].update2(x_contour, x2, y)

        # 5. Free GPU memory
        del batch, feature_maps, x_contour, sym_maps, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images_processed_this_run += 1

        # ─── Progress & ETA ───
        if img_idx % 100 == 0:
            elapsed = time.time() - start_time
            avg     = elapsed / images_processed_this_run
            eta     = time.strftime("%H:%M:%S", time.gmtime(avg * (num_images - img_idx)))
            print(f"  [Processing] {img_idx}/{num_images} | ETA: {eta} | "
                  f"({avg:.3f} sec/img)")

        # ─── Periodic Checkpoint ─────────────────────────────────────────────
        if img_idx % save_interval == 0:
            print(f"  [Checkpoint] Saving progress at image {img_idx}...")
            torch.save(
                {'reduced_stats': reduced_stats,
                 'full_stats':    full_stats,
                 'next_img_idx':  img_idx + 1},
                ckpt_path)

    for h in handles:
        h.remove()

    # ─── Save final checkpoint (kept permanently) ─────────────────────────────
    print(f"\n[Checkpoint] Saving final checkpoint (will be kept): {ckpt_path}")
    torch.save(
        {'reduced_stats': reduced_stats,
         'full_stats':    full_stats,
         'next_img_idx':  num_images + 1},   # signals "fully done"
        ckpt_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPUTE RESULTS  (all heavy math on CPU via numpy/scipy)
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Reduced model ─────────────────────────────────────────────────────────
    print("\n[Finalizing] Computing Reduced Model (Pass 1)...")
    reduced_results  = []
    R2_reduced_list     = []
    SS_res_reduced_list = []
    for l in range(13):
        b0, b1, t, p, r2, ss_res = reduced_stats[l].compute_reduced()
        reduced_results.append((b0, b1, t, p, r2))
        R2_reduced_list.append(r2)
        SS_res_reduced_list.append(ss_res)

    reduced_path = os.path.join(
        txt_dir, f"{sym_type}_reduced_model_{num_images}img_0{batch_no}.txt")
    write_reduced_results(reduced_path, reduced_results)

    # ── Full model ────────────────────────────────────────────────────────────
    print("[Finalizing] Computing Full Models and Delta R2 (Pass 2)...")

    full_results_dict = {}          # mt -> list[13] of result tuples
    for mt in MAP_TYPES:
        layer_results = []
        for l in range(13):
            res = full_stats[mt][l].compute_full(R2_reduced_list[l], SS_res_reduced_list[l])
            layer_results.append(res)
        full_results_dict[mt] = layer_results

        # Per-map-type Delta R2 summary file
        dr2_path = os.path.join(
            txt_dir,
            f"{sym_type}_full_delta_R2_{mt}_{num_images}img_0{batch_no}.txt")
        write_delta_r2_summary(dr2_path, layer_results)

    # Combined full model file (all map types, all stats)
    full_path = os.path.join(
        txt_dir,
        f"{sym_type}_full_model_all_maps_{num_images}img_0{batch_no}.txt")
    write_full_model_all_maps(full_path, full_results_dict, MAP_TYPES)

    # ── Accumulation dump ─────────────────────────────────────────────────────
    acc_path = os.path.join(
        txt_dir,
        f"{sym_type}_accumulation_{num_images}img_0{batch_no}.txt")
    write_accumulation(acc_path, reduced_stats, full_stats)

    print(f"\n✓ Process Complete. All results saved to: {txt_dir}")
    print(f"  Files produced:")
    print(f"    {sym_type}_reduced_model_{num_images}img_0{batch_no}.txt")
    print(f"    {sym_type}_full_model_all_maps_{num_images}img_0{batch_no}.txt")
    for mt in MAP_TYPES:
        print(f"    {sym_type}_full_delta_R2_{mt}_{num_images}img_0{batch_no}.txt")
    print(f"    {sym_type}_checkpoint_{num_images}img_0{batch_no}.pt  (kept)")
    print(f"    {sym_type}_accumulation_{num_images}img_0{batch_no}.txt + .pkl")
