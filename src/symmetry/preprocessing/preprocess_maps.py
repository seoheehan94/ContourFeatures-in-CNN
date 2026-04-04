# src/symmetry/preprocessing/preprocess_maps.py

"""
Batch Pre-processor: Converts .mat symmetry/contour maps into ready-to-use PyTorch tensors.
Applies Gaussian blur and normalization ONCE and saves as .pt files.
"""

import os
import glob
import time
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

def process_contour(mat_path, save_path):
    """Loads contour .mat, applies blur, normalizes, and saves as .pt"""
    if os.path.exists(save_path):
        return

    mat = loadmat(mat_path)
    oribinmap = mat.get('oribinmap')
    if oribinmap is None:
        return

    sigma, combined = 1.0, None
    for b in range(oribinmap.shape[1]):
        blurred = gaussian_filter(oribinmap[0, b] * 100, sigma=sigma, mode='wrap') / 100
        mx = np.max(blurred)
        norm = blurred / mx if mx > 0 else np.zeros_like(blurred)
        combined = norm.copy() if combined is None else combined + norm

    # Save as float32 PyTorch tensor
    tensor_data = torch.tensor(combined, dtype=torch.float32)
    torch.save(tensor_data, save_path)

def process_symmetry(mat_path, save_path, field, blur=True):
    """Loads symmetry .mat, extracts field, applies blur/norm, and saves as .pt"""
    if os.path.exists(save_path):
        return

    try:
        mat = loadmat(mat_path)
        arr = mat['model'][0][0][field].astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        
        if blur:
            arr = gaussian_filter(arr * 100, sigma=1.0, mode='wrap') / 100
            
        mx = np.max(arr)
        final_arr = (arr / mx).astype(np.float32) if mx > 0 else arr
        
        tensor_data = torch.tensor(final_arr, dtype=torch.float32)
        torch.save(tensor_data, save_path)
    except Exception as e:
        print(f"  Warning processing {mat_path}: {e}")

def run_batch_preprocessing(batch_no, contour_dir, sym_dirs, sym_type, out_base_dir, num_images=14600):
    start_no = (batch_no - 1) * 14600
    
    out_contour = os.path.join(out_base_dir, "ContourFilter")
    os.makedirs(out_contour, exist_ok=True)
    
    out_syms = {}
    for sym_key in sym_dirs.keys():
        out_syms[sym_key] = os.path.join(out_base_dir, f"{sym_key}Map")
        os.makedirs(out_syms[sym_key], exist_ok=True)

    print(f"Starting Pre-processing for Batch {batch_no}...")
    print(f"Input Contour Source: {contour_dir}")
    print(f"Output Contour Target: {out_contour}")
    
    start_time = time.time()
    skipped_count = 0

    for img_idx in range(1, num_images + 1):
        img_no = start_no + img_idx
        
        if img_idx % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {img_idx}/{num_images} | Skipped: {skipped_count}")

        # 1. Process Contour
        c_mat = os.path.join(contour_dir, f'CoribinMap_img{img_no}.mat')
        c_pt = os.path.join(out_contour, f'contour_{img_no}.pt')
        
        if os.path.exists(c_pt):
            skipped_count += 1
        elif os.path.exists(c_mat):
            process_contour(c_mat, c_pt)

        # 2. Process Symmetry Maps
        for sym_key, sym_dir in sym_dirs.items():
            s_mat = os.path.join(sym_dir, f'{sym_type}Img{img_no}.mat')
            if not os.path.exists(s_mat):
                continue
            
            contour_pt = os.path.join(out_syms[sym_key], f'contour_{img_no}.pt')
            medial_pt  = os.path.join(out_syms[sym_key], f'medialAxis_{img_no}.pt')
            area_pt    = os.path.join(out_syms[sym_key], f'area_{img_no}.pt')
            
            if not (os.path.exists(contour_pt) and os.path.exists(medial_pt) and os.path.exists(area_pt)):
                process_symmetry(s_mat, contour_pt, 'contour', blur=True)
                process_symmetry(s_mat, medial_pt, 'medialAxis', blur=True)
                process_symmetry(s_mat, area_pt, 'area', blur=False)
            else:
                skipped_count += 3

    print(f"Pre-processing complete! Total files skipped: {skipped_count}")
    
if __name__ == "__main__":
    import os
    batch_no = 4
    path = os.environ.get("PROJECT_ROOT", "")

    contour_dir = f"{path}/orientation/OribinMapsResults/ContourFilter{batch_no:02d}"

    sym_dirs = {
        'mir': f"{path}/SymbinMapsResults/mirMap"
    }

    # Define your new target save directory here
    out_base_dir = f"{path}/symmetry/maps_pt"

    run_batch_preprocessing(batch_no, contour_dir, sym_dirs, sym_type="mir", out_base_dir=out_base_dir)

    # run all for mir

# Preprocess for mirror sym analysis   (119 min)  
for batch_no in range(2, 6):
    path = os.environ.get("PROJECT_ROOT", "")
    contour_dir = f"{path}/orientation/OribinMapsResults/ContourFilter{batch_no:02d}"
    sym_dirs = {
        'mir': f"{path}/SymbinMapsResults/mirMap"
    }
    # Define your new target save directory here
    out_base_dir = f"{path}/symmetry/maps_pt1"
    
    run_batch_preprocessing(batch_no, contour_dir, sym_dirs, sym_type="mir", out_base_dir=out_base_dir)

    # mirror nested model -- 31 hr as expected (1643 + 251 min)
    for batch_no in range(1, 6):
        path       = os.environ.get("PROJECT_ROOT", "")
        image_dir  = f"{os.environ.get('STIMULI_ROOT', '')}/images0{batch_no}"

        contour_dir_pt     = f"{path}/symmetry/maps_pt1/ContourFilter"
        mir_symmetry_dir_pt = f"{path}/symmetry/maps_pt1/mirMap"

        num_images = 14600
        txt_dir    = f"{path}/symmetry/NestedModelResults_FINAL"
        sym_type   = "mir"

        # run_full_pipeline(
        #     batch_no, image_dir,
        #     contour_dir_pt, mir_symmetry_dir_pt,
        #     num_images, txt_dir, sym_type,
        #     save_interval=100)
