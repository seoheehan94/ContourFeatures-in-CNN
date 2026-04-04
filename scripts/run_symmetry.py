#!/usr/bin/env python
# scripts/run_symmetry.py
"""
Entry-point script for the symmetry nested-model pipeline.

Reads all paths and parameters from config/symmetry.yaml (or a path
passed via --config).  Sensitive lab paths are kept out of the repo:
set the two environment variables below before running, or edit
config/symmetry.yaml to point at an override file.

Required environment variables
--------------------------------
  PROJECT_ROOT   – root directory of the project on the lab machine
                   e.g. /bwlab/Projects/NeuralNetOrientation_Napasorn
  STIMULI_ROOT   – root directory of the NSD stimuli
                   e.g. /bwdata/NSDData/stimuli

Usage
-----
  # preprocess maps for a single batch
  python scripts/run_symmetry.py --stage preprocess --sym_type mir --batch 1

  # run nested model for a single batch
  python scripts/run_symmetry.py --stage model --sym_type mir --batch 1

  # run statistical testing (all sym types)
  python scripts/run_symmetry.py --stage stats

  # generate plots
  python scripts/run_symmetry.py --stage plot
"""

import argparse
import os
import sys

# ── make src/ importable when running from repo root ─────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.symmetry.preprocessing.preprocess_maps import run_batch_preprocessing
from src.symmetry.modeling.nested_model import run_full_pipeline
from src.symmetry.modeling.statistics_testing import (
    parse_full_model_txt, apply_bh_correction
)


def get_paths():
    """Resolve all paths from environment variables."""
    project_root  = os.environ.get("PROJECT_ROOT", "")
    stimuli_root  = os.environ.get("STIMULI_ROOT", "")

    if not project_root:
        print("[WARNING] PROJECT_ROOT env var is not set. "
              "Paths will be relative — set it before running on the lab machine.")
    if not stimuli_root:
        print("[WARNING] STIMULI_ROOT env var is not set.")

    return {
        "project_root"      : project_root,
        "stimuli_root"      : stimuli_root,
        "contour_base"      : os.path.join(project_root, "orientation", "OribinMapsResults"),
        "sym_maps_base"     : os.path.join(project_root, "SymbinMapsResults"),
        "preprocessed_pt"   : os.path.join(project_root, "symmetry", "maps_pt1"),
        "results_dir"       : os.environ.get(
                                  "RESULTS_DIR",
                                  os.path.join(project_root, "symmetry", "NestedModelResults_FINAL")),
        "plots_dir"         : os.environ.get(
                                  "PLOTS_DIR",
                                  os.path.join(project_root, "symmetry",
                                               "NestedModelResults_FINAL", "plots")),
    }


SYM_MAP_FOLDER = {
    "mir": "mirMap",
    "par": "parMap",
    "tap": "tapMap",
}

NUM_IMAGES = 14600
NUM_BATCHES = 5


def stage_preprocess(sym_type: str, batch_no: int, paths: dict):
    contour_dir = os.path.join(
        paths["contour_base"], f"ContourFilter{batch_no:02d}")
    sym_dirs = {
        sym_type: os.path.join(paths["sym_maps_base"], SYM_MAP_FOLDER[sym_type])
    }
    run_batch_preprocessing(
        batch_no, contour_dir, sym_dirs,
        sym_type=sym_type,
        out_base_dir=paths["preprocessed_pt"],
        num_images=NUM_IMAGES,
    )


def stage_model(sym_type: str, batch_no: int, paths: dict):
    image_dir         = os.path.join(paths["stimuli_root"], f"images0{batch_no}")
    contour_dir_pt    = os.path.join(paths["preprocessed_pt"], "ContourFilter")
    symmetry_dir_pt   = os.path.join(paths["preprocessed_pt"], SYM_MAP_FOLDER[sym_type])

    run_full_pipeline(
        batch_no,
        image_dir,
        contour_dir_pt,
        symmetry_dir_pt,
        num_images=NUM_IMAGES,
        txt_dir=paths["results_dir"],
        sym_type=sym_type,
        save_interval=100,
    )


def stage_stats(paths: dict):
    MAP_TYPES = ['contour', 'medial', 'area']
    for sym_type in ['mir', 'par', 'tap']:
        filepath = os.path.join(
            paths["results_dir"],
            f"{sym_type}_accumulation_73000img_all_full.txt"
        )
        print(f"\n{'='*60}")
        print(f"Processing: {sym_type.upper()} symmetry")
        print(f"{'='*60}")
        full_results_dict = parse_full_model_txt(filepath, sym_type)
        apply_bh_correction(full_results_dict, MAP_TYPES, alpha=0.05)


def stage_plot():
    # Set env vars so the plot scripts pick up the right directories,
    # then import and run them as modules.
    import importlib
    import src.symmetry.visualization.contour_model_plot   # noqa: F401
    import src.symmetry.visualization.delta_r2_plot        # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Run symmetry nested-model pipeline stages.")
    parser.add_argument("--stage", required=True,
                        choices=["preprocess", "model", "stats", "plot"],
                        help="Which pipeline stage to run.")
    parser.add_argument("--sym_type", default=None,
                        choices=["mir", "par", "tap"],
                        help="Symmetry type (required for preprocess/model).")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch number 1-5 (required for preprocess/model).")
    parser.add_argument("--all_batches", action="store_true",
                        help="Loop over all 5 batches (preprocess/model only).")
    args = parser.parse_args()

    paths = get_paths()

    if args.stage == "preprocess":
        if not args.sym_type:
            parser.error("--sym_type is required for --stage preprocess")
        batches = range(1, NUM_BATCHES + 1) if args.all_batches else [args.batch]
        for b in batches:
            print(f"\n=== Preprocessing {args.sym_type} | batch {b} ===")
            stage_preprocess(args.sym_type, b, paths)

    elif args.stage == "model":
        if not args.sym_type:
            parser.error("--sym_type is required for --stage model")
        batches = range(1, NUM_BATCHES + 1) if args.all_batches else [args.batch]
        for b in batches:
            print(f"\n=== Nested model {args.sym_type} | batch {b} ===")
            stage_model(args.sym_type, b, paths)

    elif args.stage == "stats":
        stage_stats(paths)

    elif args.stage == "plot":
        stage_plot()


if __name__ == "__main__":
    main()
