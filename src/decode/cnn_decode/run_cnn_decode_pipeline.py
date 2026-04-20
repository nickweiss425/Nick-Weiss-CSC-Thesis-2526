#!/usr/bin/env python3
"""
Wrapper for the full CNN decoding/evaluation pipeline.

Runs, in order:
  1) infer_windows.py
  2) smooth_inferences.py
  3) hysteresis.py
  4) min_duration_filter_cnn.py
  5) gap_merge_cnn.py
  6) segment_primitives.py
  7) decode_eval.py

Design goals:
- one shared decode directory for all per-participant .npz files
- one command for either a single participant or all folds
- pass-through of the most important decoding hyperparameters
- support for the new min-duration and gap-merge cleanup steps

IMPORTANT:
- This wrapper assumes your downstream CNN scripts have been updated so that:
    * segment_primitives.py reads:
        {decode_root}/{pid}/{out_tag}_window_gap_merged.npz
      and uses:
        y_gap_merged
    * decode_eval.py reads:
        {decode_root}/{pid}/{out_tag}_window_gap_merged.npz
      and uses:
        y_gap_merged
- If those scripts still read the hysteresis output, the new cleanup steps will
  not affect final evaluation metrics.

Example:
  python run_cnn_decode_pipeline.py \
      --all_folds \
      --scripts_dir ./ \
      --data_root ../../../data/ \
      --folds_root ../../../runs/window_folds/ \
      --models_root ../../../runs/training_results/cnn_lopo/ \
      --decode_root ../../../decoded/cnn_decoded/ \
      --out_tag cnn \
      --alpha 0.35 \
      --K 2 \
      --p_switch 0.45 \
      --default_min_dur_s 0.0 \
      --min_dur_thresholds_s "Reach=0.30,Reposition=0.40,Stabilize=0.20" \
      --default_gap_dur_s 0.0 \
      --gap_thresholds_s "Reach=0.20,Reposition=0.30,Transport=0.20"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_ORDER = [
    "infer_windows.py",
    "smooth_inferences.py",
    "hysteresis.py",
    "min_duration_filter_cnn.py",
    "gap_merge_cnn.py",
    "segment_primitives.py",
    "decode_eval.py",
]


def default_script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_script(scripts_dir: Path, name: str) -> Path:
    path = scripts_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")
    return path


def validate_required_scripts(scripts_dir: Path) -> None:
    missing = [str(scripts_dir / name) for name in SCRIPT_ORDER if not (scripts_dir / name).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing required scripts:\n"
            f"  - {joined}\n"
            "Use --scripts_dir to point to the folder containing the CNN decode scripts."
        )


def build_mode_args(args: argparse.Namespace) -> List[str]:
    if args.all_folds:
        return ["--all_folds", "--folds_root", args.folds_root]
    if args.pid:
        return ["--pid", args.pid]
    raise ValueError("Provide either --pid or --all_folds.")


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pid", type=str, default=None,
                      help="Single participant ID, e.g. P22")
    mode.add_argument("--all_folds", action="store_true",
                      help="Run the full pipeline for every fold_* under --folds_root")

    # Script locations
    ap.add_argument("--scripts_dir", type=str, default=None,
                    help="Folder containing the CNN decode scripts. Defaults to the wrapper's directory.")

    # Shared locations
    ap.add_argument("--data_root", type=str, default="../../../data/")
    ap.add_argument("--folds_root", type=str, default="../../../runs/window_folds/")
    ap.add_argument("--models_root", type=str, default="../../../runs/training_results/cnn_lopo/")
    ap.add_argument("--decode_root", type=str, default="../../../decoded/cnn_decoded/",
                    help="Single shared directory where all per-pid decode .npz files live")

    # Model naming
    ap.add_argument("--model_dir_pattern", type=str, default="cnn_fold_{pid}")
    ap.add_argument("--model_filename", type=str, default="best.keras")

    # Decode naming
    ap.add_argument("--out_tag", type=str, default="cnn")

    # Decoding hyperparameters
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="EMA smoothing alpha")
    ap.add_argument("--K", type=int, default=2,
                    help="Hysteresis persistence length")
    ap.add_argument("--p_switch", type=float, default=0.45,
                    help="Hysteresis confidence threshold for switching")

    # Min-duration cleanup
    ap.add_argument("--default_min_dur_s", type=float, default=0.0,
                    help="Default minimum segment duration in seconds for classes not explicitly overridden")
    ap.add_argument("--min_dur_thresholds_s", type=str, default="",
                    help='Class-specific min-duration thresholds, e.g. "Reach=0.30,Reposition=0.40,Stabilize=0.20"')

    # Gap-merge cleanup
    ap.add_argument("--default_gap_dur_s", type=float, default=0.0,
                    help="Default maximum short-gap duration in seconds for A-B-A merging when class B is not explicitly overridden")
    ap.add_argument("--gap_thresholds_s", type=str, default="",
                    help='Class-specific gap thresholds, e.g. "Reach=0.20,Reposition=0.30,Transport=0.20"')

    # Eval output
    ap.add_argument("--metrics_json", type=str, default=None,
                    help="Optional explicit path for final metrics JSON. Defaults to <decode_root>/metrics_results.json")
    ap.add_argument("--print_per_pid", action="store_true")
    ap.add_argument("--print_segment_summary", action="store_true")
    ap.add_argument("--overwrite_metrics", action="store_true")

    # Utility
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without executing them")

    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir else default_script_dir()
    validate_required_scripts(scripts_dir)

    decode_root = os.path.abspath(args.decode_root)
    os.makedirs(decode_root, exist_ok=True)

    metrics_json = args.metrics_json
    if metrics_json is None:
        metrics_json = os.path.join(decode_root, "metrics_results.json")

    mode_args = build_mode_args(args)

    # Step 1: windowed inference
    cmd1 = [
        sys.executable,
        str(resolve_script(scripts_dir, "infer_windows.py")),
        *mode_args,
        "--data_root", args.data_root,
        "--folds_root", args.folds_root,
        "--models_root", args.models_root,
        "--model_dir_pattern", args.model_dir_pattern,
        "--model_filename", args.model_filename,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
    ]

    # Step 2: EMA smoothing
    cmd2 = [
        sys.executable,
        str(resolve_script(scripts_dir, "smooth_inferences.py")),
        *mode_args,
        "--folds_root", args.folds_root,
        "--predictions_root", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--alpha", str(args.alpha),
    ]

    # Step 3: hysteresis decoding
    cmd3 = [
        sys.executable,
        str(resolve_script(scripts_dir, "hysteresis.py")),
        *mode_args,
        "--folds_root", args.folds_root,
        "--in_dir", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--K", str(args.K),
        "--p_switch", str(args.p_switch),
    ]

    # Step 4: minimum-duration filtering
    cmd4 = [
        sys.executable,
        str(resolve_script(scripts_dir, "min_duration_filter_cnn.py")),
        *mode_args,
        "--folds_root", args.folds_root,
        "--in_dir", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--default_min_dur_s", str(args.default_min_dur_s),
    ]
    if args.min_dur_thresholds_s.strip():
        cmd4.extend(["--class_thresholds_s", args.min_dur_thresholds_s])

    # Step 5: gap merge
    cmd5 = [
        sys.executable,
        str(resolve_script(scripts_dir, "gap_merge_cnn.py")),
        *mode_args,
        "--folds_root", args.folds_root,
        "--in_dir", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--default_gap_dur_s", str(args.default_gap_dur_s),
    ]
    if args.gap_thresholds_s.strip():
        cmd5.extend(["--class_gap_thresholds_s", args.gap_thresholds_s])

    # Step 6: segment extraction
    cmd6 = [
        sys.executable,
        str(resolve_script(scripts_dir, "segment_primitives.py")),
        *mode_args,
        "--folds_root", args.folds_root,
        "--in_dir", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
    ]
    if args.print_segment_summary:
        cmd6.append("--print_summary")

    # Step 7: evaluation
    cmd7 = [
        sys.executable,
        str(resolve_script(scripts_dir, "decode_eval.py")),
        *mode_args,
        "--folds_root", args.folds_root,
        "--decoded_root", decode_root,
        "--segment_root", decode_root,
        "--out_tag", args.out_tag,
        "--out_json", metrics_json,
    ]
    if args.print_per_pid:
        cmd7.append("--print_per_pid")
    if args.overwrite_metrics:
        cmd7.append("--overwrite")

    for cmd in [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7]:
        run_cmd(cmd, dry_run=args.dry_run)

    print("\n[DONE] CNN decode pipeline finished.")
    print(f"[DECODE ROOT] {decode_root}")
    print(f"[METRICS JSON] {metrics_json}")


if __name__ == "__main__":
    main()