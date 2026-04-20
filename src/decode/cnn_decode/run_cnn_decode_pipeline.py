#!/usr/bin/env python3
"""
run_cnn_decode_pipeline.py

Wrapper to run the full CNN decoding pipeline in order:
  1) infer_windows.py
  2) smooth_inferences.py
  3) hysteresis.py
  4) segment_primitives.py
  5) decode_eval.py

Design goal:
- one shared decode directory for all intermediate/final .npz files
- supports either a single participant (--pid) or all folds (--all_folds)
- keeps per-script naming conventions intact
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pid", type=str, default=None, help="Single participant ID, e.g. P22")
    mode.add_argument("--all_folds", action="store_true", help="Run full pipeline for all fold_* dirs")

    ap.add_argument("--scripts_dir", type=str, default=None,
                    help="Directory containing infer_windows.py, smooth_inferences.py, hysteresis.py, segment_primitives.py, decode_eval.py. Defaults to this wrapper's directory.")

    ap.add_argument("--data_root", type=str, default="../../../data/")
    ap.add_argument("--folds_root", type=str, default="../../../runs/window_folds/")
    ap.add_argument("--models_root", type=str, default="../../../runs/training_results/cnn_lopo/")
    ap.add_argument("--model_dir_pattern", type=str, default="cnn_fold_{pid}")
    ap.add_argument("--model_filename", type=str, default="best.keras")

    ap.add_argument("--decode_root", type=str, default="../../../decoded/cnn_decoded/")
    ap.add_argument("--out_tag", type=str, default="cnn")
    ap.add_argument("--drop_label", type=str, default="Unknown")

    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--p_switch", type=float, default=0.45)

    ap.add_argument("--metrics_json", type=str, default=None,
                    help="Final metrics JSON path. Defaults to {decode_root}/metrics_results.json")
    ap.add_argument("--overwrite_metrics", action="store_true",
                    help="Pass --overwrite to decode_eval.py")
    ap.add_argument("--print_per_pid", action="store_true",
                    help="Pass --print_per_pid to decode_eval.py")
    ap.add_argument("--print_segment_summary", action="store_true",
                    help="Pass --print_summary to segment_primitives.py")

    args = ap.parse_args()

    wrapper_dir = Path(__file__).resolve().parent
    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir else wrapper_dir

    infer_script = scripts_dir / "infer_windows.py"
    smooth_script = scripts_dir / "smooth_inferences.py"
    hysteresis_script = scripts_dir / "hysteresis.py"
    segment_script = scripts_dir / "segment_primitives.py"
    eval_script = scripts_dir / "decode_eval.py"

    required = [infer_script, smooth_script, hysteresis_script, segment_script, eval_script]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing required scripts:\n  - " + "\n  - ".join(missing) +
            "\nUse --scripts_dir to point to the folder containing the CNN decode scripts."
        )

    decode_root = os.path.abspath(args.decode_root)
    metrics_json = os.path.abspath(args.metrics_json) if args.metrics_json else os.path.join(decode_root, "metrics_results.json")
    os.makedirs(decode_root, exist_ok=True)

    mode_flags = ["--pid", args.pid] if args.pid else ["--all_folds"]

    infer_cmd = [
        sys.executable, str(infer_script),
        *mode_flags,
        "--data_root", args.data_root,
        "--folds_root", args.folds_root,
        "--models_root", args.models_root,
        "--model_dir_pattern", args.model_dir_pattern,
        "--model_filename", args.model_filename,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--drop_label", args.drop_label,
    ]

    smooth_cmd = [
        sys.executable, str(smooth_script),
        *mode_flags,
        "--folds_root", args.folds_root,
        "--predictions_root", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--alpha", str(args.alpha),
    ]

    hysteresis_cmd = [
        sys.executable, str(hysteresis_script),
        *mode_flags,
        "--folds_root", args.folds_root,
        "--in_dir", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
        "--K", str(args.K),
        "--p_switch", str(args.p_switch),
    ]

    segment_cmd = [
        sys.executable, str(segment_script),
        *mode_flags,
        "--folds_root", args.folds_root,
        "--in_dir", decode_root,
        "--out_dir", decode_root,
        "--out_tag", args.out_tag,
    ]
    if args.print_segment_summary:
        segment_cmd.append("--print_summary")

    eval_cmd = [
        sys.executable, str(eval_script),
        *mode_flags,
        "--folds_root", args.folds_root,
        "--decoded_root", decode_root,
        "--segment_root", decode_root,
        "--out_tag", args.out_tag,
        "--out_json", metrics_json,
    ]
    if args.overwrite_metrics:
        eval_cmd.append("--overwrite")
    if args.print_per_pid:
        eval_cmd.append("--print_per_pid")

    run_cmd(infer_cmd)
    run_cmd(smooth_cmd)
    run_cmd(hysteresis_cmd)
    run_cmd(segment_cmd)
    run_cmd(eval_cmd)

    print("\n[DONE] CNN decode pipeline complete.")
    print(f"[DECODE ROOT] {decode_root}")
    print(f"[METRICS JSON] {metrics_json}")


if __name__ == "__main__":
    main()
