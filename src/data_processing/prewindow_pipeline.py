"""
run_prewindow_pipeline.py

Wrapper to run the full *pre-windowing* pipeline for a single participant:

1) Build labeled continuous file(s) for the participant
2) check for missing Primitive labels 
3) Feature engineering: write engineered.csv 

"""

import argparse
import subprocess
import sys
import os


STEP1_SCRIPT = "sync_and_label.py"  

def run(cmd: list[str]) -> None:
    """Run a command and raise if it fails."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(data_root: str, participant: str, sensors: list[str] | None):
    # basic sanity check
    pdir = os.path.join(data_root, participant)
    if not os.path.isdir(pdir):
        raise FileNotFoundError(f"Participant folder not found: {pdir}")

    # ---- Step 1: build + sync + trim + label (creates labeled.csv) ----
    step1_cmd = [
        sys.executable,
        STEP1_SCRIPT,
        "--data_root", data_root,
        "--participant", participant,
    ]
    if sensors:
        step1_cmd += ["--sensors", *sensors]

    run(step1_cmd)

    # ---- Step 2: QC missing labels (participant-only) ----
    run([
        sys.executable,
        "check_missing_labels.py",
        "--data_root", data_root,
        "--participant", participant,
    ])

    # ---- Step 3: feature engineering (participant-only) ----
    step3_cmd = [
        sys.executable,
        "feature_engineer.py",
        "--data_root", data_root,
        "--participant", participant,
    ]
    if sensors:
        step3_cmd += ["--sensors", *sensors]

    run(step3_cmd)

    print("\n[DONE] Pre-windowing pipeline completed for", participant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--participant", type=str, required=True)
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="Optional sensor IDs (e.g., A5F2 A19E). Passed through to step 1 and feature_engineer.",
    )
    args = parser.parse_args()

    main(args.data_root, args.participant, args.sensors)
