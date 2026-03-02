# smooth_inferences_tcn.py
#
# Step 2: EMA probability smoothing for full-sequence TCN predictions
#
# Input (per pid):
#   {predictions_root}/{pid}/{out_tag}_probs_raw.npz
#
# Output (per pid):
#   {out_dir}/{pid}/{out_tag}_probs_ema.npz

import os
import argparse
import numpy as np


def list_pids_from_folds(folds_root):
    """Scan folds_root for directories like 'fold_PXX' and return participant IDs."""
    pids = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            pid = name.split("fold_", 1)[-1]
            if pid:
                pids.append(pid)
    return pids


def ema_smooth(probs_raw, alpha):
    """Apply EMA along time axis: p_s[t] = alpha*p_s[t-1] + (1-alpha)*p_raw[t]."""
    if probs_raw.shape[0] == 0:
        return np.zeros_like(probs_raw, dtype=np.float32)

    probs_raw = probs_raw.astype(np.float32)
    probs_smooth = np.zeros_like(probs_raw, dtype=np.float32)
    probs_smooth[0] = probs_raw[0]

    for i in range(1, probs_raw.shape[0]):
        probs_smooth[i] = alpha * probs_smooth[i - 1] + (1.0 - alpha) * probs_raw[i]

    return probs_smooth


def run_one(pid, args):
    # input path: predictions_root/pid/out_tag_probs_raw.npz
    pid_in_dir = os.path.join(args.predictions_root, pid)
    in_path = os.path.join(pid_in_dir, f"{args.out_tag}_probs_raw.npz")

    data = np.load(in_path, allow_pickle=True)
    raw_probs = data["probs"].astype(np.float32)

    # EMA smoothing
    smooth_probs = ema_smooth(raw_probs, args.alpha)

    # argmax labels
    y_pred_raw = np.argmax(raw_probs, axis=1).astype(np.int32) if raw_probs.shape[0] else np.zeros((0,), dtype=np.int32)
    y_pred_smooth = np.argmax(smooth_probs, axis=1).astype(np.int32) if smooth_probs.shape[0] else np.zeros((0,), dtype=np.int32)

    # output path: out_dir/pid/out_tag_probs_ema.npz
    pid_out_dir = os.path.join(args.out_dir, pid)
    os.makedirs(pid_out_dir, exist_ok=True)
    out_path = os.path.join(pid_out_dir, f"{args.out_tag}_probs_ema.npz")

    np.savez(
        out_path,
        t=data["t"],
        probs_raw=raw_probs,
        probs_smooth=smooth_probs,
        y_pred_raw=y_pred_raw,
        y_pred_smooth=y_pred_smooth,
        class_list=data["class_list"],
        y_true=data["y_true"],
        alpha=args.alpha,
    )

    print(f"[OK] pid={pid} | saved={out_path} | T={raw_probs.shape[0]}")


def main():
    ap = argparse.ArgumentParser()

    # single or batch mode
    ap.add_argument("--pid", default=None, help="Single participant ID, e.g., P40")
    ap.add_argument("--all_folds", action="store_true", help="Run for every fold_* under folds_root")

    # where to find pids (TCN uses full_sequence_folds)
    ap.add_argument("--folds_root", default="../../../runs/full_sequence_folds/")

    # where raw predictions are + where to write EMA outputs
    ap.add_argument("--predictions_root", default="../../../decoded/causal_tcn_decoded/")
    ap.add_argument("--out_dir", default="../../../decoded/causal_tcn_decoded/")

    # file naming
    ap.add_argument("--out_tag", default="tcn", help="Matches inference out_tag (folder file prefix)")
    ap.add_argument("--alpha", type=float, default=0.6)

    args = ap.parse_args()

    if not args.all_folds and not args.pid:
        raise SystemExit("Provide --pid or --all_folds")

    pids = [args.pid] if args.pid else []
    if args.all_folds:
        pids = list_pids_from_folds(args.folds_root)

    n_ok = 0
    for pid in pids:
        try:
            run_one(pid, args)
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] pid={pid}: {e}")

    print(f"[DONE] Successfully processed {n_ok}/{len(pids)} participants.")


if __name__ == "__main__":
    main()