# hysteresis_cnn.py
#
# Step 3: Hysteresis decoding for CNN window predictions (after EMA)
#
# Input (per pid):
#   {in_dir}/{pid}/{out_tag}_window_probs_ema.npz
#
# Output (per pid):
#   {out_dir}/{pid}/{out_tag}_window_hysteresis_decoded.npz

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


def hysteresis_decode(probs, K=3, p_switch=0.60):
    """
    Causal hysteresis decoding:
      - candidate must be confident (>= p_switch)
      - must persist for K consecutive steps before switching state
    """
    N = probs.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int32)

    decoded = np.zeros(N, dtype=np.int32)
    state = int(np.argmax(probs[0]))
    count = 0
    decoded[0] = state

    for i in range(1, N):
        cand = int(np.argmax(probs[i]))
        conf = float(np.max(probs[i]))

        if cand == state:
            count = 0
        else:
            if conf >= p_switch:
                count += 1
            else:
                count = 0

            if count >= K:
                state = cand
                count = 0

        decoded[i] = state

    return decoded


def run_one(pid, args):
    # input path: in_dir/pid/out_tag_window_probs_ema.npz
    pid_in_dir = os.path.join(args.in_dir, pid)
    in_path = os.path.join(pid_in_dir, f"{args.out_tag}_window_probs_ema.npz")

    data = np.load(in_path, allow_pickle=True)
    probs = data["probs_smooth"].astype(np.float32)

    y_decoded = hysteresis_decode(probs, K=args.K, p_switch=args.p_switch)

    # output path: out_dir/pid/out_tag_window_hysteresis_decoded.npz
    pid_out_dir = os.path.join(args.out_dir, pid)
    os.makedirs(pid_out_dir, exist_ok=True)
    out_path = os.path.join(pid_out_dir, f"{args.out_tag}_window_hysteresis_decoded.npz")

    np.savez(
        out_path,
        t_center=data["t_center"],
        probs_smooth=probs,
        y_pred_smooth=data["y_pred_smooth"],
        y_decoded=y_decoded,
        y_pred_raw=data["y_pred_raw"],
        y_true=data["y_true"],
        class_list=data["class_list"],
        alpha=data["alpha"],
        win=data.get("win", None),
        stride=data.get("stride", None),
        K=args.K,
        p_switch=args.p_switch,
    )

    frac_changed = float(np.mean(y_decoded != data["y_pred_smooth"])) if y_decoded.shape[0] else 0.0
    print(f"[OK] pid={pid} | saved={out_path} | windows={probs.shape[0]} | frac_changed={frac_changed:.4f}")


def main():
    ap = argparse.ArgumentParser()

    # single or batch mode
    ap.add_argument("--pid", default=None, help="Single participant ID, e.g., P40")
    ap.add_argument("--all_folds", action="store_true", help="Run for every fold_* under folds_root")

    # where to find pids
    ap.add_argument("--folds_root", default="../../../runs/window_folds/")

    # input/output directories
    ap.add_argument("--in_dir", default="../../../decoded/cnn_decoded/")
    ap.add_argument("--out_dir", default="../../../decoded/cnn_decoded/")

    # file naming
    ap.add_argument("--out_tag", default="cnn", help="Must match EMA out_tag")

    # hysteresis params
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--p_switch", type=float, default=0.45)

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