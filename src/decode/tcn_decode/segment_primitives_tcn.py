# segment_primitives_tcn.py
#
# Step 4: Segment Extraction and Primitive Quantification (TCN full sequence)
#
# Input (per pid):
#   {in_dir}/{pid}/{out_tag}_hysteresis_decoded.npz
#
# Output (per pid):
#   {out_dir}/{pid}/{out_tag}_segments.npz
#
# Supports:
#   --pid PXX
#   --all_folds  (enumerates fold_* under --folds_root)

import os
import argparse
import numpy as np


def labels_to_segments(t: np.ndarray, y: np.ndarray):
    """
    Convert per-sample labels into contiguous temporal segments.

    Parameters
    ----------
    t : (T,) float32
        Timestamp for each sample.
    y : (T,) int32
        Decoded primitive label for each sample.

    Returns
    -------
    segs : list of dicts
        label_idx, t_start, t_end, duration_s, n_steps
    """
    T = len(y)
    if T == 0:
        return []

    segs = []
    start_i = 0
    cur = int(y[0])

    for i in range(1, T):
        lab = int(y[i])
        if lab != cur:
            t0 = float(t[start_i])
            t1 = float(t[i - 1])
            segs.append({
                "label_idx": cur,
                "t_start": t0,
                "t_end": t1,
                "duration_s": max(0.0, t1 - t0),
                "n_steps": int(i - start_i),
            })
            start_i = i
            cur = lab

    # last segment
    t0 = float(t[start_i])
    t1 = float(t[T - 1])
    segs.append({
        "label_idx": cur,
        "t_start": t0,
        "t_end": t1,
        "duration_s": max(0.0, t1 - t0),
        "n_steps": int(T - start_i),
    })
    return segs


def count_and_duration(segs, n_classes: int):
    """Compute per-class segment counts and total durations."""
    counts = np.zeros((n_classes,), dtype=np.int32)
    durs = np.zeros((n_classes,), dtype=np.float32)

    for s in segs:
        k = int(s["label_idx"])
        if 0 <= k < n_classes:
            counts[k] += 1
            durs[k] += float(s["duration_s"])

    return counts, durs


def list_pids_from_folds(folds_root):
    """Scan folds_root for directories like 'fold_PXX' and return participant IDs."""
    pids = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            pid = name.split("fold_", 1)[-1]
            if pid:
                pids.append(pid)
    return pids


def run_one(pid, args):
    # input: in_dir/pid/out_tag_hysteresis_decoded.npz
    pid_in_dir = os.path.join(args.in_dir, pid)
    in_path = os.path.join(pid_in_dir, f"{args.out_tag}_hysteresis_decoded.npz")
    d = np.load(in_path, allow_pickle=True)

    t = d["t"].astype(np.float32)
    y_dec = d["y_decoded"].astype(np.int32)
    class_list = d["class_list"]

    # segments
    segs = labels_to_segments(t, y_dec)

    # counts + durations
    counts, durs = count_and_duration(segs, n_classes=len(class_list))

    # output: out_dir/pid/out_tag_segments.npz
    pid_out_dir = os.path.join(args.out_dir, pid)
    os.makedirs(pid_out_dir, exist_ok=True)
    out_path = os.path.join(pid_out_dir, f"{args.out_tag}_segments.npz")

    seg_label = np.array([s["label_idx"] for s in segs], dtype=np.int32)
    seg_t_start = np.array([s["t_start"] for s in segs], dtype=np.float32)
    seg_t_end = np.array([s["t_end"] for s in segs], dtype=np.float32)
    seg_dur = np.array([s["duration_s"] for s in segs], dtype=np.float32)
    seg_n = np.array([s["n_steps"] for s in segs], dtype=np.int32)

    np.savez(
        out_path,
        class_list=class_list,
        n_classes=len(class_list),
        seg_label=seg_label,
        seg_t_start=seg_t_start,
        seg_t_end=seg_t_end,
        seg_dur=seg_dur,
        seg_n=seg_n,
        counts=counts,
        durations=durs,
    )

    if args.print_summary:
        print(f"[OK] pid={pid} | saved={out_path} | segments={len(segs)}")
        for i, name in enumerate(class_list):
            print(f"  {str(name):10s}  count={int(counts[i]):4d}  dur={float(durs[i]):8.2f}s")
    else:
        print(f"[OK] pid={pid} | saved={out_path} | segments={len(segs)}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pid", default=None, help="Single participant ID, e.g., P40")
    ap.add_argument("--all_folds", action="store_true", help="Run for every fold_* under folds_root")

    ap.add_argument("--folds_root", default="../../../runs/full_sequence_folds/")
    ap.add_argument("--in_dir", default="../../../decoded/causal_tcn_decoded/")
    ap.add_argument("--out_dir", default="../../../decoded/causal_tcn_decoded/")
    ap.add_argument("--out_tag", default="tcn")

    ap.add_argument("--print_summary", action="store_true")

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