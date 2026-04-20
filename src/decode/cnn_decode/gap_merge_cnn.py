#!/usr/bin/env python3
# gap_merge_cnn.py
#
# Step 5: Short-gap merging for CNN window predictions
#
# Input (per pid):
#   {in_dir}/{pid}/{out_tag}_window_mindur_filtered.npz
#
# Output (per pid):
#   {out_dir}/{pid}/{out_tag}_window_gap_merged.npz
#
# Supports class-specific gap thresholds via:
#   --default_gap_dur_s
#   --class_gap_thresholds_s "Reach=0.20,Reposition=0.30,Transport=0.20"

import os
import argparse
import numpy as np


def list_pids_from_folds(folds_root):
    pids = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            pid = name.split("fold_", 1)[-1]
            if pid:
                pids.append(pid)
    return pids


def parse_class_thresholds(spec: str):
    out = {}
    spec = (spec or "").strip()
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad class threshold item: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def labels_to_segments(y: np.ndarray):
    T = len(y)
    if T == 0:
        return []

    segs = []
    start = 0
    cur = int(y[0])
    for i in range(1, T):
        lab = int(y[i])
        if lab != cur:
            segs.append({"label": cur, "start": start, "end": i - 1})
            start = i
            cur = lab
    segs.append({"label": cur, "start": start, "end": T - 1})
    return segs


def segment_duration_s(seg, t_center: np.ndarray, dt_default: float):
    if seg["end"] < seg["start"]:
        return 0.0
    if seg["end"] == seg["start"]:
        return dt_default
    return float(max(dt_default, t_center[seg["end"]] - t_center[seg["start"]]))


def estimate_dt(t_center: np.ndarray):
    if len(t_center) < 2:
        return 0.0
    dt = np.diff(t_center.astype(np.float32))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0
    return float(np.median(dt))


def apply_gap_merge(y_in, t_center, class_list, default_gap_dur_s, class_gap_thresholds):
    y = np.asarray(y_in, dtype=np.int32).copy()
    if len(y) == 0:
        return y

    dt_default = estimate_dt(t_center)

    changed = True
    while changed:
        changed = False
        segs = labels_to_segments(y)
        if len(segs) < 3:
            break

        for idx in range(1, len(segs) - 1):
            left = segs[idx - 1]
            mid = segs[idx]
            right = segs[idx + 1]

            if left["label"] != right["label"]:
                continue

            surround_name = str(class_list[int(left["label"])])
            thr = float(class_gap_thresholds.get(surround_name, default_gap_dur_s))
            if thr <= 0:
                continue

            gap_dur = segment_duration_s(mid, t_center, dt_default)
            if gap_dur <= thr:
                y[mid["start"]:mid["end"] + 1] = int(left["label"])
                changed = True
                break

    return y


def run_one(pid, args):
    pid_in_dir = os.path.join(args.in_dir, pid)
    in_path = os.path.join(pid_in_dir, f"{args.out_tag}_window_mindur_filtered.npz")
    d = np.load(in_path, allow_pickle=True)

    t_center = d["t_center"].astype(np.float32)
    y_filtered = d["y_mindur_filtered"].astype(np.int32)
    class_list = [str(x) for x in d["class_list"].tolist()]

    class_gap_thresholds = parse_class_thresholds(args.class_gap_thresholds_s)
    y_gap_merged = apply_gap_merge(
        y_filtered,
        t_center,
        class_list,
        default_gap_dur_s=args.default_gap_dur_s,
        class_gap_thresholds=class_gap_thresholds,
    )

    pid_out_dir = os.path.join(args.out_dir, pid)
    os.makedirs(pid_out_dir, exist_ok=True)
    out_path = os.path.join(pid_out_dir, f"{args.out_tag}_window_gap_merged.npz")

    frac_changed = float(np.mean(y_gap_merged != y_filtered)) if len(y_filtered) else 0.0

    np.savez(
        out_path,
        t_center=t_center,
        y_decoded=d["y_decoded"],
        y_mindur_filtered=y_filtered,
        y_gap_merged=y_gap_merged,
        y_true=d["y_true"],
        class_list=d["class_list"],
        probs_smooth=d["probs_smooth"],
        y_pred_smooth=d["y_pred_smooth"],
        y_pred_raw=d["y_pred_raw"],
        alpha=d["alpha"],
        K=d["K"],
        p_switch=d["p_switch"],
        win=d.get("win", None),
        stride=d.get("stride", None),
        default_min_dur_s=d.get("default_min_dur_s", None),
        class_thresholds_s=d.get("class_thresholds_s", None),
        default_gap_dur_s=float(args.default_gap_dur_s),
        class_gap_thresholds_s=str(args.class_gap_thresholds_s),
    )

    print(f"[OK] pid={pid} | saved={out_path} | windows={len(y_gap_merged)} | frac_changed={frac_changed:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=None, help="Single participant ID, e.g., P40")
    ap.add_argument("--all_folds", action="store_true", help="Run for every fold_* under folds_root")

    ap.add_argument("--folds_root", default="../../../runs/window_folds/")
    ap.add_argument("--in_dir", default="../../../decoded/cnn_decoded/")
    ap.add_argument("--out_dir", default="../../../decoded/cnn_decoded/")
    ap.add_argument("--out_tag", default="cnn")

    ap.add_argument("--default_gap_dur_s", type=float, default=0.0)
    ap.add_argument("--class_gap_thresholds_s", type=str, default="")

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
