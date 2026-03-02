# decode_eval_basic_tcn.py
#
# Basic clinically-relevant decoding metrics (TCN full sequence)
#
# Per pid:
#   - TWA (time-weighted accuracy) of decoded labels vs y_true
#   - Per-class count error (pred - gt)
#   - Per-class duration error (pred - gt) in seconds
#   - Aggregate MAE across classes for counts + durations
#
# Inputs (per pid):
#   Segments: {segment_root}/{pid}/{out_tag}_segments.npz
#   Decoded : {decoded_root}/{pid}/{out_tag}_hysteresis_decoded.npz
#
# Outputs:
#   CSV appended at {out_csv} (one row per pid)
#   Prints mean ± std across folds when --all_folds is used

import os
import argparse
import numpy as np
import csv


def labels_to_segments(t: np.ndarray, y: np.ndarray):
    """Convert per-sample labels into segments. Returns list of (label, t0, t1)."""
    T = len(y)
    if T == 0:
        return []

    segs = []
    start_i = 0
    cur = int(y[0])

    for i in range(1, T):
        lab = int(y[i])
        if lab != cur:
            segs.append((cur, float(t[start_i]), float(t[i - 1])))
            start_i = i
            cur = lab

    segs.append((cur, float(t[start_i]), float(t[T - 1])))
    return segs


def counts_and_durations(segs, n_classes: int):
    """Compute per-class segment counts and total durations from (label, t0, t1) segments."""
    counts = np.zeros(n_classes, dtype=np.int32)
    durs = np.zeros(n_classes, dtype=np.float32)
    for lab, t0, t1 in segs:
        if 0 <= int(lab) < n_classes:
            counts[int(lab)] += 1
            durs[int(lab)] += max(0.0, (t1 - t0))
    return counts, durs


def time_weighted_accuracy(y_true, y_pred, t):
    """
    Time-weighted accuracy on the per-sample timeline.
    Uses median dt as a uniform weight (robust to occasional irregularity).
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if len(y_true) != len(y_pred):
        return np.nan

    dt = np.diff(t)
    if len(dt) == 0:
        return np.nan

    dt_med = float(np.median(dt))
    weights = np.full_like(y_true, dt_med, dtype=np.float32)
    correct = (y_true == y_pred).astype(np.float32)
    return float((correct * weights).sum() / weights.sum())


def append_metrics_row(out_csv_path: str, row: dict, fieldnames: list[str]):
    """Append a row to CSV; create file + header if it doesn't exist."""
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(out_csv_path)

    if not file_exists:
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
    else:
        with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)


def list_pids_from_folds(folds_root):
    """Scan folds_root for directories like 'fold_PXX' and return participant IDs."""
    pids = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            pid = name.split("fold_", 1)[-1]
            if pid:
                pids.append(pid)
    return pids


def sanitize_col(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def run_one(pid: str, args):
    # Predicted segments
    seg_path = os.path.join(args.segment_root, pid, f"{args.out_tag}_segments.npz")
    pred = np.load(seg_path, allow_pickle=True)

    class_list = pred["class_list"]
    pred_counts = pred["counts"].astype(np.int32)
    pred_durs = pred["durations"].astype(np.float32)

    # Decoded labels + GT
    dec_path = os.path.join(args.decoded_root, pid, f"{args.out_tag}_hysteresis_decoded.npz")
    decoded = np.load(dec_path, allow_pickle=True)

    t = decoded["t"].astype(np.float32)
    y_true = decoded["y_true"].astype(np.int32)
    y_pred_dec = decoded["y_decoded"].astype(np.int32)

    # Build GT segments
    gt_segs = labels_to_segments(t, y_true)
    gt_counts, gt_durs = counts_and_durations(gt_segs, n_classes=len(class_list))

    # Metrics
    twa = time_weighted_accuracy(y_true, y_pred_dec, t)
    count_err = pred_counts - gt_counts
    dur_err = pred_durs - gt_durs

    mae_count = float(np.mean(np.abs(count_err))) if len(count_err) else np.nan
    mae_dur = float(np.mean(np.abs(dur_err))) if len(dur_err) else np.nan

    if args.print_per_pid:
        print(f"[EVAL] pid={pid} | twa={twa:.4f} | mae_count={mae_count:.3f} | mae_dur_s={mae_dur:.3f}")

    row = {"pid": pid, "twa": twa, "mae_count": mae_count, "mae_dur_s": mae_dur}

    for i, name in enumerate(class_list):
        col = sanitize_col(name)
        row[f"count_err_{col}"] = int(count_err[i])
        row[f"dur_err_{col}_s"] = float(dur_err[i])
        row[f"gt_count_{col}"] = int(gt_counts[i])
        row[f"pred_count_{col}"] = int(pred_counts[i])
        row[f"gt_dur_{col}_s"] = float(gt_durs[i])
        row[f"pred_dur_{col}_s"] = float(pred_durs[i])

    return row, list(class_list)


def summarize_rows(rows: list[dict], class_list: list):
    def arr(key):
        return np.array([r.get(key, np.nan) for r in rows], dtype=np.float32)

    summary = {}
    for k in ["twa", "mae_count", "mae_dur_s"]:
        a = arr(k)
        summary[k] = (float(np.nanmean(a)), float(np.nanstd(a)))

    for name in class_list:
        col = sanitize_col(name)
        for k in [f"count_err_{col}", f"dur_err_{col}_s"]:
            a = arr(k)
            summary[k] = (float(np.nanmean(a)), float(np.nanstd(a)))

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pid", default=None)
    ap.add_argument("--all_folds", action="store_true")
    ap.add_argument("--folds_root", default="../../../runs/full_sequence_folds/")

    ap.add_argument("--decoded_root", default="../../../decoded/causal_tcn_decoded/")
    ap.add_argument("--segment_root", default="../../../decoded/causal_tcn_decoded/")
    ap.add_argument("--out_tag", default="tcn")

    ap.add_argument("--out_csv", default="../../../decoded/causal_tcn_decoded/metrics_results.csv")
    ap.add_argument("--overwrite_csv", action="store_true")
    ap.add_argument("--print_per_pid", action="store_true")

    args = ap.parse_args()

    if not args.all_folds and not args.pid:
        raise SystemExit("Provide --pid or --all_folds")

    pids = [args.pid] if args.pid else []
    if args.all_folds:
        pids = list_pids_from_folds(args.folds_root)

    if args.overwrite_csv and os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    rows = []
    class_list_ref = None

    n_ok = 0
    for pid in pids:
        try:
            row, class_list = run_one(pid, args)
            if class_list_ref is None:
                class_list_ref = class_list

            append_metrics_row(args.out_csv, row, list(row.keys()))
            rows.append(row)
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] pid={pid}: {e}")

    print(f"[DONE] Successfully evaluated {n_ok}/{len(pids)} participants.")
    print(f"[SAVED] {args.out_csv}")

    if rows and class_list_ref:
        summary = summarize_rows(rows, class_list_ref)

        print("\n[SUMMARY ACROSS FOLDS] mean ± std")
        print(f"  twa:        {summary['twa'][0]:.4f} ± {summary['twa'][1]:.4f}")
        print(f"  mae_count:  {summary['mae_count'][0]:.3f} ± {summary['mae_count'][1]:.3f}")
        print(f"  mae_dur_s:  {summary['mae_dur_s'][0]:.3f} ± {summary['mae_dur_s'][1]:.3f}")

        print("\n[PER-CLASS ERROR MEANS] (pred - gt)")
        for name in class_list_ref:
            col = sanitize_col(name)
            mc = summary[f"count_err_{col}"][0]
            md = summary[f"dur_err_{col}_s"][0]
            print(f"  {str(name):10s}  mean_count_err={mc:+.3f}  mean_dur_err_s={md:+.3f}")


if __name__ == "__main__":
    main()