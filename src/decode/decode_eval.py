import os, argparse
import numpy as np
import csv

def labels_to_segments(t_center: np.ndarray, y: np.ndarray):
    """Convert per-window labels into segments. Returns list of (label, t0, t1)."""
    N = len(y)
    if N == 0:
        return []

    segs = []
    start_i = 0
    cur = int(y[0])

    for i in range(1, N):
        lab = int(y[i])
        if lab != cur:
            t0 = float(t_center[start_i])
            t1 = float(t_center[i - 1])
            segs.append((cur, t0, t1))
            start_i = i
            cur = lab

    segs.append((cur, float(t_center[start_i]), float(t_center[N - 1])))
    return segs

def counts_and_durations(segs, n_classes: int):
    """
    Compute per-class segment counts and total durations.

    For each segment (label, t0, t1), increments the class count
    and accumulates duration (t1 - t0).

    Returns:
        counts : (n_classes,) segment counts
        durs   : (n_classes,) total duration (s)
    """
    counts = np.zeros(n_classes, dtype=np.int32)
    durs = np.zeros(n_classes, dtype=np.float32)
    for lab, t0, t1 in segs:
        counts[lab] += 1
        durs[lab] += max(0.0, (t1 - t0))
    return counts, durs

def time_weighted_accuracy(y_true, y_pred, t_center):
    """
    Compute time-weighted accuracy between predicted and true labels.

    Each window is assigned a time weight equal to the median spacing
    between adjacent t_center values. Accuracy is calculated as:

        (total time correctly labeled) / (total time)

    Under uniform stride, this reduces to standard window-level accuracy.
    """
    if len(y_true) == 0:
        return np.nan

    dt = np.diff(t_center)
    if len(dt) == 0:
        return np.nan

    dt_med = float(np.median(dt))
    weights = np.full_like(y_true, dt_med, dtype=np.float32)

    correct = (y_true == y_pred).astype(np.float32)
    return float((correct * weights).sum() / weights.sum())

def boundary_times(y):
    """Return indices where label changes (boundary occurs between i-1 and i)."""
    return np.where(y[1:] != y[:-1])[0] + 1

def append_metrics_row(out_csv_path: str, row: dict, fieldnames: list[str]):
    """
    Append a single row to a CSV. If file doesn't exist, create it with header.
    Assumes consistent fieldnames across runs.
    """
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(out_csv_path)

    if not file_exists:
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)
    else:
        # If the CSV exists, assume the same header; append.
        with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--segment_dir", default="../../decoded/cnn_decoded/P32")
    ap.add_argument("--decoded_dir", default="../../decoded/cnn_decoded/P32")
    ap.add_argument("--out_csv", default="../../decoded/cnn_decoded/metrics_results.csv",
                    help="CSV file to append metrics (one row per pid).")
    args = ap.parse_args()

    pid = args.pid

    # load predicted segments
    pred_path = os.path.join(args.segment_dir, f"{pid}_segments.npz")
    pred = np.load(pred_path, allow_pickle=True)

    class_list = pred["class_list"]

    # get counts and durations per primitive
    pred_counts = pred["counts"].astype(np.int32)
    pred_durs = pred["durations"].astype(np.float32)

    # load ground-truth window labels and timestamps (from hysteresis output file)
    decoded_path = os.path.join(args.decoded_dir, f"{pid}_window_hysteresis_decoded.npz")
    decoded = np.load(decoded_path, allow_pickle=True)
    t_center = decoded["t_center"].astype(np.float32)
    y_true = decoded["y_true"].astype(np.int32)

    # build ground truth segments from y_true using same t_center grid
    gt_segs = labels_to_segments(t_center, y_true)
    gt_counts, gt_durs = counts_and_durations(gt_segs, n_classes=len(class_list))

    # get time weighted accuracy
    y_pred_dec = decoded["y_decoded"].astype(np.int32)
    twa = time_weighted_accuracy(y_true, y_pred_dec, t_center)

    # boundary timing error (approx)
    b_gt = boundary_times(y_true)
    b_pr = boundary_times(y_pred_dec)

    boundary_errs = []
    if len(b_gt) > 0 and len(b_pr) > 0:
        pr_times = t_center[b_pr]
        for idx in b_gt:
            t = float(t_center[idx])
            j = int(np.argmin(np.abs(pr_times - t)))
            boundary_errs.append(abs(float(pr_times[j]) - t))
    mean_boundary_err = float(np.mean(boundary_errs)) if boundary_errs else np.nan

    # report count/duration errors
    count_err = pred_counts - gt_counts
    dur_err = pred_durs - gt_durs

    mae_count = float(np.mean(np.abs(count_err))) if len(count_err) else np.nan
    mae_dur = float(np.mean(np.abs(dur_err))) if len(dur_err) else np.nan

    print(f"\n[STEP 6 EVAL] pid={pid}")
    print(f"Time-weighted accuracy (decoded vs y_true): {twa:.4f}")
    if not np.isnan(mean_boundary_err):
        print(f"Mean boundary timing error (s): {mean_boundary_err:.3f}")
    else:
        print("Mean boundary timing error (s): n/a")

    print("\nPer-class counts and durations:")
    for i, name in enumerate(class_list):
        print(
            f"{name:10s}  "
            f"GT count={int(gt_counts[i]):4d}  Pred count={int(pred_counts[i]):4d}  Err={int(count_err[i]):+4d}  |  "
            f"GT dur={float(gt_durs[i]):8.2f}s  Pred dur={float(pred_durs[i]):8.2f}s  Err={float(dur_err[i]):+8.2f}s"
        )

    print("\nAggregate errors:")
    print(f"  Mean abs count error per class: {mae_count:.3f}")
    print(f"  Mean abs duration error per class (s): {mae_dur:.3f}")

    # ----------------------------
    # SAVE METRICS (one row / pid)
    # ----------------------------
    row = {
        "pid": pid,
        "twa": twa,
        "mean_boundary_err_s": mean_boundary_err,
        "mae_count": mae_count,
        "mae_dur_s": mae_dur,
    }

    # per-class errors as separate columns (easy to average later)
    for i, name in enumerate(class_list):
        # sanitize class name for column keys
        col = str(name).strip().replace(" ", "_")
        row[f"count_err_{col}"] = int(count_err[i])
        row[f"dur_err_{col}_s"] = float(dur_err[i])
        row[f"gt_count_{col}"] = int(gt_counts[i])
        row[f"pred_count_{col}"] = int(pred_counts[i])
        row[f"gt_dur_{col}_s"] = float(gt_durs[i])
        row[f"pred_dur_{col}_s"] = float(pred_durs[i])

    fieldnames = list(row.keys())
    append_metrics_row(args.out_csv, row, fieldnames)

    print(f"\n[SAVED METRICS] {args.out_csv} (appended row for {pid})")

if __name__ == "__main__":
    main()
