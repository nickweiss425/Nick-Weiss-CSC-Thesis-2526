#!/usr/bin/env python3
# decode_eval_basic_json_v2.py
#
# Save per-participant metrics + overall summary into a single JSON file.
#
# This version evaluates the FINAL post-processed CNN decoded sequence from:
#   {decoded_root}/{pid}/{out_tag}_window_gap_merged.npz
# using:
#   y_gap_merged
#
# It also reads predicted segment counts/durations from:
#   {segment_root}/{pid}/{out_tag}_segments.npz
#
# Metrics:
#   - time-weighted accuracy (TWA)
#   - relative count error
#   - relative duration error
#   - temporal IoU
#   - legacy MAE count / duration for backward compatibility

import os
import argparse
import numpy as np
import json
from datetime import datetime


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
            segs.append((int(cur), t0, t1))
            start_i = i
            cur = lab

    segs.append((int(cur), float(t_center[start_i]), float(t_center[N - 1])))
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


def time_weighted_accuracy(y_true, y_pred, t_center):
    """
    Time-weighted accuracy on the window-center grid.
    Under constant stride, this matches standard window-level accuracy.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if len(y_true) != len(y_pred):
        return np.nan

    dt = np.diff(t_center)
    if len(dt) == 0:
        return np.nan

    dt_med = float(np.median(dt))
    weights = np.full_like(y_true, dt_med, dtype=np.float32)
    correct = (y_true == y_pred).astype(np.float32)
    return float((correct * weights).sum() / weights.sum())


def macro_relative_error(pred_vals: np.ndarray, gt_vals: np.ndarray):
    """
    Mean absolute relative error over classes with nonzero GT.
    Returns:
      macro_relative_error, per_class_relative_error_array
    """
    pred_vals = np.asarray(pred_vals, dtype=np.float32)
    gt_vals = np.asarray(gt_vals, dtype=np.float32)

    rel = np.full(gt_vals.shape, np.nan, dtype=np.float32)
    valid = gt_vals > 0
    rel[valid] = np.abs(pred_vals[valid] - gt_vals[valid]) / gt_vals[valid]

    return float(np.nanmean(rel)) if np.any(valid) else np.nan, rel


def class_iou(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """Per-class temporal IoU on aligned timeline labels."""
    ious = np.full((n_classes,), np.nan, dtype=np.float32)
    for k in range(n_classes):
        true_k = (y_true == k)
        pred_k = (y_pred == k)
        union = np.sum(true_k | pred_k)
        if union > 0:
            inter = np.sum(true_k & pred_k)
            ious[k] = inter / union
    return ious


def sanitize_col(name: str) -> str:
    """Make class names safe for JSON keys."""
    return str(name).strip().replace(" ", "_")


def run_one(pid: str, args):
    # Load predicted segments
    seg_path = os.path.join(args.segment_root, pid, f"{args.out_tag}_segments.npz")
    pred = np.load(seg_path, allow_pickle=True)

    class_list = [str(x) for x in pred["class_list"].tolist()]
    pred_counts = pred["counts"].astype(np.int32)
    pred_durs = pred["durations"].astype(np.float32)

    # Load FINAL decoded labels + GT
    dec_path = os.path.join(args.decoded_root, pid, f"{args.out_tag}_window_gap_merged.npz")
    decoded = np.load(dec_path, allow_pickle=True)

    t_center = decoded["t_center"].astype(np.float32)
    y_true = decoded["y_true"].astype(np.int32)
    y_pred_dec = decoded["y_gap_merged"].astype(np.int32)

    # Build GT segments on same time grid
    gt_segs = labels_to_segments(t_center, y_true)
    gt_counts, gt_durs = counts_and_durations(gt_segs, n_classes=len(class_list))

    # Metrics
    twa = time_weighted_accuracy(y_true, y_pred_dec, t_center)

    count_err = pred_counts - gt_counts
    dur_err = pred_durs - gt_durs
    mae_count = float(np.mean(np.abs(count_err))) if len(count_err) else np.nan
    mae_dur = float(np.mean(np.abs(dur_err))) if len(dur_err) else np.nan

    rel_count_err_macro, rel_count_err_per_class = macro_relative_error(pred_counts, gt_counts)
    rel_dur_err_macro, rel_dur_err_per_class = macro_relative_error(pred_durs, gt_durs)
    iou_per_class = class_iou(y_true, y_pred_dec, n_classes=len(class_list))
    mean_iou = float(np.nanmean(iou_per_class)) if np.any(~np.isnan(iou_per_class)) else np.nan

    row = {
        "pid": pid,
        "twa": float(twa) if not np.isnan(twa) else None,
        "rel_count_err": float(rel_count_err_macro) if not np.isnan(rel_count_err_macro) else None,
        "rel_dur_err": float(rel_dur_err_macro) if not np.isnan(rel_dur_err_macro) else None,
        "mean_iou": float(mean_iou) if not np.isnan(mean_iou) else None,
        "mae_count": float(mae_count) if not np.isnan(mae_count) else None,
        "mae_dur_s": float(mae_dur) if not np.isnan(mae_dur) else None,
        "per_class": {}
    }

    for i, name in enumerate(class_list):
        col = sanitize_col(name)
        row["per_class"][col] = {
            "name": name,
            "gt_count": int(gt_counts[i]),
            "pred_count": int(pred_counts[i]),
            "count_err": int(count_err[i]),
            "rel_count_err": float(rel_count_err_per_class[i]) if not np.isnan(rel_count_err_per_class[i]) else None,
            "gt_dur_s": float(gt_durs[i]),
            "pred_dur_s": float(pred_durs[i]),
            "dur_err_s": float(dur_err[i]),
            "rel_dur_err": float(rel_dur_err_per_class[i]) if not np.isnan(rel_dur_err_per_class[i]) else None,
            "iou": float(iou_per_class[i]) if not np.isnan(iou_per_class[i]) else None,
        }

    return row, class_list


def list_pids_from_folds(folds_root):
    """Scan folds_root for directories like 'fold_PXX' and return participant IDs."""
    pids = []
    if not os.path.isdir(folds_root):
        return pids
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            pid = name.split("fold_", 1)[-1]
            if pid:
                pids.append(pid)
    return pids


def summarize_rows(rows: list[dict], class_list: list):
    """Compute mean±std across pids for core metrics and per-class metrics."""
    def arr_float(key):
        vals = []
        for r in rows:
            v = r.get(key)
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=np.float32)

    summary = {}

    for k in ["twa", "rel_count_err", "rel_dur_err", "mean_iou", "mae_count", "mae_dur_s"]:
        a = arr_float(k)
        summary[k] = {
            "mean": float(np.nanmean(a)) if a.size > 0 else None,
            "std": float(np.nanstd(a)) if a.size > 0 else None
        }

    per_class = {}
    for name in class_list:
        col = sanitize_col(name)
        count_errs, rel_count_errs, dur_errs, rel_dur_errs, ious = [], [], [], [], []

        for r in rows:
            p = r.get("per_class", {})
            if col in p:
                count_errs.append(float(p[col]["count_err"]))
                rel_count_errs.append(np.nan if p[col]["rel_count_err"] is None else float(p[col]["rel_count_err"]))
                dur_errs.append(float(p[col]["dur_err_s"]))
                rel_dur_errs.append(np.nan if p[col]["rel_dur_err"] is None else float(p[col]["rel_dur_err"]))
                ious.append(np.nan if p[col]["iou"] is None else float(p[col]["iou"]))
            else:
                count_errs.append(np.nan)
                rel_count_errs.append(np.nan)
                dur_errs.append(np.nan)
                rel_dur_errs.append(np.nan)
                ious.append(np.nan)

        ce = np.array(count_errs, dtype=np.float32)
        rce = np.array(rel_count_errs, dtype=np.float32)
        de = np.array(dur_errs, dtype=np.float32)
        rde = np.array(rel_dur_errs, dtype=np.float32)
        iou = np.array(ious, dtype=np.float32)

        per_class[col] = {
            "name": name,
            "count_err_mean": float(np.nanmean(ce)) if ce.size > 0 else None,
            "count_err_std": float(np.nanstd(ce)) if ce.size > 0 else None,
            "rel_count_err_mean": float(np.nanmean(rce)) if np.any(~np.isnan(rce)) else None,
            "rel_count_err_std": float(np.nanstd(rce)) if np.any(~np.isnan(rce)) else None,
            "dur_err_mean_s": float(np.nanmean(de)) if de.size > 0 else None,
            "dur_err_std_s": float(np.nanstd(de)) if de.size > 0 else None,
            "rel_dur_err_mean": float(np.nanmean(rde)) if np.any(~np.isnan(rde)) else None,
            "rel_dur_err_std": float(np.nanstd(rde)) if np.any(~np.isnan(rde)) else None,
            "iou_mean": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else None,
            "iou_std": float(np.nanstd(iou)) if np.any(~np.isnan(iou)) else None,
        }

    summary["per_class"] = per_class
    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pid", default=None, help="Single participant ID, e.g., P40")
    ap.add_argument("--all_folds", action="store_true", help="Evaluate all fold_* dirs under folds_root")
    ap.add_argument("--folds_root", default="../../../runs/window_folds/")

    ap.add_argument("--decoded_root", default="../../../decoded/cnn_decoded/")
    ap.add_argument("--segment_root", default="../../../decoded/cnn_decoded/")
    ap.add_argument("--out_tag", default="cnn")
    ap.add_argument("--out_json", default="../../../decoded/cnn_decoded/metrics_results.json")
    ap.add_argument("--print_per_pid", action="store_true", help="Print one-line metrics per pid")
    ap.add_argument("--overwrite", action="store_true", help="If set, delete existing out_json before writing new results")

    args = ap.parse_args()

    if not args.all_folds and not args.pid:
        raise SystemExit("Provide --pid or --all_folds")

    pids = [args.pid] if args.pid else []
    if args.all_folds:
        pids = list_pids_from_folds(args.folds_root)

    if args.overwrite and os.path.exists(args.out_json):
        os.remove(args.out_json)

    participants = {}
    rows_for_summary = []
    class_list_ref = None
    failed = []
    n_ok = 0

    for pid in pids:
        try:
            row, class_list = run_one(pid, args)
            if class_list_ref is None:
                class_list_ref = class_list
            participants[pid] = row
            rows_for_summary.append(row)
            n_ok += 1

            if args.print_per_pid:
                twa_str = f"{row['twa']:.4f}" if row['twa'] is not None else "nan"
                rce_str = f"{row['rel_count_err']:.4f}" if row['rel_count_err'] is not None else "nan"
                rde_str = f"{row['rel_dur_err']:.4f}" if row['rel_dur_err'] is not None else "nan"
                miou_str = f"{row['mean_iou']:.4f}" if row['mean_iou'] is not None else "nan"
                print(f"[EVAL] pid={pid} | twa={twa_str} | rel_count_err={rce_str} | rel_dur_err={rde_str} | mean_iou={miou_str}")

        except Exception as e:
            print(f"[ERROR] pid={pid}: {e}")
            participants[pid] = {"pid": pid, "error": str(e)}
            failed.append(pid)

    summary = None
    if rows_for_summary and class_list_ref:
        summary = summarize_rows(rows_for_summary, class_list_ref)

    out_struct = {
        "meta": {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "n_requested": len(pids),
            "n_success": n_ok,
            "n_failed": len(failed),
            "failed_pids": failed,
        },
        "class_list": class_list_ref if class_list_ref is not None else [],
        "participants": participants,
        "summary": summary
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_struct, f, indent=2)

    print(f"[DONE] Successfully evaluated {n_ok}/{len(pids)} participants.")
    print(f"[SAVED JSON] {args.out_json}")


if __name__ == "__main__":
    main()