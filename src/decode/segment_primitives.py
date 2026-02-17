import os, argparse
import numpy as np

"""
Step 4: Segment Extraction and Primitive Quantification

Purpose
-------
Convert temporally decoded window-level labels into contiguous primitive
segments with start/end times and durations. Then compute per-class
primitive counts and total durations.

This step transforms a window-level classification sequence into
functional movement summaries suitable for evaluation and clinical
interpretation.
"""

def labels_to_segments(t_center: np.ndarray, y: np.ndarray):
    """
    Convert per-window labels into contiguous temporal segments.

    Parameters
    ----------
    t_center : np.ndarray, shape (N,)
        Timestamp corresponding to the center of each sliding window.
    y : np.ndarray, shape (N,)
        Decoded primitive label (integer index) for each window.

    Returns
    -------
    segs : list of dict
        Each dictionary contains:
            - label_idx : int
            - t_start   : float (segment start time)
            - t_end     : float (segment end time)
            - duration_s: float (t_end - t_start)
            - n_steps   : int (number of windows in segment)

    Notes
    -----
    - Segment boundaries are defined at window center timestamps.
    - Consecutive identical labels are merged into a single segment.
    - Temporal resolution is limited by the window stride.
    """
    N = len(y)
    if N == 0:
        return []

    segs = []
    start_i = 0
    cur = int(y[0])

    # loop through all window labels
    for i in range(1, N):
        # get label for particular window
        lab = int(y[i])

        # if label is new, then we need to create segment
        if lab != cur:
            # get starting time
            t0 = float(t_center[start_i])

            # get ending time
            t1 = float(t_center[i - 1])

            # create dictionary object for segment, append to list
            segs.append({
                "label_idx": cur,
                "t_start": t0,
                "t_end": t1,
                "duration_s": max(0.0, t1 - t0),
                "n_steps": int(i - start_i),
            })

            # reset starting index and current label
            start_i = i
            cur = lab

    # last segment
    t0 = float(t_center[start_i])
    t1 = float(t_center[N - 1])
    segs.append({
        "label_idx": cur,
        "t_start": t0,
        "t_end": t1,
        "duration_s": max(0.0, t1 - t0),
        "n_steps": int(N - start_i),
    })
    return segs


def count_and_duration(segs, n_classes: int):
    """
    Compute total primitive counts and cumulative durations per class.

    Parameters
    ----------
    segs : list of dict
        Segment list produced by labels_to_segments().
    n_classes : int
        Total number of primitive classes.

    Returns
    -------
    counts : np.ndarray, shape (n_classes,)
        Number of segments detected for each class.
    durs : np.ndarray, shape (n_classes,)
        Total time (seconds) spent in each class.

    Notes
    -----
    - Each contiguous segment contributes exactly one count.
    - Durations are summed across all segments of the same class.
    - Does not account for temporal alignment or ordering.
    """
    # initialize lists of 0s for counts and durations
    counts = np.zeros((n_classes,), dtype=np.int32)
    durs = np.zeros((n_classes,), dtype=np.float32)

    # loop through all segments
    for s in segs:
        # get class label
        k = int(s["label_idx"])

        # increment count of that primitive class
        counts[k] += 1

        # increment duration of that primitive class
        durs[k] += float(s["duration_s"])

    return counts, durs


def main():
    """
    Load hysteresis-decoded window predictions, extract temporal segments,
    compute per-class primitive counts and durations, and save results.

    Input
    -----
    {pid}_window_hysteresis_decoded.npz

    Output
    ------
    {pid}_segments.npz containing:
        - segment-level arrays (labels, start/end times, durations, window counts)
        - per-class counts
        - per-class total durations
        - class metadata

    This output is used for downstream evaluation (count error,
    duration error, boundary timing analysis, etc.).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--in_dir", default="../../decoded/cnn_decoded/P32")
    ap.add_argument("--out_dir", default="../../decoded/cnn_decoded/P32")
    args = ap.parse_args()

    pid = args.pid
    in_path = os.path.join(args.in_dir, f"{pid}_window_hysteresis_decoded.npz")
    d = np.load(in_path, allow_pickle=True)

    t_center = d["t_center"].astype(np.float32)
    y_dec = d["y_decoded"].astype(np.int32)
    class_list = d["class_list"]

    # convert window labels into segments with specific start/stop times
    segs = labels_to_segments(t_center, y_dec)

    # get total overall counts and durations per primitive
    counts, durs = count_and_duration(segs, n_classes=len(class_list))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{pid}_segments.npz")

    # store segments as arrays for easy loading later
    seg_label = np.array([s["label_idx"] for s in segs], dtype=np.int32)
    seg_t_start    = np.array([s["t_start"] for s in segs], dtype=np.float32)
    seg_t_end    = np.array([s["t_end"] for s in segs], dtype=np.float32)
    seg_dur   = np.array([s["duration_s"] for s in segs], dtype=np.float32)
    seg_n     = np.array([s["n_steps"] for s in segs], dtype=np.int32)

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
        durations=durs
    )

    print(f"[SAVED] {out_path}")
    print("\n[COUNTS + TOTAL DURATION (s)]")
    for i, name in enumerate(class_list):
        print(f"  {name:10s}  count={int(counts[i]):4d}  dur={float(durs[i]):8.2f}s")
    print(f"\nTotal segments: {len(segs)}")

if __name__ == "__main__":
    main()
