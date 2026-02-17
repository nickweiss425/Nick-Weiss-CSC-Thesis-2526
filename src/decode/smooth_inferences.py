import os, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

def main():
    """
    Probability Smoothing (Exponential Moving Average)

    Purpose
    -------
    The CNN produces a probability vector at every sliding window (e.g., every 50 ms).
    Although the model may be highly accurate, predictions near primitive boundaries
    can fluctuate due to overlapping windows and mixed-label regions. Even in stable
    segments, occasional single-window misclassifications may occur.

    Probability smoothing reduces short-term jitter by applying an exponential
    moving average (EMA) across time:

        p_smooth[t] = alpha * p_smooth[t-1] + (1 - alpha) * p_raw[t]

    This introduces temporal inertia, ensuring that predictions reflect the
    continuous nature of human movement rather than independent per-window decisions.

    Why This Is Needed
    ------------------
    Sliding windows overlap heavily (e.g., 1-second window with 50 ms stride).
    Near a true transition from "Reach" to "Transport", several consecutive windows
    contain mixed motion patterns. The raw model output may look like:

        Time    Reach   Transport   Pred
        --------------------------------
        t0      0.92     0.05       Reach
        t1      0.85     0.12       Reach
        t2      0.61     0.34       Reach
        t3      0.49     0.47       Reach
        t4      0.44     0.52       Transport
        t5      0.28     0.70       Transport
        t6      0.10     0.88       Transport

    At t3–t4 the top class may briefly flip back and forth, even though the
    physical transition is gradual.

    After EMA smoothing (alpha=0.9), the transition becomes more stable:

        Time    Reach_s  Transport_s   Pred
        ------------------------------------
        t0      0.92       0.05         Reach
        t1      0.91       0.06         Reach
        t2      0.88       0.09         Reach
        t3      0.84       0.13         Reach
        t4      0.78       0.19         Reach
        t5      0.70       0.28         Reach
        t6      0.60       0.38         Reach
        t7      0.49       0.50         Transport

    The model no longer reacts to single-window uncertainty spikes.
    Transitions still occur, but in a physically plausible, smooth manner.

    Inputs
    ------
    - probs_raw: (N, K) array of softmax probabilities
    - alpha: smoothing coefficient (0 < alpha < 1)

    Outputs
    -------
    - probs_smooth: (N, K) smoothed probabilities
    - y_pred_smooth: argmax labels derived from probs_smooth

    Effect
    ------
    - Reduces one-frame misclassifications
    - Stabilizes boundary regions
    - Preserves true transitions
    - Improves segment extraction in later decoding steps
    """


    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True, help="Held-out participant ID, e.g., P40")
    ap.add_argument("--predictions_root", default="../../decoded/cnn_decoded/P32")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--out_dir", default="../../decoded/cnn_decoded/P32")

    args = ap.parse_args()

    # participant id to perform inference on
    pid = args.pid

    raw_infer = np.load(os.path.join(args.predictions_root, f"{pid}_window_probs_raw.npz"), allow_pickle=True)
    raw_probs = raw_infer["probs"].astype(np.float32)
    t_center = raw_infer["t_center"]
    class_list = raw_infer["class_list"]
    y_true = raw_infer["y_true"]

    # --- EMA smoothing ---

    # make np array same shape as raw probabilities
    smooth_probs = np.zeros_like(raw_probs, dtype=np.float32)

    smooth_probs[0] = raw_probs[0]

    for i in range(1, len(raw_probs)):
        smooth_probs[i] = (
            args.alpha * smooth_probs[i - 1]
            + (1 - args.alpha) * raw_probs[i]
        )

    # new predicted labels
    y_pred_raw = np.argmax(raw_probs, axis=1)
    y_pred_smooth = np.argmax(smooth_probs, axis=1)

    # --- Save ---
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"{pid}_window_probs_ema.npz")

    np.savez(
        out_path,
        t_center=t_center,
        probs_raw=raw_probs,
        probs_smooth=smooth_probs,
        y_pred_raw=y_pred_raw,
        y_pred_smooth=y_pred_smooth,
        class_list=class_list,
        y_true=y_true,
        alpha=args.alpha
    )

    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()