"""
Step 3: Hysteresis-Based Temporal Decoding

Purpose
-------
Convert smoothed window-level probabilities into a temporally stable
label sequence using hysteresis.

Even after probability smoothing (EMA), brief fluctuations or early
boundary crossings may remain. Hysteresis introduces decision inertia:
a state transition is only allowed if a new class is both sufficiently
confident and persistent over multiple consecutive windows.

This reduces spurious short segments and enforces physically plausible
primitive transitions.

Inputs
------
- probs_smooth (N, C): Smoothed class probabilities per window
- K: Minimum consecutive confirmations required before switching
- p_switch: Minimum confidence threshold required to consider switching

Outputs
-------
- y_decoded (N,): Final temporally stabilized label sequence
"""

import os
import argparse
import numpy as np


def hysteresis_decode(probs, K=3, p_switch=0.60):
    """
    Apply hysteresis-based decoding to smoothed probabilities.

    Parameters
    ----------
    probs : np.ndarray, shape (N, C)
        Smoothed probability matrix (e.g., after EMA).
    K : int
        Number of consecutive high-confidence windows required
        before switching to a new state.
    p_switch : float
        Minimum confidence threshold required to consider a switch.

    Returns
    -------
    decoded : np.ndarray, shape (N,)
        Final decoded label sequence with temporal persistence enforced.

    Notes
    -----
    - The decoder is causal (uses only past information).
    - State transitions are delayed by up to K windows.
    - Prevents single-window flips and low-confidence oscillations.
    """

    N = probs.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int32)

    decoded = np.zeros(N, dtype=np.int32)

    # initialize state from first window
    state = int(np.argmax(probs[0]))
    count = 0
    decoded[0] = state

    for i in range(1, N):
        # get output candidate and confidence in that prediction
        cand = int(np.argmax(probs[i]))
        conf = float(np.max(probs[i]))

        # if model agrees with current state, reset counter
        if cand == state:
            count = 0
        # if predicted state is different than current state
        else:
            # onyl consider switching if confidence exceeds threshold
            if conf >= p_switch:
                count += 1
            else:
                count = 0

            # switch only after K consecutive confirmations
            if count >= K:
                state = cand
                count = 0

        decoded[i] = state

    return decoded


def main():
    """
    Load smoothed window probabilities, apply hysteresis decoding,
    and save the final temporally stabilized label sequence.

    This step operates after EMA smoothing and before segment extraction.
    It produces the final window-level labels used for primitive
    segmentation, counting, and duration estimation.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--in_dir", default="../../decoded/cnn_decoded/P32")
    ap.add_argument("--out_dir", default="../../decoded/cnn_decoded/P32")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--p_switch", type=float, default=0.60)
    args = ap.parse_args()

    pid = args.pid

    in_path = os.path.join(args.in_dir, f"{pid}_window_probs_ema.npz")
    data = np.load(in_path, allow_pickle=True)

    probs = data["probs_smooth"]

    # run hysteresis on smoothed probabilities
    y_decoded = hysteresis_decode(
        probs,
        K=args.K,
        p_switch=args.p_switch
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{pid}_window_hysteresis_decoded.npz")

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
        K=args.K,
        p_switch=args.p_switch
    )

    print(f"[SAVED] {out_path}")

    frac_changed = np.mean(y_decoded != data["y_pred_smooth"])
    print(f"Fraction changed by hysteresis vs EMA: {frac_changed:.4f}")


if __name__ == "__main__":
    main()
