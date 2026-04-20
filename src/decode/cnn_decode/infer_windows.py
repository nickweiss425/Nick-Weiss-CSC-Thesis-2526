# infer_windows_cnn.py
#
# Runs windowed inference for trained CNN models.
# Can process a single participant (--pid) or all LOPO folds (--all_folds).
#
# Output: one .npz per participant saved to:
#   {out_dir}/{pid}/{out_tag}_window_probs_raw.npz
#
# Contents:
#   - t_center  : window center timestamps
#   - probs     : per-window class probabilities (N, K)
#   - y_pred    : argmax predictions (N,)
#   - y_true    : integer labels for window centers (N,) (empty if not available)
#   - class_list, feature_cols, win, stride

import os, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

TIME_COL = "Time (s)"
LABEL_COL = "Primitive"


def make_windows(df, feature_cols, win, stride, drop_label="Unknown"):
    """
    Build overlapping windows and label each window using its center sample label.

    Returns:
      Xw_raw     : (N, win, C) float32
      t_center   : (N,) float32
      y_center_s : (N,) object (class names) or None
    """
    # feature matrix and time vector
    X = df[feature_cols].to_numpy(np.float32)
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(np.float32)

    # label vector (strings), if present
    y = df[LABEL_COL].astype(str).to_numpy() if LABEL_COL in df.columns else None

    X_list, t_list, y_list = [], [], []
    n = len(df)

    for start in range(0, n - win + 1, stride):
        center = start + win // 2

        # If labels exist, drop windows whose center label is Unknown
        if y is not None:
            center_label = y[center]
            if center_label == drop_label:
                continue
            y_list.append(center_label)

        X_list.append(X[start:start + win])
        t_list.append(float(t[center]))

    if len(X_list) == 0:
        # Return empty arrays (still well-typed)
        C = X.shape[1]
        return (
            np.zeros((0, win, C), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            None
        )

    return (
        np.stack(X_list).astype(np.float32),
        np.array(t_list, dtype=np.float32),
        (np.array(y_list, dtype=object) if y is not None else None)
    )


def y_to_int(y_str, class_list):
    """Map class-name strings to integer labels using class_list ordering."""
    m = {c: i for i, c in enumerate(class_list)}
    return np.array([m[v] for v in y_str], dtype=np.int32)


def list_pids_from_folds(folds_root):
    """
    Scan folds_root for directories like 'fold_PXX'
    and return the extracted participant IDs.
    """
    pids = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            pid = name.split("fold_", 1)[-1]
            if pid:
                pids.append(pid)
    return pids


def run_one(pid, args):
    """
    Run inference for a single participant.
    """

    # --------------------------------------------------
    # 1) Load engineered data (continuous full sequence)
    # --------------------------------------------------
    engineered_csv = os.path.join(args.data_root, pid, "engineered.csv")
    df = pd.read_csv(engineered_csv)

    # --------------------------------------------------
    # 2) Load fold metadata + scaler + EMG scaling
    #    (ensures inference matches training transforms)
    # --------------------------------------------------
    fold_dir = os.path.join(args.folds_root, f"fold_{pid}")

    with open(os.path.join(fold_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    emg_env_cols = meta.get("emg_env_cols", [])
    class_list = meta["class_list"]
    win = int(meta["win_samples"])
    stride = int(meta["stride_samples"])

    # Apply EMG p95 scaling (if used in training)
    if emg_env_cols:
        p95 = np.load(os.path.join(fold_dir, "p95_median_train.npy")).astype(np.float32)
        df.loc[:, emg_env_cols] = df[emg_env_cols].to_numpy(np.float32) / p95

    # --------------------------------------------------
    # 3) Build windowed dataset
    # --------------------------------------------------
    Xw_raw, t_center, y_center_str = make_windows(
        df, feature_cols, win, stride, drop_label=args.drop_label
    )

    # If no windows were produced, still write a valid empty output
    if Xw_raw.shape[0] == 0:
        out_dir = os.path.join(args.out_dir, f"{pid}")
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, f"{args.out_tag}_window_probs_raw.npz")

        np.savez(
            out_npz,
            t_center=t_center,
            probs=np.zeros((0, len(class_list)), dtype=np.float32),
            y_pred=np.zeros((0,), dtype=np.int32),
            y_true=np.zeros((0,), dtype=np.int32),
            class_list=np.array(class_list, dtype=object),
            feature_cols=np.array(feature_cols, dtype=object),
            win=win,
            stride=stride,
        )

        print(f"[OK] pid={pid} | windows=0 (saved empty)")
        return

    # --------------------------------------------------
    # 4) Feature standardization using fold scaler
    # --------------------------------------------------
    scaler = joblib.load(os.path.join(fold_dir, "scaler.joblib"))
    n, w, c = Xw_raw.shape
    Xw = scaler.transform(Xw_raw.reshape(-1, c)).reshape(n, w, c).astype(np.float32)

    # --------------------------------------------------
    # 5) Load trained CNN model for this fold
    # --------------------------------------------------
    model_dir = os.path.join(
        args.models_root,
        args.model_dir_pattern.format(pid=pid)
    )
    model_path = os.path.join(model_dir, args.model_filename)
    model = tf.keras.models.load_model(model_path)

    # --------------------------------------------------
    # 6) Inference on all windows
    # --------------------------------------------------
    probs = model.predict(Xw, verbose=0).astype(np.float32)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    # Build y_true if label strings exist
    y_true = np.array([], dtype=np.int32)
    if y_center_str is not None:
        y_true = y_to_int(y_center_str, class_list)

    # --------------------------------------------------
    # 7) Save results to disk (match TCN layout)
    # --------------------------------------------------
    out_dir = os.path.join(args.out_dir, f"{pid}")
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"{args.out_tag}_window_probs_raw.npz")

    np.savez(
        out_npz,
        t_center=t_center,
        probs=probs,
        y_pred=y_pred,
        y_true=y_true,
        class_list=np.array(class_list, dtype=object),
        feature_cols=np.array(feature_cols, dtype=object),
        win=win,
        stride=stride,
    )

    print(f"[OK] pid={pid} | windows={len(t_center)} | probs={probs.shape} | has_y_true={len(y_true) > 0}")


def main():
    """
    Entry point.
    Either:
      --pid PXX        (single participant)
    or:
      --all_folds      (process all fold_* dirs)
    """
    ap = argparse.ArgumentParser()

    # single or batch mode
    ap.add_argument("--pid", default=None,
                    help="Single participant ID, e.g., P40")
    ap.add_argument("--all_folds", action="store_true",
                    help="Run inference for every fold_* directory")

    # data + fold locations (match your updated TCN defaults style)
    ap.add_argument("--data_root", default="../../../data/")
    ap.add_argument("--folds_root", default="../../../runs/window_folds/")

    # Model locations (CNN has the same pattern idea as TCN)
    ap.add_argument("--models_root",
                    default="../../../runs/training_results/cnn_lopo/")
    ap.add_argument("--model_dir_pattern",
                    default="cnn_fold_{pid}")
    ap.add_argument("--model_filename", default="best.keras")

    # Output (match TCN layout: out_dir/pid/out_tag_*.npz)
    ap.add_argument("--out_dir", default="../../../decoded/cnn_decoded/")
    ap.add_argument("--out_tag", default="cnn")

    ap.add_argument("--drop_label", default="Unknown")

    args = ap.parse_args()

    # Must specify either single pid or batch mode
    if not args.all_folds and not args.pid:
        raise SystemExit("Provide --pid or --all_folds")

    # Determine which participants to process
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