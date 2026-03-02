# infer_fullseq_tcn.py
#
# Runs full-sequence inference for trained TCN models.
# Can process a single participant (--pid) or all LOPO folds (--all_folds).
#
# Output: one .npz file per participant containing:
#   - t           : per-sample timestamps
#   - probs       : per-sample class probabilities (T, K)
#   - y_pred      : argmax predictions (T,)
#   - y_true      : ground-truth labels (Unknown = -1)
#   - class_list  : class name list
#   - feature_cols: feature column names used by the model

import os, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

TIME_COL = "Time (s)"
LABEL_COL = "Primitive"


def y_to_int_fullseq(y_str, class_list, unknown_label="Unknown"):
    """
    Convert string labels to integer indices.
    Unknown (or anything not in class_list) -> -1.
    """
    mapping = {c: i for i, c in enumerate(class_list)}
    y_int = np.full((len(y_str),), -1, dtype=np.int32)

    for i, label in enumerate(y_str):
        if label == unknown_label:
            continue
        if label in mapping:
            y_int[i] = mapping[label]

    return y_int


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

    # Extract time vector
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(np.float32)

    # Apply EMG p95 scaling (if used in training)
    if emg_env_cols:
        p95 = np.load(os.path.join(fold_dir, "p95_median_train.npy")).astype(np.float32)
        df.loc[:, emg_env_cols] = df[emg_env_cols].to_numpy(np.float32) / p95

    # --------------------------------------------------
    # 3) Feature extraction + standardization
    # --------------------------------------------------
    X_raw = df[feature_cols].to_numpy(np.float32)

    scaler = joblib.load(os.path.join(fold_dir, "scaler.joblib"))
    X = scaler.transform(X_raw).astype(np.float32)

    # --------------------------------------------------
    # 4) Build ground-truth label vector
    #    Unknown timesteps are set to -1
    # --------------------------------------------------
    y_true = np.full((len(df),), -1, dtype=np.int32)

    if LABEL_COL in df.columns:
        y_str = df[LABEL_COL].astype(str).to_numpy(dtype=object)
        y_true = y_to_int_fullseq(y_str, class_list, unknown_label=args.unknown_label)

    # --------------------------------------------------
    # 5) Load trained TCN model for this fold
    # --------------------------------------------------
    model_dir = os.path.join(
        args.models_root,
        args.model_dir_pattern.format(pid=pid)
    )
    model_path = os.path.join(model_dir, args.model_filename)

    model = tf.keras.models.load_model(model_path)

    # --------------------------------------------------
    # 6) Run full-sequence inference
    #    Model expects shape (batch, T, C)
    # --------------------------------------------------
    probs = model.predict(X[None, :, :], verbose=0).astype(np.float32)

    # Handle possible (1, T, K) shape
    if probs.ndim == 3:
        probs = probs[0]

    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    # --------------------------------------------------
    # 7) Save results to disk
    # --------------------------------------------------
    out_dir = os.path.join(args.out_dir, f"{pid}")
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"{args.out_tag}_probs_raw.npz")

    np.savez(
        out_npz,
        t=t,
        probs=probs,
        y_pred=y_pred,
        y_true=y_true,
        class_list=np.array(class_list, dtype=object),
        feature_cols=np.array(feature_cols, dtype=object),
    )

    print(
        f"[OK] pid={pid} | T={len(t)} | probs={probs.shape} "
        f"| unknown_frac={float(np.mean(y_true == -1)):.3f}"
    )


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

    # data + fold locations
    ap.add_argument("--data_root", default="../../../data/")
    ap.add_argument("--folds_root", default="../../../runs/full_sequence_folds/")

    # Model locations
    ap.add_argument("--models_root",
                    default="../../../runs/training_results/tcn_fullseq_causal_lopo/")
    ap.add_argument("--model_dir_pattern",
                    default="tcn_fullseq_causal_fold_{pid}")
    ap.add_argument("--model_filename", default="best.keras")

    # Output
    ap.add_argument("--out_dir", default="../../../decoded/causal_tcn_decoded/")
    ap.add_argument("--out_tag", default="tcn")

    ap.add_argument("--unknown_label", default="Unknown")

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