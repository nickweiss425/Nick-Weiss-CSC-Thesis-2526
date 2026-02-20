#!/usr/bin/env python3
"""
prepare_fullseq_folds.py (batch_size=1 friendly, simplest version)

Creates LOPO folds for FULL-SEQUENCE dense supervision (per-timestep labels),
keeping variable-length sequences (NO padding) so training can use batch_size=1.

Key points:
- Unknown occurs inside trials (station change). We KEEP it in the sequence but MASK it out.
- y uses UNKNOWN_ID=-1 for Unknown timesteps.
- mask is 1 for supervised timesteps (known primitives), 0 for Unknown.
- EMG envelope p95 scaling is computed from TRAINING participants only (excluding Unknown rows),
  per participant when available; held-out uses median(training p95).
- StandardScaler is fit on ALL TRAINING TIMESTEPS after p95 scaling (concatenated),
  then applied to train/val/test sequences.
- Saves X/y/mask as NumPy object arrays so each element retains its natural length:
    X_train.npy: object array of length N_train_seqs, each item shape (T_i, C)
    y_train.npy: object array of length N_train_seqs, each item shape (T_i,) int32
    mask_train.npy: object array of length N_train_seqs, each item shape (T_i,) float32
  (Same for val/test)

Output structure:
  ../../runs/full_sequence_folds/fold_PXX/
    meta.json
    scaler.joblib
    p95_median_train.npy
    p95_train_by_pid.npy
    train_pids_order.json
    X_train.npy / y_train.npy / mask_train.npy / seq_lens_train.npy
    X_val.npy   / y_val.npy   / mask_val.npy   / seq_lens_val.npy
    X_test.npy  / y_test.npy  / mask_test.npy  / seq_lens_test.npy
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

TIME_COL  = "Time (s)"
LABEL_COL = "Primitive"
PID_COL   = "participant_id"

UNKNOWN_LABEL = "Unknown"
UNKNOWN_ID    = -1

# ---------- IO ----------
def list_pids(data_root):
    pids = []
    for d in sorted(os.listdir(data_root)):
        pdir = os.path.join(data_root, d)
        if not os.path.isdir(pdir):
            continue
        if os.path.exists(os.path.join(pdir, "engineered.csv")):
            pids.append(d)
    return pids

def load_engineered(data_root, pid):
    path = os.path.join(data_root, pid, "engineered.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing engineered.csv for {pid}: {path}")
    df = pd.read_csv(path)
    df[PID_COL] = pid
    return df

def estimate_fs(df):
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(float)
    t = t[~np.isnan(t)]
    if len(t) < 3:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else 0.0

# ---------- Feature selection ----------
def get_feature_cols(df):
    excl = {TIME_COL, LABEL_COL, PID_COL}
    cols = [c for c in df.columns if c not in excl]
    cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    return sorted(cols)

def get_emg_env_cols(feature_cols):
    return [c for c in feature_cols if ("_EMG" in c and c.endswith("_ENV"))]

# ---------- Labels ----------
def build_class_list(dfs):
    classes = set()
    for df in dfs.values():
        classes.update(df[LABEL_COL].astype(str).unique().tolist())
    classes.discard(UNKNOWN_LABEL)
    return sorted(classes)

def make_label_map(class_list):
    return {c: i for i, c in enumerate(class_list)}

def labels_to_int_and_mask(label_str_arr, label_map):
    """
    label_str_arr: (T,) dtype str
    Returns:
      y_int: (T,) int32 with Unknown -> -1
      mask:  (T,) float32, 1 for known labels, 0 for Unknown
    """
    T = len(label_str_arr)
    y_int = np.full((T,), UNKNOWN_ID, dtype=np.int32)
    mask = np.zeros((T,), dtype=np.float32)
    for i, s in enumerate(label_str_arr):
        if s in label_map:
            y_int[i] = label_map[s]
            mask[i] = 1.0
        else:
            y_int[i] = UNKNOWN_ID
            mask[i] = 0.0
    return y_int, mask

# ---------- Normalization (EMG p95 + StandardScaler) ----------
def compute_p95_from_dfs(dfs, emg_env_cols, pids, min_p95=1e-6):
    """
    Compute per-participant EMG envelope p95 from NON-Unknown rows only.
    Returns dict pid -> p95_vec (n_emg,)
    """
    out = {}
    for pid in pids:
        df = dfs[pid]
        keep = df[LABEL_COL].astype(str).to_numpy() != UNKNOWN_LABEL
        if not np.any(keep):
            continue
        emg = df.loc[keep, emg_env_cols].to_numpy(np.float32)
        if emg.size == 0:
            continue
        p95 = np.percentile(emg, 95, axis=0)
        p95 = np.maximum(p95, min_p95).astype(np.float32)
        out[pid] = p95
    return out

def apply_p95_to_df(df, emg_env_cols, p95_vec):
    df2 = df.copy()
    if len(emg_env_cols) > 0:
        df2.loc[:, emg_env_cols] = df2.loc[:, emg_env_cols].to_numpy(np.float32) / p95_vec
    return df2

def fit_scaler_on_training_seqs(X_train_list):
    """
    Fit StandardScaler on ALL TRAINING TIMESTEPS after p95 scaling.
    X_train_list: list of arrays each (T_i, C)
    """
    if len(X_train_list) == 0:
        raise RuntimeError("No training sequences provided to fit scaler.")
    flat = np.concatenate(X_train_list, axis=0)  # (sum_T, C)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler

def transform_seqs_with_scaler(X_list, scaler):
    out = []
    for X in X_list:
        out.append(scaler.transform(X).astype(np.float32))
    return out

# ---------- Sequence extraction ----------
def df_to_sequence(df, feature_cols, label_map):
    """
    Extract full-sequence features + per-timestep labels + mask from a participant dataframe.
    Unknown is kept but masked out (y=-1, mask=0).
    """
    X = df[feature_cols].to_numpy(np.float32)  # (T, C)
    y_str = df[LABEL_COL].astype(str).to_numpy()
    y_int, mask = labels_to_int_and_mask(y_str, label_map)
    return X, y_int, mask

def save_object_array(path, seq_list):
    """
    Save a python list of numpy arrays as a numpy object array.
    Use allow_pickle=True when loading.
    """
    arr = np.empty((len(seq_list),), dtype=object)
    for i, x in enumerate(seq_list):
        arr[i] = x
    np.save(path, arr)

# ---------- Fold assembly ----------
def prepare_fold(data_root, out_root, held_out_pid):
    pids = list_pids(data_root)
    if held_out_pid not in pids:
        raise ValueError(f"held_out_pid {held_out_pid} not found. Available: {pids}")

    dfs = {pid: load_engineered(data_root, pid) for pid in pids}

    # sampling info (stored in meta; not required)
    fs_list = [estimate_fs(dfs[pid]) for pid in pids]
    fs_list = [f for f in fs_list if f > 0]
    fs_used = float(np.median(fs_list)) if len(fs_list) else None

    feature_cols = get_feature_cols(dfs[pids[0]])
    emg_env_cols = get_emg_env_cols(feature_cols)
    class_list = build_class_list(dfs)
    label_map = make_label_map(class_list)

    # split: LOPO + 1 val pid from training pool (same as your window script)
    train_pids = [p for p in pids if p != held_out_pid]
    val_idx = pids.index(held_out_pid) % len(train_pids)
    val_pid = train_pids[val_idx]
    print(f"[INFO] Participant {val_pid} being used for validation for held out participant split {held_out_pid}")

    val_pids = [val_pid]
    train_for_model = [p for p in train_pids if p != val_pid]
    test_pids = [held_out_pid]

    # --- EMG robust scaling computed BEFORE sequence extraction ---
    p95_by_pid = compute_p95_from_dfs(dfs, emg_env_cols, train_pids)

    train_pids_order = [pid for pid in train_pids if pid in p95_by_pid]
    if len(train_pids_order) == 0:
        raise RuntimeError("No training participants had any non-Unknown rows for EMG p95 computation.")

    p95_mat = np.stack([p95_by_pid[pid] for pid in train_pids_order], axis=0)
    p95_median = np.median(p95_mat, axis=0).astype(np.float32)

    def p95_for(pid):
        return p95_by_pid.get(pid, p95_median)

    dfs_p95 = {pid: apply_p95_to_df(dfs[pid], emg_env_cols, p95_for(pid)) for pid in pids}

    # --- Extract sequences (NO padding) ---
    def extract(pid_list):
        X_list, y_list, m_list, lens = [], [], [], []
        for pid in pid_list:
            X, y, m = df_to_sequence(dfs_p95[pid], feature_cols, label_map)
            X_list.append(X)
            y_list.append(y)
            m_list.append(m)
            lens.append(int(X.shape[0]))
        return X_list, y_list, m_list, np.array(lens, dtype=np.int32)

    X_train_list, y_train_list, m_train_list, lens_train = extract(train_for_model)
    X_val_list,   y_val_list,   m_val_list,   lens_val   = extract(val_pids)
    X_test_list,  y_test_list,  m_test_list,  lens_test  = extract(test_pids)

    # --- StandardScaler fit on training only (ALL training timesteps after p95) ---
    scaler = fit_scaler_on_training_seqs(X_train_list)
    X_train_list = transform_seqs_with_scaler(X_train_list, scaler)
    X_val_list   = transform_seqs_with_scaler(X_val_list, scaler)
    X_test_list  = transform_seqs_with_scaler(X_test_list, scaler)

    # --- Save fold artifacts ---
    fold_dir = os.path.join(out_root, f"fold_{held_out_pid}")
    os.makedirs(fold_dir, exist_ok=True)

    meta = {
        "held_out_pid": held_out_pid,
        "train_pids": train_for_model,
        "val_pids": val_pids,
        "test_pids": test_pids,
        "fs_used": fs_used,
        "feature_cols": feature_cols,
        "emg_env_cols": emg_env_cols,
        "class_list": class_list,
        "unknown_label": UNKNOWN_LABEL,
        "unknown_id": UNKNOWN_ID,
        "note": "Full-sequence folds saved as object arrays (variable length). Intended for batch_size=1 training.",
        "normalization": {
            "emg_env": "per-training-participant p95 computed from dfs excluding Unknown rows; held-out uses median(training p95)",
            "global": "StandardScaler fit on ALL training timesteps after p95 scaling (concatenated across training sequences)"
        },
        "mask_meaning": "mask==1 => supervised timestep; mask==0 => ignore (Unknown).",
        "counts": {
            "n_train_seqs": int(len(X_train_list)),
            "n_val_seqs": int(len(X_val_list)),
            "n_test_seqs": int(len(X_test_list)),
            "train_seq_lens": {"min": int(lens_train.min()) if len(lens_train) else 0,
                               "max": int(lens_train.max()) if len(lens_train) else 0,
                               "sum": int(lens_train.sum()) if len(lens_train) else 0},
        }
    }

    with open(os.path.join(fold_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # p95 artifacts
    np.save(os.path.join(fold_dir, "p95_train_by_pid.npy"),
            np.stack([p95_by_pid[pid] for pid in train_pids_order], axis=0))
    with open(os.path.join(fold_dir, "train_pids_order.json"), "w") as f:
        json.dump(train_pids_order, f, indent=2)
    np.save(os.path.join(fold_dir, "p95_median_train.npy"), p95_median)

    # scaler
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))

    # sequences (object arrays)
    save_object_array(os.path.join(fold_dir, "X_train.npy"), X_train_list)
    save_object_array(os.path.join(fold_dir, "y_train.npy"), y_train_list)
    save_object_array(os.path.join(fold_dir, "mask_train.npy"), m_train_list)
    np.save(os.path.join(fold_dir, "seq_lens_train.npy"), lens_train)

    save_object_array(os.path.join(fold_dir, "X_val.npy"), X_val_list)
    save_object_array(os.path.join(fold_dir, "y_val.npy"), y_val_list)
    save_object_array(os.path.join(fold_dir, "mask_val.npy"), m_val_list)
    np.save(os.path.join(fold_dir, "seq_lens_val.npy"), lens_val)

    save_object_array(os.path.join(fold_dir, "X_test.npy"), X_test_list)
    save_object_array(os.path.join(fold_dir, "y_test.npy"), y_test_list)
    save_object_array(os.path.join(fold_dir, "mask_test.npy"), m_test_list)
    np.save(os.path.join(fold_dir, "seq_lens_test.npy"), lens_test)

    print(f"[DONE] full-seq fold {held_out_pid}: "
          f"train_seqs={len(X_train_list)}, val_seqs={len(X_val_list)}, test_seqs={len(X_test_list)} saved to {fold_dir}")

def prepare_all_lopo(data_root, out_root, skip_existing=True):
    pids = list_pids(data_root)
    print(f"[INFO] Found {len(pids)} participants: {pids}")

    for held_out_pid in pids:
        fold_dir = os.path.join(out_root, f"fold_{held_out_pid}")
        if skip_existing and os.path.exists(os.path.join(fold_dir, "meta.json")):
            print(f"[SKIP] fold {held_out_pid} already exists at {fold_dir}")
            continue
        prepare_fold(data_root=data_root, out_root=out_root, held_out_pid=held_out_pid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", default="../../runs/full_sequence_folds")

    ap.add_argument("--held_out", default=None, help="Participant ID to hold out, e.g., P40")
    ap.add_argument("--all_lopo", action="store_true", help="Run LOPO over all participants")
    ap.add_argument("--skip_existing", action="store_true", help="Skip folds that already exist")

    args = ap.parse_args()

    if args.all_lopo:
        prepare_all_lopo(args.data_root, args.out_root, args.skip_existing)
    else:
        if args.held_out is None:
            raise SystemExit("Provide --held_out PXX or use --all_lopo")
        prepare_fold(args.data_root, args.out_root, args.held_out)

if __name__ == "__main__":
    main()