#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

TIME_COL = "Time (s)"
LABEL_COL = "Primitive"
PID_COL = "participant_id"

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
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt))

# ---------- Feature selection ----------
def get_feature_cols(df):
    # engineered columns only: everything numeric except Time/Primitive/participant_id
    excl = {TIME_COL, LABEL_COL, PID_COL}
    cols = [c for c in df.columns if c not in excl]
    cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    return sorted(cols)

def get_emg_env_cols(feature_cols):
    # your engineered naming: "*_EMG*_ENV"
    return [c for c in feature_cols if ("_EMG" in c and c.endswith("_ENV"))]

# ---------- Windowing ----------
def make_windows(df, feature_cols, win, stride, drop_label="Unknown"):
    X = df[feature_cols].to_numpy(np.float32)
    y = df[LABEL_COL].astype(str).to_numpy()

    X_list, y_list = [], []
    n = len(df)
    for start in range(0, n - win + 1, stride):
        center = start + win // 2
        lab = y[center]
        if lab == drop_label:
            continue
        X_list.append(X[start:start+win])
        y_list.append(lab)

    if not X_list:
        return np.zeros((0, win, len(feature_cols)), np.float32), np.array([], dtype=object)

    return np.stack(X_list), np.array(y_list, dtype=object)

# ---------- Normalization ----------
def compute_p95_per_pid(windows_by_pid, emg_idxs, pids, min_p95=1e-6):
    # returns dict pid -> p95 vector (len = n_emg_channels)
    out = {}
    for pid in pids:
        X = windows_by_pid[pid]
        if X.size == 0:
            continue
        emg_flat = X[:, :, emg_idxs].reshape(-1, len(emg_idxs))
        p95 = np.percentile(emg_flat, 95, axis=0)
        p95 = np.maximum(p95, min_p95)
        out[pid] = p95.astype(np.float32)
    return out

def apply_p95(X, emg_idxs, p95_vec):
    if X.size == 0:
        return X
    X2 = X.copy()
    for j, idx in enumerate(emg_idxs):
        X2[:, :, idx] = X2[:, :, idx] / p95_vec[j]
    return X2

def fit_scaler_on_training(X_train):
    # X_train: (N, W, C)
    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler

def transform_with_scaler(X, scaler):
    if X.size == 0:
        return X
    n, w, c = X.shape
    flat = X.reshape(-1, c)
    flat2 = scaler.transform(flat)
    return flat2.reshape(n, w, c).astype(np.float32)

# ---------- Labels ----------
def build_class_list(dfs):
    classes = set()
    for df in dfs.values():
        classes.update(df[LABEL_COL].astype(str).unique().tolist())
    classes.discard("Unknown")
    return sorted(classes)

def y_to_int(y, class_list):
    m = {c:i for i,c in enumerate(class_list)}
    return np.array([m[v] for v in y], dtype=np.int32)

# ---------- Fold assembly ----------
def prepare_fold(data_root, out_root, held_out_pid, win_s=1.0, stride_s=0.05):
    pids = list_pids(data_root)
    assert held_out_pid in pids

    # load all dfs into easy to access dictionary
    dfs = {pid: load_engineered(data_root, pid) for pid in pids}

    # determine fs and win/stride in units of # samples
    fs_used = float(np.median([estimate_fs(dfs[pid]) for pid in pids]))
    win = int(round(win_s * fs_used))
    stride = int(round(stride_s * fs_used))

    # feature cols based on first participant 
    feature_cols = get_feature_cols(dfs[pids[0]])
    emg_env_cols = get_emg_env_cols(feature_cols)
    emg_idxs = [feature_cols.index(c) for c in emg_env_cols]

    class_list = build_class_list(dfs)

    # Split: LOPO with a simple 1-participant val pulled from training set
    train_pids = [p for p in pids if p != held_out_pid]
    val_pid = train_pids[-1] if len(train_pids) > 1 else train_pids[0]
    train_for_model = [p for p in train_pids if p != val_pid]
    val_pids = [val_pid]
    test_pids = [held_out_pid]

    # window all participants 
    windows = {}
    labels = {}
    for pid in pids:
        Xw, yw = make_windows(dfs[pid], feature_cols, win, stride)
        windows[pid], labels[pid] = Xw, yw

    # --- EMG robust scaling ---
    # 1) p95 per training participant (for training participants only)
    p95_by_pid = compute_p95_per_pid(windows, emg_idxs, train_pids)

    # 2) median p95 across training participants (used for held-out)
    p95_mat = np.stack([p95_by_pid[pid] for pid in train_pids if pid in p95_by_pid], axis=0)
    p95_median = np.median(p95_mat, axis=0).astype(np.float32)

    # 3) apply p95: train participants use own; held-out uses median
    def p95_for(pid):
        return p95_by_pid.get(pid, p95_median)

    windows_p95 = {pid: apply_p95(windows[pid], emg_idxs, p95_for(pid)) for pid in pids}

    # --- Build split arrays ---
    def cat(pid_list):
        X = np.concatenate([windows_p95[p] for p in pid_list], axis=0)
        y = np.concatenate([labels[p] for p in pid_list], axis=0)
        return X, y

    X_train, y_train = cat(train_for_model)
    X_val,   y_val   = cat(val_pids)
    X_test,  y_test  = cat(test_pids)

    # --- StandardScaler on training only ---
    scaler = fit_scaler_on_training(X_train)
    X_train = transform_with_scaler(X_train, scaler)
    X_val   = transform_with_scaler(X_val, scaler)
    X_test  = transform_with_scaler(X_test, scaler)

    # Labels to ints
    y_train_i = y_to_int(y_train, class_list)
    y_val_i   = y_to_int(y_val, class_list)
    y_test_i  = y_to_int(y_test, class_list)

    # --- Save fold artifacts ---
    fold_dir = os.path.join(out_root, f"fold_{held_out_pid}")
    os.makedirs(fold_dir, exist_ok=True)

    meta = {
        "held_out_pid": held_out_pid,
        "train_pids": train_for_model,
        "val_pids": val_pids,
        "test_pids": test_pids,
        "fs_used": fs_used,
        "win_s": win_s, "stride_s": stride_s,
        "win_samples": win, "stride_samples": stride,
        "feature_cols": feature_cols,
        "emg_env_cols": emg_env_cols,
        "class_list": class_list,
        "normalization": {
            "emg_env": "per-training-participant p95; held-out uses median(training p95)",
            "global": "StandardScaler fit on training windows after p95 scaling"
        }
    }
    with open(os.path.join(fold_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save p95 artifacts
    train_pids_order = [pid for pid in train_pids if pid in p95_by_pid]
    np.save(os.path.join(fold_dir, "p95_train_by_pid.npy"),
            np.stack([p95_by_pid[pid] for pid in train_pids_order], axis=0))
    with open(os.path.join(fold_dir, "train_pids_order.json"), "w") as f:
        json.dump(train_pids_order, f, indent=2)

    np.save(os.path.join(fold_dir, "p95_median_train.npy"), p95_median)

    # Save scaler + arrays
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
    np.save(os.path.join(fold_dir, "X_train.npy"), X_train)
    np.save(os.path.join(fold_dir, "y_train.npy"), y_train_i)
    np.save(os.path.join(fold_dir, "X_val.npy"), X_val)
    np.save(os.path.join(fold_dir, "y_val.npy"), y_val_i)
    np.save(os.path.join(fold_dir, "X_test.npy"), X_test)
    np.save(os.path.join(fold_dir, "y_test.npy"), y_test_i)

    print(f"[DONE] fold {held_out_pid}: "
          f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape} saved to {fold_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", default="runs/prep")
    ap.add_argument("--held_out", required=True, help="Participant ID to hold out, e.g., P40")
    ap.add_argument("--win_s", type=float, default=1.0)
    ap.add_argument("--stride_s", type=float, default=0.05)
    args = ap.parse_args()

    prepare_fold(args.data_root, args.out_root, args.held_out, args.win_s, args.stride_s)

if __name__ == "__main__":
    main()
