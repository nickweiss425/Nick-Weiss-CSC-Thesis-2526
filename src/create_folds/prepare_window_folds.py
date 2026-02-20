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
    excl = {TIME_COL, LABEL_COL, PID_COL}
    cols = [c for c in df.columns if c not in excl]
    cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    return sorted(cols)

def get_emg_env_cols(feature_cols):
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
def compute_p95_from_dfs(dfs, emg_env_cols, pids, drop_label="Unknown", min_p95=1e-6):
    """
    Compute per-participant EMG envelope p95 directly from dfs[pid],
    excluding rows whose label is drop_label.
    Returns: dict pid -> p95 vector (len = n_emg_env_cols)
    """
    out = {}
    for pid in pids:
        df = dfs[pid]

        # exclude Unknown rows
        mask = df[LABEL_COL].astype(str).to_numpy() != drop_label
        if not np.any(mask):
            continue

        emg = df.loc[mask, emg_env_cols].to_numpy(np.float32)  # shape: (N_kept, n_emg)
        if emg.size == 0:
            continue

        p95 = np.percentile(emg, 95, axis=0)
        p95 = np.maximum(p95, min_p95).astype(np.float32)
        out[pid] = p95
    return out

def apply_p95_to_df(df, emg_env_cols, p95_vec):
    """
    Return a copy of df where EMG envelope columns are divided by p95_vec.
    """
    df2 = df.copy()
    # ensure broadcast works: (N, n_emg) / (n_emg,)
    df2.loc[:, emg_env_cols] = df2.loc[:, emg_env_cols].to_numpy(np.float32) / p95_vec
    return df2

def fit_scaler_on_training(X_train):
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

    # load all dfs
    dfs = {pid: load_engineered(data_root, pid) for pid in pids}

    # fs and win/stride in samples
    fs_used = float(np.median([estimate_fs(dfs[pid]) for pid in pids]))
    win = int(round(win_s * fs_used))
    stride = int(round(stride_s * fs_used))

    # features + EMG envelope columns
    feature_cols = get_feature_cols(dfs[pids[0]])
    emg_env_cols = get_emg_env_cols(feature_cols)
    class_list = build_class_list(dfs)

    # split: LOPO + 1 val pid from training pool
    train_pids = [p for p in pids if p != held_out_pid]
    # rotate val deterministically based on which fold we’re on
    val_idx = pids.index(held_out_pid) % len(train_pids)
    val_pid = train_pids[val_idx]
    print(f"[INFO] Participant {val_pid} being used for validation for held out participant split {held_out_pid}")
    val_pids = [val_pid]
    train_for_model = [p for p in train_pids if p != val_pid]
    test_pids = [held_out_pid]

    # --- EMG robust scaling computed before windowing ---
    # STEP 1: get p95 per training participant from dfs (excluding Unknown rows)
    p95_by_pid = compute_p95_from_dfs(dfs, emg_env_cols, train_pids, drop_label="Unknown")

    # STEP 2: get median p95 across training participants (used for held-out, and as fallback)
    train_pids_order = [pid for pid in train_pids if pid in p95_by_pid]
    if len(train_pids_order) == 0:
        raise RuntimeError("No training participants had any non-Unknown rows for EMG p95 computation.")

    p95_mat = np.stack([p95_by_pid[pid] for pid in train_pids_order], axis=0)
    p95_median = np.median(p95_mat, axis=0).astype(np.float32)

    # STEP 3: normalize dfs per participant (train/val use own when available; held-out uses median)
    def p95_for(pid):
        return p95_by_pid.get(pid, p95_median)

    dfs_p95 = {pid: apply_p95_to_df(dfs[pid], emg_env_cols, p95_for(pid)) for pid in pids}

    # --- Window all participants AFTER EMG normalization ---
    windows = {}
    labels = {}
    for pid in pids:
        Xw, yw = make_windows(dfs_p95[pid], feature_cols, win, stride, drop_label="Unknown")
        windows[pid], labels[pid] = Xw, yw

    # --- Build split arrays ---
    def cat(pid_list):
        X = np.concatenate([windows[p] for p in pid_list], axis=0)
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
            "emg_env": "per-training-participant p95 computed from dfs excluding Unknown rows; held-out uses median(training p95)",
            "global": "StandardScaler fit on training windows after p95 scaling"
        }
    }
    with open(os.path.join(fold_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # save p95 artifacts
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

def prepare_all_lopo(data_root, out_root, win_s=1.0, stride_s=0.05, skip_existing=True):
    pids = list_pids(data_root)
    print(f"[INFO] Found {len(pids)} participants: {pids}")

    for held_out_pid in pids:
        fold_dir = os.path.join(out_root, f"fold_{held_out_pid}")
        if skip_existing and os.path.exists(os.path.join(fold_dir, "meta.json")):
            print(f"[SKIP] fold {held_out_pid} already exists at {fold_dir}")
            continue

        prepare_fold(
            data_root=data_root,
            out_root=out_root,
            held_out_pid=held_out_pid,
            win_s=win_s,
            stride_s=stride_s,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", default="../../runs/window_folds/")

    # either specify one held-out pid or run all
    ap.add_argument("--held_out", default=None, help="Participant ID to hold out, e.g., P40")
    ap.add_argument("--all_lopo", action="store_true", help="Run LOPO over all participants")

    ap.add_argument("--win_s", type=float, default=1.0)
    ap.add_argument("--stride_s", type=float, default=0.05)
    ap.add_argument("--skip_existing", action="store_true", help="Skip folds that already exist")

    args = ap.parse_args()

    if args.all_lopo:
        prepare_all_lopo(args.data_root, args.out_root, args.win_s, args.stride_s, args.skip_existing)
    else:
        if args.held_out is None:
            raise SystemExit("Provide --held_out PXX or use --all_lopo")
        prepare_fold(args.data_root, args.out_root, args.held_out, args.win_s, args.stride_s)


if __name__ == "__main__":
    main()
