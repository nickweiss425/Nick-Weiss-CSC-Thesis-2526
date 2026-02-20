#!/usr/bin/env python3
"""
plot_normalization_hist.py  (fixed)

Recreates the LOPO fold like your prepare_fold then produces histogram checks:
 - Raw windows (no p95, no scaler)
 - After p95 normalization (per-training p95, median fallback for held-out)
 - After StandardScaler (scaler fit on training windows after p95)

Usage:
  python plot_normalization_hist.py --data_root ./data --held_out P32 --out_png hist_P32.png
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------- Config constants (match your pipeline) ----------
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
def make_windows_from_df(df, feature_cols, win, stride, drop_label="Unknown"):
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

# ---------- p95 scaling ----------
def compute_p95_from_dfs(dfs, emg_env_cols, pids, drop_label="Unknown", min_p95=1e-6):
    out = {}
    for pid in pids:
        df = dfs[pid]
        mask = df[LABEL_COL].astype(str).to_numpy() != drop_label
        if not np.any(mask):
            continue
        emg = df.loc[mask, emg_env_cols].to_numpy(np.float32)
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

# ---------- scaler helpers ----------
def fit_scaler_on_windows(X_train):
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

# ---------- sampling / plotting helpers ----------
def sample_channel_values(X_windows, channel_idx, max_samples=50000, random_state=0):
    """
    X_windows: ndarray with shape (n_windows, win, n_channels)
    Returns a 1-D array of shape (min(total_values, max_samples),) of scalar values
    sampled from the channel flattened across windows and time.
    """
    if X_windows.size == 0:
        return np.array([], dtype=float)
    # flatten across windows and time -> shape (n_windows*win,)
    flat = X_windows[..., channel_idx].reshape(-1)
    n = flat.shape[0]
    rng = np.random.RandomState(random_state)
    if n <= max_samples:
        return flat.astype(float)
    idx = rng.choice(n, size=max_samples, replace=False)
    return flat[idx].astype(float)

def plot_three_stage_overlay(cname, raw_train, raw_val, raw_test,
                              p95_train, p95_val, p95_test,
                              sc_train, sc_val, sc_test,
                              out_png_base=None, bins=120):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    axes[0].hist([raw_train, raw_val, raw_test], bins=bins, label=['train','val','test'], alpha=0.6, density=True)
    axes[0].set_title(f"{cname} — raw")
    axes[1].hist([p95_train, p95_val, p95_test], bins=bins, label=['train','val','test'], alpha=0.6, density=True)
    axes[1].set_title(f"{cname} — after p95")
    axes[2].hist([sc_train, sc_val, sc_test], bins=bins, label=['train','val','test'], alpha=0.6, density=True)
    axes[2].set_title(f"{cname} — after StandardScaler")
    for ax in axes:
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    if out_png_base:
        safe_name = cname.replace(" ", "_").replace("/", "_")
        outname = f"{out_png_base}__{safe_name}.png"
        plt.savefig(outname, dpi=200)
        print(f"[INFO] Saved {outname}")
        plt.close(fig)
    else:
        plt.show()

# ---------- main logic ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path containing participant folders with engineered.csv")
    ap.add_argument("--held_out", required=True, help="Held-out participant ID, e.g., P32")
    ap.add_argument("--win_s", type=float, default=1.0)
    ap.add_argument("--stride_s", type=float, default=0.05)
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--channels", default=None,
                    help="Comma-separated channel names to plot (exact feature column names). If omitted, picks defaults.")
    ap.add_argument("--channel_indices", default=None,
                    help="Comma-separated channel indices (0-based) corresponding to feature_cols order.")
    ap.add_argument("--out_png", default=None, help="If provided, saves PNG(s) with this base filename.")
    args = ap.parse_args()

    data_root = args.data_root
    held_out_pid = args.held_out

    pids = list_pids(data_root)
    if held_out_pid not in pids:
        raise SystemExit(f"Held-out pid {held_out_pid} not found in data_root. Found: {pids}")

    # load all dfs
    dfs = {pid: load_engineered(data_root, pid) for pid in pids}

    # sampling rate and window/stride in samples (median across pids)
    fs_used = float(np.median([estimate_fs(dfs[pid]) for pid in pids]))
    win = int(round(args.win_s * fs_used))
    stride = int(round(args.stride_s * fs_used))
    print(f"[INFO] fs_used={fs_used:.2f}Hz win={win} stride={stride}")

    feature_cols = get_feature_cols(dfs[pids[0]])
    emg_env_cols = get_emg_env_cols(feature_cols)
    print(f"[INFO] Detected {len(feature_cols)} feature cols, {len(emg_env_cols)} EMG_ENV cols.")

    # build class list like prepare_fold
    classes = set()
    for df in dfs.values():
        classes.update(df[LABEL_COL].astype(str).unique().tolist())
    if "Unknown" in classes:
        classes.remove("Unknown")
    class_list = sorted(classes)

    # LOPO split & val rotation (same logic as your script)
    train_pids = [p for p in pids if p != held_out_pid]
    val_idx = pids.index(held_out_pid) % len(train_pids)
    val_pid = train_pids[val_idx]
    val_pids = [val_pid]
    train_for_model = [p for p in train_pids if p != val_pid]
    test_pids = [held_out_pid]
    print(f"[INFO] Using val pid {val_pid}; train_pids (for model)={train_for_model}; test_pids={test_pids}")

    # --- compute p95 per training pid and median fallback (before windowing) ---
    p95_by_pid = compute_p95_from_dfs(dfs, emg_env_cols, train_for_model, drop_label="Unknown")
    train_pids_order = [pid for pid in train_for_model if pid in p95_by_pid]
    if len(train_pids_order) == 0:
        raise RuntimeError("No training participants had EMG p95 info to compute.")
    p95_mat = np.stack([p95_by_pid[pid] for pid in train_pids_order], axis=0)
    p95_median = np.median(p95_mat, axis=0).astype(np.float32)

    def p95_for(pid):
        return p95_by_pid.get(pid, p95_median)

    # --- Build three sets of data: raw windows (no p95), p95-normalized windows, and scaled windows ---
    windows_raw = {}
    windows_p95 = {}
    labels = {}
    for pid in pids:
        df = dfs[pid]
        # raw windows (no p95)
        Xw_raw, yw = make_windows_from_df(df, feature_cols, win, stride, drop_label="Unknown")
        windows_raw[pid] = Xw_raw
        # p95-normalized df then windows
        p95_vec = p95_for(pid)
        df_p95 = apply_p95_to_df(df, emg_env_cols, p95_vec)
        Xw_p95, _ = make_windows_from_df(df_p95, feature_cols, win, stride, drop_label="Unknown")
        windows_p95[pid] = Xw_p95
        labels[pid] = yw

    # concatenate splits like prepare_fold
    def cat(pid_list, mapping):
        if len(pid_list) == 0:
            return np.zeros((0, win, len(feature_cols)), np.float32)
        arrs = [mapping[p] for p in pid_list]
        if any(a.size == 0 for a in arrs):
            arrs = [a for a in arrs if a.size != 0]
            if len(arrs) == 0:
                return np.zeros((0, win, len(feature_cols)), np.float32)
        return np.concatenate(arrs, axis=0)

    X_train_raw = cat(train_for_model, windows_raw)
    X_val_raw   = cat(val_pids, windows_raw)
    X_test_raw  = cat(test_pids, windows_raw)

    X_train_p95 = cat(train_for_model, windows_p95)
    X_val_p95   = cat(val_pids, windows_p95)
    X_test_p95  = cat(test_pids, windows_p95)

    # fit scaler on training windows AFTER p95 (same as prepare_fold)
    if X_train_p95.size == 0:
        raise RuntimeError("No training windows after p95 to fit scaler.")
    scaler = fit_scaler_on_windows(X_train_p95)

    X_train_scaled = transform_with_scaler(X_train_p95, scaler)
    X_val_scaled   = transform_with_scaler(X_val_p95, scaler)
    X_test_scaled  = transform_with_scaler(X_test_p95, scaler)

    # Choose channels to plot
    if args.channels:
        ch_names = [s.strip() for s in args.channels.split(",")]
        channel_indices = [feature_cols.index(n) for n in ch_names]
    elif args.channel_indices:
        channel_indices = [int(s.strip()) for s in args.channel_indices.split(",")]
        ch_names = [feature_cols[i] for i in channel_indices]
    else:
        accel = next((c for c in feature_cols if ("ACC" in c.upper() or "ACCEL" in c.upper() or "acc" in c.lower())), None)
        gyro  = next((c for c in feature_cols if ("GYRO" in c.upper() or "GYR" in c.upper())), None)
        emg   = emg_env_cols[0] if len(emg_env_cols) > 0 else None
        ch_names = []
        for c in (accel, gyro, emg):
            if c and c not in ch_names:
                ch_names.append(c)
        if len(ch_names) == 0:
            ch_names = feature_cols[:3]
        channel_indices = [feature_cols.index(c) for c in ch_names]

    print(f"[INFO] Plotting channels: {ch_names}  (indices {channel_indices})")

    # For each chosen channel, sample up to max_samples and plot raw->p95->scaled
    for ci, cname in zip(channel_indices, ch_names):
        raw_train = sample_channel_values(X_train_raw, ci, max_samples=args.max_samples)
        raw_val   = sample_channel_values(X_val_raw, ci, max_samples=args.max_samples)
        raw_test  = sample_channel_values(X_test_raw, ci, max_samples=args.max_samples)

        p95_train = sample_channel_values(X_train_p95, ci, max_samples=args.max_samples)
        p95_val   = sample_channel_values(X_val_p95, ci, max_samples=args.max_samples)
        p95_test  = sample_channel_values(X_test_p95, ci, max_samples=args.max_samples)

        sc_train = sample_channel_values(X_train_scaled, ci, max_samples=args.max_samples)
        sc_val   = sample_channel_values(X_val_scaled, ci, max_samples=args.max_samples)
        sc_test  = sample_channel_values(X_test_scaled, ci, max_samples=args.max_samples)

        # Plot overlayed train/val/test for each stage
        plot_three_stage_overlay(
            cname,
            raw_train, raw_val, raw_test,
            p95_train, p95_val, p95_test,
            sc_train, sc_val, sc_test,
            out_png_base=args.out_png,
            bins=120
        )

    print("[DONE]")

if __name__ == "__main__":
    main()
