import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch


# =====================
# Utility functions
# =====================

def estimate_fs_from_time(df, time_col="Time (s)"):
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    t = t[~np.isnan(t)]
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Cannot estimate sampling rate.")
    return float(1.0 / np.median(dt))


def dropout_report(df, time_col="Time (s)"):
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    t = t[~np.isnan(t)]
    dt = np.diff(t)
    dt = dt[dt > 0]
    med = np.median(dt)
    return {
        "median_dt": med,
        "p95_dt": np.percentile(dt, 95),
        "gap_frac_gt2x": np.mean(dt > 2 * med),
    }


def interpolate_small_gaps(series, limit=10):
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = x.interpolate("linear", limit=limit, limit_direction="both")
    x = x.ffill().bfill()
    return x.to_numpy()


def lowpass(x, fs, cutoff, order=4):
    nyq = 0.5 * fs
    if cutoff >= 0.99 * nyq:
        return x
    sos = butter(order, cutoff / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)


def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    high = min(high, 0.99 * nyq)
    if low >= high:
        return x
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def notch(x, fs, freq, q=35):
    nyq = 0.5 * fs
    if freq >= nyq:
        return x
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, x)


# =====================
# Feature engineering
# =====================

def add_emg_features(df, sensor_ids, fs, interp_limit=10):
    out = df.copy()
    nyq = 0.5 * fs
    bp_high = 0.95 * nyq  # Nyquist-safe

    for sid in sensor_ids:
        for ch in ["EMG1", "EMG2"]:
            col = f"{sid}_{ch}"
            if col not in out.columns:
                continue

            x = interpolate_small_gaps(out[col], interp_limit)

            # notch powerline
            x = notch(x, fs, 60)
            x = notch(x, fs, 120)

            # bandpass EMG
            x_bp = bandpass(x, fs, low=20.0, high=bp_high, order=4)

            # envelope
            env = lowpass(np.abs(x_bp), fs, cutoff=5.0, order=4)

            out[f"{col}_FILTERED"] = x_bp.astype(np.float32)
            out[f"{col}_ENV"] = env.astype(np.float32)

    return out


def add_accel_features(df, sensor_ids, fs, interp_limit=10):
    out = df.copy()

    for sid in sensor_ids:
        ax, ay, az = f"{sid}_AccelX", f"{sid}_AccelY", f"{sid}_AccelZ"
        if not all(c in out.columns for c in [ax, ay, az]):
            continue

        for col in [ax, ay, az]:
            x = interpolate_small_gaps(out[col], interp_limit)
            x_dn = lowpass(x, fs, 20.0, 4)
            x_grav = lowpass(x_dn, fs, 0.5, 2)
            x_dyn = x_dn - x_grav

            out[f"{col}_DENOISED"] = x_dn.astype(np.float32)
            out[f"{col}_DYN"] = x_dyn.astype(np.float32)

        out[f"{sid}_AccelMag_DYN"] = np.sqrt(
            out[f"{ax}_DYN"]**2 +
            out[f"{ay}_DYN"]**2 +
            out[f"{az}_DYN"]**2
        ).astype(np.float32)

    return out


def add_gyro_features(df, sensor_ids, fs, interp_limit=10):
    out = df.copy()

    for sid in sensor_ids:
        gx, gy, gz = f"{sid}_GyroX", f"{sid}_GyroY", f"{sid}_GyroZ"
        if not all(c in out.columns for c in [gx, gy, gz]):
            continue

        for col in [gx, gy, gz]:
            x = interpolate_small_gaps(out[col], interp_limit)
            x_dn = lowpass(x, fs, 20.0, 4)
            x_bias = lowpass(x_dn, fs, 0.2, 2)
            x_dyn = x_dn - x_bias

            out[f"{col}_DENOISED"] = x_dn.astype(np.float32)
            out[f"{col}_DYN"] = x_dyn.astype(np.float32)

        out[f"{sid}_GyroMag_DYN"] = np.sqrt(
            out[f"{gx}_DYN"]**2 +
            out[f"{gy}_DYN"]**2 +
            out[f"{gz}_DYN"]**2
        ).astype(np.float32)

    return out


# =====================
# Main
# =====================

def main(data_root):
    sensor_ids = ["A5F2", "A19E"] 

    for pid in sorted(os.listdir(data_root)):
        pdir = os.path.join(data_root, pid)
        if not os.path.isdir(pdir):
            continue

        in_csv = os.path.join(pdir, "labeled.csv")
        out_csv = os.path.join(pdir, "engineered.csv")

        if not os.path.exists(in_csv):
            print(f"[SKIP] {pid}: labeled.csv not found")
            continue

        df = pd.read_csv(in_csv)

        if "Event_Marker" in df.columns:
            df = df.drop(columns=["Event_Marker"])

        fs = estimate_fs_from_time(df)
        rep = dropout_report(df)

        print(
            f"[INFO] {pid}: fs≈{fs:.2f} Hz | "
            f"median_dt={rep['median_dt']:.6f}s | "
            f"gap_frac>2x={rep['gap_frac_gt2x']:.2%}"
        )

        df["participant_id"] = pid

        df = add_emg_features(df, sensor_ids, fs)
        df = add_accel_features(df, sensor_ids, fs)
        df = add_gyro_features(df, sensor_ids, fs)

        # keep only engineered + metadata
        keep_cols = ["Time (s)", "Primitive", "participant_id"]
        keep_cols += [c for c in df.columns if any(
            c.endswith(suf) for suf in (
                "_FILTERED", "_ENV",
                "_DENOISED", "_DYN",
                "_AccelMag_DYN", "_GyroMag_DYN"
            )
        )]

        df[keep_cols].to_csv(out_csv, index=False)
        print(f"[DONE] {pid}: wrote engineered.csv ({len(keep_cols)} cols)")

    print("All participants processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    main(args.data_root)
