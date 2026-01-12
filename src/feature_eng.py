import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

def apply_feature_eng(trial_folder: str, sensors_used: list):
    df = pd.read_csv(trial_folder + "labeled.csv")
    df = emg_feature_eng(df, sensors_used)
    df = imu_feature_eng(df, sensors_used)
    df.to_csv(trial_folder + "finalized_data.csv", index=False)   


def emg_feature_eng(df: pd.DataFrame, sensors_used: list):
    # gather list of emg columns
    emg_columns = []
    for sensor_id in sensors_used:
        emg_columns.append(f'{sensor_id}_EMG1')
        emg_columns.append(f'{sensor_id}_EMG2')


    # get sampling rate from time differences
    dt = np.diff(df["Time (s)"].to_numpy())
    fs = float(np.round(1.0/np.median(dt[dt>0])))
    nyq = 0.5*fs

    # bandpass filter: keeps the core EMG frequency band where motor-unit signals live and rejects everything else
    def bp_sos(low, high, order=4):
        return butter(order, [low/nyq, high/nyq], btype='bandpass', output='sos')

    # high pass filter: removes slow drift and movement artifacts so only muscle activity remains
    def hp_sos(cut, order=2):
        return butter(order, cut/nyq, btype='highpass', output='sos')

    # narrow notch filter to remove power line interference
    def notch_once(x, f0, Q=35):
        if f0 >= nyq: return x
        w0 = f0/(fs/2); b, a = iirnotch(w0, Q)
        return filtfilt(b, a, x)

    # linear envelope: full-wave rectification followed by low-pass filter to smooth
    def linear_envelope(x, lp=5.0):
        rect = np.abs(x)
        sos_lp = butter(4, lp/nyq, btype='lowpass', output='sos')
        return sosfiltfilt(sos_lp, rect)

    # filters
    sos_hp = hp_sos(20.0, order=2)
    sos_bp = bp_sos(20.0, min(240.0, 0.45*fs), order=4)

    for col in emg_columns:
        x = df[col].astype(float).interpolate('linear').to_numpy()

        # high pass
        x = sosfiltfilt(sos_hp, x)

        # notch 60 & 120
        x = notch_once(x, 60); x = notch_once(x, 120)

        # bandpass
        x_bp = sosfiltfilt(sos_bp, x)

        # envelope
        env = linear_envelope(x_bp, lp=5.0)

        # save both normally filtered and enveloped signal
        df[col] = x_bp
        df[col + "_ENV"] = env
    
    return df

def imu_feature_eng(df: pd.DataFrame, sensors_used: list):

    fs_accel = estimate_fs_modality(df, [f'{sensors_used[0]}_AccelX',f'{sensors_used[0]}_AccelX',f'{sensors_used[0]}_AccelZ'])
    fs_gyro  = estimate_fs_modality(df, [f'{sensors_used[0]}_GyroX',f'{sensors_used[0]}_GyroY',f'{sensors_used[0]}_GyroZ'])
    fs_mag   = estimate_fs_modality(df, [f'{sensors_used[0]}_MagX',f'{sensors_used[0]}_MagY',f'{sensors_used[0]}_MagZ'])

    # gather list of accelerometer columns
    accel_cols = []
    for sensor_id in sensors_used:
        accel_cols.append(f'{sensor_id}_AccelX')
        accel_cols.append(f'{sensor_id}_AccelY')
        accel_cols.append(f'{sensor_id}_AccelZ')

    for col in accel_cols:
        df[col] = lowpass_sos(df[col], cutoff=20, fs=fs_accel)
        df[col + "_DYN"] = highpass_sos(df[col], cutoff=0.3, fs=50) # cut out gravity with high pass filter
    
    # gather list of accelerometer columns
    gyro_cols = []
    for sensor_id in sensors_used:
        gyro_cols.append(f'{sensor_id}_GyroX')
        gyro_cols.append(f'{sensor_id}_GyroY')
        gyro_cols.append(f'{sensor_id}_GyroZ')

    for col in gyro_cols:
        df[col] = lowpass_sos(df[col], cutoff=20, fs=fs_gyro)

    # gather list of accelerometer columns
    mag_cols = []
    for sensor_id in sensors_used:
        mag_cols.append(f'{sensor_id}_MagX')
        mag_cols.append(f'{sensor_id}_MagY')
        mag_cols.append(f'{sensor_id}_MagZ')

    for col in mag_cols:
        df[col] = lowpass_sos(df[col], cutoff=10, fs=fs_mag)

    import numpy as np

    for sensor_id in sensors_used:
        df[f"{sensor_id}_AccelMag"] = np.sqrt(df[f'{sensor_id}_AccelX']**2 + df[f'{sensor_id}_AccelY']**2 + df[f'{sensor_id}_AccelZ']**2)
        df[f"{sensor_id}_GyroMag"]  = np.sqrt(df[f'{sensor_id}_GyroX']**2  + df[f'{sensor_id}_GyroY']**2  + df[f'{sensor_id}_GyroZ']**2)
        df[f"{sensor_id}_MagnetMag"]   = np.sqrt(df[f'{sensor_id}_MagX']**2   + df[f'{sensor_id}_MagY']**2   + df[f'{sensor_id}_MagZ']**2)

    for col in accel_cols + gyro_cols + mag_cols:
        mu, sigma = df[col].mean(), df[col].std()
        df[col + "_Z"] = (df[col] - mu) / (sigma + 1e-8)

    for sensor_id in sensors_used:
        for col in [f"{sensor_id}_AccelMag"] + [f"{sensor_id}_GyroMag"] + [f"{sensor_id}_MagnetMag"]:
            mu, sigma = df[col].mean(), df[col].std()
            df[col + "_Z"] = (df[col] - mu) / (sigma + 1e-8)
    
    return df




# lowpass filter
def lowpass_sos(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, cutoff/nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data)

def highpass_sos(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    sos = butter(order, cutoff/nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, data)

def estimate_fs_modality(df, axes, time_col="Time (s)", eps=1e-6):
    """
    Estimate IMU sampling rate for a modality (e.g., ['AccelX','AccelY','AccelZ'])
    by detecting rows where ANY axis changes value. Returns a float fs_est (Hz).
    """
    t = df[time_col].to_numpy()
    X = df[axes].to_numpy(dtype=float)
    # mark where any axis changes between rows
    change = (np.abs(np.diff(X, axis=0)) > eps).any(axis=1)
    change_times = t[1:][change]
    if change_times.size < 2:
        return None
    dt = np.diff(change_times)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    fs_est = 1.0 / np.median(dt)  # robust estimate
    return float(fs_est)
