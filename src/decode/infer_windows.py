import os, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

TIME_COL = "Time (s)"
LABEL_COL = "Primitive"

def make_windows(df, feature_cols, win, stride, drop_label="Unknown"):
    # make sure all columns are in floats of same type
    X = df[feature_cols].to_numpy(np.float32)

    # make sure time column is numeric float
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(np.float32)

    # get label column
    y = df[LABEL_COL].astype(str).to_numpy() if LABEL_COL in df.columns else None

    X_list, t_list, y_list = [], [], []
    n = len(df)
    # loop through dataset and make windows
    for start in range(0, n - win + 1, stride):
        center = start + win // 2
        center_label = y[center]
        if center_label == drop_label:
            continue

        # append label (center of window) to label list
        y_list.append(center_label)

        # append window of feature data to list
        X_list.append(X[start:start + win])

        # append center time to list
        t_list.append(float(t[center]))

    # return arrays of all lists
    return np.stack(X_list).astype(np.float32), np.array(t_list, np.float32), (np.array(y_list, dtype=object) if y is not None else None)

def y_to_int(y_str, class_list):
    m = {c:i for i, c in enumerate(class_list)}
    return np.array([m[v] for v in y_str], dtype=np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True, help="Held-out participant ID, e.g., P40")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--folds_root", default=os.path.join("runs", "prep"))
    ap.add_argument("--models_root", default=os.path.join("runs", "models", "cnn_lopo"))
    ap.add_argument("--out_dir", default=os.path.join("runs", "decoded"))
    ap.add_argument("--drop_label", default="Unknown")
    args = ap.parse_args()

    # participant id to perform inference on
    pid = args.pid

    # load participant data
    engineered_csv = os.path.join(args.data_root, pid, "engineered.csv")
    df = pd.read_csv(engineered_csv)

    # get paths for fold information and model
    fold_dir = os.path.join(args.folds_root, f"fold_{pid}")
    model_path = os.path.join(args.models_root, f"cnn_fold_{pid}", "best.keras")

    # open up json file that contains information about window sizes, columns, etc.
    with open(os.path.join(fold_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    # extract feature columns, emg columns, class list, window information 
    feature_cols = meta["feature_cols"]
    emg_env_cols = meta.get("emg_env_cols", [])
    class_list = meta["class_list"]
    win = int(meta["win_samples"])
    stride = int(meta["stride_samples"])


    # p95 scaling (median-from-training for this fold)
    if emg_env_cols:
        # load median p95 value for each emg envelope channel
        p95 = np.load(os.path.join(fold_dir, "p95_median_train.npy")).astype(np.float32)

        # scale each emg envelope channel by median value across training participants
        df.loc[:, emg_env_cols] = df[emg_env_cols].to_numpy(np.float32) / p95

    # make windows
    Xw_raw, t_center, y_center_str = make_windows(df, feature_cols, win, stride, drop_label=args.drop_label)

    # StandardScaler from used fold
    scaler = joblib.load(os.path.join(fold_dir, "scaler.joblib"))
    n, w, c = Xw_raw.shape
    Xw = scaler.transform(Xw_raw.reshape(-1, c)).reshape(n, w, c).astype(np.float32)

    # inference
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(Xw, verbose=0).astype(np.float32)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    y_true = np.array([], dtype=np.int32)
    if y_center_str is not None:
        y_true = y_to_int(y_center_str, class_list)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"{pid}_window_probs_raw.npz")

    # output file will contain: ['t_center', 'probs', 'y_pred', 'y_true', 'class_list', 'win', 'stride']
    np.savez(
        out_npz,
        t_center=t_center,
        probs=probs,
        y_pred=y_pred,
        y_true=y_true,
        class_list=np.array(class_list, dtype=object),
        win=win,
        stride=stride
    )

    print(f"[SAVED] {out_npz}")
    print(f"  windows: {len(t_center)} | probs: {probs.shape} | has_y_true: {len(y_true) > 0}")

if __name__ == "__main__":
    main()
