import os
import json
import argparse
import numpy as np
import tensorflow as tf

# ---------- Utils ----------
def load_fold(fold_dir: str):
    X_train = np.load(os.path.join(fold_dir, "X_train.npy"))
    y_train = np.load(os.path.join(fold_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(fold_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(fold_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(fold_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(fold_dir, "y_test.npy"))

    with open(os.path.join(fold_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    class_list = meta["class_list"]
    return (X_train, y_train, X_val, y_val, X_test, y_test, class_list, meta)

def list_fold_dirs(folds_root: str):
    fold_dirs = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            p = os.path.join(folds_root, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
                fold_dirs.append(p)
    return fold_dirs

def make_tf_dataset(X, y, batch_size=256, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(y), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def compute_class_weights(y: np.ndarray, num_classes: int):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    counts = np.maximum(counts, 1.0)
    weights = total / (num_classes * counts)
    return {i: float(weights[i]) for i in range(num_classes)}

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def macro_f1_from_cm(cm: np.ndarray):
    num_classes = cm.shape[0]
    f1s = []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), [float(x) for x in f1s]

# ---------- Model ----------
def build_hybrid_cnn(input_shape, num_classes,
                     depth_kernel=7,
                     depth_multiplier=1,
                     pointwise_filters=64):
    """
    Hybrid front-end:
      1) Depthwise temporal conv (per-channel) using groups=input_channels
      2) Pointwise conv (kernel_size=1) to mix channels
    Then continues with a standard CNN stack similar to your baseline.

    input_shape: (T, C)
    """
    inp = tf.keras.Input(shape=input_shape)  # (T, C)
    C = int(input_shape[-1])

    # ---- Depthwise Conv1D (per-channel temporal filtering) ----
    # Conv1D supports "groups" in modern TF; groups=C means each input channel is convolved independently.
    # Output channels must be divisible by groups. We set filters = C * depth_multiplier.
    depth_filters = C * int(depth_multiplier)
    try:
        x = tf.keras.layers.Conv1D(
            filters=depth_filters,
            kernel_size=int(depth_kernel),
            padding="same",
            groups=C,          # <- key: no channel mixing here
            use_bias=False
        )(inp)
    except TypeError as e:
        raise RuntimeError(
            "Your TensorFlow/Keras version may not support Conv1D(groups=...). "
            "Upgrade TensorFlow (2.9+ recommended) or tell me your TF version "
            "and I’ll give you a compatible alternative."
        ) from e

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # ---- Pointwise Conv1D (1x1) to mix channels ----
    # This learns combinations across channels at each time step.
    x = tf.keras.layers.Conv1D(
        filters=int(pointwise_filters),
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Now we’re back to a standard feature map representation over time.
    # Shape here: (T, pointwise_filters)

    # ---- Rest of your CNN (same spirit) ----
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv1D(256, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    return model

def train_one_fold(fold_dir: str, out_dir: str, args):
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, class_list, meta = load_fold(fold_dir)
    num_classes = len(class_list)

    train_ds = make_tf_dataset(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds   = make_tf_dataset(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_ds  = make_tf_dataset(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    model = build_hybrid_cnn(
        input_shape=X_train.shape[1:],
        num_classes=num_classes,
        depth_kernel=args.depth_kernel,
        depth_multiplier=args.depth_multiplier,
        pointwise_filters=args.pointwise_filters,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "best.keras"),
            monitor="val_acc",
            save_best_only=True,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=7,
            mode="max",
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "training_log.csv"))
    ]

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(y_train, num_classes)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate + metrics
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix_np(y_test, y_pred, num_classes)
    macro_f1, f1_per_class = macro_f1_from_cm(cm)

    results = {
        "fold_dir": fold_dir,
        "out_dir": out_dir,
        "held_out_pid": meta.get("held_out_pid", None),
        "classes": class_list,
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "f1_per_class": {class_list[i]: float(f1_per_class[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "config": {
            "model": "hybrid_depthwise_pointwise_cnn",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "use_class_weights": bool(args.use_class_weights),
            "seed": args.seed,
            "depth_kernel": args.depth_kernel,
            "depth_multiplier": args.depth_multiplier,
            "pointwise_filters": args.pointwise_filters,
        }
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(out_dir, "final.keras"))
    return results

def aggregate_results(all_results):
    accs = np.array([r["test_acc"] for r in all_results], dtype=np.float32)
    f1s  = np.array([r["test_macro_f1"] for r in all_results], dtype=np.float32)

    cms = [np.array(r["confusion_matrix"], dtype=np.int64) for r in all_results]
    cm_sum = np.sum(cms, axis=0)

    summary = {
        "n_folds": int(len(all_results)),
        "test_acc_mean": float(accs.mean()),
        "test_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "test_macro_f1_mean": float(f1s.mean()),
        "test_macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "confusion_matrix_sum": cm_sum.tolist(),
        "per_fold": [
            {
                "held_out_pid": r.get("held_out_pid"),
                "test_acc": r["test_acc"],
                "test_macro_f1": r["test_macro_f1"],
                "out_dir": r["out_dir"],
            } for r in all_results
        ],
    }
    return summary

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()

    # --- Single-fold vs LOPO-all ---
    ap.add_argument("--fold_dir", type=str, default=None,
                    help="Path to a single fold dir, e.g., runs/prep/fold_P22")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir for a single-fold run, e.g., runs/models/hybrid_cnn_fold_P22")
    ap.add_argument("--all_folds", action="store_true",
                    help="Train/eval on every fold_* directory under --folds_root")
    ap.add_argument("--folds_root", type=str, default="runs/prep",
                    help="Directory containing fold_* folders")
    ap.add_argument("--out_root", type=str, default="runs/models/hybrid_cnn_lopo",
                    help="Root output directory for LOPO runs (one subdir per fold)")

    # --- Training hyperparams ---
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # --- Hybrid front-end hyperparams ---
    ap.add_argument("--depth_kernel", type=int, default=7,
                    help="Kernel size for depthwise temporal filtering.")
    ap.add_argument("--depth_multiplier", type=int, default=1,
                    help="Depthwise channel multiplier (output channels = C * multiplier).")
    ap.add_argument("--pointwise_filters", type=int, default=64,
                    help="Number of 1x1 conv filters after depthwise conv (controls mixing capacity).")

    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------
    # LOPO: train on all folds
    # ---------------------------
    if args.all_folds:
        fold_dirs = list_fold_dirs(args.folds_root)
        if len(fold_dirs) == 0:
            raise SystemExit(f"No fold_* directories found in: {args.folds_root}")

        os.makedirs(args.out_root, exist_ok=True)

        all_results = []
        for fold_dir in fold_dirs:
            held_out = os.path.basename(fold_dir).replace("fold_", "")
            out_dir = os.path.join(args.out_root, f"hybrid_cnn_fold_{held_out}")

            print(f"\n[RUN] {fold_dir} -> {out_dir}")
            r = train_one_fold(fold_dir, out_dir, args)
            all_results.append(r)

            tf.keras.backend.clear_session()

        summary = aggregate_results(all_results)
        with open(os.path.join(args.out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n[LOPO SUMMARY]")
        print(f"  folds: {summary['n_folds']}")
        print(f"  acc:   {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
        print(f"  f1:    {summary['test_macro_f1_mean']:.4f} ± {summary['test_macro_f1_std']:.4f}")
        print(f"\n[SAVED] {os.path.join(args.out_root, 'summary.json')}")
        return

    # ---------------------------
    # Single-fold: train once
    # ---------------------------
    if args.fold_dir is None:
        raise SystemExit("Provide --fold_dir for a single-fold run, or use --all_folds.")
    if args.out_dir is None:
        raise SystemExit("Provide --out_dir for a single-fold run.")

    train_one_fold(args.fold_dir, args.out_dir, args)

if __name__ == "__main__":
    main()
