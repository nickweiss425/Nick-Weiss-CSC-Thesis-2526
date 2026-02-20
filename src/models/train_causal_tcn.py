# train_tcn_causal.py
"""
Causal TCN trainer (full-sequence ready, but works with windowed folds too).

This follows the structure of your CNN trainer:
- load_fold() reads X_train/y_train/X_val/y_val/X_test/y_test + meta.json
- supports single-fold and LOPO-all
- supports class weights (for sparse labels)
- writes best.keras, final.keras, results.json, summary.json

IMPORTANT:
- This script supports TWO label formats:
  (A) Window classification: y has shape (N,)  -> model output (N, K)
  (B) Dense per-timestep:   y has shape (N, T) -> model output (N, T, K)
  It auto-detects based on y.ndim.

For full-sequence training with variable trial lengths:
- easiest is batch_size=1 and store each trial as one example (N = num_trials)
- if you pad within a batch, you must also provide sample weights/mask;
  this script keeps it simple and assumes either fixed-length per batch or batch_size=1.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

# -----------------------
# Utils (same as your CNN)
# -----------------------
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
    """
    For sparse integer labels.

    If y is (N,), weights are computed directly.
    If y is (N,T), weights are computed over all timesteps.
    """
    if y.ndim == 2:
        y_flat = y.reshape(-1)
    else:
        y_flat = y
    counts = np.bincount(y_flat, minlength=num_classes).astype(np.float32)
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


# -----------------------
# TCN building blocks
# -----------------------
def tcn_residual_block(
    x,
    filters: int,
    kernel_size: int,
    dilation: int,
    dropout: float,
    name: str,
):
    """
    A standard TCN residual block:
      Conv1D (causal, dilated) -> BN -> ReLU -> Dropout
      Conv1D (causal, dilated) -> BN -> ReLU -> Dropout
      Residual connection (1x1 conv if needed)
    """
    in_channels = x.shape[-1]

    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",
        use_bias=False,
        name=f"{name}_conv2",
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = tf.keras.layers.Activation("relu", name=f"{name}_relu2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop2")(y)

    if in_channels != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters, kernel_size=1, padding="same", use_bias=False, name=f"{name}_proj"
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    else:
        shortcut = x

    out = tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])
    out = tf.keras.layers.Activation("relu", name=f"{name}_out_relu")(out)
    return out


def build_tcn(
    input_shape,
    num_classes: int,
    *,
    filters: int = 64,
    kernel_size: int = 3,
    n_blocks: int = 6,
    dropout: float = 0.2,
    dense_per_timestep: bool = True,
):
    """
    If dense_per_timestep=True:
        Input:  (T, C) -> Output: (T, K) softmax per timestep
    Else (window classification):
        Input:  (T, C) -> Output: (K,) softmax for the window
    """
    inp = tf.keras.Input(shape=input_shape, name="input")  # (T, C)

    x = inp
    # Exponential dilations: 1,2,4,8,...
    for i in range(n_blocks):
        d = 2 ** i
        x = tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=d,
            dropout=dropout,
            name=f"tcn_b{i+1}_d{d}",
        )

    if dense_per_timestep:
        # 1x1 conv to K classes at each timestep
        logits = tf.keras.layers.Conv1D(
            num_classes, kernel_size=1, padding="same", name="logits_1x1"
        )(x)  # (B, T, K)
        out = tf.keras.layers.Activation("softmax", name="softmax")(logits)
    else:
        # collapse time and classify window
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
        x = tf.keras.layers.Dropout(0.3, name="fc1_drop")(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = tf.keras.Model(inp, out, name="causal_tcn")
    return model


# -----------------------
# Train / Eval (same style)
# -----------------------
def train_one_fold(fold_dir: str, out_dir: str, args):
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, class_list, meta = load_fold(fold_dir)
    num_classes = len(class_list)

    # Auto-detect dense vs window labels
    # y.ndim==2 => dense per timestep (N,T)
    dense_labels = (y_train.ndim == 2)
    if args.force_dense:
        dense_labels = True
    if args.force_window:
        dense_labels = False

    train_ds = make_tf_dataset(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds   = make_tf_dataset(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_ds  = make_tf_dataset(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    model = build_tcn(
        input_shape=X_train.shape[1:],  # (T, C)
        num_classes=num_classes,
        filters=args.filters,
        kernel_size=args.kernel_size,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        dense_per_timestep=dense_labels,
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss,
        metrics=[metric],
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
    if args.use_class_weights and not dense_labels:
        # Keras class_weight works directly for (N,) sparse labels.
        # For dense (N,T) labels, you typically need sample_weight masking/weights instead.
        class_weight = compute_class_weights(y_train, num_classes)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # ----------------
    # Evaluate + metrics
    # ----------------
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    y_prob = model.predict(test_ds, verbose=0)

    if dense_labels:
        # y_prob: (N, T, K), y_test: (N, T)
        y_pred = np.argmax(y_prob, axis=-1)  # (N,T)

        # Flatten for confusion matrix / macro-F1
        y_true_flat = y_test.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        cm = confusion_matrix_np(y_true_flat, y_pred_flat, num_classes)
    else:
        # y_prob: (N, K), y_test: (N,)
        y_pred = np.argmax(y_prob, axis=1)
        cm = confusion_matrix_np(y_test, y_pred, num_classes)

    macro_f1, f1_per_class = macro_f1_from_cm(cm)

    results = {
        "fold_dir": fold_dir,
        "out_dir": out_dir,
        "held_out_pid": meta.get("held_out_pid", None),
        "classes": class_list,
        "dense_per_timestep": bool(dense_labels),
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "f1_per_class": {class_list[i]: float(f1_per_class[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "tcn_config": {
            "filters": int(args.filters),
            "kernel_size": int(args.kernel_size),
            "n_blocks": int(args.n_blocks),
            "dropout": float(args.dropout),
            "causal": True,
        },
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "use_class_weights": bool(args.use_class_weights),
            "seed": args.seed,
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
                "dense_per_timestep": r["dense_per_timestep"],
                "out_dir": r["out_dir"],
            } for r in all_results
        ],
    }
    return summary


# -----------------------
# Main (same as your CNN)
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    # --- Single-fold vs LOPO-all ---
    ap.add_argument("--fold_dir", type=str, default=None,
                    help="Path to a single fold dir, e.g., runs/prep/fold_P22")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir for a single-fold run, e.g., runs/models/tcn_causal_fold_P22")
    ap.add_argument("--all_folds", action="store_true",
                    help="Train/eval on every fold_* directory under --folds_root")
    ap.add_argument("--folds_root", type=str, default="runs/prep",
                    help="Directory containing fold_* folders")
    ap.add_argument("--out_root", type=str, default="runs/models/tcn_causal_lopo",
                    help="Root output directory for LOPO runs (one subdir per fold)")

    # --- Training hyperparams ---
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # --- TCN hyperparams ---
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)

    # --- Label mode overrides ---
    ap.add_argument("--force_dense", action="store_true",
                    help="Force dense per-timestep supervision (expects y shaped (N,T))")
    ap.add_argument("--force_window", action="store_true",
                    help="Force window classification (expects y shaped (N,))")

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
            out_dir = os.path.join(args.out_root, f"tcn_causal_fold_{held_out}")

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