# train_noncausal_tcn.py
"""
Non-causal (bidirectional context) Dense TCN trainer for FULL-SEQUENCE folds saved as NumPy object arrays
(from prepare_fullseq_folds.py), intended for batch_size=1.

Difference vs causal:
- Uses padding="same" in all Conv1D layers (so prediction at time t can use past + future context).
- Everything else (masking, batch_size=1 object arrays, LOPO structure, metrics) matches the causal script.

Expected fold_dir contents:
  X_train.npy, y_train.npy, mask_train.npy
  X_val.npy,   y_val.npy,   mask_val.npy
  X_test.npy,  y_test.npy,  mask_test.npy
  meta.json (with class_list)

Each X_*.npy is an object array of length N_seqs:
  X[i]: (T_i, C) float32
  y[i]: (T_i,) int32 with Unknown = -1
  mask[i]: (T_i,) float32 with supervised=1, Unknown=0

Training uses sample_weight=mask so Unknown timesteps do not contribute to loss/metrics.
Because SparseCategoricalCrossentropy cannot accept -1 labels, we replace -1 with 0
ONLY for computation; those positions are weight=0 so they are ignored.

Outputs per fold:
  best.keras, final.keras, training_log.csv, results.json
If --all_folds:
  summary.json in out_root
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf


# -----------------------
# IO / fold loading
# -----------------------
def load_fullseq_fold(fold_dir: str):
    def load_obj(name):
        return np.load(os.path.join(fold_dir, name), allow_pickle=True)

    X_train = load_obj("X_train.npy")
    y_train = load_obj("y_train.npy")
    m_train = load_obj("mask_train.npy")

    X_val = load_obj("X_val.npy")
    y_val = load_obj("y_val.npy")
    m_val = load_obj("mask_val.npy")

    X_test = load_obj("X_test.npy")
    y_test = load_obj("y_test.npy")
    m_test = load_obj("mask_test.npy")

    with open(os.path.join(fold_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    class_list = meta["class_list"]
    return (X_train, y_train, m_train, X_val, y_val, m_val, X_test, y_test, m_test, class_list, meta)


def list_fold_dirs(folds_root: str):
    fold_dirs = []
    for name in sorted(os.listdir(folds_root)):
        if name.startswith("fold_"):
            p = os.path.join(folds_root, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
                fold_dirs.append(p)
    return fold_dirs


# -----------------------
# Metrics helpers
# -----------------------
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


def masked_time_weighted_accuracy(y_true_list, y_pred_list, mask_list):
    correct = 0.0
    total = 0.0
    for yt, yp, m in zip(y_true_list, y_pred_list, mask_list):
        keep = (m > 0.5)
        if not np.any(keep):
            continue
        correct += float(np.sum((yt[keep] == yp[keep]).astype(np.float32)))
        total += float(np.sum(keep.astype(np.float32)))
    return float(correct / total) if total > 0 else 0.0


# -----------------------
# tf.data for batch_size=1 full sequences (infinite)
# -----------------------
def make_fullseq_dataset(X_obj, y_obj, m_obj, shuffle: bool, seed: int):
    """
    Infinite dataset yielding (X, y_fixed, sample_weight) with shapes:
      X: (T_i, C)
      y_fixed: (T_i,)
      sample_weight: (T_i,)

    Because this generator is infinite, you MUST pass steps_per_epoch
    (and validation_steps / test_steps) to fit/evaluate.
    """
    n = len(X_obj)
    indices = np.arange(n, dtype=np.int32)
    rng = np.random.RandomState(seed)

    def gen():
        while True:
            idxs = indices.copy()
            if shuffle:
                rng.shuffle(idxs)

            for i in idxs:
                X = X_obj[i].astype(np.float32)
                y = y_obj[i].astype(np.int32)
                m = m_obj[i].astype(np.float32)

                if X.ndim != 2:
                    raise ValueError(f"X[{i}] expected shape (T,C), got {X.shape}")
                if y.ndim != 1 or m.ndim != 1:
                    raise ValueError(f"y/m[{i}] expected shape (T,), got y={y.shape}, m={m.shape}")
                if X.shape[0] != y.shape[0] or y.shape[0] != m.shape[0]:
                    raise ValueError(f"Length mismatch at i={i}: X={X.shape}, y={y.shape}, m={m.shape}")

                y_fixed = np.where(y < 0, 0, y).astype(np.int32)
                yield (X, y_fixed, m)

    output_signature = (
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # (T, C)
        tf.TensorSpec(shape=(None,), dtype=tf.int32),         # (T,)
        tf.TensorSpec(shape=(None,), dtype=tf.float32),       # (T,)
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------
# TCN model (NON-CAUSAL dense)
# -----------------------
def tcn_residual_block(
    x,
    filters: int,
    kernel_size: int,
    dilation: int,
    dropout: float,
    use_layernorm: bool,
    name: str,
):
    """
    Residual TCN block:
      Conv1D(same, dilated) -> Norm -> ReLU -> Dropout
      Conv1D(same, dilated) -> Norm -> ReLU -> Dropout
      Residual add (+ 1x1 projection if needed)

    Non-causal: padding="same" so the output at t can depend on both past and future.
    """
    in_ch = x.shape[-1]

    def norm_layer(nm):
        return tf.keras.layers.LayerNormalization(name=nm) if use_layernorm else tf.keras.layers.BatchNormalization(name=nm)

    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    y = norm_layer(f"{name}_norm1")(y)
    y = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(y)
    y = norm_layer(f"{name}_norm2")(y)
    y = tf.keras.layers.Activation("relu", name=f"{name}_relu2")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop2")(y)

    if in_ch != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(x)
        shortcut = norm_layer(f"{name}_proj_norm")(shortcut)
    else:
        shortcut = x

    out = tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])
    out = tf.keras.layers.Activation("relu", name=f"{name}_out_relu")(out)
    return out


def build_noncausal_dense_tcn(
    num_channels: int,
    num_classes: int,
    filters: int,
    kernel_size: int,
    n_blocks: int,
    dropout: float,
    use_layernorm: bool,
):
    """
    Input:  (T, C)
    Output: (T, K) softmax per timestep
    """
    inp = tf.keras.Input(shape=(None, num_channels), name="input")  # variable T
    x = inp
    for i in range(n_blocks):
        d = 2 ** i
        x = tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=d,
            dropout=dropout,
            use_layernorm=use_layernorm,
            name=f"tcn_b{i+1}_d{d}",
        )

    logits = tf.keras.layers.Conv1D(num_classes, 1, padding="same", name="logits_1x1")(x)
    out = tf.keras.layers.Activation("softmax", name="softmax")(logits)
    return tf.keras.Model(inp, out, name="noncausal_dense_tcn")


# -----------------------
# Training per fold
# -----------------------
def infer_num_channels(X_obj):
    for x in X_obj:
        if x is not None and len(x) > 0:
            return int(x.shape[1])
    raise RuntimeError("Could not infer num_channels: all sequences empty?")


def train_one_fold(fold_dir: str, out_dir: str, args):
    os.makedirs(out_dir, exist_ok=True)

    (X_train, y_train, m_train,
     X_val, y_val, m_val,
     X_test, y_test, m_test,
     class_list, meta) = load_fullseq_fold(fold_dir)

    num_classes = len(class_list)
    num_channels = infer_num_channels(X_train)

    # datasets (infinite generators)
    train_ds = make_fullseq_dataset(X_train, y_train, m_train, shuffle=True,  seed=args.seed)
    val_ds   = make_fullseq_dataset(X_val,   y_val,   m_val,   shuffle=False, seed=args.seed)
    test_ds  = make_fullseq_dataset(X_test,  y_test,  m_test,  shuffle=False, seed=args.seed)

    # define epoch lengths explicitly (one pass over sequences)
    steps_per_epoch  = max(1, len(X_train))
    validation_steps = max(1, len(X_val))
    test_steps       = max(1, len(X_test))

    model = build_noncausal_dense_tcn(
        num_channels=num_channels,
        num_classes=num_classes,
        filters=args.filters,
        kernel_size=args.kernel_size,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "best.keras"),
            monitor="val_acc",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=7,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "training_log.csv")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Keras evaluate() uses sample_weight from dataset
    test_loss, test_acc = model.evaluate(test_ds, steps=test_steps, verbose=0)

    # Predict per test sequence and compute masked confusion/F1 offline
    y_pred_list = []
    for i in range(len(X_test)):
        X_i = X_test[i].astype(np.float32)[None, ...]  # (1, T, C)
        prob = model.predict(X_i, verbose=0)[0]        # (T, K)
        y_pred = np.argmax(prob, axis=-1).astype(np.int32)  # (T,)
        y_pred_list.append(y_pred)

    y_true_flat = []
    y_pred_flat = []
    for yt, yp, m in zip(y_test, y_pred_list, m_test):
        keep = (m.astype(np.float32) > 0.5) & (yt.astype(np.int32) >= 0)
        if not np.any(keep):
            continue
        y_true_flat.append(yt[keep].astype(np.int32))
        y_pred_flat.append(yp[keep].astype(np.int32))

    if len(y_true_flat) == 0:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        macro_f1 = 0.0
        f1_per_class = [0.0] * num_classes
        masked_acc = 0.0
    else:
        y_true_flat = np.concatenate(y_true_flat, axis=0)
        y_pred_flat = np.concatenate(y_pred_flat, axis=0)
        cm = confusion_matrix_np(y_true_flat, y_pred_flat, num_classes)
        macro_f1, f1_per_class = macro_f1_from_cm(cm)
        masked_acc = masked_time_weighted_accuracy(y_test, y_pred_list, m_test)

    results = {
        "fold_dir": fold_dir,
        "out_dir": out_dir,
        "held_out_pid": meta.get("held_out_pid", None),
        "classes": class_list,
        "test_acc_keras_masked": float(test_acc),
        "test_acc_time_weighted_masked": float(masked_acc),
        "test_macro_f1_masked": float(macro_f1),
        "f1_per_class": {class_list[i]: float(f1_per_class[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "tcn_config": {
            "filters": int(args.filters),
            "kernel_size": int(args.kernel_size),
            "n_blocks": int(args.n_blocks),
            "dropout": float(args.dropout),
            "causal": False,
            "padding": "same",
            "normalization": "LayerNorm" if args.use_layernorm else "BatchNorm",
        },
        "train_config": {
            "epochs": args.epochs,
            "batch_size": 1,
            "lr": args.lr,
            "seed": args.seed,
        },
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(out_dir, "final.keras"))
    return results


def aggregate_results(all_results):
    accs = np.array([r["test_acc_time_weighted_masked"] for r in all_results], dtype=np.float32)
    f1s  = np.array([r["test_macro_f1_masked"] for r in all_results], dtype=np.float32)

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
                "test_acc": r["test_acc_time_weighted_masked"],
                "test_macro_f1": r["test_macro_f1_masked"],
                "out_dir": r["out_dir"],
            } for r in all_results
        ],
    }
    return summary


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fold_dir", type=str, default=None,
                    help="Path to a single fold dir, e.g., ../../runs/full_sequence_folds/fold_P22")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir for a single-fold run")
    ap.add_argument("--all_folds", action="store_true",
                    help="Train/eval on every fold_* directory under --folds_root")
    ap.add_argument("--folds_root", type=str, default="../../runs/full_sequence_folds",
                    help="Directory containing fold_* folders created by prepare_fullseq_folds.py")
    ap.add_argument("--out_root", type=str, default="../../runs/training_results/tcn_fullseq_noncausal_lopo",
                    help="Root output directory for LOPO runs (one subdir per fold)")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--n_blocks", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--use_layernorm", action="store_true",
                    help="Use LayerNorm instead of BatchNorm (recommended for batch_size=1).")

    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Avoid TF greedily grabbing memory (harmless on CPU too)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    if args.all_folds:
        fold_dirs = list_fold_dirs(args.folds_root)
        if len(fold_dirs) == 0:
            raise SystemExit(f"No fold_* directories found in: {args.folds_root}")

        os.makedirs(args.out_root, exist_ok=True)

        all_results = []
        for fold_dir in fold_dirs:
            held_out = os.path.basename(fold_dir).replace("fold_", "")
            out_dir = os.path.join(args.out_root, f"tcn_fullseq_noncausal_fold_{held_out}")

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

    if args.fold_dir is None:
        raise SystemExit("Provide --fold_dir for a single-fold run, or use --all_folds.")
    if args.out_dir is None:
        raise SystemExit("Provide --out_dir for a single-fold run.")

    train_one_fold(args.fold_dir, args.out_dir, args)


if __name__ == "__main__":
    main()