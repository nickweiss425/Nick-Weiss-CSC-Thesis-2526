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


def make_tf_dataset(X, y, batch_size=256, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(y), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(y: np.ndarray, num_classes: int):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    # avoid division by zero
    counts = np.maximum(counts, 1.0)
    weights = total / (num_classes * counts)
    return {i: float(weights[i]) for i in range(num_classes)}


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray):
    # cm rows = true, cols = pred
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
def build_cnn(input_shape, num_classes):
    inp = tf.keras.Input(shape=input_shape)  # (T, C)

    x = tf.keras.layers.Conv1D(64, kernel_size=7, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
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


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", type=str, default="runs/prep/fold_P22")
    ap.add_argument("--out_dir", type=str, default="runs/models/cnn_fold_P22")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, class_list, meta = load_fold(args.fold_dir)
    num_classes = len(class_list)

    print("[INFO] Loaded fold:", args.fold_dir)
    print("  Train:", X_train.shape, y_train.shape)
    print("  Val:  ", X_val.shape, y_val.shape)
    print("  Test: ", X_test.shape, y_test.shape)
    print("  Classes:", class_list)

    train_ds = make_tf_dataset(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds   = make_tf_dataset(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_ds  = make_tf_dataset(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, "best.keras"),
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
        tf.keras.callbacks.CSVLogger(os.path.join(args.out_dir, "training_log.csv"))
    ]

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(y_train, num_classes)
        print("[INFO] Using class weights:", class_weight)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate on test
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # Predictions for metrics
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix_np(y_test, y_pred, num_classes)
    macro_f1, f1_per_class = macro_f1_from_cm(cm)

    print("\n[TEST RESULTS]")
    print(f"  acc      = {test_acc:.4f}")
    print(f"  macro_f1 = {macro_f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save results
    results = {
        "fold_dir": args.fold_dir,
        "out_dir": args.out_dir,
        "classes": class_list,
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "f1_per_class": {class_list[i]: f1_per_class[i] for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "use_class_weights": bool(args.use_class_weights),
            "seed": args.seed,
        }
    }

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save final model too (in addition to best.keras checkpoint)
    model.save(os.path.join(args.out_dir, "final.keras"))

    print(f"\n[SAVED] {args.out_dir}")
    print("  - best.keras")
    print("  - final.keras")
    print("  - results.json")
    print("  - training_log.csv")


if __name__ == "__main__":
    main()
