import os
import json
import argparse
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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


# ---------- Window featurization ----------
def featurize_windows(X: np.ndarray, mode: str) -> np.ndarray:
    """
    X: (N, T, C)
    Returns 2D array (N, D)

    mode="stats": mean/std/min/max over time per channel => D = 4*C
    mode="flatten": flatten time+channels => D = T*C
    """
    if X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,T,C); got {X.shape}")

    if mode == "stats":
        mean = X.mean(axis=1)
        std  = X.std(axis=1)
        mn   = X.min(axis=1)
        mx   = X.max(axis=1)
        F = np.concatenate([mean, std, mn, mx], axis=1).astype(np.float32)
        return F

    if mode == "flatten":
        N, T, C = X.shape
        return X.reshape(N, T * C).astype(np.float32)

    raise ValueError(f"Unknown featurize mode: {mode}")


# ---------- Training ----------
def train_one_fold(fold_dir: str, out_dir: str, args):
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, class_list, meta = load_fold(fold_dir)
    num_classes = len(class_list)

    # Featurize windows
    F_train = featurize_windows(X_train, args.featurize)
    F_val   = featurize_windows(X_val, args.featurize)
    F_test  = featurize_windows(X_test, args.featurize)

    # Model (RF baseline)
    # class_weight='balanced' is a reasonable analogue to your class-weight option.
    class_weight = "balanced" if args.use_class_weights else None

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        class_weight=class_weight,
        bootstrap=True,
    )

    rf.fit(F_train, y_train)

    # Evaluate on test
    y_pred = rf.predict(F_test)

    test_acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    macro_f1, f1_per_class = macro_f1_from_cm(cm)

    results = {
        "fold_dir": fold_dir,
        "out_dir": out_dir,
        "held_out_pid": meta.get("held_out_pid", None),
        "classes": class_list,
        "test_acc": test_acc,
        "test_macro_f1": float(macro_f1),
        "f1_per_class": {class_list[i]: float(f1_per_class[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "config": {
            "model": "RandomForestClassifier",
            "featurize": args.featurize,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "use_class_weights": bool(args.use_class_weights),
            "seed": args.seed,
        }
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(rf, os.path.join(out_dir, "model.joblib"))

    # optional: also save feature dimension info
    with open(os.path.join(out_dir, "feature_info.json"), "w") as f:
        json.dump({
            "X_shape": list(X_train.shape),
            "F_shape": list(F_train.shape),
            "featurize": args.featurize
        }, f, indent=2)

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
                    help="Output dir for a single-fold run, e.g., runs/models/rf_fold_P22")
    ap.add_argument("--all_folds", action="store_true",
                    help="Train/eval on every fold_* directory under --folds_root")
    ap.add_argument("--folds_root", type=str, default="runs/prep",
                    help="Directory containing fold_* folders")
    ap.add_argument("--out_root", type=str, default="runs/models/rf_lopo",
                    help="Root output directory for LOPO runs (one subdir per fold)")

    # --- Featurization ---
    ap.add_argument("--featurize", type=str, default="stats",
                    choices=["stats", "flatten"],
                    help="How to convert (T,C) windows into vectors. stats is recommended baseline.")

    # --- RF hyperparams ---
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--max_features", type=str, default="sqrt",
                    help="RF max_features (e.g., 'sqrt', 'log2', float, int).")
    ap.add_argument("--n_jobs", type=int, default=-1)

    # --- Options ---
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
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
            out_dir = os.path.join(args.out_root, f"rf_fold_{held_out}")

            print(f"\n[RUN] {fold_dir} -> {out_dir}")
            r = train_one_fold(fold_dir, out_dir, args)
            all_results.append(r)

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

    r = train_one_fold(args.fold_dir, args.out_dir, args)
    print("\n[DONE]")
    print(f"  held_out: {r.get('held_out_pid')}")
    print(f"  test acc: {r['test_acc']:.4f}")
    print(f"  macro f1: {r['test_macro_f1']:.4f}")
    print(f"  saved:    {args.out_dir}")


if __name__ == "__main__":
    main()
