import os
import argparse
import pandas as pd

LABEL_COL = "Primitive"


def scan_one_participant(pdir: str, pid: str) -> bool:
    """
    Scan one participant folder for missing Primitive labels in labeled.csv.

    Returns True if missing labels were found, otherwise False.
    """


    csv_path = os.path.join(pdir, "labeled.csv")
    # no file -> skip, not an error
    if not os.path.exists(csv_path):
        return False  

    df = pd.read_csv(csv_path)

    if LABEL_COL not in df.columns:
        print(f"[WARN] {pid}: no '{LABEL_COL}' column found")
        return False

    missing_mask = df[LABEL_COL].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0:
        print(f"[FOUND] {pid}: {n_missing} missing '{LABEL_COL}' values")
        preview = df.loc[missing_mask, ["Time (s)", LABEL_COL]].head()
        print(preview)
        print("-" * 40)
        return True

    return False


def main(data_root: str, participant: str | None):
    print("Scanning labeled.csv files for missing Primitive values...\n")

    found_any = False

    if participant is not None:
        # Single-participant mode
        pdir = os.path.join(data_root, participant)
        if not os.path.isdir(pdir):
            raise FileNotFoundError(f"Participant folder not found: {pdir}")
        found_any = scan_one_participant(pdir, participant)
    else:
        # Batch mode: scan all participant folders in data_root
        for pid in sorted(os.listdir(data_root)):
            pdir = os.path.join(data_root, pid)
            if not os.path.isdir(pdir):
                continue
            found_any = scan_one_participant(pdir, pid) or found_any

    if not found_any:
        print("No missing Primitive labels found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing participant directories (e.g., .../data)",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default=None,
        help="Optional participant ID (e.g., P32). If omitted, scans all participants.",
    )
    args = parser.parse_args()

    main(args.data_root, args.participant)
