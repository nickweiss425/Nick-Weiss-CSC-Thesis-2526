import os
import pandas as pd

DATA_ROOT = "C:/Users/nicho/Desktop/Nick-Weiss-CSC-Thesis-2526/data"        
LABEL_COL = "Primitive"

def main():
    print("Scanning labeled.csv files for missing Primitive values...\n")

    found_any = False

    for pid in sorted(os.listdir(DATA_ROOT)):
        pdir = os.path.join(DATA_ROOT, pid)
        if not os.path.isdir(pdir):
            continue

        csv_path = os.path.join(pdir, "labeled.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        if LABEL_COL not in df.columns:
            print(f"[WARN] {pid}: no '{LABEL_COL}' column found")
            continue

        # find missing labels
        missing_mask = df[LABEL_COL].isna()
        n_missing = missing_mask.sum()

        if n_missing > 0:
            found_any = True
            print(f"[FOUND] {pid}: {n_missing} missing '{LABEL_COL}' values")

            # show first few indices + timestamps
            preview = df.loc[missing_mask, ["Time (s)", LABEL_COL]].head()
            print(preview)
            print("-" * 40)

    if not found_any:
        print("No missing Primitive labels found")

if __name__ == "__main__":
    main()
