from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # OpenML dataset "creditcard" (anonymized PCA features V1..V28 + Amount + Class)
    print("Fetching dataset from OpenML (this may take a minute)...")
    Xy = fetch_openml(name="creditcard", version=1, as_frame=True)
    df = Xy.frame  # includes features + target ("Class")
    # Normalize column names just in case
    df.columns = [str(c) for c in df.columns]

    csv_path = out_dir / "creditcard.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (rows={len(df):,}, cols={len(df.columns)})")

if __name__ == "__main__":
    main()
