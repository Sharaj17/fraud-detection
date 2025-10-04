from pathlib import Path
import pandas as pd

csv_path = Path("data/raw/creditcard.csv")
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nClass balance:\n", df["Class"].value_counts(normalize=True).rename({0:"legit",1:"fraud"}))

print("\nBasic stats (Amount):\n", df["Amount"].describe())

# quick null check
nulls = df.isna().sum()
print("\nNulls per column (top 10):\n", nulls[nulls>0].sort_values(ascending=False).head(10))