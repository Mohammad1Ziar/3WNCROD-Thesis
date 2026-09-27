import json
from pathlib import Path
import pandas as pd
import numpy as np

data_dir = Path("3.Datasets")

print("=" * 60)
print("1. HEART CHECK")
print("=" * 60)
df_h = pd.read_csv(data_dir / "heart270_2_16_variant1ori.csv", header=None)
h1_match = (df_h.iloc[:, 1] == 1).sum()
h2_match = (df_h.iloc[:, 5] == 0).sum()
h3_match = (df_h.iloc[:, 8] == 0).sum()
print(f"H1 (col 1 == 1): {h1_match} (Expected: 91)")
print(f"H2 (col 5 == 0): {h2_match} (Expected: 137)")
print(f"H3 (col 8 == 0): {h3_match} (Expected: 131)")

print("\n" + "=" * 60)
print("2. LYMPHOGRAPHY CHECK")
print("=" * 60)
df_l = pd.read_csv(data_dir / "lymphographyori.csv", header=None)
print("Shape:", df_l.shape)
print("Column 1 unique values:", df_l.iloc[:, 1].unique()[:5])
print("Column 2 unique values:", df_l.iloc[:, 2].unique()[:5])
for c in range(df_l.shape[1]):
    vc = df_l.iloc[:, c].astype(str).str.strip().value_counts().to_dict()
    for val, count in vc.items():
        if count in [122, 66, 85]:
            print(f"-> Found match in col {c}: value '{val}' has {count} samples")

print("\n" + "=" * 60)
print("3. MUSHROOM CHECK")
print("=" * 60)
df_m = pd.read_csv(data_dir / "mushroom_p_221_variant1ori.csv", header=None)
print("Shape:", df_m.shape)
for c in range(df_m.shape[1]):
    vc = df_m.iloc[:, c].value_counts().to_dict()
    for val, count in vc.items():
        if count in [1649, 3218, 1211]:
            print(f"-> Found match in col {c}: value {val} has {count} samples")

print("\n" + "=" * 60)
print("4. MUSK NOTEBOOK CONDITIONS")
print("=" * 60)
nb_mu = Path("2. 3WNCROD+/1. Implementation/1st - FeatureWise/8. MU/WNCROD_Full_MU.ipynb")
if nb_mu.exists():
    with open(nb_mu) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "mu_conditions" in src or "MU2" in src or "get_ma_indices" in src or "get_mu_indices" in src:
            print(src.strip())
            print("-" * 40)

print("\n" + "=" * 60)
print("5. WDBC COLUMN INSPECTION")
print("=" * 60)
df_w = pd.read_csv(data_dir / "wdbc_M_39_variant1ori.csv")
print("WDBC CSV Shape:", df_w.shape)
print("WDBC Columns:", df_w.columns.tolist()[:10])
for c in range(df_w.shape[1]):
    vc = df_w.iloc[:, c].value_counts().to_dict()
    for val, count in vc.items():
        if count in [42, 363, 194]:
            print(f"-> Found match in col {c} ({df_w.columns[c]}): value {val} has {count} samples")
