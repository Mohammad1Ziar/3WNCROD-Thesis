import json
from pathlib import Path
import pandas as pd
import numpy as np

data_dir = Path("3.Datasets")

# 1. LYMPHOGRAPHY L3 CHECK
print("=" * 60)
print("1. LYMPHOGRAPHY L3 SEARCH (Target: 85)")
print("=" * 60)
df_l = pd.read_csv(data_dir / "lymphographyori.csv", header=0) # skip header row
print("Columns after header:", df_l.columns[:5].tolist())
# Check Table V: c13 == 'yes' or c18 == 'no' -> in 0-indexed: col 12 or col 17
for c1 in range(df_l.shape[1]):
    for c2 in range(c1 + 1, df_l.shape[1]):
        for v1 in ['1', '2', 1, 2, 'yes', 'no']:
            for v2 in ['1', '2', 1, 2, 'yes', 'no']:
                cnt = ((df_l.iloc[:, c1] == v1) | (df_l.iloc[:, c2] == v2)).sum()
                if cnt == 85:
                    print(f"-> L3 Match found: (col {c1} == {v1}) | (col {c2} == {v2}) -> {cnt} samples")

# 2. MUSK NOTEBOOK INSPECTION
print("\n" + "=" * 60)
print("2. MUSK NOTEBOOK FULL CODE EXTRACTION")
print("=" * 60)
nb_mu = Path("2. 3WNCROD+/1. Implementation/1st - FeatureWise/8. MU/WNCROD_Full_MU.ipynb")
if nb_mu.exists():
    with open(nb_mu) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "mu_conditions" in src or "MU2" in src or "low" in src or "high" in src:
            print("Found Musk Logic:")
            print(src)
            print("-" * 40)

# 3. WDBC CHECK
print("\n" + "=" * 60)
print("3. WDBC CONDITIONS INSPECTION")
print("=" * 60)
df_w = pd.read_csv(data_dir / "wdbc_M_39_variant1ori.csv")
print("WDBC Shape:", df_w.shape)
# Table V states:
# W1: c4 in [x, y] or c4 == ?
# Let's inspect unique counts or ranges
for c in range(df_w.shape[1]):
    col = df_w.iloc[:, c]
    if col.nunique() < 15:
        print(f"Categorical col {c} ({df_w.columns[c]}): {col.value_counts().to_dict()}")

# Check Table V paper mentions for WDBC
# Check if there is an ori.xls
xls_w = data_dir / "wdbc_M_39_variant1ori.xls"
if xls_w.exists():
    df_wx = pd.read_excel(xls_w)
    print("XLS Shape:", df_wx.shape, "Columns:", df_wx.columns[:5].tolist())
