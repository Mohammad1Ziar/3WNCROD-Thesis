import pandas as pd
import numpy as np
from scipy.io import loadmat
from pathlib import Path

data_dir = Path("3.Datasets")

print("=" * 60)
print("1. CHECKING REPO CODES FOR WDBC LOGIC")
print("=" * 60)
found_ref = False
for p in Path("baseline/code").glob("*.*"):
    try:
        txt = p.read_text(errors="ignore")
        for term in ["wdbc", "w1", "w2", "w3", "363", "194"]:
            if term in txt.lower():
                print(f"-> Found '{term}' in baseline/code/{p.name}")
                found_ref = True
                break
    except Exception:
        pass
if not found_ref:
    print("No direct reference found in baseline/code.")

print("\n" + "=" * 60)
print("2. CHECKING ALL NON-EMPTY SHEETS OF WDBC XLS")
print("=" * 60)
xls_p = data_dir / "wdbc_M_39_variant1ori.xls"
if xls_p.exists():
    xls = pd.ExcelFile(xls_p)
    for s_name in xls.sheet_names:
        df = pd.read_excel(xls_p, sheet_name=s_name)
        if len(df) > 0:
            print(f"Sheet: {s_name} | Shape: {df.shape}")
        else:
            print(f"Sheet: {s_name} is empty.")

print("\n" + "=" * 60)
print("3. SCANNING WDBC CSV FOR EXACT COUNTS (42, 363, 194)")
print("=" * 60)
df_csv = pd.read_csv(data_dir / "wdbc_M_39_variant1ori.csv")
mat = loadmat(data_dir / "wdbc_M_39_variant1.mat")
labels = mat["trandata"][:, -1]
print(f"Total objects in WDBC: {len(df_csv)}, Outliers count: {(labels == 1).sum()}")

# بررسی تک‌ستونه برای 42 (W1)
for c in range(df_csv.shape[1] - 1):
    col = df_csv.iloc[:, c]
    vc = col.value_counts()
    for val, cnt in vc.items():
        if cnt in [42, 363, 194]:
            outliers = (labels[col == val] == 1).sum()
            print(f"Match found! Col {c} ({df_csv.columns[c]}) == {val}: {cnt} samples ({outliers} outliers)")

# بررسی ترکیب دو ستون برای 363 (W2) و 194 (W3) بر اساس مقادیر صحیح ممکن
print("\nScanning for OR conditions on integer columns...")
for c1 in range(1, 10):
    for c2 in range(1, 10):
        if c1 == c2:
            continue
        s1 = pd.to_numeric(df_csv.iloc[:, c1], errors="coerce").round()
        s2 = pd.to_numeric(df_csv.iloc[:, c2], errors="coerce").round()
        
        cnt_363 = ((s1 == 6) | (s2 == 2)).sum()
        if cnt_363 == 363:
            print(f"-> W2 candidate: (col {c1}==6 | col {c2}==2) -> 363 samples!")
        
        cnt_194 = ((s1 == 5) | (s2 == 3)).sum()
        if cnt_194 == 194:
            print(f"-> W3 candidate: (col {c1}==5 | col {c2}==3) -> 194 samples!")
