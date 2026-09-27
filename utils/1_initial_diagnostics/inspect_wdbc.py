import pandas as pd
from pathlib import Path

data_dir = Path("3.Datasets")
xls_p = data_dir / "wdbc_M_39_variant1ori.xls"

if xls_p.exists():
    xls = pd.ExcelFile(xls_p)
    print("=== WDBC XLS SHEETS INSPECTION ===")
    for s_name in xls.sheet_names:
        df = pd.read_excel(xls_p, sheet_name=s_name)
        print(f"Sheet: {s_name:<8} | Shape: {df.shape} | Columns (first 5): {df.columns[:5].tolist()}")
        # چاپ اولین سطر
        print("First row:", df.iloc[0, :min(6, df.shape[1])].tolist())
        print("-" * 60)

    # بررسی شیت‌های ۲ و ۳ برای یافتن شروط ۴۲، ۳۶۳ و ۱۹۴
    for s_idx, s_name in enumerate(xls.sheet_names):
        df = pd.read_excel(xls_p, sheet_name=s_name)
        # اگر خود شیت تعداد سطرش برابر ساب‌ست‌ها باشد:
        if len(df) in [42, 363, 194]:
            print(f"-> Direct subset match: Sheet '{s_name}' has exactly {len(df)} rows!")
        
        # اگر شروط در ستون‌ها برقرار باشند:
        for c in range(min(10, df.shape[1])):
            s = pd.to_numeric(df.iloc[:, c], errors='coerce')
            if (s == 2).sum() == 42:
                print(f"-> W1 match in sheet '{s_name}', col {c} == 2 (42 samples)!")
            for c2 in range(min(10, df.shape[1])):
                s2 = pd.to_numeric(df.iloc[:, c2], errors='coerce')
                if ((s == 6) | (s2 == 2)).sum() == 363:
                    print(f"-> W2 match in sheet '{s_name}', (col {c}==6 | col {c2}==2) (363 samples)!")
                if ((s == 5) | (s2 == 3)).sum() == 194:
                    print(f"-> W3 match in sheet '{s_name}', (col {c}==5 | col {c2}==3) (194 samples)!")
