import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.criteria import SUBSET_REGISTRY
from data_loaders.base import load_csv_dataframe, load_mat_matrix

DATASETS_DIR = PROJECT_ROOT / "baseline" / "datasets"
if not DATASETS_DIR.exists():
    DATASETS_DIR = PROJECT_ROOT / "datasets"

def verify_all():
    print(f"{'Subset':<8} | {'Family':<12} | {'N (Act/Exp)':<16} | {'Outliers (Act/Exp)':<20} | {'Status'}")
    print("-" * 70)

    all_passed = True

    for name, spec in SUBSET_REGISTRY.items():
        csv_path = DATASETS_DIR / spec["csv_file"]
        mat_path = DATASETS_DIR / spec["mat_file"]

        if not csv_path.exists() or not mat_path.exists():
            print(f"{name:<8} | {spec['family']:<12} | {'FILE MISSING':<16} | {'FILE MISSING':<20} | ❌ FAIL")
            all_passed = False
            continue

        try:
            df = load_csv_dataframe(csv_path, has_header=spec["has_csv_header"])
            mask = spec["filter"](df)
            selected_indices = df[mask].index.to_numpy()
            n_actual = len(selected_indices)

            X_all, y_all = load_mat_matrix(mat_path, key=spec["mat_key"])
            outliers_actual = int(np.sum(y_all[selected_indices]))

            n_expected = spec["expected_objects"]
            out_expected = spec["expected_outliers"]

            n_match = (n_actual == n_expected)
            out_match = (outliers_actual == out_expected)

            status = "✅ PASS" if (n_match and out_match) else "❌ FAIL"
            if not (n_match and out_match):
                all_passed = False

            n_str = f"{n_actual}/{n_expected}"
            out_str = f"{outliers_actual}/{out_expected}"

            print(f"{name:<8} | {spec['family']:<12} | {n_str:<16} | {out_str:<20} | {status}")

        except Exception as e:
            print(f"{name:<8} | {spec['family']:<12} | {f'ERR: {str(e)[:10]}':<16} | -                    | ❌ FAIL")
            all_passed = False

    print("-" * 70)
    if all_passed:
        print("🎉 All 30 subsets perfectly match Table V specifications!")
    else:
        print("⚠ Mismatches found. Please inspect the failed subsets above.")

if __name__ == "__main__":
    verify_all()
