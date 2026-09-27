"""
generate_paper_tables.py
Generates exactly 2 clean comparison tables with exact columns requested.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_BASE_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_DIR = RESULTS_BASE_DIR / "0_summary"
OUT_DIR = RESULTS_BASE_DIR / "paper_comparisons"

# مقادیر دقیق ستون 3WNCROD از جداول VI، VII و VIII مقاله
PAPER_RESULTS = {
    "A1":  (0.732, 0.437, 15.748),
    "A2":  (0.692, 0.427, 19.689),
    "A3":  (0.775, 0.278, 18.799),
    "C1":  (0.990, 0.800, 0.219),
    "C2":  (0.999, 0.993, 0.084),
    "C3":  (0.997, 0.978, 0.467),
    "G1":  (0.987, 0.561, 0.831),
    "G2":  (0.986, 0.543, 0.398),
    "G3":  (0.996, 0.501, 0.393),
    "H1":  (0.989, 0.931, 0.009),
    "H2":  (0.998, 0.973, 0.012),
    "H3":  (0.988, 0.817, 0.011),
    "L1":  (1.000, 1.000, 0.040),
    "L2":  (1.000, 1.000, 0.011),
    "L3":  (1.000, 1.000, 0.013),
    "MA1": (0.893, 0.355, 45.042),
    "MA2": (0.775, 0.073, 61.293),
    "MA3": (0.788, 0.187, 53.314),
    "M1":  (0.985, 0.920, 18.533),
    "M2":  (0.939, 0.898, 26.294),
    "M3":  (0.978, 0.753, 17.812),
    "MU1": (1.000, 1.000, 110.730),
    "MU2": (1.000, 1.000, 93.271),
    "MU3": (1.000, 1.000, 101.388),
    "TH1": (0.985, 0.716, 3.940),
    "TH2": (0.985, 0.761, 6.714),
    "TH3": (0.992, 0.852, 6.941),
    "W1":  (1.000, 1.000, 0.064),
    "W2":  (1.000, 1.000, 0.131),
    "W3":  (0.995, 0.921, 0.061),
}

def load_reproduced_matrices():
    p_auc = SUMMARY_DIR / "summary_ROC_AUC.csv"
    p_ap = SUMMARY_DIR / "summary_AP.csv"
    p_time = SUMMARY_DIR / "summary_running_time.csv"

    df_auc = pd.read_csv(p_auc, index_col="subset") if p_auc.exists() else None
    df_ap = pd.read_csv(p_ap, index_col="subset") if p_ap.exists() else None
    df_time = pd.read_csv(p_time, index_col="subset") if p_time.exists() else None
    return df_auc, df_ap, df_time

def get_row_safely(df, sub):
    """پشتیبانی همزمان از نام‌های TH و T در دیتاسورس"""
    if df is None:
        return None
    if sub in df.index:
        return df.loc[sub]
    alt = sub.replace("TH", "T") if "TH" in sub else sub.replace("T", "TH")
    if alt in df.index:
        return df.loc[alt]
    return None

def generate_two_tables():
    df_auc, df_ap, df_time = load_reproduced_matrices()
    if df_auc is None:
        print(f"Error: {SUMMARY_DIR / 'summary_ROC_AUC.csv'} not found!")
        return

    perf_records = []
    time_records = []

    for sub, (paper_auc, paper_ap, paper_t) in PAPER_RESULTS.items():
        row_auc_raw = get_row_safely(df_auc, sub)
        row_ap_raw = get_row_safely(df_ap, sub)
        row_time_raw = get_row_safely(df_time, sub)

        if row_auc_raw is not None:
            row_auc = row_auc_raw.dropna()
            best_lambda_col = row_auc.idxmax()
            best_lambda_val = best_lambda_col.replace("lambda_", "")

            our_auc = round(float(row_auc.max()), 3)
            
            our_ap_val = row_ap_raw[best_lambda_col] if row_ap_raw is not None and best_lambda_col in row_ap_raw else np.nan
            our_ap = round(float(our_ap_val), 3) if pd.notna(our_ap_val) else np.nan

            our_t_val = row_time_raw[best_lambda_col] if row_time_raw is not None and best_lambda_col in row_time_raw else np.nan
            our_t = round(float(our_t_val), 3) if pd.notna(our_t_val) else np.nan

            if row_time_raw is not None:
                total_sweep_raw = float(row_time_raw.dropna().sum())
                total_sweep_time = round(total_sweep_raw, 3)
                sweep_speedup = round(paper_t / total_sweep_raw, 3) if total_sweep_raw > 0 else np.nan
            else:
                total_sweep_time = np.nan
                sweep_speedup = np.nan

            diff_auc = round(our_auc - paper_auc, 3)
            if abs(diff_auc) < 1e-9: diff_auc = 0.0
            diff_ap = round(our_ap - paper_ap, 3) if pd.notna(our_ap) else np.nan
            if pd.notna(diff_ap) and abs(diff_ap) < 1e-9: diff_ap = 0.0
            
            speedup = round(paper_t / our_t, 3) if (pd.notna(our_t) and our_t > 0) else np.nan
        else:
            best_lambda_val = "-"
            our_auc, our_ap, our_t, total_sweep_time = np.nan, np.nan, np.nan, np.nan
            diff_auc, diff_ap, speedup, sweep_speedup = np.nan, np.nan, np.nan, np.nan

        perf_records.append({
            "Subset": sub,
            "Best_λ": best_lambda_val,
            "Paper_AUC": f"{paper_auc:.3f}",
            "Ours_AUC": f"{our_auc:.3f}" if pd.notna(our_auc) else "-",
            "Diff_AUC": f"{diff_auc:+.3f}" if pd.notna(diff_auc) else "-",
            "Paper_AP": f"{paper_ap:.3f}",
            "Ours_AP": f"{our_ap:.3f}" if pd.notna(our_ap) else "-",
            "Diff_AP": f"{diff_ap:+.3f}" if pd.notna(diff_ap) else "-",
        })

        time_records.append({
            "Subset": sub,
            "Best_λ": best_lambda_val,
            "Paper_Time(s)": f"{paper_t:.3f}",
            "Best_Lambda_Time(s)": f"{our_t:.3f}" if pd.notna(our_t) else "-",
            "Speedup_Best": f"{speedup:.3f}x" if pd.notna(speedup) else "-",
            "Total_Sweep_Time(s)": f"{total_sweep_time:.3f}" if pd.notna(total_sweep_time) else "-",
            "Speedup_Full_Sweep": f"{sweep_speedup:.3f}x" if pd.notna(sweep_speedup) else "-"
        })

    df_perf = pd.DataFrame(perf_records)
    df_time = pd.DataFrame(time_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file_perf = OUT_DIR / "WNCROD_Performance_Comparison.csv"
    file_time = OUT_DIR / "WNCROD_RunningTime_Comparison.csv"

    df_perf.to_csv(file_perf, index=False)
    df_time.to_csv(file_time, index=False)

    print("\n" + "=" * 80)
    print("⚡ RUNNING TIME COMPARISON (Updated with exact requested columns)")
    print("=" * 80)
    print(df_time.to_string(index=False))
    print("=" * 80)
    print(f"\nFiles successfully saved to:\n- {file_perf}\n- {file_time}")

if __name__ == "__main__":
    generate_two_tables()