"""
generate_paper_tables.py
Generates official paper comparison tables for the New Idea (Feature-Wise Multi-Lambda 3WNCROD).
"""

from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_PATH = Path(__file__).resolve()
if CURRENT_PATH.parent.name == "results":
    RESULTS_BASE_DIR = CURRENT_PATH.parent
    PROJECT_ROOT = CURRENT_PATH.parent.parent.parent
else:
    RESULTS_BASE_DIR = CURRENT_PATH.parent / "results"
    PROJECT_ROOT = CURRENT_PATH.parent.parent

SUMMARY_DIR = RESULTS_BASE_DIR / "0_summary"
OUT_DIR = RESULTS_BASE_DIR / "paper_comparisons"

BASELINE_PERF = PROJECT_ROOT / "baseline" / "reproduction" / "results" / "paper_comparisons" / "WNCROD_Performance_Comparison.csv"

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

CANONICAL_ORDER = list(PAPER_RESULTS.keys())

def get_row_safely(df, sub):
    if df is None:
        return None
    if sub in df.index:
        return df.loc[sub]
    alt = sub.replace("TH", "T") if "TH" in sub else sub.replace("T", "TH")
    if alt in df.index:
        return df.loc[alt]
    return None

def generate_comparison_tables():
    summary_file = SUMMARY_DIR / "new_idea_summary.csv"
    if not summary_file.exists():
        print(f"❌ Error: {summary_file} does not exist yet! Run run_new_idea.py first.")
        return

    df_summary = pd.read_csv(summary_file).set_index("subset")
    df_base_perf = pd.read_csv(BASELINE_PERF).set_index("Subset") if BASELINE_PERF.exists() else None

    perf_records = []
    time_records = []

    eval_paper_auc, eval_base_auc, eval_idea_auc = [], [], []
    eval_paper_ap, eval_base_ap, eval_idea_ap = [], [], []
    eval_paper_time, eval_idea_time = [], []

    for sub in CANONICAL_ORDER:
        paper_auc, paper_ap, paper_t = PAPER_RESULTS[sub]
        row = get_row_safely(df_summary, sub)

        if row is not None:
            our_auc = round(float(row["final_auroc"]), 3)
            our_ap = round(float(row["final_ap"]), 3)
            search_t = round(float(row["search_time_sec"]), 2)
            feat_cnt = int(row["features_count"])
            lam_roc = row["best_lambdas_roc"]
            lam_ap = row["best_lambdas_ap"]

            base_row = get_row_safely(df_base_perf, sub)
            base_auc = base_row["Ours_AUC"] if base_row is not None else "-"
            base_ap = base_row["Ours_AP"] if base_row is not None else "-"

            eval_paper_auc.append(paper_auc)
            eval_idea_auc.append(our_auc)
            eval_paper_ap.append(paper_ap)
            eval_idea_ap.append(our_ap)
            eval_paper_time.append(paper_t)
            eval_idea_time.append(search_t)

            if base_auc != "-":
                eval_base_auc.append(float(base_auc))
            if base_ap != "-":
                eval_base_ap.append(float(base_ap))

            perf_records.append({
                "Subset": sub,
                "Paper_AUC": f"{paper_auc:.3f}",
                "Baseline_AUC": base_auc,
                "NewIdea_AUC": f"{our_auc:.3f}",
                "Paper_AP": f"{paper_ap:.3f}",
                "Baseline_AP": base_ap,
                "NewIdea_AP": f"{our_ap:.3f}",
                "Best_λ_ROC": lam_roc,
                "Best_λ_AP": lam_ap
            })

            time_records.append({
                "Subset": sub,
                "Features": feat_cnt,
                "Paper_Time(s)": f"{paper_t:.3f}",
                "NewIdea_Total_Search_Time(s)": f"{search_t:.2f}",
                "Avg_Time_Per_Feature(s)": f"{(search_t / feat_cnt):.3f}"
            })
        else:
            perf_records.append({
                "Subset": sub,
                "Paper_AUC": f"{paper_auc:.3f}",
                "Baseline_AUC": "-",
                "NewIdea_AUC": "-",
                "Paper_AP": f"{paper_ap:.3f}",
                "Baseline_AP": "-",
                "NewIdea_AP": "-",
                "Best_λ_ROC": "-",
                "Best_λ_AP": "-"
            })
            time_records.append({
                "Subset": sub,
                "Features": "-",
                "Paper_Time(s)": f"{paper_t:.3f}",
                "NewIdea_Total_Search_Time(s)": "-",
                "Avg_Time_Per_Feature(s)": "-"
            })

    if len(eval_idea_auc) > 0:
        mean_paper_auc = np.mean(eval_paper_auc)
        mean_base_auc = np.mean(eval_base_auc) if len(eval_base_auc) > 0 else np.nan
        mean_idea_auc = np.mean(eval_idea_auc)

        mean_paper_ap = np.mean(eval_paper_ap)
        mean_base_ap = np.mean(eval_base_ap) if len(eval_base_ap) > 0 else np.nan
        mean_idea_ap = np.mean(eval_idea_ap)

        avg_perf_row = {
            "Subset": "Average",
            "Paper_AUC": f"{mean_paper_auc:.3f}",
            "Baseline_AUC": f"{mean_base_auc:.3f}" if pd.notna(mean_base_auc) else "-",
            "NewIdea_AUC": f"{mean_idea_auc:.3f}",
            "Paper_AP": f"{mean_paper_ap:.3f}",
            "Baseline_AP": f"{mean_base_ap:.3f}" if pd.notna(mean_base_ap) else "-",
            "NewIdea_AP": f"{mean_idea_ap:.3f}",
            "Best_λ_ROC": "-",
            "Best_λ_AP": "-"
        }
        perf_records.append(avg_perf_row)

        mean_paper_t = np.mean(eval_paper_time)
        mean_idea_t = np.mean(eval_idea_time)
        avg_time_row = {
            "Subset": "Average",
            "Features": "-",
            "Paper_Time(s)": f"{mean_paper_t:.3f}",
            "NewIdea_Total_Search_Time(s)": f"{mean_idea_t:.2f}",
            "Avg_Time_Per_Feature(s)": "-"
        }
        time_records.append(avg_time_row)

    df_perf_out = pd.DataFrame(perf_records)
    df_time_out = pd.DataFrame(time_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perf_path = OUT_DIR / "NewIdea_Performance_Comparison.csv"
    time_path = OUT_DIR / "NewIdea_RunningTime_Comparison.csv"

    df_perf_out.to_csv(perf_path, index=False)
    df_time_out.to_csv(time_path, index=False)

    print("\n" + "=" * 90)
    print("📊 NEW IDEA PERFORMANCE COMPARISON (ROC-AUC & AP - Rounded to 3 Decimals)")
    print("=" * 90)
    cols_to_show = ["Subset", "Paper_AUC", "Baseline_AUC", "NewIdea_AUC", 
                    "Paper_AP", "Baseline_AP", "NewIdea_AP"]
    print(df_perf_out[cols_to_show].to_string(index=False))

    print("\n" + "=" * 90)
    print("⚡ NEW IDEA RUNNING TIME & COMPUTATIONAL COST")
    print("=" * 90)
    print(df_time_out.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    generate_comparison_tables()