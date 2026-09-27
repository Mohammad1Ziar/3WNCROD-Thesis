"""
generate_paper_tables.py
Generates official paper comparison tables for Sub-Idea 3 (Expanded Boundary Lambda Space).
Outputs:
  - New_idea/2/sub_ideas/3/results/paper_comparisons/Expanded_Performance_Comparison.csv
  - New_idea/2/sub_ideas/3/results/paper_comparisons/Expanded_RunningTime_Comparison.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_BASE_DIR = CURRENT_DIR / "results"
SUMMARY_DIR = RESULTS_BASE_DIR / "0_summary"
OUT_DIR = RESULTS_BASE_DIR / "paper_comparisons"

PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
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
    "T1":  (0.985, 0.716, 3.940),
    "T2":  (0.985, 0.761, 6.714),
    "T3":  (0.992, 0.852, 6.941),
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
    summary_file = SUMMARY_DIR / "expanded_boundary_summary.csv"
    if not summary_file.exists():
        print(f"❌ Error: {summary_file} not found! Run run_expanded_boundary_benchmark.py first.")
        return

    df_summary = pd.read_csv(summary_file).set_index("subset")
    df_base_perf = pd.read_csv(BASELINE_PERF).set_index("Subset") if BASELINE_PERF.exists() else None

    perf_records = []
    time_records = []

    eval_paper_auc, eval_base_auc, eval_hybrid_auc = [], [], []
    eval_paper_ap, eval_base_ap, eval_hybrid_ap = [], [], []
    eval_paper_time, eval_hybrid_time = [], []

    for sub in CANONICAL_ORDER:
        paper_auc, paper_ap, paper_t = PAPER_RESULTS[sub]
        row_raw = get_row_safely(df_summary, sub)

        if row_raw is not None:
            our_auc = round(float(row_raw["final_auroc"]), 3)
            our_ap = round(float(row_raw["final_ap"]), 3)
            our_t = round(float(row_raw["total_search_time_sec"]), 2)

            best_pair_roc = f"({row_raw['best_w_roc']}, {row_raw['best_lam_roc']})"
            best_pair_ap = f"({row_raw['best_w_ap']}, {row_raw['best_lam_ap']})"

            base_row = get_row_safely(df_base_perf, sub)
            base_auc = base_row["Ours_AUC"] if base_row is not None and "Ours_AUC" in base_row else "-"
            base_ap = base_row["Ours_AP"] if base_row is not None and "Ours_AP" in base_row else "-"

            eval_paper_auc.append(paper_auc)
            eval_hybrid_auc.append(our_auc)
            eval_paper_ap.append(paper_ap)
            eval_hybrid_ap.append(our_ap)
            eval_paper_time.append(paper_t)
            eval_hybrid_time.append(our_t)

            if base_auc != "-":
                try: eval_base_auc.append(float(base_auc))
                except: pass
            if base_ap != "-":
                try: eval_base_ap.append(float(base_ap))
                except: pass

            perf_records.append({
                "Subset": sub,
                "Best_(w,λ)_ROC": best_pair_roc,
                "Paper_AUC": f"{paper_auc:.3f}",
                "Baseline_AUC": base_auc,
                "Expanded_AUC": f"{our_auc:.3f}",
                "Best_(w,λ)_AP": best_pair_ap,
                "Paper_AP": f"{paper_ap:.3f}",
                "Baseline_AP": base_ap,
                "Expanded_AP": f"{our_ap:.3f}"
            })

            speedup = round(paper_t / our_t, 3) if our_t > 0 else np.nan
            time_records.append({
                "Subset": sub,
                "Paper_Time(s)": f"{paper_t:.3f}",
                "Search_Time(s)": f"{our_t:.2f}",
                "Speedup_vs_Paper": f"{speedup:.3f}x" if pd.notna(speedup) else "-"
            })
        else:
            perf_records.append({
                "Subset": sub,
                "Best_(w,λ)_ROC": "-",
                "Paper_AUC": f"{paper_auc:.3f}",
                "Baseline_AUC": "-",
                "Expanded_AUC": "-",
                "Best_(w,λ)_AP": "-",
                "Paper_AP": f"{paper_ap:.3f}",
                "Baseline_AP": "-",
                "Expanded_AP": "-"
            })
            time_records.append({
                "Subset": sub,
                "Paper_Time(s)": f"{paper_t:.3f}",
                "Search_Time(s)": "-",
                "Speedup_vs_Paper": "-"
            })

    if len(eval_hybrid_auc) > 0:
        mean_paper_auc = np.mean(eval_paper_auc)
        mean_base_auc = np.mean(eval_base_auc) if len(eval_base_auc) > 0 else np.nan
        mean_hybrid_auc = np.mean(eval_hybrid_auc)

        mean_paper_ap = np.mean(eval_paper_ap)
        mean_base_ap = np.mean(eval_base_ap) if len(eval_base_ap) > 0 else np.nan
        mean_hybrid_ap = np.mean(eval_hybrid_ap)

        avg_perf_row = {
            "Subset": "Average",
            "Best_(w,λ)_ROC": "-",
            "Paper_AUC": f"{mean_paper_auc:.3f}",
            "Baseline_AUC": f"{mean_base_auc:.3f}" if pd.notna(mean_base_auc) else "-",
            "Expanded_AUC": f"{mean_hybrid_auc:.3f}",
            "Best_(w,λ)_AP": "-",
            "Paper_AP": f"{mean_paper_ap:.3f}",
            "Baseline_AP": f"{mean_base_ap:.3f}" if pd.notna(mean_base_ap) else "-",
            "Expanded_AP": f"{mean_hybrid_ap:.3f}"
        }
        perf_records.append(avg_perf_row)

        mean_paper_t = np.mean(eval_paper_time)
        mean_hybrid_t = np.mean(eval_hybrid_time)
        avg_speedup = round(mean_paper_t / mean_hybrid_t, 3) if mean_hybrid_t > 0 else np.nan

        avg_time_row = {
            "Subset": "Average",
            "Paper_Time(s)": f"{mean_paper_t:.3f}",
            "Search_Time(s)": f"{mean_hybrid_t:.2f}",
            "Speedup_vs_Paper": f"{avg_speedup:.3f}x" if pd.notna(avg_speedup) else "-"
        }
        time_records.append(avg_time_row)

    df_perf_out = pd.DataFrame(perf_records)
    df_time_out = pd.DataFrame(time_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perf_path = OUT_DIR / "Expanded_Performance_Comparison.csv"
    time_path = OUT_DIR / "Expanded_RunningTime_Comparison.csv"

    df_perf_out.to_csv(perf_path, index=False)
    df_time_out.to_csv(time_path, index=False)

    print("\n" + "=" * 90)
    print("📊 EXPANDED BOUNDARY PERFORMANCE COMPARISON (ROC-AUC & AP)")
    print("=" * 90)
    cols_to_show = ["Subset", "Paper_AUC", "Baseline_AUC", "Expanded_AUC", 
                    "Paper_AP", "Baseline_AP", "Expanded_AP"]
    print(df_perf_out[cols_to_show].to_string(index=False))

    print("\n" + "=" * 90)
    print("⚡ EXPANDED BOUNDARY RUNNING TIME")
    print("=" * 90)
    print(df_time_out.to_string(index=False))
    print("=" * 90)

    if len(eval_hybrid_auc) > 0:
        print(f"\n📈 Overall Performance Averages ({len(eval_hybrid_auc)} Subsets Evaluated):")
        print(f"  • ROC-AUC Average: Paper = {mean_paper_auc:.3f} | Expanded = {mean_hybrid_auc:.3f}")
        print(f"  • AP Average:      Paper = {mean_paper_ap:.3f}  | Expanded = {mean_hybrid_ap:.3f}")
        print(f"\n📁 Files successfully exported to:")
        print(f"  1. {perf_path}")
        print(f"  2. {time_path}")


if __name__ == "__main__":
    generate_comparison_tables()