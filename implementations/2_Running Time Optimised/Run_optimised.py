"""
Optimized-time benchmark runner for the 3WNCROD algorithm.
This script reproduces the same evaluation flow as the baseline reproduction,
but uses the optimized implementation from `wncrod_optimized.py` and writes the
final artifacts to `result/4_Running Time Optimised`.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# Headless Matplotlib setup
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_thesis"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions

OPTIMIZED_PATH = Path(__file__).resolve().with_name("wncrod_optimized.py")
SPEC = importlib.util.spec_from_file_location("wncrod_optimized", OPTIMIZED_PATH)
OPTIMIZED_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTIMIZED_MODULE)
WNCROD_optimized = OPTIMIZED_MODULE.WNCROD_optimized

BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "2_Running Time Optimised"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = BASE_RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAMBDAS = [round(x, 1) for x in np.arange(0.1, 2.05, 0.1)]
NOMINAL_SUBSETS = {"L1", "L2", "L3", "M1", "M2", "M3"}

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


def generate_group_plots(group_name: str, group_subsets: list, sweep_data: dict, out_dir: Path):
    """Plot all three subsets of one family in a single merged AP figure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    plt.figure(figsize=(8, 5))
    for i, sub_name in enumerate(group_subsets):
        vals = sweep_data[sub_name]["aps"]
        plt.plot(DEFAULT_LAMBDAS, vals, marker=markers[i], linewidth=2.0, markersize=5, color=colors[i], label=sub_name)
    plt.title(f"Average Precision (AP) vs. λ — {group_name}", fontsize=13, fontweight="bold")
    plt.xlabel("λ (Radius Adjustment Parameter)", fontsize=11)
    plt.ylabel("Average Precision (AP)", fontsize=11)
    plt.xticks(DEFAULT_LAMBDAS, rotation=45, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{group_name}_AP_vs_lambda.png", dpi=300)
    plt.close()


def generate_group_roc_plots(group_name: str, group_subsets: list, sweep_data: dict, out_dir: Path):
    """Plot all three subsets of one family in a single merged ROC-AUC figure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    plt.figure(figsize=(8, 5))
    for i, sub_name in enumerate(group_subsets):
        vals = sweep_data[sub_name]["aucs"]
        plt.plot(DEFAULT_LAMBDAS, vals, marker=markers[i], linewidth=2.0, markersize=5, color=colors[i], label=sub_name)
    plt.title(f"ROC-AUC vs. λ — {group_name}", fontsize=13, fontweight="bold")
    plt.xlabel("λ (Radius Adjustment Parameter)", fontsize=11)
    plt.ylabel("ROC-AUC", fontsize=11)
    plt.xticks(DEFAULT_LAMBDAS, rotation=45, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{group_name}_ROC_AUC_vs_lambda.png", dpi=300)
    plt.close()


def run_optimized_pipeline():
    print("=" * 85)
    print("🚀 Running Optimized 3WNCROD on All 30 Subsets")
    print(f"📁 Output Directory: {BASE_RESULTS_DIR}")
    print(f"📊 24 Numerical Subsets: Full Sweep over {len(DEFAULT_LAMBDAS)} Lambdas")
    print(f"🏷️  6 Nominal Subsets: {sorted(list(NOMINAL_SUBSETS))} (Single execution, λ=1.0)")
    print("=" * 85)

    table1_rows = []
    table2_rows = []
    sweep_data = {}
    subsets = list(PAPER_RESULTS.keys())

    for sub_name in subsets:
        is_nominal = sub_name in NOMINAL_SUBSETS
        paper_auc, paper_ap, paper_time = PAPER_RESULTS[sub_name]

        _, y, X_tem = load_subset(sub_name, validate=False)
        mat_file = SUBSET_REGISTRY[sub_name]["mat_file"]
        mat_path = PROJECT_ROOT / "baseline" / "datasets" / mat_file
        data_raw = scipy.io.loadmat(str(mat_path))["trandata"][:, :-1].astype(float)

        if is_nominal:
            t0 = time.perf_counter()
            scores = WNCROD_optimized(data_raw, X_tem, lammda=1.0)
            elapsed = time.perf_counter() - t0

            metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem, elapsed_time=elapsed)
            best_ap = metrics["ap"]
            best_auc = metrics["auroc"]
            best_lam_ap = "-"
            best_lam_auc = "-"
            time_best_ap = elapsed
            time_best_auc = elapsed
            total_search_time_ap = elapsed
            total_search_time_auc = elapsed

            print(f"  [{sub_name:4s}] (Nominal) Done in {elapsed:.3f}s | AP: {best_ap:.4f} (Paper: {paper_ap:.3f}) | ROC: {best_auc:.4f} (Paper: {paper_auc:.3f})")
        else:
            sweep_aucs = []
            sweep_aps = []
            sweep_times = []

            for lam in DEFAULT_LAMBDAS:
                t0 = time.perf_counter()
                scores = WNCROD_optimized(data_raw, X_tem, lammda=lam)
                elapsed = time.perf_counter() - t0
                metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem, elapsed_time=elapsed)
                sweep_aucs.append(metrics["auroc"])
                sweep_aps.append(metrics["ap"])
                sweep_times.append(elapsed)

            best_idx_ap = int(np.argmax(sweep_aps))
            best_ap = sweep_aps[best_idx_ap]
            best_lam_ap = DEFAULT_LAMBDAS[best_idx_ap]
            time_best_ap = sweep_times[best_idx_ap]

            best_idx_auc = int(np.argmax(sweep_aucs))
            best_auc = sweep_aucs[best_idx_auc]
            best_lam_auc = DEFAULT_LAMBDAS[best_idx_auc]
            time_best_auc = sweep_times[best_idx_auc]

            total_search_time_ap = sum(sweep_times)
            total_search_time_auc = sum(sweep_times)
            sweep_data[sub_name] = {"aps": sweep_aps, "aucs": sweep_aucs}

            print(f"  [{sub_name:4s}] (Numeric) Done in {total_search_time_ap:.2f}s | Best AP: {best_ap:.4f} (λ={best_lam_ap}, Paper: {paper_ap:.3f}) | Best ROC: {best_auc:.4f} (λ={best_lam_auc}, Paper: {paper_auc:.3f})")

        table1_rows.append({
            "Subset": sub_name,
            "Best_Lambda_AP": str(best_lam_ap),
            "Best_AP": float(best_ap),
            "Paper_AP": float(paper_ap),
            "Best_Lambda_ROC": str(best_lam_auc),
            "Best_ROC_AUC": float(best_auc),
            "Paper_ROC_AUC": float(paper_auc),
        })

        table2_rows.append({
            "Subset": sub_name,
            "Time_Best_AP_sec": float(time_best_ap),
            "Time_Best_ROC_sec": float(time_best_auc),
            "Total_Search_Time_AP_sec": float(total_search_time_ap),
            "Total_Search_Time_ROC_sec": float(total_search_time_auc),
            "Paper_Time_sec": float(paper_time),
        })

    family_groups = {
        "A": ["A1", "A2", "A3"],
        "C": ["C1", "C2", "C3"],
        "G": ["G1", "G2", "G3"],
        "H": ["H1", "H2", "H3"],
        "MA": ["MA1", "MA2", "MA3"],
        "MU": ["MU1", "MU2", "MU3"],
        "TH": ["TH1", "TH2", "TH3"],
        "W": ["W1", "W2", "W3"],
    }
    for group_name, group_subsets in family_groups.items():
        if all(sub in sweep_data for sub in group_subsets):
            generate_group_plots(group_name, group_subsets, sweep_data, CHARTS_DIR)
            generate_group_roc_plots(group_name, group_subsets, sweep_data, CHARTS_DIR)

    df_table1 = pd.DataFrame(table1_rows)
    df_table2 = pd.DataFrame(table2_rows)

    avg_best_ap = df_table1["Best_AP"].mean()
    avg_paper_ap = df_table1["Paper_AP"].mean()
    avg_best_roc = df_table1["Best_ROC_AUC"].mean()
    avg_paper_roc = df_table1["Paper_ROC_AUC"].mean()

    avg_row_t1 = pd.DataFrame([{
        "Subset": "Average",
        "Best_Lambda_AP": "-",
        "Best_AP": avg_best_ap,
        "Paper_AP": avg_paper_ap,
        "Best_Lambda_ROC": "-",
        "Best_ROC_AUC": avg_best_roc,
        "Paper_ROC_AUC": avg_paper_roc,
    }])
    df_table1_final = pd.concat([df_table1, avg_row_t1], ignore_index=True)

    avg_time_best_ap = df_table2["Time_Best_AP_sec"].mean()
    avg_time_best_roc = df_table2["Time_Best_ROC_sec"].mean()
    avg_total_search_ap = df_table2["Total_Search_Time_AP_sec"].mean()
    avg_total_search_roc = df_table2["Total_Search_Time_ROC_sec"].mean()
    avg_paper_time = df_table2["Paper_Time_sec"].mean()

    avg_row_t2 = pd.DataFrame([{
        "Subset": "Average",
        "Time_Best_AP_sec": avg_time_best_ap,
        "Time_Best_ROC_sec": avg_time_best_roc,
        "Total_Search_Time_AP_sec": avg_total_search_ap,
        "Total_Search_Time_ROC_sec": avg_total_search_roc,
        "Paper_Time_sec": avg_paper_time,
    }])
    df_table2_final = pd.concat([df_table2, avg_row_t2], ignore_index=True)

    table1_csv_path = BASE_RESULTS_DIR / "Table1_Performance_Comparison.csv"
    table2_csv_path = BASE_RESULTS_DIR / "Table2_RunningTime_Comparison.csv"

    df_table1_export = df_table1_final.copy()
    for col in ["Best_AP", "Paper_AP", "Best_ROC_AUC", "Paper_ROC_AUC"]:
        df_table1_export[col] = df_table1_export[col].apply(lambda v: f"{v:.3f}" if pd.notnull(v) else "-")

    df_table2_export = df_table2_final.copy()
    for col in ["Time_Best_AP_sec", "Time_Best_ROC_sec", "Total_Search_Time_AP_sec", "Total_Search_Time_ROC_sec", "Paper_Time_sec"]:
        df_table2_export[col] = df_table2_export[col].apply(lambda v: f"{v:.4f}" if pd.notnull(v) else "-")

    df_table1_export.to_csv(table1_csv_path, index=False)
    df_table2_export.to_csv(table2_csv_path, index=False)

    print("\n" + "=" * 90)
    print("📊 TABLE 1: PERFORMANCE COMPARISON")
    print("=" * 90)
    print(df_table1_export.to_string(index=False))

    print("\n" + "=" * 90)
    print("⏱️ TABLE 2: RUNNING TIME COMPARISON (Seconds)")
    print("=" * 90)
    print(df_table2_export.to_string(index=False))

    print("\n" + "=" * 90)
    print("✅ All optimized runs completed successfully!")
    print(f"📁 Tables saved to:\n  - {table1_csv_path}\n  - {table2_csv_path}")
    print(f"📈 {len(family_groups)} Family charts saved in: {CHARTS_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    run_optimized_pipeline()


