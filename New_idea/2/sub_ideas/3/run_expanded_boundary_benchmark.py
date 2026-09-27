"""
run_expanded_boundary_benchmark.py
Modular execution script for Sub-Idea 3: Boundary-Hit Lambda Space Expansion.
Expands lambda search space selectively:
  - Lower Expansion (5 steps: [0.01, 0.02, 0.04, 0.06, 0.08]) for:
    ['A1', 'A2', 'A3', 'MA1', 'MU1', 'MU2', 'MU3', 'T2']
  - Upper Expansion (5 steps: [2.1, 2.2, 2.3, 2.4, 2.5]) for:
    ['C2', 'G2', 'G3', 'T1', 'T2', 'T3']
Reuses:
  - New_idea/2/scale_estimators.py
  - New_idea/2/wncrod_hybrid.py
  - Cached non-boundary results from New_idea/2/results/0_summary/hybrid_grid_summary.csv
Outputs stored in:
  New_idea/2/sub_ideas/3/results/
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# تنظیم دقیق مسیرها برای ایمپورت از ریشه پروژه و پوشه ایده ۲
CURRENT_DIR = Path(__file__).resolve().parent
IDEA2_DIR = CURRENT_DIR.parent.parent
PROJECT_ROOT = IDEA2_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(IDEA2_DIR))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions, identify_outliers
from scale_estimators import sigma_mad, sigma_sn
from wncrod_hybrid import WNCROD_optimized_hybrid

RESULTS_BASE_DIR = CURRENT_DIR / "results"
RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = RESULTS_BASE_DIR / "0_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

IDEA2_SUMMARY_FILE = IDEA2_DIR / "results" / "0_summary" / "hybrid_grid_summary.csv"

# دامنه‌های لامبدا
BASE_LAMBDAS = [round(x, 1) for x in np.arange(0.1, 2.05, 0.1)]
LOWER_EXPANSION = [0.01, 0.02, 0.04, 0.06, 0.08]
UPPER_EXPANSION = [2.1, 2.2, 2.3, 2.4, 2.5]
WEIGHT_SPACE = [round(x, 1) for x in np.arange(0.0, 1.05, 0.1)]

# زیرمجموعه‌های نیازمند گسترش دامنه
LOWER_SUBSETS = ["A1", "A2", "A3", "MA1", "MU1", "MU2", "MU3", "T2"]
UPPER_SUBSETS = ["C2", "G2", "G3", "T1", "T2", "T3"]
BOUNDARY_SUBSETS = sorted(list(set(LOWER_SUBSETS + UPPER_SUBSETS)))

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


def get_lambda_space_for_subset(sub_name: str) -> list:
    lams = list(BASE_LAMBDAS)
    if sub_name in LOWER_SUBSETS:
        lams = LOWER_EXPANSION + lams
    if sub_name in UPPER_SUBSETS:
        lams = lams + UPPER_EXPANSION
    return sorted(list(set([round(x, 2) for x in lams])))


def precompute_base_scales(X: np.ndarray, id_mask: np.ndarray):
    m = X.shape[1]
    mad_vec = np.zeros(m, dtype=np.float64)
    sn_vec = np.zeros(m, dtype=np.float64)
    numerical_cols = np.flatnonzero(id_mask)
    for col in numerical_cols:
        col_data = X[:, col]
        mad_vec[col] = sigma_mad(col_data)
        sn_vec[col] = sigma_sn(col_data)
    return mad_vec, sn_vec


def save_incremental_summary(df_new: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_order = list(SUBSET_REGISTRY.keys())
    
    if file_path.exists():
        try:
            df_existing = pd.read_csv(file_path, index_col="subset")
            df_new_idx = df_new.set_index("subset")
            df_combined = df_new_idx.combine_first(df_existing)
            ordered_idx = [s for s in canonical_order if s in df_combined.index]
            remaining_idx = [s for s in df_combined.index if s not in ordered_idx]
            df_combined = df_combined.reindex(ordered_idx + remaining_idx)
            df_combined.reset_index().to_csv(file_path, index=False)
            return
        except Exception:
            pass
            
    df_new_idx = df_new.set_index("subset")
    ordered_idx = [s for s in canonical_order if s in df_new_idx.index]
    remaining_idx = [s for s in df_new_idx.index if s not in ordered_idx]
    df_new_idx = df_new_idx.reindex(ordered_idx + remaining_idx)
    df_new_idx.reset_index().to_csv(file_path, index=False)


def seed_summary_with_cached_results():
    """نتایج ۱۷ ساب‌ست غیرمرزی را از ایده ۲ کپی می‌کند تا نیازی به اجرای مجدد نباشد."""
    summary_file = SUMMARY_DIR / "expanded_boundary_summary.csv"
    if not summary_file.exists() and IDEA2_SUMMARY_FILE.exists():
        try:
            df_cached = pd.read_csv(IDEA2_SUMMARY_FILE)
            df_cached.to_csv(summary_file, index=False)
            print(f"ℹ️ Seeded baseline results from: {IDEA2_SUMMARY_FILE}")
        except Exception:
            pass


def run_expanded_benchmark(target_subsets=None, force_rerun: bool = False):
    seed_summary_with_cached_results()

    if target_subsets is None:
        target_subsets = list(BOUNDARY_SUBSETS)

    family_map = defaultdict(list)
    for sub in target_subsets:
        family = SUBSET_REGISTRY[sub]["family"]
        family_map[family].append(sub)

    print("=" * 90)
    print("🚀 Running Sub-Idea 3: Selective Boundary Lambda Expansion")
    print(f"   Target Subsets ({len(target_subsets)}): {target_subsets}")
    print(f"   Lower Extension: {LOWER_EXPANSION} for {LOWER_SUBSETS}")
    print(f"   Upper Extension: {UPPER_EXPANSION} for {UPPER_SUBSETS}")
    print("=" * 90)

    for family_name, subs in family_map.items():
        family_dir = RESULTS_BASE_DIR / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Processing Family: {family_name} ({', '.join(subs)})")

        family_runtimes = []

        for sub_name in subs:
            paper_auc, paper_ap, paper_t = PAPER_RESULTS.get(sub_name, (np.nan, np.nan, np.nan))
            lambda_space = get_lambda_space_for_subset(sub_name)

            metrics_file = family_dir / f"{sub_name}_expanded_grid_metrics.csv"
            if metrics_file.exists() and not force_rerun:
                try:
                    df_check = pd.read_csv(metrics_file)
                    if len(df_check) == (len(WEIGHT_SPACE) * len(lambda_space)):
                        print(f"  └─ {sub_name} [Already Computed! Skipped]")
                        continue
                except Exception:
                    pass

            print(f"  ├─ Loading {sub_name} (Testing {len(lambda_space)} lambdas: [{min(lambda_space)} ... {max(lambda_space)}])...")
            X, y, X_tem = load_subset(sub_name)

            t_sub_start = time.perf_counter()
            ID = np.all(X <= 1.0, axis=0)

            t_scale_start = time.perf_counter()
            mad_vec, sn_vec = precompute_base_scales(X, ID)
            t_scale = time.perf_counter() - t_scale_start

            best_auc = -1.0
            best_ap = -1.0
            best_pair_roc = (0.5, 1.0)
            best_pair_ap = (0.5, 1.0)
            best_scores_roc = None
            best_scores_ap = None

            grid_records = []
            both_saturated = False

            for w in WEIGHT_SPACE:
                if both_saturated:
                    break
                scale_vector = (1.0 - w) * sn_vec + w * mad_vec

                for lam in lambda_space:
                    t0 = time.perf_counter()
                    scores = WNCROD_optimized_hybrid(
                        X, X_tem, lammda=lam, w=w, scale_vector=scale_vector
                    )
                    elapsed = time.perf_counter() - t0

                    m = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem, elapsed_time=elapsed)

                    grid_records.append({
                        "subset": sub_name,
                        "weight": w,
                        "lambda": lam,
                        "auroc": m["auroc"],
                        "ap": m["ap"],
                        "time_sec": m["time_sec"]
                    })

                    if m["auroc"] > best_auc:
                        best_auc = m["auroc"]
                        best_pair_roc = (w, lam)
                        best_scores_roc = scores

                    if m["ap"] > best_ap:
                        best_ap = m["ap"]
                        best_pair_ap = (w, lam)
                        best_scores_ap = scores

                    if best_auc >= 1.0 - 1e-5 and best_ap >= 1.0 - 1e-5:
                        both_saturated = True
                        break

            total_sub_time = time.perf_counter() - t_sub_start

            df_grid = pd.DataFrame(grid_records)
            df_grid.to_csv(family_dir / f"{sub_name}_expanded_grid_metrics.csv", index=False)

            df_det_roc, df_miss_roc = identify_outliers(y, best_scores_roc, X_tem)
            df_det_roc.to_csv(family_dir / f"{sub_name}_detected_outliers_roc.csv", index=False)
            df_miss_roc.to_csv(family_dir / f"{sub_name}_missed_outliers_roc.csv", index=False)

            df_det_ap, df_miss_ap = identify_outliers(y, best_scores_ap, X_tem)
            df_det_ap.to_csv(family_dir / f"{sub_name}_detected_outliers_ap.csv", index=False)
            df_miss_ap.to_csv(family_dir / f"{sub_name}_missed_outliers_ap.csv", index=False)

            family_runtimes.append({
                "subset": sub_name,
                "objects": len(X_tem),
                "grid_evaluations": len(grid_records),
                "scale_precomp_time_sec": round(t_scale, 3),
                "total_grid_time_sec": round(total_sub_time, 2),
                "avg_time_per_eval_sec": round(total_sub_time / max(1, len(grid_records)), 4)
            })

            summary_row = {
                "subset": sub_name,
                "family": family_name,
                "best_w_roc": best_pair_roc[0],
                "best_lam_roc": best_pair_roc[1],
                "final_auroc": f"{best_auc:.3f}",
                "best_w_ap": best_pair_ap[0],
                "best_lam_ap": best_pair_ap[1],
                "final_ap": f"{best_ap:.3f}",
                "total_search_time_sec": round(total_sub_time, 2)
            }
            save_incremental_summary(pd.DataFrame([summary_row]), SUMMARY_DIR / "expanded_boundary_summary.csv")

            print(f"  └─ {sub_name} Done in {total_sub_time:.2f}s | Best ROC: {best_auc:.3f} (w={best_pair_roc[0]}, λ={best_pair_roc[1]}) | Best AP: {best_ap:.3f} (w={best_pair_ap[0]}, λ={best_pair_ap[1]})")

        if len(family_runtimes) > 0:
            pd.DataFrame(family_runtimes).to_csv(family_dir / f"{family_name}_runtime.csv", index=False)

    print("\n" + "=" * 90)
    print("✅ Expanded Boundary Benchmark Completed Successfully!")
    print(f"📁 Reports saved in: {RESULTS_BASE_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    subs = args if len(args) > 0 else None
    run_expanded_benchmark(target_subsets=subs, force_rerun=force)