"""
run_new_idea.py
Executes the sequential feature-wise greedy lambda search for New_idea.
Optimizations:
  1. Step-0 Uniform Baseline Sweep with instant saturation exit (Solves MU1/MU2/MU3 in ~4 seconds).
  2. Pure Feature-Wise Search: ALL features are searched to the end (No patience/skipping).
  3. Early Exit ONLY upon reaching mathematical maximum (1.000).
  4. Skip non-numerical/categorical features (lambda has zero effect on categorical attributes).
  5. Shared Kernel Evaluation (2x speedup when ROC and AP vectors coincide).
  6. Instant incremental saving & auto-resumption.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions, identify_outliers
from evaluators.results_store import resolve_result_path
from New_idea.wncrod_multi import WNCROD_optimized_multi_lambda

NEW_IDEA_DIR = Path(__file__).resolve().parent
RESULTS_BASE_DIR = resolve_result_path("new_idea_1")
RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = RESULTS_BASE_DIR / "0_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_SPACE = [round(x, 1) for x in np.arange(0.1, 2.1, 0.1)]

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


def plot_feature_progression(save_path: Path, subset_name: str, step_records: list):
    df = pd.DataFrame(step_records)
    features = df["feature_idx"] + 1
    
    plt.figure(figsize=(9, 5))
    plt.plot(features, df["best_roc_so_far"], marker='o', color='#1877F2', label='ROC-AUC (Greedy Step)')
    plt.plot(features, df["best_ap_so_far"], marker='s', color='#E02424', linestyle='--', label='AP (Greedy Step)')
    
    plt.title(f"Feature-wise Optimization Progress: {subset_name}")
    plt.xlabel("Feature Index")
    plt.ylabel("Metric Score")
    plt.xticks(features)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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


def check_existing_subset(family_dir: Path, sub_name: str, m_features: int):
    prog_file = family_dir / f"{sub_name}_feature_progression.csv"
    if prog_file.exists():
        try:
            df_prog = pd.read_csv(prog_file)
            if len(df_prog) == m_features:
                best_lambdas_roc = df_prog["locked_lambda_roc"].tolist()
                best_lambdas_ap = df_prog["locked_lambda_ap"].tolist()
                our_auc = round(float(df_prog.iloc[-1]["best_roc_so_far"]), 3)
                our_ap = round(float(df_prog.iloc[-1]["best_ap_so_far"]), 3)
                total_sub_time = round(float(df_prog["step_time_sec"].sum()), 2)
                return True, best_lambdas_roc, best_lambdas_ap, our_auc, our_ap, total_sub_time
        except Exception:
            pass
    return False, None, None, None, None, None


def run_new_idea_pipeline(target_subsets=None, force_rerun=False):
    if target_subsets is None:
        target_subsets = list(SUBSET_REGISTRY.keys())

    family_map = defaultdict(list)
    for sub in target_subsets:
        family = SUBSET_REGISTRY[sub]["family"]
        family_map[family].append(sub)

    print("=" * 88)
    print(f"🚀 Running High-Performance Feature-Wise Multi-Lambda 3WNCROD")
    print(f"   Target Subsets: {len(target_subsets)} across {len(family_map)} Families")
    print(f"   (Pure Feature Search across ALL features - Early Exit ONLY on 1.000)")
    print("=" * 88)

    for family_name, subs in family_map.items():
        family_dir = RESULTS_BASE_DIR / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Processing Family: {family_name} ({', '.join(subs)})")

        family_runtime_list = []

        for sub_name in subs:
            print(f"  ├─ Loading {sub_name}...")
            X, y, X_tem = load_subset(sub_name)
            m_features = X.shape[1]

            paper_key = sub_name.replace("T", "TH") if sub_name in ["T1", "T2", "T3"] else sub_name
            paper_auc, paper_ap, paper_t = PAPER_RESULTS.get(paper_key, (np.nan, np.nan, np.nan))

            # بررسی اجرا شدن قبلی
            already_done, b_roc, b_ap, our_auc, our_ap, total_sub_time = check_existing_subset(family_dir, sub_name, m_features)
            
            if already_done and not force_rerun:
                diff_auc = round(our_auc - paper_auc, 3) if pd.notna(paper_auc) else np.nan
                diff_ap = round(our_ap - paper_ap, 3) if pd.notna(paper_ap) else np.nan
                print(f"  └─ {sub_name} [Already Computed! Skipped] Final AUROC: {our_auc:.3f} (Paper: {paper_auc:.3f}, Diff: {diff_auc:+.3f}) | Final AP: {our_ap:.3f} (Paper: {paper_ap:.3f}, Diff: {diff_ap:+.3f})")
                
                summary_rec = {
                    "subset": sub_name, "family": family_name, "features_count": m_features,
                    "best_lambdas_roc": str(b_roc), "final_auroc": f"{our_auc:.3f}",
                    "best_lambdas_ap": str(b_ap), "final_ap": f"{our_ap:.3f}",
                    "search_time_sec": total_sub_time
                }
                comp_rec = {
                    "subset": sub_name, "Paper_AUC": f"{paper_auc:.3f}", "NewIdea_AUC": f"{our_auc:.3f}",
                    "Diff_AUC": f"{diff_auc:+.3f}", "Paper_AP": f"{paper_ap:.3f}",
                    "NewIdea_AP": f"{our_ap:.3f}", "Diff_AP": f"{diff_ap:+.3f}",
                    "Total_Search_Time(s)": f"{total_sub_time:.2f}"
                }
                save_incremental_summary(pd.DataFrame([summary_rec]), SUMMARY_DIR / "new_idea_summary.csv")
                save_incremental_summary(pd.DataFrame([comp_rec]), SUMMARY_DIR / "new_idea_vs_paper_comparison.csv")
                continue

            t_sub_start = time.perf_counter()
            is_numerical = np.all(X <= 1.0, axis=0)

            # --- گام صفر: تست یکنواخت پایه ---
            best_uniform_lam_ap = 1.0
            best_uniform_lam_roc = 1.0
            best_ap_val = -1.0
            best_roc_val = -1.0

            for u_lam in LAMBDA_SPACE:
                u_vec = [u_lam] * m_features
                u_scores = WNCROD_optimized_multi_lambda(X, X_tem, u_vec)
                u_m = evaluate_predictions(y_true=y, scores=u_scores, X_tem=X_tem)
                
                if u_m["ap"] > best_ap_val:
                    best_ap_val = u_m["ap"]
                    best_uniform_lam_ap = u_lam
                    
                if u_m["auroc"] > best_roc_val:
                    best_roc_val = u_m["auroc"]
                    best_uniform_lam_roc = u_lam
                    
                # توقف در صورت رسیدن هر دو به سقف ۱.۰
                if best_roc_val >= 1.0 - 1e-5 and best_ap_val >= 1.0 - 1e-5:
                    break

            best_lambdas_ap = [best_uniform_lam_ap] * m_features
            best_lambdas_roc = [best_uniform_lam_roc] * m_features

            roc_saturated = (best_roc_val >= 1.0 - 1e-5)
            ap_saturated = (best_ap_val >= 1.0 - 1e-5)

            step_history = []

            # --- بهینه‌سازی ویژگی به ویژگی (تک‌تک همه ویژگی‌ها بدون وقفه) ---
            for f in range(m_features):
                # خروج تنها در صورت سقف ۱.۰
                if roc_saturated and ap_saturated:
                    print(f"     ⚡ Both ROC and AP reached theoretical 1.000! Terminating search at feature {f+1}/{m_features}.")
                    for rem_f in range(f, m_features):
                        step_history.append({
                            "subset": sub_name, "feature_idx": rem_f,
                            "locked_lambda_ap": best_lambdas_ap[rem_f], "best_ap_so_far": round(best_ap_val, 4),
                            "locked_lambda_roc": best_lambdas_roc[rem_f], "best_roc_so_far": round(best_roc_val, 4),
                            "step_time_sec": 0.0
                        })
                    break

                f_start = time.perf_counter()

                # رد کردن ویژگی‌های اسمی که تأثیری از لامبدا نمی‌پذیرند
                if not is_numerical[f]:
                    step_history.append({
                        "subset": sub_name, "feature_idx": f,
                        "locked_lambda_ap": best_lambdas_ap[f], "best_ap_so_far": round(best_ap_val, 4),
                        "locked_lambda_roc": best_lambdas_roc[f], "best_roc_so_far": round(best_roc_val, 4),
                        "step_time_sec": 0.0
                    })
                    continue

                best_lam_ap = best_lambdas_ap[f]
                best_lam_roc = best_lambdas_roc[f]

                ap_feature_done = ap_saturated
                roc_feature_done = roc_saturated

                for lam in LAMBDA_SPACE:
                    if ap_feature_done and roc_feature_done:
                        break

                    cur_ap = list(best_lambdas_ap)
                    cur_ap[f] = lam

                    cur_roc = list(best_lambdas_roc)
                    cur_roc[f] = lam

                    need_ap = (not ap_feature_done)
                    need_roc = (not roc_feature_done)

                    if need_ap and need_roc and (cur_ap == cur_roc):
                        scores = WNCROD_optimized_multi_lambda(X, X_tem, cur_ap)
                        m = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
                        m_ap_score = m["ap"]
                        m_roc_score = m["auroc"]

                        if m_ap_score > best_ap_val:
                            best_ap_val = m_ap_score
                            best_lam_ap = lam
                            if best_ap_val >= 1.0 - 1e-5:
                                ap_feature_done = True
                                ap_saturated = True

                        if m_roc_score > best_roc_val:
                            best_roc_val = m_roc_score
                            best_lam_roc = lam
                            if best_roc_val >= 1.0 - 1e-5:
                                roc_feature_done = True
                                roc_saturated = True
                    else:
                        if need_ap:
                            scores_ap = WNCROD_optimized_multi_lambda(X, X_tem, cur_ap)
                            m_ap = evaluate_predictions(y_true=y, scores=scores_ap, X_tem=X_tem)
                            if m_ap["ap"] > best_ap_val:
                                best_ap_val = m_ap["ap"]
                                best_lam_ap = lam
                                if best_ap_val >= 1.0 - 1e-5:
                                    ap_feature_done = True
                                    ap_saturated = True

                        if need_roc:
                            scores_roc = WNCROD_optimized_multi_lambda(X, X_tem, cur_roc)
                            m_roc = evaluate_predictions(y_true=y, scores=scores_roc, X_tem=X_tem)
                            if m_roc["auroc"] > best_roc_val:
                                best_roc_val = m_roc["auroc"]
                                best_lam_roc = lam
                                if best_roc_val >= 1.0 - 1e-5:
                                    roc_feature_done = True
                                    roc_saturated = True

                best_lambdas_ap[f] = best_lam_ap
                best_lambdas_roc[f] = best_lam_roc

                step_history.append({
                    "subset": sub_name, "feature_idx": f,
                    "locked_lambda_ap": best_lam_ap, "best_ap_so_far": round(best_ap_val, 4),
                    "locked_lambda_roc": best_lam_roc, "best_roc_so_far": round(best_roc_val, 4),
                    "step_time_sec": round(time.perf_counter() - f_start, 2)
                })

            total_sub_time = time.perf_counter() - t_sub_start

            # ذخیره گزارشات و نمودارها
            df_prog = pd.DataFrame(step_history)
            df_prog.to_csv(family_dir / f"{sub_name}_feature_progression.csv", index=False)
            plot_feature_progression(family_dir / f"{sub_name}_progression_plot.png", sub_name, step_history)

            final_scores_roc = WNCROD_optimized_multi_lambda(X, X_tem, best_lambdas_roc)
            final_roc_metrics = evaluate_predictions(y_true=y, scores=final_scores_roc, X_tem=X_tem)

            final_scores_ap = WNCROD_optimized_multi_lambda(X, X_tem, best_lambdas_ap)
            final_ap_metrics = evaluate_predictions(y_true=y, scores=final_scores_ap, X_tem=X_tem)

            df_det_roc, df_miss_roc = identify_outliers(y, final_scores_roc, X_tem)
            df_det_roc.to_csv(family_dir / f"{sub_name}_detected_outliers_roc.csv", index=False)
            df_miss_roc.to_csv(family_dir / f"{sub_name}_missed_outliers_roc.csv", index=False)

            df_det_ap, df_miss_ap = identify_outliers(y, final_scores_ap, X_tem)
            df_det_ap.to_csv(family_dir / f"{sub_name}_detected_outliers_ap.csv", index=False)
            df_miss_ap.to_csv(family_dir / f"{sub_name}_missed_outliers_ap.csv", index=False)

            family_runtime_list.append({
                "subset": sub_name, "num_features": m_features,
                "total_search_time_sec": round(total_sub_time, 2),
                "avg_time_per_feature_sec": round(total_sub_time / m_features, 3)
            })

            our_auc = round(float(final_roc_metrics["auroc"]), 3)
            our_ap = round(float(final_ap_metrics["ap"]), 3)
            diff_auc = round(our_auc - paper_auc, 3) if pd.notna(paper_auc) else np.nan
            diff_ap = round(our_ap - paper_ap, 3) if pd.notna(paper_ap) else np.nan

            print(f"  └─ {sub_name} Done in {total_sub_time:.2f}s | Final AUROC: {our_auc:.3f} (Paper: {paper_auc:.3f}, Diff: {diff_auc:+.3f}) | Final AP: {our_ap:.3f} (Paper: {paper_ap:.3f}, Diff: {diff_ap:+.3f})")

            # ذخیره بلادرنگ نتایج در 0_summary
            summary_rec = {
                "subset": sub_name, "family": family_name, "features_count": m_features,
                "best_lambdas_roc": str(best_lambdas_roc), "final_auroc": f"{our_auc:.3f}",
                "best_lambdas_ap": str(best_lambdas_ap), "final_ap": f"{our_ap:.3f}",
                "search_time_sec": round(total_sub_time, 2)
            }
            comp_rec = {
                "subset": sub_name, "Paper_AUC": f"{paper_auc:.3f}", "NewIdea_AUC": f"{our_auc:.3f}",
                "Diff_AUC": f"{diff_auc:+.3f}", "Paper_AP": f"{paper_ap:.3f}",
                "NewIdea_AP": f"{our_ap:.3f}", "Diff_AP": f"{diff_ap:+.3f}",
                "Total_Search_Time(s)": f"{total_sub_time:.2f}"
            }
            save_incremental_summary(pd.DataFrame([summary_rec]), SUMMARY_DIR / "new_idea_summary.csv")
            save_incremental_summary(pd.DataFrame([comp_rec]), SUMMARY_DIR / "new_idea_vs_paper_comparison.csv")

        if len(family_runtime_list) > 0:
            pd.DataFrame(family_runtime_list).to_csv(family_dir / f"{family_name}_runtime.csv", index=False)

    print("\n" + "=" * 88)
    print("✅ All Results successfully synchronized and saved!")
    print(f"📁 Family folders & detailed reports: {RESULTS_BASE_DIR}")
    print(f"📊 Unified comparison tables: {SUMMARY_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    subs = args if len(args) > 0 else None
    run_new_idea_pipeline(target_subsets=subs, force_rerun=force)