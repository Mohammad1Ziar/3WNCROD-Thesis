"""
Multi-pass Feature-wise (per-attribute lambda) benchmark runner for 3WNCROD.

Idea under test (5_FeatureWise3)
--------------------------------
Executes up to ``NUM_SEARCH_PASSES`` sequential greedy coordinate-ascent passes.
Each pass starts from the optimal lambda vector found at the end of the
previous pass.

Early Stopping & Per-Pass Output Rules
--------------------------------------
1. **Perfect Score Stop (1.0):** As soon as a metric reaches 1.0 during any
   pass, the search for that metric stops immediately; subsequent passes are
   not executed.
2. **Convergence Stop (No Improvement):** Stop as soon as the number of
   consecutive active-attribute searches without improvement reaches the
   number of active attributes. The streak continues across pass boundaries
   and resets whenever an attribute improves.
3. **Dynamic CSV Export:** Only tables up to the maximum number of passes
   actually executed across the benchmark are exported (e.g., if all subsets
   stop by Pass 3 when ``NUM_SEARCH_PASSES = 5``, only Pass 1..3 tables are
   created). Subsets that converged earlier carry forward their final best
   metrics and cumulative time so the ``Average`` row remains valid across all
   30 subsets.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# ---------------------------------------------------------------------------
# USER CONFIGURATION: Maximum number of sequential coordinate-ascent passes
# ---------------------------------------------------------------------------
# Set this manually to 2, 3, 5, etc. (must be >= 1; default is 5).
NUM_SEARCH_PASSES = 4

# ---------------------------------------------------------------------------
# project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions

# core algorithm of this idea
CORE_PATH = Path(__file__).resolve().with_name("wncrod_featurewise3.py")
_CORE_SPEC = importlib.util.spec_from_file_location("wncrod_featurewise3", CORE_PATH)
_CORE_MODULE = importlib.util.module_from_spec(_CORE_SPEC)
_CORE_SPEC.loader.exec_module(_CORE_MODULE)
WNCROD_featurewise = _CORE_MODULE.WNCROD_featurewise

# destination of the tables
BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "5_FeatureWise3"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# benchmark constants
# ---------------------------------------------------------------------------
DEFAULT_LAMBDAS = [round(x, 1) for x in np.arange(0.1, 3.05, 0.1)]

# Official Paper (Table 6, Table 7, Table 8) results: (Paper_ROC, Paper_AP, Paper_Time)
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

DATASET_GROUPS = {
    "A": ("A1", "A2", "A3"),
    "C": ("C1", "C2", "C3"),
    "G": ("G1", "G2", "G3"),
    "H": ("H1", "H2", "H3"),
    "L": ("L1", "L2", "L3"),
    "MA": ("MA1", "MA2", "MA3"),
    "M": ("M1", "M2", "M3"),
    "MU": ("MU1", "MU2", "MU3"),
    "TH": ("TH1", "TH2", "TH3"),
    "W": ("W1", "W2", "W3"),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def expand_dataset_arguments(dataset_arguments: list[str]) -> list[str]:
    """Expand dataset groups or subset names, preserving order and removing duplicates."""
    all_subsets = [subset for group in DATASET_GROUPS.values() for subset in group]
    expanded = []
    seen = set()

    for argument in dataset_arguments:
        name = argument.upper()
        if name == "ALL":
            subsets = all_subsets
        elif name in DATASET_GROUPS:
            subsets = DATASET_GROUPS[name]
        elif name in PAPER_RESULTS:
            subsets = (name,)
        else:
            valid_names = sorted((*DATASET_GROUPS.keys(), *PAPER_RESULTS.keys(), "ALL"))
            raise ValueError(
                f"Unknown dataset '{argument}'. Choose from: {', '.join(valid_names)}"
            )

        for subset in subsets:
            if subset not in seen:
                expanded.append(subset)
                seen.add(subset)

    return expanded


def active_attributes(data: np.ndarray) -> np.ndarray:
    """Indices of numerical attributes whose radius depends on lambda."""
    m = data.shape[1]
    ID = np.all(data <= 1.0, axis=0)
    active = np.zeros(m, dtype=bool)
    if ID.any():
        stds = np.std(data[:, ID], axis=0)
        active[ID] = stds > 0.0
    return np.flatnonzero(active)


def timed_run(data: np.ndarray, X_tem: np.ndarray, lam_vec: np.ndarray):
    """One execution of the algorithm; returns (scores, elapsed seconds)."""
    t0 = time.perf_counter()
    scores = WNCROD_featurewise(data, X_tem, lam_vec)
    return scores, time.perf_counter() - t0


def global_lambda_sweep(data, X_tem, y, candidates):
    """Classical sweep of the paper: one *global* lambda for every attribute."""
    m = data.shape[1]
    aps, rocs = [], []
    t0 = time.perf_counter()
    for lam in candidates:
        scores = WNCROD_featurewise(data, X_tem, np.full(m, lam))
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        aps.append(metrics["ap"])
        rocs.append(metrics["auroc"])
    return aps, rocs, time.perf_counter() - t0


def greedy_lambda_search_history(data, X_tem, y, candidates, metric_key, active,
                                 subset_name, metric_label, start_vec=None, max_passes=5):
    """Multi-pass greedy coordinate ascent with strict early stopping.

    Stops immediately when:
      1. ``best_value >= 1.0`` (maximum possible score reached).
      2. The number of consecutive active-attribute searches without
         improvement reaches the number of active attributes.
         Improvements reset the streak, including across pass boundaries.

    Returns
    -------
    history : list of dicts (length == number of passes actually executed, >= 1)
    """
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    t_start = time.perf_counter()
    m = data.shape[1]
    lam_vec = np.ones(m, dtype=float) if start_vec is None else np.array(start_vec, dtype=float)
    counter = {"runs": 0}

    def evaluate(vec):
        counter["runs"] += 1
        scores = WNCROD_featurewise(data, X_tem, vec)
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        return metrics[metric_key]

    best_value = evaluate(lam_vec)
    history = []

    # Already optimal at initial point (before even starting Pass 1)
    if best_value >= 1.0:
        elapsed = time.perf_counter() - t_start
        history.append({
            "vec": lam_vec.copy(),
            "value": best_value,
            "cum_time": elapsed,
            "cum_runs": counter["runs"],
        })
        return history

    consecutive_no_improvements = 0
    stopped_for_stagnation = False

    for pass_idx in range(1, max_passes + 1):
        for position, j in enumerate(active):
            current = float(lam_vec[j])
            best_c, best_v = current, best_value
            for c in candidates:
                if c == current:
                    continue
                trial = lam_vec.copy()
                trial[j] = c
                value = evaluate(trial)
                if value > best_v:
                    best_v, best_c = value, c
                if value >= 1.0:
                    break

            if best_c != current:
                lam_vec[j] = best_c
                best_value = best_v
                consecutive_no_improvements = 0
            else:
                consecutive_no_improvements += 1

            print(f"      [{subset_name}/{metric_label} | Pass {pass_idx}/{max_passes}] "
                  f"attr {position + 1}/{len(active)} (c{j + 1}) -> "
                  f"best {metric_label} = {best_value:.4f}", flush=True)

            if best_value >= 1.0:
                break
            if consecutive_no_improvements >= len(active):
                stopped_for_stagnation = True
                print(
                    f"      [{subset_name}/{metric_label}] "
                    f"{consecutive_no_improvements} consecutive attribute searches "
                    "without improvement — stopping search.",
                    flush=True,
                )
                break

        elapsed_now = time.perf_counter() - t_start
        history.append({
            "vec": lam_vec.copy(),
            "value": best_value,
            "cum_time": elapsed_now,
            "cum_runs": counter["runs"],
        })

        # Stop immediately if the maximum score or no-improvement streak is reached.
        if best_value >= 1.0:
            print(f"      [{subset_name}/{metric_label}] reached 1.0 at pass {pass_idx} — stopping search.", flush=True)
            break
        if stopped_for_stagnation:
            break

    return history


def format_lambda_vector(vec, max_items=None) -> str:
    """Compact textual form of a lambda vector, e.g. ``[1.8,1.0,0.9]``."""
    if vec is None:
        return "-"
    values = [f"{v:g}" for v in np.asarray(vec, dtype=float)]
    if max_items is not None and len(values) > max_items:
        return "[" + ",".join(values[:max_items]) + f",...({len(values)} values)]"
    return "[" + ",".join(values) + "]"


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------
def run_featurewise_pipeline(
    max_passes: int = NUM_SEARCH_PASSES,
    subset_names: list[str] | None = None,
):
    selected_subsets = list(PAPER_RESULTS) if subset_names is None else subset_names
    print("=" * 95)
    print(f"🚀 Running 3WNCROD+ Multi-Pass FeatureWise (max {max_passes} passes) "
          f"on {len(selected_subsets)} subset(s): {', '.join(selected_subsets)}")
    print(f"📁 Output Directory: {BASE_RESULTS_DIR}")
    print("⏹️  Early stopping active: halts on metric == 1.0 or after "
          "consecutive no-improvement searches reach the number of active attributes.")
    print("⏱️  Total_Search_Time_sec in each table is CUMULATIVE across executed passes.")
    print("=" * 95)

    # Store per-subset list of pass records: subset_results[sub_name] = [(row_t1, row_t2), ...]
    subset_results = {}
    max_executed_passes_overall = 1

    for sub_name in selected_subsets:
        paper_auc, paper_ap, paper_time = PAPER_RESULTS[sub_name]
        print(f"\n  ▶ [{sub_name}] loading data ...", flush=True)

        try:
            # ---- data ------------------------------------------------------
            _, y, X_tem = load_subset(sub_name, validate=False)
            mat_file = SUBSET_REGISTRY[sub_name]["mat_file"]
            mat_path = PROJECT_ROOT / "baseline" / "datasets" / mat_file
            data_raw = scipy.io.loadmat(str(mat_path))["trandata"][:, :-1].astype(float)
            active = active_attributes(data_raw)

            if active.size == 0:
                lam_vec = np.ones(data_raw.shape[1], dtype=float)
                scores, elapsed = timed_run(data_raw, X_tem, lam_vec)
                metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
                best_ap, best_auc = metrics["ap"], metrics["auroc"]

                row_t1 = {
                    "Subset": sub_name,
                    "Best_Lambda_AP": "-",
                    "Best_AP": float(best_ap),
                    "Paper_AP": float(paper_ap),
                    "Best_Lambda_ROC": "-",
                    "Best_ROC_AUC": float(best_auc),
                    "Paper_ROC_AUC": float(paper_auc),
                }
                row_t2 = {
                    "Subset": sub_name,
                    "Time_Best_AP_sec": float(elapsed),
                    "Time_Best_ROC_sec": float(elapsed),
                    "Total_Search_Time_sec": float(elapsed),
                    "Paper_Time_sec": float(paper_time),
                }
                subset_results[sub_name] = [(row_t1, row_t2)]
                print(f"  ✔ [{sub_name}] lambda-independent (no numerical attribute) "
                      f"| AP={best_ap:.4f}, ROC={best_auc:.4f}")
            else:
                # ---- (1) global sweep of the paper (one lambda for all) ----
                sweep_aps, sweep_rocs, t_sweep = global_lambda_sweep(
                    data_raw, X_tem, y, DEFAULT_LAMBDAS)
                idx_ap = int(np.argmax(sweep_aps))
                idx_roc = int(np.argmax(sweep_rocs))
                lam_global_ap = DEFAULT_LAMBDAS[idx_ap]
                lam_global_roc = DEFAULT_LAMBDAS[idx_roc]
                print(f"      global sweep ({t_sweep:.2f}s): best AP  = {sweep_aps[idx_ap]:.4f} "
                      f"(lambda={lam_global_ap}, paper {paper_ap:.3f}) | "
                      f"best ROC = {sweep_rocs[idx_roc]:.4f} "
                      f"(lambda={lam_global_roc}, paper {paper_auc:.3f})", flush=True)

                # ---- (2) feature-wise search from each metric's global best ----
                hist_ap = greedy_lambda_search_history(
                    data_raw, X_tem, y, DEFAULT_LAMBDAS, "ap", active,
                    sub_name, "AP", start_vec=np.full(data_raw.shape[1], lam_global_ap),
                    max_passes=max_passes)
                hist_roc = greedy_lambda_search_history(
                    data_raw, X_tem, y, DEFAULT_LAMBDAS, "auroc", active,
                    sub_name, "ROC", start_vec=np.full(data_raw.shape[1], lam_global_roc),
                    max_passes=max_passes)

                sub_passes = max(len(hist_ap), len(hist_roc))
                if sub_passes > max_executed_passes_overall:
                    max_executed_passes_overall = sub_passes

                records = []
                for p in range(sub_passes):
                    entry_ap = hist_ap[min(p, len(hist_ap) - 1)]
                    entry_roc = hist_roc[min(p, len(hist_roc) - 1)]

                    lam_ap_p = entry_ap["vec"]
                    lam_roc_p = entry_roc["vec"]

                    scores_ap, time_best_ap = timed_run(data_raw, X_tem, lam_ap_p)
                    scores_roc, time_best_roc = timed_run(data_raw, X_tem, lam_roc_p)

                    metrics_ap = evaluate_predictions(y_true=y, scores=scores_ap, X_tem=X_tem)
                    metrics_roc = evaluate_predictions(y_true=y, scores=scores_roc, X_tem=X_tem)

                    cum_search_time = (
                        t_sweep
                        + entry_ap["cum_time"]
                        + entry_roc["cum_time"]
                        + time_best_ap
                        + time_best_roc
                    )

                    row_t1 = {
                        "Subset": sub_name,
                        "Best_Lambda_AP": format_lambda_vector(lam_ap_p),
                        "Best_AP": float(metrics_ap["ap"]),
                        "Paper_AP": float(paper_ap),
                        "Best_Lambda_ROC": format_lambda_vector(lam_roc_p),
                        "Best_ROC_AUC": float(metrics_roc["auroc"]),
                        "Paper_ROC_AUC": float(paper_auc),
                    }
                    row_t2 = {
                        "Subset": sub_name,
                        "Time_Best_AP_sec": float(time_best_ap),
                        "Time_Best_ROC_sec": float(time_best_roc),
                        "Total_Search_Time_sec": float(cum_search_time),
                        "Paper_Time_sec": float(paper_time),
                    }
                    records.append((row_t1, row_t2))

                    print(f"  ✔ [{sub_name} | Pass {p + 1}/{sub_passes}] "
                          f"AP={metrics_ap['ap']:.4f} (Paper:{paper_ap:.3f}) | "
                          f"ROC={metrics_roc['auroc']:.4f} (Paper:{paper_auc:.3f}) | "
                          f"CumTime={cum_search_time:.2f}s")

                subset_results[sub_name] = records

        except Exception as exc:
            print(f"  ✘ [{sub_name}] FAILED: {type(exc).__name__}: {exc}")
            row_t1 = {
                "Subset": sub_name,
                "Best_Lambda_AP": "-",
                "Best_AP": float("nan"),
                "Paper_AP": float(paper_ap),
                "Best_Lambda_ROC": "-",
                "Best_ROC_AUC": float("nan"),
                "Paper_ROC_AUC": float(paper_auc),
            }
            row_t2 = {
                "Subset": sub_name,
                "Time_Best_AP_sec": float("nan"),
                "Time_Best_ROC_sec": float("nan"),
                "Total_Search_Time_sec": float("nan"),
                "Paper_Time_sec": float(paper_time),
            }
            subset_results[sub_name] = [(row_t1, row_t2)]

    # ----------------------------------------------------------------------
    # Export tables only for Pass 1 .. max_executed_passes_overall
    # ----------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"💾 EXPORTING TABLES FOR {max_executed_passes_overall} EXECUTED PASS(ES) "
          f"(configured max: {max_passes})")
    print("=" * 95)

    for p in range(max_executed_passes_overall):
        pass_num = p + 1
        t1_rows = []
        t2_rows = []

        for sub_name in selected_subsets:
            recs = subset_results[sub_name]
            # If this subset stopped early, carry forward its final converged result
            r1, r2 = recs[min(p, len(recs) - 1)]
            t1_rows.append(r1)
            t2_rows.append(r2)

        df_table1 = pd.DataFrame(t1_rows)
        df_table2 = pd.DataFrame(t2_rows)

        avg_row_t1 = pd.DataFrame([{
            "Subset": "Average",
            "Best_Lambda_AP": "-",
            "Best_AP": df_table1["Best_AP"].mean(),
            "Paper_AP": df_table1["Paper_AP"].mean(),
            "Best_Lambda_ROC": "-",
            "Best_ROC_AUC": df_table1["Best_ROC_AUC"].mean(),
            "Paper_ROC_AUC": df_table1["Paper_ROC_AUC"].mean(),
        }])
        df_table1_final = pd.concat([df_table1, avg_row_t1], ignore_index=True)

        avg_row_t2 = pd.DataFrame([{
            "Subset": "Average",
            "Time_Best_AP_sec": df_table2["Time_Best_AP_sec"].mean(),
            "Time_Best_ROC_sec": df_table2["Time_Best_ROC_sec"].mean(),
            "Total_Search_Time_sec": df_table2["Total_Search_Time_sec"].mean(),
            "Paper_Time_sec": df_table2["Paper_Time_sec"].mean(),
        }])
        df_table2_final = pd.concat([df_table2, avg_row_t2], ignore_index=True)

        table1_csv_path = BASE_RESULTS_DIR / f"Table1_Performance_Comparison_Pass{pass_num}.csv"
        table2_csv_path = BASE_RESULTS_DIR / f"Table2_RunningTime_Comparison_Pass{pass_num}.csv"

        df_table1_export = df_table1_final.copy()
        for col in ["Best_AP", "Paper_AP", "Best_ROC_AUC", "Paper_ROC_AUC"]:
            df_table1_export[col] = df_table1_export[col].apply(
                lambda v: f"{v:.3f}" if pd.notnull(v) else "-")

        df_table2_export = df_table2_final.copy()
        for col in ["Time_Best_AP_sec", "Time_Best_ROC_sec",
                    "Total_Search_Time_sec", "Paper_Time_sec"]:
            df_table2_export[col] = df_table2_export[col].apply(
                lambda v: f"{v:.4f}" if pd.notnull(v) else "-")

        df_table1_export.to_csv(table1_csv_path, index=False)
        df_table2_export.to_csv(table2_csv_path, index=False)

        # Console summary for this pass
        shown_t1 = df_table1_export.copy()
        for col in ["Best_Lambda_AP", "Best_Lambda_ROC"]:
            shown_t1[col] = shown_t1[col].apply(
                lambda v: v if v == "-" else format_lambda_vector(
                    [float(x) for x in v.strip("[]").split(",")], max_items=8))

        print(f"\n--- 📊 PASS {pass_num} PERFORMANCE COMPARISON ---")
        print(shown_t1.to_string(index=False))

        print(f"\n--- ⏱️ PASS {pass_num} RUNNING TIME COMPARISON (Cumulative Seconds) ---")
        print(df_table2_export.to_string(index=False))

        print(f"\n✅ Saved Pass {pass_num} files:")
        print(f"   - {table1_csv_path}")
        print(f"   - {table2_csv_path}")

    print("\n" + "=" * 95)
    print(f"🎉 Completed! Exported {max_executed_passes_overall} pass table set(s).")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-pass feature-wise 3WNCROD on selected datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["ALL"],
        metavar="DATASET",
        help=(
            "dataset group(s) (A C G H L MA M MU TH W), individual subset(s) "
            "(such as A1 or MA2), or ALL; defaults to ALL"
        ),
    )
    args = parser.parse_args()
    try:
        selected_subsets = expand_dataset_arguments(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))

    run_featurewise_pipeline(
        max_passes=NUM_SEARCH_PASSES,
        subset_names=selected_subsets,
    )