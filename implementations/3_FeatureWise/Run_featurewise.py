"""
Feature-wise (per-attribute lambda) benchmark runner for the 3WNCROD algorithm.

Idea under test
---------------
The baseline 3WNCROD uses a **single global** radius adjustment parameter
``lambda`` for every condition attribute (Eq. 8 of the paper):

    delta_j = std(c_j) / lambda

This runner replaces it by **one lambda per attribute** (the "FeatureWise"
idea of this thesis):

    delta_j = std(c_j) / lambda_j

and searches the vector ``(lambda_1, ..., lambda_m)`` with a greedy coordinate
ascent: the attributes are visited one after the other, every candidate of the
grid {0.1, 0.2, ..., 2.0} is tried for the current attribute while the other
lambdas stay fixed, and the value that maximises the metric (AP or ROC-AUC) is
kept.  As soon as a metric reaches its maximum value 1.0 the search for that
metric is stopped -- nothing can be better than 1.

Two starting points
-------------------
A greedy coordinate ascent is a *local* method, so the starting point matters.
The search is therefore run from two initial points and the better result is
kept (multi-start):

* **neutral** -- every ``lambda_j = 1`` (the plain definition of the idea);
* **global** -- every ``lambda_j`` equal to the best *global* lambda of the
  paper's own sweep (20 extra algorithm executions per subset).

Measured on the benchmark, each of the two starts wins on some subsets
(neutral: H3 -> AP 1.0000 against 0.9500, global: W3 -> AP 1.0000 against
0.8469), so keeping both guarantees that the feature-wise variant is never
worse than the neutral-start variant.  The global start is tried first
because on the subsets whose global optimum is already optimal for the
feature-wise variant (Musk, W1, W2) it reaches 1.0 immediately and the
neutral start is skipped -- otherwise the 166 attributes of Musk would cost
166 x 20 x 2 algorithm executions per subset.

The algorithm itself is the runtime-optimised 3WNCROD of
``implementations/2_Running Time Optimised`` with the per-attribute divisor
added; it lives in ``wncrod_featurewise.py`` next to this file.

Only two artefacts are written (no plots):

    result/3_FeatureWise/Table1_Performance_Comparison.csv
        Subset | Best_Lambda_AP | Best_AP | Paper_AP |
                Best_Lambda_ROC | Best_ROC_AUC | Paper_ROC_AUC
    result/3_FeatureWise/Table2_RunningTime_Comparison.csv
        Subset | Time_Best_AP_sec | Time_Best_ROC_sec |
                Total_Search_Time_sec | Paper_Time_sec

Both tables end with an "Average" row.

Run it from the project root with:

    .venv/bin/python implementations/3_FeatureWise/Run_featurewise.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# ---------------------------------------------------------------------------
# project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions

# core algorithm of this idea (optimised 3WNCROD + per-attribute lambda)
CORE_PATH = Path(__file__).resolve().with_name("wncrod_featurewise.py")
_CORE_SPEC = importlib.util.spec_from_file_location("wncrod_featurewise", CORE_PATH)
_CORE_MODULE = importlib.util.module_from_spec(_CORE_SPEC)
_CORE_SPEC.loader.exec_module(_CORE_MODULE)
WNCROD_featurewise = _CORE_MODULE.WNCROD_featurewise

# destination of the two tables
BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "3_FeatureWise"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# benchmark constants
# ---------------------------------------------------------------------------
# candidate values of every single lambda (same grid as the paper)
DEFAULT_LAMBDAS = [round(x, 1) for x in np.arange(0.1, 2.05, 0.1)]

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def active_attributes(data: np.ndarray) -> np.ndarray:
    """Indices of the attributes whose radius really depends on lambda.

    A column is *numerical* when all of its values lie in [0, 1]; only then
    Eq. (8) uses a positive radius ``std(c_j) / lambda_j``.  For a nominal
    column (values coded as distinct integers) the radius is 0 whatever
    lambda_j is, and a constant numerical column has ``std = 0`` and therefore
    a radius of 0 as well.  Searching such attributes would be a waste of
    time, so they are left out of the search.
    """
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
    """Classical sweep of the paper: one *global* lambda for every attribute.

    Returns the AP and the ROC-AUC of every candidate lambda together with
    the elapsed time of the sweep.  The best global lambda of each metric is
    used as a starting point of the feature-wise search.
    """
    m = data.shape[1]
    aps, rocs = [], []
    t0 = time.perf_counter()
    for lam in candidates:
        scores = WNCROD_featurewise(data, X_tem, np.full(m, lam))
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        aps.append(metrics["ap"])
        rocs.append(metrics["auroc"])
    return aps, rocs, time.perf_counter() - t0


def greedy_lambda_search(data, X_tem, y, candidates, metric_key, active,
                         subset_name, metric_label, start_vec=None):
    """Greedy coordinate ascent over one lambda per numerical attribute.

    The search starts from ``start_vec`` (by default the neutral vector
    ``lambda_j = 1``) and visits the attributes one after the other.  For the
    current attribute every candidate of ``candidates`` is evaluated while the
    other lambdas stay fixed; the best candidate is kept.  The search for the
    given metric stops as soon as the metric reaches 1.0, its maximum value.

    Returns
    -------
    lam_vec    : (m,) float64   the lambda vector that produced the best metric
    best_value : float          the value of the metric at that vector
    n_runs     : int            number of algorithm executions of this search
    """
    m = data.shape[1]
    lam_vec = np.ones(m, dtype=float) if start_vec is None else np.array(start_vec, dtype=float)
    counter = {"runs": 0}

    def evaluate(vec):
        counter["runs"] += 1
        scores = WNCROD_featurewise(data, X_tem, vec)
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        return metrics[metric_key]

    best_value = evaluate(lam_vec)
    if best_value >= 1.0:                      # maximum reached at once
        return lam_vec, best_value, counter["runs"]

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
            if value >= 1.0:                   # maximum reached, stop here
                break
        lam_vec[j] = best_c
        best_value = best_v
        print(f"      [{subset_name}/{metric_label}] attribute {position + 1}/{len(active)} "
              f"(c{j + 1}) -> best {metric_label} = {best_value:.4f}", flush=True)
        if best_value >= 1.0:                  # maximum reached, stop the search
            break
    return lam_vec, best_value, counter["runs"]


def multi_start_search(data, X_tem, y, candidates, metric_key, active,
                       subset_name, metric_label, warm_start):
    """Feature-wise search started from two initial points; the better wins.

    The greedy coordinate ascent is a local method: two different starting
    points can lead to two different local optima.  Measured on the benchmark
    subsets, starting from the neutral vector ``lambda = 1`` is better on some
    subsets (e.g. H3: AP 1.0000 against 0.9500) while starting from the best
    *global* lambda of the paper is better on others (e.g. W3: AP 1.0000
    against 0.8469).  Both starting points are therefore searched and the
    better result is kept, so the feature-wise variant can never be worse
    than the neutral-start variant.

    The global start is tried first: on the subsets whose global optimum is
    already optimal for the feature-wise variant (Musk, W1, W2, ...) it
    reaches the maximum 1.0 immediately and the neutral start is skipped
    altogether -- without that shortcut the 166 attributes of Musk would cost
    166 x 20 x 2 algorithm executions per subset.

    Returns
    -------
    lam_vec    : (m,) float64   the best lambda vector found
    best_value : float          the value of the metric at that vector
    n_runs     : int            number of algorithm executions of this search
    origin     : str            "global" or "neutral" (which start won)
    """
    warm_vec, warm_value, warm_runs = greedy_lambda_search(
        data, X_tem, y, candidates, metric_key, active, subset_name,
        metric_label, start_vec=warm_start)

    if warm_value >= 1.0:
        # the maximum of the metric is reached -- nothing can beat it
        return warm_vec, warm_value, warm_runs, "global"

    cold_vec, cold_value, cold_runs = greedy_lambda_search(
        data, X_tem, y, candidates, metric_key, active, subset_name,
        metric_label, start_vec=None)

    if cold_value >= 1.0:
        return cold_vec, cold_value, cold_runs + warm_runs, "neutral"
    if cold_value > warm_value:
        return cold_vec, cold_value, cold_runs + warm_runs, "neutral"
    return warm_vec, warm_value, cold_runs + warm_runs, "global"


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
def run_featurewise_pipeline():
    print("=" * 95)
    print("🚀 Running 3WNCROD+ FeatureWise (independent lambda per attribute) on all 30 subsets")
    print(f"📁 Output Directory: {BASE_RESULTS_DIR}")
    print(f"📊 Lambda grid per attribute: {len(DEFAULT_LAMBDAS)} candidates "
          f"({DEFAULT_LAMBDAS[0]} ... {DEFAULT_LAMBDAS[-1]})")
    print("⏹️  Search of a metric stops as soon as the metric reaches 1.0")
    print("🔀 Per metric the search runs from two starts (neutral lambda=1 and the best")
    print("   global lambda of the paper); the better result is kept.")
    print("=" * 95)

    table1_rows = []
    table2_rows = []

    for sub_name in PAPER_RESULTS.keys():
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
                # lambda cannot change anything on this subset -> single run
                lam_vec = np.ones(data_raw.shape[1], dtype=float)
                scores, elapsed = timed_run(data_raw, X_tem, lam_vec)
                metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
                best_ap, best_auc = metrics["ap"], metrics["auroc"]
                best_lam_ap = best_lam_auc = "-"
                time_best_ap = time_best_roc = elapsed
                total_search_time = elapsed
                n_runs = 1
                note = "lambda-independent (no numerical attribute)"
            else:
                # ---- (1) global sweep of the paper (one lambda for all) ----
                t_search = time.perf_counter()
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

                # ---- (2) feature-wise search, one metric at a time ------------
                lam_ap, val_ap, runs_ap, org_ap = multi_start_search(
                    data_raw, X_tem, y, DEFAULT_LAMBDAS, "ap", active,
                    sub_name, "AP", warm_start=np.full(data_raw.shape[1], lam_global_ap))
                lam_roc, val_roc, runs_roc, org_roc = multi_start_search(
                    data_raw, X_tem, y, DEFAULT_LAMBDAS, "auroc", active,
                    sub_name, "ROC", warm_start=np.full(data_raw.shape[1], lam_global_roc))

                # ---- (3) final timed executions with the two optimal vectors --
                scores_ap, time_best_ap = timed_run(data_raw, X_tem, lam_ap)
                scores_roc, time_best_roc = timed_run(data_raw, X_tem, lam_roc)

                total_search_time = time.perf_counter() - t_search
                n_runs = len(DEFAULT_LAMBDAS) + runs_ap + runs_roc + 2

                metrics_ap = evaluate_predictions(y_true=y, scores=scores_ap, X_tem=X_tem)
                metrics_roc = evaluate_predictions(y_true=y, scores=scores_roc, X_tem=X_tem)
                best_ap, best_auc = metrics_ap["ap"], metrics_roc["auroc"]
                best_lam_ap = format_lambda_vector(lam_ap)
                best_lam_auc = format_lambda_vector(lam_roc)
                note = (f"{len(active)} numerical attribute(s) tuned, {n_runs} runs "
                        f"(start: AP={org_ap}, ROC={org_roc})")

            print(f"  ✔ [{sub_name}] {note}")
            print(f"      Best AP  = {best_ap:.4f} (Paper: {paper_ap:.3f})  "
                  f"| Best ROC = {best_auc:.4f} (Paper: {paper_auc:.3f})  "
                  f"| search {total_search_time:.2f}s")

        except Exception as exc:                                   # keep going
            print(f"  ✘ [{sub_name}] FAILED: {type(exc).__name__}: {exc}")
            best_ap = best_auc = float("nan")
            best_lam_ap = best_lam_auc = "-"
            time_best_ap = time_best_roc = total_search_time = float("nan")

        # ---- Table 1: performance ------------------------------------------
        table1_rows.append({
            "Subset": sub_name,
            "Best_Lambda_AP": str(best_lam_ap),
            "Best_AP": float(best_ap),
            "Paper_AP": float(paper_ap),
            "Best_Lambda_ROC": str(best_lam_auc),
            "Best_ROC_AUC": float(best_auc),
            "Paper_ROC_AUC": float(paper_auc),
        })

        # ---- Table 2: running time -----------------------------------------
        table2_rows.append({
            "Subset": sub_name,
            "Time_Best_AP_sec": float(time_best_ap),
            "Time_Best_ROC_sec": float(time_best_roc),
            "Total_Search_Time_sec": float(total_search_time),
            "Paper_Time_sec": float(paper_time),
        })

    # ----------------------------------------------------------------------
    # tables + Average row
    # ----------------------------------------------------------------------
    df_table1 = pd.DataFrame(table1_rows)
    df_table2 = pd.DataFrame(table2_rows)

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

    table1_csv_path = BASE_RESULTS_DIR / "Table1_Performance_Comparison.csv"
    table2_csv_path = BASE_RESULTS_DIR / "Table2_RunningTime_Comparison.csv"

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

    # ---- console report (lambda vectors shortened for readability) ---------
    shown_t1 = df_table1_export.copy()
    for col in ["Best_Lambda_AP", "Best_Lambda_ROC"]:
        shown_t1[col] = shown_t1[col].apply(
            lambda v: v if v == "-" else format_lambda_vector(
                [float(x) for x in v.strip("[]").split(",")], max_items=8))

    print("\n" + "=" * 95)
    print("📊 TABLE 1: PERFORMANCE COMPARISON (3WNCROD+ FeatureWise)")
    print("=" * 95)
    print(shown_t1.to_string(index=False))

    print("\n" + "=" * 95)
    print("⏱️ TABLE 2: RUNNING TIME COMPARISON (Seconds)")
    print("=" * 95)
    print(df_table2_export.to_string(index=False))

    print("\n" + "=" * 95)
    print("✅ FeatureWise run completed!")
    print(f"📁 Tables saved to:\n  - {table1_csv_path}\n  - {table2_csv_path}")
    print("   (the CSV files hold the complete lambda vector of every subset;")
    print("    the console print above shortens it to the first 8 values)")
    print("=" * 95)


if __name__ == "__main__":
    run_featurewise_pipeline()
