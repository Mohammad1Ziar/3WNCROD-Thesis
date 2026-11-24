"""
Bayesian Optimization (TPE via Optuna) benchmark runner for the 3WNCROD algorithm.

Idea under test (6_Optimization1)
---------------------------------
Replaces greedy coordinate ascent with Bayesian Optimization using the Tree-structured
Parzen Estimator (TPE) algorithm provided by Optuna.

Key Advantages:
1. Joint Multi-Attribute Exploration: Captures non-linear feature interactions
   rather than optimizing each coordinate in isolation.
2. Sample-Efficient: Typically converges within 60-120 evaluations per metric,
   significantly faster than multi-pass coordinate ascent for large datasets.
3. Early Stopping on Perfect Score (1.0): Stops immediately when AP or ROC reaches 1.0.
4. Baseline Warm-Start: Enqueues the neutral lambda vector (all ones) as trial 0.

Outputs:
    result/6_Optimization1/Table1_Performance_Comparison_<subsets>.csv
    result/6_Optimization1/Table2_RunningTime_Comparison_<subsets>.csv

Usage from project root:
    python implementations\\6_Optimization1\\Run_optuna.py A
    python implementations\\6_Optimization1\\Run_optuna.py A TH2
    python implementations\\6_Optimization1\\Run_optuna.py ALL
    python implementations\\6_Optimization1\\Run_optuna.py A --trials 80 --seed 42
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import scipy.io

# Suppress verbose default logs from Optuna so stdout stays clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# CONFIGURATION: Key Hyperparameters
# ---------------------------------------------------------------------------
# Default number of Optuna trials per metric search (AP / ROC-AUC)
DEFAULT_TRIALS = 400

# Lambda search bounds and step size
# Set LAMBDA_STEP to 0.1 to match previous benchmark grids (0.1, 0.2, ..., 3.0).
# Set LAMBDA_STEP to None or 0.0 for pure continuous search in [0.1, 3.0].
LAMBDA_MIN = 0.1
LAMBDA_MAX = 3.0
LAMBDA_STEP = 0.1

# Random seed for reproducible TPE sampling
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Project Layout & Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions

# Core algorithm module in current directory (dynamically loaded)
CORE_PATH = Path(__file__).resolve().with_name("wncrod_optuna.py")
_CORE_SPEC = importlib.util.spec_from_file_location("wncrod_optuna", CORE_PATH)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load wncrod_optuna from {CORE_PATH}")
_CORE_MODULE = importlib.util.module_from_spec(_CORE_SPEC)
_CORE_SPEC.loader.exec_module(_CORE_MODULE)
WNCROD_featurewise = _CORE_MODULE.WNCROD_featurewise

# Destination folder for generated tables
BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "6_Optimization1"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark Metadata
# ---------------------------------------------------------------------------
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
# Helper Functions
# ---------------------------------------------------------------------------
def expand_dataset_arguments(dataset_arguments: list[str]) -> list[str]:
    """Expand main dataset codes (A, C, etc.) into individual subsets."""
    all_subsets = [subset for group in DATASET_GROUPS.values() for subset in group]
    expanded = []
    seen = set()

    for name in dataset_arguments:
        if name == "ALL":
            subsets = all_subsets
        else:
            subsets = DATASET_GROUPS.get(name, (name,))
        for subset in subsets:
            if subset not in seen:
                expanded.append(subset)
                seen.add(subset)

    return expanded


def active_attributes(data: np.ndarray) -> np.ndarray:
    """Identify indices of continuous numerical attributes where std > 0."""
    m = data.shape[1]
    ID = np.all(data <= 1.0, axis=0)
    active = np.zeros(m, dtype=bool)
    if ID.any():
        stds = np.std(data[:, ID], axis=0)
        active[ID] = stds > 0.0
    return np.flatnonzero(active)


def timed_run(data: np.ndarray, X_tem: np.ndarray, lam_vec: np.ndarray):
    """Execute algorithm once; returns (scores, elapsed_seconds)."""
    t0 = time.perf_counter()
    scores = WNCROD_featurewise(data, X_tem, lam_vec)
    return scores, time.perf_counter() - t0


def format_lambda_vector(vec, max_items=None) -> str:
    """Format lambda vector cleanly for CSV and console output."""
    if vec is None:
        return "-"
    values = [f"{v:g}" for v in np.asarray(vec, dtype=float)]
    if max_items is not None and len(values) > max_items:
        return "[" + ",".join(values[:max_items]) + f",...({len(values)} values)]"
    return "[" + ",".join(values) + "]"


# ---------------------------------------------------------------------------
# Optuna Search Routine
# ---------------------------------------------------------------------------
def optuna_lambda_search(data, X_tem, y, metric_key, active, subset_name,
                         metric_label, n_trials=100, seed=42, step=0.1):
    """Bayesian Optimization (TPE) over the per-attribute lambda vector.

    Parameters
    ----------
    data         : (n, m) feature matrix
    X_tem        : concept X indices
    y            : true anomaly labels
    metric_key   : 'ap' or 'auroc'
    active       : array of active numerical feature indices
    subset_name  : subset name for progress printing
    metric_label : 'AP' or 'ROC'
    n_trials     : max number of trials
    seed         : random seed for reproducibility
    step         : step size for grid discretization (0.1) or None for continuous

    Returns
    -------
    best_lam   : (m,) array of best lambdas
    best_value : float best metric value achieved
    n_executed : int total trials evaluated
    """
    m = data.shape[1]
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start: enqueue trial 0 with neutral lambda = 1.0 for all active features
    init_params = {f"l_{j}": 1.0 for j in active}
    study.enqueue_trial(init_params)

    best_so_far = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far
        lam_vec = np.ones(m, dtype=float)

        for j in active:
            if step and step > 0:
                lam_vec[j] = trial.suggest_float(f"l_{j}", LAMBDA_MIN, LAMBDA_MAX, step=step)
            else:
                lam_vec[j] = trial.suggest_float(f"l_{j}", LAMBDA_MIN, LAMBDA_MAX)

        scores = WNCROD_featurewise(data, X_tem, lam_vec)
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        val = float(metrics[metric_key])

        if val > best_so_far:
            best_so_far = val
            print(f"      [{subset_name}/{metric_label}] Trial {trial.number + 1:3d}/{n_trials} "
                  f"-> New Best {metric_label} = {best_so_far:.4f}", flush=True)

        # Early Stopping if perfect score achieved
        if val >= 1.0:
            print(f"      [{subset_name}/{metric_label}] Perfect score 1.0 reached! Stopping early.", flush=True)
            trial.study.stop()

        return val

    study.optimize(objective, n_trials=n_trials)

    best_lam = np.ones(m, dtype=float)
    for j in active:
        best_lam[j] = study.best_params[f"l_{j}"]

    return best_lam, study.best_value, len(study.trials)


# ---------------------------------------------------------------------------
# Main Benchmark Pipeline
# ---------------------------------------------------------------------------
def run_optuna_pipeline(dataset_names: list[str], n_trials: int = DEFAULT_TRIALS,
                        seed: int = DEFAULT_SEED, step: float = LAMBDA_STEP) -> None:
    if not dataset_names:
        raise ValueError("Provide at least one subset name.")
    unknown = sorted(set(dataset_names) - PAPER_RESULTS.keys())
    if unknown:
        raise ValueError(f"Unknown subset(s): {', '.join(unknown)}")
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Subset names must not be repeated.")

    print("=" * 95)
    print("Running 3WNCROD + Bayesian Optimization (Optuna TPE)")
    print(f"Selected subsets : {', '.join(dataset_names)}")
    print(f"Output directory : {BASE_RESULTS_DIR}")
    print(f"Max Trials / Run : {n_trials}")
    print(f"Lambda Range     : [{LAMBDA_MIN}, {LAMBDA_MAX}] (step={step if step else 'continuous'})")
    print(f"Random Seed      : {seed}")
    print("=" * 95)

    table1_rows = []
    table2_rows = []

    for sub_name in PAPER_RESULTS.keys():
        paper_auc, paper_ap, paper_time = PAPER_RESULTS[sub_name]

        if sub_name not in dataset_names:
            # Subsets not requested remain blank in output tables
            table1_rows.append({
                "Subset": sub_name,
                "Best_Lambda_AP": "",
                "Best_AP": float("nan"),
                "Paper_AP": float(paper_ap),
                "Best_Lambda_ROC": "",
                "Best_ROC_AUC": float("nan"),
                "Paper_ROC_AUC": float(paper_auc),
            })
            table2_rows.append({
                "Subset": sub_name,
                "Time_Best_AP_sec": float("nan"),
                "Time_Best_ROC_sec": float("nan"),
                "Total_Search_Time_sec": float("nan"),
                "Paper_Time_sec": float(paper_time),
            })
            continue

        print(f"\n  > [{sub_name}] loading data ...", flush=True)

        try:
            _, y, X_tem = load_subset(sub_name, validate=False)
            mat_file = SUBSET_REGISTRY[sub_name]["mat_file"]
            mat_path = PROJECT_ROOT / "baseline" / "datasets" / mat_file
            data_raw = scipy.io.loadmat(str(mat_path))["trandata"][:, :-1].astype(float)
            active = active_attributes(data_raw)

            if active.size == 0:
                # Nominal dataset (e.g. Lymphography): radius is 0, single evaluation
                lam_vec = np.ones(data_raw.shape[1], dtype=float)
                scores, elapsed = timed_run(data_raw, X_tem, lam_vec)
                metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
                best_ap, best_auc = metrics["ap"], metrics["auroc"]
                best_lam_ap = best_lam_auc = "-"
                time_best_ap = time_best_roc = elapsed
                total_search_time = elapsed
                note = "nominal attributes only (no active continuous features)"
            else:
                t_start = time.perf_counter()

                # 1. Search for Best AP
                lam_ap, val_ap, runs_ap = optuna_lambda_search(
                    data_raw, X_tem, y, "ap", active, sub_name, "AP",
                    n_trials=n_trials, seed=seed, step=step)

                # 2. Search for Best ROC-AUC
                lam_roc, val_roc, runs_roc = optuna_lambda_search(
                    data_raw, X_tem, y, "auroc", active, sub_name, "ROC",
                    n_trials=n_trials, seed=seed, step=step)

                # 3. Timed execution of optimal vectors
                scores_ap, time_best_ap = timed_run(data_raw, X_tem, lam_ap)
                scores_roc, time_best_roc = timed_run(data_raw, X_tem, lam_roc)

                total_search_time = time.perf_counter() - t_start
                metrics_ap = evaluate_predictions(y_true=y, scores=scores_ap, X_tem=X_tem)
                metrics_roc = evaluate_predictions(y_true=y, scores=scores_roc, X_tem=X_tem)

                best_ap, best_auc = metrics_ap["ap"], metrics_roc["auroc"]
                best_lam_ap = format_lambda_vector(lam_ap)
                best_lam_auc = format_lambda_vector(lam_roc)
                note = f"{len(active)} active attributes, {runs_ap + runs_roc} trials total"

            print(f"  OK [{sub_name}] {note}")
            print(f"      Best AP  = {best_ap:.4f} (Paper: {paper_ap:.3f})  "
                  f"| Best ROC = {best_auc:.4f} (Paper: {paper_auc:.3f})  "
                  f"| Search Time = {total_search_time:.2f}s")

        except Exception as exc:
            print(f"  FAILED [{sub_name}]: {type(exc).__name__}: {exc}")
            best_ap = best_auc = float("nan")
            best_lam_ap = best_lam_auc = "-"
            time_best_ap = time_best_roc = total_search_time = float("nan")

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
            "Time_Best_ROC_sec": float(time_best_roc),
            "Total_Search_Time_sec": float(total_search_time),
            "Paper_Time_sec": float(paper_time),
        })

    # ----------------------------------------------------------------------
    # Export Tables + Average Summary
    # ----------------------------------------------------------------------
    df_table1 = pd.DataFrame(table1_rows)
    df_table2 = pd.DataFrame(table2_rows)

    avg_row_t1 = pd.DataFrame([{
        "Subset": "Average",
        "Best_Lambda_AP": "",
        "Best_AP": df_table1["Best_AP"].mean(),
        "Paper_AP": df_table1["Paper_AP"].mean(),
        "Best_Lambda_ROC": "",
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

    output_suffix = "_".join(dataset_names)
    table1_csv_path = BASE_RESULTS_DIR / f"Table1_Performance_Comparison_{output_suffix}.csv"
    table2_csv_path = BASE_RESULTS_DIR / f"Table2_RunningTime_Comparison_{output_suffix}.csv"

    df_table1_export = df_table1_final.copy()
    for col in ["Best_AP", "Paper_AP", "Best_ROC_AUC", "Paper_ROC_AUC"]:
        df_table1_export[col] = df_table1_export[col].apply(
            lambda v: f"{v:.3f}" if pd.notnull(v) else "")

    df_table2_export = df_table2_final.copy()
    for col in ["Time_Best_AP_sec", "Time_Best_ROC_sec",
                "Total_Search_Time_sec", "Paper_Time_sec"]:
        df_table2_export[col] = df_table2_export[col].apply(
            lambda v: f"{v:.4f}" if pd.notnull(v) else "")

    df_table1_export.to_csv(table1_csv_path, index=False, na_rep="")
    df_table2_export.to_csv(table2_csv_path, index=False, na_rep="")

    # Shorten vector representation for readable console printout
    shown_t1 = df_table1_export.copy()
    for col in ["Best_Lambda_AP", "Best_Lambda_ROC"]:
        shown_t1[col] = shown_t1[col].apply(
            lambda v: v if not v or v == "-" else format_lambda_vector(
                [float(x) for x in v.strip("[]").split(",")], max_items=8))

    print("\n" + "=" * 95)
    print("TABLE 1: PERFORMANCE COMPARISON (3WNCROD + Optuna TPE)")
    print("=" * 95)
    print(shown_t1.to_string(index=False))

    print("\n" + "=" * 95)
    print("TABLE 2: RUNNING TIME COMPARISON (Seconds)")
    print("=" * 95)
    print(df_table2_export.to_string(index=False))

    print("\n" + "=" * 95)
    print("Optuna benchmark run completed successfully.")
    print(f"Results saved to:\n  - {table1_csv_path}\n  - {table2_csv_path}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization (Optuna TPE) on selected benchmark datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=str.upper,
        choices=sorted(set(PAPER_RESULTS) | set(DATASET_GROUPS) | {"ALL"}),
        metavar="DATASET",
        help="Main dataset codes (A, C, G, H, L, MA, M, MU, TH, W), subset names (A1, TH2), or ALL.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Number of Optuna trials per metric search (default: {DEFAULT_TRIALS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=LAMBDA_STEP,
        help=f"Discretization step for lambda (default: {LAMBDA_STEP}; set 0 for continuous).",
    )

    args = parser.parse_args()
    selected = expand_dataset_arguments(args.datasets)
    run_optuna_pipeline(selected, n_trials=args.trials, seed=args.seed, step=args.step)
