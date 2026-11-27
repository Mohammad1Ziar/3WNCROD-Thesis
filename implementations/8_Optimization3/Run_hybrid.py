"""
Hybrid Memetic (pymoo Global Search + Grid Coordinate Polish) benchmark runner for 3WNCROD.

Idea under test (8_Optimization3)
---------------------------------
Combines two complementary optimization paradigms into a two-phase memetic pipeline:

Phase 1 — Global Exploration (Continuous via pymoo):
    Runs a population-based metaheuristic (DE/rand/1/bin or CMA-ES with tolfun=0)
    directly in the *continuous* box [0.1, 3.0]^m_active. Keeping Phase 1
    continuous avoids piecewise-constant grid plateaus that cause premature
    stagnation, allowing the optimizer to learn feature interactions and locate
    the best basin of attraction.

Phase 2 — Local Exploitation (Coordinate Polish on 0.1 Grid):
    Snaps the best continuous vector from Phase 1 onto the standard 0.1-step
    grid {0.1, 0.2, ..., 3.0} (also comparing against the all-ones baseline),
    then performs up to ``polish_passes`` of greedy coordinate ascent with early
    stopping (stops immediately at 1.0 or when a full streak of active features
    yields no improvement).

Outputs:
    result/8_Optimization3/Table1_Performance_Comparison_<subsets>.csv
    result/8_Optimization3/Table2_RunningTime_Comparison_<subsets>.csv

Usage from project root:
    python implementations\\8_Optimization3\\Run_hybrid.py A H MA
    python implementations\\8_Optimization3\\Run_hybrid.py ALL
    python implementations\\8_Optimization3\\Run_hybrid.py A --global-algo CMAES --global-evals 250 --polish-passes 2
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ---------------------------------------------------------------------------
# CONFIGURATION: Key Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_GLOBAL_ALGO = "DE"       # "DE" or "CMAES"
DEFAULT_GLOBAL_EVALS = 300       # Phase 1 evaluation budget per metric
DEFAULT_POLISH_PASSES = 2        # Phase 2 max coordinate-ascent passes
DEFAULT_SEED = 42

LAMBDA_MIN = 0.1
LAMBDA_MAX = 3.0
LAMBDA_GRID = np.round(np.arange(LAMBDA_MIN, LAMBDA_MAX + 0.05, 0.1), 2)

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
CORE_PATH = Path(__file__).resolve().with_name("wncrod_hybrid.py")
_CORE_SPEC = importlib.util.spec_from_file_location("wncrod_hybrid", CORE_PATH)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load wncrod_hybrid from {CORE_PATH}")
_CORE_MODULE = importlib.util.module_from_spec(_CORE_SPEC)
_CORE_SPEC.loader.exec_module(_CORE_MODULE)
WNCROD_featurewise = _CORE_MODULE.WNCROD_featurewise

# Destination folder for generated tables
BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "8_Optimization3"
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


def snap_to_grid(vec: np.ndarray) -> np.ndarray:
    """Snap continuous lambda values to the nearest 0.1-step grid point."""
    x = np.clip(np.asarray(vec, dtype=float), LAMBDA_MIN, LAMBDA_MAX)
    idx = np.clip(np.rint((x - LAMBDA_MIN) / 0.1), 0, len(LAMBDA_GRID) - 1).astype(int)
    return LAMBDA_GRID[idx]


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
# Pymoo Problem Definition
# ---------------------------------------------------------------------------
class _WNCRODProblem(ElementwiseProblem):
    """Continuous single-objective pymoo problem for 3WNCROD lambda search."""

    def __init__(self, data: np.ndarray, X_tem: np.ndarray, y: np.ndarray,
                 metric_key: str, active: np.ndarray, subset_name: str,
                 metric_label: str, max_evals: int):
        self._X_data = data
        self._X_tem = X_tem
        self._y = y
        self._metric_key = metric_key
        self._active = active
        self._m_total = data.shape[1]
        self._subset_name = subset_name
        self._metric_label = metric_label
        self._max_evals = max_evals

        self.eval_count = 0
        self.best_score = -1.0
        self.best_active_vec = np.ones(len(active), dtype=float)

        super().__init__(
            n_var=len(active),
            n_obj=1,
            xl=np.full(len(active), LAMBDA_MIN, dtype=float),
            xu=np.full(len(active), LAMBDA_MAX, dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        if self.best_score >= 1.0 or self.eval_count >= self._max_evals:
            # Fast no-op return if budget exhausted or perfect score already found
            out["F"] = -self.best_score
            return

        self.eval_count += 1
        full_vec = np.ones(self._m_total, dtype=float)
        full_vec[self._active] = np.clip(x, LAMBDA_MIN, LAMBDA_MAX)

        scores = WNCROD_featurewise(self._X_data, self._X_tem, full_vec)
        metrics = evaluate_predictions(y_true=self._y, scores=scores, X_tem=self._X_tem)
        val = float(metrics[self._metric_key])

        if val > self.best_score:
            self.best_score = val
            self.best_active_vec = np.clip(x, LAMBDA_MIN, LAMBDA_MAX).copy()
            print(f"      [{self._subset_name}/{self._metric_label}] Phase 1 Eval "
                  f"{self.eval_count:3d}/{self._max_evals} -> New Best = {val:.4f}", flush=True)

        out["F"] = -val


# ---------------------------------------------------------------------------
# Hybrid Memetic Search Routine
# ---------------------------------------------------------------------------
def hybrid_lambda_search(data, X_tem, y, metric_key, active, subset_name,
                         metric_label, global_algo="DE", global_evals=200,
                         polish_passes=2, seed=42):
    """Two-phase Hybrid Memetic optimization:
    Phase 1: Continuous global search via pymoo (DE or CMA-ES).
    Phase 2: Multi-pass greedy coordinate polish on the 0.1-step grid.
    """
    m = data.shape[1]
    d = len(active)
    total_evals = 0

    def eval_full(full_vec: np.ndarray) -> float:
        nonlocal total_evals
        total_evals += 1
        scores = WNCROD_featurewise(data, X_tem, full_vec)
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        return float(metrics[metric_key])

    # 0. Evaluate neutral all-ones baseline first
    base_vec = np.ones(m, dtype=float)
    base_score = eval_full(base_vec)
    print(f"      [{subset_name}/{metric_label}] Baseline (all-1s) = {base_score:.4f}", flush=True)
    if base_score >= 1.0:
        return base_vec, base_score, total_evals

    # 1. Phase 1: Continuous Global Exploration with pymoo
    problem = _WNCRODProblem(
        data=data, X_tem=X_tem, y=y, metric_key=metric_key, active=active,
        subset_name=subset_name, metric_label=metric_label, max_evals=global_evals,
    )

    if global_algo == "CMAES":
        pop_size = max(8, 4 + int(3 * np.log(max(d, 2))))
        algorithm = CMAES(
            x0=np.ones(d, dtype=float),
            sigma=0.5,
            pop_size=pop_size,
            tolfun=0.0,
            tolx=0.0,
            restarts=0,
        )
    else:  # DE
        pop_size = max(12, min(40, 4 * d))
        algorithm = DE(
            pop_size=pop_size,
            variant="DE/rand/1/bin",
            F=0.7,
            CR=0.9,
        )

    termination = get_termination("n_eval", global_evals)
    minimize(problem, algorithm, termination, seed=seed, verbose=False)
    total_evals += problem.eval_count

    if problem.best_score >= 1.0:
        best_vec = np.ones(m, dtype=float)
        best_vec[active] = snap_to_grid(problem.best_active_vec)
        snapped_score = eval_full(best_vec)
        if snapped_score >= 1.0:
            return best_vec, snapped_score, total_evals
        # If snapping reduced from 1.0, keep continuous or polish from snapped
        cur_vec = best_vec
        cur_best = snapped_score
    else:
        # Snap Phase 1 best to 0.1 grid and compare against all-ones baseline
        snapped_active = snap_to_grid(problem.best_active_vec)
        snapped_vec = np.ones(m, dtype=float)
        snapped_vec[active] = snapped_active
        snapped_score = eval_full(snapped_vec)

        if snapped_score >= base_score:
            cur_vec = snapped_vec
            cur_best = snapped_score
        else:
            cur_vec = base_vec
            cur_best = base_score

    print(f"      [{subset_name}/{metric_label}] Phase 1 Complete -> "
          f"Snapped Best = {cur_best:.4f} (starting Phase 2 polish)", flush=True)

    # 2. Phase 2: Local Exploitation (Coordinate Polish on 0.1 grid)
    no_improve_streak = 0
    for p in range(1, polish_passes + 1):
        improved_in_pass = False
        for pos, j in enumerate(active):
            current_val = float(cur_vec[j])
            best_c, best_v = current_val, cur_best

            for c in LAMBDA_GRID:
                if abs(c - current_val) < 1e-9:
                    continue
                trial_vec = cur_vec.copy()
                trial_vec[j] = float(c)
                val = eval_full(trial_vec)
                if val > best_v:
                    best_v, best_c = val, float(c)
                if val >= 1.0:
                    break

            if best_v > cur_best:
                cur_vec[j] = best_c
                cur_best = best_v
                improved_in_pass = True
                no_improve_streak = 0
                print(f"      [{subset_name}/{metric_label}] Polish P{p} attr {pos + 1}/{d} "
                      f"(c{j + 1}) -> New Best = {cur_best:.4f}", flush=True)
            else:
                no_improve_streak += 1

            if cur_best >= 1.0:
                print(f"      [{subset_name}/{metric_label}] Perfect score 1.0 reached! Stopping early.", flush=True)
                return cur_vec, cur_best, total_evals

            if no_improve_streak >= d:
                print(f"      [{subset_name}/{metric_label}] Converged (no improvement over {d} attrs).", flush=True)
                return cur_vec, cur_best, total_evals

        if not improved_in_pass:
            break

    return cur_vec, cur_best, total_evals


# ---------------------------------------------------------------------------
# Main Benchmark Pipeline
# ---------------------------------------------------------------------------
def run_hybrid_pipeline(dataset_names: list[str], global_algo: str = DEFAULT_GLOBAL_ALGO,
                        global_evals: int = DEFAULT_GLOBAL_EVALS,
                        polish_passes: int = DEFAULT_POLISH_PASSES,
                        seed: int = DEFAULT_SEED) -> None:
    if not dataset_names:
        raise ValueError("Provide at least one subset name.")
    unknown = sorted(set(dataset_names) - PAPER_RESULTS.keys())
    if unknown:
        raise ValueError(f"Unknown subset(s): {', '.join(unknown)}")
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Subset names must not be repeated.")

    print("=" * 95)
    print(f"Running 3WNCROD + Hybrid Memetic (pymoo {global_algo} + Coordinate Polish)")
    print(f"Selected subsets  : {', '.join(dataset_names)}")
    print(f"Output directory  : {BASE_RESULTS_DIR}")
    print(f"Phase 1 Global    : {global_algo} ({global_evals} evals, continuous)")
    print(f"Phase 2 Polish    : Coordinate Ascent (max {polish_passes} passes on 0.1 grid)")
    print(f"Random Seed       : {seed}")
    print("=" * 95)

    table1_rows = []
    table2_rows = []

    for sub_name in PAPER_RESULTS.keys():
        paper_auc, paper_ap, paper_time = PAPER_RESULTS[sub_name]

        if sub_name not in dataset_names:
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

                lam_ap, _, runs_ap = hybrid_lambda_search(
                    data_raw, X_tem, y, "ap", active, sub_name, "AP",
                    global_algo=global_algo, global_evals=global_evals,
                    polish_passes=polish_passes, seed=seed)

                lam_roc, _, runs_roc = hybrid_lambda_search(
                    data_raw, X_tem, y, "auroc", active, sub_name, "ROC",
                    global_algo=global_algo, global_evals=global_evals,
                    polish_passes=polish_passes, seed=seed)

                scores_ap, time_best_ap = timed_run(data_raw, X_tem, lam_ap)
                scores_roc, time_best_roc = timed_run(data_raw, X_tem, lam_roc)

                total_search_time = time.perf_counter() - t_start
                metrics_ap = evaluate_predictions(y_true=y, scores=scores_ap, X_tem=X_tem)
                metrics_roc = evaluate_predictions(y_true=y, scores=scores_roc, X_tem=X_tem)

                best_ap, best_auc = metrics_ap["ap"], metrics_roc["auroc"]
                best_lam_ap = format_lambda_vector(lam_ap)
                best_lam_auc = format_lambda_vector(lam_roc)
                note = f"{len(active)} active attributes, {runs_ap + runs_roc} evaluations total"

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

    shown_t1 = df_table1_export.copy()
    for col in ["Best_Lambda_AP", "Best_Lambda_ROC"]:
        shown_t1[col] = shown_t1[col].apply(
            lambda v: v if not v or v == "-" else format_lambda_vector(
                [float(x) for x in v.strip("[]").split(",")], max_items=8))

    print("\n" + "=" * 95)
    print(f"TABLE 1: PERFORMANCE COMPARISON (3WNCROD + Hybrid {global_algo} & Polish)")
    print("=" * 95)
    print(shown_t1.to_string(index=False))

    print("\n" + "=" * 95)
    print("TABLE 2: RUNNING TIME COMPARISON (Seconds)")
    print("=" * 95)
    print(df_table2_export.to_string(index=False))

    print("\n" + "=" * 95)
    print("Hybrid Memetic benchmark run completed successfully.")
    print(f"Results saved to:\n  - {table1_csv_path}\n  - {table2_csv_path}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Hybrid Memetic (pymoo global + coordinate polish) optimization."
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
        "--global-algo",
        type=str.upper,
        choices=["DE", "CMAES"],
        default=DEFAULT_GLOBAL_ALGO,
        help=f"Phase 1 pymoo global algorithm (default: {DEFAULT_GLOBAL_ALGO}).",
    )
    parser.add_argument(
        "--global-evals",
        type=int,
        default=DEFAULT_GLOBAL_EVALS,
        help=f"Phase 1 max evaluations per metric (default: {DEFAULT_GLOBAL_EVALS}).",
    )
    parser.add_argument(
        "--polish-passes",
        type=int,
        default=DEFAULT_POLISH_PASSES,
        help=f"Phase 2 max coordinate polish passes (default: {DEFAULT_POLISH_PASSES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )

    args = parser.parse_args()
    selected = expand_dataset_arguments(args.datasets)
    run_hybrid_pipeline(
        selected,
        global_algo=args.global_algo,
        global_evals=args.global_evals,
        polish_passes=args.polish_passes,
        seed=args.seed,
    )
