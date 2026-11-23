"""Multi-pass greedy coordinate search for feature-wise 3WNCROD lambdas.

Each AP and ROC-AUC search starts with all lambdas equal to 1. Every later
pass starts from the best vector found in the previous pass. Search stops for
a metric when its score reaches 1.0 or a complete pass makes no improvement.

Run from the project root, for example:
    python implementations\\4_FeatureWise2\\Run_multipass_featurewise.py A C
    python implementations\\4_FeatureWise2\\Run_multipass_featurewise.py ALL

Change ``NUM_SEARCH_PASSES`` below to set the maximum number of passes.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions

CORE_PATH = (
    PROJECT_ROOT
    / "implementations"
    / "3_FeatureWise"
    / "wncrod_featurewise.py"
)
CORE_SPEC = importlib.util.spec_from_file_location("wncrod_featurewise", CORE_PATH)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise ImportError(f"Unable to load 3WNCROD implementation from {CORE_PATH}")
CORE_MODULE = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(CORE_MODULE)
WNCROD_featurewise = CORE_MODULE.WNCROD_featurewise

BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "4_FeatureWise2"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Maximum full coordinate-ascent passes, configurable by editing this value.
NUM_SEARCH_PASSES = 5

# Keep the candidate grid aligned with the current FeatureWise implementation.
DEFAULT_LAMBDAS = [round(x, 1) for x in np.arange(0.1, 3.05, 0.1)]

# Official Paper (Table 6, Table 7, Table 8): (ROC-AUC, AP, time in seconds).
PAPER_RESULTS = {
    "A1": (0.732, 0.437, 15.748),
    "A2": (0.692, 0.427, 19.689),
    "A3": (0.775, 0.278, 18.799),
    "C1": (0.990, 0.800, 0.219),
    "C2": (0.999, 0.993, 0.084),
    "C3": (0.997, 0.978, 0.467),
    "G1": (0.987, 0.561, 0.831),
    "G2": (0.986, 0.543, 0.398),
    "G3": (0.996, 0.501, 0.393),
    "H1": (0.989, 0.931, 0.009),
    "H2": (0.998, 0.973, 0.012),
    "H3": (0.988, 0.817, 0.011),
    "L1": (1.000, 1.000, 0.040),
    "L2": (1.000, 1.000, 0.011),
    "L3": (1.000, 1.000, 0.013),
    "MA1": (0.893, 0.355, 45.042),
    "MA2": (0.775, 0.073, 61.293),
    "MA3": (0.788, 0.187, 53.314),
    "M1": (0.985, 0.920, 18.533),
    "M2": (0.939, 0.898, 26.294),
    "M3": (0.978, 0.753, 17.812),
    "MU1": (1.000, 1.000, 110.730),
    "MU2": (1.000, 1.000, 93.271),
    "MU3": (1.000, 1.000, 101.388),
    "TH1": (0.985, 0.716, 3.940),
    "TH2": (0.985, 0.761, 6.714),
    "TH3": (0.992, 0.852, 6.941),
    "W1": (1.000, 1.000, 0.064),
    "W2": (1.000, 1.000, 0.131),
    "W3": (0.995, 0.921, 0.061),
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


def expand_dataset_arguments(dataset_arguments: list[str]) -> list[str]:
    """Expand main dataset codes into subset names, preserving order."""
    all_subsets = [subset for group in DATASET_GROUPS.values() for subset in group]
    expanded = []
    seen = set()

    for name in dataset_arguments:
        subsets = all_subsets if name == "ALL" else DATASET_GROUPS.get(name, (name,))
        for subset in subsets:
            if subset not in seen:
                expanded.append(subset)
                seen.add(subset)
    return expanded


def active_attributes(data: np.ndarray) -> np.ndarray:
    """Return numerical, non-constant attributes whose radii depend on lambda."""
    numerical = np.all(data <= 1.0, axis=0)
    active = np.zeros(data.shape[1], dtype=bool)
    if numerical.any():
        active[numerical] = np.std(data[:, numerical], axis=0) > 0.0
    return np.flatnonzero(active)


def timed_run(
    data: np.ndarray, X_tem: np.ndarray, lam_vec: np.ndarray
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    scores = WNCROD_featurewise(data, X_tem, lam_vec)
    return scores, time.perf_counter() - start


def greedy_coordinate_search(
    data: np.ndarray,
    X_tem: np.ndarray,
    y: np.ndarray,
    candidates: list[float],
    metric_key: str,
    active: np.ndarray,
    subset_name: str,
    metric_label: str,
    max_passes: int = NUM_SEARCH_PASSES,
) -> list[dict]:
    """Run coordinate passes until perfect score or a full no-improvement streak."""
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    m = data.shape[1]
    lam_vec = np.ones(m, dtype=float)
    evaluation_count = 0
    search_start = time.perf_counter()

    def evaluate(vector: np.ndarray) -> float:
        nonlocal evaluation_count
        evaluation_count += 1
        scores = WNCROD_featurewise(data, X_tem, vector)
        metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
        return float(metrics[metric_key])

    best_value = evaluate(lam_vec)
    history = []
    if best_value >= 1.0:
        history.append(
            {
                "lambda_vector": lam_vec.copy(),
                "score": best_value,
                "search_time_sec": time.perf_counter() - search_start,
                "evaluations": evaluation_count,
            }
        )
        return history

    consecutive_no_improvement = 0
    no_improvement_limit = len(active)
    stopped_for_stagnation = False

    for pass_number in range(1, max_passes + 1):
        for position, feature_index in enumerate(active):
            current_lambda = float(lam_vec[feature_index])
            best_lambda = current_lambda
            best_feature_score = best_value

            for candidate in candidates:
                if candidate == current_lambda:
                    continue
                trial = lam_vec.copy()
                trial[feature_index] = candidate
                score = evaluate(trial)

                if score > best_feature_score:
                    best_feature_score = score
                    best_lambda = candidate
                if score >= 1.0:
                    break

            if best_lambda != current_lambda:
                lam_vec[feature_index] = best_lambda
                best_value = best_feature_score
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            print(
                f"      [{subset_name}/{metric_label} | pass "
                f"{pass_number}/{max_passes}] feature {position + 1}/"
                f"{len(active)} (c{feature_index + 1}) -> "
                f"{best_value:.4f}",
                flush=True,
            )
            if best_value >= 1.0:
                break
            if consecutive_no_improvement >= no_improvement_limit:
                stopped_for_stagnation = True
                print(
                    f"      [{subset_name}/{metric_label}] no improvement "
                    f"across {no_improvement_limit} consecutive features; stopping."
                )
                break

        history.append(
            {
                "lambda_vector": lam_vec.copy(),
                "score": best_value,
                "search_time_sec": time.perf_counter() - search_start,
                "evaluations": evaluation_count,
            }
        )

        if best_value >= 1.0:
            print(f"      [{subset_name}/{metric_label}] reached 1.0; stopping.")
            break
        if stopped_for_stagnation:
            break

    return history


def format_lambda_vector(vector: np.ndarray, max_items: int | None = None) -> str:
    values = [f"{value:g}" for value in vector]
    if max_items is not None and len(values) > max_items:
        return "[" + ",".join(values[:max_items]) + f",...({len(values)} values)]"
    return "[" + ",".join(values) + "]"


def make_empty_rows(subset_name: str) -> tuple[dict, dict]:
    paper_auc, paper_ap, paper_time = PAPER_RESULTS[subset_name]
    return (
        {
            "Subset": subset_name,
            "Best_Lambda_AP": "",
            "Best_AP": float("nan"),
            "Paper_AP": paper_ap,
            "Best_Lambda_ROC": "",
            "Best_ROC_AUC": float("nan"),
            "Paper_ROC_AUC": paper_auc,
        },
        {
            "Subset": subset_name,
            "Time_Best_AP_sec": float("nan"),
            "Time_Best_ROC_sec": float("nan"),
            "Total_Search_Time_sec": float("nan"),
            "Paper_Time_sec": paper_time,
        },
    )


def run_multipass_pipeline(
    dataset_names: list[str], max_passes: int = NUM_SEARCH_PASSES
) -> None:
    if max_passes < 1:
        raise ValueError("NUM_SEARCH_PASSES must be at least 1")
    if not dataset_names:
        raise ValueError("Provide at least one dataset or subset name.")

    unknown = sorted(set(dataset_names) - PAPER_RESULTS.keys())
    if unknown:
        raise ValueError(f"Unknown subset(s): {', '.join(unknown)}")
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Subset names must not be repeated.")

    selected_subsets = set(dataset_names)
    subset_results = {}
    max_executed_passes = 1

    print("=" * 90)
    print("3WNCROD FeatureWise multi-pass coordinate search")
    print(f"Selected subsets: {', '.join(dataset_names)}")
    print(f"Maximum search passes: {max_passes}")
    print(f"Lambda grid: {DEFAULT_LAMBDAS[0]} to {DEFAULT_LAMBDAS[-1]}")
    print("Each metric starts from all lambdas = 1; unselected table rows stay blank.")
    print("=" * 90)

    for subset_name in PAPER_RESULTS:
        if subset_name not in selected_subsets:
            continue

        paper_auc, paper_ap, paper_time = PAPER_RESULTS[subset_name]
        print(f"\n  > [{subset_name}] loading data ...", flush=True)

        try:
            _, y, X_tem = load_subset(subset_name, validate=False)
            mat_file = SUBSET_REGISTRY[subset_name]["mat_file"]
            mat_path = PROJECT_ROOT / "baseline" / "datasets" / mat_file
            data = scipy.io.loadmat(str(mat_path))["trandata"][:, :-1].astype(float)
            active = active_attributes(data)

            if active.size == 0:
                lambda_vector = np.ones(data.shape[1], dtype=float)
                start = time.perf_counter()
                scores = WNCROD_featurewise(data, X_tem, lambda_vector)
                elapsed = time.perf_counter() - start
                metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem)
                ap_history = [
                    {
                        "lambda_vector": lambda_vector,
                        "score": metrics["ap"],
                        "search_time_sec": elapsed,
                    }
                ]
                roc_history = [
                    {
                        "lambda_vector": lambda_vector,
                        "score": metrics["auroc"],
                        "search_time_sec": elapsed,
                    }
                ]
            else:
                ap_history = greedy_coordinate_search(
                    data,
                    X_tem,
                    y,
                    DEFAULT_LAMBDAS,
                    "ap",
                    active,
                    subset_name,
                    "AP",
                    max_passes,
                )
                roc_history = greedy_coordinate_search(
                    data,
                    X_tem,
                    y,
                    DEFAULT_LAMBDAS,
                    "auroc",
                    active,
                    subset_name,
                    "ROC-AUC",
                    max_passes,
                )

            pass_records = []
            extra_runtime = 0.0
            pass_count = max(len(ap_history), len(roc_history))
            for pass_index in range(pass_count):
                ap_entry = ap_history[min(pass_index, len(ap_history) - 1)]
                roc_entry = roc_history[min(pass_index, len(roc_history) - 1)]

                ap_scores, ap_runtime = timed_run(
                    data, X_tem, ap_entry["lambda_vector"]
                )
                roc_scores, roc_runtime = timed_run(
                    data, X_tem, roc_entry["lambda_vector"]
                )
                extra_runtime += ap_runtime + roc_runtime
                ap_metrics = evaluate_predictions(
                    y_true=y, scores=ap_scores, X_tem=X_tem
                )
                roc_metrics = evaluate_predictions(
                    y_true=y, scores=roc_scores, X_tem=X_tem
                )

                total_search_time = (
                    ap_entry["search_time_sec"]
                    + roc_entry["search_time_sec"]
                    + extra_runtime
                )
                pass_records.append(
                    (
                        {
                            "Subset": subset_name,
                            "Best_Lambda_AP": format_lambda_vector(
                                ap_entry["lambda_vector"]
                            ),
                            "Best_AP": float(ap_metrics["ap"]),
                            "Paper_AP": float(paper_ap),
                            "Best_Lambda_ROC": format_lambda_vector(
                                roc_entry["lambda_vector"]
                            ),
                            "Best_ROC_AUC": float(roc_metrics["auroc"]),
                            "Paper_ROC_AUC": float(paper_auc),
                        },
                        {
                            "Subset": subset_name,
                            "Time_Best_AP_sec": float(ap_runtime),
                            "Time_Best_ROC_sec": float(roc_runtime),
                            "Total_Search_Time_sec": float(total_search_time),
                            "Paper_Time_sec": float(paper_time),
                        },
                    )
                )

            subset_results[subset_name] = pass_records
            max_executed_passes = max(max_executed_passes, len(pass_records))
            print(
                f"  OK [{subset_name}] AP={pass_records[-1][0]['Best_AP']:.4f}, "
                f"ROC-AUC={pass_records[-1][0]['Best_ROC_AUC']:.4f}, "
                f"recorded passes={len(pass_records)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"  FAILED [{subset_name}]: {type(exc).__name__}: {exc}",
                flush=True,
            )
            subset_results[subset_name] = [make_empty_rows(subset_name)]

    output_suffix = "_".join(dataset_names)
    for pass_number in range(1, max_executed_passes + 1):
        table1_rows = []
        table2_rows = []
        for subset_name in PAPER_RESULTS:
            if subset_name not in selected_subsets:
                empty_t1, empty_t2 = make_empty_rows(subset_name)
                table1_rows.append(empty_t1)
                table2_rows.append(empty_t2)
                continue

            records = subset_results[subset_name]
            row_t1, row_t2 = records[min(pass_number - 1, len(records) - 1)]
            table1_rows.append(row_t1)
            table2_rows.append(row_t2)

        df_table1 = pd.DataFrame(table1_rows)
        df_table2 = pd.DataFrame(table2_rows)
        df_table1 = pd.concat(
            [
                df_table1,
                pd.DataFrame(
                    [
                        {
                            "Subset": "Average",
                            "Best_Lambda_AP": "",
                            "Best_AP": df_table1["Best_AP"].mean(),
                            "Paper_AP": df_table1["Paper_AP"].mean(),
                            "Best_Lambda_ROC": "",
                            "Best_ROC_AUC": df_table1["Best_ROC_AUC"].mean(),
                            "Paper_ROC_AUC": df_table1["Paper_ROC_AUC"].mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        df_table2 = pd.concat(
            [
                df_table2,
                pd.DataFrame(
                    [
                        {
                            "Subset": "Average",
                            "Time_Best_AP_sec": df_table2["Time_Best_AP_sec"].mean(),
                            "Time_Best_ROC_sec": df_table2["Time_Best_ROC_sec"].mean(),
                            "Total_Search_Time_sec": df_table2[
                                "Total_Search_Time_sec"
                            ].mean(),
                            "Paper_Time_sec": df_table2["Paper_Time_sec"].mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        for column in ("Best_AP", "Paper_AP", "Best_ROC_AUC", "Paper_ROC_AUC"):
            df_table1[column] = df_table1[column].apply(
                lambda value: f"{value:.3f}" if pd.notnull(value) else ""
            )
        for column in (
            "Time_Best_AP_sec",
            "Time_Best_ROC_sec",
            "Total_Search_Time_sec",
            "Paper_Time_sec",
        ):
            df_table2[column] = df_table2[column].apply(
                lambda value: f"{value:.4f}" if pd.notnull(value) else ""
            )

        table1_path = (
            BASE_RESULTS_DIR
            / f"Table1_Performance_Comparison_Pass{pass_number}_{output_suffix}.csv"
        )
        table2_path = (
            BASE_RESULTS_DIR
            / f"Table2_RunningTime_Comparison_Pass{pass_number}_{output_suffix}.csv"
        )
        df_table1.to_csv(table1_path, index=False, na_rep="")
        df_table2.to_csv(table2_path, index=False, na_rep="")
        print(f"Saved pass {pass_number}: {table1_path.name}, {table2_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-pass FeatureWise search on selected datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=str.upper,
        choices=sorted(set(PAPER_RESULTS) | set(DATASET_GROUPS) | {"ALL"}),
        metavar="DATASET",
        help="Main dataset codes, subset names, or ALL; for example A C or ALL.",
    )
    args = parser.parse_args()
    dataset_names = expand_dataset_arguments(args.datasets)
    run_multipass_pipeline(dataset_names)


if __name__ == "__main__":
    main()
