"""Benchmark the metaheuristic lambda-search strategies for 3WNCROD+.

Every method gets the *same* evaluation budget (the number of 3WNCROD
executions), so the comparison is budget-fair rather than time-fair.  For every
subset, method and seed the run records the best metric, the grid-snapped
metric, the number of evaluations and the wall-clock time.

Usage
-----
    py -3 implementations/6_Metaheuristics/Run_metaheuristic_benchmark.py --subsets A1 G1 --budget 400
    py -3 implementations/6_Metaheuristics/Run_metaheuristic_benchmark.py --subsets ALL --budget 600 --seeds 1 2 3
    py -3 implementations/6_Metaheuristics/Run_metaheuristic_benchmark.py --methods CMA-ES DE PSO

The tables are written to ``result/6_Metaheuristics``.
"""

from __future__ import annotations

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

_CORE = Path(__file__).resolve().with_name("lambda_search.py")
_SPEC = importlib.util.spec_from_file_location("lambda_search", _CORE)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["lambda_search"] = _MODULE          # required by @dataclass
_SPEC.loader.exec_module(_MODULE)

LambdaObjective = _MODULE.LambdaObjective
METHODS = _MODULE.METHODS
KEY_METHODS = _MODULE.KEY_METHODS
active_attributes = _MODULE.active_attributes

BASE_RESULTS_DIR = PROJECT_ROOT / "result" / "6_Metaheuristics"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Groups of the 30 benchmark subsets.
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

#: Subset without any lambda-dependent attribute -- the search is trivial there.
TRIVIAL_SUBSETS = ("L1", "L2", "L3", "M1", "M2", "M3")


def expand_subsets(arguments: list[str]) -> list[str]:
    """Expand dataset groups, subset names or ALL into a list of subset names."""
    all_subsets = [s for group in DATASET_GROUPS.values() for s in group]
    expanded: list[str] = []
    seen: set[str] = set()
    for argument in arguments:
        name = argument.upper()
        if name == "ALL":
            candidates = all_subsets
        elif name in DATASET_GROUPS:
            candidates = DATASET_GROUPS[name]
        elif name in SUBSET_REGISTRY:
            candidates = (name,)
        else:
            raise ValueError(f"Unknown dataset '{argument}'.")
        for subset in candidates:
            if subset not in seen:
                expanded.append(subset)
                seen.add(subset)
    return expanded


def run_one(subset: str, method: str, metric: str, budget: int, seed: int,
            project: bool = True):
    """Run one (subset, method, metric, seed) combination."""
    _, y, X_tem = load_subset(subset, validate=False)
    mat_file = SUBSET_REGISTRY[subset]["mat_file"]
    data = scipy.io.loadmat(str(PROJECT_ROOT / "baseline" / "datasets" / mat_file))[
        "trandata"][:, :-1].astype(float)

    active = active_attributes(data)
    mode = "key" if method in KEY_METHODS else "continuous"
    obj = LambdaObjective(data, X_tem, y, metric, active, mode=mode,
                          subset=subset, project=project)

    started = time.perf_counter()
    try:
        result = METHODS[method](obj, budget, seed)
    except Exception as exc:  # a failing method must not kill the whole run
        result = _MODULE.SearchResult(
            method=method, subset=subset, metric=metric, m_active=int(active.size),
            budget=budget, evaluations=obj.n_evals, seconds=obj.seconds,
            best_value=float("nan"), snapped_value=float("nan"),
            best_lambda="-", seed=seed, status=f"{type(exc).__name__}: {exc}",
        )
    result.extra["wall_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=["ALL"],
                        help="Dataset groups (A C G H L MA M MU TH W), subset names, or ALL.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        help=f"Methods to run: {', '.join(METHODS)}")
    parser.add_argument("--metric", default="ap", choices=["ap", "auroc"],
                        help="Metric to optimise (default: ap).")
    parser.add_argument("--budget", type=int, default=400,
                        help="Evaluation budget per method and subset (default: 400).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="Random seeds (default: 1).")
    parser.add_argument("--skip-trivial", action="store_true",
                        help="Skip subsets without lambda-dependent attributes.")
    parser.add_argument("--continuous", action="store_true",
                        help="Evaluate the raw continuous lambda instead of the "
                             "0.1-step grid (measures the continuous relaxation).")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of runs executed in parallel (threads).")
    parser.add_argument("--tag", default="",
                        help="Extra suffix for the output file names.")
    args = parser.parse_args()

    subsets = expand_subsets(args.subsets)
    if args.skip_trivial:
        subsets = [s for s in subsets if s not in TRIVIAL_SUBSETS]
    methods = [m for m in args.methods if m in METHODS]
    project = not args.continuous

    print("=" * 100)
    print(f"Metaheuristic lambda-search benchmark | metric={args.metric} "
          f"budget={args.budget} | seeds={args.seeds}")
    print(f"landscape: {'grid 0.1 (projected)' if project else 'raw continuous'}")
    print(f"subsets: {', '.join(subsets)}")
    print(f"methods: {', '.join(methods)}")
    print(f"jobs: {args.jobs}")
    print("=" * 100)

    tasks = [(subset, method, seed)
             for subset in subsets for method in methods for seed in args.seeds]

    def _execute(task):
        subset, method, seed = task
        result = run_one(subset, method, args.metric, args.budget, seed,
                         project=project)
        print(f"  [{subset:4s}] {method:18s} seed={seed} "
              f"m={result.m_active:3d} evals={result.evaluations:4d} "
              f"best={result.best_value:.4f} snapped={result.snapped_value:.4f} "
              f"{result.extra.get('wall_seconds', 0.0):7.2f}s "
              f"{'' if result.status == 'ok' else result.status}", flush=True)
        return {
            "Subset": result.subset,
            "Method": result.method,
            "Metric": result.metric,
            "m_active": result.m_active,
            "Seed": result.seed,
            "Evaluations": result.evaluations,
            "Best_Metric": round(result.best_value, 4),
            "Best_Metric_Snapped": round(result.snapped_value, 4),
            "Lambda_Vector": result.best_lambda,
            "Raw_Lambda_Vector": result.extra.get("raw_lambda", ""),
            "Algorithm_Sec": round(result.seconds, 3),
            "Wall_Sec": round(result.extra.get("wall_seconds", 0.0), 3),
            "Status": result.status,
        }

    if args.jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=args.jobs, backend="threading")(
            delayed(_execute)(task) for task in tasks)
    else:
        rows = [_execute(task) for task in tasks]

    df = pd.DataFrame(rows)
    tag = f"{args.metric}_budget{args.budget}" + (f"_{args.tag}" if args.tag else "")
    table_path = BASE_RESULTS_DIR / f"Metaheuristic_Comparison_{tag}.csv"
    df.to_csv(table_path, index=False)
    print(f"\nSaved full table: {table_path}")

    # ---- summary: mean over subsets, per method -----------------------------
    summary = (df.groupby("Method")
                 .agg(mean_best=("Best_Metric", "mean"),
                      mean_snapped=("Best_Metric_Snapped", "mean"),
                      mean_evals=("Evaluations", "mean"),
                      mean_seconds=("Wall_Sec", "mean"),
                      runs=("Best_Metric", "size"))
                 .sort_values("mean_snapped", ascending=False))
    summary_path = BASE_RESULTS_DIR / f"Metaheuristic_Summary_{tag}.csv"
    summary.to_csv(summary_path)
    print(f"\n--- summary ({tag}) ---")
    print(summary.to_string())
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
