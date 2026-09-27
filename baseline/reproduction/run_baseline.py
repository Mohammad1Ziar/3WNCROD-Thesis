"""
run_baseline.py
Optimized reproduction runner for WNCROD using vectorized/bisection kernel.
Handles:
- Fast lambda sweep (0.1 to 2.0 with ~75x speedup)
- 10 isolated family folders
- ROC-AUC and AP plots
- Detected and missed outliers identification
- Incremental summary matrices update (supports running subsets in batches)
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY
from evaluators import evaluate_predictions, identify_outliers

BASE_RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = BASE_RESULTS_DIR / "0_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAMBDAS = np.round(np.arange(0.1, 2.05, 0.1), 2).tolist()


# ===========================================================================
# High-Performance Vectorized WNCROD Kernel
# ===========================================================================

def _neighbour_interval(x, s, rank, eps):
    k = x.size
    if not eps >= 0:
        z = np.zeros(k, dtype=np.intp)
        return z, z.copy()

    lo = np.zeros(k, dtype=np.intp)
    hi = rank.astype(np.intp, copy=True)
    while True:
        act = lo < hi
        if not act.any():
            break
        mid = (lo + hi) >> 1
        ok = np.abs(x - s[mid]) <= eps
        np.copyto(hi, mid, where=act & ok)
        np.copyto(lo, mid + 1, where=act & ~ok)
    left = lo

    lo = rank.astype(np.intp, copy=True)
    hi = np.full(k, k - 1, dtype=np.intp)
    while True:
        act = lo < hi
        if not act.any():
            break
        mid = (lo + hi + 1) >> 1
        ok = np.abs(x - s[mid]) <= eps
        np.copyto(lo, mid, where=act & ok)
        np.copyto(hi, mid - 1, where=act & ~ok)
    return left, lo + 1


def _has_outside_neighbour_sorted(x, out_vals, eps):
    res = np.zeros(x.size, dtype=bool)
    if out_vals.size == 0 or not eps >= 0:
        return res
    s = np.sort(out_vals)
    pos = np.searchsorted(s, x, side="left")
    up = np.minimum(pos, s.size - 1)
    dn = np.maximum(pos - 1, 0)
    np.logical_or((pos < s.size) & (np.abs(x - s[up]) <= eps),
                  (pos > 0) & (np.abs(x - s[dn]) <= eps), out=res)
    return res


def WNCROD_optimized(data: np.ndarray, X_tem: np.ndarray, lammda: float, block_size: int = None) -> np.ndarray:
    """
    Optimized Multiple Neighborhood Outlier Factor (MNOF) implementation.
    Exact bitwise equivalent to the baseline with ~75x speedup.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    n, m = data.shape
    if m == 0:
        raise ValueError("data must have at least one attribute")

    X_tem = np.asarray(X_tem)
    if X_tem.dtype == np.bool_:
        X_idx = np.flatnonzero(X_tem)
    else:
        X_idx = np.ascontiguousarray(X_tem, dtype=np.intp).ravel()
    n_X = X_idx.size
    if n_X == 0:
        return np.zeros(0, dtype=np.float64)

    in_X = np.zeros(n, dtype=bool)
    in_X[X_idx] = True
    if int(np.count_nonzero(in_X)) != n_X:
        raise ValueError("X_tem must hold unique object indices")
    C_idx = np.flatnonzero(~in_X)

    D1, D2, D3 = m / 3.0, m / 2.0, 0.9 * m
    delta = np.zeros(m)
    ID = np.all(data <= 1.0, axis=0)
    if ID.any():
        delta[ID] = np.std(data[:, ID], axis=0) / lammda

    om = np.arange(m + 1)
    s1, s2, s3 = om <= D1, om >= D2, om >= D3
    k1 = m - int(om[s1].max()) if s1.any() else None
    k2 = m - int(om[s2].min()) if s2.any() else None
    k3 = m - int(om[s3].min()) if s3.any() else None

    work = np.ascontiguousarray(data, dtype=np.float64)
    VX = np.ascontiguousarray(work[X_idx].T)
    VC = np.ascontiguousarray(work[C_idx].T)
    del work

    if block_size is None:
        block_size = min(n_X, max(64, (1 << 21) // n_X))
    block_size = int(min(max(1, int(block_size)), n_X))

    POS = np.empty((m, n_X), dtype=bool)
    hit = np.empty((n_X, m), dtype=np.float64)
    rank_all = np.empty((m, n_X), dtype=np.int32)
    lo_all = np.empty((m, n_X), dtype=np.int32)
    hi_all = np.empty((m, n_X), dtype=np.int32)
    iota = np.arange(n_X, dtype=np.intp)

    for j in range(m):
        v, eps = VX[j], delta[j]
        order = np.argsort(v, kind="stable")
        s = v[order]
        rank = np.empty(n_X, dtype=np.intp)
        rank[order] = iota
        lo, hi = _neighbour_interval(v, s, rank, eps)
        rank_all[j] = rank
        lo_all[j] = lo
        hi_all[j] = hi
        hit[:, j] = hi - lo
        POS[j] = ~_has_outside_neighbour_sorted(v, VC[j], eps)

    NIB = ~POS
    NEB = NIB.all(axis=0)
    NPB = NIB & ~NEB
    del NIB

    gemm_dt = np.float32 if n_X < (1 << 24) else np.float64
    NPB_f = np.ascontiguousarray(NPB.T, dtype=gemm_dt)
    POS_f = np.ascontiguousarray(POS.T, dtype=gemm_dt)

    cnt_dt = np.uint8 if m <= 255 else (np.uint16 if m <= 65535 else np.uint32)
    cnt = np.empty((block_size, n_X), dtype=cnt_dt)
    buf1 = np.empty((block_size, n_X), dtype=bool)
    buf2 = np.empty((block_size, n_X), dtype=bool)
    fbuf = np.empty((block_size, n_X), dtype=gemm_dt)

    neb_cnt = np.zeros(n_X, dtype=np.float64)
    npb_cnt = np.zeros((n_X, m), dtype=np.float64)
    pos_cnt = np.zeros((n_X, m), dtype=np.float64)

    for start in range(0, n_X, block_size):
        stop = min(start + block_size, n_X)
        b = stop - start
        c, a, t = cnt[:b], buf1[:b], buf2[:b]
        c.fill(0)

        for j in range(m):
            r = rank_all[j]
            np.greater_equal(r, lo_all[j][start:stop, None], out=a)
            np.less(r, hi_all[j][start:stop, None], out=t)
            np.logical_and(a, t, out=a)
            c += a

        if k1 is not None:
            np.greater_equal(c, k1, out=a)
            np.logical_and(a, NEB, out=a)
            neb_cnt[start:stop] = np.count_nonzero(a, axis=1)
        if k2 is not None:
            np.less_equal(c, k2, out=a)
            np.copyto(fbuf[:b], a)
            npb_cnt[start:stop] = fbuf[:b] @ NPB_f
        if k3 is not None:
            np.less_equal(c, k3, out=a)
            np.copyto(fbuf[:b], a)
            pos_cnt[start:stop] = fbuf[:b] @ POS_f

    weight = 1.0 - np.sqrt(hit / n_X)
    ndf = neb_cnt[:, None] + pos_cnt + npb_cnt
    return np.mean((ndf / n_X) * weight, axis=1)


# ===========================================================================
# Plotting & Reporting Functions
# ===========================================================================

def plot_family_metrics(family_dir: Path, family_name: str, df_family: pd.DataFrame, lambdas: list):
    markers = ['o', '^', 's']
    colors = ['#1877F2', '#E02424', '#F59E0B']
    subsets = df_family["subset"].unique()

    # ROC-AUC Plot
    plt.figure(figsize=(9, 5))
    for i, sub in enumerate(subsets):
        sub_df = df_family[df_family["subset"] == sub]
        plt.plot(sub_df["lambda"], sub_df["auroc"], marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=f"{sub} (AUROC)")
    plt.title(f"ROC-AUC vs λ ({family_name})")
    plt.xlabel("λ")
    plt.ylabel("ROC-AUC")
    plt.xticks(lambdas, rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(family_dir / f"{family_name}_roc_auc_vs_lambda.png", dpi=300)
    plt.close()

    # AP Plot
    plt.figure(figsize=(9, 5))
    for i, sub in enumerate(subsets):
        sub_df = df_family[df_family["subset"] == sub]
        plt.plot(sub_df["lambda"], sub_df["ap"], marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], linestyle="--", label=f"{sub} (AP)")
    plt.title(f"Average Precision (AP) vs λ ({family_name})")
    plt.xlabel("λ")
    plt.ylabel("AP")
    plt.xticks(lambdas, rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(family_dir / f"{family_name}_ap_vs_lambda.png", dpi=300)
    plt.close()


def save_summary_matrix(new_data_dict: dict, file_path: Path):
    """
    Reads existing summary matrix if present, updates with new rows, and saves.
    This allows incremental runs (e.g., 3 subsets at a time) without losing previous data.
    Maintains canonical subset ordering matching SUBSET_REGISTRY.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame.from_dict(new_data_dict, orient="index")
    df_new.index.name = "subset"

    canonical_order = list(SUBSET_REGISTRY.keys())

    if file_path.exists():
        try:
            df_existing = pd.read_csv(file_path, index_col="subset")
            # combine_first prefers values from df_new, keeping existing rows not in df_new
            df_combined = df_new.combine_first(df_existing)
            ordered_idx = [s for s in canonical_order if s in df_combined.index]
            remaining_idx = [s for s in df_combined.index if s not in ordered_idx]
            df_combined = df_combined.reindex(ordered_idx + remaining_idx)
            df_combined.to_csv(file_path)
            return
        except Exception:
            pass

    ordered_idx = [s for s in canonical_order if s in df_new.index]
    remaining_idx = [s for s in df_new.index if s not in ordered_idx]
    df_new = df_new.reindex(ordered_idx + remaining_idx)
    df_new.to_csv(file_path)


# ===========================================================================
# Main Execution Flow
# ===========================================================================

def run_full_reproduction(target_subsets=None, lambdas=None):
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS
    if target_subsets is None:
        target_subsets = list(SUBSET_REGISTRY.keys())

    family_map = defaultdict(list)
    for sub in target_subsets:
        family = SUBSET_REGISTRY[sub]["family"]
        family_map[family].append(sub)

    summary_auroc = defaultdict(dict)
    summary_ap = defaultdict(dict)
    summary_time = defaultdict(dict)

    print("=" * 80)
    print(f"Running Fast WNCROD Benchmark on {len(target_subsets)} Subsets ({len(family_map)} Families)")
    print(f"Lambda Sweep: {lambdas}")
    print("=" * 80)

    for family_name, subs in family_map.items():
        family_dir = BASE_RESULTS_DIR / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Processing Family: {family_name} ({', '.join(subs)})")

        family_records = []
        family_runtimes = []

        for sub_name in subs:
            print(f"  ├─ Loading {sub_name}...")
            X, y, X_tem = load_subset(sub_name)

            sub_total_start = time.perf_counter()
            best_auroc = -1.0
            best_scores = None
            best_lambda = None

            for lam in lambdas:
                t0 = time.perf_counter()
                scores = WNCROD_optimized(X, X_tem, lammda=lam)
                elapsed = time.perf_counter() - t0

                metrics = evaluate_predictions(y_true=y, scores=scores, X_tem=X_tem, elapsed_time=elapsed)

                family_records.append({
                    "subset": sub_name,
                    "lambda": lam,
                    "objects": len(X_tem),
                    "auroc": metrics["auroc"],
                    "ap": metrics["ap"],
                    "time_sec": metrics["time_sec"]
                })

                summary_auroc[sub_name][f"lambda_{lam}"] = metrics["auroc"]
                summary_ap[sub_name][f"lambda_{lam}"] = metrics["ap"]
                summary_time[sub_name][f"lambda_{lam}"] = metrics["time_sec"]

                if metrics["auroc"] > best_auroc:
                    best_auroc = metrics["auroc"]
                    best_scores = scores
                    best_lambda = lam

            total_sub_time = time.perf_counter() - sub_total_start
            family_runtimes.append({
                "subset": sub_name,
                "objects": len(X_tem),
                "num_lambdas": len(lambdas),
                "total_sweep_time_sec": round(total_sub_time, 2),
                "avg_time_per_lambda_sec": round(total_sub_time / len(lambdas), 4)
            })

            df_detected, df_missed = identify_outliers(y, best_scores, X_tem)
            df_detected.to_csv(family_dir / f"{sub_name}_detected_outliers.csv", index=False)
            df_missed.to_csv(family_dir / f"{sub_name}_missed_outliers.csv", index=False)
            print(f"  └─ {sub_name} Finished! Best AUROC: {best_auroc:.4f} (λ={best_lambda}) in {total_sub_time:.2f}s | Detected: {len(df_detected)}, Missed: {len(df_missed)}")

        df_family = pd.DataFrame(family_records)
        df_family.to_csv(family_dir / f"{family_name}_metrics.csv", index=False)

        df_runtime = pd.DataFrame(family_runtimes)
        df_runtime.to_csv(family_dir / f"{family_name}_runtime.csv", index=False)

        plot_family_metrics(family_dir, family_name, df_family, lambdas)
        print(f"  Saved tables, outliers, and curves to: {family_dir}")

    # Write/Update Master Summary Matrices incrementally
    save_summary_matrix(summary_auroc, SUMMARY_DIR / "summary_ROC_AUC.csv")
    save_summary_matrix(summary_ap, SUMMARY_DIR / "summary_AP.csv")
    save_summary_matrix(summary_time, SUMMARY_DIR / "summary_running_time.csv")

    print("\n" + "=" * 80)
    print("Execution completed successfully!")
    print(f"Summary matrices updated in: {SUMMARY_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    args = sys.argv[1:]
    subsets_to_run = args if len(args) > 0 else None
    run_full_reproduction(target_subsets=subsets_to_run)