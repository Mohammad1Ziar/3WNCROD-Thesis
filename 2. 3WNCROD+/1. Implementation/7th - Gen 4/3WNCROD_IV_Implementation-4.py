#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WNCROD Modular Ablation Suite (Generations 1-4 Synthesis)
Evaluates V0 (Baseline), V2 (Global Weights), V4 (Tiered NDF), V3 (Subspace SPE),
and their combinations across a 20-point grid (λ ∈ [0.1, 2.0]).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

_EPS = 1e-12
_LAMBDA_GRID = np.round(np.arange(0.1, 2.05, 0.1), 2)


# =============================================================================
# 1. SUBSPACE SPE SYNTHESIS (for V3 and V_star)
# =============================================================================

def synthesize_spe_feature(data: np.ndarray, X_tem: np.ndarray) -> np.ndarray:
    """Generates min-max scaled SPE attribute from numeric columns using X statistics."""
    numeric_mask = np.all(data <= 1.0, axis=0) & ~np.all(np.isnan(data), axis=0)
    num_idx = np.flatnonzero(numeric_mask)
    m_num = int(num_idx.size)
    if m_num < 2:
        return data

    r_target = max(1, min(int(np.ceil(np.log2(m_num + 1))), m_num - 1))
    Z = np.empty((data.shape[0], m_num), dtype=np.float64)

    for k, j in enumerate(num_idx):
        col = data[:, j]
        x_col = col[X_tem]
        med = float(np.median(x_col))
        mad = float(1.4826 * np.median(np.abs(x_col - med)))
        scale = mad if mad > _EPS else float(np.std(x_col))
        scale = scale if scale > _EPS else 1.0
        Z[:, k] = (col - med) / scale

    Z_X = Z[X_tem]
    cov = np.cov(Z_X, rowvar=False, bias=True)
    _, eigvecs = np.linalg.eigh(cov)
    V_r = eigvecs[:, ::-1][:, :r_target]

    projected = Z @ V_r
    spe = np.einsum("ij,ij->i", Z, Z) - np.einsum("ik,ik->i", projected, projected)
    spe = np.maximum(spe, 0.0)

    lo, hi = float(spe[X_tem].min()), float(spe[X_tem].max())
    c_spe = np.clip((spe - lo) / (hi - lo + _EPS), 0.0, 1.0)
    return np.column_stack([data, c_spe])


# =============================================================================
# 2. CORE GRANULATION & ABLATION ENGINE
# =============================================================================

def evaluate_granulation_block(
    data: np.ndarray,
    X_tem: np.ndarray,
    lammda: float,
) -> Dict[str, np.ndarray]:
    """
    Computes granulation and evaluates V0, V2, V4, and V2+4 in a single pass.
    """
    n, m = data.shape
    n_X = X_tem.size

    in_X = np.zeros(n, dtype=bool)
    in_X[X_tem] = True
    out_X = ~in_X

    # Attribute Radii (standard baseline rule)
    delta = np.zeros(m, dtype=np.float64)
    numeric_mask = np.all(data <= 1.0, axis=0)
    delta[numeric_mask] = np.std(data[:, numeric_mask], axis=0) / max(lammda, _EPS)

    pos_full = np.empty((m, n), dtype=bool)
    neg_full = np.empty((m, n), dtype=bool)
    rel_X_list: List[np.ndarray] = []
    hit = np.empty((n_X, m), dtype=np.int64)

    for col in range(m):
        if numeric_mask[col]:
            diff = np.abs(data[:, col][:, None] - data[:, col][None, :])
            rel = diff <= delta[col]
        else:
            rel = (data[:, col][:, None] == data[:, col][None, :])

        # True definitions over universe U:
        pos_full[col] = ~rel[:, out_X].any(axis=1)
        neg_full[col] = ~rel[:, in_X].any(axis=1)

        rel_X = rel[np.ix_(X_tem, X_tem)]
        hit[:, col] = rel_X.sum(axis=1)
        rel_X_list.append(rel_X)

    # Three-way partitions on X
    lower = pos_full[:, X_tem]
    ib = (~lower).astype(np.int8)
    neb = ib.all(axis=0)
    npb = ib & ~neb[None, :]

    neb_size = int(neb.sum())
    npb_sizes = npb.sum(axis=1)
    lower_sizes = lower.sum(axis=1)

    # NOM matrix on X x X
    overlap = np.zeros((n_X, n_X), dtype=np.int32)
    for col in range(m):
        overlap += rel_X_list[col].astype(np.int32)
    nom = m - overlap

    d1, d2, d3 = m / 3.0, m / 2.0, 0.9 * m

    neb_idx = np.flatnonzero(neb)
    neb_cnt = (nom[:, neb_idx] <= d1).sum(axis=1).astype(np.float64) if neb_idx.size > 0 else np.zeros(n_X)

    npb_cnt = np.zeros((n_X, m), dtype=np.float64)
    lower_cnt = np.zeros((n_X, m), dtype=np.float64)
    for col in range(m):
        idx_npb = np.flatnonzero(npb[col])
        if idx_npb.size > 0:
            npb_cnt[:, col] = (nom[:, idx_npb] >= d2).sum(axis=1).astype(np.float64)
        idx_lower = np.flatnonzero(lower[col])
        if idx_lower.size > 0:
            lower_cnt[:, col] = (nom[:, idx_lower] >= d3).sum(axis=1).astype(np.float64)

    # --- Variant 0 Elements ---
    ndf_v0 = (neb_cnt[:, None] + npb_cnt + lower_cnt) / float(n_X)
    w_base = (1.0 - np.sqrt(hit / float(n_X))) / float(m)
    scores_v0 = (ndf_v0 * w_base).sum(axis=1)

    # --- Variant 2 Elements (Calibrated Global Weighting) ---
    gamma = (pos_full.sum(axis=1) + neg_full.sum(axis=1)) / float(n)
    gamma_max = float(gamma.max()) if gamma.size > 0 else 0.0
    gamma_hat = np.clip(gamma / (gamma_max + _EPS), 0.10, 1.0)
    # Geometric mean blend restoring 1/m factor
    w_v2 = np.sqrt(np.clip(w_base, 0.0, None)) * np.sqrt(gamma_hat[None, :] / float(m))
    scores_v2 = (ndf_v0 * w_v2).sum(axis=1)

    # --- Variant 4 Elements (Tiered Evidentiary NDF) ---
    w_neb_r = 1.0 - neb_size / float(n_X)
    w_npb_r = 1.0 - npb_sizes / float(n_X)
    w_pos_r = 1.0 - lower_sizes / float(n_X)

    denom_v4 = w_neb_r * neb_size + w_npb_r * npb_sizes + w_pos_r * lower_sizes
    numer_v4 = (
        w_neb_r * neb_cnt[:, None]
        + w_npb_r[None, :] * npb_cnt
        + w_pos_r[None, :] * lower_cnt
    )
    ndf_v4 = np.zeros_like(numer_v4)
    ok = denom_v4 > _EPS
    ndf_v4[:, ok] = numer_v4[:, ok] / denom_v4[ok][None, :]
    ndf_v4[:, ~ok] = ndf_v0[:, ~ok]

    scores_v4 = (ndf_v4 * w_base).sum(axis=1)
    scores_v2_v4 = (ndf_v4 * w_v2).sum(axis=1)

    return {
        "V0": scores_v0,
        "V2": scores_v2,
        "V4": scores_v4,
        "V2+V4": scores_v2_v4,
    }


# =============================================================================
# 3. DATASET HANDLING
# =============================================================================

def _generate_synthetic_mammography(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_total = 11200
    n_outliers = int(np.round(n_total * 0.015))
    n_inliers = n_total - n_outliers

    c1_in = rng.choice([0, 1, 2, 3, 4, 5], size=n_inliers, p=[0.18, 0.22, 0.20, 0.18, 0.12, 0.10])
    c1_out = rng.choice([0, 1, 2, 3, 4, 5], size=n_outliers)

    latent1 = rng.normal(0.0, 1.0, size=n_inliers)
    latent2 = rng.normal(0.0, 0.6, size=n_inliers)
    c2_in = 0.70 * latent1 + 0.30 * latent2 + 0.20 * rng.normal(size=n_inliers)
    c3_in = 0.60 * latent1 - 0.40 * latent2 + 0.25 * rng.normal(size=n_inliers)
    c4_in = 0.50 * latent1 + 0.20 * latent2 + 0.30 * rng.normal(size=n_inliers)
    c5_in = -0.40 * latent1 + 0.60 * latent2 + 0.20 * rng.normal(size=n_inliers)
    c6_in = 0.30 * latent1 + 0.50 * latent2 + 0.25 * rng.normal(size=n_inliers)

    c2_out = rng.normal(loc=rng.choice([-3.5, 3.5], size=n_outliers), scale=0.5, size=n_outliers)
    c3_out = rng.normal(loc=rng.choice([-3.0, 3.0], size=n_outliers), scale=0.6, size=n_outliers)
    c4_out = rng.normal(loc=rng.choice([-2.5, 2.5], size=n_outliers), scale=0.7, size=n_outliers)
    c5_out = rng.normal(loc=rng.choice([-3.5, 3.5], size=n_outliers), scale=0.5, size=n_outliers)
    c6_out = rng.normal(loc=rng.choice([-3.0, 3.0], size=n_outliers), scale=0.6, size=n_outliers)

    c1 = np.concatenate([c1_in, c1_out])
    c2 = np.concatenate([c2_in, c2_out])
    c3 = np.concatenate([c3_in, c3_out])
    c4 = np.concatenate([c4_in, c4_out])
    c5 = np.concatenate([c5_in, c5_out])
    c6 = np.concatenate([c6_in, c6_out])
    labels = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])

    df = pd.DataFrame({"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6, "label": labels})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    full_data = df[["c1", "c2", "c3", "c4", "c5", "c6", "label"]].values
    return full_data[:, :6], full_data[:, 6].astype(int), df


def load_mammography() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    csv_path = "/media/mine/Mine/3.Main-Ubuntu-Envs/2.Thesis-Env/3.Datasets/mammographyori.csv"
    mat_path = "/media/mine/Mine/3.Main-Ubuntu-Envs/2.Thesis-Env/3.Datasets/mammography.mat"

    use_synthetic = not (os.path.isfile(csv_path) and os.path.isfile(mat_path))

    if use_synthetic:
        warnings.warn(
            "Original Mammography dataset not found. Using synthetic generator.",
            UserWarning,
        )
        X_raw, y, df = _generate_synthetic_mammography(seed=42)
    else:
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["c1", "c2", "c3", "c4", "c5", "c6", "label"]
        mat = loadmat(mat_path)
        full_data = mat["trandata"]
        X_raw = full_data[:, :6]
        y = full_data[:, 6].astype(int)

    scaler = MinMaxScaler()
    X = X_raw.copy()
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    ma_conditions = {
        "MA1": ("c4", -0.9, 2.0),
        "MA2": ("c5", -0.4, 1.0),
        "MA3": ("c6", -1.0, 1.0),
    }

    subsets = {}
    if use_synthetic:
        X_raw_df = pd.DataFrame(X_raw, columns=["c1", "c2", "c3", "c4", "c5", "c6"])
        for name, (feat, lo, hi) in ma_conditions.items():
            mask = (X_raw_df[feat] >= lo) & (X_raw_df[feat] <= hi)
            subsets[name] = X_raw_df[mask].index.to_numpy()
    else:
        for name, (feat, lo, hi) in ma_conditions.items():
            mask = (df[feat] >= lo) & (df[feat] <= hi)
            subsets[name] = mask[mask].index.to_numpy()

    return X, y, subsets


# =============================================================================
# 4. BENCHMARK HARNESS
# =============================================================================

def run_ablation_on_subset(
    name: str,
    X_data: np.ndarray,
    labels: np.ndarray,
    X_tem: np.ndarray,
    grid: np.ndarray = _LAMBDA_GRID,
) -> Dict:
    y_true = labels[X_tem]
    X_aug = synthesize_spe_feature(X_data, X_tem)

    variants = ["V0", "V2", "V4", "V2+V4", "V3 (SPE)", "V* (SPE+V2+V4)"]
    ap_records = {v: [] for v in variants}
    roc_records = {v: [] for v in variants}

    for lam in grid:
        # Pass 1: Original 6 attributes
        res_orig = evaluate_granulation_block(X_data, X_tem, lammda=lam)
        ap_records["V0"].append(average_precision_score(y_true, res_orig["V0"]))
        roc_records["V0"].append(roc_auc_score(y_true, res_orig["V0"]))

        ap_records["V2"].append(average_precision_score(y_true, res_orig["V2"]))
        roc_records["V2"].append(roc_auc_score(y_true, res_orig["V2"]))

        ap_records["V4"].append(average_precision_score(y_true, res_orig["V4"]))
        roc_records["V4"].append(roc_auc_score(y_true, res_orig["V4"]))

        ap_records["V2+V4"].append(average_precision_score(y_true, res_orig["V2+V4"]))
        roc_records["V2+V4"].append(roc_auc_score(y_true, res_orig["V2+V4"]))

        # Pass 2: Augmented 7 attributes (SPE)
        res_aug = evaluate_granulation_block(X_aug, X_tem, lammda=lam)
        ap_records["V3 (SPE)"].append(average_precision_score(y_true, res_aug["V0"]))
        roc_records["V3 (SPE)"].append(roc_auc_score(y_true, res_aug["V0"]))

        ap_records["V* (SPE+V2+V4)"].append(average_precision_score(y_true, res_aug["V2+V4"]))
        roc_records["V* (SPE+V2+V4)"].append(roc_auc_score(y_true, res_aug["V2+V4"]))

    summary = {}
    v0_best_ap = max(ap_records["V0"])
    v0_best_roc = max(roc_records["V0"])

    for v in variants:
        best_ap_idx = int(np.argmax(ap_records[v]))
        best_roc_idx = int(np.argmax(roc_records[v]))
        b_ap = ap_records[v][best_ap_idx]
        b_roc = roc_records[v][best_roc_idx]
        summary[v] = {
            "best_ap": b_ap,
            "best_ap_lam": grid[best_ap_idx],
            "delta_ap": b_ap - v0_best_ap,
            "best_roc": b_roc,
            "best_roc_lam": grid[best_roc_idx],
            "delta_roc": b_roc - v0_best_roc,
        }

    return {
        "subset": name,
        "n_X": len(X_tem),
        "n_out": int(y_true.sum()),
        "prev%": 100.0 * y_true.sum() / len(X_tem),
        "summary": summary,
    }


def main():
    print("=" * 90)
    print("3WNCROD MODULAR ABLATION STUDY (Grid λ ∈ [0.1, 2.0], step 0.1)")
    print("=" * 90)

    X_data, labels, subsets = load_mammography()
    print(f"\nUniverse: N = {X_data.shape[0]}, attributes = {X_data.shape[1]}")

    results = []
    for name in ["MA1", "MA2", "MA3"]:
        print(f"\nRunning full ablation grid on [{name}] ...", end="", flush=True)
        res = run_ablation_on_subset(name, X_data, labels, subsets[name])
        results.append(res)
        print(" Done.")

    for r in results:
        print("\n" + "=" * 90)
        print(f"SUBSET: {r['subset']} (N = {r['n_X']}, Outliers = {r['n_out']}, Prevalence = {r['prev%']:.2f}%)")
        print("=" * 90)
        header = (
            f"{'Variant':<18} | {'Best AP':>8} (λ)   | {'Δ AP vs V0':>12} | "
            f"{'Best ROC':>8} (λ)   | {'Δ ROC vs V0':>12}"
        )
        print(header)
        print("-" * 90)
        for v_name, metrics in r["summary"].items():
            print(
                f"{v_name:<18} | {metrics['best_ap']:>8.4f} ({metrics['best_ap_lam']:>3.1f}) | "
                f"{metrics['delta_ap']:>+12.4f} | "
                f"{metrics['best_roc']:>8.4f} ({metrics['best_roc_lam']:>3.1f}) | "
                f"{metrics['delta_roc']:>+12.4f}"
            )
        print("-" * 90)

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()