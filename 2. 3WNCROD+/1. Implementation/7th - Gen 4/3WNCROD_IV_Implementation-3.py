#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WNCROD-IV: Topologically-Stabilized Neighborhood Characteristic Region Outlier Detection
with Grid-Searched Lambda Formulation (Gen 4 Grid Optimization)

Components:
  - Alg. 0† : Conditional Subspace Augmentation (SPE with η_r >= 0.60 gate)
  - Eq. 8†  : Robust-Scale / λ Radius Sweep (λ in [0.1, 2.0], step 0.1)
  - Eq. 11† : Adaptive NOM Thresholds indexed to effective dimension m_eff
  - Eq. 10† : Calibrated Discriminative Weights with regularized gamma_hat
  - Eq. 4†  : Tiered Evidentiary NDF (IDF-style region rarity)
  - Eq. 5†  : Stabilized MNOF Fusion

Evaluates 20 grid points across MA1, MA2, and MA3 subsets.
Reports per-metric maxima (Best AP, Best ROC) and full grid trajectory.
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

# =============================================================================
# 0. CONFIGURATION & CONSTANTS
# =============================================================================

_MADN_C = 1.4826          # MAD -> Gaussian consistency factor
_IQR_C = 1.34             # IQR -> Gaussian consistency factor
_EPS = 1e-12              # Numerical stability zero guard
_ETA_MIN = 0.60           # Minimum explained variance for subspace augmentation
_ALPHA = 0.5              # Jeffreys-symmetric blend exponent (Eq. 10†)

# Grid parameters: 0.1 to 2.0 with step 0.1 (20 evaluation points)
_LAMBDA_GRID = np.round(np.arange(0.1, 2.05, 0.1), 2)


# =============================================================================
# 1. ROBUST STATISTICAL PRIMITIVES
# =============================================================================

def _madn(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    return float(_MADN_C * np.median(np.abs(v - med)))


def _iqr(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    q1, q3 = np.percentile(v, [25.0, 75.0])
    return float(q3 - q1)


def _std(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    return 0.0 if v.size == 0 else float(np.std(v))


def _min_positive(*values: float) -> float:
    positive = [float(v) for v in values if np.isfinite(v) and v > 0.0]
    return min(positive) if positive else 0.0


# =============================================================================
# 2. CONDITIONAL SUBSPACE AUGMENTATION — Alg. 0†
# =============================================================================

def conditional_subspace_augmentation(
    data: np.ndarray,
    numeric_mask: np.ndarray,
    X_tem: np.ndarray,
    eta_min: float = _ETA_MIN,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Synthesizes c_{m+1} = minmax(SPE) only if dominant correlation explains >= eta_min variance.
    Computed once per dataset before sweeping lambda.
    """
    n, m = data.shape
    num_idx = np.flatnonzero(numeric_mask)
    m_num = int(num_idx.size)

    info: Dict = {"used": False, "eta_r": 0.0, "r": 0, "reason": ""}

    if m_num < 2:
        info["reason"] = "m_num < 2: C† = C"
        return data, numeric_mask, info

    r_target = max(0, min(int(np.ceil(np.log2(m_num + 1))), m_num - 1))
    if r_target == 0:
        info["reason"] = "r_target = 0: C† = C"
        return data, numeric_mask, info

    # Step 1: Robust standardization using X statistics
    Z = np.empty((n, m_num), dtype=np.float64)
    for k, j in enumerate(num_idx):
        col = data[:, j]
        x_col = col[X_tem]
        centre = float(np.median(x_col))
        scale = _min_positive(_madn(x_col), _iqr(x_col) / _IQR_C)
        if scale <= 0.0:
            scale = _std(x_col)
        if scale <= 0.0:
            scale = 1.0
        z = (col - centre) / scale
        Z[:, k] = np.where(np.isnan(z), 0.0, z)

    # Step 2: Explained variance ratio
    Z_X = Z[X_tem]
    if Z_X.shape[0] < 2:
        info["reason"] = "|X| < 2: C† = C"
        return data, numeric_mask, info

    cov = np.cov(Z_X, rowvar=False, bias=True)
    eigvals = np.linalg.eigvalsh(cov)[::-1]

    total_var = eigvals.sum()
    if total_var <= _EPS:
        info["reason"] = "total_var <= 0: C† = C"
        return data, numeric_mask, info

    eta_r = eigvals[:r_target].sum() / total_var
    info["eta_r"] = float(eta_r)
    info["r"] = r_target

    if eta_r < eta_min:
        info["reason"] = f"η_r = {eta_r:.3f} < {eta_min}: no dominant subspace"
        return data, numeric_mask, info

    # Step 3: Dominant projection & SPE
    _, eigvecs = np.linalg.eigh(cov)
    V_r = eigvecs[:, ::-1][:, :r_target]

    projected = Z @ V_r
    spe = np.einsum("ij,ij->i", Z, Z) - np.einsum("ik,ik->i", projected, projected)
    spe = np.maximum(spe, 0.0)

    # Step 4: Minmax normalization
    lo, hi = float(spe[X_tem].min()), float(spe[X_tem].max())
    if hi - lo <= _EPS:
        info["reason"] = "SPE constant: C† = C"
        return data, numeric_mask, info

    c_m1 = np.clip((spe - lo) / (hi - lo), 0.0, 1.0)
    data_aug = np.column_stack([data, c_m1])
    numeric_mask_aug = np.append(numeric_mask, True)
    info["used"] = True
    info["reason"] = f"augmented with c_{{m+1}} = minmax(SPE), η_r={eta_r:.3f}"
    return data_aug, numeric_mask_aug, info


# =============================================================================
# 3. 3WNCROD-IV GRID ENGINE — Algorithm 1† (Parameterized by λ)
# =============================================================================

def compute_gen4_radii(
    data: np.ndarray,
    numeric_mask: np.ndarray,
    X_tem: np.ndarray,
    lammda: float,
) -> np.ndarray:
    """
    Computes radii using robust scale over lambda:
    ε_j(λ) = max( min+(MADN, IQR/1.34) / λ, 1/n_X )
    Categorical attributes receive ε = 0.
    """
    _, m = data.shape
    n_X = X_tem.size
    floor = 1.0 / float(n_X)
    radii = np.zeros(m, dtype=np.float64)

    for j in range(m):
        if not numeric_mask[j]:
            radii[j] = 0.0
            continue

        x_vals = data[X_tem, j]
        scale = _min_positive(_madn(x_vals), _iqr(x_vals) / _IQR_C)
        if scale <= 0.0:
            scale = _std(x_vals)

        radii[j] = max(scale / max(lammda, _EPS), floor)
    return radii


def wncrod_iv_core(
    data_aug: np.ndarray,
    numeric_mask_aug: np.ndarray,
    X_tem: np.ndarray,
    lammda: float,
    alpha: float = _ALPHA,
) -> np.ndarray:
    """
    Executes one evaluation of 3WNCROD-IV at a given lambda.
    """
    n, m_dagger = data_aug.shape
    n_X = X_tem.size

    in_X = np.zeros(n, dtype=bool)
    in_X[X_tem] = True
    out_X = ~in_X

    # Compute radii at lambda
    radii = compute_gen4_radii(data_aug, numeric_mask_aug, X_tem, lammda)

    pos = np.empty((m_dagger, n), dtype=bool)
    neg = np.empty((m_dagger, n), dtype=bool)
    hit = np.empty((n_X, m_dagger), dtype=np.int64)
    rel_X_list: List[np.ndarray] = []

    for j in range(m_dagger):
        col = data_aug[:, j]
        if numeric_mask_aug[j]:
            dist = np.abs(col[:, None] - col[None, :])
            rel = dist <= radii[j]
        else:
            rel = (col[:, None] == col[None, :])

        pos[j] = ~rel[:, out_X].any(axis=1)
        neg[j] = ~rel[:, in_X].any(axis=1)
        rel_X = rel[np.ix_(X_tem, X_tem)]
        hit[:, j] = rel_X.sum(axis=1)
        rel_X_list.append(rel_X)

    # Granulation partitions
    nib = in_X[None, :] & ~pos
    neb = nib.all(axis=0) if m_dagger > 0 else np.zeros(n, dtype=bool)
    npb = nib & ~neb[None, :]

    neb_size = int(neb[X_tem].sum())
    npb_sizes = npb[:, X_tem].sum(axis=1).astype(np.int64)
    pos_sizes = pos[:, X_tem].sum(axis=1).astype(np.int64)
    neg_sizes = neg[:, X_tem].sum(axis=1).astype(np.int64)

    # Informative attributes selection (Def. 6)
    informative = (neb_size + npb_sizes) > 0
    I = np.flatnonzero(informative)
    m_eff = int(I.size)

    if m_eff == 0:
        return np.zeros(n_X, dtype=np.float64)

    # Eq. 11†: Adaptive NOM thresholds
    d1 = m_eff / 3.0
    d2 = m_eff / 2.0
    d3 = 0.9 * m_eff

    # Accumulate NOM strictly over informative attributes
    overlap_I = np.zeros((n_X, n_X), dtype=np.int32)
    for j in I:
        overlap_I += rel_X_list[j].astype(np.int32)
    nom = m_eff - overlap_I

    # Characteristic counts (Eq. 3)
    neb_idx = np.flatnonzero(neb[X_tem])
    neb_cnt = np.zeros(n_X, dtype=np.float64)
    if neb_idx.size > 0:
        neb_cnt = (nom[:, neb_idx] <= d1).sum(axis=1).astype(np.float64)

    npb_cnt = np.zeros((n_X, m_eff), dtype=np.float64)
    pos_cnt = np.zeros((n_X, m_eff), dtype=np.float64)

    for idx_j, j in enumerate(I):
        sel_npb = np.flatnonzero(npb[j, X_tem])
        if sel_npb.size > 0:
            npb_cnt[:, idx_j] = (nom[:, sel_npb] >= d2).sum(axis=1).astype(np.float64)
        sel_pos = np.flatnonzero(pos[j, X_tem])
        if sel_pos.size > 0:
            pos_cnt[:, idx_j] = (nom[:, sel_pos] >= d3).sum(axis=1).astype(np.float64)

    # Eq. 4†: Tiered evidentiary NDF
    n_x_f = float(n_X)
    w_neb = 1.0 - neb_size / n_x_f
    w_npb = 1.0 - npb_sizes[I] / n_x_f
    w_pos = 1.0 - pos_sizes[I] / n_x_f

    denom = w_neb * neb_size + w_npb * npb_sizes[I] + w_pos * pos_sizes[I]
    numer = (
        w_neb * neb_cnt[:, None]
        + w_npb[None, :] * npb_cnt
        + w_pos[None, :] * pos_cnt
    )
    ndf = np.zeros_like(numer)
    ok = denom > _EPS
    if np.any(ok):
        ndf[:, ok] = numer[:, ok] / denom[ok][None, :]

    # Eq. 10†: Calibrated discriminative weights with Laplace regularization
    hit_I = hit[:, I].astype(np.float64)
    prevalence = hit_I / n_x_f
    omega_raw = 1.0 - np.sqrt(np.clip(prevalence, 0.0, 1.0))

    gamma = (pos_sizes + neg_sizes) / float(n)
    gamma_max = float(gamma[I].max()) if gamma[I].size > 0 else 0.0
    if gamma_max <= _EPS:
        gamma_hat = np.ones(m_eff, dtype=np.float64)
    else:
        gamma_hat = np.clip(gamma[I] / gamma_max, 0.05, 1.0)

    omega = (1.0 / m_eff) * (omega_raw ** alpha) * (gamma_hat[None, :] ** (1.0 - alpha))

    # Eq. 5†: Stabilized MNOF fusion
    mnof = (ndf * omega).sum(axis=1)
    return np.clip(mnof, 0.0, 1.0)


# =============================================================================
# 4. BASELINE REFERENCE IMPLEMENTATION (Exact Paper Formulation)
# =============================================================================

def wncrod_baseline_core(data: np.ndarray, X_tem: np.ndarray, lammda: float) -> np.ndarray:
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1
    n_X = int(X.sum())

    D1, D2, D3 = m / 3.0, m / 2.0, 0.9 * m
    delta = np.zeros(m)
    ID = np.all(data <= 1.0, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / lammda

    Lower = np.zeros((m, n))
    RM_list = []
    for col in range(m):
        diff = np.abs(data[:, col][:, None] - data[:, col][None, :])
        RM = diff <= delta[col]
        RM_list.append(RM)
        Lower_temp = np.min(np.maximum(1 - RM, np.tile(X, (n, 1))), axis=1)
        Lower[col, :] = Lower_temp

    IB = np.tile(X, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    weight = np.zeros((n_X, m))
    for col in range(m):
        RM_X = RM_list[col][X_tem, :]
        for i in range(n_X):
            weight[i, col] = 1.0 - np.sqrt(np.sum(np.minimum(RM_X[i, :], X)) / n_X)

    D_tem = np.zeros((n, n), dtype=np.int32)
    for col in range(m):
        D_tem += RM_list[col].astype(np.int32)
    NOM = m - D_tem
    X_OM = NOM[np.ix_(X_tem, X_tem)]

    NEB_num = np.zeros((n_X, m))
    Lower_num = np.zeros((n_X, m))
    NPB_num = np.zeros((n_X, m))
    for col in range(m):
        temp2 = Lower[col, :]
        temp3 = NPB[col, :]
        for i in range(n_X):
            temp1 = X_OM[i, :]
            NEB_num[i, col] = np.sum(np.minimum(NEB[X_tem], temp1 <= D1))
            NPB_num[i, col] = np.sum(np.minimum(temp3[X_tem], temp1 >= D2))
            Lower_num[i, col] = np.sum(np.minimum(temp2[X_tem], temp1 >= D3))

    MNOF = np.mean(((NEB_num + Lower_num + NPB_num) / n_X) * weight, axis=1)
    return MNOF


# =============================================================================
# 5. DATA LOADING
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
            "Original Mammography dataset not found. Using statistically faithful synthetic generator.",
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
# 6. GRID SEARCH & EVALUATION SUITE
# =============================================================================

def run_grid_search_on_subset(
    name: str,
    X_data: np.ndarray,
    labels: np.ndarray,
    X_tem: np.ndarray,
    grid: np.ndarray = _LAMBDA_GRID,
) -> Dict:
    """
    Sweeps lambda across the grid for both Baseline and 3WNCROD-IV.
    Finds independent best AP and best ROC for both.
    """
    y_true = labels[X_tem]
    n_X = len(X_tem)
    n_out = int(y_true.sum())
    numeric_mask = np.all(X_data <= 1.0, axis=0) & ~np.all(np.isnan(X_data), axis=0)

    # Precompute subspace augmentation once
    data_aug, numeric_mask_aug, subspace_info = conditional_subspace_augmentation(
        X_data, numeric_mask, X_tem, eta_min=_ETA_MIN
    )

    base_ap_list, base_roc_list = [], []
    iv_ap_list, iv_roc_list = [], []

    for lam in grid:
        # Baseline evaluation
        scores_base = wncrod_baseline_core(X_data, X_tem, lammda=lam)
        base_roc_list.append(roc_auc_score(y_true, scores_base))
        base_ap_list.append(average_precision_score(y_true, scores_base))

        # 3WNCROD-IV evaluation
        scores_iv = wncrod_iv_core(data_aug, numeric_mask_aug, X_tem, lammda=lam)
        iv_roc_list.append(roc_auc_score(y_true, scores_iv))
        iv_ap_list.append(average_precision_score(y_true, scores_iv))

    # Baseline best extraction
    idx_b_best_ap = int(np.argmax(base_ap_list))
    idx_b_best_roc = int(np.argmax(base_roc_list))

    # Gen 4 best extraction
    idx_iv_best_ap = int(np.argmax(iv_ap_list))
    idx_iv_best_roc = int(np.argmax(iv_roc_list))

    return {
        "subset": name,
        "n_X": n_X,
        "n_out": n_out,
        "prev%": 100.0 * n_out / n_X,
        "spe_used": subspace_info["used"],
        "spe_reason": subspace_info["reason"],
        # Baseline Best
        "base_best_ap": base_ap_list[idx_b_best_ap],
        "base_best_ap_lam": grid[idx_b_best_ap],
        "base_best_roc": base_roc_list[idx_b_best_roc],
        "base_best_roc_lam": grid[idx_b_best_roc],
        # Gen 4 Best
        "iv_best_ap": iv_ap_list[idx_iv_best_ap],
        "iv_best_ap_lam": grid[idx_iv_best_ap],
        "iv_best_roc": iv_roc_list[idx_iv_best_roc],
        "iv_best_roc_lam": grid[idx_iv_best_roc],
        # Deltas
        "delta_best_ap": iv_ap_list[idx_iv_best_ap] - base_ap_list[idx_b_best_ap],
        "delta_best_roc": iv_roc_list[idx_iv_best_roc] - base_roc_list[idx_b_best_roc],
        # Full trajectories
        "grid": grid,
        "base_ap_list": base_ap_list,
        "base_roc_list": base_roc_list,
        "iv_ap_list": iv_ap_list,
        "iv_roc_list": iv_roc_list,
    }


def main():
    print("=" * 85)
    print("3WNCROD-IV — Generation 4 Grid Search Optimization (λ ∈ [0.1, 2.0], step 0.1)")
    print("=" * 85)

    X_data, labels, subsets = load_mammography()
    print(f"\nUniverse: N = {X_data.shape[0]}, attributes = {X_data.shape[1]}")

    results = []
    for name in ["MA1", "MA2", "MA3"]:
        print(f"\nRunning 20-point grid search on [{name}] ...", end="", flush=True)
        res = run_grid_search_on_subset(name, X_data, labels, subsets[name])
        results.append(res)
        print(" Done.")

    # 1. High-Level Summary Table
    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY: BASELINE vs 3WNCROD-IV (Grid-Optimal)")
    print("=" * 85)
    header = (
        f"{'Subset':>6} | {'N':>6} | {'Out':>4} | {'Prev%':>6} | "
        f"{'Base AP':>8} (λ) | {'IV AP':>8} (λ) | {'Δ AP':>7} | "
        f"{'Base ROC':>8} (λ) | {'IV ROC':>8} (λ) | {'Δ ROC':>7}"
    )
    print(header)
    print("-" * 85)
    for r in results:
        print(
            f"{r['subset']:>6} | {r['n_X']:>6} | {r['n_out']:>4} | {r['prev%']:>6.2f} | "
            f"{r['base_best_ap']:>8.4f} ({r['base_best_ap_lam']:>3.1f}) | "
            f"{r['iv_best_ap']:>8.4f} ({r['iv_best_ap_lam']:>3.1f}) | "
            f"{r['delta_best_ap']:>+7.4f} | "
            f"{r['base_best_roc']:>8.4f} ({r['base_best_roc_lam']:>3.1f}) | "
            f"{r['iv_best_roc']:>8.4f} ({r['iv_best_roc_lam']:>3.1f}) | "
            f"{r['delta_best_roc']:>+7.4f}"
        )
    print("-" * 85)

    # 2. Detailed Per-Lambda Trajectory Table
    print("\n" + "=" * 85)
    print("DETAILED GRID TRAJECTORY PER SUBSET")
    print("=" * 85)
    for r in results:
        print(f"\n--- SubsetData: {r['subset']} (SPE Augmented: {'YES' if r['spe_used'] else 'NO'}) ---")
        print(f"{'λ':>5} | {'Base AP':>10} | {'IV AP':>10} | {'Base ROC':>10} | {'IV ROC':>10}")
        print("-" * 55)
        for i, lam in enumerate(r['grid']):
            print(
                f"{lam:>5.1f} | {r['base_ap_list'][i]:>10.4f} | {r['iv_ap_list'][i]:>10.4f} | "
                f"{r['base_roc_list'][i]:>10.4f} | {r['iv_roc_list'][i]:>10.4f}"
            )

    print("\n" + "=" * 85)
    print("Grid search evaluation complete.")
    print("=" * 85)


if __name__ == "__main__":
    main()