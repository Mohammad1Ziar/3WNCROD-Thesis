#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WNCROD-IV: Topologically-Stabilized Neighborhood Characteristic Region Outlier Detection
with Adaptive Multi-Granular Fusion — Corrected Production Implementation (Gen 4)

Fixes applied:
  - Corrected Eq. 8†: Applied 0.5 factor to gap radius and stabilized covering search
    against outlier-void inflation (restoring radii to optimal 0.02–0.08 range).
  - Regularized gamma_hat in Eq. 10† to prevent attribute blackout when POS is small.
  - Aligned NOM accumulation strictly to the informative attribute set (I).
  - Stabilized subspace SPE injection radius.

Author: Research Team
Date: 2026-09-10
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
_EPS = 1e-12              # Numerical zero threshold
_ETA_MIN = 0.60           # Minimum explained-variance ratio for subspace aug.
_ALPHA = 0.5              # Jeffreys-symmetric blend exponent (Eq. 10†)


# =============================================================================
# 1. ROBUST STATISTICAL PRIMITIVES
# =============================================================================

def _madn(vals: np.ndarray) -> float:
    """Normalised MAD: 1.4826 * med|x - med(x)|."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    return float(_MADN_C * np.median(np.abs(v - med)))


def _iqr(vals: np.ndarray) -> float:
    """Inter-quartile range Q3 - Q1."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    q1, q3 = np.percentile(v, [25.0, 75.0])
    return float(q3 - q1)


def _std(vals: np.ndarray) -> float:
    """Standard deviation."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    return 0.0 if v.size == 0 else float(np.std(v))


def _min_positive(*values: float) -> float:
    """Minimum over strictly positive candidates; 0.0 if none exist."""
    positive = [float(v) for v in values if np.isfinite(v) and v > 0.0]
    return min(positive) if positive else 0.0


# =============================================================================
# 2. STABILITY-MATCHED LAMBDA SEARCH (for Eq. 8†)
# =============================================================================

def _neighbor_counts_sorted(sorted_vals: np.ndarray, radius: float) -> np.ndarray:
    """
    Two-pointer sliding window computing the neighborhood count for every point
    in a sorted 1D array: O(n).
    """
    n = sorted_vals.size
    counts = np.empty(n, dtype=np.int32)
    left = 0
    right = 0
    for i in range(n):
        v = sorted_vals[i]
        while left < n and v - sorted_vals[left] > radius:
            left += 1
        while right < n and sorted_vals[right] - v <= radius:
            right += 1
        counts[i] = right - left
    return counts


def _find_stability_lambda(
    sorted_vals: np.ndarray,
    scale: float,
    k_star: int,
    target_tail_pct: float = 2.5,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Binary search for λ† such that the target tail percentile of inliers
    maintains at least k* neighbors, preventing extreme outliers from
    artificially blowing up the neighborhood radius.
    """
    if scale <= 0.0 or k_star <= 0:
        return 1.0

    lo, hi = 0.1, 40.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        radius = scale / mid
        counts = _neighbor_counts_sorted(sorted_vals, radius)
        eff_count = float(np.percentile(counts, target_tail_pct))
        if eff_count > k_star:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# =============================================================================
# 3. TOPOLOGICAL STABILITY RADIUS — Eq. 8†
# =============================================================================

def _compute_stability_radii(
    data: np.ndarray,
    numeric_mask: np.ndarray,
    X_tem: np.ndarray,
) -> np.ndarray:
    """
    Compute ε†_j for every attribute j ∈ C†.
    Enforces Definition 2 (0.5 * gap) and prevents outlier-void inflation.
    """
    n, m = data.shape
    n_X = X_tem.size
    k_star = int(np.ceil(np.log2(max(n_X, 2))))
    floor = 1.0 / float(n_X)
    radii = np.zeros(m, dtype=np.float64)

    for j in range(m):
        if not numeric_mask[j]:
            radii[j] = 0.0
            continue

        col = data[:, j]
        x_vals = col[X_tem]
        x_sorted = np.sort(x_vals)

        # 1. Robust scale σ̂
        scale = _min_positive(_madn(x_vals), _iqr(x_vals) / _IQR_C)
        if scale <= 0.0:
            scale = _std(x_vals)

        # 2. Covering radius ρ (Definition 2: 0.5 * k*-gap, trimmed against outlier voids)
        if x_sorted.size > k_star:
            gaps = x_sorted[k_star:] - x_sorted[:-k_star]
            rho = 0.5 * float(np.percentile(gaps, 98.0)) if gaps.size > 0 else 0.0
        else:
            rho = 0.0

        # 3. Stability-matched scale
        if scale > 0.0:
            lambda_dagger = _find_stability_lambda(x_sorted, scale, k_star, target_tail_pct=2.5)
            radius_scale = scale / lambda_dagger if lambda_dagger > 0.0 else 0.0
        else:
            radius_scale = 0.0

        raw_r = max(rho, radius_scale, floor)
        # Numerical guard: an attribute radius should not exceed 30% of its total range
        span = float(x_sorted[-1] - x_sorted[0]) if x_sorted.size > 0 else 1.0
        radii[j] = min(raw_r, max(0.15, span * 0.30))

    return radii


# =============================================================================
# 4. CONDITIONAL SUBSPACE AUGMENTATION — Alg. 0†
# =============================================================================

def _conditional_subspace_augmentation(
    data: np.ndarray,
    numeric_mask: np.ndarray,
    X_tem: np.ndarray,
    eta_min: float = _ETA_MIN,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
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

    # Step 1: Robust standardization
    Z = np.empty((n, m_num), dtype=np.float64)
    for k, j in enumerate(num_idx):
        col = data[:, j]
        x_col = col[X_tem]
        centre = float(np.median(x_col))
        scale = _madn(x_col)
        if scale <= 0.0:
            scale = _iqr(x_col) / _IQR_C
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

    # Step 4: Normalised injection
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
# 5. THREE-WAY GRANULATION & MNOF FUSION — Algorithm 1†
# =============================================================================

def wncrod_iv(
    data: np.ndarray,
    X_tem: np.ndarray,
    eta_min: float = _ETA_MIN,
    alpha: float = _ALPHA,
    return_diagnostics: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict]:
    """
    3WNCROD-IV Corrected Engine.
    """
    data = np.asarray(data, dtype=np.float64)
    n, m = data.shape
    X_tem = np.asarray(X_tem, dtype=int)
    n_X = X_tem.size

    if n_X == 0:
        return (np.zeros(0), {}) if return_diagnostics else np.zeros(0)

    in_X = np.zeros(n, dtype=bool)
    in_X[X_tem] = True
    out_X = ~in_X

    # Numeric attribute mask
    numeric_mask = np.all(data <= 1.0, axis=0) & ~np.all(np.isnan(data), axis=0)

    # Alg. 0†: Conditional subspace augmentation
    data_aug, numeric_mask_aug, subspace_info = _conditional_subspace_augmentation(
        data, numeric_mask, X_tem, eta_min
    )
    m_dagger = data_aug.shape[1]

    # Eq. 8†: Topological stability radii
    radii = _compute_stability_radii(data_aug, numeric_mask_aug, X_tem)

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

    # Three-way boundary partitions
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
        scores = np.zeros(n_X, dtype=np.float64)
        if return_diagnostics:
            return scores, {"m_dagger": m_dagger, "m_eff": 0, "radii": radii}
        return scores

    # Adaptive NOM thresholds (Eq. 11†)
    d1 = m_eff / 3.0
    d2 = m_eff / 2.0
    d3 = 0.9 * m_eff

    # Accumulate overlap strictly over informative attributes
    overlap_I = np.zeros((n_X, n_X), dtype=np.int32)
    for j in I:
        overlap_I += rel_X_list[j].astype(np.int32)
    nom = m_eff - overlap_I

    # Characteristic region counts (Eq. 3)
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

    # Tiered evidentiary NDF (Eq. 4†)
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

    # Calibrated discriminative weights with regularization (Eq. 10†)
    hit_I = hit[:, I].astype(np.float64)
    prevalence = hit_I / n_x_f
    omega_raw = 1.0 - np.sqrt(np.clip(prevalence, 0.0, 1.0))

    gamma = (pos_sizes + neg_sizes) / float(n)
    gamma_max = float(gamma[I].max()) if gamma[I].size > 0 else 0.0
    if gamma_max <= _EPS:
        gamma_hat = np.ones(m_eff, dtype=np.float64)
    else:
        # Regularization: clip at 0.05 to preserve all informative attributes from zeroing out
        gamma_hat = np.clip(gamma[I] / gamma_max, 0.05, 1.0)

    omega = (1.0 / m_eff) * (omega_raw ** alpha) * (gamma_hat[None, :] ** (1.0 - alpha))

    # Stabilized MNOF fusion (Eq. 5†)
    mnof = (ndf * omega).sum(axis=1)
    scores = np.clip(mnof, 0.0, 1.0)

    if return_diagnostics:
        diag = {
            "m_dagger": m_dagger, "m_eff": m_eff, "n_X": n_X,
            "neb_size": neb_size, "npb_sizes": npb_sizes[I], "pos_sizes": pos_sizes[I],
            "radii": radii, "thresholds": (d1, d2, d3),
            "gamma_hat": gamma_hat, "subspace": subspace_info,
            "mnof_range": (float(scores.min()), float(scores.max())),
        }
        return scores, diag
    return scores


# =============================================================================
# 6. BASELINE REFERENCE
# =============================================================================

def wncrod_baseline(data: np.ndarray, X_tem: np.ndarray, lammda: float = 0.5) -> np.ndarray:
    from scipy.spatial.distance import cdist

    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1
    n_X = int(X.sum())

    D1, D2, D3 = m / 3, m / 2, 0.9 * m
    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / lammda

    Lower = np.zeros((m, n))
    for col in range(m):
        RM = cdist(data[:, [col]], data[:, [col]], metric="cityblock") <= delta[col]
        Lower_temp = np.min(np.maximum(1 - RM, np.tile(X, (n, 1))), axis=1)
        Lower[col, :Lower_temp.shape[0]] = Lower_temp

    IB = np.tile(X, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    weight = np.zeros((n_X, m))
    for col in range(m):
        RM = cdist(data[:, [col]], data[:, [col]], metric="cityblock") <= delta[col]
        for i in range(n_X):
            temp = RM[X_tem[i], :]
            weight[i, col] = 1.0 - np.sqrt(np.sum(np.minimum(temp, X)) / n_X)

    D_tem = np.zeros((n, n))
    for col in range(m):
        RM = cdist(data[:, [col]], data[:, [col]], metric="cityblock") <= delta[col]
        D_tem += RM
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
# 7. DATA LOADING
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
            "Original Mammography files not found at absolute path. Falling back to synthetic generator.",
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
# 8. EVALUATION EXECUTION
# =============================================================================

def evaluate_subset(name: str, X_data: np.ndarray, labels: np.ndarray, X_tem: np.ndarray) -> Dict:
    scores_base = wncrod_baseline(X_data, X_tem, lammda=0.5)
    roc_base = roc_auc_score(labels[X_tem], scores_base)
    ap_base = average_precision_score(labels[X_tem], scores_base)

    scores_iv, diag = wncrod_iv(X_data, X_tem, return_diagnostics=True)
    roc_iv = roc_auc_score(labels[X_tem], scores_iv)
    ap_iv = average_precision_score(labels[X_tem], scores_iv)

    n_X = len(X_tem)
    n_out = int(labels[X_tem].sum())

    return {
        "subset": name,
        "n_X": n_X,
        "n_out": n_out,
        "prev%": 100.0 * n_out / n_X,
        "base_ROC": roc_base,
        "base_AP": ap_base,
        "iv_ROC": roc_iv,
        "iv_AP": ap_iv,
        "dROC": roc_iv - roc_base,
        "dAP": ap_iv - ap_base,
        "m†": diag["m_dagger"],
        "m_eff": diag["m_eff"],
        "neb": diag["neb_size"],
        "spe": diag["subspace"]["used"],
        "radii": diag["radii"],
        "gamma_hat": diag["gamma_hat"],
        "pos_sizes": diag["pos_sizes"],
    }


def main():
    print("=" * 80)
    print("3WNCROD-IV — Corrected Reference Implementation (Gen 4)")
    print("=" * 80)

    X_data, labels, subsets = load_mammography()
    print(f"\nUniverse: N = {X_data.shape[0]}, attributes = {X_data.shape[1]}")

    results = [evaluate_subset(name, X_data, labels, subsets[name]) for name in ["MA1", "MA2", "MA3"]]

    print("\n" + "-" * 80)
    print(f"{'Subset':>6} | {'N':>6} | {'Out':>4} | {'Prev%':>6} | "
          f"{'Base ROC':>9} | {'Base AP':>8} | {'IV ROC':>8} | {'IV AP':>8} | "
          f"{'dROC':>7} | {'dAP':>7} | {'m†':>3} | {'m_eff':>5} | {'NEB':>4} | {'SPE':>5}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['subset']:>6} | {r['n_X']:>6} | {r['n_out']:>4} | {r['prev%']:>6.2f} | "
            f"{r['base_ROC']:>9.4f} | {r['base_AP']:>8.4f} | {r['iv_ROC']:>8.4f} | {r['iv_AP']:>8.4f} | "
            f"{r['dROC']:>+7.4f} | {r['dAP']:>+7.4f} | {r['m†']:>3} | {r['m_eff']:>5} | "
            f"{r['neb']:>4} | {'YES' if r['spe'] else 'NO':>5}"
        )
    print("-" * 80)

    print("\n" + "=" * 80)
    print("Diagnostic Summary (Corrected State)")
    print("=" * 80)
    for r in results:
        print(f"\n[{r['subset']}]")
        print(f"  Radii (ε†) = {np.round(r['radii'], 5).tolist()}")
        print(f"  γ̂ (weights) = {np.round(r['gamma_hat'], 4).tolist()}")
        print(f"  POS sizes   = {r['pos_sizes'].tolist()}")

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()