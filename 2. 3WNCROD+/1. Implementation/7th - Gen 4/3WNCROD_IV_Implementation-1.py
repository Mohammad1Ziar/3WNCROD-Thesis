#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WNCROD-IV: Topologically-Stabilized Neighborhood Characteristic Region Outlier Detection
with Adaptive Multi-Granular Fusion — Complete Reference Implementation (Gen 4)

Implements Algorithm 1† from the Fourth-Generation Mathematical Treatise:
  - Eq. 8†   : Topological Stability Radius (k*-gap + robust scale + stability-matched λ)
  - Alg. 0†  : Conditional Subspace Augmentation (SPE with η_r gate)
  - Eq. 11†  : Adaptive NOM Thresholds (m_eff-informed)
  - Eq. 10†  : Calibrated Discriminative Weights (codomain-restored)
  - Eq. 4†   : Tiered Evidentiary NDF (IDF-style region rarity)
  - Eq. 5†   : Stabilized MNOF Fusion

Benchmarks on Mammography MA1/MA2/MA3 subsets. Reports AP (primary) and ROC-AUC (secondary).
If the original Mammography files are unavailable, a statistically faithful synthetic dataset
is generated automatically with a warning.

Author: Research Team
Date: 2026-09-08
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

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
    """Normalised MAD: 1.4826 * med|x - med(x)|. Breakdown point 0.5."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    return float(_MADN_C * np.median(np.abs(v - med)))


def _iqr(vals: np.ndarray) -> float:
    """Inter-quartile range Q3 - Q1 (type-7 linear quantile)."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    q1, q3 = np.percentile(v, [25.0, 75.0])
    return float(q3 - q1)


def _std(vals: np.ndarray) -> float:
    """Population standard deviation (identical to np.std default)."""
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

def _min_neighbors_count(sorted_vals: np.ndarray, radius: float) -> int:
    """
    For a sorted 1-D array, return the minimum number of points that fall
    within `radius` of any point (including the point itself).
    Two-pointer sliding window: O(n).
    """
    n = sorted_vals.size
    if n == 0 or radius < 0:
        return 0
    min_count = n
    left = 0
    right = 0
    for i in range(n):
        while left < n and sorted_vals[i] - sorted_vals[left] > radius:
            left += 1
        while right < n and sorted_vals[right] - sorted_vals[i] <= radius:
            right += 1
        count = right - left
        if count < min_count:
            min_count = count
            if min_count <= 1:
                break
    return min_count


def _find_stability_lambda(
    sorted_vals: np.ndarray,
    scale: float,
    k_star: int,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> float:
    """
    Binary search for the unique λ† such that
        min_x |{y ∈ X : |x - y| ≤ scale / λ†}| = k_star.
    Existence and uniqueness follow from Theorem 1 (treatise).
    """
    if scale <= 0.0 or k_star <= 0:
        return 1.0

    lo, hi = 1e-6, 1e6
    for _ in range(max_iter):
        if hi - lo < tol * lo:
            break
        mid = (lo + hi) / 2.0
        radius = scale / mid
        min_n = _min_neighbors_count(sorted_vals, radius)
        if min_n > k_star:
            lo = mid
        else:
            hi = mid
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

    Per numeric attribute:
        ε† = max( ρ^(k*),  σ̂ / λ†,  1/n_X )
    where
        k*      = ceil(log2 n_X)
        ρ^(k*)  = max_i (v_(i+k*) - v_(i))       [k*-gap covering radius]
        σ̂       = min+(MADN, IQR/1.34)           [robust scale]
        λ†      = stability-matched adjustment   [binary search]

    Categorical attributes receive ε = 0 (exact-match granules).
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

        # --- Robust scale σ̂ ---
        scale = _min_positive(_madn(x_vals), _iqr(x_vals) / _IQR_C)

        # --- k*-gap covering radius ρ ---
        if x_sorted.size > k_star:
            gaps = x_sorted[k_star:] - x_sorted[:-k_star]
            rho = float(gaps.max()) if gaps.size > 0 else 0.0
        else:
            rho = 0.0

        # --- Stability-matched λ† ---
        if scale > 0.0:
            lambda_dagger = _find_stability_lambda(x_sorted, scale, k_star)
            radius_scale = scale / lambda_dagger if lambda_dagger > 0.0 else 0.0
        else:
            radius_scale = 0.0

        radii[j] = max(rho, radius_scale, floor)

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
    """
    Algorithm 0†: synthesise c_{m+1} = minmax(SPE) only when the dominant
    correlation subspace explains at least η_min of total variance.

    Returns:
        data_aug      : (n, m†) augmented data matrix
        numeric_mask_aug : (m†,) updated type mask
        info          : diagnostic dict with keys {used, eta_r, r, reason}
    """
    n, m = data.shape
    num_idx = np.flatnonzero(numeric_mask)
    m_num = int(num_idx.size)

    info: Dict = {"used": False, "eta_r": 0.0, "r": 0, "reason": ""}

    if m_num < 2:
        info["reason"] = "m_num < 2 (degenerate): C† = C"
        return data, numeric_mask, info

    r_target = max(0, min(int(np.ceil(np.log2(m_num + 1))), m_num - 1))
    if r_target == 0:
        info["reason"] = "r_target = 0: no residual direction"
        return data, numeric_mask, info

    # --- Step 1: robust standardisation using statistics from X ---
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

    # --- Step 2: explained variance ratio on X ---
    Z_X = Z[X_tem]
    if Z_X.shape[0] < 2:
        info["reason"] = "|X| < 2: cannot estimate covariance"
        return data, numeric_mask, info

    cov = np.cov(Z_X, rowvar=False, bias=True)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[::-1]  # descending

    total_var = eigvals.sum()
    if total_var <= _EPS:
        info["reason"] = "zero total variance: C† = C"
        return data, numeric_mask, info

    eta_r = eigvals[:r_target].sum() / total_var
    info["eta_r"] = float(eta_r)
    info["r"] = r_target

    if eta_r < eta_min:
        info["reason"] = f"η_r = {eta_r:.3f} < {eta_min}: no dominant subspace"
        return data, numeric_mask, info

    # --- Step 3: dominant directions & SPE ---
    _, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, ::-1]
    V_r = eigvecs[:, :r_target]

    projected = Z @ V_r
    spe = np.einsum("ij,ij->i", Z, Z) - np.einsum("ik,ik->i", projected, projected)
    spe = np.maximum(spe, 0.0)

    # --- Step 4: min-max injection ---
    lo, hi = float(spe.min()), float(spe.max())
    if hi - lo <= _EPS:
        info["reason"] = "SPE constant: carries no information"
        return data, numeric_mask, info

    c_m1 = (spe - lo) / (hi - lo)
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
    3WNCROD-IV: compute MNOF†(x) for every x ∈ X.

    Parameters
    ----------
    data : (n, m) ndarray
        Condition attributes. Numeric attributes must be min-max normalised
        into [0, 1]; nominal attributes must be integer codes.
    X_tem : (n_X,) int array
        Indices of the concept X (selected condition subdata).
    eta_min : float
        Explained-variance threshold for conditional subspace augmentation.
    alpha : float
        Blend exponent for Eq. 10† (0.5 = Jeffreys-symmetric).
    return_diagnostics : bool
        If True, also return a dict with internal state for inspection.

    Returns
    -------
    MNOF† : (n_X,) float array
        Outlier scores in [0, 1]; larger = more outlying.
    diagnostics : dict (optional)
        Granulation sizes, radii, thresholds, subspace info, etc.
    """
    data = np.asarray(data, dtype=np.float64)
    n, m = data.shape
    X_tem = np.asarray(X_tem, dtype=int)
    n_X = X_tem.size

    if n_X == 0:
        return (np.zeros(0, dtype=np.float64), {}) if return_diagnostics else np.zeros(0, dtype=np.float64)

    in_X = np.zeros(n, dtype=bool)
    in_X[X_tem] = True
    out_X = ~in_X

    # --- Attribute typing (identical to reference: numeric iff all ≤ 1) ---
    numeric_mask = np.all(data <= 1.0, axis=0) & ~np.all(np.isnan(data), axis=0)

    # --- Alg. 0†: conditional subspace augmentation ---
    data_aug, numeric_mask_aug, subspace_info = _conditional_subspace_augmentation(
        data, numeric_mask, X_tem, eta_min
    )
    m_dagger = data_aug.shape[1]

    # --- Eq. 8†: topological stability radii ---
    radii = _compute_stability_radii(data_aug, numeric_mask_aug, X_tem)

    # --- Algorithm 1†: three-way granulation ---
    pos = np.empty((m_dagger, n), dtype=bool)      # POS_j over all n objects
    neg = np.empty((m_dagger, n), dtype=bool)      # NEG_j
    hit = np.empty((n_X, m_dagger), dtype=np.int64)  # |n_j(x) ∩ X| for x ∈ X
    overlap = np.zeros((n_X, n_X), dtype=np.int32)   # NOM accumulation on X×X

    for j in range(m_dagger):
        col = data_aug[:, j]
        if numeric_mask_aug[j]:
            dist = np.abs(col[:, None] - col[None, :])
            rel = dist <= radii[j]
        else:
            rel = (col[:, None] == col[None, :])

        # POS_j(x)  ⇔  n_j(x) ∩ (U \ X) = ∅
        pos[j] = ~rel[:, out_X].any(axis=1)
        # NEG_j(x)  ⇔  n_j(x) ∩ X = ∅
        neg[j] = ~rel[:, in_X].any(axis=1)
        # hit counts for objects in X
        hit[:, j] = rel[X_tem][:, X_tem].sum(axis=1)
        # accumulate overlap on X×X for NOM
        overlap += rel[np.ix_(X_tem, X_tem)].astype(np.int32)

    # NIB, NEB, NPB
    nib = in_X[None, :] & ~pos
    neb = nib.all(axis=0) if m_dagger > 0 else np.zeros(n, dtype=bool)
    npb = nib & ~neb[None, :]

    # --- Def. 6: informative attributes ---
    neb_size = int(neb[X_tem].sum())
    npb_sizes = npb[:, X_tem].sum(axis=1).astype(np.int64)
    pos_sizes = pos[:, X_tem].sum(axis=1).astype(np.int64)
    neg_sizes = neg[:, X_tem].sum(axis=1).astype(np.int64)

    informative = (neb_size + npb_sizes) > 0
    I = np.flatnonzero(informative)
    m_eff = int(I.size)

    # Edge case: no informative attributes
    if m_eff == 0:
        scores = np.zeros(n_X, dtype=np.float64)
        if return_diagnostics:
            diag = {
                "m_dagger": m_dagger, "m_eff": 0, "n_X": n_X,
                "neb_size": neb_size, "radii": radii, "subspace": subspace_info,
                "reason": "m_eff = 0: all attributes uninformative",
            }
            return scores, diag
        return scores

    # --- Eq. 11†: adaptive NOM thresholds ---
    d1 = m_eff / 3.0
    d2 = m_eff / 2.0
    d3 = 0.9 * m_eff

    # NOM restricted to X×X (Def. 9)
    nom = m_dagger - overlap

    # --- Eq. 3: characteristic region counts ---
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

    # --- Eq. 4†: tiered evidentiary NDF ---
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

    # --- Eq. 10†: calibrated discriminative weights ---
    hit_I = hit[:, I].astype(np.float64)  # (n_X, m_eff)
    prevalence = hit_I / n_x_f
    omega_raw = 1.0 - np.sqrt(np.clip(prevalence, 0.0, None))

    # Discriminability γ_j = (|POS_j| + |NEG_j|) / n  ;  γ̂_j = γ_j / max_k γ_k
    gamma = (pos_sizes + neg_sizes) / float(n)
    gamma_max = float(gamma[I].max()) if gamma[I].size > 0 else 0.0
    gamma_hat = np.ones(m_eff, dtype=np.float64) if gamma_max <= _EPS else gamma[I] / gamma_max

    omega = (1.0 / m_eff) * (omega_raw ** alpha) * (gamma_hat[None, :] ** (1.0 - alpha))

    # --- Eq. 5†: stabilized MNOF fusion ---
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
# 6. BASELINE WNCROD (for comparison)
# =============================================================================

def wncrod_baseline(data: np.ndarray, X_tem: np.ndarray, lammda: float = 0.5) -> np.ndarray:
    """
    Original 3WNCROD baseline (Algorithms 1–2, Eq. 8/4/10).
    Transcribed faithfully from the published reference implementation.
    """
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
# 7. DATA LOADING — Mammography benchmark with synthetic fallback
# =============================================================================

def _generate_synthetic_mammography(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate a statistically faithful synthetic Mammography dataset.
    Produces ~11 200 samples with ~1.5% outlier prevalence and the same
    feature structure (c1 nominal 0–5, c2–c6 continuous, c4/c5/c6 filters).
    """
    rng = np.random.default_rng(seed)
    n_total = 11200
    n_outliers = int(np.round(n_total * 0.015))  # ~168 outliers
    n_inliers = n_total - n_outliers

    # c1: nominal-like integer 0–5 (slightly imbalanced)
    c1_in = rng.choice([0, 1, 2, 3, 4, 5], size=n_inliers, p=[0.18, 0.22, 0.20, 0.18, 0.12, 0.10])
    c1_out = rng.choice([0, 1, 2, 3, 4, 5], size=n_outliers)

    # c2–c6: correlated continuous features driven by 2 latent factors
    latent1 = rng.normal(loc=0.0, scale=1.0, size=n_inliers)
    latent2 = rng.normal(loc=0.0, scale=0.6, size=n_inliers)
    c2_in = 0.70 * latent1 + 0.30 * latent2 + 0.20 * rng.normal(size=n_inliers)
    c3_in = 0.60 * latent1 - 0.40 * latent2 + 0.25 * rng.normal(size=n_inliers)
    c4_in = 0.50 * latent1 + 0.20 * latent2 + 0.30 * rng.normal(size=n_inliers)
    c5_in = -0.40 * latent1 + 0.60 * latent2 + 0.20 * rng.normal(size=n_inliers)
    c6_in = 0.30 * latent1 + 0.50 * latent2 + 0.25 * rng.normal(size=n_inliers)

    # Outliers: break correlation, push to extremes on subsets of features
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
    """
    Load Mammography benchmark. Tries original file paths first;
    falls back to synthetic data generation with a warning.

    Returns
    -------
    X : (n, 6) ndarray — condition attributes (c1–c6, c1 already integer-coded)
    y : (n,) ndarray   — binary labels (0 = inlier, 1 = outlier)
    subsets : dict     — {"MA1": indices, "MA2": indices, "MA3": indices}
    """
    csv_path = "/media/mine/Mine/3.Main-Ubuntu-Envs/2.Thesis-Env/3.Datasets/mammographyori.csv"
    mat_path = "/media/mine/Mine/3.Main-Ubuntu-Envs/2.Thesis-Env/3.Datasets/mammography.mat"

    use_synthetic = False
    if not (os.path.isfile(csv_path) and os.path.isfile(mat_path)):
        use_synthetic = True

    if use_synthetic:
        warnings.warn(
            "Original Mammography files not found. Using statistically faithful "
            "synthetic data for self-contained execution. Benchmark against "
            "original data is required for publication.",
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

    # Normalise numeric block (c2–c6) into [0, 1]; c1 stays as integer codes
    scaler = MinMaxScaler()
    X = X_raw.copy()
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    # Subset filters (identical to reference)
    ma_conditions = {
        "MA1": ("c4", -0.9, 2.0),
        "MA2": ("c5", -0.4, 1.0),
        "MA3": ("c6", -1.0, 1.0),
    }

    if use_synthetic:
        X_raw_df = pd.DataFrame(X_raw, columns=["c1", "c2", "c3", "c4", "c5", "c6"])
        X_raw_df["label"] = y
        subsets = {}
        for name, (feat, lo, hi) in ma_conditions.items():
            mask = (X_raw_df[feat] >= lo) & (X_raw_df[feat] <= hi)
            subsets[name] = X_raw_df[mask].index.to_numpy()
    else:
        subsets = {}
        for name, (feat, lo, hi) in ma_conditions.items():
            mask = (df[feat] >= lo) & (df[feat] <= hi)
            subsets[name] = mask[mask].index.to_numpy()

    return X, y, subsets


# =============================================================================
# 8. EVALUATION & REPORTING
# =============================================================================

def evaluate_subset(
    name: str,
    X_data: np.ndarray,
    labels: np.ndarray,
    X_tem: np.ndarray,
    lam_baseline: float = 0.5,
) -> Dict[str, float]:
    """Evaluate both baseline and 3WNCROD-IV on a single subset."""
    # Baseline
    scores_base = wncrod_baseline(X_data, X_tem, lammda=lam_baseline)
    roc_base = roc_auc_score(labels[X_tem], scores_base)
    ap_base = average_precision_score(labels[X_tem], scores_base)

    # 3WNCROD-IV
    scores_iv, diag = wncrod_iv(X_data, X_tem, return_diagnostics=True)
    roc_iv = roc_auc_score(labels[X_tem], scores_iv)
    ap_iv = average_precision_score(labels[X_tem], scores_iv)

    n_X = len(X_tem)
    n_out = int(labels[X_tem].sum())
    prev = 100.0 * n_out / n_X

    return {
        "subset": name,
        "n_X": n_X,
        "n_out": n_out,
        "prevalence_%": prev,
        "base_ROC": roc_base,
        "base_AP": ap_base,
        "iv_ROC": roc_iv,
        "iv_AP": ap_iv,
        "delta_ROC": roc_iv - roc_base,
        "delta_AP": ap_iv - ap_base,
        "m_dagger": diag["m_dagger"],
        "m_eff": diag["m_eff"],
        "neb_size": diag["neb_size"],
        "subspace_used": diag["subspace"]["used"],
        "subspace_reason": diag["subspace"]["reason"],
    }


def main():
    print("=" * 80)
    print("3WNCROD-IV — Fourth-Generation Reference Implementation")
    print("=" * 80)

    X_data, labels, subsets = load_mammography()
    print(f"\nUniverse: N = {X_data.shape[0]}, attributes = {X_data.shape[1]}")

    results: List[Dict] = []
    for name in ["MA1", "MA2", "MA3"]:
        X_tem = subsets[name]
        res = evaluate_subset(name, X_data, labels, X_tem)
        results.append(res)

    # Tabular report
    print("\n" + "-" * 80)
    print(f"{'Subset':>6} | {'N':>6} | {'Out':>4} | {'Prev%':>6} | "
          f"{'Base ROC':>9} | {'Base AP':>8} | {'IV ROC':>8} | {'IV AP':>8} | "
          f"{'dROC':>7} | {'dAP':>7} | {'m†':>3} | {'m_eff':>5} | {'NEB':>4} | {'SPE':>5}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['subset']:>6} | {r['n_X']:>6} | {r['n_out']:>4} | {r['prevalence_%']:>6.2f} | "
            f"{r['base_ROC']:>9.4f} | {r['base_AP']:>8.4f} | {r['iv_ROC']:>8.4f} | {r['iv_AP']:>8.4f} | "
            f"{r['delta_ROC']:>+7.4f} | {r['delta_AP']:>+7.4f} | {r['m_dagger']:>3} | {r['m_eff']:>5} | "
            f"{r['neb_size']:>4} | {'YES' if r['subspace_used'] else 'NO':>5}"
        )
    print("-" * 80)

    # Detailed diagnostics per subset
    print("\n" + "=" * 80)
    print("Per-Subset Diagnostic Detail")
    print("=" * 80)
    for name in ["MA1", "MA2", "MA3"]:
        X_tem = subsets[name]
        _, diag = wncrod_iv(X_data, X_tem, return_diagnostics=True)
        print(f"\n[{name}]  n_X = {diag['n_X']},  m† = {diag['m_dagger']},  m_eff = {diag['m_eff']}")
        print(f"  NEB size = {diag['neb_size']},  NPB sizes = {diag['npb_sizes'].tolist()}")
        print(f"  POS sizes = {diag['pos_sizes'].tolist()}")
        print(f"  Radii (ε†) = {np.round(diag['radii'], 6).tolist()}")
        print(f"  Thresholds (d1†, d2†, d3†) = ({diag['thresholds'][0]:.2f}, {diag['thresholds'][1]:.2f}, {diag['thresholds'][2]:.2f})")
        print(f"  γ̂ (informative attrs) = {np.round(diag['gamma_hat'], 4).tolist()}")
        print(f"  Subspace: {diag['subspace']['reason']}")
        print(f"  MNOF† range = [{diag['mnof_range'][0]:.6f}, {diag['mnof_range'][1]:.6f}]")

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()