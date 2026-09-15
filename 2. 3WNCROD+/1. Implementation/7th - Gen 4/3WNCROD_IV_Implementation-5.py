#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WNCROD-Adaptive: Unified Data-Driven Autonomous Outlier Detection Architecture
Eliminates static hyperparameters via continuous data-induced modulation:
  1. Continuous Subspace Spectral Gating: β(η_r) = σ(12 * (η_r - 0.70))
  2. Entropy/Divergence-Adaptive Blending Exponent: α = 1.0 - 0.5 * tanh(2.5 * CV(γ))
  3. Continuous Tiered Evidentiary Granulation (Regularized Soft-Rarity)
  4. Unified Master Fusion Engine evaluated across λ ∈ [0.1, 2.0]
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
# 1. CONTINUOUS SUBSPACE ESTIMATION & SPECTRAL GATING
# =============================================================================

def compute_adaptive_subspace(data: np.ndarray, X_tem: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Computes SPE and its continuous activation weight β based on explained variance ratio η_r.
    Returns:
        X_extended : data matrix with synthesized SPE appended
        beta       : continuous gating factor in [0, 1]
        eta_r      : raw spectral energy ratio
    """
    numeric_mask = np.all(data <= 1.0, axis=0) & ~np.all(np.isnan(data), axis=0)
    num_idx = np.flatnonzero(numeric_mask)
    m_num = int(num_idx.size)

    if m_num < 2:
        return data, 0.0, 0.0

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
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    total_var = float(eigvals.sum())
    eta_r = float(eigvals[:r_target].sum() / max(total_var, _EPS))

    # Continuous Sigmoidal Gating: sharp transition centered at η_0 = 0.70
    beta = float(1.0 / (1.0 + np.exp(-12.0 * (eta_r - 0.70))))

    V_r = eigvecs[:, :r_target]
    projected = Z @ V_r
    spe = np.einsum("ij,ij->i", Z, Z) - np.einsum("ik,ik->i", projected, projected)
    spe = np.maximum(spe, 0.0)

    lo, hi = float(spe[X_tem].min()), float(spe[X_tem].max())
    c_spe = np.clip((spe - lo) / (hi - lo + _EPS), 0.0, 1.0)

    X_ext = np.column_stack([data, c_spe])
    return X_ext, beta, eta_r


# =============================================================================
# 2. 3WNCROD-ADAPTIVE MASTER ENGINE
# =============================================================================

def wncrod_adaptive_core(
    data_ext: np.ndarray,
    X_tem: np.ndarray,
    lammda: float,
    beta_spe: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Unified Adaptive Engine with endogenous parameter induction.
    """
    n, m = data_ext.shape
    n_X = X_tem.size

    in_X = np.zeros(n, dtype=bool)
    in_X[X_tem] = True
    out_X = ~in_X

    numeric_mask = np.all(data_ext <= 1.0, axis=0)
    delta = np.zeros(m, dtype=np.float64)
    delta[numeric_mask] = np.std(data_ext[:, numeric_mask], axis=0) / max(lammda, _EPS)

    pos_full = np.empty((m, n), dtype=bool)
    neg_full = np.empty((m, n), dtype=bool)
    rel_X_list: List[np.ndarray] = []
    hit = np.empty((n_X, m), dtype=np.int64)

    for col in range(m):
        if numeric_mask[col]:
            diff = np.abs(data_ext[:, col][:, None] - data_ext[:, col][None, :])
            rel = diff <= delta[col]
        else:
            rel = (data_ext[:, col][:, None] == data_ext[:, col][None, :])

        pos_full[col] = ~rel[:, out_X].any(axis=1)
        neg_full[col] = ~rel[:, in_X].any(axis=1)

        rel_X = rel[np.ix_(X_tem, X_tem)]
        hit[:, col] = rel_X.sum(axis=1)
        rel_X_list.append(rel_X)

    # Three-Way Granulation on X
    lower = pos_full[:, X_tem]
    ib = (~lower).astype(np.int8)
    neb = ib.all(axis=0)
    npb = ib & ~neb[None, :]

    neb_size = int(neb.sum())
    npb_sizes = npb.sum(axis=1)
    lower_sizes = lower.sum(axis=1)

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

    # 1. Tiered Evidentiary NDF
    w_neb_r = 1.0 - neb_size / float(n_X)
    w_npb_r = 1.0 - npb_sizes / float(n_X)
    w_pos_r = 1.0 - lower_sizes / float(n_X)

    denom_v4 = w_neb_r * neb_size + w_npb_r * npb_sizes + w_pos_r * lower_sizes
    numer_v4 = (
        w_neb_r * neb_cnt[:, None]
        + w_npb_r[None, :] * npb_cnt
        + w_pos_r[None, :] * lower_cnt
    )
    ndf = np.zeros_like(numer_v4)
    ok = denom_v4 > _EPS
    ndf[:, ok] = numer_v4[:, ok] / denom_v4[ok][None, :]
    ndf[:, ~ok] = (neb_cnt[:, None] + npb_cnt + lower_cnt)[:, ~ok] / float(n_X)

    # 2. Endogenous Exponent Induction from Attribute Divergence
    gamma = (pos_full.sum(axis=1) + neg_full.sum(axis=1)) / float(n)
    gamma_mean = float(np.mean(gamma))
    gamma_std = float(np.std(gamma))
    cv_gamma = gamma_std / max(gamma_mean, _EPS)

    # α adapts smoothly between 0.5 (high contrast) and 1.0 (uniform/low signal)
    alpha = float(1.0 - 0.5 * np.tanh(2.5 * cv_gamma))

    # Base prevalence weights
    w_base = (1.0 - np.sqrt(hit / float(n_X))) / float(m)

    # Continuous modulation of SPE attribute
    if beta_spe > 0.0:
        w_base[:, -1] *= beta_spe

    gamma_max = float(gamma.max()) if gamma.size > 0 else 0.0
    gamma_hat = np.clip(gamma / max(gamma_max, _EPS), 0.10, 1.0)

    # Calibrated Geometric Synthesis
    omega = (np.clip(w_base, 0.0, None) ** alpha) * ((gamma_hat[None, :] / float(m)) ** (1.0 - alpha))

    # Master Fusion
    mnof = (ndf * omega).sum(axis=1)
    diagnostics = {"alpha": alpha, "cv_gamma": cv_gamma, "beta_spe": beta_spe}
    return mnof, diagnostics


# =============================================================================
# 3. BASELINE WNCROD ENGINE (Reference)
# =============================================================================

def wncrod_baseline_core(data: np.ndarray, X_tem: np.ndarray, lammda: float) -> np.ndarray:
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1
    n_X = int(X.sum())

    D1, D2, D3 = m / 3.0, m / 2.0, 0.9 * m
    delta = np.zeros(m)
    ID = np.all(data <= 1.0, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / max(lammda, _EPS)

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

    return np.mean(((NEB_num + Lower_num + NPB_num) / n_X) * weight, axis=1)


# =============================================================================
# 4. DATA LOADING (Synthetic Fallback Enabled)
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
        warnings.warn("Original Mammography files not found. Using synthetic generator.", UserWarning)
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
# 5. BENCHMARK SUITE
# =============================================================================

def run_adaptive_benchmark(
    name: str,
    X_data: np.ndarray,
    labels: np.ndarray,
    X_tem: np.ndarray,
    grid: np.ndarray = _LAMBDA_GRID,
) -> Dict:
    y_true = labels[X_tem]
    X_ext, beta_spe, eta_r = compute_adaptive_subspace(X_data, X_tem)

    base_ap, base_roc = [], []
    adp_ap, adp_roc = [], []
    last_diag = {}

    for lam in grid:
        # Baseline execution
        s_base = wncrod_baseline_core(X_data, X_tem, lammda=lam)
        base_ap.append(average_precision_score(y_true, s_base))
        base_roc.append(roc_auc_score(y_true, s_base))

        # Adaptive Master execution
        s_adp, diag = wncrod_adaptive_core(X_ext, X_tem, lammda=lam, beta_spe=beta_spe)
        adp_ap.append(average_precision_score(y_true, s_adp))
        adp_roc.append(roc_auc_score(y_true, s_adp))
        last_diag = diag

    idx_b_ap = int(np.argmax(base_ap))
    idx_b_roc = int(np.argmax(base_roc))
    idx_a_ap = int(np.argmax(adp_ap))
    idx_a_roc = int(np.argmax(adp_roc))

    return {
        "subset": name,
        "n_X": len(X_tem),
        "n_out": int(y_true.sum()),
        "prev%": 100.0 * y_true.sum() / len(X_tem),
        "eta_r": eta_r,
        "beta_spe": beta_spe,
        "alpha": last_diag["alpha"],
        "cv_gamma": last_diag["cv_gamma"],
        "base_best_ap": base_ap[idx_b_ap],
        "base_best_ap_lam": grid[idx_b_ap],
        "base_best_roc": base_roc[idx_b_roc],
        "base_best_roc_lam": grid[idx_b_roc],
        "adp_best_ap": adp_ap[idx_a_ap],
        "adp_best_ap_lam": grid[idx_a_ap],
        "adp_best_roc": adp_roc[idx_a_roc],
        "adp_best_roc_lam": grid[idx_a_roc],
        "delta_ap": adp_ap[idx_a_ap] - base_ap[idx_b_ap],
        "delta_roc": adp_roc[idx_a_roc] - base_roc[idx_b_roc],
    }


def main():
    print("=" * 90)
    print("3WNCROD-ADAPTIVE — Autonomous Master Architecture (λ ∈ [0.1, 2.0], step 0.1)")
    print("=" * 90)

    X_data, labels, subsets = load_mammography()
    print(f"\nUniverse: N = {X_data.shape[0]}, attributes = {X_data.shape[1]}")

    results = []
    for name in ["MA1", "MA2", "MA3"]:
        print(f"\nEvaluating Adaptive Architecture on [{name}] ...", end="", flush=True)
        res = run_adaptive_benchmark(name, X_data, labels, subsets[name])
        results.append(res)
        print(" Done.")

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS: BASELINE vs 3WNCROD-ADAPTIVE")
    print("=" * 90)
    header = (
        f"{'Subset':>6} | {'N':>6} | {'Out':>4} | {'Prev%':>6} | "
        f"{'Base AP':>8} (λ) | {'Adp AP':>8} (λ) | {'Δ AP':>7} | "
        f"{'Base ROC':>8} (λ) | {'Adp ROC':>8} (λ) | {'Δ ROC':>7}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r['subset']:>6} | {r['n_X']:>6} | {r['n_out']:>4} | {r['prev%']:>6.2f} | "
            f"{r['base_best_ap']:>8.4f} ({r['base_best_ap_lam']:>3.1f}) | "
            f"{r['adp_best_ap']:>8.4f} ({r['adp_best_ap_lam']:>3.1f}) | "
            f"{r['delta_ap']:>+7.4f} | "
            f"{r['base_best_roc']:>8.4f} ({r['base_best_roc_lam']:>3.1f}) | "
            f"{r['adp_best_roc']:>8.4f} ({r['adp_best_roc_lam']:>3.1f}) | "
            f"{r['delta_roc']:>+7.4f}"
        )
    print("-" * 90)

    print("\n" + "=" * 90)
    print("ENDOGENOUSLY INDUCED PARAMETERS PER DATASET")
    print("=" * 90)
    print(f"{'Subset':>6} | {'Spectral η_r':>12} | {'Subspace β':>10} | {'Divergence CV(γ)':>16} | {'Induced α':>10}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['subset']:>6} | {r['eta_r']:>12.4f} | {r['beta_spe']:>10.4f} | "
            f"{r['cv_gamma']:>16.4f} | {r['alpha']:>10.4f}"
        )
    print("-" * 65)
    print("Execution complete.")


if __name__ == "__main__":
    main()