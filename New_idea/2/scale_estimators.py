"""
scale_estimators.py
Robust scale estimators based on Yaro et al. (2024):
- MAD (Median Absolute Deviation) with Bn = 1.4826
- Sn estimator with finite-sample correction factor Cn
- Weighted Hybrid Scale Estimator
"""

import numpy as np


def get_correction_factor_cn(n: int) -> float:
    """
    Computes finite-sample correction factor Cn for Sn estimator (Eq. 6 in Yaro et al., 2024).
    """
    if n <= 1:
        return 1.0
    return (n / (n - 0.9)) if (n % 2 == 1) else 1.0


def sigma_mad(x: np.ndarray, b_n: float = 1.4826) -> float:
    """
    Computes MAD scale estimator (Eq. 3 in Yaro et al., 2024).
    sigma_MAD = B_n * Median(|x - Median(x)|)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n <= 1:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(b_n * mad)


def sigma_sn(x: np.ndarray, block_size: int = 2000) -> float:
    """
    Computes Sn scale estimator (Eq. 5 and 6 in Yaro et al., 2024).
    sigma_Sn = C_n * Median_m { Median_{k != m} |x_m - x_k| }
    Vectorized and chunked by block_size to prevent memory overflow on large N.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n <= 1:
        return 0.0

    c_n = get_correction_factor_cn(n)
    inner_medians = np.empty(n, dtype=np.float64)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        b = stop - start

        diff = np.abs(x[start:stop, None] - x[None, :])
        row_indices = np.arange(b)
        col_indices = np.arange(start, stop)
        diff[row_indices, col_indices] = np.nan

        inner_medians[start:stop] = np.nanmedian(diff, axis=1)

    return float(c_n * np.median(inner_medians))


def sigma_hybrid(x: np.ndarray, w: float = 0.5, b_n: float = 1.4826) -> float:
    """
    Computes Weighted Hybrid Scale Estimator (Section III-A in Yaro et al., 2024).
    sigma_hybrid = w * sigma_MAD + (1 - w) * sigma_Sn
    """
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"Weight w must be in [0, 1], got {w}")

    s_mad = sigma_mad(x, b_n=b_n)
    s_sn = sigma_sn(x)
    return float(w * s_mad + (1.0 - w) * s_sn)


def compute_hybrid_scale_vector(data: np.ndarray, id_mask: np.ndarray, w: float = 0.5) -> np.ndarray:
    """
    Computes sigma_hybrid for all numerical attributes (columns where id_mask is True).
    """
    m = data.shape[1]
    scale_vec = np.zeros(m, dtype=np.float64)
    numerical_cols = np.flatnonzero(id_mask)

    for col in numerical_cols:
        scale_vec[col] = sigma_hybrid(data[:, col], w=w)

    return scale_vec