#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3WNCROD -- baseline + isolated closed-form enhancement variants
===============================================================

Baseline
--------
X. Zhang, Z. Yuan, D. Miao, "Outlier Detection Using Three-Way Neighborhood
Characteristic Regions and Corresponding Fusion Measurement", IEEE TKDE 2023,
DOI 10.1109/TKDE.2023.3312108.

Equation / definition numbers below are those of the paper (Eq. 1-11,
Def. 1-9, Algorithms 1-2).  Starred objects (Eq. 4*, 8*, 9*, 10*, 5*,
Algorithm 0*) are those of the companion treatise
"3WNCROD_closed_form_enhancements.md".

Each enhancement is implemented as an *isolated* variant: exactly one
mechanism deviates from the baseline, every other stage is the untouched
baseline pipeline, so standalone effects are measurable.

Variant map
-----------
  V0   baseline                  Eq. 8 / 4 / 10, attribute set C      (oracle)
  V1   adaptive radius           Eq. 8*  (MADN + Silverman + 1/n)     (Sec. 2.1)
  V2   discriminative weights    Eq. 10* (gamma_hat blend)            (Sec. 2.3)
  V3   subspace augmentation     Alg. 0* + Eq. 9*/5* (SPE attribute)  (Sec. 2.2)
  V4   tiered evidence NDF       Eq. 4*  (region rarity weights w_R)  (Sec. 2.4)
  V5   robust scale only         MADN(c_j) / lambda                   (Sec. 2.1 ablation)
  V6   Silverman selector only   min+(MADN, IQR/1.34) / lambda        (Sec. 2.1 ablation)
  V7   AMISE rate only           0.9 * std(c_j) * n^(-1/5)            (Sec. 2.1 ablation)
  V8   resolution floor only     max(std(c_j)/lambda, 1/n)            (Sec. 2.1/4.1 ablation)
  VF   full                      Eq. 8* + 9* + 10* + 4* together

V5-V8 decompose Eq. 8* into its four independent factors (robust scale,
IQR co-selector, AMISE decay rate, resolution floor); each toggles exactly
one factor away from Eq. 8, so the contribution of every factor of the
adaptive-radius rule is separately measurable.

Deviations from the treatise, and why
-------------------------------------
1. Eq. 8*'s ``min(sigma_hat, IQR/1.34)`` is taken over the *strictly positive*
   candidates.  A literal ``min`` returns 0 whenever MADN = 0 (>= 50% ties),
   contradicting Sec. 4.1 ("MAD = 0 ... falls through to IQR").  With no
   positive candidate the rule returns 0, i.e. Sec. 4.1's "behaviourally
   categorical" exact-match case.
2. Algorithm 0*'s r = ceil(log2(m_num + 1)) is clamped to m_num - 1.  For
   m_num = 2 the unclamped value is 2 = m_num, which annihilates the residual
   space (J == 0) and would inject a constant attribute; the clamp keeps at
   least one residual direction.  If J is still numerically constant the
   augmentation is skipped, which is Sec. 2.2's C* = C degeneration.
3. Corollary 4 claims sum_j omega* <= 1, hence MNOF* in [0, 1].  That bound
   does not hold for the geometric blend of Eq. 10*: omega_j <= 1/m* gives
   sqrt(omega_j * gamma_hat_j) <= 1/sqrt(m*), so sum_j <= sqrt(m*) and
   MNOF* lies in [0, sqrt(m*)].  Rankings (ROC_AUC / AP) are unaffected, but an
   absolute cut mu (Def. 8) must be rescaled for V2/VF.  Eq. 10* is
   implemented exactly as written; alpha = 1 recovers Eq. 10 identically
   (asserted in __main__).
4. Eq. 4*'s denominator can vanish (POS_j = X gives w_POS,j = 0 while the two
   boundary regions are empty).  Guarded to NDF* = 0, the coherent IDF-style
   reading: a region that contains all of X carries no evidence.

Dependencies: NumPy, SciPy, Pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist  # baseline-fidelity reference path
from scipy.stats import rankdata  # ROC_AUC / AP helpers

__all__ = [
    "as_matrix", "infer_numeric_mask", "minmax_normalize", "attribute_distance",
    "heom_distance", "neighborhood_relation", "madn", "iqr", "compute_radii",
    "RADIUS_RULES", "synthesize_subspace_attribute", "SubspaceInfo", "granulate",
    "Granulation", "ndf_baseline", "ndf_tiered", "omega_baseline",
    "discriminability", "omega_discriminative", "ThreeWNCROD",
    "Variant0Baseline", "Variant1AdaptiveRadius", "Variant2DiscriminativeWeights",
    "Variant3SubspaceAugmented", "Variant4TieredEvidence", "Variant5RobustScale",
    "Variant6SilvermanSelector", "Variant7AmiseRate", "Variant8ResolutionFloor",
    "VariantFull", "VARIANTS", "build_all_variants", "roc_auc",
    "average_precision", "WNCROD_reference",
]

_EPS = 1e-12
_MADN_C = 1.4826  # MAD -> sigma consistency factor at the Gaussian
_SILVERMAN_C = 0.9  # Silverman (1986) rule-of-thumb constant
_IQR_C = 1.34  # IQR -> sigma consistency factor at the Gaussian

ArrayLike = Union[np.ndarray, pd.DataFrame, Sequence]


# =============================================================================
# 1. SHARED CORE ENGINE -- primitives common to every variant
# =============================================================================

def as_matrix(data: ArrayLike) -> np.ndarray:
    """Return a fresh float64 ``(n, m)`` matrix.  NaN = unknown value (Eq. 7)."""
    raw = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    arr = np.array(raw, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"data must be 2-D (n, m); got shape {arr.shape!r}.")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("data needs at least one object and one attribute.")
    if np.isinf(arr).any():
        raise ValueError("data contains +/-inf (NaN is allowed as HEOM 'unknown', inf is not).")
    return arr


def infer_numeric_mask(data: np.ndarray) -> np.ndarray:
    """
    Numeric/nominal split of the reference implementation: a column is numeric
    iff all its values are <= 1 (Eq. 6 maps numeric attributes into [0, 1] while
    nominal attributes carry integer codes).  NaN is neutral so that an unknown
    value cannot flip an attribute's type.
    """
    filled = np.where(np.isnan(data), -np.inf, data)
    return np.all(filled <= 1.0, axis=0)

def minmax_normalize(data: ArrayLike, numeric_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Eq. 6 -- min-max normalisation of the numeric block into [0, 1]."""
    out = as_matrix(data)
    mask = infer_numeric_mask(out) if numeric_mask is None else np.asarray(numeric_mask, dtype=bool)
    if mask.shape != (out.shape[1],):
        raise ValueError("numeric_mask must have one entry per attribute.")
    for j in np.flatnonzero(mask):
        col = out[:, j]
        if np.all(np.isnan(col)):
            continue
        lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        span = hi - lo
        if span <= _EPS:  # constant attribute -> collapse to 0, keep unknowns
            out[:, j] = np.where(np.isnan(col), np.nan, 0.0)
        else:
            out[:, j] = (col - lo) / span
    return out


def attribute_distance(col: np.ndarray, is_numeric: bool) -> np.ndarray:
    """
    ``d_{c_j}(x, y)`` of Eq. 7 for a single attribute -- the B = {c_j} case, for
    which ``HEOM_{c_j} = d_{c_j}``.  Unknown (NaN) values give distance 1; the
    diagonal is forced to 0 so ``x in n_{c_j}(x)`` always holds (reflexivity is
    required for POS ⊆ X, Prop. 1).  For numeric columns this is identical to
    ``cdist(col[:, None], col[:, None], metric='cityblock')``.
    """
    col = np.asarray(col, dtype=np.float64).ravel()
    unknown = np.isnan(col)
    safe = np.where(unknown, 0.0, col)
    if is_numeric:
        dist = np.abs(safe[:, None] - safe[None, :])
    else:
        dist = (safe[:, None] != safe[None, :]).astype(np.float64)
    if unknown.any():
        dist[unknown, :] = 1.0
        dist[:, unknown] = 1.0
    np.fill_diagonal(dist, 0.0)
    return dist


def heom_distance(data: ArrayLike, numeric_mask: Optional[np.ndarray] = None,
                  subset: Optional[Sequence[int]] = None) -> np.ndarray:
    """Eq. 7 -- full HEOM over an attribute subset B (shared primitive)."""
    arr = as_matrix(data)
    mask = infer_numeric_mask(arr) if numeric_mask is None else np.asarray(numeric_mask, dtype=bool)
    cols = np.arange(arr.shape[1]) if subset is None else np.atleast_1d(np.asarray(subset, dtype=int))
    acc = np.zeros((arr.shape[0], arr.shape[0]), dtype=np.float64)
    for j in cols:
        d = attribute_distance(arr[:, j], bool(mask[j]))
        acc += d * d
    return np.sqrt(acc)

def neighborhood_relation(col: np.ndarray, eps: float, is_numeric: bool) -> np.ndarray:
    """``nr_{c_j} = {(x, y) | d_{c_j}(x, y) <= eps}`` as an (n, n) boolean matrix."""
    return attribute_distance(col, is_numeric) <= eps


def _finite(col: np.ndarray) -> np.ndarray:
    return col[~np.isnan(col)]


def madn(col: np.ndarray) -> float:
    """Normalised MAD ``1.4826 * med|x - med(x)|`` -- breakdown point 0.5."""
    vals = _finite(np.asarray(col, dtype=np.float64))
    if vals.size == 0:
        return 0.0
    return float(_MADN_C * np.median(np.abs(vals - np.median(vals))))


def iqr(col: np.ndarray) -> float:
    """``Q3 - Q1`` with the classical type-7 (linear) quantile estimator (Sec. 4.1)."""
    vals = _finite(np.asarray(col, dtype=np.float64))
    if vals.size == 0:
        return 0.0
    q1, q3 = np.percentile(vals, [25.0, 75.0])
    return float(q3 - q1)


def _std(col: np.ndarray) -> float:
    """Population standard deviation, exactly the ``np.std`` of Eq. 8."""
    vals = _finite(np.asarray(col, dtype=np.float64))
    return 0.0 if vals.size == 0 else float(np.std(vals))


def _min_positive(*values: float) -> float:
    """min over the strictly positive candidates, 0.0 if there is none (Sec. 4.1)."""
    positive = [float(v) for v in values if np.isfinite(v) and v > 0.0]
    return min(positive) if positive else 0.0


# =============================================================================
# 2. NEIGHBORHOOD RADII -- Eq. 8 and the Eq. 8* family
# =============================================================================

RADIUS_RULES = (
    "baseline",          # Eq. 8   : std(c_j) / lambda
    "baseline_floor",    # Eq. 8   + 1/n resolution floor            (V8)
    "madn",              # Eq. 8   with MADN replacing std            (V5)
    "silverman_lambda",  # min+(MADN, IQR/1.34) / lambda              (V6)
    "amise_std",         # 0.9 * std * n^(-1/5), lambda-free          (V7)
    "silverman_amise",   # Eq. 8*  : full closed-form rule            (V1, VF)
)

_LAMBDA_RULES = ("baseline", "baseline_floor", "madn", "silverman_lambda")


def compute_radii(data: np.ndarray, numeric_mask: np.ndarray, rule: str = "baseline",
                  lammda: float = 1.0) -> np.ndarray:
    """
    Neighborhood radius per attribute.  Categorical attributes always receive
    ``eps = 0`` (exact-match granules), identically under every rule -- Eq. 8*'s
    first case is Eq. 8's first case verbatim (Sec. 4.3).

    Numeric branches
    ----------------
    baseline          std(c_j) / lambda                                    (Eq. 8)
    baseline_floor    max(std(c_j)/lambda, 1/n)   if std > 0 else 0
    madn              MADN(c_j) / lambda
    silverman_lambda  min+(MADN, IQR/1.34) / lambda
    amise_std         0.9 * std(c_j) * n^(-1/5)   if std > 0 else 0
    silverman_amise   max(0.9 * min+(MADN, IQR/1.34) * n^(-1/5), 1/n)     (Eq. 8*)

    ``min+`` is the minimum over the strictly positive candidates, so MADN = 0
    falls through to IQR/1.34 and MADN = IQR = 0 yields eps = 0 (Sec. 4.1).
    The 1/n floor is applied only where the scale estimate is positive, so a
    constant numeric attribute keeps the exact-match behaviour of Eq. 8.
    """
    if rule not in RADIUS_RULES:
        raise ValueError(f"unknown radius rule {rule!r}; expected one of {RADIUS_RULES}.")
    n, m = data.shape
    if rule in _LAMBDA_RULES:
        if lammda is None or not np.isfinite(lammda) or lammda <= 0.0:
            raise ValueError(f"lammda must be finite and > 0 for radius rule {rule!r}.")
    floor = 1.0 / float(n)
    eps = np.zeros(m, dtype=np.float64)
    for j in np.flatnonzero(np.asarray(numeric_mask, dtype=bool)):
        col = data[:, j]
        if rule == "baseline":
            value = _std(col) / lammda
        elif rule == "baseline_floor":
            scale = _std(col)
            value = max(scale / lammda, floor) if scale > 0.0 else 0.0
        elif rule == "madn":
            value = madn(col) / lammda
        elif rule == "silverman_lambda":
            value = _min_positive(madn(col), iqr(col) / _IQR_C) / lammda
        elif rule == "amise_std":
            scale = _std(col)
            value = _SILVERMAN_C * scale * n ** (-0.2) if scale > 0.0 else 0.0
        else:  # "silverman_amise" == Eq. 8*
            scale = _min_positive(madn(col), iqr(col) / _IQR_C)
            value = max(_SILVERMAN_C * scale * n ** (-0.2), floor) if scale > 0.0 else 0.0
        eps[j] = max(float(value), 0.0)
    return eps

# =============================================================================
# 3. ALGORITHM 0* -- subspace attribute synthesis (SPE / Q-statistic)
# =============================================================================

@dataclass
class SubspaceInfo:
    """Outcome of Algorithm 0*."""
    used: bool
    m_num: int
    r: int
    reason: str = ""
    attribute: Optional[np.ndarray] = None  # c_{m+1} = minmax(J) in [0, 1]
    spe: Optional[np.ndarray] = None  # raw J(x) >= 0


def _power_subspace(Z: np.ndarray, r: int, power_iters: int, random_state: int) -> np.ndarray:
    """
    Matrix-free deflationary power iteration (Step 2 of Algorithm 0*).
    Neither ``Z^T Z`` nor any inverse is formed; one iteration costs two
    matrix-vector products, i.e. O(n * m_num).  Returns an orthonormal
    ``(m_num, r')`` basis with ``r' <= r`` (short if the residual rank runs out).
    """
    rng = np.random.default_rng(random_state)
    deflated = Z.copy()
    basis: list = []
    for _ in range(r):
        vec = rng.standard_normal(Z.shape[1])
        norm = float(np.linalg.norm(vec))
        if norm <= _EPS:
            break
        vec /= norm
        converged = False
        for _t in range(max(1, int(power_iters))):
            candidate = deflated.T @ (deflated @ vec)
            for prev in basis:  # re-orthogonalisation: numerical hygiene only
                candidate -= (candidate @ prev) * prev
            norm = float(np.linalg.norm(candidate))
            if norm <= _EPS:
                converged = False
                break
            vec = candidate / norm
            converged = True
        if not converged:
            break
        basis.append(vec)
        deflated = deflated - np.outer(deflated @ vec, vec)
    if not basis:
        return np.zeros((Z.shape[1], 0), dtype=np.float64)
    return np.column_stack(basis)

def synthesize_subspace_attribute(data: np.ndarray, numeric_mask: np.ndarray,
                                  n_components: Optional[int] = None,
                                  power_iters: int = 10,
                                  random_state: int = 0) -> SubspaceInfo:
    """
    Algorithm 0* -- synthesise one numeric attribute
    ``c_{m+1}(x) = minmax(J(x))`` where ``J`` is the squared prediction error
    (SPE / PCA Q-statistic) of ``x`` against the dominant rank-r correlation
    subspace of the robustly standardised numeric block.

    Steps: (1) robust standardisation reusing ``sigma_hat`` of Eq. 8*;
    (2) deflationary power iteration for ``v_1..v_r``,
    ``r = ceil(log2(m_num + 1))`` clamped to ``m_num - 1``;
    (3) ``J(x) = ||z_x||^2 - sum_k <z_x, v_k>^2 >= 0`` (Bessel);
    (4) min-max injection as the (m+1)-th numeric attribute.

    Degenerations (Sec. 2.2 / 4.3) return ``used=False`` so that ``C* = C``:
    fewer than two numeric attributes, no residual direction, zero numeric
    rank, or a constant SPE (which would inject a useless constant attribute).
    """
    n = data.shape[0]
    num_idx = np.flatnonzero(np.asarray(numeric_mask, dtype=bool))
    m_num = int(num_idx.size)
    if m_num < 2:
        return SubspaceInfo(False, m_num, 0, "m_num < 2 (Sec. 2.2 degenerate case): C* = C")

    # ---- Step 1: robust standardisation -------------------------------------
    Z = np.empty((n, m_num), dtype=np.float64)
    for k, j in enumerate(num_idx):
        col = data[:, j]
        centre = 0.0 if np.all(np.isnan(col)) else float(np.nanmedian(col))
        scale = madn(col)
        if scale <= 0.0:  # >= 50% ties -> Sec. 4.1 fall-through chain
            scale = iqr(col) / _IQR_C
        if scale <= 0.0:
            scale = _std(col)
        if scale <= 0.0:  # constant attribute: z == 0, contributes no residual
            scale = 1.0
        z = (col - centre) / scale
        Z[:, k] = np.where(np.isnan(z), 0.0, z)  # unknown -> on-manifold

    # ---- Step 2: dominant correlation directions ----------------------------
    r_target = int(np.ceil(np.log2(m_num + 1))) if n_components is None else int(n_components)
    r_target = max(0, min(r_target, m_num - 1))
    if r_target == 0:
        return SubspaceInfo(False, m_num, 0, "r clamped to 0: no residual direction available")
    basis = _power_subspace(Z, r_target, power_iters, random_state)
    r = int(basis.shape[1])
    if r == 0:
        return SubspaceInfo(False, m_num, 0, "numeric block has numerically zero rank")

    # ---- Step 3: squared prediction error (Jackson & Mudholkar, 1979) -------
    projected = Z @ basis
    spe = np.einsum("ij,ij->i", Z, Z) - np.einsum("ik,ik->i", projected, projected)
    spe = np.maximum(spe, 0.0)  # Bessel's inequality, guarded against round-off

    # ---- Step 4: attribute injection ---------------------------------------
    lo, hi = float(spe.min()), float(spe.max())
    if hi - lo <= _EPS:
        return SubspaceInfo(False, m_num, r, "SPE constant: augmentation carries no information",
                            spe=spe)
    return SubspaceInfo(True, m_num, r, "augmented with c_{m+1} = minmax(SPE)",
                        attribute=(spe - lo) / (hi - lo), spe=spe)


# =============================================================================
# 4. ALGORITHM 1 / 1* -- three-way granulation and characteristic cardinalities
# =============================================================================

@dataclass
class Granulation:
    """Everything Algorithms 1/1* produce, for the (possibly augmented) set C*."""
    n: int
    m: int  # = m* : number of granulated attributes
    n_X: int
    radii: np.ndarray  # (m,)     eps_{c_j}
    x_index: np.ndarray  # (n_X,) object ids of X, in caller order
    pos: np.ndarray  # (m, n) bool  POS_{nr_{c_j}}(X)              Eq. 1
    neg: np.ndarray  # (m, n) bool  NEG_{nr_{c_j}}(X)              Eq. 1  (Alg. 1*)
    nib: np.ndarray  # (m, n) bool  NIB_{nr_{c_j}}(X) = X - POS    Eq. 2
    neb: np.ndarray  # (n,)   bool  NEB(X) = ∩_j NIB_{nr_{c_j}}(X) Def. 3
    npb: np.ndarray  # (m, n) bool  NPB_{nr_{c_j}}(X) = NIB - NEB  Def. 4
    inter: np.ndarray  # (m, n_X) int  |n_{c_j}(x) ∩ X|            Eq. 10
    nom: np.ndarray  # (n_X, n_X) int  NOM restricted to X x X     Eq. 9
    neb_cnt: np.ndarray  # (n_X,)   |NEB^X(x)|                     Eq. 3
    npb_cnt: np.ndarray  # (n_X, m) |NPB^X_{nr_{c_j}}(x)|          Eq. 3
    pos_cnt: np.ndarray  # (n_X, m) |POS^X_{nr_{c_j}}(x)|          Eq. 3
    neb_size: float  # |NEB(X)|
    npb_size: np.ndarray  # (m,) |NPB_{nr_{c_j}}(X)|
    pos_size: np.ndarray  # (m,) |POS_{nr_{c_j}}(X)|
    neg_size: np.ndarray  # (m,) |NEG_{nr_{c_j}}(X)|
    d1: float
    d2: float
    d3: float

def granulate(data: np.ndarray, numeric_mask: np.ndarray, radii: np.ndarray,
              x_index: np.ndarray) -> Granulation:
    """
    Algorithms 1 and 1* fused into a single pass over the attributes.

    Per attribute one boolean relation ``nr_{c_j}`` is built and immediately
    reduced, so nothing of size ``O(m n^2)`` is ever materialised:

      POS_{nr_{c_j}}(X) = {x | n_{c_j}(x) ⊆ X}                 (Eq. 1)
      NEG_{nr_{c_j}}(X) = {x | n_{c_j}(x) ∩ X = ∅}             (Eq. 1, Alg. 1* Step 4)
      |n_{c_j}(x) ∩ X|                                         (Eq. 10)
      NOM(x, y) = |{c | (x, y) ∉ nr_{c}}|                      (Eq. 9)

    NIB / NEB / NPB then follow from Eq. 2, Def. 3 and Def. 4, and the
    characteristic cardinalities of Eq. 3 are counted against the Eq. 11
    thresholds ``d1 = m*/3 < d2 = m*/2 < d3 = 0.9 m*``.  ``dist_C`` is NOM
    (Def. 9), restricted to ``X x X`` in caller order.
    """
    n, m = data.shape
    mask = np.asarray(numeric_mask, dtype=bool)
    x_index = np.asarray(x_index, dtype=int)
    n_X = int(x_index.size)
    in_x = np.zeros(n, dtype=bool)
    in_x[x_index] = True
    out_x = ~in_x

    pos = np.zeros((m, n), dtype=bool)
    neg = np.zeros((m, n), dtype=bool)
    inter = np.zeros((m, n_X), dtype=np.int64)
    overlap = np.zeros((n, n), dtype=np.int32)  # #attributes with (x, y) in nr_{c}

    for j in range(m):
        rel = neighborhood_relation(data[:, j], float(radii[j]), bool(mask[j]))
        rel_x = rel[:, x_index]
        pos[j] = ~rel[:, out_x].any(axis=1)  # empty slice -> POS = U when X = U
        neg[j] = ~rel_x.any(axis=1)
        inter[j] = rel_x[x_index].sum(axis=1)
        overlap += rel

    nib = in_x[None, :] & ~pos  # Eq. 2 (POS ⊆ X by reflexivity)
    neb = nib.all(axis=0) if m > 0 else np.zeros(n, dtype=bool)  # Def. 3
    npb = nib & ~neb[None, :]  # Def. 4
    nom = (m - overlap)[np.ix_(x_index, x_index)]  # Eq. 9

    d1, d2, d3 = m / 3.0, m / 2.0, 0.9 * m  # Eq. 11

    # Eq. 3: characteristic regions of x, counted over y in the relevant region.
    neb_cols = np.flatnonzero(neb[x_index])
    neb_cnt = (nom[:, neb_cols] <= d1).sum(axis=1).astype(np.float64)
    npb_cnt = np.zeros((n_X, m), dtype=np.float64)
    pos_cnt = np.zeros((n_X, m), dtype=np.float64)
    for j in range(m):
        sel = np.flatnonzero(npb[j][x_index])
        npb_cnt[:, j] = (nom[:, sel] >= d2).sum(axis=1)
        sel = np.flatnonzero(pos[j][x_index])
        pos_cnt[:, j] = (nom[:, sel] >= d3).sum(axis=1)

    return Granulation(
        n=n, m=m, n_X=n_X, radii=np.asarray(radii, dtype=np.float64), x_index=x_index,
        pos=pos, neg=neg, nib=nib, neb=neb, npb=npb, inter=inter, nom=nom,
        neb_cnt=neb_cnt, npb_cnt=npb_cnt, pos_cnt=pos_cnt,
        neb_size=float(neb.sum()),
        npb_size=npb.sum(axis=1).astype(np.float64),
        pos_size=pos.sum(axis=1).astype(np.float64),
        neg_size=neg.sum(axis=1).astype(np.float64),
        d1=d1, d2=d2, d3=d3,
    )


# =============================================================================
# 5. MEASUREMENT LAYER -- Eq. 4 / 4*, Eq. 10 / 10*
# =============================================================================

def ndf_baseline(g: Granulation) -> np.ndarray:
    """Eq. 4 -- neighborhood deviation factor, ``(n_X, m)``, values in [0, 1]."""
    return (g.neb_cnt[:, None] + g.npb_cnt + g.pos_cnt) / float(g.n_X)


def ndf_tiered(g: Granulation) -> np.ndarray:
    """
    Eq. 4* -- tiered evidentiary weighting with the self-normalising rarity
    weight ``w_R = 1 - |R| / |X|``.  The three regions partition X
    (``|POS_j| + |NEB| + |NPB_j| = |X|``), hence with ``w == 1`` the denominator
    collapses to ``|X|`` and Eq. 4* reduces to Eq. 4 exactly.  A vanishing
    denominator (e.g. ``POS_j = X``, so the only non-empty region has zero
    rarity weight) yields ``NDF* = 0``.
    """
    n_x = float(g.n_X)
    w_neb = 1.0 - g.neb_size / n_x
    w_npb = 1.0 - g.npb_size / n_x
    w_pos = 1.0 - g.pos_size / n_x
    numer = (w_neb * g.neb_cnt[:, None]
             + w_npb[None, :] * g.npb_cnt
             + w_pos[None, :] * g.pos_cnt)
    denom = w_neb * g.neb_size + w_npb * g.npb_size + w_pos * g.pos_size  # (m,)
    out = np.zeros_like(numer)
    ok = denom > _EPS
    if ok.any():
        out[:, ok] = numer[:, ok] / denom[ok][None, :]
    return out

def omega_baseline(g: Granulation, m_star: int) -> np.ndarray:
    """Eq. 10 -- ``(1 - sqrt(|n_{c_j}(x) ∩ X| / |X|)) / |C*|``, shape ``(n_X, m)``."""
    prevalence = g.inter.T.astype(np.float64) / float(g.n_X)
    return (1.0 - np.sqrt(np.clip(prevalence, 0.0, None))) / float(m_star)


def discriminability(g: Granulation) -> tuple:
    """
    Single-attribute approximation quality (rough-set dependency degree)
    ``gamma_{c_j}(X) = (|POS_{nr_{c_j}}(X)| + |NEG_{nr_{c_j}}(X)|) / n`` and its
    self-normalisation ``gamma_hat = gamma / max_k gamma_k``.  If every attribute
    is fully ambiguous (``max_k gamma_k = 0``) then ``gamma_hat := 1``, which
    recovers Eq. 10 exactly.
    """
    gamma = (g.pos_size + g.neg_size) / float(g.n)
    top = float(gamma.max()) if gamma.size else 0.0
    gamma_hat = np.ones_like(gamma) if top <= _EPS else gamma / top
    return gamma, gamma_hat


def omega_discriminative(omega: np.ndarray, gamma_hat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Eq. 10* -- ``omega* = omega^alpha * gamma_hat^(1 - alpha)``, the Jeffreys-symmetric
    geometric blend at the fixed ``alpha = 1/2``.  Boundedness in [0, 1] holds for
    any ``alpha in [0, 1]`` (Prop. 2); ``alpha = 1`` returns Eq. 10 unchanged.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    if alpha >= 1.0 - _EPS:
        return np.array(omega, dtype=np.float64, copy=True)
    base = np.clip(omega, 0.0, None)
    factor = np.clip(np.asarray(gamma_hat, dtype=np.float64), 0.0, None)[None, :]
    if alpha <= _EPS:
        return np.repeat(factor, base.shape[0], axis=0)
    return base ** alpha * factor ** (1.0 - alpha)


# =============================================================================
# 6. ALGORITHM 2 / 2* -- the configurable engine (all variants share it)
# =============================================================================

class ThreeWNCROD:
    """
    Configurable 3WNCROD engine.  ``MNOF^X(x) = sum_j NDF_j(x) * omega_j(x)``
    (Eq. 5 / 5*); each switch selects one stage's formula, and the default
    configuration is the untouched baseline of Algorithms 1-2.

    Parameters
    ----------
    lammda : float
        Eq. 8 radius adjustment.  Ignored by the lambda-free radius rules
        (``amise_std``, ``silverman_amise``).

    radius_rule : str
        One of :data:`RADIUS_RULES`.
    weight_rule : {'baseline', 'discriminative'}
        Eq. 10 or Eq. 10*.
    ndf_rule : {'baseline', 'tiered'}
        Eq. 4 or Eq. 4*.
    augment_subspace : bool
        Run Algorithm 0* and, if it is non-degenerate, granulate over
        ``C* = C ∪ {c_{m+1}}`` (Eq. 9*, Eq. 5*).
    alpha : float
        Blend exponent of Eq. 10* (1/2 as specified; 1.0 reproduces Eq. 10).
    n_components, power_iters, random_state
        Algorithm 0* controls.  ``n_components=None`` uses
        ``r = ceil(log2(m_num + 1))`` clamped to ``m_num - 1``.

    Notes
    -----
    ``fit_predict`` returns MNOF for the objects of ``X`` only, aligned with the
    given ``X_index`` order -- exactly the contract of the reference
    implementation.  Diagnostics land in ``self.details_``.
    """

    def __init__(self, lammda: float = 1.0, radius_rule: str = "baseline",
                 weight_rule: str = "baseline", ndf_rule: str = "baseline",
                 augment_subspace: bool = False, alpha: float = 0.5,
                 n_components: Optional[int] = None, power_iters: int = 10,
                 random_state: int = 0, name: Optional[str] = None):
        if radius_rule not in RADIUS_RULES:
            raise ValueError(f"radius_rule must be one of {RADIUS_RULES}.")
        if weight_rule not in ("baseline", "discriminative"):
            raise ValueError("weight_rule must be 'baseline' or 'discriminative'.")
        if ndf_rule not in ("baseline", "tiered"):
            raise ValueError("ndf_rule must be 'baseline' or 'tiered'.")
        self.lammda = float(lammda)
        self.radius_rule = radius_rule
        self.weight_rule = weight_rule
        self.ndf_rule = ndf_rule
        self.augment_subspace = bool(augment_subspace)
        self.alpha = float(alpha)
        self.n_components = n_components
        self.power_iters = int(power_iters)
        self.random_state = int(random_state)
        self.name = name or type(self).__name__
        self.details_: Dict = {}

    # -- helpers -------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{type(self).__name__}(name={self.name!r}, lammda={self.lammda}, "
                f"radius_rule={self.radius_rule!r}, weight_rule={self.weight_rule!r}, "
                f"ndf_rule={self.ndf_rule!r}, augment_subspace={self.augment_subspace})")

    @staticmethod
    def _resolve_x_index(x_index, n: int) -> np.ndarray:
        """Accept ``None`` (X = U), a boolean mask, or an integer index array."""
        if x_index is None:
            return np.arange(n, dtype=int)
        arr = np.asarray(x_index)
        if arr.dtype == bool:
            if arr.shape != (n,):
                raise ValueError("boolean X_index must have one entry per object.")
            return np.flatnonzero(arr).astype(int)
        arr = arr.astype(int, copy=False).ravel()
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            raise ValueError("X_index contains out-of-range object ids.")
        if np.unique(arr).size != arr.size:
            raise ValueError("X_index contains duplicated object ids.")
        return arr

    # -- main entry point ----------------------------------------------------
    def fit_predict(self, data: ArrayLike, X_index=None,
                    numeric_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute ``MNOF^X(x)`` for every ``x in X``.

        Parameters
        ----------
        data : (n, m) array-like
            Condition attributes only.  Numeric attributes must already be
            min-max normalised into [0, 1] (Eq. 6, see :func:`minmax_normalize`);
            nominal attributes must be integer codes.  NaN = unknown value.
        X_index : None | bool mask | int array
            The concept ``X`` (the selected condition subdata).  ``None`` means
            ``X = U``, which degenerates the boundary structure and is only
            meaningful as a sanity check.
        numeric_mask : (m,) bool array, optional
            Explicit attribute typing; defaults to :func:`infer_numeric_mask`.

        Returns
        -------
        (|X|,) float array
            MNOF scores aligned with ``X_index``; larger = more outlying.
        """
        arr = as_matrix(data)
        n, m = arr.shape
        mask = infer_numeric_mask(arr) if numeric_mask is None else np.asarray(numeric_mask, dtype=bool).copy()
        if mask.shape != (m,):
            raise ValueError("numeric_mask must have one entry per attribute.")
        x_index = self._resolve_x_index(X_index, n)

        subspace = SubspaceInfo(False, int(mask.sum()), 0, "subspace augmentation disabled")
        if x_index.size == 0:
            self.details_ = {"variant": self.name, "n": n, "m": m, "m_star": m, "n_X": 0,
                             "radii": np.zeros(m), "subspace": subspace,
                             "warning": "X is empty: no object to score"}
            return np.zeros(0, dtype=np.float64)

        # ---- Algorithm 0*: optional attribute-space augmentation ------------
        if self.augment_subspace:
            subspace = synthesize_subspace_attribute(
                arr, mask, self.n_components, self.power_iters, self.random_state)
            if subspace.used:
                arr = np.column_stack([arr, subspace.attribute])
                mask = np.append(mask, True)

        m_star = arr.shape[1]

        # ---- metric layer (Eq. 8 / 8*) --------------------------------------
        radii = compute_radii(arr, mask, self.radius_rule, self.lammda)

        # ---- region layer (Algorithms 1 / 1*, unchanged in form) ------------
        g = granulate(arr, mask, radii, x_index)

        # ---- measurement layer (Eq. 4 / 4*, Eq. 10 / 10*) -------------------
        ndf = ndf_tiered(g) if self.ndf_rule == "tiered" else ndf_baseline(g)
        omega = omega_baseline(g, m_star)
        gamma, gamma_hat = discriminability(g)
        if self.weight_rule == "discriminative":
            omega = omega_discriminative(omega, gamma_hat, self.alpha)

        # ---- fusion (Eq. 5 / 5*) and decision input (Def. 8) ----------------
        mnof = (ndf * omega).sum(axis=1)

        self.details_ = {
            "variant": self.name, "n": n, "m": m, "m_star": m_star, "n_X": g.n_X,
            "radius_rule": self.radius_rule, "radii": g.radii,
            "weight_rule": self.weight_rule, "ndf_rule": self.ndf_rule,
            "thresholds": (g.d1, g.d2, g.d3), "gamma": gamma, "gamma_hat": gamma_hat,
            "region_sizes": {"NEB": g.neb_size, "NPB": g.npb_size, "POS": g.pos_size,
                             "NEG": g.neg_size},
            "subspace": subspace, "mnof_range": (float(mnof.min()), float(mnof.max())),
        }
        return mnof

    # Identical calling interface under either name.
    def compute_mnof(self, data: ArrayLike, X_index=None,
                     numeric_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Alias of :meth:`fit_predict`."""
        return self.fit_predict(data, X_index=X_index, numeric_mask=numeric_mask)

    def __call__(self, data: ArrayLike, X_index=None,
                 numeric_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit_predict(data, X_index=X_index, numeric_mask=numeric_mask)


# =============================================================================
# 7. ISOLATED VARIANTS -- one mechanism each, identical constructor and API
# =============================================================================

class _FixedVariant(ThreeWNCROD):
    """Base of the isolated variants: freezes the stage switches, keeps the API."""

    _NAME = "variant"
    _CONFIG: Dict = {}

    def __init__(self, lammda: float = 1.0, alpha: float = 0.5,
                 n_components: Optional[int] = None, power_iters: int = 10,
                 random_state: int = 0):
        super().__init__(lammda=lammda, alpha=alpha, n_components=n_components,
                         power_iters=power_iters, random_state=random_state,
                         name=self._NAME, **self._CONFIG)


class Variant0Baseline(_FixedVariant):
    """
    V0 -- untouched 3WNCROD (Algorithms 1-2).

    Eq. 8 radius ``std(c_j)/lambda``, Eq. 4 NDF, Eq. 10 weights, attribute set
    ``C``.  Regression-tested against :func:`WNCROD_reference` in ``__main__``.
    """
    _NAME = "V0 baseline"
    _CONFIG = dict(radius_rule="baseline", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=False)


class Variant1AdaptiveRadius(_FixedVariant):
    """
    V1 -- Enhancement 2.1 only: Eq. 8* parameter-free adaptive radius

        eps* = max(0.9 * min+(MADN, IQR/1.34) * n^(-1/5), 1/n)

    per numeric attribute (0 for categorical, and 0 when MADN = IQR = 0).
    Resolves Weaknesses 1, 4 and 5: radii decouple across attributes, the
    20-step lambda grid disappears, and there is no search domain to truncate.
    ``lammda`` is accepted for interface uniformity but unused.
    NDF, weights and the attribute set stay baseline.
    """
    _NAME = "V1 adaptive radius (Eq. 8*)"
    _CONFIG = dict(radius_rule="silverman_amise", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=False)


class Variant2DiscriminativeWeights(_FixedVariant):
    """
    V2 -- Enhancement 2.3 only: Eq. 10* discriminability-augmented weight

        omega* = sqrt(omega) * sqrt(gamma_hat),
        gamma_{c_j}(X) = (|POS_{nr_{c_j}}(X)| + |NEG_{nr_{c_j}}(X)|) / n

    Resolves Weakness 2: the per-object prevalence term of Eq. 10 is multiplied
    by a feature-level informativeness term, reusing Algorithm 1*'s NEG output
    at no extra asymptotic cost.  Radius, NDF and attribute set stay baseline.
    ``alpha=1.0`` collapses this variant onto V0 (identity check in ``__main__``).
    """
    _NAME = "V2 discriminative weights (Eq. 10*)"
    _CONFIG = dict(radius_rule="baseline", weight_rule="discriminative",
                   ndf_rule="baseline", augment_subspace=False)


class Variant3SubspaceAugmented(_FixedVariant):
    """
    V3 -- Enhancement 2.2 only: Algorithm 0* + Eq. 9*/5*.

    One synthetic numeric attribute ``c_{m+1} = minmax(SPE)`` is appended, where
    SPE is the residual energy orthogonal to the dominant rank-r correlation
    subspace found by deflationary power iteration.  It then flows through the
    unchanged pipeline as an ordinary numeric attribute, so NOM (Eq. 9), the
    Eq. 11 thresholds and the Eq. 10 normaliser re-index over ``C*``.
    Resolves Weakness 3 (linear correlation-breaking / off-diagonal outliers)
    without any ``2^m`` subspace search.  Radius rule, NDF and weights stay
    baseline; ``m_num < 2`` degenerates losslessly to V0.
    """
    _NAME = "V3 subspace attribute (Alg. 0*, Eq. 9*)"
    _CONFIG = dict(radius_rule="baseline", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=True)


class Variant4TieredEvidence(_FixedVariant):
    """
    V4 -- Enhancement 2.4 only: Eq. 4* tiered evidentiary weighting

        NDF* = (w_NEB |NEB^X| + w_NPB,j |NPB^X_j| + w_POS,j |POS^X_j|)
               / (w_NEB |NEB|  + w_NPB,j |NPB_j|  + w_POS,j |POS_j|),
        w_R = 1 - |R| / |X|

    Operationalises the paper's own (previously unweighted) claim that NEB
    evidence dominates POS evidence, via IDF-style region rarity.  Strictly
    generalises Eq. 4 (``w == 1`` reproduces it, since the three regions
    partition ``X``).  Radius, weights and attribute set stay baseline.
    """
    _NAME = "V4 tiered evidence NDF (Eq. 4*)"
    _CONFIG = dict(radius_rule="baseline", weight_rule="baseline",
                   ndf_rule="tiered", augment_subspace=False)


class Variant5RobustScale(_FixedVariant):
    """
    V5 -- Eq. 8* factor 1 in isolation: robust scale, ``eps = MADN(c_j)/lambda``.

    Only the breakdown point changes (0 -> 0.5, Sec. 4.2): lambda, the grid and
    the radius scaling law are all kept, so this measures how much of Eq. 8*'s
    effect comes purely from de-masking the scale estimator.
    """
    _NAME = "V5 robust scale only (MADN/lambda)"
    _CONFIG = dict(radius_rule="madn", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=False)


class Variant6SilvermanSelector(_FixedVariant):
    """
    V6 -- Eq. 8* factors 1+2: ``eps = min+(MADN, IQR/1.34) / lambda``.

    Adds Silverman's IQR co-selector on top of V5 while still keeping lambda,
    isolating the effect of the min-selector (heavy-tail / multimodality guard)
    from the AMISE decay rate.
    """
    _NAME = "V6 Silverman selector only"
    _CONFIG = dict(radius_rule="silverman_lambda", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=False)


class Variant7AmiseRate(_FixedVariant):
    """
    V7 -- Eq. 8* factor 3 in isolation: ``eps = 0.9 * std(c_j) * n^(-1/5)``.

    Keeps the non-robust std of Eq. 8 but replaces ``1/lambda`` by the
    AMISE-optimal decay rate, so the grid search vanishes (Weaknesses 4-5) while
    the breakdown point stays 0.  Measures the value of parameter elimination
    alone, without robustness and without the 1/n floor.
    """
    _NAME = "V7 AMISE rate only (lambda-free)"
    _CONFIG = dict(radius_rule="amise_std", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=False)


class Variant8ResolutionFloor(_FixedVariant):
    """
    V8 -- Eq. 8* factor 4 in isolation: ``eps = max(std(c_j)/lambda, 1/n)``.

    Adds only the minimum-resolvable-gap floor of Sec. 4.1 to Eq. 8, which
    prevents neighborhoods from fragmenting into singletons on near-degenerate
    attributes.  Everything else is baseline.
    """
    _NAME = "V8 resolution floor only (1/n)"
    _CONFIG = dict(radius_rule="baseline_floor", weight_rule="baseline",
                   ndf_rule="baseline", augment_subspace=False)


class VariantFull(_FixedVariant):
    """
    VF -- all four enhancements at once: Eq. 8* radius, Algorithm 0*/Eq. 9*
    attribute augmentation, Eq. 10* weights and Eq. 4* tiered NDF, fused by
    Eq. 5* over ``C*``.  Reference point for the integrated proposal.
    """
    _NAME = "VF full (8* + 9* + 10* + 4*)"
    _CONFIG = dict(radius_rule="silverman_amise", weight_rule="discriminative",
                   ndf_rule="tiered", augment_subspace=True)


VARIANTS: Dict[str, type] = {
    "V0": Variant0Baseline,
    "V1": Variant1AdaptiveRadius,
    "V2": Variant2DiscriminativeWeights,
    "V3": Variant3SubspaceAugmented,
    "V4": Variant4TieredEvidence,
    "V5": Variant5RobustScale,
    "V6": Variant6SilvermanSelector,
    "V7": Variant7AmiseRate,
    "V8": Variant8ResolutionFloor,
    "VF": VariantFull,
}


def build_all_variants(**kwargs) -> Dict[str, ThreeWNCROD]:
    """Instantiate every variant with a shared configuration (e.g. ``lammda=1.0``)."""
    return {key: cls(**kwargs) for key, cls in VARIANTS.items()}


# =============================================================================
# 8. BENCHMARKING UTILITIES
# =============================================================================

def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC_AUC via the Mann-Whitney statistic with mid-ranks for ties."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """AP = sum_k (R_k - R_{k-1}) * P_k, ties broken by descending score order."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    hits = y[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / hits.sum())



# =============================================================================
# 9. REFERENCE TRANSCRIPTION -- the published implementation, kept verbatim
#    (regression oracle for V0; not used by any variant)
# =============================================================================

def WNCROD_reference(data, X_tem, lammda):
    """
    Original 3WNCROD implementation by Z. Yuan (2023-08-29), transcribed
    unchanged from ``main_code.py``.  Used only to prove that V0 reproduces the
    published algorithm bit-for-bit.
    """
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1 = m / 3
    D2 = m / 2
    D3 = 0.9 * m

    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / lammda

    Lower = np.zeros((m, n))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        Lower_temp = np.min(np.maximum(1 - RM_tem, np.tile(X, (n, 1))), axis=1)
        Lower[col, :Lower_temp.shape[0]] = Lower_temp

    IB = np.tile(X, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    n_X = int(sum(X))

    weight = np.zeros((n_X, m))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        weight_x = []

        for i in range(n_X):
            temp1 = RM_tem[X_tem[i], :]
            weight_temp = 1 - (np.sqrt((np.sum(np.minimum(temp1, X))) / n_X))
            weight_x.append(weight_temp)

        weight[:len(weight_x), col] = weight_x

    D_tem = np.zeros((n, n))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        D_tem += RM_tem

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
# 10. VERIFICATION / DEMONSTRATION
# =============================================================================

def _synthetic_dataset(n_inlier: int = 180, seed: int = 7):
    """
    Hybrid universe with three planted outlier mechanisms, so that the isolated
    variants can be told apart:

      c1, c2, c3  numeric, driven by one latent factor (linear correlation)
      c4          numeric context attribute used to carve the concept X
      c5          numeric, independent
      c6          nominal in {1, 2} for inliers

      'marginal'  extreme on c5                       -> visible to any 1-D factor
      'subspace'  c2, c3 sign-flipped w.r.t. c1       -> marginally ordinary,
                                                         jointly impossible
      'rare_cat'  c6 = 3, numerics on-manifold        -> nominal-only anomaly
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n_inlier)
    inliers = np.column_stack([
        latent + 0.20 * rng.normal(size=n_inlier),
        0.85 * latent + 0.20 * rng.normal(size=n_inlier),
        -0.70 * latent + 0.25 * rng.normal(size=n_inlier),
        rng.uniform(0.0, 1.0, size=n_inlier),
        rng.normal(size=n_inlier),
        rng.choice([1.0, 2.0], size=n_inlier, p=[0.62, 0.38]),
    ])
    kinds = ["inlier"] * n_inlier
    blocks = [inliers]

    # (a) marginal outliers: extreme on c5, on-manifold elsewhere
    k = 4
    t = rng.normal(size=k)
    blocks.append(np.column_stack([
        t + 0.2 * rng.normal(size=k), 0.85 * t, -0.70 * t,
        rng.uniform(0.0, 0.5, size=k),
        rng.choice([-4.5, 4.5], size=k) + 0.2 * rng.normal(size=k),
        rng.choice([1.0, 2.0], size=k),
    ]))
    kinds += ["marginal"] * k

    # (b) correlation-breaking outliers: every margin plausible, joint pattern wrong
    k = 5
    t = rng.uniform(1.0, 1.6, size=k) * rng.choice([-1.0, 1.0], size=k)
    blocks.append(np.column_stack([
        t, -0.85 * t + 0.05 * rng.normal(size=k), 0.70 * t + 0.05 * rng.normal(size=k),
        rng.uniform(0.0, 0.5, size=k),
        rng.normal(size=k),
        rng.choice([1.0, 2.0], size=k),
    ]))
    kinds += ["subspace"] * k

    # (c) rare-category outliers: nominal value unseen among inliers
    k = 3
    t = rng.normal(size=k)
    blocks.append(np.column_stack([
        t + 0.2 * rng.normal(size=k), 0.85 * t, -0.70 * t,
        rng.uniform(0.0, 0.5, size=k),
        rng.normal(size=k),
        np.full(k, 3.0),
    ]))
    kinds += ["rare_cat"] * k

    raw = np.vstack(blocks)
    numeric_mask = np.array([True, True, True, True, True, False])
    data = minmax_normalize(raw, numeric_mask)
    kinds = np.array(kinds)
    labels = (kinds != "inlier").astype(int)
    # Concept X (a condition subdata, as in Table 5): X = {x | f(x, c4) <= 0.6}
    x_index = np.flatnonzero(data[:, 3] <= 0.6)
    return data, numeric_mask, labels, kinds, x_index


def _fmt(values, width=8, digits=5):
    return " ".join(f"{v:{width}.{digits}f}" for v in values)

def _demo() -> None:
    from time import perf_counter

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    data, numeric_mask, labels, kinds, x_index = _synthetic_dataset()
    n, m = data.shape
    y_x, kinds_x = labels[x_index], kinds[x_index]
    lammda = 1.0

    print("=" * 100)
    print("3WNCROD -- isolated enhancement variants, synthetic hybrid benchmark")
    print("=" * 100)
    print(f"universe n = {n}, attributes m = {m} (numeric mask {numeric_mask.astype(int)})")
    print(f"concept X = {{x | f(x, c4) <= 0.6}} -> |X| = {x_index.size}, "
          f"outliers in X = {int(y_x.sum())}")
    for kind in ("marginal", "subspace", "rare_cat"):
        print(f"    planted '{kind}' inside X: {int((kinds_x == kind).sum())}")
    assert np.array_equal(infer_numeric_mask(data), numeric_mask), \
        "inferred attribute typing disagrees with the declared one"

    # ---- 1. V0 must reproduce the published implementation ------------------
    reference = WNCROD_reference(data.copy(), x_index.copy(), lammda)
    v0_scores = Variant0Baseline(lammda=lammda).fit_predict(data, x_index, numeric_mask)
    np.testing.assert_allclose(v0_scores, reference, rtol=1e-12, atol=1e-12)
    print(f"\n[check] V0 == WNCROD_reference (published code): max|diff| = "
          f"{np.max(np.abs(v0_scores - reference)):.3e}  OK")

    # ---- 2. run every variant ----------------------------------------------
    rows, scores, details = [], {}, {}
    n_out = int(y_x.sum())
    for key, cls in VARIANTS.items():
        estimator = cls(lammda=lammda)
        tic = perf_counter()
        s = estimator.fit_predict(data, x_index, numeric_mask)
        elapsed = perf_counter() - tic
        scores[key], details[key] = s, estimator.details_
        top = np.argsort(-s, kind="mergesort")[:n_out]
        rows.append({
            "key": key, "variant": estimator.name, "m*": estimator.details_["m_star"],
            "ROC_AUC": roc_auc(y_x, s), "AP": average_precision(y_x, s),
            f"hits@{n_out}": int(y_x[top].sum()),
            "marg": int((kinds_x[top] == "marginal").sum()),
            "subsp": int((kinds_x[top] == "subspace").sum()),
            "rare": int((kinds_x[top] == "rare_cat").sum()),
            "MNOF_min": s.min(), "MNOF_max": s.max(),
            "d_vs_V0": float(np.max(np.abs(s - scores["V0"]))),
            "sec": elapsed,
        })
    print("\n---- 2. standalone effect of each isolated enhancement " + "-" * 44)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 3. raw MNOF outputs ----------------------------------------------
    show = list(range(6)) + list(np.flatnonzero(y_x)[:4])
    print("\n---- 3. MNOF^X(x) for the first 6 objects of X and 4 planted outliers " + "-" * 26)
    print(f"{'variant':38s} " + " ".join(f"x[{i}]{'*' if y_x[i] else ' '}".rjust(9) for i in show))
    for key in VARIANTS:
        print(f"{key + ' ' + details[key]['variant']:38.38s} " + _fmt(scores[key][show], 9, 5))
    print("  (* = planted outlier;  full score vectors are the fit_predict return values)")

    # ---- 4. per-variant diagnostics ---------------------------------------
    print("\n---- 4. metric layer: neighborhood radii eps_{c_j} " + "-" * 48)
    for key in ("V0", "V1", "V5", "V6", "V7", "V8", "VF"):
        rule = details[key]["radius_rule"]
        print(f"{key:3s} {rule:17s} eps = [{_fmt(details[key]['radii'], 7, 4)}]")
    print("\n---- 5. Eq. 10* discriminability and Alg. 0* subspace " + "-" * 45)
    print(f"gamma_hat(c_j) [V2] = [{_fmt(details['V2']['gamma_hat'], 7, 4)}]")
    print(f"gamma(c_j)     [V2] = [{_fmt(details['V2']['gamma'], 7, 4)}]")
    for key in ("V3", "VF"):
        info = details[key]["subspace"]
        print(f"{key:3s} subspace: used={info.used}, m_num={info.m_num}, r={info.r}, "
              f"m* = {details[key]['m_star']} ({info.reason})")
        if info.attribute is not None:
            print(f"    c_{{m+1}} on X: min={info.attribute[x_index].min():.4f}, "
                  f"max={info.attribute[x_index].max():.4f}, "
                  f"mean over planted 'subspace' outliers = "
                  f"{info.attribute[x_index][kinds_x == 'subspace'].mean():.4f} "
                  f"vs inliers {info.attribute[x_index][kinds_x == 'inlier'].mean():.4f}")

    # ---- 6. structural / formal checks ------------------------------------
    print("\n---- 6. structural checks " + "-" * 73)
    sizes = details["V0"]["region_sizes"]
    partition = sizes["NEB"] + sizes["NPB"] + sizes["POS"]
    assert np.allclose(partition, details["V0"]["n_X"]), "POS_j + NEB + NPB_j must partition X"
    print(f"[check] |POS_j| + |NEB| + |NPB_j| == |X| = {details['V0']['n_X']} for every j  OK"
          "  (this is why Eq. 4* with w == 1 is exactly Eq. 4)")

    identity = ThreeWNCROD(lammda=lammda, weight_rule="discriminative", alpha=1.0,
                           name="alpha=1 probe").fit_predict(data, x_index, numeric_mask)
    np.testing.assert_allclose(identity, scores["V0"], rtol=1e-12, atol=1e-12)
    print("[check] Eq. 10* with alpha = 1 reproduces Eq. 10 (V2 -> V0)  OK")

    # Theorem 1: augmentation can only sharpen the exceptional boundary.
    radii_c = compute_radii(data, numeric_mask, "baseline", lammda)
    g_c = granulate(data, numeric_mask, radii_c, x_index)
    sub = details["V3"]["subspace"]
    if sub.used:
        aug = np.column_stack([data, sub.attribute])
        mask_aug = np.append(numeric_mask, True)
        g_star = granulate(aug, mask_aug, compute_radii(aug, mask_aug, "baseline", lammda), x_index)
        assert np.all(g_star.neb <= g_c.neb), "Theorem 1 violated: NEB* must be a subset of NEB"
        print(f"[check] Theorem 1: |NEB(X)| = {int(g_c.neb.sum())} -> "
              f"|NEB*(X)| = {int(g_star.neb.sum())}, NEB* subset of NEB  OK")

    # Sec. 2.2 / 4.3: purely nominal data degenerates V1 and V3 onto V0.
    rng = np.random.default_rng(3)
    cat_data = rng.integers(1, 4, size=(40, 4)).astype(float)
    cat_mask = np.zeros(4, dtype=bool)
    cat_x = np.arange(0, 26)
    cat_v0 = Variant0Baseline(lammda=lammda).fit_predict(cat_data, cat_x, cat_mask)
    cat_v1 = Variant1AdaptiveRadius().fit_predict(cat_data, cat_x, cat_mask)
    cat_v3 = Variant3SubspaceAugmented(lammda=lammda)
    cat_v3_scores = cat_v3.fit_predict(cat_data, cat_x, cat_mask)
    np.testing.assert_allclose(cat_v1, cat_v0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cat_v3_scores, cat_v0, rtol=1e-12, atol=1e-12)
    print("[check] all-nominal data: Eq. 8* == Eq. 8 (eps = 0) and Alg. 0* returns C* = C, "
          "so V1 == V3 == V0  OK")
    print(f"        ({cat_v3.details_['subspace'].reason})")

    # Codomains: Eq. 4/4* keep MNOF in [0, 1]; Eq. 10* rescales to [0, sqrt(m*)].
    for key in ("V0", "V1", "V3", "V4", "V5", "V6", "V7", "V8"):
        s = scores[key]
        assert s.min() >= -_EPS and s.max() <= 1.0 + 1e-9, f"{key} left [0, 1]"
    for key in ("V2", "VF"):
        cap = np.sqrt(details[key]["m_star"])
        assert scores[key].min() >= -_EPS and scores[key].max() <= cap + 1e-9, f"{key} left [0, sqrt(m*)]"
    print("[check] MNOF in [0, 1] for the Eq. 10 weight family; "
          "MNOF in [0, sqrt(m*)] under Eq. 10* (see module docstring, note 3)  OK")

    print("\nAll variants share the interface  variant.fit_predict(data, X_index, numeric_mask)"
          "  ->  (|X|,) MNOF array")
    print("Registry:  VARIANTS  /  build_all_variants(lammda=1.0)")
    print("=" * 100)


if __name__ == "__main__":
    _demo()
