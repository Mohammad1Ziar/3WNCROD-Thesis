"""
3WNCROD — Baseline and Isolated Closed-Form Enhancement Variants
===================================================================

Source materials
-----------------
1. Baseline algorithm: Zhang X., Yuan Z., Miao D. "Outlier Detection Using
   Three-Way Neighborhood Characteristic Regions and Corresponding Fusion
   Measurement," IEEE TKDE, 2023 (reference `main_code.py`, function
   `WNCROD`, kept verbatim below as Variant 0).
2. Enhancement document: "Closed-Form Enhancements to 3WNCROD: Attribute-
   Space Augmentation and Robust Nonparametric Granulation for Three-Way
   Neighborhood Outlier Detection" (`3WNCROD_closed_form_enhancements.md`).

Distinct mathematical ideas identified in the enhancement document
--------------------------------------------------------------------
A careful audit of the document (Sections 1-6) surfaces exactly FOUR
independently testable mathematical mechanisms. Section 2.5 (fusion, Eq. 5*)
is purely a re-indexing of the existing sum over an enlarged attribute set
and is not a separable idea; the Section 1 lemmas/theorems and Section 3-6
material are proofs/analysis, not new computational mechanisms. Hence:

    Idea 1 -> Eq. 8*  : parameter-free adaptive radius (Silverman/MAD/IQR,
                         1/n floor)                          -> Variant 1
    Idea 2 -> Eq. 10* : discriminability-augmented weight via rough
                         approximation quality gamma_{cj}(X)  -> Variant 2
    Idea 3 -> Alg. 0*, Eq. 9*: rank-r subspace/SPE synthetic attribute
                         (deflationary power iteration)        -> Variant 3
    Idea 4 -> Eq. 4*  : tiered evidentiary NDF weighting (region rarity
                         weights w_R)                           -> Variant 4

No further separable mechanisms exist in the document, so no Variant 5/6
are fabricated. Variant "Full" integrates all four.

All variants expose the identical calling interface:

    detector.fit_predict(data, X_tem, lammda=..., numeric_mask=None) -> np.ndarray

`data`   : (n, m) float ndarray. Numeric columns pre-normalized to [0,1];
           nominal columns integer-coded (matches the baseline convention).
`X_tem`  : 1-D array of row indices defining the target concept X subset.
`lammda` : radius-adjustment parameter for the *baseline* radius (Eq. 8).
           Ignored (accepted for interface uniformity only) by variants
           whose radius is parameter-free (Variant 1, Variant Full).
`numeric_mask` : optional bool mask overriding the auto-detected
           numeric/nominal split (defaults to the baseline heuristic
           `np.all(data <= 1, axis=0)`).

Dependencies: NumPy, SciPy, Pandas only.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

EPS = 1e-12


# =====================================================================
# 1. SHARED CORE ENGINE
# =====================================================================

def infer_numeric_mask(data):
    """Baseline heuristic for numeric/nominal detection: numeric attributes
    are normalized into [0,1]; nominal attributes are integer-coded and
    assumed to take values > 1 somewhere. Mirrors `WNCROD`'s
    `np.all(data <= 1, axis=0)` so every variant sees the same attribute
    typing as the original algorithm unless explicitly overridden."""
    return np.all(data <= 1, axis=0)


def min_max_normalize(data, numeric_mask=None):
    """Eq. 6 min-max normalization (public utility for preprocessing raw
    data before it is handed to any variant). Applied only to numeric
    columns; nominal columns are left untouched."""
    data = np.asarray(data, dtype=float).copy()
    if numeric_mask is None:
        numeric_mask = np.ones(data.shape[1], dtype=bool)
    for j in np.where(numeric_mask)[0]:
        col = data[:, j]
        lo, hi = col.min(), col.max()
        data[:, j] = (col - lo) / (hi - lo) if (hi - lo) > EPS else 0.0
    return data


def _minmax_vector(v):
    """1-D min-max normalization with a zero-division guard."""
    lo, hi = np.min(v), np.max(v)
    if hi - lo > EPS:
        return (v - lo) / (hi - lo)
    return np.zeros_like(v)


def heom_neighbor_relation(data, col, delta):
    """Single-attribute HEOM-thresholded neighbor relation n_{c_j}: an
    (n, n) boolean matrix, True where the city-block distance on this
    single column is <= delta (equivalent, per column, to the numeric and
    nominal branches of Eq. 7)."""
    return cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta


def compute_all_neighbor_relations(data, delta):
    """Precomputes every single-attribute neighbor relation exactly once,
    eliminating the redundant repeated `cdist` calls of a naive port
    (Section 3 / Theorem 2 of the enhancement document relies on this)."""
    n, m = data.shape
    RMs = np.zeros((m, n, n), dtype=bool)
    for col in range(m):
        RMs[col] = heom_neighbor_relation(data, col, delta[col])
    return RMs


# ---- Eq. 8 (baseline) vs Eq. 8* (adaptive, parameter-free) -----------

def baseline_radius(data, numeric_mask, lammda):
    """Eq. 8 (original): std(c_j)/lambda for numeric attributes, 0 for
    nominal attributes."""
    m = data.shape[1]
    delta = np.zeros(m)
    if np.any(numeric_mask):
        delta[numeric_mask] = np.std(data[:, numeric_mask], axis=0) / lammda
    return delta


def adaptive_radius(data, numeric_mask, apply_floor=True):
    """Eq. 8*: parameter-free Silverman/MAD/IQR bandwidth.

    Edge-case handling (Section 4.1 of the document):
      - MAD == 0            -> fall back to IQR/1.34
      - MAD == IQR == 0      -> feature is behaviorally categorical, eps=0
      - otherwise            -> eps = max(0.9 * min(MAD, IQR/1.34) * n^-1/5, 1/n)
    """
    n, m = data.shape
    delta = np.zeros(m)
    for j in np.where(numeric_mask)[0]:
        col = data[:, j].astype(float)
        med = np.median(col)
        mad = 1.4826 * np.median(np.abs(col - med))
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        candidates = [s for s in (mad, iqr / 1.34) if s > EPS]
        if not candidates:
            delta[j] = 0.0  # near-constant / behaviorally categorical
            continue
        scale = min(candidates)
        bandwidth = 0.9 * scale * (n ** (-0.2))
        delta[j] = max(bandwidth, 1.0 / n) if apply_floor else bandwidth
    return delta


# ---- Region layer (Def. 1-4), shared and unmodified in form ----------

def region_indicators_cached(X, RMs):
    """Returns POS ('Lower'), NEB, NPB single-attribute indicators
    (Definitions 1-4), reusing precomputed neighbor relations."""
    m, n, _ = RMs.shape
    Lower = np.zeros((m, n))
    for col in range(m):
        RM = RMs[col]
        Lower[col, :] = np.min(np.maximum(1 - RM, np.tile(X, (n, 1))), axis=1)
    IB = np.tile(X, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))
    return Lower, NEB, NPB


def neg_indicator_cached(X, RMs):
    """NEG_{nr_{c_j}}(X) indicator per attribute. Not produced by the
    baseline; a single extra O(1)-per-object check inside the same loop
    already used for POS (Section 2.3 / Algorithm 1* delta)."""
    m, n, _ = RMs.shape
    Neg = np.zeros((m, n))
    for col in range(m):
        overlap = np.max(RMs[col].astype(float) * X[np.newaxis, :], axis=1)
        Neg[col, :] = (overlap <= EPS).astype(float)
    return Neg


def compute_NOM_cached(RMs):
    """Eq. 9 (unchanged in form): NOM(x,y) = |C| - (# attributes where x,y
    ARE neighbors). Domain automatically extends to C* when RMs includes
    the synthetic subspace attribute (Eq. 9*)."""
    m = RMs.shape[0]
    D_tem = np.sum(RMs.astype(float), axis=0)
    return m - D_tem


# ---- Eq. 4 (baseline NDF) vs Eq. 4* (tiered evidentiary NDF) ---------

def compute_NDF_baseline_cached(X_tem, X, Lower, NEB, NPB, RMs, D1, D2, D3):
    """Eq. 4 (original): unweighted cardinality-fusion NDF."""
    n_X = len(X_tem)
    NOM = compute_NOM_cached(RMs)
    X_OM = NOM[np.ix_(X_tem, X_tem)]
    m = RMs.shape[0]
    NEB_sub = NEB[X_tem]
    NDF = np.zeros((n_X, m))
    for col in range(m):
        pos_ind = Lower[col, X_tem]
        npb_ind = NPB[col, X_tem]
        for i in range(n_X):
            row = X_OM[i, :]
            neb_n = np.sum(np.minimum(NEB_sub, row <= D1))
            npb_n = np.sum(np.minimum(npb_ind, row >= D2))
            pos_n = np.sum(np.minimum(pos_ind, row >= D3))
            NDF[i, col] = (neb_n + npb_n + pos_n) / n_X
    return NDF


def compute_NDF_tiered_cached(X_tem, X, Lower, NEB, NPB, RMs, D1, D2, D3):
    """Eq. 4*: region-rarity tiered NDF, w_R = 1 - |R|/|X|. Strictly
    generalizes Eq. 4 (recovered when all w_R = 1)."""
    n_X = len(X_tem)
    NOM = compute_NOM_cached(RMs)
    X_OM = NOM[np.ix_(X_tem, X_tem)]
    m = RMs.shape[0]
    NEB_sub = NEB[X_tem]
    NEB_size = np.sum(NEB_sub)
    w_NEB = 1.0 - NEB_size / n_X
    NDF = np.zeros((n_X, m))
    for col in range(m):
        pos_ind = Lower[col, X_tem]
        npb_ind = NPB[col, X_tem]
        POS_size = np.sum(pos_ind)
        NPB_size = np.sum(npb_ind)
        w_POS = 1.0 - POS_size / n_X
        w_NPB = 1.0 - NPB_size / n_X
        denom = w_NEB * NEB_size + w_NPB * NPB_size + w_POS * POS_size
        denom = denom if denom > EPS else 1.0  # zero-division guard
        for i in range(n_X):
            row = X_OM[i, :]
            neb_n = np.sum(np.minimum(NEB_sub, row <= D1))
            npb_n = np.sum(np.minimum(npb_ind, row >= D2))
            pos_n = np.sum(np.minimum(pos_ind, row >= D3))
            NDF[i, col] = (w_NEB * neb_n + w_NPB * npb_n + w_POS * pos_n) / denom
    return NDF


# ---- Eq. 10 (baseline weight) vs Eq. 10* (discriminative weight) -----

def compute_weight_baseline_cached(X_tem, X, RMs):
    """Eq. 10 (original, prevalence-only). The explicit 1/|C| factor is
    intentionally omitted here and supplied later via `np.mean` at the
    fusion stage, exactly mirroring the baseline code's convention."""
    n_X = len(X_tem)
    m = RMs.shape[0]
    weight = np.zeros((n_X, m))
    for col in range(m):
        RM = RMs[col]
        for i in range(n_X):
            row = RM[X_tem[i], :]
            p = np.sum(np.minimum(row, X)) / n_X
            weight[i, col] = 1.0 - np.sqrt(max(p, 0.0))
    return weight


def compute_weight_discriminative_cached(X_tem, X, RMs, Lower, n, alpha=0.5):
    """Eq. 10*: bounded geometric-mean blend of the prevalence weight and
    the normalized discriminability gamma_hat_{c_j}(X) (rough
    approximation quality, reusing POS from `Lower` + freshly computed
    NEG at zero extra asymptotic order). alpha is fixed at 1/2
    (Jeffreys-symmetric default), not tuned."""
    m = RMs.shape[0]
    Neg = neg_indicator_cached(X, RMs)
    pos_count = np.sum(Lower, axis=1)
    neg_count = np.sum(Neg, axis=1)
    gamma = (pos_count + neg_count) / n
    gmax = np.max(gamma)
    gamma_hat = gamma / gmax if gmax > EPS else np.ones(m)

    base_weight = compute_weight_baseline_cached(X_tem, X, RMs)
    weight_star = (np.power(np.clip(base_weight, 0.0, None), 1 - alpha) *
                   np.power(np.clip(gamma_hat, 0.0, None), alpha)[np.newaxis, :])
    return weight_star


# ---- Algorithm 0* / Eq. 9*: subspace attribute synthesis -------------

def synthesize_subspace_attribute(data, numeric_mask, r=None, t_max=10, random_state=0):
    """Algorithm 0*: appends ONE synthetic numeric attribute c_{m+1} equal
    to the min-max normalized Squared Prediction Error (SPE / Q-statistic)
    against the top-r correlation directions of the numeric block,
    obtained via matrix-free deflationary power iteration (no m x m
    covariance matrix is ever formed or inverted).

    Degenerate case (Section 2.2 / 4.3): if fewer than 2 numeric attributes
    are available, no correlation structure exists; the data and mask are
    returned unchanged (r effectively 0), reducing exactly to baseline.
    """
    data = np.asarray(data, dtype=float)
    n, m = data.shape
    numeric_idx = np.where(numeric_mask)[0]
    m_num = len(numeric_idx)
    if m_num < 2:
        return data, numeric_mask, None

    Znum = data[:, numeric_idx].astype(float)
    med = np.median(Znum, axis=0)
    mad = 1.4826 * np.median(np.abs(Znum - med), axis=0)
    std = np.std(Znum, axis=0)
    scale = np.where(mad > EPS, mad, std)
    scale = np.where(scale > EPS, scale, 1.0)  # constant column guard
    Z = (Znum - med) / scale

    if r is None:
        r = int(np.ceil(np.log2(m_num + 1)))
    r = max(1, min(r, m_num))

    rng = np.random.RandomState(random_state)
    Zk = Z.copy()
    V = []
    for _ in range(r):
        v = rng.randn(m_num)
        norm_v = np.linalg.norm(v)
        v = v / norm_v if norm_v > EPS else v
        for _ in range(t_max):
            u = Zk.T @ (Zk @ v)
            norm_u = np.linalg.norm(u)
            if norm_u <= EPS:
                break
            v = u / norm_u
        V.append(v)
        proj = Zk @ v
        Zk = Zk - np.outer(proj, v)  # deflate for next component
    V = np.array(V)  # (r, m_num), rows approx. orthonormal

    proj_all = Z @ V.T  # (n, r): <z_x, v_k>
    J = np.sum(Z ** 2, axis=1) - np.sum(proj_all ** 2, axis=1)
    J = np.clip(J, 0.0, None)  # guard tiny negative rounding (Bessel's inequality => J >= 0)

    c_new = _minmax_vector(J)
    data_aug = np.hstack([data, c_new.reshape(-1, 1)])
    mask_aug = np.append(numeric_mask, True)
    return data_aug, mask_aug, c_new


# =====================================================================
# 2. UNIFIED CONFIGURABLE PIPELINE (used by Variants 1, 2, 3, 4, Full)
# =====================================================================

def _mnof_pipeline(data, X_tem, lammda=1.0, numeric_mask=None,
                    use_adaptive_radius=False,
                    use_discriminative_weight=False,
                    use_subspace_augmentation=False,
                    use_tiered_ndf=False,
                    alpha=0.5):
    data = np.asarray(data, dtype=float)
    X_tem = np.asarray(X_tem, dtype=int)
    n_X = len(X_tem)
    if n_X == 0:
        return np.array([])

    n = data.shape[0]
    X = np.zeros(n)
    X[X_tem] = 1.0

    if numeric_mask is None:
        numeric_mask = infer_numeric_mask(data)
    else:
        numeric_mask = np.asarray(numeric_mask, dtype=bool)

    # --- Idea 3 (Alg. 0*, Eq. 9*): subspace attribute augmentation ----
    if use_subspace_augmentation:
        data, numeric_mask, _ = synthesize_subspace_attribute(data, numeric_mask)

    m = data.shape[1]
    D1, D2, D3 = m / 3.0, m / 2.0, 0.9 * m

    # --- Idea 1 (Eq. 8 vs Eq. 8*): radius ------------------------------
    if use_adaptive_radius:
        delta = adaptive_radius(data, numeric_mask)
    else:
        if lammda is None:
            lammda = 1.0
        delta = baseline_radius(data, numeric_mask, lammda)

    RMs = compute_all_neighbor_relations(data, delta)
    Lower, NEB, NPB = region_indicators_cached(X, RMs)

    # --- Idea 4 (Eq. 4 vs Eq. 4*): NDF ---------------------------------
    if use_tiered_ndf:
        NDF = compute_NDF_tiered_cached(X_tem, X, Lower, NEB, NPB, RMs, D1, D2, D3)
    else:
        NDF = compute_NDF_baseline_cached(X_tem, X, Lower, NEB, NPB, RMs, D1, D2, D3)

    # --- Idea 2 (Eq. 10 vs Eq. 10*): weight ----------------------------
    if use_discriminative_weight:
        weight = compute_weight_discriminative_cached(X_tem, X, RMs, Lower, n, alpha=alpha)
    else:
        weight = compute_weight_baseline_cached(X_tem, X, RMs)

    MNOF = np.mean(NDF * weight, axis=1)
    return MNOF


# =====================================================================
# 3. VARIANT 0 — ORIGINAL BASELINE (UNTOUCHED, verbatim port)
# =====================================================================

def WNCROD(data, X_tem, lammda):
    """Three-way neighborhood characteristic region-based outlier
    detection (3WNCROD) algorithm — verbatim baseline, unmodified.

    Zhang Xianyong, Yuan Zhong, and Miao Duoqian. Outlier Detection Using
    Three-Way Neighborhood Characteristic Regions and Corresponding
    Fusion Measurement[J]. TKDE, 2023.
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


# =====================================================================
# 4. VARIANT CLASSES — IDENTICAL CALLING INTERFACE
# =====================================================================

class BaseOutlierDetector:
    """Common interface: `fit_predict(data, X_tem, lammda=..., numeric_mask=None)`."""
    name = "BaseOutlierDetector"

    def fit_predict(self, data, X_tem, lammda=1.0, numeric_mask=None):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Variant0_Baseline(BaseOutlierDetector):
    """Original 3WNCROD, untouched (calls `WNCROD` verbatim)."""
    name = "Variant 0: Baseline (original 3WNCROD)"

    def fit_predict(self, data, X_tem, lammda=1.0, numeric_mask=None):
        return WNCROD(np.asarray(data, dtype=float), np.asarray(X_tem, dtype=int), lammda)


class Variant1_AdaptiveRadius(BaseOutlierDetector):
    """Idea 1 ONLY: Eq. 8* parameter-free adaptive radius.
    NDF (Eq. 4) and weight (Eq. 10) remain strictly baseline."""
    name = "Variant 1: Adaptive Radius (Eq. 8*)"

    def fit_predict(self, data, X_tem, lammda=None, numeric_mask=None):
        return _mnof_pipeline(data, X_tem, lammda=lammda, numeric_mask=numeric_mask,
                               use_adaptive_radius=True)


class Variant2_DiscriminativeWeight(BaseOutlierDetector):
    """Idea 2 ONLY: Eq. 10* discriminability-augmented weight.
    Radius (Eq. 8) and NDF (Eq. 4) remain strictly baseline."""
    name = "Variant 2: Discriminative Weight (Eq. 10*)"

    def fit_predict(self, data, X_tem, lammda=1.0, numeric_mask=None):
        return _mnof_pipeline(data, X_tem, lammda=lammda, numeric_mask=numeric_mask,
                               use_discriminative_weight=True)


class Variant3_SubspaceAugmentation(BaseOutlierDetector):
    """Idea 3 ONLY: Algorithm 0* + Eq. 9* rank-r SPE synthetic attribute.
    Radius (Eq. 8), weight (Eq. 10) and NDF (Eq. 4) remain strictly
    baseline, applied over the augmented attribute set C*."""
    name = "Variant 3: Subspace Attribute Augmentation (Alg. 0*, Eq. 9*)"

    def fit_predict(self, data, X_tem, lammda=1.0, numeric_mask=None):
        return _mnof_pipeline(data, X_tem, lammda=lammda, numeric_mask=numeric_mask,
                               use_subspace_augmentation=True)


class Variant4_TieredNDF(BaseOutlierDetector):
    """Idea 4 ONLY: Eq. 4* region-rarity tiered NDF.
    Radius (Eq. 8) and weight (Eq. 10) remain strictly baseline."""
    name = "Variant 4: Tiered Evidentiary NDF (Eq. 4*)"

    def fit_predict(self, data, X_tem, lammda=1.0, numeric_mask=None):
        return _mnof_pipeline(data, X_tem, lammda=lammda, numeric_mask=numeric_mask,
                               use_tiered_ndf=True)


class VariantFull_Integrated(BaseOutlierDetector):
    """All four ideas combined (Eq. 8* + Eq. 9* + Eq. 10* + Eq. 4*)."""
    name = "Variant Full: Integrated (8*+9*+10*+4*)"

    def fit_predict(self, data, X_tem, lammda=None, numeric_mask=None):
        return _mnof_pipeline(data, X_tem, lammda=lammda, numeric_mask=numeric_mask,
                               use_adaptive_radius=True,
                               use_discriminative_weight=True,
                               use_subspace_augmentation=True,
                               use_tiered_ndf=True)


# =====================================================================
# 5. VERIFICATION / DEMONSTRATION BLOCK
# =====================================================================

if __name__ == '__main__':
    np.random.seed(42)
    LAMBDA_DEMO = 1.5

    # ---- Synthetic mixed (numeric + nominal) dataset -------------------
    n_normal = 40
    numeric_normal = np.random.normal(loc=0.5, scale=0.08, size=(n_normal, 4))
    numeric_normal[:, 1] = numeric_normal[:, 0] + np.random.normal(0, 0.01, n_normal)  # x0 ~ x1 correlated

    numeric_outliers = np.array([
        [0.92, 0.90, 0.88, 0.91],   # (a) classic marginal outlier: extreme on every axis
        [0.50, 0.05, 0.50, 0.50],   # (b) subspace/correlation-breaking outlier: each coordinate
                                    #     individually unremarkable, but breaks the x0~x1 correlation
    ])
    numeric_block = np.clip(np.vstack([numeric_normal, numeric_outliers]), 0.0, 1.0)

    n_total = numeric_block.shape[0]
    nominal_block = np.random.randint(2, 5, size=(n_total, 2)).astype(float)  # integer-coded, values > 1

    data = np.hstack([numeric_block, nominal_block])
    X_tem = np.arange(n_total)  # concept X = the whole synthetic universe

    classic_idx = n_normal
    subspace_idx = n_normal + 1

    print("Detected numeric mask:", infer_numeric_mask(data))
    print(f"n = {n_total}, injected outliers at indices {classic_idx} (classic) "
          f"and {subspace_idx} (subspace/correlation-breaking)\n")

    variants = [
        Variant0_Baseline(),
        Variant1_AdaptiveRadius(),
        Variant2_DiscriminativeWeight(),
        Variant3_SubspaceAugmentation(),
        Variant4_TieredNDF(),
        VariantFull_Integrated(),
    ]

    rows = []
    all_scores = {}
    for detector in variants:
        scores = detector.fit_predict(data, X_tem, lammda=LAMBDA_DEMO)
        all_scores[detector.name] = scores
        rows.append({
            "Variant": detector.name,
            "MNOF[classic outlier]": round(float(scores[classic_idx]), 4),
            "MNOF[subspace outlier]": round(float(scores[subspace_idx]), 4),
            "MNOF[normal mean]": round(float(scores[:n_normal].mean()), 4),
            "MNOF[normal std]": round(float(scores[:n_normal].std()), 4),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))

    print("\nFull MNOF vectors:")
    for name, scores in all_scores.items():
        print(f"\n{name}\n{np.round(scores, 4)}")
