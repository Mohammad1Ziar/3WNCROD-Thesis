"""
Unified 3WNCROD Implementation and Theoretical Enhancements.

This script implements the baseline 3WNCROD algorithm[cite: 3] and five theoretical 
closed-form enhancements[cite: 2] addressing neighborhood granulation, feature discriminability, 
and subspace outlier detection[cite: 1]. 

References:
[cite: 1] Zhang, Yuan & Miao, "Outlier Detection Using Three-Way Neighborhood Characteristic Regions..."
[cite: 2] 3WNCROD_closed_form_enhancements.md
[cite: 3] main_code.py
"""

import time
import numpy as np
from scipy.spatial.distance import cdist


# =============================================================================
# SHARED MATHEMATICAL PRIMITIVES[cite: 2]
# =============================================================================

def _get_silverman_radii(data, ID):
    """
    Computes parameter-free adaptive radii using Silverman's robust KDE bandwidth[cite: 2].
    Replaces global standard deviation with MADN and IQR[cite: 2].
    """
    n, m = data.shape
    delta = np.zeros(m)
    for j in range(m):
        if ID[j]:
            col = data[:, j]
            med = np.median(col)
            mad = np.median(np.abs(col - med))
            madn = 1.4826 * mad
            q75, q25 = np.percentile(col, [75, 25])
            iqr = q75 - q25
            
            if madn == 0 and iqr == 0:
                delta[j] = 0.0
            else:
                val = 0.9 * min(madn, iqr / 1.34) * (n ** -0.2)
                delta[j] = max(val, 1.0 / n)
    return delta


def _get_spe_feature(data, ID, t_max=10):
    """
    Computes the Squared Prediction Error (SPE / Q-statistic) using matrix-free
    deflationary power iteration to capture linear subspace outliers[cite: 2].
    """
    num_cols = np.where(ID)[0]
    m_num = len(num_cols)
    if m_num < 2:
        return None
        
    n = data.shape[0]
    Z = np.zeros((n, m_num))
    
    # Robust standardization
    for idx, c in enumerate(num_cols):
        col = data[:, c]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        sigma_hat = 1.4826 * mad
        if sigma_hat == 0:
            sigma_hat = 1.0
        Z[:, idx] = (col - med) / sigma_hat

    # Power Iteration
    r = int(np.ceil(np.log2(m_num + 1)))
    V = []
    Z_k = Z.copy()
    for _ in range(r):
        u = np.ones(m_num) / np.sqrt(m_num)
        for _ in range(t_max):
            Zu = Z_k.dot(u)
            u_next = Z_k.T.dot(Zu)
            norm_u = np.linalg.norm(u_next)
            if norm_u < 1e-12:
                break
            u = u_next / norm_u
        V.append(u)
        Z_k = Z_k - np.outer(Z_k.dot(u), u)

    # SPE residual
    Z_sq_norm = np.sum(Z**2, axis=1)
    proj_sq_norm = np.zeros(n)
    for u in V:
        proj_sq_norm += (Z.dot(u))**2
        
    J = Z_sq_norm - proj_sq_norm
    J[J < 0] = 0
    
    j_min, j_max = np.min(J), np.max(J)
    if j_max == j_min:
        return np.zeros(n)
    return (J - j_min) / (j_max - j_min)


# =============================================================================
# VARIANT 0: BASELINE
# =============================================================================

def WNCROD_v0_baseline(data, X_tem, lammda=0.5):
    """
    Original baseline implementation exactly as given in main_code.py[cite: 3].
    """
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1 = m / 3
    D2 = m / 2
    D3 = 0.9 * m

    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    if np.any(ID):
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
# VARIANT 1: ADAPTIVE RADIUS
# =============================================================================

def WNCROD_v1_adaptive_radius(data, X_tem):
    """
    Isolates Enhancement 2.1 (Eq. 8*). Parameter-free Silverman rule of thumb[cite: 2].
    """
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1, D2, D3 = m / 3, m / 2, 0.9 * m

    ID = np.all(data <= 1, axis=0)
    delta = _get_silverman_radii(data, ID)  #[cite: 2]

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
        weight[:, col] = weight_x

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
# VARIANT 2: DISCRIMINATIVE WEIGHTS
# =============================================================================

def WNCROD_v2_discriminative_weights(data, X_tem, lammda=0.5):
    """
    Isolates Enhancement 2.3 (Eq. 10*). Blends prevalence with approximation quality[cite: 2].
    """
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1, D2, D3 = m / 3, m / 2, 0.9 * m

    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    if np.any(ID):
        delta[ID] = np.std(data[:, ID], axis=0) / lammda

    Lower = np.zeros((m, n))
    gamma = np.zeros(m)

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        Lower_temp = np.min(np.maximum(1 - RM_tem, np.tile(X, (n, 1))), axis=1)
        Lower[col, :Lower_temp.shape[0]] = Lower_temp
        
        # Track NEG region for discriminability[cite: 2]
        NEG_temp = np.min(np.maximum(1 - RM_tem, np.tile(1 - X, (n, 1))), axis=1)
        gamma[col] = (np.sum(Lower_temp) + np.sum(NEG_temp)) / n

    max_g = np.max(gamma)
    hat_gamma = gamma / max_g if max_g > 0 else np.ones(m)

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
            # Blend geometric mean[cite: 2]
            omega_star = np.sqrt(weight_temp * hat_gamma[col])
            weight_x.append(omega_star)
        weight[:, col] = weight_x

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
# VARIANT 3: SUBSPACE AUGMENTATION
# =============================================================================

def WNCROD_v3_subspace_augmentation(data, X_tem, lammda=0.5):
    """
    Isolates Enhancement 2.2 (Algorithm 0* and Eq. 9*). Augments data with SPE feature[cite: 2].
    """
    ID = np.all(data <= 1, axis=0)
    c_m1 = _get_spe_feature(data, ID)  #[cite: 2]
    
    if c_m1 is not None:
        data = np.column_stack((data, c_m1))
        
    return WNCROD_v0_baseline(data, X_tem, lammda)


# =============================================================================
# VARIANT 4: TIERED NDF
# =============================================================================

def WNCROD_v4_tiered_ndf(data, X_tem, lammda=0.5):
    """
    Isolates Enhancement 2.4 (Eq. 4*). Replaces unweighted NDF with tiered rarity weights[cite: 2].
    """
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1, D2, D3 = m / 3, m / 2, 0.9 * m

    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    if np.any(ID):
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
        weight[:, col] = weight_x

    D_tem = np.zeros((n, n))
    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        D_tem += RM_tem

    NOM = m - D_tem
    X_OM = NOM[np.ix_(X_tem, X_tem)]

    NEB_num = np.zeros((n_X, m))
    Lower_num = np.zeros((n_X, m))
    NPB_num = np.zeros((n_X, m))

    # Regional evidentiary weights[cite: 2]
    w_neb = 1.0 - np.sum(NEB) / n_X
    w_npb = np.zeros(m)
    w_pos = np.zeros(m)
    for col in range(m):
        w_npb[col] = 1.0 - np.sum(NPB[col, :]) / n_X
        w_pos[col] = 1.0 - np.sum(Lower[col, :]) / n_X

    NDF_tiered = np.zeros((n_X, m))

    for col in range(m):
        temp2 = Lower[col, :]
        temp3 = NPB[col, :]
        for i in range(n_X):
            temp1 = X_OM[i, :]
            NEB_num[i, col] = np.sum(np.minimum(NEB[X_tem], temp1 <= D1))
            NPB_num[i, col] = np.sum(np.minimum(temp3[X_tem], temp1 >= D2))
            Lower_num[i, col] = np.sum(np.minimum(temp2[X_tem], temp1 >= D3))
            
        denom = w_neb * np.sum(NEB) + w_npb[col] * np.sum(NPB[col, :]) + w_pos[col] * np.sum(Lower[col, :])
        if denom > 0:
            num = (w_neb * NEB_num[:, col] + w_npb[col] * NPB_num[:, col] + w_pos[col] * Lower_num[:, col])
            NDF_tiered[:, col] = num / denom

    MNOF = np.mean(NDF_tiered * weight, axis=1)
    return MNOF


# =============================================================================
# VARIANT FULL: COMBINED ENHANCEMENTS
# =============================================================================

def WNCROD_v_full(data, X_tem):
    """
    Unified model combining all theoretical enhancements (Alg 0*, Eq. 8*, 10*, 4*, 5*)[cite: 2].
    Fully parameter-free.
    """
    ID = np.all(data <= 1, axis=0)
    c_m1 = _get_spe_feature(data, ID)
    if c_m1 is not None:
        data = np.column_stack((data, c_m1))
        ID = np.append(ID, True)

    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1, D2, D3 = m / 3, m / 2, 0.9 * m
    delta = _get_silverman_radii(data, ID)

    Lower = np.zeros((m, n))
    gamma = np.zeros(m)

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        Lower_temp = np.min(np.maximum(1 - RM_tem, np.tile(X, (n, 1))), axis=1)
        Lower[col, :Lower_temp.shape[0]] = Lower_temp
        
        NEG_temp = np.min(np.maximum(1 - RM_tem, np.tile(1 - X, (n, 1))), axis=1)
        gamma[col] = (np.sum(Lower_temp) + np.sum(NEG_temp)) / n

    max_g = np.max(gamma)
    hat_gamma = gamma / max_g if max_g > 0 else np.ones(m)

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
            omega_star = np.sqrt(weight_temp * hat_gamma[col])
            weight_x.append(omega_star)
        weight[:, col] = weight_x

    D_tem = np.zeros((n, n))
    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        D_tem += RM_tem

    NOM = m - D_tem
    X_OM = NOM[np.ix_(X_tem, X_tem)]

    NEB_num = np.zeros((n_X, m))
    Lower_num = np.zeros((n_X, m))
    NPB_num = np.zeros((n_X, m))

    w_neb = 1.0 - np.sum(NEB) / n_X
    w_npb = np.zeros(m)
    w_pos = np.zeros(m)
    for col in range(m):
        w_npb[col] = 1.0 - np.sum(NPB[col, :]) / n_X
        w_pos[col] = 1.0 - np.sum(Lower[col, :]) / n_X

    NDF_tiered = np.zeros((n_X, m))

    for col in range(m):
        temp2 = Lower[col, :]
        temp3 = NPB[col, :]
        for i in range(n_X):
            temp1 = X_OM[i, :]
            NEB_num[i, col] = np.sum(np.minimum(NEB[X_tem], temp1 <= D1))
            NPB_num[i, col] = np.sum(np.minimum(temp3[X_tem], temp1 >= D2))
            Lower_num[i, col] = np.sum(np.minimum(temp2[X_tem], temp1 >= D3))
            
        denom = w_neb * np.sum(NEB) + w_npb[col] * np.sum(NPB[col, :]) + w_pos[col] * np.sum(Lower[col, :])
        if denom > 0:
            num = (w_neb * NEB_num[:, col] + w_npb[col] * NPB_num[:, col] + w_pos[col] * Lower_num[:, col])
            NDF_tiered[:, col] = num / denom

    MNOF = np.mean(NDF_tiered * weight, axis=1)
    return MNOF


# =============================================================================
# EXECUTION GUARD & VERIFICATION
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    n_samples, n_features = 150, 6
    
    # Generate synthetic mixed data
    X_num = np.random.rand(n_samples, n_features - 1)
    X_cat = np.random.randint(0, 3, size=(n_samples, 1))
    synth_data = np.column_stack((X_num, X_cat))
    
    X_tem_indices = np.arange(60) # Target concept spanning first 60 instances
    
    algorithms = [
        ("v0_Baseline", lambda: WNCROD_v0_baseline(synth_data, X_tem_indices, lammda=0.5)),
        ("v1_Adaptive_Radius", lambda: WNCROD_v1_adaptive_radius(synth_data, X_tem_indices)),
        ("v2_Discriminative_Weights", lambda: WNCROD_v2_discriminative_weights(synth_data, X_tem_indices, lammda=0.5)),
        ("v3_Subspace_Augmentation", lambda: WNCROD_v3_subspace_augmentation(synth_data, X_tem_indices, lammda=0.5)),
        ("v4_Tiered_NDF", lambda: WNCROD_v4_tiered_ndf(synth_data, X_tem_indices, lammda=0.5)),
        ("v_Full_Unified", lambda: WNCROD_v_full(synth_data, X_tem_indices))
    ]

    print(f"Running Validation on Synthetic Data ({n_samples} objects, {n_features} features, Target size: {len(X_tem_indices)})...\n")
    for name, func in algorithms:
        start_t = time.perf_counter()
        mnof_scores = func()
        exec_t = time.perf_counter() - start_t
        print(f"[{name:<25}] Shape: {mnof_scores.shape} | Time: {exec_t:.4f}s | Range: [{np.min(mnof_scores):.4f}, {np.max(mnof_scores):.4f}]")