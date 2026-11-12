# ---------------------------------------------------------------------------
#  3WNCROD -- optimised multiple neighborhood outlier factor (MNOF)
#
#  X. Zhang, Z. Yuan, D. Miao, "Outlier Detection Using Three-Way Neighborhood
#  Characteristic Regions and Corresponding Fusion Measurement", IEEE TKDE,
#  2023.  (Definitions 2-7, Eq. (3)-(11), Algorithm 1 and Algorithm 2)
#
#  Drop-in replacement for WNCROD() of main_code.py: same signature, same
#  output -- bit-for-bit identical, see benchmark_wncrod.py.
# ---------------------------------------------------------------------------
"""Vectorised 3WNCROD.

Symbol map (paper -> code)
--------------------------
================================  =========  ==============================
paper                             equation   code
================================  =========  ==============================
eps_{c_j}                         Eq. (8)    ``delta[j]``
d1 < d2 < d3                      Eq. (11)   ``D1, D2, D3``
n_{c_j}(x)  (HEOM, one feature)   Eq. (7)    ``|x - y| <= delta[j]``
POS_{nr_{c_j}}(X)                 Eq. (1)    ``POS[j]``   (POS subset of X)
NIB_{nr_{c_j}}(X)                 Eq. (2)    ``NIB[j] = ~POS[j]``
NEB(X)                            Def. 3     ``NEB = NIB.all(axis=0)``
NPB_{nr_{c_j}}(X)                 Def. 4     ``NPB[j] = NIB[j] & ~NEB``
NOM(x, y)                         Eq. (9)    ``m - cnt[x, y]``
|NEB^X(x)|                        Eq. (3)    ``neb_cnt[x]``
|NPB^X_{nr_{c_j}}(x)|             Eq. (3)    ``npb_cnt[x, j]``
|POS^X_{nr_{c_j}}(x)|             Eq. (3)    ``pos_cnt[x, j]``
NDF^X_{nr_{c_j}}(x)               Eq. (4)    ``ndf / n_X``
omega^X_{nr_{c_j}}(x)             Eq. (10)   ``weight[x, j] / m``
MNOF^X(x)                         Eq. (5)    return value
================================  =========  ==============================

Why this is fast
----------------
* Each single-attribute neighborhood relation is characterised **once**, as an
  interval of the sorted attribute order (O(n log n) -- the SANC complexity of
  Algorithm 1), instead of three full ``cdist`` calls over n x n per attribute.
* NOM (Eq. 9) is only ever consumed on X x X, so the pair loop costs
  O(m |X|^2) -- the complexity actually claimed for Algorithm 2 -- and runs on
  cache-resident row blocks of ``uint8`` counters.
* The Eq. (3) cardinalities become two BLAS matrix products and one masked
  popcount, so no Python-level loop over X survives.
* ``|n_{c_j}(x) ∩ X|`` of Eq. (10) is read off the interval widths for free.
* Peak memory is O(block x |X| + m x n) instead of a stack of n x n float64
  temporaries (``cdist``, ``np.tile``, ``D_tem``, ...).

Why the result is *identical*, not just close
---------------------------------------------
* Every decision uses the same float64 expression as the baseline,
  ``|x - y| <= eps``.  ``cdist`` promotes its input to double, so ``data`` is
  upcast here as well and float32 / integer input behaves identically, while
  ``delta`` is derived from the untouched ``data`` (same dtype, same reduction)
  as in the baseline.
* NOM is integral, so the float thresholds d1, d2, d3 are pre-resolved into
  exact integer cut-offs on the integer counter -- no float comparison of a
  reconstructed NOM value is needed.
* All cardinalities are sums of 0/1, hence exact in float32 up to 2**24 (a
  float64 gemm is used beyond that), and the closing
  ``mean((ndf / |X|) * weight, axis=1)`` is evaluated on an array of the same
  shape, dtype and layout, so NumPy's pairwise summation matches term by term.
* The sorted kernel is exact because IEEE rounding is monotone: for sorted
  values ``s`` the rounded difference ``fl(x - s_p)`` is non-increasing in
  ``p``, so ``{p : |fl(x - s_p)| <= eps}`` is a contiguous index range.  It is
  located by bisection *with that very predicate*, never by comparing against a
  perturbed bound such as ``x - eps``.  For the same reason only the two
  nearest values of the complement decide ``POS`` (Eq. 1).

``sorted_kernel=False`` falls back to the literal dense ``|x - y| <= eps``
evaluation (identical results, ~2-3x slower); it is also the path to use for
non-finite input, which the sorted kernel rejects instead of returning
silently wrong values.
"""

import numpy as np

__all__ = ["WNCROD_optimized"]


# ---------------------------------------------------------------------------
#  single-attribute neighborhood characterisation
# ---------------------------------------------------------------------------
def _neighbour_interval(x, s, rank, eps):
    """Neighborhood ``n_{c_j}(x)`` as an interval of the sorted value order.

    Parameters
    ----------
    x    : (k,) float64  attribute values, original object order
    s    : (k,) float64  the same values sorted ascending
    rank : (k,) intp     position of every object inside ``s`` (``s[rank] == x``)
    eps  : float         neighborhood radius of Eq. (8)

    Returns
    -------
    lo, hi : (k,) intp   ``{p : |x[i] - s[p]| <= eps} == range(lo[i], hi[i])``

    The set really is an interval: rounding to nearest is monotone, so
    ``fl(x - s_p)`` is non-increasing in ``p``; it is >= 0 left of ``rank[i]``
    and <= 0 right of it, hence the absolute value decreases then increases and
    the ``<= eps`` sub-level set is contiguous.  ``p = rank[i]`` always belongs
    to it (distance 0), which makes it a valid, non-empty bisection bracket.
    """
    k = x.size
    if not eps >= 0:                       # NaN radius: empty relation
        z = np.zeros(k, dtype=np.intp)
        return z, z.copy()

    # lower end: first p in [0, rank] that satisfies the predicate
    lo = np.zeros(k, dtype=np.intp)
    hi = rank.astype(np.intp, copy=True)
    while True:
        act = lo < hi
        if not act.any():
            break
        mid = (lo + hi) >> 1               # lo <= mid < hi
        ok = np.abs(x - s[mid]) <= eps
        np.copyto(hi, mid, where=act & ok)
        np.copyto(lo, mid + 1, where=act & ~ok)
    left = lo

    # upper end: last p in [rank, k - 1] that satisfies the predicate
    lo = rank.astype(np.intp, copy=True)
    hi = np.full(k, k - 1, dtype=np.intp)
    while True:
        act = lo < hi
        if not act.any():
            break
        mid = (lo + hi + 1) >> 1           # lo < mid <= hi
        ok = np.abs(x - s[mid]) <= eps
        np.copyto(lo, mid, where=act & ok)
        np.copyto(hi, mid - 1, where=act & ~ok)
    return left, lo + 1


def _has_outside_neighbour_sorted(x, out_vals, eps):
    """``n_{c_j}(x) ∩ ¬X != {}`` for every x in X, in O(k log |¬X|).

    Only the two values of ``¬X`` bracketing ``x`` can be its nearest ones on
    either side -- monotone rounding again -- so two exact distance tests per
    object replace a full row scan.
    """
    res = np.zeros(x.size, dtype=bool)
    if out_vals.size == 0 or not eps >= 0:
        return res
    s = np.sort(out_vals)
    pos = np.searchsorted(s, x, side="left")        # s[pos-1] < x <= s[pos]
    up = np.minimum(pos, s.size - 1)
    dn = np.maximum(pos - 1, 0)
    np.logical_or((pos < s.size) & (np.abs(x - s[up]) <= eps),
                  (pos > 0) & (np.abs(x - s[dn]) <= eps), out=res)
    return res


def _has_outside_neighbour_dense(x, out_vals, eps, block):
    """Literal ``any(|x - y| <= eps)`` over ``y`` in ``¬X``, in row blocks."""
    res = np.zeros(x.size, dtype=bool)
    if out_vals.size == 0:
        return res
    for start in range(0, x.size, block):
        stop = min(start + block, x.size)
        d = np.abs(x[start:stop, None] - out_vals[None, :])
        np.any(d <= eps, axis=1, out=res[start:stop])
    return res


# ---------------------------------------------------------------------------
#  3WNCROD (Algorithm 1 + Algorithm 2)
# ---------------------------------------------------------------------------
def WNCROD_optimized(data, X_tem, lammda, block_size=None, sorted_kernel=True):
    """Multiple neighborhood outlier factor ``MNOF^X(x)`` for every x in X.

    Parameters
    ----------
    data : (n, m) array_like
        Condition data without decisions.  Numerical attributes normalised into
        [0, 1] (Eq. 6), nominal attributes coded as distinct integers; exactly
        as in the baseline a column counts as nominal (eps = 0) as soon as it
        holds a value > 1.
    X_tem : (|X|,) integer array, or (n,) bool mask
        The selected condition subdata X.  Integer indices must be unique; the
        order of the returned factors follows their order.
    lammda : float
        Radius adjustment lambda of Eq. (8).
    block_size : int, optional
        Number of X-rows handled per block in the O(m |X|^2) pass.  Default:
        about 2M counters per block, i.e. a working set that stays in L2/L3.
    sorted_kernel : bool, optional
        True (default): characterise every neighborhood as an interval of the
        sorted attribute order.  False: literal dense ``|x - y| <= eps``
        evaluation -- identical results, slower, tolerates non-finite input.

    Returns
    -------
    MNOF : (|X|,) float64
        ``np.allclose(WNCROD(data, X_tem, lammda), <this>, atol=1e-12)`` -- in
        fact the two are bit-for-bit equal.

    Notes
    -----
    Two silent failure modes of the baseline are turned into errors here:
    duplicated indices in ``X_tem`` (the baseline then mixes ``len(X_tem)`` and
    ``|X|``) and ``m == 0`` (an empty reduction).  A boolean mask is accepted in
    addition to indices.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D (samples x attributes) array")
    n, m = data.shape
    if m == 0:
        raise ValueError("data must have at least one attribute")

    # -- 0. the concept X ---------------------------------------------------
    X_tem = np.asarray(X_tem)
    if X_tem.dtype == np.bool_:
        X_idx = np.flatnonzero(X_tem)
    else:
        X_idx = np.ascontiguousarray(X_tem, dtype=np.intp).ravel()
    n_X = X_idx.size
    if n_X == 0:
        return np.zeros(0, dtype=np.float64)

    in_X = np.zeros(n, dtype=bool)                 # characteristic function
    in_X[X_idx] = True
    if int(np.count_nonzero(in_X)) != n_X:
        raise ValueError("X_tem must hold unique object indices")
    C_idx = np.flatnonzero(~in_X)                  # ¬X

    # -- 1. radii (Eq. 8) and the NOM cut-offs (Eq. 11) ---------------------
    D1, D2, D3 = m / 3, m / 2, 0.9 * m
    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)                 # numerical attributes
    if ID.any():
        delta[ID] = np.std(data[:, ID], axis=0) / lammda

    # NOM(x, y) = m - cnt(x, y) is an integer, so each float threshold becomes
    # one exact integer comparison on the counter.
    om = np.arange(m + 1)
    s1, s2, s3 = om <= D1, om >= D2, om >= D3
    k1 = m - int(om[s1].max()) if s1.any() else None   # cnt >= k1 <=> NOM <= d1
    k2 = m - int(om[s2].min()) if s2.any() else None   # cnt <= k2 <=> NOM >= d2
    k3 = m - int(om[s3].min()) if s3.any() else None   # cnt <= k3 <=> NOM >= d3

    # -- 2. attribute-major float64 working copies -------------------------
    work = np.ascontiguousarray(data, dtype=np.float64)   # cdist promotes too
    if sorted_kernel and not np.isfinite(work).all():
        raise ValueError("sorted_kernel=True requires finite data; use "
                         "sorted_kernel=False for NaN/inf input")
    VX = np.ascontiguousarray(work[X_idx].T)              # (m, |X|)
    VC = np.ascontiguousarray(work[C_idx].T)              # (m, |¬X|)
    del work

    if block_size is None:
        block_size = min(n_X, max(64, (1 << 21) // n_X))
    block_size = int(min(max(1, int(block_size)), n_X))

    # -- 3. one pass per attribute (Algorithm 1) ----------------------------
    #    POS_{nr_{c_j}}(X) of Eq. (1) and |n_{c_j}(x) ∩ X| of Eq. (10)
    POS = np.empty((m, n_X), dtype=bool)
    hit = np.empty((n_X, m), dtype=np.float64)
    if sorted_kernel:
        rank_all = np.empty((m, n_X), dtype=np.int32)
        lo_all = np.empty((m, n_X), dtype=np.int32)
        hi_all = np.empty((m, n_X), dtype=np.int32)
        iota = np.arange(n_X, dtype=np.intp)
    for j in range(m):
        v, eps = VX[j], delta[j]
        if sorted_kernel:
            order = np.argsort(v, kind="stable")
            s = v[order]
            rank = np.empty(n_X, dtype=np.intp)
            rank[order] = iota
            lo, hi = _neighbour_interval(v, s, rank, eps)
            rank_all[j] = rank
            lo_all[j] = lo
            hi_all[j] = hi
            hit[:, j] = hi - lo                    # |n_{c_j}(x) ∩ X|, for free
            POS[j] = ~_has_outside_neighbour_sorted(v, VC[j], eps)
        else:
            POS[j] = ~_has_outside_neighbour_dense(v, VC[j], eps, block_size)

    # -- 4. three-way inner regions (Def. 2, 3, 4) --------------------------
    NIB = ~POS                     # X - POS_{nr_{c_j}}(X)          (Eq. 2)
    NEB = NIB.all(axis=0)          # intersection over attributes   (Def. 3)
    NPB = NIB & ~NEB               # NIB - NEB                      (Def. 4)
    del NIB

    # -- 5. blocked X x X pass: NOM (Eq. 9) and the Eq. (3) cardinalities ---
    gemm_dt = np.float32 if n_X < (1 << 24) else np.float64   # 0/1 dots exact
    NPB_f = np.ascontiguousarray(NPB.T, dtype=gemm_dt)        # (|X|, m)
    POS_f = np.ascontiguousarray(POS.T, dtype=gemm_dt)        # (|X|, m)

    cnt_dt = np.uint8 if m <= 255 else (np.uint16 if m <= 65535 else np.uint32)
    cnt = np.empty((block_size, n_X), dtype=cnt_dt)
    buf1 = np.empty((block_size, n_X), dtype=bool)
    buf2 = np.empty((block_size, n_X), dtype=bool)
    fbuf = np.empty((block_size, n_X), dtype=gemm_dt)
    dbuf = None if sorted_kernel else np.empty((block_size, n_X), np.float64)

    neb_cnt = np.zeros(n_X, dtype=np.float64)          # |NEB^X(x)|
    npb_cnt = np.zeros((n_X, m), dtype=np.float64)     # |NPB^X_{nr_{c_j}}(x)|
    pos_cnt = np.zeros((n_X, m), dtype=np.float64)     # |POS^X_{nr_{c_j}}(x)|

    for start in range(0, n_X, block_size):
        stop = min(start + block_size, n_X)
        b = stop - start
        c, a, t = cnt[:b], buf1[:b], buf2[:b]
        c.fill(0)

        # cnt[i, y] = #{j : (x_i, y) in nr_{c_j}}  ->  NOM = m - cnt  (Eq. 9)
        for j in range(m):
            if sorted_kernel:
                r = rank_all[j]
                np.greater_equal(r, lo_all[j][start:stop, None], out=a)
                np.less(r, hi_all[j][start:stop, None], out=t)
                np.logical_and(a, t, out=a)
            else:
                v, d = VX[j], dbuf[:b]
                np.subtract(v[start:stop, None], v[None, :], out=d)
                np.abs(d, out=d)
                np.less_equal(d, delta[j], out=a)
                hit[start:stop, j] = np.count_nonzero(a, axis=1)
            c += a

        # three-way neighborhood characteristic regions of x  (Eq. 3)
        if k1 is not None:                            # NOM <= d1, y in NEB(X)
            np.greater_equal(c, k1, out=a)
            np.logical_and(a, NEB, out=a)
            neb_cnt[start:stop] = np.count_nonzero(a, axis=1)
        if k2 is not None:                            # NOM >= d2, y in NPB
            np.less_equal(c, k2, out=a)
            np.copyto(fbuf[:b], a)
            npb_cnt[start:stop] = fbuf[:b] @ NPB_f
        if k3 is not None:                            # NOM >= d3, y in POS
            np.less_equal(c, k3, out=a)
            np.copyto(fbuf[:b], a)
            pos_cnt[start:stop] = fbuf[:b] @ POS_f

    # -- 6. weights (Eq. 10) and the fused factor (Eq. 4 -> Eq. 5) ----------
    weight = 1 - np.sqrt(hit / n_X)                 # m * omega^X_{nr_{c_j}}(x)
    ndf = neb_cnt[:, None] + pos_cnt + npb_cnt      # numerator of Eq. (4)
    return np.mean((ndf / n_X) * weight, axis=1)    # Eq. (5), the 1/m of Eq. 10
