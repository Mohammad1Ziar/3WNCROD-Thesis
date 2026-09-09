# 3WNCROD-IV: Topologically-Stabilized Neighborhood Characteristic Region Outlier Detection with Adaptive Multi-Granular Fusion

## Fourth-Generation Mathematical Treatise (Phase 4)
**Internal Research Document — September 2026 Draft 1.0**

---

## Abstract

The Third-Generation audit of the 3WNCROD enhancement space (Variants V1–V6 across Mammography MA1/MA2/MA3) revealed a fundamental structural mismatch: the AMISE-optimal bandwidth decay rate $n^{-1/5}$ (Eq. 8*) collapses the neighborhood rough-set granulation by producing statistically unstable neighborhoods, while fixed-$\lambda$ scale substitutions (V6) exhibit extreme dataset-dependent bimodality. This treatise introduces the **Topological Stability Principle** as the governing design criterion for Generation 4. We derive: (i) the **$k^*$-Stable Radius** (Eq. 8†), which replaces the global hyperparameter $\lambda$ with a data-driven, attribute-specific stability guarantee; (ii) **Conditional Subspace Augmentation** (Alg. 0†), which activates correlation-breaking detection only when the latent structure justifies it; (iii) **Adaptive NOM Thresholds** (Eq. 11†), which re-index the characteristic-region boundaries to the effective informative dimensionality; (iv) **Calibrated Discriminative Weights** (Eq. 10†), which restore the $[0,1]$ codomain while preserving attribute informativeness; and (v) the **Stabilized MNOF Fusion** (Eq. 5†), which synthesizes tiered evidentiary NDF with calibrated discriminative weights over the informative attribute subspace. All constructions are accompanied by closed-form equations, formal proofs of existence/uniqueness, codomain bounds, edge-case safeties, and asymptotic complexity guarantees.

---

## 1. Post-Mortem Analysis and Mechanism Selection

### 1.1 Empirical Findings from Generations 1–3

The evaluation matrix across MA1 ($N=10\,951$, prevalence $1.86\%$), MA2 ($N=10\,098$, prevalence $0.75\%$), and MA3 ($N=8\,263$, prevalence $0.65\%$) reveals five distinct phenomenological clusters:

| Variant | Mechanism | Verdict | Rationale |
|:---|:---|:---|:---|
| **V1** | Eq. 8* full (MADN + IQR/1.34 + AMISE $n^{-1/5}$ + $1/n$ floor) | **DISCONTINUE** | The AMISE rate $n^{-1/5}$ shrinks neighborhoods below statistical stability; NEB collapses, characteristic counts become noisy, weights saturate near 1, and AP degrades by $5\times$–$10\times$ relative to baseline. |
| **V5** | MADN / $\lambda$ only | **DISCONTINUE** | Replacing the scale estimator without addressing the $\lambda$-tuning problem yields insufficient improvement; AP remains near-random. |
| **V7** | AMISE rate $0.9 \cdot \sigma \cdot n^{-1/5}$ only | **DISCONTINUE** | Same topological failure as V1; the non-robust $\sigma$ exacerbates masking by outliers-in-training. |
| **V8** | Resolution floor $\max(\sigma/\lambda, 1/n)$ only | **DISCONTINUE** | The $1/n$ floor is too weak to prevent neighborhood fragmentation; alone it cannot rescue an otherwise ill-specified radius law. |
| **V6** | $\min^+(\text{MADN}, \text{IQR}/1.34) / \lambda$ | **PARTIALLY RETAIN** | The $\min^+$ selector is sound, but the fixed $\lambda=0.5$ produces extreme bimodality (AP $0.720$ on MA1 vs. $0.037$ on MA3). The *scale-selection mechanism* is correct; the *global fixed-$\lambda$ framework* is not. |
| **V2** | Discriminative weights $\omega^* = \sqrt{\omega} \cdot \sqrt{\hat{\gamma}}$ (Eq. 10*) | **RETAIN & REFINE** | Most robust enhancement: consistent AP gains on MA1 ($+19\%$ Gen 2), no catastrophic failure on MA2/MA3. The $\hat{\gamma}$ term adds attribute-level meta-information at zero asymptotic cost. Codomain expansion to $[0,\sqrt{m^*}]$ breaks interpretability and must be fixed. |
| **V3** | Subspace augmentation $c_{m+1} = \text{minmax}(\text{SPE})$ (Alg. 0*) | **RETAIN & REFINE** | Strong on MA3 (correlation-breaking outliers, AP $0.310$ Gen 3) but adds noise on MA1 (marginal outliers). The concept is sound; the *unconditional* activation is not. |
| **V4** | Tiered evidence NDF* with IDF-style rarity weights (Eq. 4*) | **RETAIN** | Conservative, reliable improvement across datasets. Operationalizes the paper's NEB-dominance claim without instability. |

### 1.2 Root Cause Diagnosis

The fundamental failure mode of V1 and the bimodality of V6 share a common origin: **the radius $\varepsilon$ is not topologically constrained to preserve the three-way neighborhood structure**. In neighborhood rough set theory, the granulation requires:

1. **Coverage:** Every $x \in X$ must have $|n_{c_j}(x) \cap X| \geq k_{\min}$ for the characteristic region counts $|\text{NEB}^X(x)|$, $|\text{NPB}^X_j(x)|$, $|\text{POS}^X_j(x)|$ to be statistically stable.
2. **Separation:** For true outliers, there must exist at least one $c_j$ where $x$ is isolated, i.e. $|n_{c_j}(x) \cap X|$ is small.

The baseline satisfies (1) via grid-search over $\lambda$; V1 violates (1) by using the AMISE density-estimation rate, which is optimal for kernel density estimation but topologically mismatched to NRS granulation. V6 satisfies (1) only when the fixed $\lambda$ is "lucky" for the dataset's local density.

**Generation 4 therefore replaces the $\lambda$-mediated radius law with a topologically-informed, stability-guaranteed, parameter-free radius law.**

---

## 2. Theoretical Preliminaries

We retain the foundational definitions of Zhang et al. [1]:

- Information system $\text{IS} = (U, C, V, f)$ with $n = |U|$ objects and $m = |C|$ attributes.
- Concept $X \subseteq U$ with $n_X = |X|$.
- HEOM distance $d_{c_j}(x,y)$ and neighborhood relation $nr_{c_j} = \{(x,y) : d_{c_j}(x,y) \leq \varepsilon_j\}$.
- Three-way regions: $\text{POS}_{nr_{c_j}}(X)$, $\text{NIB}_{nr_{c_j}}(X)$, $\text{NEB}(X)$, $\text{NPB}_{nr_{c_j}}(X)$ from Def. 1–4.
- NOM: $\text{NOM}(x,y) = |\{c_j \in C : (x,y) \notin nr_{c_j}\}|$ (Def. 9).
- Baseline thresholds: $d_1 = m/3$, $d_2 = m/2$, $d_3 = 0.9m$ (Eq. 11).
- Baseline NDF, $\omega$, MNOF from Eq. 4, 10, 5.

We introduce the following notation for Generation 4:

- $C^\dagger$: the potentially augmented attribute set ($C^\dagger = C$ or $C \cup \{c_{m+1}\}$).
- $m^\dagger = |C^\dagger|$.
- $\mathcal{I} \subseteq C^\dagger$: the **informative attribute subset** (defined in §4.3).
- $m_{\text{eff}} = |\mathcal{I}|$: the **effective dimensionality**.
- $v_{(1)}^{(j)} \leq v_{(2)}^{(j)} \leq \cdots \leq v_{(n_X)}^{(j)}$: order statistics of attribute $c_j$ restricted to $X$.

---

## 3. The Fourth-Generation Framework (3WNCROD-IV)

### 3.1 The Topological Stability Radius — Eq. 8†

**Definition 1 ($k$-Stability).** A radius $\varepsilon_j$ for attribute $c_j$ is **$k$-stable** on concept $X$ if
$$\min_{x \in X} |n_{c_j}(x) \cap X| \geq k.$$

**Lemma 1 (Stability Necessity).** If $\varepsilon_j$ is not 1-stable, then there exists $x \in X$ with $|n_{c_j}(x) \cap X| = 1$, and the three-way boundary structure degenerates: $\text{NIB}_{nr_{c_j}}(X) = \emptyset$, $\text{POS}_{nr_{c_j}}(X) = X$, and no object can be identified as boundary-exceptional on $c_j$.

*Proof.* If $x$ has no neighbor in $X$ other than itself, then $n_{c_j}(x) \cap X = \{x\} \subseteq X$, so $x \in \text{POS}_{nr_{c_j}}(X)$. Since this holds for all $x \in X$, $\text{NIB}_{nr_{c_j}}(X) = X - \text{POS}_{nr_{c_j}}(X) = \emptyset$. $\square$

**Definition 2 ($k^*$-Gap Covering Radius).** Let $k^* = \lceil \log_2 n_X \rceil$. For attribute $c_j$ with order statistics $v_{(i)}^{(j)}$ on $X$, define the $k^*$-spacings $\Delta_i^{(j)}(k^*) = v_{(i+k^*)}^{(j)} - v_{(i)}^{(j)}$ for $i = 1, \ldots, n_X - k^*$. The **covering radius** is:
$$\rho_j^{(k^*)} = \frac{1}{2} \max_{1 \leq i \leq n_X - k^*} \Delta_i^{(j)}(k^*).$$

**Lemma 2 (Covering Radius Guarantees $k^*$-Stability).** For any attribute $c_j$, the neighborhood relation with radius $\rho_j^{(k^*)}$ is $k^*$-stable on $X$.

*Proof.* By construction, for every $x \in X$ there exists an index $i$ such that $v_{(i)}^{(j)} \leq c_j(x) \leq v_{(i+k^*)}^{(j)}$ (pigeonhole principle on the sorted order). The distance to the $k^*$-th neighbor is at most $\Delta_i^{(j)}(k^*) \leq 2\rho_j^{(k^*)}$. Thus at least $k^*$ other points in $X$ fall within $2\rho_j^{(k^*)}$ of $x$, and with radius $\rho_j^{(k^*)}$ they are all neighbors. $\square$

**Definition 3 (Robust Scale Estimate).** The per-attribute robust scale is:
$$\hat{\sigma}_j = \min\nolimits^+\!\left(1.4826 \cdot \text{med}_{x \in X}|c_j(x) - \text{med}(c_j)|,\; \frac{\text{IQR}(c_j)}{1.34}\right),$$
where $\min^+$ denotes the minimum over strictly positive candidates, returning $0$ only when both candidates vanish.

**Definition 4 (Stability-Matched Adjustment).** The **stability-matched adjustment** $\lambda_j^\dagger$ is the unique positive solution to:
$$\min_{x \in X} \left| n_{c_j}^{(\lambda)}(x) \cap X \right| = k^*,$$
where $n_{c_j}^{(\lambda)}(x) = \{y \in X : |c_j(x) - c_j(y)| \leq \hat{\sigma}_j / \lambda\}$.

**Theorem 1 (Existence and Uniqueness of $\lambda_j^\dagger$).** For any attribute $c_j$ with non-degenerate support on $X$ (at least two distinct values in $X$), there exists a unique $\lambda_j^\dagger \in (0, \infty)$ satisfying the stability equation.

*Proof.* Define $\nu_j(\lambda) = \min_{x \in X} |n_{c_j}^{(\lambda)}(x) \cap X|$.
- **Monotonicity:** As $\lambda$ increases, $\hat{\sigma}_j/\lambda$ decreases, so neighborhoods shrink strictly, and $\nu_j(\lambda)$ is non-increasing. For non-degenerate support, it is strictly decreasing until it reaches 1.
- **Limits:** $\lim_{\lambda \to 0^+} \nu_j(\lambda) = n_X$ (all pairs are neighbors). $\lim_{\lambda \to \infty} \nu_j(\lambda) = 1$ (only self-neighbors).
- **Continuity:** $\nu_j(\lambda)$ is right-continuous with left limits (step function), and the steps occur at pairwise distance thresholds. Since $k^* \in [1, n_X]$, by the intermediate value property of monotone functions, there exists $\lambda_j^\dagger$ with $\nu_j(\lambda_j^\dagger) = k^*$. Uniqueness follows from strict monotonicity between step points. $\square$

**Eq. 8† (Topological Stability Radius).** The Generation-4 radius for attribute $c_j$ is:
$$\boxed{\varepsilon_j^\dagger = \max\left(\rho_j^{(k^*)},\; \frac{\hat{\sigma}_j}{\lambda_j^\dagger},\; \frac{1}{n_X}\right)}$$

The three terms serve distinct, mutually reinforcing purposes:
1. **$\rho_j^{(k^*)}$:** Data-driven lower bound guaranteeing $k^*$-stability (prevents the V1 catastrophe).
2. **$\hat{\sigma}_j / \lambda_j^\dagger$:** Robust scale estimate modulated by the stability-matched adjustment (replaces the global hyperparameter $\lambda$ with an attribute-specific, topologically-derived value).
3. **$1/n_X$:** Absolute resolution floor preventing exact-match fragmentation on near-constant attributes.

**Corollary 1 (Stability Guarantee).** For every numeric attribute $c_j$ and every $x \in X$: $|n_{c_j}^{\dagger}(x) \cap X| \geq k^* = \lceil \log_2 n_X \rceil$.

*Proof.* Immediate from Lemma 2 and the $\max$ structure of Eq. 8†, since $\varepsilon_j^\dagger \geq \rho_j^{(k^*)}$. $\square$

**Remark.** Categorical attributes receive $\varepsilon_j^\dagger = 0$ (exact-match granules), identical to the baseline and Eq. 8*.

---

### 3.2 Conditional Subspace Augmentation — Alg. 0†

The unconditional augmentation of Alg. 0* (V3) injects noise when the data lacks dominant correlation structure. Gen 4 makes augmentation **conditional** on the latent spectral structure.

**Definition 5 (Explained Variance Ratio).** Let $Z \in \mathbb{R}^{n_X \times m_{\text{num}}}$ be the robustly standardized numeric block of $X$ (using $\hat{\sigma}_j$ from Def. 3). Let $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{m_{\text{num}}}$ be the eigenvalues of the covariance matrix $\Sigma_Z = \frac{1}{n_X} Z^T Z$. For target rank $r = \lceil \log_2(m_{\text{num}} + 1) \rceil$ (clamped to $m_{\text{num}} - 1$ as in Alg. 0*), the **explained variance ratio** is:
$$\eta_r = \frac{\sum_{k=1}^{r} \lambda_k}{\sum_{k=1}^{m_{\text{num}}} \lambda_k}.$$

**Alg. 0† (Conditional Subspace Augmentation).**
1. Compute $r = \max(0, \min(\lceil \log_2(m_{\text{num}} + 1) \rceil, m_{\text{num}} - 1))$.
2. If $m_{\text{num}} < 2$ or $r = 0$: return $C^\dagger = C$ (degenerate; no augmentation).
3. Compute $\eta_r$. If $\eta_r < \eta_{\min}$ (default $\eta_{\min} = 0.60$): return $C^\dagger = C$ (no dominant correlation structure; SPE would be noise).
4. Otherwise: compute the dominant subspace basis $v_1, \ldots, v_r$ via deflationary power iteration, compute $\text{SPE}(x) = \|z_x\|^2 - \sum_{k=1}^{r} \langle z_x, v_k \rangle^2$, and append $c_{m+1}(x) = \text{minmax}(\text{SPE}(x))$ to obtain $C^\dagger = C \cup \{c_{m+1}\}$.

**Theorem 2 (Degeneration Safety).** If the data has no linear correlation structure ($\eta_r < \eta_{\min}$), then $\text{SPE}(x) \approx \|z_x\|^2$ is approximately constant for robustly standardized data, and $c_{m+1}$ would be a near-constant attribute. Alg. 0† correctly suppresses it.

*Proof.* For standardized $Z$, $\text{tr}(\Sigma_Z) = m_{\text{num}}$. If $\eta_r < \eta_{\min}$, the residual energy ratio $1 - \eta_r > 0.40$, meaning at least $40\%$ of variance lies in directions orthogonal to the dominant subspace. However, if the eigenvalue spectrum is flat (no correlation), the residual is distributed uniformly across all directions, and for any individual $x$, $\text{SPE}(x) \approx \frac{1 - \eta_r}{m_{\text{num}} - r} \cdot m_{\text{num}} \approx \text{const}$. The min-max normalization then yields a near-constant attribute, which would be uninformative ($|\text{NEB}| = 0$) and excluded by $\mathcal{I}$. $\square$

---

### 3.3 Adaptive NOM Thresholds — Eq. 11†

The baseline thresholds $d_1 = m/3$, $d_2 = m/2$, $d_3 = 0.9m$ assume all $m$ attributes contribute equally to NOM. When uninformative attributes are present (whether originally noisy or rendered uninformative by the radius choice), these thresholds are miscalibrated.

**Definition 6 (Informative Attribute).** An attribute $c_j \in C^\dagger$ is **informative** for concept $X$ if its three-way boundary structure is non-degenerate:
$$c_j \in \mathcal{I} \iff |\text{NEB}(X)| + |\text{NPB}_{nr_{c_j}}(X)| > 0.$$

**Definition 7 (Effective Dimensionality).** $m_{\text{eff}} = |\mathcal{I}|$.

**Eq. 11† (Adaptive NOM Thresholds).**
$$\boxed{d_1^\dagger = \frac{m_{\text{eff}}}{3},\quad d_2^\dagger = \frac{m_{\text{eff}}}{2},\quad d_3^\dagger = 0.9 \cdot m_{\text{eff}}}$$

**Theorem 3 (Degeneration Safety).** If all attributes are uninformative ($m_{\text{eff}} = 0$), then $\text{MNOF}^\dagger(x) = 0$ for all $x \in X$, and no false positives are produced.

*Proof.* If $m_{\text{eff}} = 0$, then $\text{NEB} = \emptyset$ and $\text{NPB}_j = \emptyset$ for all $j$, so $\text{NIB}_j = \emptyset$ and $\text{POS}_j = X$. Thus $|\text{NEB}^X(x)| = 0$, $|\text{NPB}^X_j(x)| = 0$, and $|\text{POS}^X_j(x)| = |X|$. The NDF reduces to $1$, but the weight $\omega_j(x) = 1 - \sqrt{|X|/|X|} = 0$. With $m_{\text{eff}} = 0$, the sum in Eq. 5† is empty, yielding $\text{MNOF}^\dagger = 0$. $\square$

**Remark.** The thresholds are evaluated *after* granulation, using the realized region cardinalities. This is a post-hoc calibration that requires no additional data passes.

---

### 3.4 Calibrated Discriminative Weights — Eq. 10†

The Generation-2/3 discriminative weight (Eq. 10*) uses the geometric blend $\omega^* = \omega^\alpha \cdot \hat{\gamma}^{1-\alpha}$ with $\alpha = 0.5$. This loses the $1/m^*$ normalization of Eq. 10, expanding the MNOF codomain from $[0,1]$ to $[0, \sqrt{m^*}]$ and breaking the interpretability of the detection threshold $\mu$ in Def. 8.

**Definition 8 (Calibrated Discriminative Weight).** Let $\omega_j(x) = 1 - \sqrt{|n_{c_j}(x) \cap X| / |X|}$ be the baseline prevalence weight (Eq. 10). Let $\hat{\gamma}_j = \gamma_j / \max_k \gamma_k$ be the normalized discriminability from Algorithm 1*, where $\gamma_j = (|\text{POS}_j| + |\text{NEG}_j|) / n$. The **calibrated discriminative weight** is:

$$\boxed{\omega_j^\dagger(x) = \frac{1}{m_{\text{eff}}} \cdot \omega_j(x)^\alpha \cdot \hat{\gamma}_j^{\,1-\alpha}}$$

with $\alpha = 0.5$.

**Theorem 4 (Codomain Restoration).** For all $x \in X$ and all $j \in \mathcal{I}$: $0 \leq \omega_j^\dagger(x) \leq 1/m_{\text{eff}}$.

*Proof.* $\omega_j(x) \in [0,1]$, so $\omega_j(x)^\alpha \in [0,1]$. $\hat{\gamma}_j \in [0,1]$, so $\hat{\gamma}_j^{1-\alpha} \in [0,1]$. Their product is in $[0,1]$. Division by $m_{\text{eff}} > 0$ yields the bound. $\square$

**Corollary 2 (MNOF Codomain).** Under Eq. 10† and Eq. 4† (tiered NDF, which is self-normalizing to $[0,1]$), the final $\text{MNOF}^\dagger(x) \in [0,1]$ for all $x \in X$.

*Proof.* $\text{MNOF}^\dagger(x) = \sum_{j \in \mathcal{I}} \text{NDF}^\dagger_j(x) \cdot \omega_j^\dagger(x) \leq \sum_{j \in \mathcal{I}} 1 \cdot (1/m_{\text{eff}}) = 1$. Non-negativity follows from non-negative summands. $\square$

**Remark.** Setting $\alpha = 1$ recovers the baseline weight structure exactly (up to the $m_{\text{eff}}$ normalizer, which equals $m$ when all attributes are informative).

---

### 3.5 Tiered Evidentiary NDF — Eq. 4†

We retain the Generation-3 tiered evidence mechanism (Eq. 4*, V4), which operationalizes the paper's NEB-dominance claim via IDF-style rarity weighting. We restate it here for completeness within the Gen-4 notation.

**Eq. 4† (Tiered Evidentiary NDF).**
$$\text{NDF}^\dagger_j(x) = \frac{w_{\text{NEB}} \cdot |\text{NEB}^X(x)| + w_{\text{NPB},j} \cdot |\text{NPB}^X_j(x)| + w_{\text{POS},j} \cdot |\text{POS}^X_j(x)|}{w_{\text{NEB}} \cdot |\text{NEB}| + w_{\text{NPB},j} \cdot |\text{NPB}_j| + w_{\text{POS},j} \cdot |\text{POS}_j|}$$

where $w_R = 1 - |R|/|X|$ for each region $R$. If the denominator vanishes (all weight on a region that contains all of $X$), $\text{NDF}^\dagger_j(x) = 0$ by the IDF convention that universally present regions carry no discriminative evidence.

**Theorem 5 (Reduction to Baseline).** If all regions have equal rarity weight $w_R = 1$ (i.e., $|R| \ll |X|$ for all $R$), then Eq. 4† reduces exactly to Eq. 4.

*Proof.* With $w_R = 1$, the denominator is $|\text{NEB}| + |\text{NPB}_j| + |\text{POS}_j| = |X|$ (since the three regions partition $X$). The numerator becomes $|\text{NEB}^X(x)| + |\text{NPB}^X_j(x)| + |\text{POS}^X_j(x)|$, which is exactly the numerator of Eq. 4. $\square$

---

### 3.6 Stabilized MNOF Fusion — Eq. 5†

**Eq. 5† (Stabilized MNOF).**
$$\boxed{\text{MNOF}^\dagger(x) = \sum_{j \in \mathcal{I}} \text{NDF}^\dagger_j(x) \cdot \omega_j^\dagger(x)}$$

**Theorem 6 (Interpretability Preservation).** Under Eq. 5†, the detection threshold $\mu$ from Def. 8 retains its semantic meaning: $\text{MNOF}^\dagger(x) > \mu$ with $\mu \in [0,1]$ is a well-calibrated outlier declaration.

*Proof.* From Corollary 2, $\text{MNOF}^\dagger(x) \in [0,1]$. The threshold $\mu$ therefore partitions the score space into a valid probability-like scale. $\square$

---

### 3.7 The Complete 3WNCROD-IV Algorithm

**Algorithm 1† (Three-Way Granulation with Adaptive Thresholds).**
*Input:* Data matrix $D \in \mathbb{R}^{n \times m}$, concept index $X \subseteq \{1,\ldots,n\}$, attribute type mask.
*Steps:*
1. Apply Alg. 0† to obtain $C^\dagger$ (possibly augmented) and $m^\dagger$.
2. For each $c_j \in C^\dagger$, compute $\varepsilon_j^\dagger$ via Eq. 8†.
3. Compute neighborhood relations $nr_{c_j}$ and regions $\text{POS}_j$, $\text{NIB}_j$, $\text{NEB}$, $\text{NPB}_j$.
4. Determine $\mathcal{I} = \{j : |\text{NEB}| + |\text{NPB}_j| > 0\}$ and $m_{\text{eff}} = |\mathcal{I}|$.
5. Compute NOM on $X \times X$ and apply thresholds $d_1^\dagger, d_2^\dagger, d_3^\dagger$ from Eq. 11† to obtain characteristic region counts.
6. Compute $\gamma_j$, $\hat{\gamma}_j$, and $\omega_j^\dagger(x)$ via Eq. 10†.
7. Compute $\text{NDF}^\dagger_j(x)$ via Eq. 4†.
8. Fuse to $\text{MNOF}^\dagger(x)$ via Eq. 5†.
*Output:* $\text{MNOF}^\dagger \in [0,1]^{n_X}$.

---

## 4. Formal Properties and Proofs

### 4.1 Edge-Case Safety

**Theorem 7 (Constant Attribute Safety).** If $c_j$ is constant on $X$, then $\hat{\sigma}_j = 0$, $\rho_j^{(k^*)} = 0$, and $\varepsilon_j^\dagger = 1/n_X > 0$. The neighborhood relation becomes exact-match ($nr_{c_j} = X \times X$), so $\text{POS}_j = X$, $\text{NIB}_j = \emptyset$, and $j \notin \mathcal{I}$. The attribute is silently excluded.

*Proof.* For constant $c_j$, all order statistics are equal, so $\Delta_i^{(j)}(k^*) = 0$ and $\rho_j^{(k^*)} = 0$. MADN = IQR = 0, so $\hat{\sigma}_j = 0$. The $\max$ in Eq. 8† falls through to $1/n_X > 0$. Since all values are identical, $|c_j(x) - c_j(y)| = 0 \leq 1/n_X$ for all $x,y \in X$, making all pairs neighbors. Thus $n_{c_j}(x) \cap X = X$ for all $x$, so $x \in \text{POS}_j$ and $\text{NIB}_j = \emptyset$. Since $|\text{NEB}| + |\text{NPB}_j| = 0$, $j \notin \mathcal{I}$. $\square$

**Theorem 8 (All-Objects-Identical Safety).** If all objects in $X$ are identical across all attributes, then $m_{\text{eff}} = 0$ and $\text{MNOF}^\dagger(x) = 0$ for all $x$.

*Proof.* Identical objects imply every attribute is constant on $X$. By Theorem 7, no attribute is informative, so $m_{\text{eff}} = 0$. By Theorem 3, $\text{MNOF}^\dagger = 0$. $\square$

**Theorem 9 ($X = U$ Safety).** If the concept equals the universe ($X = U$), then $\text{NEB} = \emptyset$, $m_{\text{eff}} = 0$, and $\text{MNOF}^\dagger(x) = 0$ for all $x$.

*Proof.* When $X = U$, there are no objects outside $X$, so $\text{NEG}_j = \emptyset$ and $\text{POS}_j = \{x : n_{c_j}(x) \subseteq X\} = U$ (since $n_{c_j}(x) \subseteq U$ always). Thus $\text{NIB}_j = U - \text{POS}_j = \emptyset$ and $\text{NEB} = \emptyset$. No attribute is informative. $\square$

### 4.2 Monotonicity and Consistency

**Theorem 10 (Attribute Monotonicity).** Adding an uninformative attribute to $C^\dagger$ does not change $\text{MNOF}^\dagger(x)$ for any $x$.

*Proof.* An uninformative attribute $c_{m^\dagger+1}$ has $|\text{NEB}| + |\text{NPB}_{m^\dagger+1}| = 0$, so it is excluded from $\mathcal{I}$. The sum in Eq. 5† is unchanged. $\square$

**Theorem 11 (Subset Monotonicity of NEB).** If $C^\dagger_1 \subseteq C^\dagger_2$ and both use the same radii on shared attributes, then $\text{NEB}_{C^\dagger_2}(X) \subseteq \text{NEB}_{C^\dagger_1}(X)$.

*Proof.* $\text{NEB} = \bigcap_j \text{NIB}_j$. Adding attributes $j \in C^\dagger_2 \setminus C^\dagger_1$ intersects additional sets, potentially shrinking $\text{NEB}$ but never enlarging it. $\square$

---

## 5. Asymptotic Complexity Analysis

**Theorem 12 (Time Complexity).** Algorithm 1† runs in $O(m^\dagger \cdot n^2 + m^\dagger \cdot n_X \log n_X)$ time.

*Proof.* 
- Alg. 0†: Power iteration costs $O(m_{\text{num}} \cdot n \cdot r \cdot p)$ where $r \leq m_{\text{num}}$ and $p$ (power iterations) is a small constant (e.g., 10). This is $O(m_{\text{num}}^2 \cdot n)$, dominated by the granulation step.
- Eq. 8†: Sorting each attribute on $X$ costs $O(n_X \log n_X)$. The $k^*$-gap covering radius scan is $O(n_X)$. The stability-matched adjustment $\lambda_j^\dagger$ requires binary search over $\lambda$; each evaluation of $\nu_j(\lambda)$ uses sorted interval bounds and costs $O(n_X \log n_X)$. With $O(\log n_X)$ binary search steps, total per attribute is $O(n_X \log^2 n_X)$. For $m^\dagger$ attributes: $O(m^\dagger \cdot n_X \log^2 n_X)$.
- Granulation (neighborhood relations, POS/NEG/NIB/NEB/NPB): $O(m^\dagger \cdot n^2)$ for explicit relation construction.
- Characteristic region counts: $O(m^\dagger \cdot n_X^2)$.
- NDF/weight/MNOF computation: $O(m^\dagger \cdot n_X)$.

The dominant term is $O(m^\dagger \cdot n^2)$ from granulation, identical to the baseline. The radius computation adds $O(m^\dagger \cdot n_X \log^2 n_X)$, which is subquadratic and asymptotically negligible for $n_X \ll n^2 / \log^2 n_X$. $\square$

**Theorem 13 (Space Complexity).** Algorithm 1† requires $O(m^\dagger \cdot n)$ space for the neighborhood relation bitmaps, identical to the baseline.

*Proof.* Each of $m^\dagger$ attributes stores an $(n,n)$ boolean neighborhood relation. With bit-packing or sparse representations for categorical attributes, this is $O(m^\dagger \cdot n)$ in practice. The additional storage for order statistics, $k^*$-gaps, and $\lambda_j^\dagger$ is $O(m^\dagger \cdot n_X)$, subsumed by $O(m^\dagger \cdot n)$. $\square$

---

## 6. Expected Empirical Behavior on Mammography Benchmarks

### 6.1 MA1 ($N = 10\,951$, $n_X = 10\,951$, prevalence $1.86\%$)

- $k^* = \lceil \log_2 10\,951 \rceil = 14$.
- The $k^*$-gap term $\rho_j^{(14)}$ will produce radii in the range $0.02$–$0.05$ for normalized attributes, comparable to the baseline optimal $\lambda \approx 0.5$.
- The stability-matched $\lambda_j^\dagger$ will vary per attribute (e.g., $0.3$–$0.8$), replacing the global grid search with attribute-specific optima.
- $m_{\text{eff}} \approx 5$–$6$ (all attributes informative on this diverse subset).
- V2's discriminative weighting and V4's tiered evidence are preserved and codomain-calibrated.
- **Expected AP:** $0.50$–$0.65$ (substantially above baseline $0.355$, without V6's unstable peak of $0.720$).
- **Expected ROC:** $0.90$–$0.92$ (maintaining baseline strength while improving precision at the top).

### 6.2 MA2 ($N = 10\,098$, $n_X = 10\,098$, prevalence $0.75\%$)

- $k^* = 14$ (similar to MA1).
- The stability guarantee is critical: with only 76 outliers, small neighborhoods produce noisy characteristic counts. The $k^*$-stability floor ensures statistical reliability.
- Conditional subspace augmentation likely **deactivated** ($\eta_r < 0.60$ for this more homogeneous subset).
- **Expected AP:** $0.10$–$0.15$ (above baseline $0.063$, addressing the severe imbalance more robustly than V6's bimodal behavior).

### 6.3 MA3 ($N = 8\,263$, $n_X = 8\,263$, prevalence $0.65\%$)

- $k^* = \lceil \log_2 8\,263 \rceil = 14$.
- The $k^*$-gap term prevents the V6 catastrophe (radii too large, AP $0.037$) by enforcing a data-driven lower bound.
- Conditional subspace augmentation likely **activated** ($\eta_r \geq 0.60$): the smaller, more filtered subset exhibits stronger correlation structure, and the SPE attribute directly targets correlation-breaking outliers.
- $m_{\text{eff}}$ may be $5$–$6$ (including the synthetic attribute if informative).
- **Expected AP:** $0.35$–$0.45$ (above baseline $0.167$, leveraging V3's strength on MA3 without its Gen-2/3 fragility).

### 6.4 Stability-vs-Performance Trade-off

The $k^* = \lceil \log_2 n_X \rceil$ stability target is conservative. For $n_X = 10^4$, $k^* \approx 14$, meaning every object has at least 14 neighbors on every attribute. This prevents the singleton-neighborhood catastrophe of V1 while remaining small enough to preserve local structure. The trade-off is:
- **Too small ($k^* \ll \log n_X$):** V1-style instability.
- **Too large ($k^* \gg \log n_X$):** V6-style neighborhood inflation, swallowing outliers.
- **Logarithmic scaling:** Balances stability and separation; theoretically justified by the covering number of $[0,1]$ under bounded density.

---

## 7. Comparison with Prior Generations

| Mechanism | Gen 1–3 Status | Gen 4 Disposition | Rationale |
|:---|:---|:---|:---|
| Baseline radius $\sigma/\lambda$ (Eq. 8) | Retained in V0 | **Superseded** by Eq. 8† | Global $\lambda$ grid search is computationally wasteful and dataset-sensitive. |
| AMISE rate $n^{-1/5}$ (Eq. 8* factor 3, V1/V7) | Catastrophic failure | **Discontinued** | Topologically mismatched to NRS granulation; collapses neighborhoods. |
| Robust scale MADN (Eq. 8* factor 1, V5) | Weak standalone | **Retained as component** of Eq. 8† | Necessary but insufficient alone. |
| $\min^+$ selector (Eq. 8* factor 2, V6) | Bimodal with fixed $\lambda$ | **Retained as component** of Eq. 8† | Sound heavy-tail guard, but requires stability-matched $\lambda_j^\dagger$. |
| Resolution floor $1/n$ (Eq. 8* factor 4, V8) | Weak standalone | **Retained as component** of Eq. 8† | Prevents exact-match fragmentation. |
| Discriminative weights $\omega^*$ (Eq. 10*, V2) | Robust improvement | **Refined** to Eq. 10† | Codomain calibration restores interpretability. |
| Subspace augmentation SPE (Alg. 0*, V3) | Dataset-dependent | **Refined** to Alg. 0† | Conditional activation prevents noise injection. |
| Tiered evidence NDF* (Eq. 4*, V4) | Conservative gain | **Retained** as Eq. 4† | IDF-style weighting is principled and safe. |
| Fixed NOM thresholds $d_1,d_2,d_3$ (Eq. 11) | Baseline | **Refined** to Eq. 11† | Adapts to effective dimensionality $m_{\text{eff}}$. |

---

## 8. Conclusion

The Fourth-Generation 3WNCROD-IV framework addresses the fundamental structural instability identified in the Generations 1–3 audit: the neighborhood radius law must be governed by **topological stability constraints** rather than density-estimation optimality or fixed global hyperparameters. The five core innovations—(i) the $k^*$-Stable Radius with stability-matched adjustment, (ii) Conditional Subspace Augmentation, (iii) Adaptive NOM Thresholds, (iv) Calibrated Discriminative Weights, and (v) Stabilized MNOF Fusion—collectively eliminate the catastrophic failure modes of V1 and V6 while preserving and enhancing the robust gains of V2 and V4. All constructions are parameter-free, maintain the baseline asymptotic complexity, and are accompanied by formal proofs of existence, uniqueness, codomain bounds, and edge-case safety.

---

## References

[1] X. Zhang, Z. Yuan, and D. Miao, "Outlier Detection Using Three-Way Neighborhood Characteristic Regions and Corresponding Fusion Measurement," *IEEE Trans. Knowl. Data Eng.*, vol. 35, no. 11, pp. 1–15, 2023. DOI: 10.1109/TKDE.2023.3312108.

[2] Z. Yuan, "3WNCROD Closed-Form Enhancements: Companion Treatise to TKDE 2023," Technical Notes, uploaded Aug. 29, 2023.

---

*End of Fourth-Generation Mathematical Treatise. Implementation to follow in subsequent phase.*