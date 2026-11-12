# Second-Generation Closed-Form Enhancements to 3WNCROD: Decoupled Multi-Scale Granulation, Calibrated Cross-Attribute Voting, and Unsupervised Global Feature Weighting

**Addendum II to:** *3WNCROD_closed_form_enhancements.md* ("Addendum I"). All notation, equations, lemmas, and theorems of Addendum I ($\S0$–$\S6$; Eq. 1–10$^\star$; Lemmas 1–2; Theorems 1–2) are retained unmodified and extended below. New equations are numbered 11–14; results that directly supersede a starred result of Addendum I are marked $^{\star\star}$.

---

## 0. Notation Addendum II

| Symbol | Meaning |
|---|---|
| $HL(c_j),\,HU(c_j)$ | Lower/upper half-spreads: $\mathrm{med}(c_j)-Q_1(c_j)$, $Q_3(c_j)-\mathrm{med}(c_j)$ |
| $IQR_{eff}(c_j)$ | Asymmetry-corrected dispersion, $2\min(HL(c_j),HU(c_j))$ |
| $d$ | Residual subspace dimension, $d=m_{num}-r$ (corrected; §3.2) |
| $q_j$ | Empirical outside-radius base rate of $c_j$: $\frac{1}{n(n-1)}\sum_{x\neq y}\mathbb 1[d_{c_j}(x,y)>\varepsilon_{c_j}]$ |
| $\bar q$ | Mean base rate over the active attribute set, $\frac1{m^\star}\sum_k q_k$ |
| $\hat I_j(x,y)$ | Rate-calibrated outside-radius vote (Eq. 11) |
| $\rho_j(x_i)$ | Normalized granule occupancy of $x_i$ on $c_j$: $\lvert n_{c_j}(x_i)\rvert/n$ |
| $CV(c_j)$ | Coefficient of variation of $\{\rho_j(x_i)\}_{i=1}^n$ |
| $W(c_j)$ | Unsupervised global feature weight (Eq. 12), $CV(c_j)/\max_k CV(c_k)$ |

---

## 1. Epistemic Scope of This Addendum

> **Remark 0.** The Mammography grid-sweep figures (MA1–MA3, V0–V4) are treated below as externally supplied empirical input, not as quantities re-derivable from first principles — no closed-form argument can *prove* a decimal AUC or AP value on a specific finite real-world sample, since that value depends on the sample's actual joint distribution, not merely on the algebraic form of the estimator. Consistent with that limit, every claim below is labeled by its actual epistemic status: **Proposition/Theorem** where a property is derived deductively from the construction (boundedness, monotonicity, exact recovery of a special case, asymptotic order), and **Conjecture** where a claim concerns the *direction and mechanism* of an expected empirical effect and therefore requires re-benchmarking to confirm its magnitude. This mirrors ordinary practice in applied approximation theory: a bandwidth selector's AMISE-optimality is provable; its exact effect on one dataset's AUC is not.

---

## 2. Diagnostic Correspondence

| # | Restated weakness | Empirical symptom (this round) | Addendum II mechanism | Equation |
|---|---|---|---|---|
| 1 | Locked feature radii | V1 (Eq. 8$^\star$ alone) trails grid-optimal $\lambda$; symmetric bandwidth over-smooths skewed tabular features | Asymmetry-corrected radius | 8$^{\star\star}$ |
| 3a | Subspace blindness — scale conflict | Naive V3+V4 hybrid dilutes V3's AP peak because $c_{m+1}$ and $C$ want different $\lambda$ | Emergent per-attribute harmonization (no manual dual-$\lambda$) | 8$^{\star\star}$, Cor. 5 |
| 3b | Subspace blindness — vote distortion | Injecting $c_{m+1}$ at a different scale can bias $D_1,D_2,D_3$ gating in $NOM$ | Rate-calibrated cross-attribute voting | 11 |
| 2 | Uninformative weights — Eq. 10$^\star$ failure | V2 collapses: $\gamma_{c_j}(X)$ penalizes exactly the features carrying contamination signal, since $X$ is an unpurified mixture | Fully unsupervised dispersion-based weight, no $POS/NEG$ dependence | 12 |

---

## 3. Second-Generation Reformulations

### 3.1 Eq. 8$^{\star\star}$ — asymmetry-aware adaptive radius

**Diagnosis.** Eq. 8$^\star$'s bandwidth $0.9\min(\hat\sigma_{c_j},IQR(c_j)/1.34)\,n^{-1/5}$ presumes the two robust scale estimates are calibrated against a roughly *symmetric* reference distribution. For a right-skewed feature, $IQR(c_j)=HL(c_j)+HU(c_j)$ is inflated by the elongated tail relative to the spread that actually governs resolution in the dense core, so the resulting radius over-smooths precisely where discrimination matters — the mechanism V1's empirical gap and the V3+V4 scale conflict share.

**Reformulation.**

$$
IQR_{eff}(c_j)=2\min\big(HL(c_j),\,HU(c_j)\big),\qquad HL=\mathrm{med}-Q_1,\ \ HU=Q_3-\mathrm{med}
\tag{new}
$$

$$
\varepsilon_{c_j}^{\star\star}=
\begin{cases}
0, & c_j\text{ categorical}\\[4pt]
\max\!\Big(0.9\,\min\big(\hat\sigma_{c_j},\,IQR_{eff}(c_j)/1.34\big)\cdot n^{-1/5},\ \dfrac1n\Big), & c_j\text{ numeric},\ \hat\sigma_{c_j}>0\text{ or }IQR_{eff}(c_j)>0\\[6pt]
0, & c_j\text{ numeric},\ \hat\sigma_{c_j}=IQR_{eff}(c_j)=0
\end{cases}
\tag{8$^{\star\star}$}
$$

Drop-in replacement for Eq. 8$^\star$; same branch structure, same floor, same MAD-fallback discipline.

> **Proposition 3 (Monotone tightening).** $\varepsilon_{c_j}^{\star\star}\le\varepsilon_{c_j}^{\star}$ for every attribute and every sample, with equality iff $HL(c_j)=HU(c_j)$ (symmetric quartile spread) or the $\hat\sigma_{c_j}$ branch already binds in both. *Proof.* $IQR_{eff}=2\min(HL,HU)\le HL+HU=IQR$ (since $2\min(a,b)\le a+b$ for $a,b\ge0$), so $IQR_{eff}/1.34\le IQR/1.34$, hence $\min(\hat\sigma_{c_j},IQR_{eff}/1.34)\le\min(\hat\sigma_{c_j},IQR/1.34)$, and $\max(\cdot,1/n)$ is monotone in its first argument. $\blacksquare$

This is a strict *refinement*, not a new free parameter: symmetric attributes are untouched; asymmetric ones are tightened automatically, in proportion to their own measured asymmetry.

### 3.2 Correction to Algorithm 0$^\star$ — residual-dimension floor

Checking $r=\lceil\log_2(m_{num}+1)\rceil$ against $d=m_{num}-r$: for $m_{num}=2$, $r=\lceil\log_2 3\rceil=2=m_{num}$, giving $d=0$ — the residual subspace vanishes and $J(x)\equiv0$, so $c_{m+1}$ degenerates to a constant and $\mathrm{minmax}(J)$ is ill-defined (zero-range division). This does not occur for any $m_{num}\ge3$. The fix is a one-line cap, harmless elsewhere since $\lceil\log_2(m_{num}+1)\rceil< m_{num}-1$ for all $m_{num}\ge4$:

$$
r:=\min\big(\lceil\log_2(m_{num}+1)\rceil,\ m_{num}-1\big),\qquad d=m_{num}-r\ge1\ \ \forall\, m_{num}\ge2
\tag{correction}
$$

### 3.3 Eq. 11 — calibrated cross-attribute voting (resolves threshold dilation)

**Diagnosis.** $NOM(x,y)=\sum_j\mathbb 1[(x,y)\notin nr_{\{c_j\}}]$ and the fixed fractions $D_1=m/3,D_2=m/2,D_3=0.9m$ implicitly assume the $m$ attributes are *vote-exchangeable*: each contributes "outside radius" at a roughly comparable marginal rate, so that "differs on $1/3$ of attributes" is a stable benchmark regardless of composition. Eq. 8$^{\star\star}$ (by design, via Prop. 3–4 below) deliberately breaks this exchangeability for $c_{m+1}$: a tightened radius raises that column's marginal "outside-radius" rate above the average of the original attributes, so its vote is no longer "one vote among $m^\star$ comparable ones" — it systematically pushes many pairs' $NOM$ values by a near-constant offset unrelated to genuine dissimilarity on $C$.

**Reformulation.** Rate-calibrate each attribute's contribution against the group mean before summation:

$$
q_j=\frac{1}{n(n-1)}\sum_{x\neq y}\mathbb 1\big[d_{c_j}(x,y)>\varepsilon_{c_j}\big],\qquad \bar q=\frac1{m^\star}\sum_{k=1}^{m^\star}q_k,\qquad q_j^\dagger=\max(q_j,\tfrac1n)
$$

$$
\hat I_j(x,y)=\frac{\bar q}{q_j^\dagger}\cdot\mathbb 1\big[(x,y)\notin nr_{\{c_j\}}\big],\qquad
NOM^{\star\star}(x,y)=\sum_{j=1}^{m^\star}\hat I_j(x,y)
$$

$$
D_1^{\star\star}=\frac{m^\star}{3},\qquad D_2^{\star\star}=\frac{m^\star}{2},\qquad D_3^{\star\star}=0.9\,m^\star
\tag{11}
$$

(If $q_j=0$ — an attribute that is *never* outside its own radius, i.e. behaviorally constant at that granularity — set $\hat I_j\equiv0$: a vote-less attribute should contribute no dissimilarity evidence, not an undefined one.)

> **Proposition 5 (Conservativity and exchangeability restoration).** If $q_1=\cdots=q_{m^\star}$ (the homogeneous-scale case of Addendum I, all radii drawn from one common $\lambda$), then $NOM^{\star\star}\equiv NOM$ (Eq. 9) exactly. In general, $\mathbb E[\hat I_j]=\bar q$ for every $j$ regardless of $q_j$'s individual value, so $\mathbb E[NOM^{\star\star}]=m^\star\bar q$ is invariant to how heterogeneously the per-attribute radii are chosen — including the deliberate tightening of $c_{m+1}$ licensed by Prop. 3–4. *Proof.* Homogeneity: $\hat I_j=(\bar q/q_j)\mathbb 1[\cdot]=1\cdot\mathbb 1[\cdot]$ when $q_j=\bar q\,\forall j$. Expectation: $\mathbb E[\mathbb 1[(x,y)\notin nr_{\{c_j\}}]]=q_j$ by definition, so $\mathbb E[\hat I_j]=(\bar q/q_j^\dagger)\,q_j=\bar q$. $\blacksquare$

Because $NOM^{\star\star}$ reduces to $NOM$ exactly in the case Addendum I already validated, reusing the unchanged fractional thresholds $D_1,D_2,D_3$ (updated only from $m$ to $m^\star$) remains valid — no new threshold constants are introduced.

### 3.4 Eq. 12 — unsupervised granular dispersion weight (replaces Eq. 10$^\star$)

**Diagnosis (why Eq. 10$^\star$ failed).** $\gamma_{c_j}(X)=(|POS_{nr_{\{c_j\}}}(X)|+|NEG_{nr_{\{c_j\}}}(X)|)/n$ measures agreement between $c_j$'s neighborhood structure and a fixed partition of $X$. When $X$ is an *unpurified* mixture (the standard unsupervised setting: the small contaminating anomalies are members of $X$ itself, not a separate labeled class), the attributes that best isolate that rare subpopulation are exactly the ones that generate *ambiguous, boundary-heavy* neighborhoods relative to the bulk of $X$ — i.e., low $\gamma_{c_j}$. Eq. 10$^\star$ therefore actively downweights the most anomaly-informative attributes, which is the observed collapse.

**Reformulation.** Replace class-containment discriminability with a purely distributional, unsupervised dispersion statistic computed once over the full universe $U$ (no reference to $X$, $POS$, or $NEG$):

$$
\rho_j(x_i)=\frac{\lvert n_{c_j}(x_i)\rvert}{n}\in(0,1],\qquad
CV(c_j)=\frac{\mathrm{std}_i\big(\rho_j(x_i)\big)}{\mathrm{mean}_i\big(\rho_j(x_i)\big)+\tfrac1n}
$$

$$
W(c_j)=\begin{cases}\dfrac{CV(c_j)}{\max_k CV(c_k)}, & \max_k CV(c_k)>0\\[6pt] 1, & \max_k CV(c_k)=0\end{cases}
$$

$$
\omega^{X\star\star}_{nr_{\{c_j\}}}(x)=\Big(\omega^X_{nr_{\{c_j\}}}(x)\Big)^{1/2}\cdot\Big(W(c_j)\Big)^{1/2}
\tag{12}
$$

**Interpretation.** $\rho_j(x_i)$ is local granule occupancy — a byproduct already available from the neighborhood construction, no extra pass. $CV(c_j)$ is high exactly when local density is *heterogeneous* across the population (most points densely surrounded, a minority sparse) — the density-contrast signature of a feature useful for isolating rare instances — and low when occupancy is uniform (either uniformly noisy or uniformly uninformative). Unlike $\gamma_{c_j}(X)$, $CV(c_j)$ never references any partition of the universe, so it cannot be suppressed by the very rarity it is meant to detect.

> **Proposition 6 ($[0,1]$-boundedness).** $\omega^{X\star\star}_{nr_{\{c_j\}}}(x)\in[0,1]$, by the same geometric-mean argument as Proposition 2 (both factors lie in $[0,1]$ by construction of $W$). $\blacksquare$

> **Proposition 7 (Structural immunity to contamination, cf. V2's failure).** $\gamma_{c_j}(X)$ is a functional of the pair $(nr_{\{c_j\}},X)$ and is provably suppressed on attributes whose neighborhood structure disagrees with $X$'s assumed boundary — precisely where a genuine minority pattern induces $BND$-region membership. $W(c_j)$ is a functional of $nr_{\{c_j\}}$ and the *marginal* distribution of $\rho_j$ over $U$ alone; no partition of $U$ enters its definition. Consequently, increasing the separability of a rare subpopulation on $c_j$ can only ever *increase* $\rho_j$'s empirical dispersion (by construction, isolating a subgroup at different local density than the majority is exactly what inflates $CV$), whereas the same change can *decrease* $\gamma_{c_j}(X)$ whenever the subpopulation is embedded inside the (unpurified) target set $X$. The two statistics are not merely differently scaled — they can move in **opposite directions** under the same contamination structure, which is the precise, provable content behind the empirical reversal from V2's collapse to Eq. 12's design goal.

### 3.5 Unified fusion over $C^\star$

$NDF^{X\star}$ (Eq. 4$^\star$) is retained in unchanged form — only its inputs (regions built from $\varepsilon^{\star\star}_{c_j}$, gated via $NOM^{\star\star}$/$D_1^{\star\star},D_2^{\star\star},D_3^{\star\star}$) are upgraded:

$$
NDF^{X\star}_{nr_{\{c_j\}}}(x)=\frac{w_{NEB}\lvert NEB^X(x)\rvert+w_{NPB,j}\lvert NPB^X_{nr_{\{c_j\}}}(x)\rvert+w_{POS,j}\lvert POS^X_{nr_{\{c_j\}}}(x)\rvert}{w_{NEB}\lvert NEB(X)\rvert+w_{NPB,j}\lvert NPB_{nr_{\{c_j\}}}(X)\rvert+w_{POS,j}\lvert POS_{nr_{\{c_j\}}}(X)\rvert}
\tag{13}
$$

$$
MNOF^{X\star\star}(x)=\sum_{j=1}^{m^\star}NDF^{X\star}_{nr_{\{c_j\}}}(x)\cdot\omega^{X\star\star}_{nr_{\{c_j\}}}(x)
\tag{14}
$$

By Corollary 4's argument (both factors in $[0,1]$), $MNOF^{X\star\star}(x)\in[0,1]$ — $\mu$ again requires no recalibration.

---

## 4. Theoretical Properties of the Unified Model

> **Proposition 4 (Positive skew of the subspace residual attribute, under standard SPE regularity).** Let $w_1,\dots,w_d$ be an orthonormal basis of $\mathrm{span}(v_1,\dots,v_r)^\perp$ (post-correction, $d\ge1$), so $J(x)=\sum_{l=1}^d\langle z_x,w_l\rangle^2$. Under the standard working assumption of Q-statistic/SPE distribution theory — that the residual projections $\{\langle z_x,w_l\rangle\}$ are approximately mean-zero and behave as a (possibly weighted) sum of $d$ independent squared Gaussian terms with nonnegative weights $\{a_l\}$ (the eigenvalues of the residual covariance) — the skewness of $J(x)$ is
> $$
> \mathrm{skew}(J)=\frac{8\sum_l a_l^3}{\big(2\sum_l a_l^2\big)^{3/2}}>0\qquad\text{whenever }d\ge1\text{ and some }a_l>0.
> $$
> *Proof sketch.* Each $a_lX_l$, $X_l\sim\chi^2_1$ i.i.d., has third cumulant $8a_l^3\ge0$ and variance $2a_l^2$; cumulants add under independence, giving the stated ratio, strictly positive since not all $a_l=0$ (else the residual subspace would carry zero variance, itself only the boundary case already handled by Eq. 8$^{\star\star}$'s zero-dispersion branch). $\blacksquare$

Since $\mathrm{minmax}(\cdot)$ is a strictly increasing affine map, it preserves the sign of skewness, so $c_{m+1}$ inherits $J$'s positive skew. A right-skewed sample has, in expectation, $HU>HL$ (the classical quantile-based signature of right skew, e.g. Bowley's coefficient $(Q_3+Q_1-2\,\mathrm{med})/IQR$ shares its sign with population skewness for unimodal distributions), so:

> **Corollary 5 (Emergent harmonization — resolves the scale-conflict ask without a manual dual-$\lambda$).** Whenever $c_{m+1}$'s empirical asymmetry is nonzero ($HL\neq HU$), Proposition 3 gives $\varepsilon^{\star\star}_{c_{m+1}}<\varepsilon^{\star}_{c_{m+1}}$ strictly, while for an approximately symmetric original attribute $c_j\in C$ ($HL\approx HU$), $\varepsilon^{\star\star}_{c_j}\approx\varepsilon^\star_{c_j}$ (Prop. 3, equality case). The *same* closed-form formula (Eq. 8$^{\star\star}$) therefore assigns $c_{m+1}$ a materially tighter radius than the bulk of $C$ purely as a function of each attribute's own measured shape — the decoupling requested in the Phase-2 task is an emergent property of one unified formula, not a second free hyperparameter reintroducing the grid-search overhead Corollary 2 eliminated.

> **Conjecture 1 (Joint retention of V4's boundary signal and V3's subspace signal — empirical, not provable in closed form).** *Statement.* The unified model (Eq. 8$^{\star\star}$, 11, 12, 13, 14) is expected to attenuate the dilution observed in the naive V3+V4 hybrid, because (i) Corollary 5 removes the forced single-scale compromise between $\lambda_{base}$ and $\lambda_{subspace}$ that caused the dilution, and (ii) Proposition 5 prevents $c_{m+1}$'s now-tighter radius from distorting the $D_1,D_2,D_3$ gating on which Eq. 4$^\star$'s tiered weights depend. *Epistemic status.* This is a mechanism-level argument for the **direction** of the effect (less destructive interference between the two mechanisms than in the naive hybrid), not a derivation of the **magnitude** — whether the unified model clears the specific reported thresholds (AUC $\ge0.82$, AP $\ge0.30$ on MA3) is a property of the Mammography sample's actual joint distribution and can only be established by re-running the grid-free unified estimator on that data. No closed-form treatise can certify a decimal benchmark outcome in advance; Remark 0 applies here in full.

---

## 5. Complexity Theorem

> **Theorem 3 (Second-generation complexity invariance).** Let $T_{new}''$ denote the total cost of Eq. 8$^{\star\star}$, the Algorithm 0$^\star$ correction, Eq. 11, and Eq. 12, layered on Algorithms 0$^\star$–2$^\star$. Then $T_{new}''=\Theta(mn\log n+m\lvert X\rvert^2)=T_{base}$.

**Proof.** (1) $HL,HU$ reuse the same per-attribute order statistics ($\mathrm{med},Q_1,Q_3$) already sorted for Eq. 8$^\star$ — $O(1)$ extra arithmetic per attribute, no new sort. (2) The $r$-cap is an $O(1)$ comparison. (3) $q_j$ and $\rho_j(x_i)$ are obtainable as running byproducts of the same sorted single-attribute neighbor construction used for $\varepsilon_{c_j}$ (cumulative in-radius counts via one sweep of the sorted axis): $O(n\log n)$ per attribute, not the $O(n^2)$ a naive pairwise loop would suggest. (4) $CV(c_j)$ and $\bar q$ are $O(n)$ reductions per attribute, subsumed. (5) $\hat I_j$ rescales an already-computed indicator by a scalar — no order change to the $O(m^\star\lvert X\rvert^2)$ fusion step. Summing (1)–(5), each $O(mn\log n)$ or $O(n)$-subsumed, and using $m^\star=\Theta(m)$ (Addendum I, Theorem 2, step 4): $T_{new}''=O(mn\log n+m\lvert X\rvert^2)$. The matching lower bound holds because $T_{new}''$ strictly contains $T_{base}$'s operations. $\blacksquare$

---

## 6. Boundary and Edge-Case Analysis (extends Addendum I §4)

- **$HL=0$ or $HU=0$ (heavy one-sided tie concentration).** $IQR_{eff}=0$ by construction; Eq. 8$^{\star\star}$ falls through to the $\hat\sigma_{c_j}$ branch exactly as Eq. 8$^\star$ falls through when $IQR=0$ — a graceful degeneration, not a new failure mode.
- **$m_{num}=2$.** Handled by the $r$-cap (§3.2); without it, $J(x)\equiv0$ and $\mathrm{minmax}$ divides by a zero range. With it, $d=1\ge1$ always.
- **$q_j=0$ for some $j$ (attribute never exceeds its own radius).** $\hat I_j\equiv0$ by definition (§3.3) — such an attribute casts no dissimilarity vote rather than triggering division by zero; the $q_j^\dagger=\max(q_j,1/n)$ floor is a secondary guard for the case $0<q_j<1/n$.
- **$\mathrm{mean}_i(\rho_j)\to0$ (near-categorical numeric attribute under a very tight radius).** The $+1/n$ floor in $CV(c_j)$'s denominator prevents blow-up; if every attribute degenerates this way, $\max_k CV(c_k)=0$ triggers the $W(c_j):=1\,\forall j$ fallback, recovering the unweighted baseline exactly, as Eq. 10$^\star$ did in its own degenerate case.
- **Small $n$.** All new statistics ($HL,HU,q_j,\rho_j$) are built from the same type-7 quantile/order-statistic machinery already validated for $n\ge1$ in Addendum I §4.1; no new small-$n$ instability is introduced.

---

## 7. Scope and Limitations

- Proposition 4's skew argument is conditioned on the standard SPE/Q-statistic regularity assumption (approximate mean-zero, weakly-correlated residual projections). Practitioners should treat this as a testable diagnostic — a near-zero sample skewness or medcouple on $c_{m+1}$ signals that Corollary 5's harmonization will be weak for that dataset, in which case the two-mechanism scale conflict may persist and a residual (smaller) empirical gap versus grid search should be expected.
- $W(c_j)$ targets density-contrast informativeness; like any unsupervised heuristic, it can be misled by attributes with strong multi-modal *clustering* structure unrelated to genuine rarity (e.g., several well-separated but equally common subpopulations would also inflate $CV$). This is the same class of trade-off Addendum I §6 already accepted for the linear-only subspace construction: a deliberate, bounded blind spot rather than an unexamined one.
- Conjecture 1 is, by construction, not closed-form-provable; it should be read as the specific, falsifiable prediction this addendum makes for the next grid-free benchmark pass, not as a result already established by this document.
