# Third-Generation Enhancements to 3WNCROD: Multi-Dimensional Subspace Banks, Distribution-Adaptive Radii, Robust Global Weighting, Calibrated Voting, and Two Novel Mechanisms

**Addendum III to:** *3WNCROD_closed_form_enhancements.md* (Addendum I) and its Second-Generation companion (Addendum II). All prior notation, equations (1–14), lemmas, theorems, propositions, and corollaries are retained. New results are numbered 15 onward.

---

## 0. Epistemic Note for Thesis-Stage Work

The Series-2 figures are again treated as externally supplied empirical input. Three points are worth stating plainly before proceeding, precisely because this is now feeding thesis-level claims:

1. No code has passed through this conversation, so the reported "~89× speedup" and the per-variant AUC/AP numbers cannot be independently checked here — they should be treated as your own experimental record, not as something this document has verified.
2. One figure is worth flagging rather than silently rationalizing: Series-1 V3 reported AP 0.3718 (MA1) / 0.3092 (MA3), while Series-2 V3 reports AP 0.1690 (MA1) / 0.0744 (MA3) — a 2–4× decline. Addendum II's Eq. 8$^{\star\star}$ was designed to *preserve or sharpen* V3's subspace signal (Corollary 5), not weaken it, so this magnitude of drop is larger than the harmonization mechanism alone predicts. Before drawing a theoretical lesson from it, it is worth confirming by ablation exactly which radius/weight/voting configuration was active for "Series-2 V3" (isolated Eq. 8$^\star$ vs. 8$^{\star\star}$; Eq. 9 vs. 11 voting) — the label alone does not disambiguate this.
3. Every claim below is marked **Proposition/Theorem** (derived deductively from the stated construction) or **Conjecture** (a mechanism-level expectation that requires re-benchmarking to confirm magnitude). §9 proposes a concrete validation protocol suited to thesis-level reporting standards (multiple folds, confidence intervals, ablation against re-implemented baselines) — worth applying before any of the Conjectures below are written up as findings.

---

## 1. Notation Addendum III

| Symbol | Meaning |
|---|---|
| $\hat\tau_{jk}$ | Sample Kendall's $\tau$ between numeric attributes $c_j,c_k$ |
| $\widehat{MI}_{jk},\,MI^{lin}_{jk}$ | Gaussian-copula MI estimate; linear (Pearson-based) MI estimate |
| $\Delta_{jk}$ | Nonlinear dependence gap, $\widehat{MI}_{jk}-MI^{lin}_{jk}$ |
| $U_j(x)$ | Continuity-corrected pseudo-observation (empirical copula rank), $(\mathrm{rank}(x;c_j)-0.5)/n$ |
| $\tau_3(c_j),\tau_4(c_j)$ | Sample L-skewness, L-kurtosis (Hosking 1990) |
| $\tau_4^G\approx0.1226$ | L-kurtosis of the Gaussian reference distribution |
| $Gini(c_j)$ | Gini coefficient of granule occupancy $\{\rho_j(x_i)\}$ |
| $S(c_j),W^{(2)}(c_j)$ | Dual-dispersion statistic; rank-normalized global weight (Eq. 17) |
| $w_R^{(2)}$ | Smoothed log-rarity (IDF-style) region weight (Eq. 18) |
| $\varepsilon^{kNN}_{c_j}(x)$ | Per-point $k$-NN radius on attribute $c_j$ |
| $RN_{c_j}(x)$ | Reverse-neighbor count of $x$ on attribute $c_j$ |
| $T,\ \varepsilon^{(t)}_{c_j}$ | Number of granulation levels; the $t$-th dyadic radius scale |

---

## 2. Directive 1 — Multi-Dimensional Subspace Augmentation (Refining V3)

### 2.1 Diagnosis given Series-2 evidence

Algorithm 0$^\star$ captures only deviation from the **dominant linear** correlation manifold (one scalar $J(x)$). Two real limitations follow: (a) a single scalar cannot separate *distinct* multi-attribute interaction patterns — a point anomalous in the $(c_1,c_3)$ relationship and a point anomalous in the $(c_2,c_5)$ relationship are conflated into the same residual axis; (b) purely linear (power-iteration/PCA) decorrelation is blind to **monotonic-but-nonlinear** dependence, e.g. two attributes related by a threshold or saturation effect rather than a line.

### 2.2 Reformulation — MI-Guided Copula-Residual Feature Bank

**Step 1 (rank-based nonlinear-dependence screen, no density estimation required).** For every numeric pair $(c_j,c_k)$, $j<k\le m_{num}$, compute Kendall's $\tau$ (Knight's $O(n\log n)$ algorithm) and convert to Gaussian-copula-equivalent correlation and mutual information:
$$
\hat\rho_{jk}=\sin\!\Big(\tfrac{\pi}{2}\hat\tau_{jk}\Big),\qquad \widehat{MI}_{jk}=-\tfrac12\ln\big(1-\hat\rho_{jk}^2\big),\qquad MI^{lin}_{jk}=-\tfrac12\ln\big(1-\hat r_{jk}^2\big)
\tag{15a}
$$
where $\hat r_{jk}$ is ordinary Pearson correlation. Define the **nonlinear dependence gap** $\Delta_{jk}=\widehat{MI}_{jk}-MI^{lin}_{jk}$: large positive $\Delta_{jk}$ flags pairs whose monotonic association is not already summarized by the linear directions $v_1,\dots,v_r$ of Algorithm 0$^\star$.

**Step 2 (top-$K$ pair selection, closed-form budget).** Given a user-set expansion target $\alpha\in[0.5,1]$ (the "50–100%" budget), set $K=\lceil\alpha m\rceil-1$ (one slot reserved for the existing linear residual $J(x)$), capped at $\binom{m_{num}}{2}$, and take the $K$ pairs with largest $\Delta_{jk}$.

**Step 3 (copula deviation feature per selected pair).** With pseudo-observations $U_j(x)=\frac{\mathrm{rank}(x;c_j)-0.5}{n}$ and a shrunk correlation $\tilde\rho_{jk}=\mathrm{sign}(\hat\rho_{jk})\min(|\hat\rho_{jk}|,1-\tfrac1n)$:
$$
c(u,v;\rho)=\frac{1}{\sqrt{1-\rho^2}}\exp\!\left(-\frac{\rho^2\big(\Phi^{-1}(u)^2+\Phi^{-1}(v)^2\big)-2\rho\,\Phi^{-1}(u)\Phi^{-1}(v)}{2(1-\rho^2)}\right)
$$
$$
c^{(j,k)}_{new}(x)=\mathrm{minmax}\Big(-\ln c\big(U_j(x),U_k(x);\tilde\rho_{jk}\big)\Big)\in[0,1]
\tag{15b}
$$
— the negative log-density under the pair's fitted Gaussian copula, i.e. a closed-form pairwise "joint-surprise" score, in the same spirit as copula-based outlier scoring (cf. COPOD, Li et al. 2020).

**Step 4 (augmentation).** $C^{\star\star}=C\cup\{c_{m+1}=\mathrm{minmax}(J)\}\cup\{c^{(j,k)}_{new}\}_{(j,k)\in\text{Top-}K}$, $m^{\star\star}=m+1+K\approx(1+\alpha)m$.

### 2.3 Analytical Justification

> **Proposition 9 (Complementarity of the two residual families).** $J(x)$ is a function only of the projection of $z_x$ orthogonal to $\mathrm{span}(v_1,\dots,v_r)$ — i.e., it is invariant to any reparametrization *within* that dominant linear subspace. $c^{(j,k)}_{new}(x)$ is a function only of the *rank pair* $(U_j(x),U_k(x))$ and is, by construction, invariant to any strictly monotone marginal transform of $c_j$ or $c_k$ individually — it is sensitive only to violations of the pair's *joint rank structure*, not to marginal outlyingness (which $NDF_{nr_{\{c_j\}}}$ and $NDF_{nr_{\{c_k\}}}$ already cover). The two families are therefore not measuring overlapping information by construction: one is a linear-subspace residual, the other a pairwise rank-dependency residual. *This is a structural non-redundancy argument, not a claim about their relative discriminative power on any specific dataset.*

> **Conjecture 2.** Because $\Delta_{jk}$ specifically screens for dependence *not already captured by the linear directions*, the copula-residual bank is expected to recover AP on subspace-anomaly-bearing subsets (such as MA3, if its features carry threshold/saturation-type dependencies) closer to the Series-1 single-feature peak than a naive re-run of the same single linear feature — but whether it does, and by how much, is an empirical question given Note 2 in §0.

### 2.4 Edge Cases, Scale Constraints, Complexity

- $m_{num}<2$: no pairs exist; $K:=0$, reduces exactly to Addendum I's §2.2 degenerate case.
- $|\hat\rho_{jk}|\to1$: the shrinkage $\tilde\rho_{jk}=\mathrm{sign}(\cdot)\min(|\hat\rho_{jk}|,1-1/n)$ prevents the $1-\rho^2\to0$ singularity in the copula density.
- Ties in $c_j$: handled by average-rank pseudo-observations; the $-0.5$ continuity correction keeps $U_j(x)\in(0,1)$ strictly, so $\Phi^{-1}(U_j(x))$ is always finite.
- **Complexity (relaxed budget acknowledged):** the pairwise screen is $O(m_{num}^2\,n\log n)$ (Knight's algorithm per pair) — quadratic in $m_{num}$, a genuine increase over Theorem 2/3's linear-in-$m$ baseline, justified only because computational constraints were explicitly relaxed for this directive. For large $m_{num}$, restrict the screen to a random or locality-sensitive-hashing-prefiltered candidate set of $O(m_{num}\log m_{num})$ pairs to recover near-linear scaling; this is a scalability note, not a correctness requirement. Feature construction for the selected $K$ pairs is $O(Kn\log n)$, subsumed.

---

## 3. Directive 2 — Distribution-Agnostic Radius (Refining V1)

### 3.1 Diagnosis

Eq. 8$^{\star\star}$ corrects for *asymmetry* (skew) but still assumes a fixed reference constant ($0.9$) derived under Gaussian-kernel AMISE theory. That constant is itself only a robustness-motivated approximation (Silverman's own original constant, derived exactly under Gaussian-reference assumptions, is $\approx1.06$; $0.9$ already hedges downward). For attributes whose *peakedness* (kurtosis), not just skew, departs from Gaussian, the AMISE-optimal constant genuinely shifts: $h^\star\propto R(f'')^{-1/5}$ where $R(f'')=\int f''(u)^2du$ grows with the sharpness of the density's central peak — a fact standard in kernel-smoothing theory (Wand & Jones, *Kernel Smoothing*, 1995) and realized concretely in the generalized-Gaussian (exponential-power) family, whose excess kurtosis $\frac{\Gamma(1/\beta)\Gamma(5/\beta)}{\Gamma(3/\beta)^2}-3$ is a known, strictly monotone function of the shape parameter $\beta$ (Gaussian at $\beta=2$; heavier-tailed/more-peaked for $\beta<2$).

### 3.2 Reformulation — L-Moment Kurtosis-Adjusted Radius

Use **L-moments** (Hosking 1990) rather than classical kurtosis: they are bounded, breakdown-robust, and computable from the same order statistics already sorted for Eq. 8$^\star$/8$^{\star\star}$ (no new sort).

$$
\Delta\tau_4(c_j)=\hat\tau_4(c_j)-\tau_4^G,\qquad \tau_4^G\approx0.1226\ \text{(Gaussian reference)}
$$
$$
\kappa(c_j)=0.9\cdot\Big(1+\max\big(\Delta\tau_4(c_j),\,-0.9\big)\Big)^{-1/5}
$$
$$
\varepsilon_{c_j}^{(3)}=
\begin{cases}
0, & c_j\text{ categorical, or }n<4\text{ (fall back to }\kappa:=0.9,\text{ i.e. Eq. 8}^{\star\star}\text{)}\\[4pt]
\max\!\Big(\kappa(c_j)\cdot\min\big(\hat\sigma_{c_j},\,IQR_{eff}(c_j)/1.34\big)\cdot n^{-1/5},\ \dfrac1n\Big), & \text{numeric, dispersion}>0\\[6pt]
0, & \text{numeric, all dispersion}=0
\end{cases}
\tag{16}
$$

> **Proposition 10 (Nested conservativity).** $\varepsilon^{(3)}_{c_j}=\varepsilon^{\star\star}_{c_j}$ exactly whenever $\hat\tau_4(c_j)=\tau_4^G$ (Gaussian-like peakedness), since $\kappa(c_j)=0.9$ in that case. Combined with Proposition 3, this gives a strict refinement chain $\varepsilon^\star\supseteq\varepsilon^{\star\star}\supseteq\varepsilon^{(3)}$ (each equal to its predecessor under a progressively weaker symmetry/normality condition) — every earlier variant is recovered exactly as a special case, not overwritten.

> **Conjecture 3.** Since leptokurtic tabular attributes are common in real measurement data (heavy central peak plus a moderate tail — exactly what inflates $\hat\sigma_{c_j}$ relative to the core's true resolvable scale), $\varepsilon^{(3)}$ is expected to track the grid-searched optimum more closely than $\varepsilon^{\star\star}$ on such attributes. Whether this closes enough of the AP gap to match or exceed the grid-searched V0 baseline on MA1–MA3 specifically is not something the AMISE argument can certify in closed form (Remark 0, §0/Addendum II) — it is the falsifiable content of this directive, to be checked empirically.

### 3.3 Edge Cases, Scale Constraints, Complexity

- $n<4$: sample L-kurtosis is not meaningfully defined from order statistics; fall back to $\kappa:=0.9$ (exactly Eq. 8$^{\star\star}$).
- $\Delta\tau_4(c_j)<-1$: floored at $-0.9$, keeping the base of the power strictly positive ($\ge0.1$) — a numerical guard, not a claim about real kurtosis values (L-kurtosis is theoretically bounded in $[-1,1]$ for genuine distributions, so this floor is a defensive margin against small-sample estimation noise pushing $\hat\tau_4$ outside its population range).
- Categorical / zero-dispersion attributes: unaffected, exactly as in Eq. 8$^{\star\star}$.
- **Complexity:** L-moments are computable in $O(n)$ given already-sorted values (linear combinations of order statistics), so $\varepsilon^{(3)}_{c_j}$ costs the same $O(n\log n)$ per attribute as its predecessors — no increase in complexity class, no relaxation needed here.

---

## 4. Directive 3 — Maximizing Unsupervised Global Weighting (Refining V2)

### 4.1 Diagnosis

Eq. 12's $CV(c_j)=\mathrm{std}_i(\rho_j)/(\mathrm{mean}_i(\rho_j)+1/n)$ is a single dispersion signal normalized by $\max_k CV(c_k)$. Two specific fragilities follow, both fixable in closed form: (a) $CV$ is a ratio to a possibly-small mean and can be dominated by a handful of near-empty granules, giving it high sampling variance; (b) max-normalization means **one** attribute with an extreme $CV$ compresses every other attribute's weight toward 0, regardless of their own merit.

### 4.2 Reformulation — Dual-Dispersion, Rank-Normalized Weight

Add a second, complementary unsupervised heterogeneity signal — the **Gini coefficient** of granule occupancy, computed from the same sorted $\{\rho_j(x_i)\}$ (ascending, $\rho_{(1)}\le\dots\le\rho_{(n)}$) at no extra sort cost:
$$
Gini(c_j)=\frac{2\sum_{i=1}^n i\,\rho_{(i)}}{n\sum_{i=1}^n\rho_{(i)}}-\frac{n+1}{n}\in[0,1)
$$
$$
S(c_j)=\sqrt{CV(c_j)\cdot Gini(c_j)}
\qquad\qquad
W^{(2)}(c_j)=\frac{\mathrm{rank}\big(S(c_j)\big)}{m^\star}\in\Big(0,1\Big]
\tag{17}
$$
($\mathrm{rank}(\cdot)$ ascending, average rank on ties, over $j=1,\dots,m^\star$), replacing $W(c_j)$ in Eq. 12's blend.

> **Proposition 11 (Non-domination).** Under max-normalization (Eq. 12), a single attribute $c_{j_0}$ with $CV(c_{j_0})\to\infty$ forces $W(c_j)\to0$ for **every** $j\ne j_0$ simultaneously. Under rank-normalization (Eq. 17), the same event changes only $c_{j_0}$'s own rank position (it simply becomes rank $m^\star$); every other attribute's rank, and hence its weight, is unchanged. *Proof.* Immediate from the definition of rank statistics: $\mathrm{rank}(S(c_j))$ depends on $S(c_j)$ only through its ordinal position among $\{S(c_k)\}$, which is invariant to how large the maximum value is, only to its relative order. $\blacksquare$ This is a structural robustness guarantee, not a performance guarantee.

**Optional second-pass refinement (contamination-robust weighting).** Compute a preliminary $MNOF^{(1)}$ using $W^{(2)}$; recompute $\rho_j(x_i)$, $CV$, $Gini$, $W^{(2)}$ restricted to $U\setminus\hat O$, where $\hat O=\{x:MNOF^{(1)}(x)>\mathrm{med}(MNOF^{(1)})+3\cdot\mathrm{MAD}(MNOF^{(1)})\}$ (the same conventional "3-MAD" constant already used elsewhere in this framework's spirit, not a newly tuned knob). This is a one-step reweighted (IRLS-style) robustification: attributes' dispersion statistics are no longer computed on a sample potentially contaminated by the very points being scored.

### 4.3 Edge Cases, Scale Constraints, Complexity

- $\rho_j(x_i)\ge1/n>0$ always (reflexivity: every point lies in its own neighborhood), so $Gini(c_j)$'s denominator never vanishes.
- All $S(c_j)$ tied (fully homogeneous dataset): every attribute receives the same mid-rank weight $W^{(2)}=\frac{(m^\star+1)/2}{m^\star}\approx\tfrac12$ — a neutral, non-degenerate fallback distinct from (and arguably more honest than) forcing $W:=1$.
- Second-pass refinement adds one full extra pass at the same per-attribute cost: a constant-factor (×2), not asymptotic, increase.
- **Complexity:** $Gini(c_j)$ from sorted $\rho_j$ is $O(n)$ given the sort already performed for $CV$/quantile machinery; ranking $m^\star$ values of $S(c_j)$ is $O(m^\star\log m^\star)$, dominated by the $O(mn\log n)$ per-attribute term. No change in complexity class.

---

## 5. Directive 4 — Stabilizing Tiered Evidence (Refining V4)

### 5.1 Diagnosis

Two independent sources of "voting miscalibration" exist in V4 as isolated: (a) $w_R=1-|R|/|X|$ is a *linear* rarity discount, which treats a change from $|R|/|X|=0.01\to0.02$ as equally significant as $0.49\to0.50$ — a poor match to the "rarer regions carry disproportionately more evidentiary weight" principle the mechanism is meant to encode; (b) $D_1,D_2,D_3$ (fixed fractions of $m^\star$) assume the empirical distribution of $NOM^{\star\star}(x,y)$ spreads roughly uniformly over $[0,m^\star]$, which need not hold — if most pairs cluster in a narrow band of $NOM^{\star\star}$, fixed fractional thresholds may rarely gate any tier at all (or gate one tier almost always), i.e. exactly a metric/voting-calibration mismatch.

### 5.2 Reformulation

**(a) Log-smoothed (IDF-style) rarity weight**, with a Jeffreys $+\tfrac12$ continuity correction to avoid $\log 0$:
$$
w_R^{(2)}=\frac{-\log_2\!\Big(\dfrac{|R|+\tfrac12}{|X|+1}\Big)}{-\log_2\!\Big(\dfrac{\tfrac12}{|X|+1}\Big)}
\tag{18a}
$$

> **Proposition 12.** $w_R^{(2)}=1$ exactly at $|R|=0$ (numerator equals denominator), and $w_R^{(2)}\to0$ as $|X|\to\infty$ with $|R|=|X|$; throughout, $w_R^{(2)}$ is strictly decreasing and convex in $|R|/|X|$, giving materially larger evidentiary contrast between "very rare" and "somewhat rare" regions than the linear $w_R$, directly targeting the tiered-evidence philosophy the paper's own qualitative discussion (preceding Eq. 3) already states but Eq. 4 under-weights.

**(b) Empirical-quantile (data-adaptive) voting thresholds**, replacing fixed fractions of $m^\star$ with quantiles of the *observed* calibrated-vote distribution:
$$
D_1^{(3)}=\widehat Q_{1/3}\big(NOM^{\star\star}(x,y)\big),\quad D_2^{(3)}=\widehat Q_{1/2}(\cdot),\quad D_3^{(3)}=\widehat Q_{0.9}(\cdot)
\tag{18b}
$$
estimated from a random subsample of $O(n\log n)$ pairs (sufficient for quantile stability at fixed precision by the Dvoretzky–Kiefer–Wolfowitz inequality), not the full $O(n^2)$ pair set.

> **Conjecture 4.** Because Eq. 18b guarantees a consistent *proportion* of pairs fall in each evidentiary tier regardless of how concentrated $NOM^{\star\star}$'s empirical distribution happens to be, the combination of 18a+18b is expected to reduce the sensitivity of V4's output to dataset-specific vote-count concentration observed informally in the Series-2 "requires further refinement" note — the size of that improvement remains to be measured.

### 5.3 Edge Cases, Scale Constraints, Complexity

- $|X|$ small: Eq. 18a is well-defined for any $|X|\ge0$; at $|X|=0$ the ratio is $0/0$ and should short-circuit to $w_R^{(2)}:=1$ (no reference set, maximal uncertainty treated as maximal weight, matching the convention already used for other degenerate fallbacks in this framework).
- Quantile subsample too small (e.g. tiny $n$): fall back to the full pair set (the $O(n^2)$ cost is immaterial for small $n$ in any case).
- **Complexity:** subsampled quantile estimation is $O(n\log n)$ (sort the subsample), preserving the established complexity class; using the *full* pair set instead would cost $O(n^2)$ per quantile call and should be avoided at scale.

---

## 6. Novel Variant V5 — Asymmetric Reverse-Neighbor Isolation

**Motivation.** Every mechanism so far uses one *shared* radius per attribute, which makes the neighborhood relation symmetric ($x\in n_{c_j}(y)\iff y\in n_{c_j}(x)$) and therefore blind to a real, distinct anomaly signature: points that others rarely "reach back" to claim as a neighbor, even though the point's own forward neighborhood looks unremarkable — the phenomenon exploited by reverse-nearest-neighbor outlier scoring (ODIN; Hautamäki, Kärkkäinen & Fränti, 2004) and related to the "hubness" phenomenon in nearest-neighbor graphs (Radovanović, Nanopoulos & Ivanović, *Hubs in Space*, JMLR 2010).

**Construction.** Introduce a *per-point* adaptive radius (only for this variant — the region-layer machinery elsewhere is untouched):
$$
\varepsilon^{kNN}_{c_j}(x)=d_{c_j}\big(x,\,\mathrm{kNN}_k(x)\big),\qquad k=\max\big(1,\lceil n^{2/5}\rceil\big)
$$
(the exponent matches the $n^{-1/5}$-type consistency order already used throughout this framework; $k$ is a deterministic function of $n$, not a tuned constant.)
$$
RN_{c_j}(x)=\big|\{y\in U\setminus\{x\}:d_{c_j}(x,y)\le\varepsilon^{kNN}_{c_j}(y)\}\big|,\qquad AIF_{c_j}(x)=\max\!\Big(0,\ 1-\frac{RN_{c_j}(x)}{k}\Big)
$$
$$
MNOF^{V5}(x)=\tfrac12\Big(MNOF^{X}(x)+\mathrm{minmax}\big(\tfrac1{m^\star}\textstyle\sum_j AIF_{c_j}(x)\big)\Big)
\tag{19}
$$

### Analytical Justification

> **Proposition 13 (Expected reverse count under local homogeneity).** In a $k$-NN graph built from an (approximately) exchangeable population, $\mathbb E[RN_{c_j}(x)]=k$ for every $x$ — a standard fact of $k$-NN graph theory (the in-degree of a $k$-NN digraph averages to $k$ by a symmetry/double-counting argument over the population). $AIF_{c_j}(x)$ is therefore, in expectation, $0$ under homogeneity and strictly positive only when $x$ is reached back by fewer than $k$ others despite claiming $k$ forward neighbors — precisely the asymmetric-isolation signature this variant targets, and a signal genuinely absent from every symmetric-radius mechanism in Addenda I–II.

> **Conjecture 5.** $AIF$ is expected to be complementary to $NDF$-based scores (which are driven by *forward*-neighborhood boundary/exceptional-region membership under a population-wide fixed radius) rather than redundant with them, since the two are provably sensitive to different graph-theoretic quantities (forward-neighborhood membership vs. reverse-degree). Whether the additive fusion in Eq. 19 improves AUC/AP beyond $MNOF^X$ alone on any given dataset is empirical.

### Edge Cases, Scale Constraints, Complexity

- $k$ floored at $1$; for $n\le2$, $k=1$ trivially (the only other point).
- Ties at the $k$-th nearest distance: include all tied points (a standard $k$-NN tie-handling convention), so $RN$ may occasionally exceed nominal $k$ for the point(s) supplying the tie — harmless, since $AIF$ is floored at $0$ regardless.
- **Complexity:** in one dimension, $k$-NN radius and reverse counts are both obtainable from the sorted attribute values via a sliding window / Fenwick-tree sweep in $O(n\log n)$ per attribute — this variant does **not** require the relaxed budget; it stays within $O(mn\log n+m|X|^2)$.

---

## 7. Novel Variant V6 — Multigranulation Consensus Ensemble

**Motivation.** Every closed-form radius proposed so far (Eq. 8$^\star$, 8$^{\star\star}$, 16) is a single best-effort estimate of "the" right granularity. Rough set theory already has an established framework for combining *several* granularities rather than committing to one: **multigranulation rough sets** (MGRS; Qian, Liang & Dang, and related work from that line, c. 2006–2010), with *optimistic* (union/OR-like agreement) and *pessimistic* (intersection/AND-like agreement) fusion operators. Re-casting "which $\lambda$" as a granulation-ensemble question sidesteps the single-scale optimization problem entirely.

**Construction.** Fix $T$ dyadic octave scales around the closed-form base radius (a structural choice, not a tuned constant — analogous to using octaves in wavelet multiresolution analysis):
$$
\varepsilon^{(t)}_{c_j}=2^{\,t-\lceil T/2\rceil}\cdot\varepsilon^{(3)}_{c_j},\qquad t=1,\dots,T\quad(\text{e.g. }T=3:\ \{\tfrac12,\,1,\,2\}\times\varepsilon^{(3)}_{c_j})
$$
$$
NDF^{X,pes}_{nr_{\{c_j\}}}(x)=\min_{t=1,\dots,T}NDF^{X\star}_{nr_{\{c_j\}},\varepsilon^{(t)}_{c_j}}(x),\qquad
NDF^{X,opt}_{nr_{\{c_j\}}}(x)=\max_{t=1,\dots,T}NDF^{X\star}_{nr_{\{c_j\}},\varepsilon^{(t)}_{c_j}}(x)
\tag{20}
$$
$MNOF^{V6,pes/opt}(x)=\sum_j NDF^{X,pes/opt}_{nr_{\{c_j\}}}(x)\cdot\omega^{X\star\star}_{nr_{\{c_j\}}}(x)$.

### Analytical Justification

> **Proposition 14 (Bracketing).** For every $x$: $NDF^{X,pes}_{nr_{\{c_j\}}}(x)\le NDF^{X\star}_{nr_{\{c_j\}},\varepsilon^{(3)}_{c_j}}(x)\le NDF^{X,opt}_{nr_{\{c_j\}}}(x)$, since the single-scale value is one of the $T$ terms being min'd/max'd over. Consequently $MNOF^{V6,pes}\le MNOF^{X\star\star}\le MNOF^{V6,opt}$ pointwise — the ensemble strictly brackets the single-scale estimate, giving a closed-form precision/recall dial (pessimistic favors precision, optimistic favors recall) with **no grid search**, since $T$ and the octave ladder are fixed structural choices, not tuned per dataset. This precision/recall trade-off direction is an established, provable property of pessimistic vs. optimistic MGRS fusion, not a claim specific to this application.

> **Conjecture 6.** Because the ensemble already brackets whichever single scale would have been grid-search-optimal (as long as that optimum falls within the $[\tfrac12,2]\times\varepsilon^{(3)}$ octave window), $MNOF^{V6,opt}$ is expected to recover AP closer to the grid-searched V0 ceiling than any single closed-form radius variant (V1 or its refinements) can alone — subject to the optimum actually lying inside that window, which is a property of the specific dataset and must be checked, not assumed.

### Edge Cases, Scale Constraints, Complexity

- $T$ is a small fixed constant (e.g. $3$ or $5$); sensitivity to its exact value is expected to be low precisely because it is a coverage window, not a point estimate — but this should be confirmed, not assumed, if reported in a thesis.
- Pessimistic fusion can drive $NDF^{X,pes}\to0$ whenever *any* scale fails to register a signal — a known, inherent precision/recall trade-off of pessimistic MGRS (not a defect introduced here).
- **Complexity:** a constant factor $T\times$ the base per-attribute cost; asymptotic class unchanged, $O(T\cdot mn\log n+T\cdot m|X|^2)=O(mn\log n+m|X|^2)$ for fixed $T$.

---

## 8. Consolidated Complexity Table

| Mechanism | Extra cost vs. baseline | Class-preserving? |
|---|---|---|
| Eq. 16 (L-moment radius) | $O(n)$/attribute (reuses sort) | Yes |
| Eq. 17 (dual-dispersion weight) | $O(n)$/attribute + $O(m^\star\log m^\star)$ | Yes |
| Eq. 18a/18b (log rarity + adaptive $D$) | $O(1)$ + $O(n\log n)$ subsample | Yes |
| V5 (reverse-neighbor) | $O(n\log n)$/attribute | Yes |
| V6 (multigranulation) | $\times T$ constant factor | Yes (fixed $T$) |
| Directive 1 (MI/copula bank) | $O(m_{num}^2 n\log n)$ screen | **No** — quadratic in $m_{num}$, relaxed budget only |

---

## 9. Scope, Limitations, and Recommended Validation Protocol

- Proposition 9's non-redundancy and Proposition 13's expected-reverse-count results are structural (hold by construction); Conjectures 2–6 are mechanism-level expectations, not proofs of magnitude — consistent with Remark 0 (Addendum II) and §0 above.
- **For thesis-grade reporting**, before writing any of Conjectures 2–6 up as findings: (i) re-run each variant across multiple random seeds / cross-validation folds on MA1–MA3 and report mean ± confidence interval, not single point estimates, since AUC/AP on datasets of this size carry non-trivial sampling variance; (ii) re-implement (or unit-test against) the V0 grid-searched baseline independently to confirm the reference numbers being improved upon are reproducible in your own harness; (iii) run the ablation flagged in §0, item 2, to resolve which exact Series-2 "V3" configuration was actually measured before attributing its AP decline to any specific mechanism; (iv) report which configuration (isolated vs. combined variants) each number corresponds to explicitly, since several mechanisms here (Eq. 16, Eq. 17, V6) are designed to compose, and composed results should not be presented under a single variant's label without saying so.
- Directive 1's quadratic screening cost is a genuine scalability limitation for high-dimensional tabular data; the random/LSH-prefiltering mitigation in §2.4 is a sketch, not a fully specified algorithm, and would need its own approximation-quality analysis if pursued further.
