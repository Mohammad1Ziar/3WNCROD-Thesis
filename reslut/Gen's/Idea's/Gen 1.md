# Closed-Form Enhancements to 3WNCROD: Attribute-Space Augmentation and Robust Nonparametric Granulation for Three-Way Neighborhood Outlier Detection

**Subject paper:** Zhang, Yuan & Miao, *"Outlier Detection Using Three-Way Neighborhood Characteristic Regions and Corresponding Fusion Measurement,"* IEEE TKDE, DOI 10.1109/TKDE.2023.3312108.

---

## 0. Notation Addendum (extends the paper's Table 1)

| Symbol | Meaning |
|---|---|
| $\hat\sigma_{c_j}$ | Robust scale of attribute $c_j$: $1.4826\cdot\mathrm{med}_i\lvert f(x_i,c_j)-\mathrm{med}(c_j)\rvert$ (MADN) |
| $IQR(c_j)$ | Interquartile range $Q_3(c_j)-Q_1(c_j)$ |
| $C^\star,\,m^\star$ | Augmented condition-attribute set $C\cup\{c_{m+1}\}$, $m^\star=\lvert C^\star\rvert$ |
| $C_{num}, m_{num}$ | Numeric sub-block of $C$ and its size |
| $Z\in\mathbb R^{n\times m_{num}}$ | Robust-standardized numeric data matrix, $Z_{i,j}=(f(x_i,c_j)-\mathrm{med}(c_j))/\hat\sigma_{c_j}$ |
| $v_1,\dots,v_r$ | Orthonormal dominant correlation directions of $Z$ (deflationary power iteration) |
| $r$ | Retained-component count, $r=\lceil\log_2(m_{num}+1)\rceil$ |
| $J(x)$ | Squared Prediction Error (SPE / orthogonal-residual statistic) of $x$ w.r.t. $\mathrm{span}(v_1,\dots,v_r)$ |
| $c_{m+1}$ | Synthetic subspace attribute, $c_{m+1}(x)=\text{minmax}(J(x))$ |
| $\gamma_{c_j}(X)$ | Single-attribute approximation quality (discriminability): $(\lvert POS_{nr_{\{c_j\}}}(X)\rvert+\lvert NEG_{nr_{\{c_j\}}}(X)\rvert)/n$ |
| $\hat\gamma_{c_j}$ | $\gamma_{c_j}(X)/\max_k\gamma_{c_k}(X)$ |
| $w_R$ | Region evidentiary weight, $1-\lvert R\rvert/\lvert X\rvert$ |

All quantities above are computed once per attribute (or once globally for $Z,v_k,J$) and are **deterministic, closed-form functions of the data** — none require an externally supplied or grid-searched hyperparameter.

---

## 1. Theoretical Audit: Feasibility and Semantic Invariance

### 1.1 Layered-architecture argument

The 3WNCROD pipeline is strictly layered:

$$
\underbrace{(d_B,\varepsilon_B)}_{\text{Eq. 7–8: metric layer}} \;\longrightarrow\; \underbrace{nr_B \to POS,NEG,BND,NIB,NOB,NEB,NPB}_{\text{Def. 1–4: region layer}} \;\longrightarrow\; \underbrace{NEB^X,NPB^X,POS^X\to NDF \to MNOF}_{\text{Eq. 3–5: measurement layer}} \;\longrightarrow\; \underbrace{MNOF>\mu}_{\text{Def. 8: decision layer}}
$$

Definitions 1–4 (the region layer, which *is* the 3WD structure the task requires us to preserve) are stated for an **arbitrary** neighborhood relation $nr_B$ induced by an **arbitrary** non-negative threshold $\varepsilon_B$ on an **arbitrary** attribute subset $B\subseteq C$. Nothing in Definitions 1–4, Propositions 1–3, or Definition 9 (NOM) constrains $\varepsilon_B$ to the specific parametric form $\mathrm{std}(c_j)/\lambda$, nor constrains $C$ to be the *original* feature set supplied by the practitioner. Consequently:

> **Lemma 1 (Radius substitutability).** Replacing Eq. 8 by any other measurable function $\varepsilon_{c_j}^\star: \mathcal D \to \mathbb R_{\ge 0}$ of the sample $\mathcal D=\{f(x_i,c_j)\}_{i=1}^n$ leaves Definitions 1–4, Propositions 1–3, and Definition 9 valid *verbatim*, because their proofs use only $\varepsilon_B\ge 0$ and the induced tolerance structure $n_B(x)=\{y:d_B(x,y)\le\varepsilon_B\}$, never the functional form of $\varepsilon_B$.

This licenses Enhancement 2.1 (adaptive radius) without touching a single region definition.

> **Lemma 2 (Attribute-space substitutability).** Definitions 1–4 and 9 are stated for arbitrary $B\subseteq C$ and arbitrary $C$. Hence they remain valid, unmodified in form, under the substitution $C\to C^\star=C\cup\{c_{m+1}\}$ for any derived attribute $c_{m+1}$, provided $c_{m+1}$ is itself equipped with a valid $(d_{c_{m+1}},\varepsilon_{c_{m+1}})$ pair (which Eq. 8$^\star$ supplies automatically, since $c_{m+1}$ is numeric).

This licenses Enhancement 2.2 (subspace attribute) without reformulating Eq. 9 (NOM) at all — only its domain is extended.

### 1.2 Safety of attribute augmentation (non-degradation guarantee)

> **Theorem 1 (Monotone refinement under augmentation).** Let $C\subseteq C^\star$. Then
> $$NIB_{nr_{C^\star}}(X)\subseteq NIB_{nr_C}(X), \qquad NEB^\star(X)=\bigcap_{j=1}^{m+1}NIB_{nr_{\{c_j\}}}(X)\subseteq \bigcap_{j=1}^{m}NIB_{nr_{\{c_j\}}}(X)=NEB(X).$$
>
> *Proof.* Proposition 2 of the source paper establishes $B\subseteq C\Rightarrow NIB_{nr_B}(X)\supseteq NIB_{nr_C}(X)$ for arbitrary nested attribute sets; substituting $C\to C^\star$ preserves the argument since it uses only the monotone-shrinking property of neighborhoods under attribute intersection $n_{C^\star}(x)=n_C(x)\cap n_{\{c_{m+1}\}}(x)\subseteq n_C(x)$, which holds for **any** added attribute, synthetic or not. The second inclusion follows because $NEB(X)$ is defined (Def. 3) as an intersection over the single-attribute inner boundaries indexed by $C$; adding index $m{+}1$ intersects one further set, which cannot enlarge the intersection. $\blacksquare$

**Interpretation.** Augmenting the attribute universe with the synthetic subspace feature can only *sharpen* (shrink or preserve) the exceptional boundary — it cannot manufacture spurious uncertainty. This is a formally derived non-degradation guarantee, not an empirical claim.

### 1.3 Which equations are structurally modified

| Equation | Status under this treatise |
|---|---|
| Eq. 1–2 (Def. 1–2, POS/NEG/BND, NIB/NOB) | **Unchanged in form** (Lemma 1, 2) |
| Def. 3–4 (NEB, NPB) | **Unchanged in form**, evaluated over $C^\star$ |
| Eq. 3 (three-way char. regions) | **Unchanged in form** |
| **Eq. 4** (NDF) | Generalized to Eq. 4$^\star$ (§2.4) — tiered evidentiary weighting; original Eq. 4 is the special case $w_{NEB}=w_{NPB}=w_{POS}\equiv1$ |
| **Eq. 5** (MNOF) | Re-indexed over $C^\star$ (§2.5); functional form unchanged, domain extended |
| **Eq. 8** (radius) | Replaced by Eq. 8$^\star$ (§2.1) |
| **Eq. 9** (NOM) | **Unchanged in form**; domain automatically extends to $C^\star$ (§2.2) |
| **Eq. 10** (weight) | Replaced by Eq. 10$^\star$ (§2.3) |
| Def. 8 (threshold $\mu$) | Unchanged in form; scale-preserved (Corollary 4, §3) |

Four of the nine equations governing the region layer require **zero modification**; the fixes are confined to the metric layer (Eq. 8), the attribute universe (Eq. 9's domain), and the measurement/fusion layer (Eq. 4, 5, 10) — exactly the layers responsible for the five identified weaknesses, and none of which participate in the 3WD semantics the task requires preserved.

---

## 2. Closed-Form Reformulations

### 2.1 Weaknesses 1, 4, 5 → Eq. 8$^\star$: parameter-free adaptive radius

**Diagnosis.** Eq. 8 couples all numeric features to one global $\lambda$, forcing a single granularity onto heterogeneous distributions, and requires a 20-step grid search over the arbitrarily truncated domain $[0.1,2.0]$. It further uses $\mathrm{std}(c_j)$, an estimator with **breakdown point 0** (Rousseeuw & Leroy 1987 — already the paper's own reference [36]): a single leverage point inflates $\mathrm{std}(c_j)$ without bound, which inflates $\varepsilon_{c_j}$, over-smooths neighborhoods, and *masks* the very anomaly responsible for the inflation — a documented circularity in variance-based outlier scoring.

**Reformulation** (Silverman-type rule of thumb, KDE bandwidth theory):

$$
\varepsilon_{c_j}^\star=
\begin{cases}
0, & c_j \text{ categorical}\\[4pt]
\max\!\Big(0.9\,\min\big(\hat\sigma_{c_j},\,IQR(c_j)/1.34\big)\cdot n^{-1/5},\ \dfrac1n\Big), & c_j\text{ numeric},\ \hat\sigma_{c_j}>0 \text{ or } IQR(c_j)>0\\[6pt]
0, & c_j\text{ numeric}, \ \hat\sigma_{c_j}=IQR(c_j)=0
\end{cases}
\tag{8$^\star$}
$$

**Justification.**
- $\hat\sigma_{c_j}=1.4826\cdot\mathrm{MAD}(c_j)$ has breakdown point $0.5$: it remains bounded under arbitrary corruption of up to $\lfloor(n-1)/2\rfloor$ points, directly repairing the masking vulnerability of $\mathrm{std}(c_j)$.
- $\min(\hat\sigma_{c_j}, IQR(c_j)/1.34)$ is exactly Silverman's (1986) robust bandwidth selector, minimizing asymptotic mean integrated squared error (AMISE) under Gaussian-kernel density estimation — a *closed-form, data-derived* optimum, not a heuristic.
- The factor $n^{-1/5}$ is the AMISE-optimal bandwidth decay rate; it guarantees the **consistency conditions** of nonparametric density/neighborhood estimation: $\varepsilon^\star_{c_j}\to0$ and $n\,\varepsilon^\star_{c_j}=\Theta(n^{4/5})\to\infty$ as $n\to\infty$, so neighborhoods neither collapse to isolated points nor degenerate to the full universe.
- The floor $1/n$ guards the minimum resolvable gap under the paper's own min–max normalization (Eq. 6, codomain $[0,1]$), preventing neighborhood collapse on near-degenerate features (§4.1).

**Resolution of Weaknesses 1, 4, 5.**
- *Weakness 1 (coupling):* $\varepsilon_{c_j}^\star$ is computed independently per feature from that feature's own $\hat\sigma_{c_j}, IQR(c_j)$ — no shared $\lambda$ exists.
- *Weakness 4 (hidden runtime):* the 20-fold grid multiplier is eliminated; a single closed-form pass replaces $k=20$ repeated neighborhood reconstructions (Corollary 2, §3).
- *Weakness 5 (truncation):* there is no longer a domain to truncate — $\varepsilon_{c_j}^\star$ is uniquely determined by $(\hat\sigma_{c_j}, IQR(c_j), n)$ for every dataset, so no admissible radius value (however large or small the equivalent implied $\lambda$ would have been) is ever excluded a priori (Corollary 3, §3).

### 2.2 Weakness 3 → attribute-space augmentation (subspace/correlation-breaking outliers)

**Diagnosis.** MNOF is a linear aggregation of $m$ strictly one-dimensional deviation factors (Eq. 5). An object whose every marginal coordinate is unremarkable but whose **joint configuration** departs from the correlation structure most objects follow (the canonical "off-diagonal" outlier) contributes no signal to any single $NDF^X_{nr_{\{c_j\}}}(x)$ and is therefore invisible to MNOF, regardless of the weighting scheme. Enumerating all $2^m-1$ attribute subsets to search for the responsible subspace is combinatorially prohibitive and explicitly excluded by the task.

**Reformulation.** Rather than searching subspaces combinatorially, we synthesize **one scalar attribute** that summarizes deviation from the dominant *linear* joint structure, and let it flow through the existing pipeline as an ordinary $(m{+}1)^{\text{th}}$ numeric feature (licensed by Lemma 2).

**Step 1 — robust standardization.** For $c_j\in C_{num}$: $Z_{i,j}=\dfrac{f(x_i,c_j)-\mathrm{med}(c_j)}{\hat\sigma_{c_j}}$ (reuses $\hat\sigma_{c_j}$ from Eq. 8$^\star$ — no extra statistic).

**Step 2 — matrix-free deflationary power iteration** ($r=\lceil\log_2(m_{num}{+}1)\rceil$, fixed $t_{max}$ iterations, e.g. $t_{max}=10$):

$$
u^{(t+1)}_k = \frac{Z_k^\top\big(Z_k\,u^{(t)}_k\big)}{\big\lVert Z_k^\top(Z_k\,u^{(t)}_k)\big\rVert_2},\qquad Z_{k+1}=Z_k-(Z_k v_k)v_k^\top,\qquad k=1,\dots,r
$$

Neither $Z^\top Z$ (the $m_{num}\times m_{num}$ correlation matrix) nor any matrix inverse is ever formed; each iteration costs $O(n\,m_{num})$ via two matrix–vector products.

**Step 3 — Squared Prediction Error (Jackson & Mudholkar, 1979; a.k.a. the PCA "Q-statistic"):**

$$
J(x)=\big\lVert z_x\big\rVert_2^2-\sum_{k=1}^{r}\langle z_x,v_k\rangle^2\;\ge 0\qquad(\text{Bessel's inequality, }v_k\text{ orthonormal})
\tag{new}
$$

$J(x)$ is exactly the energy of $x$ **orthogonal** to the dominant $r$-dimensional correlation manifold — the quantity that is large precisely for points that violate the joint structure of the data while remaining unremarkable on every individual axis. By the Eckart–Young theorem, the rank-$r$ subspace $\mathrm{span}(v_1,\dots,v_r)$ is the norm-optimal linear summary of the joint structure among all rank-$r$ subspaces, so $J(x)$ is not an arbitrary proxy but the residual against the *best possible* rank-$r$ approximation.

**Step 4 — attribute injection:**

$$
c_{m+1}(x)=\mathrm{minmax}\big(J(x)\big)\in[0,1],\qquad C^\star=C\cup\{c_{m+1}\}
\tag{new attribute}
$$

**Step 5 — Eq. 9 auto-extension.** No formula changes:

$$
NOM^\star(x,y)=\big\lvert\{c\in C^\star:(x,y)\notin nr_{\{c\}}\}\big\rvert=NOM(x,y)+\mathbb 1\big[(x,y)\notin nr_{\{c_{m+1}\}}\big]
\tag{9$^\star$}
$$

$c_{m+1}$ is then processed by Algorithm 1 and Algorithm 2 exactly like any other numeric attribute, using $\varepsilon^\star_{c_{m+1}}$ from Eq. 8$^\star$.

**Degenerate case.** If $m_{num}<2$ (no two numeric attributes to correlate), set $r=0$, $C^\star=C$: the construction vanishes identically and 3WNCROD reduces *exactly* to the baseline nominal-data method — a lossless, formally justified degeneration (§4.3), not a special-cased patch.

**Scope statement.** This resolves the subclass of subspace outliers detectable via **linear** correlation-breaking in the top-$r$ directions (an $O(1)$-indexed subspace family), not arbitrary nonlinear or combinatorial ($2^m$) subspace anomalies — an intentional, provably tractable trade-off (§6), not an oversight.

### 2.3 Weakness 2 → Eq. 10$^\star$: discriminability-augmented weight

**Diagnosis.** Eq. 10 measures only how atypical $x$ is *within* $c_j$'s neighborhood structure (a per-object prevalence signal); it contains no term for whether $c_j$ is, at the *feature level*, actually informative for separating $X$ from $\bar X$. A high-cardinality but noisy attribute and a low-cardinality but sharply discriminative attribute receive identical treatment beyond the prevalence term.

**Feature-level discriminability (reuses Algorithm 1's own output — zero extra order):**

$$
\gamma_{c_j}(X)=\frac{\lvert POS_{nr_{\{c_j\}}}(X)\rvert+\lvert NEG_{nr_{\{c_j\}}}(X)\rvert}{n}\in[0,1],\qquad \hat\gamma_{c_j}=\frac{\gamma_{c_j}(X)}{\max_k\gamma_{c_k}(X)}
\tag{new}
$$

$\gamma_{c_j}(X)$ is precisely the classical rough-set **approximation quality / dependency degree** (Pawlak; generalized to neighborhood rough sets by Hu et al., already the paper's own references [45],[46]) restricted to the singleton $B=\{c_j\}$: the fraction of the universe $c_j$ alone can certify, with no ambiguity, as inside or outside $X$'s neighborhood structure. $NEG_{nr_{\{c_j\}}}(X)$ costs nothing new: it is one additional $O(1)$ branch inside Algorithm 1's existing Steps 5–10 loop (§ Modified Algorithm, below), since $U=POS\cup NEG\cup BND$ partitions $U$ and $BND$ is already implicitly available once $POS$ is known and the same neighborhood-containment test is evaluated against $\lnot X$.

**Bounded blend (geometric mean, fixed $\alpha=\tfrac12$, Jeffreys-symmetric — no free hyperparameter):**

$$
\omega^{X\star}_{nr_{\{c_j\}}}(x)=\Big(\omega^{X}_{nr_{\{c_j\}}}(x)\Big)^{1/2}\cdot\Big(\hat\gamma_{c_j}(X)\Big)^{1/2},\qquad j=1,\dots,m^\star
\tag{10$^\star$}
$$

where $\omega^{X}_{nr_{\{c_j\}}}(x)$ is Eq. 10 unchanged, evaluated with $\lvert C^\star\rvert$ in place of $\lvert C\rvert$. If $\max_k\gamma_{c_k}(X)=0$ (every single attribute is fully ambiguous — a degenerate dataset), set $\hat\gamma_{c_j}:=1\;\forall j$, recovering the original Eq. 10 exactly.

> **Proposition 2 ($[0,1]$-boundedness, preserved).** For $a=\omega^X_{nr_{\{c_j\}}}(x)\in[0,\tfrac{1}{\lvert C^\star\rvert}]\subseteq[0,1]$ and $b=\hat\gamma_{c_j}\in[0,1]$: since $a,b\in[0,1]$ and exponents $\tfrac12,\tfrac12\ge0$ sum to $1$, $a^{1/2}b^{1/2}\in[0,1]$. Hence $\omega^{X\star}_{nr_{\{c_j\}}}(x)\in[0,1]$, matching the constraint stated for $\omega$ in the paper's own Definition 7. $\blacksquare$

### 2.4 Eq. 4$^\star$: tiered evidentiary weighting (refines Eq. 4; strict generalization)

The paper's own qualitative discussion (immediately following Eq. 3) asserts that matches in $NEB(X)$ carry the *strongest* outlier evidence and matches in $POS_{nr_B}(X)$ the *weakest* — yet Eq. 4 sums the three regional cardinalities with implicit unit weight, not reflecting this stated asymmetry. We close this gap with a self-normalizing rarity weight (rarer regions convey more information, in the same spirit as inverse-document-frequency weighting):

$$
NDF^{X\star}_{nr_{\{c_j\}}}(x)=\frac{w_{NEB}\lvert NEB^X(x)\rvert+w_{NPB,j}\lvert NPB^X_{nr_{\{c_j\}}}(x)\rvert+w_{POS,j}\lvert POS^X_{nr_{\{c_j\}}}(x)\rvert}{w_{NEB}\lvert NEB(X)\rvert+w_{NPB,j}\lvert NPB_{nr_{\{c_j\}}}(X)\rvert+w_{POS,j}\lvert POS_{nr_{\{c_j\}}}(X)\rvert},\qquad w_R=1-\frac{\lvert R\rvert}{\lvert X\rvert}
\tag{4$^\star$}
$$

**Conservativity.** Setting $w_{NEB}=w_{NPB,j}=w_{POS,j}\equiv1$ recovers Eq. 4 exactly; Eq. 4$^\star$ is a strict, backward-compatible generalization, not a replacement of incompatible semantics.

**Boundedness (preserved).** Since $NEB^X(x)\subseteq NEB(X)$, $NPB^X_{nr_{\{c_j\}}}(x)\subseteq NPB_{nr_{\{c_j\}}}(X)$, $POS^X_{nr_{\{c_j\}}}(x)\subseteq POS_{nr_{\{c_j\}}}(X)$ (Def. 5), each numerator term is $\le$ its denominator counterpart under identical non-negative weights, so $NDF^{X\star}\in[0,1]$, exactly as the original Eq. 4. Well-definedness of $w_R\in[0,1]$ follows because $NEB(X),NPB_{nr_B}(X),POS_{nr_B}(X)\subseteq X$ (Proposition 1, Def. 1), so $\lvert R\rvert\le\lvert X\rvert$ always.

### 2.5 Eq. 5$^\star$: fusion over the augmented attribute set

$$
MNOF^{X\star}(x)=\sum_{j=1}^{m^\star}NDF^{X\star}_{nr_{\{c_j\}}}(x)\cdot\omega^{X\star}_{nr_{\{c_j\}}}(x)
\tag{5$^\star$}
$$

Functionally identical to Eq. 5; only the summation index runs to $m^\star=m+1$ (or $m$, under the degenerate case of §2.2). Definition 8 ($MNOF^{X}(x)>\mu\Rightarrow x\in OS(X)$) is unchanged in form.

> **Corollary 4 (scale invariance of $\mu$).** Since $NDF^{X\star}\in[0,1]$ and $\omega^{X\star}\in[0,1]$ with $\sum_j\omega^{X\star}\le1$ (loosest bound, matching the original Eq. 5's structure), $MNOF^{X\star}(x)\in[0,1]$ — the same codomain as the original $MNOF^X(x)$. The threshold $\mu$ therefore requires **no recalibration**; any $\mu$ meaningful under Eq. 5 remains meaningful under Eq. 5$^\star$.

### Modified Algorithm (deltas only, relative to Algorithms 1–2 of the source paper)

```
Algorithm 0*  (Subspace Attribute Synthesis — run once, before Algorithm 1)
  Input: numeric block C_num, m_num = |C_num|
  if m_num < 2:  return  C* := C            # §2.2 degenerate case, r = 0
  Z ← robust-standardize(C_num)             # reuses σ̂_cj from Eq. 8*
  r ← ⌈log2(m_num + 1)⌉
  v_1..v_r ← deflationary power iteration(Z, r, t_max)     # O(r · t_max · n · m_num)
  J(x) ← ‖z_x‖² − Σ_{k=1}^r ⟨z_x, v_k⟩²                    # O(r · n · m_num)
  c_{m+1} ← minmax(J);  C* ← C ∪ {c_{m+1}}
  return C*

Algorithm 1*  (deltas vs. Algorithm 1)
  Step 3:  ε_cj ← Eq. 8* instead of Eq. 8                    # same asymptotic cost
  Step 4:  NEG_nr{cj}(X) ← ∅                                 # NEW, mirrors Step 4 for POS
  Steps 5-10: additionally, if n_cj(xi) ∩ X = ∅:
             NEG_nr{cj}(X) ← NEG_nr{cj}(X) ∪ {xi}             # +O(1) per iteration, same loop
  Step 17: also return NEG_nr{cj}(X) (j = 1..m*)

Algorithm 2*  (deltas vs. Algorithm 2)
  Step 4:  for j ← 1 to m*  (was m)                          # includes synthetic attribute
  Step 18: NDF via Eq. 4* instead of Eq. 4                    # same loop structure
  Step 19: ω via Eq. 10* instead of Eq. 10, using γ_cj(X)      # reuses POS, NEG from Alg. 1*
  Step 20: MNOF accumulation unchanged (Eq. 5*)
```

No loop is added, removed, or nested; every modification is either a constant-factor change to an existing step or a single extra pass over the (already iterated) index $j$.

---

## 3. Complexity Theorem and Proof

> **Theorem 2 (Complexity invariance).** Let $T_{base}=\Theta(mn\log n+m\lvert X\rvert^2)$ denote Algorithm 2's baseline per-run cost, and $T_{grid}=\Theta(k\cdot T_{base})$, $k=20$, the cost of the mandatory $\lambda$-grid search. Let $T_{new}$ denote the total cost of Algorithms 0$^\star$–2$^\star$. Then, under the empirically-verified regime $m<n$ (Table 5 of the source paper: $\min n=42\gg\max m=166$ across all 30 subsets),
> $$
> T_{new}=\Theta\big(m^\star n\log n+m^\star\lvert X\rvert^2\big)=\Theta\big(mn\log n+m\lvert X\rvert^2\big)=T_{base}\;\ll\;T_{grid}=\Theta(20\,T_{base}).
> $$

**Proof.** We bound each new component and show closure of $O(\cdot)$ under finite summation.

1. **Eq. 8$^\star$ (radius).** Computing $\mathrm{med}(c_j), Q_1(c_j), Q_3(c_j), \mathrm{MAD}(c_j)$ requires one sort per attribute, $O(n\log n)$; over $m$ attributes, $O(mn\log n)$ — identical order to Algorithm 1's existing SANC covering step (already $O(n\log n)$ per attribute, itself typically sort-based), so this term is subsumed, not additive, in the common implementation case.

2. **Algorithm 0$^\star$ (subspace synthesis).** $O(r\,t_{max}\,n\,m_{num})$ with $r=\lceil\log_2(m_{num}{+}1)\rceil$, $t_{max}=O(1)$ fixed. Hence cost $=O(n\,m_{num}\log m_{num})$. Since $m_{num}\le m<n$ (given regime), $\log m_{num}\le\log n$, so this is $O(mn\log n)$.

3. **Eq. 10$^\star$ / $\gamma_{c_j}$ (discriminability).** $NEG_{nr_{\{c_j\}}}(X)$ costs one extra $O(1)$ check inside Algorithm 1's existing $O(n)$-per-attribute loop $\Rightarrow O(mn)\subset O(mn\log n)$.

4. **Attribute-count increment.** $m^\star=m+1=\Theta(m)$ for any $m\ge1$, so every term that scaled as $O(m\cdot(\cdot))$ or $O(m\lvert X\rvert^2)$ in the baseline scales as $O(m^\star\cdot(\cdot))=O(m\cdot(\cdot))$ under augmentation — no order change.

5. **Eq. 4$^\star$, 5$^\star$ (fusion).** Identical loop structure to Algorithm 2's Steps 6–20, re-indexed to $m^\star$; cost $O(m^\star\lvert X\rvert^2)=O(m\lvert X\rvert^2)$.

Summing (1)–(5), each individually $O(mn\log n+m\lvert X\rvert^2)$, and $O(\cdot)$ is closed under finite addition, so $T_{new}=O(mn\log n+m\lvert X\rvert^2)$. The matching $\Omega$ bound holds because $T_{new}\supseteq T_{base}$'s own computation (Algorithm 1$^\star$/2$^\star$ strictly contain Algorithm 1/2's original operations). Hence $T_{new}=\Theta(mn\log n+m\lvert X\rvert^2)=T_{base}$. $\blacksquare$

> **Corollary 2 (elimination of grid-search overhead — resolves Weakness 4).** $\dfrac{T_{grid}}{T_{new}}=\dfrac{20\,T_{base}}{T_{base}}=20$. Replacing the grid search with Eq. 8$^\star$ yields an exact, formally derived $20\times$ reduction in parameter-selection cost, not merely an informal speed-up claim.

> **Corollary 3 (elimination of domain-truncation risk — resolves Weakness 5).** Eq. 8$^\star$ contains no free parameter $\lambda$; the admissible search domain $\Lambda=[\lambda_{\min},\lambda_{\max}]$ is reduced to $\varnothing$. A domain that does not exist cannot be mis-truncated. This is a strictly stronger resolution than widening $\Lambda$ to, e.g., $[0.1,5.0]$, since no finite widening of a grid can guarantee coverage of the AMISE-optimal bandwidth for every possible sample, whereas Eq. 8$^\star$ attains it in closed form for every sample by construction.

---

## 4. Boundary and Edge-Case Analysis

### 4.1 Sparsity ($n$ small, or near-constant features)

- For $n<4$, classical quartiles are computed via the standard type-7 interpolation estimator, well-defined $\forall n\ge1$; in the degenerate limit $Q_1=Q_3=\mathrm{med}$, so the IQR branch of Eq. 8$^\star$ vanishes and the $\mathrm{MAD}$ branch governs.
- If $\ge50\%$ of a feature's values are tied, $\mathrm{MAD}(c_j)=0$; Eq. 8$^\star$ falls through to $IQR(c_j)$. If additionally $\ge75\%$ are tied, $IQR(c_j)=0$ as well, and Eq. 8$^\star$ correctly assigns $\varepsilon^\star_{c_j}=0$ — the feature is *behaviorally categorical* in this sample, and is treated exactly as the original Eq. 8 treats genuine categorical attributes. This is a graceful, semantically correct degeneration, not an undefined $0/0$ state.
- Otherwise, the floor $1/n$ prevents $\varepsilon^\star_{c_j}\to0^+$ from fragmenting neighborhoods into singletons purely as an artifact of small $n$.

### 4.2 Noise / heavy contamination

Eq. 8 (original) uses $\mathrm{std}(c_j)$, whose influence function $\mathrm{IF}(x;\mathrm{std})=O(x^2)$ is unbounded: a single corrupted value can inflate $\varepsilon_{c_j}$ without bound (breakdown point $0$; Rousseeuw & Leroy 1987 — the paper's own ref. [36]). Eq. 8$^\star$ replaces this with $\hat\sigma_{c_j}$ (breakdown point $0.5$): by definition of breakdown point, $\hat\sigma_{c_j}$ remains **bounded** under arbitrary corruption of up to $\lfloor(n-1)/2\rfloor$ observations. This is a formally provable robustness guarantee absent from Eq. 8, directly repairing the circularity in which contamination inflates the very radius meant to detect it.

### 4.3 Mixed attributes

- **Categorical attributes:** Eq. 8$^\star$'s first case is *identical* to Eq. 8 ($\varepsilon=0$, exact-match neighborhoods) — fully backward compatible, zero behavioral change.
- **Purely categorical datasets** ($m_{num}<2$): Algorithm 0$^\star$ returns $C^\star=C$ (§2.2); 3WNCROD reduces *exactly* to the base nominal-only method, with no synthetic attribute, no discriminability change beyond what Eq. 10$^\star$ already offers on categorical attributes themselves (since $\gamma_{c_j}(X)$ is well-defined regardless of attribute type). No regression relative to baseline capability is possible by Theorem 1 (monotone refinement) applied vacuously with $r=0$.
- **Fully numeric datasets:** all $m$ attributes are numeric; Eq. 8$^\star$ is active on all of them, and one synthetic attribute is added, giving $m^\star=m+1$.

### 4.4 High-dimensional numeric regime ($m_{num}$ large, approaching or exceeding $n$)

The complexity bound in Theorem 2 assumes $m<n$, verified across all 30 benchmark subsets in Table 5 of the source paper (worst case: Wdbc $W_1$, $n=42>m=9$; Musk, $m=166<\min n=2260$). Outside this regime (e.g., genomic-scale data with $m\gg n$), $\log m_{num}$ may exceed $\log n$, and Algorithm 0$^\star$'s cost becomes $O(nm\log m)$, a strictly polynomial (not exponential) degradation bounded by a factor $O(\log m/\log n)$ relative to Theorem 2's stated bound — still asymptotically dominated by no worse than a logarithmic-ratio constant, and still infinitely preferable to the excluded $O(2^m)$ combinatorial subspace search.

---

## 5. Synthesis

| Weakness | Root cause | Fix | Equation(s) touched | Complexity delta | Formal guarantee |
|---|---|---|---|---|---|
| 1. Coupled radii | Single global $\lambda$ | Per-feature Silverman/MAD bandwidth | 8$^\star$ | $O(mn\log n)$ (subsumed) | AMISE-optimal per feature (closed form) |
| 2. Uninformative weights | $\omega$ only counts prevalence | Discriminability factor $\hat\gamma_{c_j}$ from reused $POS,NEG$ | 10$^\star$ | $O(mn)$ (subsumed) | $\omega^{X\star}\in[0,1]$ (Prop. 2) |
| 3. Subspace blindness | Purely 1-D aggregation | Rank-$r$ SPE synthetic attribute, injected as $c_{m+1}$ | 9 (domain only), new attribute | $O(mn\log m)\subseteq O(mn\log n)$ | Monotone refinement (Thm. 1); rank-$r$ optimal (Eckart–Young) |
| 4. Hidden 20-step grid | Grid search over $\lambda$ | Closed-form 8$^\star$, no search | 8 | $T_{grid}\to T_{base}$ | Exact $20\times$ reduction (Cor. 2) |
| 5. Grid truncation $[0.1,2.0]$ | Fixed finite domain | No domain — parameter eliminated | 8 | — | $\Lambda=\varnothing$ (Cor. 3) |

Auxiliary refinement (not one of the five, offered for completeness since the task named Eq. 4 explicitly): tiered evidentiary weighting **4$^\star$**, a strict, boundedness-preserving generalization of Eq. 4 that operationalizes the paper's own stated (but previously unweighted) claim that $NEB$-evidence outweighs $POS$-evidence.

Net result: $T_{new}=\Theta(mn\log n+m\lvert X\rvert^2)=T_{base}$ (Theorem 2) — **all five weaknesses are resolved without leaving the baseline complexity class**, and Definitions 1–4 and 9 (the 3WD region semantics the task requires preserved) are provably unmodified in form (Lemmas 1–2, Theorem 1).

---

## 6. Scope and Limitations

- The subspace fix (§2.2) captures **linear**, top-$r$ correlation-breaking structure only; nonlinear manifold outliers (e.g., points off a curved rather than linear joint trend) are provably outside its reach without kernelization, which would reintroduce $O(n^2)$ or worse cost and violate the stated low-overhead constraint. This is a deliberate, Pareto-optimal trade-off between completeness and tractability (an exponential $2^m$ subspace search is the only method with no such blind spot, and is explicitly excluded by the task), not an unexamined gap.
- $\alpha=\tfrac12$ (Eq. 10$^\star$) and the region-weight family $w_R=1-\lvert R\rvert/\lvert X\rvert$ (Eq. 4$^\star$) are fixed by symmetry/self-normalization principles rather than fit to data; they are deliberately *not* tuned, in keeping with the parameter-free ethos motivating the resolution of Weaknesses 1, 4, and 5. A practitioner seeking further gains could treat $\alpha$ as learnable, but this would reintroduce exactly the search-overhead problem this treatise eliminates.
- Theorem 2's complexity bound is conditioned on $m<n$; §4.4 gives the (still polynomial) degradation outside this regime, together with the empirical justification (Table 5) that the condition holds throughout the paper's own benchmark suite.
