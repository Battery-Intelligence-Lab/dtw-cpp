# Exact k-Medoids Solver — Mathematical Investigation and Design Verdict

**Date:** 2026-07-06 · **Scope:** verify `.claude/UNIMODULAR.md`, extract the implemented formulation, audit previous own-solver attempts, engage the p-median literature, recommend ONE solver architecture with registered predictions. No source code was modified. All experiments live in the session scratchpad (`tu_verify.py`, `tu_verify2.py`, `tu_verify3.py`); their registered bands and verbatim outputs are reproduced in §7.

**Notation.** N = number of series (UNIMODULAR.md calls this p), k = number of medoids (UNIMODULAR.md's k/Nc). D ∈ R^{N×N} is the DTW distance matrix: dense, symmetric, zero diagonal, **not** guaranteed to satisfy the triangle inequality [confirmed: `.claude/LESSONS.md:9`; Marteau 2009; Jain 2018].

---

## 1. The ILP as implemented [confirmed against code]

Both backends implement the Balinski (1965) p-median formulation with variables x_ij = "point j assigned to medoid candidate i", y_i ≡ x_ii = "i is a medoid":

```
min   Σ_ij D_ij x_ij / s          s = max(max_ij D_ij / 2, 1)   (objective scaling)
s.t.  Σ_i x_ij = 1        ∀j      (assignment, N rows)
      Σ_i x_ii = k                (cardinality, 1 row)
      x_ij ≤ x_ii         ∀ i≠j   (linking, N(N−1) rows)
      x_ij ∈ {0,1}                (N² binaries)
```

Verified line-by-line:

- **Gurobi** (`dtwc/mip/mip_Gurobi.cpp`): flat index x[i,j] = w[i + j·Nb] (column j = point). Assignment at lines 47–53 (`lhs += w[j + i*Nb]` summed over the block of point i), linking at 56–58, cardinality at 60–66, objective 71–76 with scaling at 69. Branch priority 100 on diagonals already set (44–45); FastPAM MIP start (88–102). **Note:** the linking loop includes i = j, adding N trivial rows `x_ii ≤ x_ii` (presolve removes them; harmless, mildly wasteful) [confirmed: lines 56–58 loop over all (i,j)].
- **HiGHS** (`dtwc/mip/mip_Highs.cpp`): flat index x[i,j] = i·Nb + j (row i = medoid). Rows: 0 = cardinality, 1..Nb = assignment, then Nb(Nb−1) linking rows built via triplets (114–132). Linking `row_lower_ = −1` instead of −∞ (line 88) — valid since x_ij − x_ii ≥ −1 under the variable bounds, so the extra side never binds [confirmed by inspection]. FastPAM warm start (172–189). Nonzeros ≈ N + N² + 2N(N−1) ≈ 3N².
- **Dispatch** (`dtwc/Problem.cpp:538–547`): `benders == "on"`, or `"auto"` with N > 200 → `MIP_clustering_byBenders`; otherwise compact MIP.

Size at the user's targets: N = 10⁴ ⇒ 10⁸ binaries, ~3·10⁸ nonzeros (≥ 12–24 GB matrix storage alone). **The compact form is not a viable vehicle at N = 10⁴ regardless of solver** [confirmed by arithmetic above]. This is why the question "can we exploit near-unimodularity?" is the right one — but the answer must avoid ever *forming* the N²-column LP.

---

## 2. Total unimodularity: what holds, what doesn't

### 2.1 UNIMODULAR.md's central claims — CONFIRMED (independently re-derived + enumerated)

**Claim (UNIMODULAR.md §2.2–2.6): the constraint matrix is TU for N ≤ 2 and not TU for N ≥ 3, the violation being the 6×6 facility-3-cycle submatrix with det = −2.**

*Hand re-derivation* [confirmed]. Rows L(0,1), Ass(1), L(1,2), Ass(2), L(2,0), Ass(0); columns x00, x01, x11, x12, x22, x20:

```
        x00 x01 x11 x12 x22 x20
L(0,1)  -1   1   0   0   0   0      (x01 − x00 ≤ 0)
Ass(1)   0   1   1   0   0   0      (x01+x11+x21 = 1; x21 not selected)
L(1,2)   0   0  -1   1   0   0
Ass(2)   0   0   0   1   1   0
L(2,0)   0   0   0   0  -1   1
Ass(0)   1   0   0   0   0   1
```

Laplace expansion down column 0 (nonzeros −1 at row 0, +1 at row 5):
det = (−1)·det(M₀₀) + (+1)·(−1)⁵·det(M₅₀). M₀₀ is upper triangular with diagonal (1,−1,1,−1,1) ⇒ det = 1; M₅₀ is lower triangular with all-ones diagonal ⇒ det = 1. Hence **det = −1 − 1 = −2**. Every step above was recomputed in this session, entry by entry, from the constraint definitions — not copied.

*Exhaustive enumeration* [confirmed: `tu_verify.py`, bands registered before running]:
- **B1 (N=2, 5×4 matrix, all 125 square submatrices): 0 violations — PASS.** (I additionally verified all five 4×4 minors by hand cofactor expansion; each has det 0.)
- **B2 (N=3, 10×9 matrix, all 92,377 square submatrices): exactly 2 violations, both 6×6, dets +2 and −2** (the two directed 3-cycles) — **PASS**, exactly as UNIMODULAR.md §3.3 claims.
- **B3 (the specific §2.4 submatrix): det = −2 — PASS.**

The n-cycle formula det = (−1)ⁿ − 1 (§2.5) was spot-verified by hand at n = 2 (det 0) and n = 3 (det −2); the two-triangular-minor structure of the proof is sound [confirmed for n ≤ 3; general n by the block-band argument in UNIMODULAR.md, which I checked and accept].

Consistency check with complexity theory: TU would give LP = IP for every D, i.e. a polynomial exact algorithm, contradicting NP-hardness of p-median (Kariv & Hakimi 1979). Non-TU for N ≥ 3 is therefore *necessary* [confirmed].

### 2.2 Strengthening: §2.8's "verified computationally for p ≤ 4" rows hold for ALL N

UNIMODULAR.md proves two blocks TU only computationally for N ≤ 4. Both admit Ghouila-Houri proofs valid for every N (derived this session):

**Cardinality + Assignment is TU for all N.** Every entry is 0/1; column x_ii has two ones (Card, Ass(i)); column x_ij, i≠j, has one (Ass(j)). A 0/1 matrix with ≤ 2 ones per column is the incidence matrix of a graph on the rows; it is TU iff that graph is bipartite (Hoffman–Kruskal). Here the graph's edges all join Card to some Ass(i) — a **star**, which is bipartite. ∎

**Cardinality + Linking is TU for all N.** Ghouila-Houri: given any row subset R, sign Card as +1 if present. For each i, the rows L(i,·) ∩ R are the only other rows touching column x_ii (each with −1), and their +1 entries sit in columns x_ij that carry no other nonzero in this subsystem. Sign the m_i rows of L(i,·) ∩ R alternately: ⌈m_i/2⌉ as +1, ⌊m_i/2⌋ as −1, so their signed sum on column x_ii is −(m_i mod 2) ∈ {−1, 0}, wait — with signs s ∈ {±1} multiplying entries −1: contribution ∈ {0, −1}·(sign choice); choosing the split gives column-i sum σ_i ∈ {0, 1} after including Card's +1, and every off-diagonal column has a single signed entry ±1. All column sums lie in {−1, 0, 1}. ∎ (Written out fully: |s_Card·1 − Σ_{L(i,·)∈R} s_L·1| = |1 − (m_i mod 2)·(±1)| ≤ 1 with the alternating split.)

So the *only* non-TU interaction is Assignment × Linking through the shared diagonal columns — precisely UNIMODULAR.md §2.8's table, now proved for all N, not just N ≤ 4.

### 2.3 What is WRONG or overstated in UNIMODULAR.md

1. **§3.3 "Fractional LP solutions are typically half-integral" — FALSIFIED empirically for this (cardinality-constrained) formulation.** Registered band B6 (≥ 90% of fractional components in [0.45, 0.55]): **FAIL — only 63.0% of 664 fractional components; observed values include 1/4, 1/3, 3/4** (quantiles 0.25/0.333/0.5/0.5/0.75) [confirmed: `tu_verify.py` output]. Baiou–Barahona's half-integrality theorems concern specific polytopes (UFL-type / restricted graph classes), and the cardinality row Σy_i = k manifestly creates non-half-integral vertices (e.g. three facilities at 1/3 sharing one cardinality unit). **Design consequence: {0,1/2}-Chvátal–Gomory (odd-cycle) cuts alone cannot close all fractional instances; branching is mandatory.** The TODO item "odd-cycle cutting planes" was demoted accordingly and is now recorded as falsified (`.claude/TODO.md`, row M01).
2. **§2.11's "fractional solution x_ij = 1/3" is a barycenter, not a vertex.** It is the average of the three integer optima, hence proves nothing about polytope non-integrality (any polytope contains convex combinations). The genuine object is a fractional *extreme point*; those exist (my B4 experiment produced 35 of them at N=10 with strictly positive gap, max 13.7%), but §2.11 as written is not evidence.
3. **§3.1 attribution error [inferred — verify before citing]:** the ≈2.675 k-median approximation is Byrka, Pensyl, Rybicki, Srinivasan, Trinh (2015/2017); Cohen-Addad et al. (2022, "breaching the 2-LMP barrier" line) improved *beyond* it (≈2.406). Also, approximation ratios and LP integrality-gap upper bounds are distinct objects; only LP-rounding proofs (Byrka et al. is one) bound the gap. The stated [2, ~2.675] gap window is roughly right for metric k-median but the citation is crossed.
4. **§3.2/§5.1 "LP integer ~80–90% of the time" is data-regime-dependent, and the dependence matters more than the number.** Measured this session: uniform-random non-metric D — 82% integral at N=10, k=3, but **26% at N=20, k=4** (band A3 FAIL) and *falling with N*; clustered non-metric D (3 line clusters, squared distances) — **100% integral at N∈{15,30}, for k ∈ {2,3,4}, including mis-specified k** (bands B5, A4, A5, A6 all PASS 50/50 or 100/100). ReVelle's folklore holds on structured data, not adversarial data. Corollary for DTWC++: real clustered series ⇒ root bound almost always exact; but the solver must not *assume* it.
5. **"Fixed medoids ⇒ transportation problem" (§3.2, §5.4) is a true but overstated crutch.** With y fixed there are no coupling constraints left at all: each point independently takes argmin_{i: y_i=1} D_ij. No LP, no network simplex, no transportation structure needed — an O(Nk) scan. (This also kills architecture option "(iv) network-simplex-based": the only network-flow substructure available is one that a trivial scan already solves.)
6. **§5.4 "O(pk) per cut generation" understates the implemented cost:** the disaggregated cut coefficients max(0, d_nearest(j) − D_ij) require the full row scan — O(N²) per Benders round as implemented (`benders.cpp:298–328`) [confirmed]. Sorted-neighbor lists (Duran-Mateluna et al. 2023) reduce this.
7. **A1/A2 findings (new, registered then run):** when the LP *is* fractional (uniform instances), the fractionality is not a small local defect — median **6 of 10** diagonals fractional (band A1 FAIL vs ≤ 4), and **0/35** fractional vertices had zero gap (band A2 FAIL) — fractionality ⇒ genuine gap, and "branch on the couple of fractional y_i" is optimistic on unstructured data. On clustered data the question is moot (no fractional roots observed).

### 2.4 What the user's "almost totally unimodular" observation actually is

Three precise statements, in decreasing strength — the user's intuition is the conjunction:

- **(a) Structural:** every square submatrix avoiding an odd facility-cycle pattern has det ∈ {0, ±1}; at N=3 exactly 2 of 92,377 submatrices violate TU [confirmed by enumeration]. The violating pattern density grows with N (number of odd cycles ~ Σ over odd L of N!/((N−L)!·2L)), so "almost TU" in the counting sense decays — but:
- **(b) Polyhedral:** fractional vertices require odd-cycle support in the LP solution (Baiou–Barahona 2009/2011); optimal solutions on *clustered* data don't activate them — measured 100% integral roots across 250 clustered instances [confirmed].
- **(c) Algorithmic:** two of the three constraint blocks pairwise-combine TU (§2.2 above); with y fixed, everything collapses to a scan. So all hardness lives in **the k-subset choice of y**, an N-dimensional binary decision — not in the N²-dimensional x. That is the actionable content: *any good exact solver works in y-space and treats x implicitly.*

---

## 3. Previous own-solver attempts — git archaeology and failure analysis

Recovered from history (all code removed in `f7064b3` "removed specialised solvers", 2023-12-07):

| Period | Attempt | Evidence | Why it failed |
|---|---|---|---|
| Oct 2023 | **Dense tableau primal simplex** (Eigen `MatrixXd`, Phase I/II, full tableau) + **Gomory cuts** | `Simplex.cpp/.hpp`, commits `5f983c9`, `3f18761` "Working Simplex functions", `98319a7` "gomory, gomoryAlgorithm", `5968fb9` **"Simplex table takes huge space!"** | Tableau is (m+1)×(n+m+1) with n = N², m ≈ N²+N+1 ⇒ ~2N⁴ doubles (N=100 ⇒ ~1.6 GB, N=316 ⇒ ~160 GB); every pivot rewrites it, O(mn) work; basis identified by *scanning for unit columns* (`getRow`, O(mn) per query, brittle at ε=1e-8). Textbook fractional Gomory cuts on a dense tableau are numerically fragile and converge glacially — the known-good cuts for this polytope are the odd-cycle family, never implemented. |
| Oct–Nov 2023 | **Sparse tableau simplex** (`SparseSimplex`, `SimplexFlatRowTable`, row/flat variants), **dual simplex** graft | `99d1912`, `6f1dba1`, `850d3a3`, `db5063c` "trying to integrate a dual simplex routine", `1ececcd` **"update still not solving but probably simplex issue"** | Still *tableau-updating* (not revised simplex): sparse tableaus fill in after a handful of pivots, so sparsity dies; no LU/product-form basis (`6e3588c` "trial for LU but just started" — abandoned); correctness never achieved. |
| ~same | **First-order LP attempts**: OSQP/ADMM, conjugate gradient | `3a87ea3` "OSQP and OSLP are removed", `581de25` "epsADMM bugfix", `2f4e8be` "sparse cg with eigen" | ADMM/CG deliver ~1e-3 accuracy; cannot certify vertex optimality or integrality; slow tail convergence — wrong tool for exact certification. |
| Apr 2026 | **Benders decomposition** (live: `dtwc/mip/benders.cpp`) | tests in `tests/unit/unit_test_benders.cpp` pass; auto-dispatch N > 200 | Works, but is the **iterative** variant: each round solves the full master **MIP to optimality**, then adds up to N dense cuts (O(N) nonzeros each ⇒ O(N²) added nonzeros/round). Modern practice (Duran-Mateluna et al. 2023; Fischetti et al.) embeds cuts lazily in ONE branch-and-bound tree. Also: master warm start via `setSolution` trips the HiGHS ≤1.14 `ub_consistent` assertion (worked around with NDEBUG — `.claude/LESSONS.md:69–72`). |

**Root-cause summary.** The 2023 attempts failed for one architectural reason with three faces: they tried to solve the **N²-variable compact LP explicitly**, with (i) the wrong simplex representation (tableau, not revised+LU+bounded-variable), (ii) the wrong cuts (Gomory, not odd-cycle — and §2.3.1 now shows even odd-cycle cuts can't finish alone), and (iii) no upper bound / reduced-cost fixing to shrink anything. The correct lesson is not "write a better simplex"; it is **never materialize the x-space LP at all** (§2.4c). The 2026 Benders code is the first attempt pointed the right way.

---

## 4. Known theory engaged (assessment of each lead)

All items below are from literature knowledge; citation details flagged [verify] should be checked against the papers before external use. Add confirmed ones to `.claude/CITATIONS.md` during implementation.

- **Trees/lines:** p-median on trees is polynomial — Kariv–Hakimi O(N²k²) DP; Tamir (1996) O(kN²); on a path with Monge structure O(kN) via SMAWK (Hassin–Tamir 1991). Kolen (1983): LP relaxation is integral on trees. **Irrelevant here:** DTW matrices carry no tree metric structure (not even a metric). Useful only as intuition for why clustered ≈ "forest-like" data yields integral LPs.
- **ReVelle–Swain (1970):** original empirical observation that the p-median LP is "almost always integral" on real geographic data. Matches my clustered-regime measurements (100%); refuted on uniform-random (26% at N=20) — the folklore is a statement about *data*, not the polytope.
- **Cornuejols–Fisher–Nemhauser (1977):** Lagrangian relaxation of the assignment constraints; dual bound = LP bound. The equality holds by **Geoffrion's theorem** because the inner polytope {Σy=k, x_ij ≤ y_i, 0 ≤ x,y ≤ 1} has the integrality property — its matrix is Cardinality+Linking, which is TU **for all N by my §2.2 proof**. This is the single most load-bearing fact for the design: *the cheap dual attains the full LP bound.* [confirmed: derivation §5.1]
- **Benders / row generation at scale:** Duran-Mateluna, Ales, Elloumi (2023, EJOR): two-phase Benders (LP phase then single-tree branch-and-cut with lazy cuts), instances to 238k points. The repo's Benders is the primitive cousin; upgrade path exists. HiGHS-side blocker: UNIMODULAR.md §6.6 names `kCallbackMipDefineLazyConstraints`; I could not verify that HiGHS exposes lazy-constraint injection as of the vendored version [OPEN — check HiGHS ≥1.8 callback API; Gurobi `addLazy` definitely works].
- **Radius/ordered formulation (García, Labbé, Marín 2011, INFORMS JoC; solver "ZEBRA" [verify name]):** variables count how many distance *values* (sorted radii) are exceeded; column-and-row generation; solved very large p-median instances exactly, often at the LP root. Exploits ties/quantization in D. Strong alternative to Benders; more implementation effort; benefits shrink for continuous DTW distances with few ties.
- **POPSTAR (Resende–Werneck 2004):** GRASP+path-relinking heuristic — an UB competitor to FastPAM, not an exact method; only worth importing if FastPAM UBs prove loose (they did not, in the repo's UCR runs).
- **Column generation / branch-and-price** (set-partitioning over clusters; Senne–Lorena–Pereira 2005; du Merle et al. 1999 for MSSC): pricing = per-candidate-medoid scan, O(N²)/round; the LP bound equals the same Lagrangian/LP bound here (identical Dantzig-Wolfe reformulation of the same blocks), so it buys nothing over subgradient except stabilized-master machinery. Rejected: same bound, higher constant, much more code.
- **Submodularity:** k-median in "max-coverage form" (maximize Σ_j (d_max − service cost)) is monotone submodular under a cardinality constraint ⇒ lazy-greedy gives (1−1/e) on the *shifted* objective and stochastic variants scale to 10⁸ — but guarantees on the shifted objective are weak for the min-form and it is inherently inexact. Use, at most, as a UB seeding alternative. BanditPAM (Tiwari et al. 2020) similarly heuristic-tier.
- **Awasthi et al. (2015):** LP relaxation is provably exact under cluster separation — the *theoretical* underwriting of my clustered-regime measurements and of prediction P1.

---

## 5. Recommended architecture (ONE): **Lagrangian-root + core Benders finish ("LR-core")**

### 5.1 Derivation of the bound engine

Dualize the assignment equalities with multipliers μ ∈ R^N (free):

```
L(μ) = Σ_j μ_j + min { Σ_ij (D_ij − μ_j)·x_ij : Σ_i y_i = k, x_ij ≤ y_i, x,y ∈ [0,1] }
```

The inner minimization decomposes per facility. Define the facility score
```
ρ_i(μ) = Σ_j min(0, D_ij − μ_j)        (open i ⇒ attract exactly its profitable points)
```
Given the cardinality constraint pick the k most negative scores S_k(μ) = argmin-k ρ_i. Then
```
L(μ) = Σ_j μ_j + Σ_{i ∈ S_k(μ)} ρ_i(μ)
```
Every step is exact because the inner polytope's matrix (Cardinality+Linking) is TU (§2.2, proved for all N) so its LP has integral optima — hence, by **Geoffrion (1974)**, max_μ L(μ) = LP relaxation bound. No x-space LP is ever formed. Subgradient: g_j = 1 − Σ_{i∈S_k(μ)} 1[D_ij < μ_j], computed in the same pass. Polyak steps use the FastPAM UB (already in-repo). Dimensional check: μ carries distance units, ρ distance units, g dimensionless counts — consistent throughout.

**Primal repair** each iteration: open S_k(μ), assign by row-min scan (O(Nk)), update incumbent. **Reduced-cost fixing** (Beasley-style): with LB = L(μ) and incumbent UB, facility i ∉ S_k is *fixed closed* if forcing it in costs more than the gap: LB + (ρ_i − ρ_(k)) > UB, where ρ_(k) is the k-th best score; facilities in S_k are fixed open by the symmetric swap test. On clustered data with near-zero root gap, this closes almost all of y.

**Exact finish:** on the surviving core (n_core candidate medoids, typically ≪ N), run the existing Benders path (`benders.cpp`) restricted to the core, warm-started with the incumbent and the Lagrangian LB — upgraded, when solver support allows, to single-tree lazy cuts (Gurobi `addLazy` now; HiGHS callback OPEN). Branching, when needed, is on y only — justified rigorously by §2.4c (fixed y ⇒ trivial scan), NOT by the falsified "few fractional y" heuristic (§2.3.7).

### 5.2 Why this beats the alternatives asked about

| Option | Verdict | Reason |
|---|---|---|
| (i) LP-first + branch on fractional y | Subsumed | LR **is** the LP bound, obtained matrix-free at O(N²)/iter instead of solving a 10⁸-column LP; integrality check = "is the repaired primal equal to LB". |
| (ii) Lagrangian + subgradient + repair | **Chosen (root engine)** | Same bound as (i)/(iii); cheapest per unit bound; streams D exactly like the existing memory-bound DTW kernels; OpenMP/H100-trivial (row-wise reductions). |
| (iii) Column generation | Rejected | Identical Dantzig-Wolfe bound (same TU blocks), higher constants, master-LP machinery, stabilization headaches. |
| (iv) Network-simplex | Rejected | The only flow substructure is the fixed-y assignment — solvable by an O(Nk) scan; no flow structure spans the y-choice (that is precisely what the TU failure proves). |

### 5.3 Cost model, N = 10⁴, k = 20 (leading-order terms named)

- **Memory:** D itself, N²/2 entries packed: 400 MB f64 / 200 MB f32 (infrastructure exists: packed triangular + mmap). Solver state O(N). **Leading term: the distance matrix — already paid.**
- **Root time:** each subgradient iteration streams D once: N² = 10⁸ reads ⇒ 0.8 GB (f64). At 20–40 GB/s effective DRAM bandwidth: **20–40 ms/iter**; T ≈ 300–1000 iterations ⇒ **10–40 s** single node. It is memory-bandwidth-bound, embarrassingly parallel over rows; on one H100 (3.3 TB/s HBM): ≲ 1 ms/iter, root in < 1 s. **Leading-order cost: T·N² memory traffic.**
- **Finish:** clustered regime ⇒ expected zero or near-zero gap at root (evidence: 250/250 clustered instances integral, §7); core MIP expected trivial. Unstructured worst case: Benders on n_core ≤ 10³ candidates — the regime the current code already handles.
- **N = 10⁵:** D f32 packed = 20 GB (fits 2 TB node); ~1–4 s/iter CPU, ~25 ms/iter H100 ⇒ root bound in minutes. No other exact approach touches this size without forming intractable masters; Benders-only would still carry a 10⁵-binary master MIP per tree.

### 5.4 What to build (implementation order)

1. `lagrangian_root()` — ρ/g/L pass (OpenMP over i), Polyak subgradient, primal repair, reduced-cost fixing. ~200 LOC, no new dependencies.
2. Report `(LB, UB, gap, n_core)`; if gap ≤ tol ⇒ **certified optimal PAM/repair solution, done** (expected path on real data).
3. Else Benders on the core (existing code path, restricted candidate set, incumbent + LB injected). Upgrade to lazy single-tree with Gurobi callback; file the HiGHS callback question upstream [OPEN].
4. Only if P4-style instances appear in practice: add odd-cycle cuts *in the master* as strengtheners — never as the sole closer (falsified, §2.3.1).

---

## 6. Registered predictions for the implementation phase (falsifiable, pre-stated)

- **P1 (root exactness on real data):** on ≥ **90%** of UCR datasets (DTW distances, k = class count, N ≤ 3000), the Lagrangian root closes the gap to the FastPAM UB within relative **0.1%** — i.e. PAM is *certified* optimal at the root with no branching. Falsified below 70%. (Basis: 250/250 clustered instances LP-integral, §7; Awasthi et al. separation theory.)
- **P2 (fixing power):** whenever root gap ≤ 1%, reduced-cost fixing eliminates ≥ **80%** of candidate medoids (n_core ≤ 0.2N). Falsified if median elimination < 50% on gap-≤1% instances.
- **P3 (throughput):** subgradient iteration time within **2×** of bytes(D)/STREAM-bandwidth on the target node; end-to-end exact solve at N = 10⁴, k = 20 in ≤ **10 min** single node (vs. compact MIP: not runnable). Falsified if > 2× off or > 1 h.
- **P4 (cut insufficiency — expected to *hold*, guarding against over-investment):** on uniform-random non-metric instances with fractional roots, {0,1/2}-CG/odd-cycle cuts alone close ≤ **80%** of them (non-half-integral vertices exist — already observed at N=10). If someone measures > 95% closure, revisit §2.3.1.

## 7. Numbers ledger (registered bands → verbatim verdicts, this session)

| Band (registered before run) | Result | Verdict |
|---|---|---|
| B1: N=2 matrix, 0 TU violations | 0 violations | **PASS** |
| B2: N=3, exactly 2 violations, 6×6, \|det\|=2 | 2 violations, sizes [6], dets {−2,+2} | **PASS** |
| B3: §2.4 submatrix det = −2 | det = −2 | **PASS** |
| B4: uniform N=10,k=3: ≥50% integral, mean gap ≤1%, max ≤5% | 82% integral, mean 0.732%, **max 13.718%** | **FAIL (max-gap)** |
| B5: clustered N=15,k=3: ≥90% integral | 100/100 | **PASS** |
| B6: fractional components ≥90% in [0.45,0.55] | **63.0%** of 664; values incl. 1/4, 1/3, 3/4 | **FAIL** (half-integrality falsified) |
| A1: fractional instances have ≤4 fractional y (median) | median **6**, max 8 (of N=10) | **FAIL** |
| A2: ≥20% of fractional vertices are zero-gap | **0/35** | **FAIL** (fractional ⇒ real gap) |
| A3: uniform N=20,k=4 ≥60% integral | **26%** | **FAIL** (integrality decays with N on unstructured data) |
| A4/A5/A6: clustered N=30, k∈{3,4,2} ≥{90,50,50}% integral | 100% / 100% / 100% | **PASS** (robust to mis-specified k) |

Scripts: scratchpad `tu_verify.py`, `tu_verify2.py`, `tu_verify3.py` (numpy + scipy `linprog(method="highs-ds")`, vertex solutions). LP experiments use N ≤ 30 with brute-force IP as oracle; extrapolation to DTW matrices at UCR scale is exactly what P1 tests — [inferred] until then.

**Rollback / most-likely-wrong claim:** nothing was modified outside this report file. The claim most likely to fail is P1's 90% (clustered-line surrogates may flatter DTW matrices, whose non-metric quirks could activate odd-cycle vertices more often); if it fails, the architecture degrades gracefully — the Benders core finish simply does more work, and the LR bound remains valid regardless.
