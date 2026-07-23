# Total Unimodularity of the k-Medoids LP Relaxation

> **Freshness (2026-07-23): CURRENT for formulation/provenance and §8 computational verdicts through `85eabcd`; §§3.2–3.3 and 5–6 retain historical proposals corrected or superseded by §8 and the implemented LR-core chain (`fd49ff4`, `8375aef`, `789cc52`); PLAN R2-D15 owns current line-by-line code conformance.**
>
> **§8 (2026-07-06, corrected 2026-07-23) consolidates the computational verification of §§2–3 and the implemented exact solver ("LR-core").** It **confirms** the TU boundary (§2.2), **strengthens** the block-TU results to all N (§2.8 → §8.2), **FALSIFIES** the half-integrality claim (§3.3 → §8.1), and records the current Kelley/subgradient root, tolerance-guarded reduced-cost fixing, and capped y-only branch-and-bound finish (§8.3–8.5). `docs/sources/lr-core-derivation.md` is the tracked 2026-07-06 derivation; its implementation roadmap is historical where §8 and the current code supersede it.

## 1. The DTWC++ Formulation

The k-medoids clustering problem in DTWC++ is formulated as a **Binary Integer Program**: a diagonal specialization of the classical p-median MILP using **Balinski linking inequalities** (Balinski 1965, ref. 1; ReVelle and Swain 1970, ref. 26).

> **Related histories, different formulations.** Vinod (1969, ref. 27) is an early statistics-side integer-programming treatment of partitional clustering, but it is not evidence for the exact constraint matrix analysed here. Section 2 depends on DTWC++'s p-median variables: the medoid indicator is the diagonal `A[i,i]`, and each linking row is `A[i,j] <= A[i,i]`. Cite Vinod for clustering history; cite Balinski and ReVelle-Swain for this p-median formulation.

**Decision variables.** A p x p binary matrix A where:
- `A[i,j] = 1` if point j is assigned to the cluster whose medoid is point i
- `A[i,i] = 1` if point i is a medoid (diagonal entries)

**Objective.** Minimize total assignment cost:

```
min  sum_{i,j} D[i,j] * A[i,j]
```

**Constraints:**

1. **Cardinality** (1 equality): `sum_i A[i,i] = k` -- exactly k medoids
2. **Assignment** (p equalities): `sum_i A[i,j] = 1  for all j` -- each point in exactly one cluster
3. **Linking** (p(p-1) inequalities): `A[i,j] <= A[i,i]  for all i != j` -- assign only to open medoids

**Implementation:** `dtwc/mip/mip_Highs.cpp` lines 56-130, `dtwc/mip/mip_Gurobi.cpp` lines 36-58.

---

## 2. Total Unimodularity Analysis

### 2.1 Definition

A matrix M is **totally unimodular (TU)** if every square submatrix has determinant in {-1, 0, +1}. If the constraint matrix is TU, then for any integral RHS vector b, every vertex of {x : Mx <= b, x >= 0} is integral -- meaning the LP relaxation always yields integer solutions.

### 2.2 Main Theorem

**Theorem.** The p-median constraint matrix is **totally unimodular for p <= 2** and **not totally unimodular for p >= 3**.

The proof is constructive: we exhibit a 6x6 submatrix with determinant -2 for any p >= 3, and verify all submatrix determinants are in {-1, 0, +1} for p <= 2.

### 2.3 The Constraint Matrix for p = 3

For p = 3, the constraint matrix has 10 rows x 9 columns. Columns correspond to variables A[i,j] at flat index i + 3j (matching the code in `mip_Gurobi.cpp` lines 39-58):

```
         A[0,0] A[1,0] A[2,0] A[0,1] A[1,1] A[2,1] A[0,2] A[1,2] A[2,2]   RHS  Type
Card:      1      0      0      0      1      0      0      0      1     = k
Ass(0):    1      1      1      0      0      0      0      0      0     = 1
Ass(1):    0      0      0      1      1      1      0      0      0     = 1
Ass(2):    0      0      0      0      0      0      1      1      1     = 1
L(0,1):   -1      0      0      1      0      0      0      0      0     <= 0
L(0,2):   -1      0      0      0      0      0      1      0      0     <= 0
L(1,0):    0      1      0      0     -1      0      0      0      0     <= 0
L(1,2):    0      0      0      0     -1      0      0      1      0     <= 0
L(2,0):    0      0      1      0      0      0      0      0     -1     <= 0
L(2,1):    0      0      0      0      0      1      0      0     -1     <= 0
```

where L(i,j) denotes the linking constraint A[i,j] - A[i,i] <= 0.

### 2.4 Proof: Constructive Violating Submatrix (p >= 3)

Consider the **3-cycle** among facilities: 0 -> 1 -> 2 -> 0 (facility 0 "assigns to" customer 1, facility 1 to customer 2, facility 2 to customer 0). Select 6 rows and 6 columns:

**Rows:** L(0,1), Ass(1), L(1,2), Ass(2), L(2,0), Ass(0)
**Columns:** A[0,0], A[0,1], A[1,1], A[1,2], A[2,2], A[2,0]

The resulting 6x6 submatrix M is:

```
              A[0,0]  A[0,1]  A[1,1]  A[1,2]  A[2,2]  A[2,0]
L(0,1):         -1      1       0       0       0       0
Ass(1):          0      1       1       0       0       0
L(1,2):          0      0      -1       1       0       0
Ass(2):          0      0       0       1       1       0
L(2,0):          0      0       0       0      -1       1
Ass(0):          1      0       0       0       0       1
```

**Determinant computation.** Expand along column 0 (only two nonzeros: M[0,0] = -1 and M[5,0] = 1):

```
det(M) = (-1) * (-1)^{0+0} * det(M_{00}) + 1 * (-1)^{5+0} * det(M_{50})
```

**M_{00}** (delete row 0, col 0) is **upper triangular**:

```
1   1   0   0   0
0  -1   1   0   0
0   0   1   1   0
0   0   0  -1   1
0   0   0   0   1
```

det(M_{00}) = 1 * (-1) * 1 * (-1) * 1 = **1**

**M_{50}** (delete row 5, col 0) is **lower triangular**:

```
1   0   0   0   0
1   1   0   0   0
0  -1   1   0   0
0   0   1   1   0
0   0   0  -1   1
```

det(M_{50}) = 1 * 1 * 1 * 1 * 1 = **1**

Therefore:

```
det(M) = (-1)(+1)(1) + (1)(-1)(1) = -1 - 1 = -2
```

Since **|det(M)| = 2 > 1**, the constraint matrix is **not totally unimodular**. **QED.**

### 2.5 General Theorem: The Odd-Cycle Determinant Formula

The 3-cycle construction generalises to arbitrary cycle lengths.

**Theorem.** For an n-cycle sigma: 0 -> 1 -> 2 -> ... -> (n-1) -> 0 among facilities, the 2n x 2n submatrix formed by rows {L(k, k+1 mod n), Ass(k+1 mod n) : k = 0,...,n-1} and columns {A[k,k], A[k, k+1 mod n] : k = 0,...,n-1} has determinant:

```
det = (-1)^n - 1
```

**Proof.** Order the rows as L(0,1), Ass(1), L(1,2), Ass(2), ..., L(n-1,0), Ass(0) and the columns as A[0,0], A[0,1], A[1,1], A[1,2], ..., A[n-1,n-1], A[n-1,0]. The matrix has the block-circulant structure:

```
Row 2k   = L(k, k+1):  entry -1 at col 2k, entry +1 at col 2k+1
Row 2k+1 = Ass(k+1):   entry +1 at col 2k+1, entry +1 at col 2(k+1) mod 2n
```

Expand along column 0 (two nonzeros: row 0 has -1, row 2n-1 has +1):

```
det(M) = (-1)·det(M_{00}) + (-1)^{2n-1}·det(M_{2n-1,0})
       = -det(M_{00}) - det(M_{2n-1,0})
```

M_{00} (delete row 0, col 0) is upper triangular with diagonal [1, -1, 1, -1, ..., 1] (2n-1 entries, n-1 negative). Thus det(M_{00}) = (-1)^{n-1}.

M_{2n-1,0} (delete last row, col 0) is lower triangular with all-ones diagonal. Thus det(M_{2n-1,0}) = 1.

Therefore:

```
det(M) = -(-1)^{n-1} - 1 = (-1)^n - 1
```

**Corollary.**
- **n odd:** |det| = |-1 - 1| = 2. TU violated.
- **n even:** det = 1 - 1 = 0. No violation from this submatrix.

**This constructed cycle family violates total unimodularity exactly when its
length is odd.** The determinant calculation supplies a violating minor and
the p >= 3 boundary; it is not a converse classification of every possible
non-TU minor.

For any p >= 3, the 3-cycle (smallest odd cycle) embeds in the constraint matrix, giving a 6x6 submatrix with det = -2. For p = 2, only the 2-cycle exists (even), so no cycle-based violation arises.

### 2.6 Proof: TU for p = 2

For p = 2, the constraint matrix has 5 rows x 4 columns:

```
         A[0,0]  A[1,0]  A[0,1]  A[1,1]   RHS   Type
Card:       1       0       0       1     = k
Ass(0):     1       1       0       0     = 1
Ass(1):     0       0       1       1     = 1
L(0,1):    -1       0       1       0     <= 0
L(1,0):     0       1       0      -1     <= 0
```

**Claim:** This matrix is TU.

**Structural argument.** The only possible cycle among 2 facilities is the 2-cycle 0 -> 1 -> 0 (length n = 2, even). By the general formula (Section 2.5), the corresponding 4x4 submatrix has det = (-1)^2 - 1 = 0, which does not violate TU. Since no odd cycle exists, the Ghouila-Houri parity obstruction cannot arise.

**Computational verification.** All square submatrices (C(5,1)·C(4,1) + C(5,2)·C(4,2) + C(5,3)·C(4,3) + C(5,4)·C(4,4) = 20 + 60 + 40 + 5 = **125** submatrices) have determinants in {-1, 0, +1}. Verified by exhaustive enumeration.

### 2.7 Ghouila-Houri Characterisation

**Theorem (Ghouila-Houri, 1962).** A {0, ±1}-matrix A is totally unimodular if and only if for every subset R of rows, there exists a partition R = R1 ∪ R2 such that for every column j:

```
|sum_{i in R1} a_{ij} - sum_{i in R2} a_{ij}| <= 1
```

**Application to the 3-cycle.** Take R = {L(0,1), Ass(1), L(1,2), Ass(2), L(2,0), Ass(0)}, the 6 rows of the violating submatrix. Assign signs s1,...,s6 in {+1, -1} to these rows respectively (R1 gets +1, R2 gets -1):

```
s1 = sign of L(0,1),   s2 = sign of Ass(1),   s3 = sign of L(1,2)
s4 = sign of Ass(2),   s5 = sign of L(2,0),   s6 = sign of Ass(0)
```

The GH condition requires |signed column sum| <= 1 for ALL 9 columns. The 3 non-cycle columns (A[1,0], A[2,1], A[0,2]) each have exactly 1 nonzero in R, so their signed sums are ±1 (automatically satisfied). The binding constraints come from the 6 cycle columns (each having exactly 2 nonzeros in R):

```
Col A[0,0]:  -s1 + s6   requires  s1 = s6
Col A[0,1]:   s1 + s2   requires  s1 = -s2
Col A[1,1]:   s2 - s3   requires  s2 = s3
Col A[1,2]:   s3 + s4   requires  s3 = -s4
Col A[2,2]:   s4 - s5   requires  s4 = s5
Col A[2,0]:   s5 + s6   requires  s5 = -s6
```

Chaining:  s2 = -s1,  s3 = s2 = -s1,  s4 = -s3 = s1,  s5 = s4 = s1,  s6 = -s5 = -s1.

But the first constraint requires s1 = s6 = -s1. Since s1 in {+1, -1}, this is a **contradiction**.

**Structural explanation.** Each linking row L(i, sigma(i)) forces A[i,i]'s diagonal column to have opposite sign to the assignment row Ass(sigma(i)). The assignment row Ass(sigma(i)) in turn forces A[sigma(i), sigma(i)]'s column to have opposite sign to L(sigma(i), sigma^2(i)). Following the cycle creates a chain:

```
s_{L(0,1)} -> s_{Ass(1)} -> s_{L(1,2)} -> s_{Ass(2)} -> s_{L(2,0)} -> s_{Ass(0)} -> s_{L(0,1)}
```

The cycle has 2n constraints: n from off-diagonal columns (which flip the sign) and n from diagonal columns (which preserve the sign). After traversing the full cycle, the sign has been flipped n times. For **odd n**, the sign returns negated -- a parity contradiction. For **even n**, the sign returns unchanged -- consistent.

This is the repository's **constructed odd-cycle obstruction**. It is related
to odd-cycle structures studied in facility-location polytopes, but this
derivation does not import a converse characterization from those different
formulations.

### 2.8 Structural Decomposition: What Breaks TU

A critical finding is that the **cardinality constraint is irrelevant** to the TU failure. The violation arises purely from the interaction between assignment and linking constraints.

| Constraint Subset           | TU?     | Reason |
|-----------------------------|---------|--------|
| Assignment only             | **Yes** | Bipartite graph incidence matrix (Hoffman & Kruskal, 1956) |
| Linking only                | **Yes** | Node-arc incidence matrix of a directed graph (each row has one +1 and one -1; each linking constraint is an arc from A[i,j] to A[i,i]) |
| Cardinality + Assignment    | **Yes** | {0,1} matrix; each column has at most 2 nonzeros; the row-graph is a **star** (all edges join the cardinality row to some Ass(i)), hence bipartite — TU for **all N** (Ghouila-Houri proof, §8.2; was only "verified for p ≤ 4") |
| Cardinality + Linking       | **Yes** | Cardinality row adds +1 on diagonals already carrying −1 in linking; a Ghouila-Houri alternating-sign split keeps every signed column sum in {−1, 0, 1} — TU for **all N** (§8.2; was only "verified for p ≤ 4") |
| **Assignment + Linking**    | **No**  | Same 6x6 cycle submatrix with det = -2 (cardinality row not used!) |
| Full (all three)            | **No**  | Inherited from assignment + linking |

**Why the interaction breaks TU:** Each diagonal column A[i,i] has coefficient:
- **+1** in the assignment row Ass(i) (bipartite incidence block)
- **-1** in (p-1) linking rows L(i,j) for j != i (network block)

In isolation, each block is TU. But combined, diagonal columns carry opposite
signs from the two blocks. For the odd-cycle row/column family constructed in
§2.4, the sign conflicts cannot be resolved by any Ghouila-Houri partition.

### 2.9 Sufficient Conditions Checklist

Every entry of the constraint matrix is in {-1, 0, +1}. The standard sufficient conditions for TU of such matrices are:

| Sufficient Condition | Satisfied? | Details |
|---------------------|-----------|---------|
| Each column has at most 2 nonzeros | **No** | Diagonal column A[i,i] has 1 (cardinality) + 1 (assignment) + (p-1) (linking) = **p+1 nonzeros** |
| Network matrix | **No** | A network matrix has at most one +1 and one -1 per row AND per column. Assignment rows have p entries of +1 |
| Consecutive-ones property | **No** | The +1 entries in column A[i,i] (cardinality, assignment, zero, ..., zero) are not consecutive with the -1 entries (linking rows) |
| Balanced / no-odd-hole route | **Not established** | The constructed 3-cycle minor proves non-TU; no complete balanced-matrix characterization was derived here |

### 2.10 Connection to Seymour's Decomposition

**Seymour's Theorem (1980).** Every totally unimodular matrix can be constructed from network matrices, their transposes, and two specific 5x5 matrices (B1, B2) via 1-sum, 2-sum, and 3-sum operations.

The assignment and linking blocks are individually TU, while their combination
through shared diagonal columns contains the constructed determinant-2 minor.
No explicit Seymour k-sum decomposition certificate—or proof that odd cycles
classify every obstruction—was produced here; this section is structural
intuition, not an additional theorem.

### 2.11 Constructive Fractional Example

For p = 3, k = 1, with a uniform distance matrix D[i,j] = 1 for i != j, D[i,i] = 0:

The LP relaxation admits the feasible point A[i,j] = 1/3 for all i,j
(cardinality: 3 * 1/3 = 1; assignment: 3 * 1/3 = 1; linking:
1/3 <= 1/3), with objective value 2. The integer optimum also equals 2 (any
single medoid).

This point is the barycenter of the three integer optima, not a fractional
extreme point, so it proves neither non-integrality nor an integrality gap.
Section 8.1 records the non-degenerate vertex experiments that do establish
fractional optima.

---

## 3. Known Integrality Results

### 3.1 Integrality Gap

> **Historical claim, not current evidence.** An earlier draft quoted a
> roughly 2-to-2.675 numeric window for metric k-median. That paragraph
> conflated approximation ratios with the LP relaxation's integrality gap and
> misattributed the upper number. No numeric metric-gap window is relied on
> here. PLAN R2 must verify a primary polyhedral source before reinstating one.

Standard DTW violates the triangle inequality, so even a correctly sourced
metric result would not transfer automatically to DTWC++ distance matrices.

### 3.2 When Is the LP Naturally Integer?

This subsection preserves a **historical proposal** while separating it from
the verified result in §8.1:

1. Structured, well-separated data can produce integral roots. The current
   evidence is computational: 250/250 clustered surrogate instances in §8.1.
   The Awasthi et al. separation theorem has not yet been shown here to use
   exactly DTWC++'s formulation and assumptions.
2. There is no verified small-N threshold. Uniform-random roots became *less*
   often integral as N increased in the registered experiment.
3. Once the medoid indicators `y_i = A[i,i]` are fixed, there is no coupled
   assignment problem to solve: each point independently chooses its nearest
   open medoid in O(Nk).
4. Odd cycles explain the constructive TU obstruction in §2, but the earlier
   universal support/vertex characterization was imported from a different
   facility-location polytope. It is not used as a theorem for this
   cardinality-constrained formulation.

### 3.3 Half-Integrality and Odd-Cycle Extreme Points

> **FALSIFIED 2026-07-06 — see §8.1.** The historical proposal that
> fractional vertices are generally half-integral or universally described by
> one odd cycle does **not** hold for this cardinality-constrained formulation.
> Only **63.0%** of 664 fractional components fell in [0.45, 0.55], and values
> 1/4, 1/3, and 3/4 occurred. Baiou-Barahona's cited half-integrality result
> concerns a different facility-location polytope. Odd-cycle cuts therefore
> cannot be the sole exact closer; branching is mandatory.

The narrower TU fact remains verified: for p = 3, exhaustive enumeration of
the 10x9 constraint matrix finds exactly two violating square submatrices, both
6x6 with determinant magnitude 2, corresponding to the two directed
three-cycles. This matrix obstruction does not imply a universal description
of every fractional vertex.

### 3.4 Role of the Metric Property

**Important caveat:** Standard DTW does NOT satisfy the triangle inequality
and is therefore not a metric (Marteau, 2009; Jain, 2018). A metric-only
integrality result would not directly apply to DTWC++ distance matrices.

However, several factors mitigate this in practice:
- DTW distances are non-negative and symmetric, which is sufficient for the MIP formulation to be well-defined
- The LP relaxation quality depends on the specific distance matrix, not just whether it is formally metric
- For truly metric distances (e.g., Euclidean on equal-length series), the full theory applies

The registered evidence in §8.1 is deliberately regime-specific; it does not
promote a clustered-surrogate observation into a theorem about all DTW data.

---

## 4. Polyhedral Analysis

### 4.1 The p-Median Polytope

The **integer hull** P_PM = conv{feasible integer solutions} is strictly contained in the **LP relaxation** Q_PM = {A in [0,1]^{p x p} : constraints 1-3}. The gap P_PM != Q_PM is the source of fractional optima.

### 4.2 Facet-Defining Inequalities

The standard formulation already includes facets:
- Non-negativity: A[i,j] >= 0
- Linking: A[i,j] <= A[i,i] (facet-defining for p >= 3)

**Historical cut proposals (not implemented in DTWC++):**

1. **{0, 1/2}-Chvatal-Gomory cuts** (Caprara and Fischetti, 1996).
   Their applicability and separation oracle for this exact formulation were
   not independently established here. The §8.1 counterexamples prove that a
   half-integral cut family cannot close every root.

2. **Strengthened linking (multi-assignment) cuts**:
   ```
   sum_{j in S} A[i,j] <= (|S| - 1) * A[i,i] + 1    for subsets S, i not in S
   ```
   This candidate inequality has not been validated against a non-degenerate
   oracle in this repository and is not part of the live solver.

3. **Odd-hole inequalities** were proposed from the earlier
   Baiou-Barahona analogy. For a candidate odd cycle C:
   ```
   sum_{(i,j) in C} A[i,j] <= (|C| - 1) / 2
   ```

   The formulation-specific validity and separation details remain
   unverified; this is a research lead, not current solver behavior.

---

## 5. Practical LP-Based Solving Strategies

### 5.1 Recommended Tiered Strategy

> **Historical proposal, superseded by §8 and `789cc52`.** The original
> three-tier design formed the compact LP, attempted cuts, then branched. Its
> useful ideas were the FastPAM incumbent and y-only decisions; its fatal cost
> was materializing N² assignment variables. The live `Method::LRCore` obtains
> the same LP bound from the Lagrangian dual, applies reduced-cost fixing, and
> finishes with matrix-free y-only branch-and-bound.

### 5.2 Cutting Plane Strategy

This was a **historical proposal**, never an implemented route. The registered
half-integrality falsification shows why odd-cycle cuts cannot serve as the
exact closer. PLAN keeps that cut family deprioritized unless new
non-degenerate measurements overturn the recorded evidence.

### 5.3 Lagrangian Relaxation

> **LIVE ENGINE — full derivation and implementation status in §8.3.** Because
> the inner polytope (Cardinality + Linking) is TU for all N (§8.2),
> Geoffrion's integrality-property theorem makes this dual attain the full LP
> bound without forming the N²-column LP.

The most effective Lagrangian relaxation dualizes the **assignment constraints** `sum_i A[i,j] = 1`:

```
L(mu) = min  sum_{i,j} (D[i,j] - mu_j) * A[i,j] + sum_j mu_j
        s.t. sum_i A[i,i] = k
             A[i,j] <= A[i,i]  for all i != j
             A[i,j] in {0,1}
```

After dualization, the problem decomposes **by facility**: for each potential
medoid i, independently decide whether to open it and which points to attract,
based on the modified costs D[i,j] - mu_j. This yields p independent
subproblems, each O(p), plus selection of the k best facilities. Geoffrion's
theorem (1974), applied to the integral inner polytope proved in §8.2,
establishes equality between this Lagrangian dual optimum and the compact LP
bound. The live solver uses stabilized Kelley optimization when HiGHS is
available and retains a solver-free subgradient implementation.

### 5.4 Benders Decomposition

This section describes the separate, older `Method::MIP` Benders route; it is
not the exact finish used by `Method::LRCore`.

**Implemented master problem** (N binary facility variables `y_i` plus N
continuous assignment-cost surrogates `theta_j`):
```
min  sum_j theta_j
s.t. sum_i y_i = k
     theta_j + sum_i max(0, d^t_j - D[i,j]) y_i >= d^t_j
         for each generated incumbent-distance cut t and point j
     y_i in {0,1}
     theta_j >= 0
```

**Subproblem** (given fixed y): assign each point to its nearest open medoid in
O(Nk); there are no remaining coupling constraints and no LP solve is needed.
For a current medoid set at round t, `d^t_j` is point j's nearest-open cost.
The implemented disaggregated coefficient generation nevertheless scans all
candidate rows and costs O(N²) per Benders round
(`docs/sources/lr-core-derivation.md:77-78`).

Duran-Mateluna, Ales, and Elloumi (2023) report solving instances with up to
238,025 clients and potential sites using their two-phase Benders approach.
That publisher claim is provenance, not a scale measurement of DTWC++'s
round-based implementation.

### 5.5 LP Rounding (historical proposal)

These ideas are retained for research provenance but are not a DTWC++ solver
route:

**Deterministic rounding:**
1. Solve LP relaxation
2. Set A[i,i] = 1 for the k largest diagonal values
3. Assign each point to nearest selected medoid (exact, TU)
4. Note: this simple rounding has no proven constant-factor guarantee. For formal guarantees, use the filtering/rounding technique of Charikar et al. (1999) which achieves ~6.67-approximation, or the dependent rounding of Li and Svensson (2016) for ~2.73-approximation.

**Iterative rounding:**
1. Solve LP. Fix variables at 0 or 1.
2. Re-solve reduced LP. Repeat until all integral.
3. Re-solve until no more variables can be fixed. No current evidence supports
   a general claim that many variables will be integral.

### 5.6 Warm-Starting from PAM

This is implemented, not merely recommended. `MIPSettings::warm_start` defaults
to true (`dtwc/Problem.hpp:70`), and both compact solver backends run FastPAM
and pass the resulting complete assignment:

- HiGHS: `dtwc/mip/mip_Highs.cpp:165-182`
- Gurobi: `dtwc/mip/mip_Gurobi.cpp:92-105`

The operational benefits are:
- Provides tight upper bound
- Gives the solver a feasible incumbent before search
- Can reduce branching, subject to the instance and backend

---

## 6. Practical Solver Tuning & Implementation

### 6.1 Live MIP Start from PAM

The warm starts described in §5.6 are live in both compact MIP backends. They
are guarded by the public `MIPSettings::warm_start` option and use each
backend's actual matrix indexing.

### 6.2 Gurobi Parameter Tuning

The prior recommendations are now defaults in `MIPSettings`:
`numeric_focus = 1`, `mip_focus = 2`, and `warm_start = true`
(`dtwc/Problem.hpp:68-75`). Gurobi applies those settings at
`dtwc/mip/mip_Gurobi.cpp:83-105`. Any performance effect remains
instance-dependent; no universal multiplier is claimed.

### 6.3 HiGHS Tuning

The live HiGHS backend applies the configured relative gap, optional time
limit, output flag, and FastPAM start (`dtwc/mip/mip_Highs.cpp:154-182`).
Earlier proposed option names were not verified against the supported HiGHS
API and have been removed. The diagonal entries identify distinct candidate
points, so an index-order constraint on them would change the feasible medoid
sets rather than break a harmless permutation symmetry.

### 6.4 LP-First Mode

This was a **historical proposal** for the compact model. It was superseded by
the LR root, which obtains the compact LP bound without materializing its N²
columns. The explicit compact MIP remains an exact backend route.

### 6.5 Report LP Bound

This historical compact-LP proposal is now realized differently:
`LagrangianResult` reports lower bound, upper bound, relative gap, and
certificate state. `lagrangian_root_exact` returns at the root when that
registered tolerance closes.

### 6.6 Benders Decomposition (for N > 200)

The live `benders.cpp` path belongs to `Method::MIP`: `Problem::cluster_by_mip`
selects it when `benders == "on"`, or when `benders == "auto"` and N > 200
(`dtwc/Problem.cpp:974-983`). It has N binary facility variables plus N
continuous `theta_j` assignment-cost surrogates. Given y, assignment is an
O(Nk) nearest-open scan; constructing the current disaggregated cut
coefficients scans all N candidate rows, making each full round O(N²).

**Algorithm:**

```text
1. Initialize with a FastPAM incumbent and theta_j values from its assignments.
2. Master: minimize sum_j theta_j subject to sum_i y_i = k, accumulated cuts,
   binary y_i, and continuous non-negative theta_j.
3. Solve master -> y*, theta*. Set LB = sum_j theta*_j.
4. For every point j, compute d_j = min_{i:y*_i=1} D[i,j]; set
   Z = sum_j d_j and update the incumbent.
5. If the registered absolute/relative bound tolerance closes, stop.
   Otherwise, for each underestimated point add:
   theta_j + sum_i max(0, d_j - D[i,j]) y_i >= d_j.
6. Go to step 2
```

The implementation resolves a master and adds rows between rounds. It does not
use a single-tree lazy-constraint callback. No universal crossover or runtime
claim is registered for this legacy route.

### 6.7 Constraint Formulation Notes

**Keep disaggregated linking constraints.** The compact formulation uses
`A[i,j] <= A[i,i]` for every off-diagonal assignment. A valid aggregated
alternative is `sum_{j != i} A[i,j] <= (N-1) A[i,i]` (or
`sum_j A[i,j] <= N A[i,i]` when the diagonal term is included).
The aggregate is weaker in the LP relaxation; the earlier version that summed
all j but retained coefficient N-1 was not valid as written.

### 6.8 Historical runtime table

The earlier size-to-runtime table contained unregistered predictions on a
shared machine and is not evidence. Current route selection is semantic:
`Method::LRCore` is the matrix-free exact LR path; `Method::MIP` retains the
solver-backed compact/Benders policy; measured results belong in dated
run-logs with preregistered bands.

---

## 7. Summary of Key Results

| Question | Answer |
|----------|--------|
| Is the constraint matrix TU? | **No** for p >= 3. **Yes** for p <= 2. Constructive proof: 6x6 submatrix with det = -2 from any 3-cycle (Section 2.4). |
| What breaks TU? | The assignment+linking interaction contains the explicit odd-cycle determinant-2 family (§2.4–2.7); cardinality is absent from that violating minor. No converse classification of all minors is asserted. |
| Precise TU boundary? | p = 2 is TU (exhaustive verification), p >= 3 is not. For p = 3, exhaustive enumeration finds exactly 2 violating submatrices (the two directed 3-cycles, both 6x6 with det = ±2). |
| General formula? | For the constructed n-cycle family, det = (-1)^n - 1, so its odd members violate TU (§2.5). |
| Integrality gap (metric D)? | No numeric window is asserted here; the earlier approximation-ratio attribution was not an LP-gap proof. DTW is not metric, so metric-only results would not transfer automatically. |
| When is the LP naturally integer? | This is data-regime-dependent. Registered clustered surrogates were 250/250 integral; uniform-random roots fell to 26% integral at N=20 (§8.1). |
| What happens after y is fixed? | Each point independently chooses its nearest open medoid in O(Nk); there is no remaining coupled LP. |
| Live decomposition? | LR-core: Kelley/subgradient Lagrangian root, reduced-cost fixing, then matrix-free y-only branch-and-bound. The older Benders code remains a separate `Method::MIP` route. |
| How should fractional values be interpreted? | They witness a genuine LP/IP gap in the registered non-degenerate instances, but are **not generally 1/2** and do not admit the earlier universal odd-cycle description (§3.3 / §8.1). |
| Practical strategy? | **LR-core** (§8.3) obtains the LP bound without x-space, fixes facilities, and closes any remaining integrality gap by branching on y. |

### k-Medoids constraint matrix is NOT totally unimodular
- TU boundary is p=3. For p≤2, the matrix IS TU.
- The constructed odd-cycle family breaks TU (det = (-1)^n - 1 for that family).
- With a fixed medoid set, assignment is a direct nearest-open scan.

---


## 8. Computational Verification (2026-07-06) and the LR-core Solver

This section records an independent computational re-derivation of the claims
above and the exact-solver design that follows from them. The tracked analytical
derivation is `docs/sources/lr-core-derivation.md`; its analytical work feeds
this durable digest, while its historical Benders-finish roadmap is superseded
by the implementation record below. Method: hand re-derivation plus scratch
experiments (NumPy + SciPy
HiGHS vertex LPs, brute-force IP as oracle); every decisive band was registered
before its run. Tags: **[confirmed]** = re-derived, enumerated, or implemented
with named evidence; **FALSIFIED** = a registered prediction that failed and is
kept visible.

### 8.1 Verification verdict — what held, what was corrected

| Earlier claim | Verdict | Evidence |
|---|---|---|
| §2.2 TU for N≤2, not for N≥3; 6×6 3-cycle det = −2 | **[confirmed]** | Hand cofactor expansion (det = −1 − 1 = −2) + exhaustive enumeration: N=2 → 0/125 submatrix violations; N=3 → exactly 2/92,377, both 6×6 with dets {−2, +2} (the two directed 3-cycles) |
| §2.5 odd-cycle formula det = (−1)ⁿ − 1 | **[confirmed]** | Hand-verified n=2 (0) and n=3 (−2); general n via the block-band / two-triangular-minor argument |
| §2.8 Card+Assignment and Card+Linking TU ("verified for p ≤ 4") | **STRENGTHENED → all N** | Ghouila-Houri proofs, §8.2 below |
| §3.3 fractional vertices "typically half-integral" | **FALSIFIED** | Only 63.0% of 664 fractional components in [0.45, 0.55]; observed 1/4, 1/3, 3/4. Cardinality row breaks half-integrality ⇒ branching mandatory |
| §2.11 "x_ij = 1/3 is a fractional solution" | **Clarified** | It is the *barycenter* of the 3 integer optima, not an extreme point — proves nothing about non-integrality. Genuine fractional vertices exist (35 at N=10, max gap 13.718%) |
| §3.2 "LP integer ~80–90%" | **Regime-dependent** | Uniform-random non-metric D: 82% integral at N=10 but **26% at N=20** (decays with N). Clustered non-metric D: **100% integral across 250 instances** (N∈{15,30}, k∈{2,3,4}, incl. mis-specified k). ReVelle folklore is a statement about *data*, not the polytope |
| §5.4 "fixed medoids ⇒ transportation problem" | **Overstated** | With y fixed there are *no* coupling constraints; each point independently takes `argmin_{i: y_i=1} D_ij` — an O(Nk) scan. No LP / network simplex / transportation machinery needed |

Consistency check: TU would give LP = IP for every D — a polynomial exact p-median algorithm, contradicting NP-hardness (Kariv & Hakimi 1979). Non-TU for N ≥ 3 is therefore *necessary*.

### 8.2 The two constraint blocks are TU for all N (strengthens §2.8)

Both were previously only "verified computationally for p ≤ 4." Ghouila-Houri (every row subset R admits a ±1 signing with all column sums in {−1,0,1}) proves them for every N:

**Cardinality + Assignment.** Every entry is 0/1. Column x_ii has two ones (Card row, Ass(i)); column x_ij (i≠j) has one (Ass(j)). A 0/1 matrix with ≤ 2 ones per column is the incidence matrix of a graph on the rows; it is TU iff that graph is bipartite (Hoffman-Kruskal). Here every edge joins the single Card row to some Ass(i) — a **star**, trivially bipartite. ∎

**Cardinality + Linking.** Take any row subset R and let `c = 1` when the
Cardinality row is present (sign it +1), otherwise `c = 0`. For each facility i,
let `m_i` selected linking rows touch diagonal column x_ii. Choose their signs
so their sum is 0 when `m_i` is even; when `m_i` is odd, choose sum +1 if
`c = 1`, and either ±1 if `c = 0`. This is always possible by balancing the
counts of + and − signs. The signed diagonal-column sum is therefore
`c - Σs ∈ {-1,0,1}`. Every off-diagonal column occurs in only its one linking
row, so its signed sum is ±1. Ghouila-Houri's condition holds for every R. ∎

**Corollary (load-bearing).** The inner polytope `{ Σ_i y_i = k, x_ij ≤ y_i, 0 ≤ x,y ≤ 1 }` has an all-TU matrix (Cardinality + Linking), hence integral optima. By **Geoffrion's theorem (1974)**, the Lagrangian dual obtained by dualizing the *assignment* equalities attains the full LP-relaxation bound — **without ever forming the N²-column LP**. This is the fact the whole solver rests on. The non-TU interaction is Assignment × Linking through the shared diagonal columns, as witnessed by §2.4's odd-cycle minor; no complete classification of its violating minors is claimed.

### 8.3 The LR-core solver

Architecture: **Kelley root when HiGHS is available (solver-free subgradient
otherwise) → reduced-cost fixing → y-only branch-and-bound.** All hardness
lives in the k-subset choice of y (N binaries); x is implicit (§2.4 (c)).

**8.3.1 Bound engine.** Dualize the assignment equalities with multipliers μ ∈ R^N (free sign):

```
L(μ) = Σ_j μ_j + min { Σ_ij (D_ij − μ_j) x_ij : Σ_i y_i = k, x_ij ≤ y_i, x,y ∈ [0,1] }
```

The inner minimization decomposes per facility. Define the **facility score**

```
ρ_i(μ) = Σ_j min(0, D_ij − μ_j)          (an open facility attracts exactly its profitable points)
```

Under the cardinality constraint, open the k most negative scores, S_k(μ) = argmin-k ρ_i. Then

```
L(μ) = Σ_j μ_j + Σ_{i ∈ S_k(μ)} ρ_i(μ)          — a valid lower bound for every μ.
```

Exact by the §8.2 corollary (Geoffrion). **Subgradient**, computed in the same O(N²) pass:

```
g_j = 1 − Σ_{i ∈ S_k(μ)} 1[D_ij < μ_j]     (= 1 − #chosen medoids that serve j)
```

Maximize L over μ by **subgradient ascent with Polyak steps**, `μ ← μ + t·g`, `t = λ·(UB − L(μ)) / ‖g‖²`, λ ∈ (0,2], UB from FastPAM (already in-repo). *Dimensional check:* μ, ρ carry distance units; g is a dimensionless count — consistent.

**8.3.2 Primal repair + reduced-cost fixing.** Each major iteration, open S_k(μ) and assign every point by row-min scan (O(Nk)) → a feasible incumbent (candidate UB). **Reduced-cost fixing (Beasley-style):** with LB = L(μ) and incumbent UB, a facility i ∉ S_k is *fixed closed* only when forcing it in clears `UB + fix_tol`, i.e.
`LB + (ρ_i − ρ_(k)) > UB + fix_tol`, where
`fix_tol = 1e-9 * (1 + max(|LB|, |UB|))`; facilities in S_k are *fixed open*
by the symmetric swap test with the same tolerance. The margin prevents
independent LB/UB/ρ summations from fixing out a tied optimal facility. The
registered clustered-surrogate fixing result is quantified under P2 below,
not generalized as "almost all."

**8.3.3 Exact finish.** If gap ≤ tol, the repaired/PAM solution is
**certified optimal** and the routine returns. Otherwise
`lagrangian_root_exact` recomputes a root-consistent fixing state, retains the
proven-open facilities, and runs a single matrix-free depth-first
branch-and-bound over the surviving y candidates
(`dtwc/mip/lagrangian_root.cpp:480+`). Its fixed-dual node bound closes the
integrality gap without invoking `benders.cpp`. Branching on y is justified by
§2.4(c), not by the falsified "few fractional y" heuristic. The tree is capped
by `LagrangianParams::max_nodes` (default 2,000,000). If the cap is reached,
the routine emits a loud diagnostic and returns the best incumbent with its
valid root lower bound and `certified_optimal=false`; only a fully explored
tree reports a zero gap and an exact certificate.

**8.3.4 Why this beats the alternatives.**

| Option | Verdict | Reason |
|---|---|---|
| LP-first + branch on fractional y | Subsumed | LR *is* the LP bound, obtained matrix-free at O(N²)/iter instead of solving a 10⁸-column LP |
| Lagrangian + Kelley/subgradient + repair | **Chosen and implemented** | Same bound without x-space; Kelley closes subgradient stalls when HiGHS is present, while the solver-free implementation remains available |
| Column generation / branch-and-price | Rejected | Identical Dantzig-Wolfe bound (same TU blocks), higher constants, stabilized-master machinery |
| Network simplex | Rejected | The only flow substructure is the fixed-y assignment — an O(Nk) scan; nothing flows across the y-choice (that is what the TU failure proves) |

**8.3.5 Cost model (N = 10⁴, k = 20).** The live LR-core interface consumes a
dense `double[N*N]` matrix, so D occupies **800,000,000 bytes** before allocator
overhead. Each subgradient iteration reads 10⁸ doubles (about 0.8 GB), giving
leading-order cost T·N² memory traffic; Kelley additionally solves its small
N-dimensional master. Earlier CPU/H100 millisecond and end-to-end runtime
figures were unregistered predictions, not measurements. They remain
**[inferred]** until a quiet-machine counter/throughput gate records the actual
implementation. Packed-triangular and Float32 storage elsewhere in the
repository do not change this dense-double LR-core contract.

### 8.4 Previous own-solver attempts — keep killed ideas killed

All removed in `f7064b3` ("removed specialised solvers", 2023-12-07). They failed for one architectural reason with three faces: they tried to solve the **N²-variable compact LP explicitly**.

- **Dense tableau primal simplex + Gomory cuts** (Oct 2023, `Simplex.cpp`): tableau ~2N⁴ doubles (N=316 ⇒ ~160 GB); basis found by scanning for unit columns; textbook Gomory cuts numerically fragile, glacial. Commit `5968fb9` "Simplex table takes huge space!" says it all.
- **Sparse tableau + dual simplex graft** (`SparseSimplex`): still tableau-updating (not revised + LU), sparsity dies after a few pivots; correctness never achieved (`1ececcd` "still not solving").
- **First-order LP (OSQP / ADMM / CG):** ~1e-3 accuracy — cannot certify vertex optimality or integrality. Wrong tool for exact certification.

**Lesson (durable):** never materialize the x-space LP for the large-N route.
Keep x implicit. The 2026 Benders and LR-core implementations are separate
y-space attempts: `Method::MIP` retains the round-based N-theta master, while
`Method::LRCore` supplies its own Lagrangian root bound, fixing, and y-only
tree. LR-core is not the bound engine for `benders.cpp`.

### 8.5 Registered predictions and numbers ledger

**Original registered predictions and current verdicts:**

- **P1 (root exactness): PARTIALLY TESTED, not UCR-wide.** The registered
  implementation gate closed the root gap to 0.1% on **40/40 clustered
  surrogate instances**, and the primal recovered the brute-force optimum
  40/40 (`fd49ff4`; plan archive §Task 4.1). This does not establish the
  original ≥90% UCR claim.
- **P2 (fixing power): FALSIFIED as a universal floor.** Across 36 qualifying
  clustered instances, mean elimination was **80.3%**, minimum **73.3%**, and
  **77.8%** reached 80%
  (`.claude/PLAN-archive-2026-07-20-phases0-9.md:428-434`).
  The failed band was retained without rescue-tuning.
- **P3 remains unmeasured.** The bytes/STREAM and N=10⁴ throughput prediction
  requires a quiet-machine benchmark and is not evidence for current runtime.
- **P4 remains an unmeasured cut-family guard.** The observed 1/4, 1/3, and 3/4
  vertices already disprove half-integrality and justify an exact branching
  path, but no percentage of gap closed by odd-cycle cuts was measured.

**Numbers ledger (registered band → verbatim verdict, 2026-07-06):**

| Band (registered before run) | Result | Verdict |
|---|---|---|
| B1: N=2, 0 TU violations | 0 | **PASS** |
| B2: N=3, exactly 2 violations, 6×6, \|det\|=2 | 2, sizes [6], dets {−2,+2} | **PASS** |
| B3: §2.4 submatrix det = −2 | −2 | **PASS** |
| B4: uniform N=10,k=3: ≥50% integral, mean gap ≤1%, max ≤5% | 82% integral, mean 0.732%, max **13.718%** | **FAIL (max-gap)** |
| B5: clustered N=15,k=3: ≥90% integral | 100/100 | **PASS** |
| B6: fractional components ≥90% in [0.45,0.55] | **63.0%** of 664; incl. 1/4,1/3,3/4 | **FAIL** (half-integrality falsified) |
| A1: fractional instances ≤4 fractional y (median) | median **6**, max 8 (of N=10) | **FAIL** |
| A2: ≥20% of fractional vertices zero-gap | **0/35** | **FAIL** (fractional ⇒ real gap) |
| A3: uniform N=20,k=4 ≥60% integral | **26%** | **FAIL** (integrality decays with N) |
| A4/A5/A6: clustered N=30, k∈{3,4,2} ≥{90,50,50}% integral | 100% / 100% / 100% | **PASS** (robust to mis-specified k) |

**Most-likely-wrong claim:** the untested UCR-wide extrapolation in P1.
Clustered-line surrogates may flatter DTW matrices. A UCR-wide registered run
would confirm or falsify it; the LR bound remains valid either way.

**Implementation status [confirmed]:** the root engines, reduced-cost fixing,
and exact y-only branch-and-bound are complete in
`dtwc/mip/lagrangian_root.{hpp,cpp}` through `789cc52`. Correctness was checked
against brute-force IP on non-degenerate N≤14 instances, including roots with
a real integrality gap.

---

## 9. References

1. **Balinski, M.L.** (1965). "Integer Programming: Methods, Uses, Computations." *Management Science* 12(3), 253-313. https://doi.org/10.1287/mnsc.12.3.253. *Introduced the disaggregated linking inequalities `x[i,j] <= x[i,i]` used in the classical p-median MILP; not, by itself, the first complete discrete p-median MILP.*

2. **Kariv, O. and Hakimi, S.L.** (1979). "An algorithmic approach to network location problems." SIAM J. Applied Mathematics 37(3), 539-560. *NP-hardness of p-median.*

3. **Cornuejols, G., Nemhauser, G.L., and Wolsey, L.A.** (1990). "The uncapacitated facility location problem." In Discrete Location Theory, Wiley, 119-171. *Foundational polyhedral study.*

4. **Caprara, A. and Fischetti, M.** (1996). "{0, 1/2}-Chvatal-Gomory cuts." Mathematical Programming 74, 221-235. *Polynomial separation of half-integral cuts.*

5. **Charikar, M., Guha, S., Tardos, E., and Shmoys, D.B.** (1999). "A constant-factor approximation algorithm for the k-median problem." STOC 1999. *LP rounding for k-median.*

6. **Jain, K. and Vazirani, V.** (2001). "Approximation algorithms for metric facility location and k-median problems." JACM 48(2), 274-296. *Primal-dual approximation algorithms for the metric problems; not the source used here for Lagrangian-dual = LP-bound equality.*

7. **Avella, P. and Sassano, A.** (2001). "On the p-median polytope." Mathematical Programming 89, 395-411. *Facet-defining inequalities.*

8. **Arya, V. et al.** (2004). "Local search heuristics for k-median and facility location problems." SIAM J. Computing 33(3), 544-562. *3+eps local search approximation.*

9. **Baiou, M. and Barahona, F.** (2009). "On the integrality of some facility location polytopes." SIAM J. Discrete Mathematics 23(2), 665-679. *Integrality structure for specific facility-location polytopes; its half-integrality results are not transferred to this cardinality-constrained formulation.*

10. **Baiou, M. and Barahona, F.** (2011). "On the p-median polytope and the intersection property." SIAM J. Discrete Mathematics 25(1), 1-20. *Intersection-property study; no universal support/vertex characterization from it is assumed in this record.*

11. **Awasthi, P. et al.** (2015). "Relax, no need to round: integrality of clustering formulations." ITCS 2015. *Separation-based LP exact-recovery result for its stated clustering formulation; applicability to DTWC++'s exact formulation has not been established here.*

12. **Li, S. and Svensson, O.** (2016). "Approximating k-median via pseudo-approximation." SIAM J. Computing 45(2), 530-547. *2.732 approximation.*

13. **Cristian Duran-Mateluna, Zacharie Ales, and Sourour Elloumi.**
(2023). "An efficient Benders decomposition for the p-median problem."
*European Journal of Operational Research* 308(1), 84–96.
https://doi.org/10.1016/j.ejor.2022.11.033. *Two-phase Benders decomposition;
the publisher abstract reports instances with up to 238,025 clients and
potential sites.*

14. **Cohen-Addad, V. et al.** (2022). "Improved approximation for k-median." STOC 2022. *Metric k-median approximation result; not an LP integrality-gap bound and not labeled "current best" here.*

15. **Nemhauser, G.L. and Wolsey, L.A.** (1988). "Integer and Combinatorial Optimization." Wiley. *Textbook: TU, cutting planes, polyhedral theory.*

16. **Hoffman, A.J. and Kruskal, J.B.** (1956). "Integral boundary points of convex polyhedra." In Linear Inequalities and Related Systems, Annals of Math. Studies 38, 223-246. *Foundational result: incidence matrices of bipartite graphs are TU.*

17. **Jain, B.J.** (2018). "Semi-Metrification of the Dynamic Time Warping Distance." arXiv:1808.09964. *DTW violates triangle inequality; proposes semi-metric conversion.*

18. **Marteau, P.F.** (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching." IEEE TPAMI 31(2), 306-318. *Proposes TWED as a proper metric alternative to DTW; documents DTW's metric failures.*

19. **Ghouila-Houri, A.** (1962). "Caracterisation des matrices totalement unimodulaires." Comptes Rendus de l'Academie des Sciences 254, 1192-1194. *Necessary and sufficient condition for TU via row partitioning.*

20. **Seymour, P.D.** (1980). "Decomposition of regular matroids." Journal of Combinatorial Theory, Series B, 28(3), 305-359. *Every TU matrix decomposes via 1/2/3-sums from network matrices.*

21. **Schrijver, A.** (1986). "Theory of Linear and Integer Programming." Wiley. *Standard textbook reference for TU theory, Ghouila-Houri characterisation, sufficient conditions.*

22. **Geoffrion, A.M.** (1974). "Lagrangean relaxation for integer programming." Mathematical Programming Study 2, 82-114. *The integrality-property theorem: when the Lagrangian subproblem has integral optima, its dual bound equals the LP-relaxation bound. Load-bearing for §8.2.*

23. **Cornuejols, G., Fisher, M.L., and Nemhauser, G.L.** (1977). "Location of bank accounts to optimize float: an analytic study of exact and approximate algorithms." Management Science 23(8), 789-810. *Historical assignment-constraint Lagrangian relaxation for a location model; the equality theorem used in §8.2 is attributed separately to Geoffrion.*

24. **Fisher, M.L.** (1981). "The Lagrangian relaxation method for solving integer programming problems." Management Science 27(1), 1-18. *Subgradient optimization and Polyak step practice (reprinted Management Science 50(12), 2004).*

25. **Beasley, J.E.** (1993). "Lagrangean heuristics for location problems." European Journal of Operational Research 65(3), 383-399. *Reduced-cost / Lagrangian variable fixing for facility location — the §8.3.2 fixing test.*

26. **ReVelle, C.S. and Swain, R.W.** (1970). "Central Facilities Location." *Geographical Analysis* 2(1), 30-42. https://doi.org/10.1111/j.1538-4632.1970.tb00142.x. *Classical complete discrete p-median MILP provenance. No empirical integrality-frequency quote is attributed here without a checked primary passage.*

27. **Vinod, H.D.** (1969). "Integer Programming and the Theory of Grouping." *Journal of the American Statistical Association* 64(326), 506-519. https://doi.org/10.1080/01621459.1969.10500990. *Early integer-programming treatment of partitional clustering. Its author-page abstract confirms the n-into-m mutually exclusive grouping formulation and the one-dimensional string-property result for minimizing within-group sums of squares. Historical context only: it is not the source for §1's diagonal p-median/Balinski constraint matrix.*
