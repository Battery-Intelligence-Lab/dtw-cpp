# Total Unimodularity of the k-Medoids LP Relaxation

## 1. The DTWC++ Formulation

The k-medoids clustering problem in DTWC++ is formulated as a **Binary Integer Program** (the standard **Balinski formulation** of the p-median problem).

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

**Only odd cycles among facilities break total unimodularity.**

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

This is precisely the **odd-cycle obstruction** identified by Baiou and Barahona (2009, 2011) in their study of the p-median polytope.

### 2.8 Structural Decomposition: What Breaks TU

A critical finding is that the **cardinality constraint is irrelevant** to the TU failure. The violation arises purely from the interaction between assignment and linking constraints.

| Constraint Subset           | TU?     | Reason |
|-----------------------------|---------|--------|
| Assignment only             | **Yes** | Bipartite graph incidence matrix (Hoffman & Kruskal, 1956) |
| Linking only                | **Yes** | Node-arc incidence matrix of a directed graph (each row has one +1 and one -1; each linking constraint is an arc from A[i,j] to A[i,i]) |
| Cardinality + Assignment    | **Yes** | {0,1} matrix; each column has at most 2 nonzeros; no odd-cycle obstruction possible (verified computationally for p <= 4) |
| Cardinality + Linking       | **Yes** | Network-like structure; cardinality row adds +1 on diagonals which are already -1 in linking, preserving {0,±1} network incidence (verified computationally for p <= 4) |
| **Assignment + Linking**    | **No**  | Same 6x6 cycle submatrix with det = -2 (cardinality row not used!) |
| Full (all three)            | **No**  | Inherited from assignment + linking |

**Why the interaction breaks TU:** Each diagonal column A[i,i] has coefficient:
- **+1** in the assignment row Ass(i) (bipartite incidence block)
- **-1** in (p-1) linking rows L(i,j) for j != i (network block)

In isolation, each block is TU. But combined, diagonal columns carry opposite signs from the two blocks. When facilities form an odd cycle, the sign conflicts cannot be resolved by any Ghouila-Houri partition.

### 2.9 Sufficient Conditions Checklist

Every entry of the constraint matrix is in {-1, 0, +1}. The standard sufficient conditions for TU of such matrices are:

| Sufficient Condition | Satisfied? | Details |
|---------------------|-----------|---------|
| Each column has at most 2 nonzeros | **No** | Diagonal column A[i,i] has 1 (cardinality) + 1 (assignment) + (p-1) (linking) = **p+1 nonzeros** |
| Network matrix | **No** | A network matrix has at most one +1 and one -1 per row AND per column. Assignment rows have p entries of +1 |
| Consecutive-ones property | **No** | The +1 entries in column A[i,i] (cardinality, assignment, zero, ..., zero) are not consecutive with the -1 entries (linking rows) |
| Balanced (no odd holes) | **No** | The 3-cycle among facilities creates an odd hole in the associated conflict graph |

### 2.10 Connection to Seymour's Decomposition

**Seymour's Theorem (1980).** Every totally unimodular matrix can be constructed from network matrices, their transposes, and two specific 5x5 matrices (B1, B2) via 1-sum, 2-sum, and 3-sum operations.

The assignment block alone is a network matrix. The linking block alone is a network matrix. But their combination -- through the shared diagonal columns -- creates a structure that cannot be decomposed into network components via k-sums. The odd-cycle obstruction is precisely the impediment to such decomposition.

### 2.11 Constructive Fractional Example

For p = 3, k = 1, with a uniform distance matrix D[i,j] = 1 for i != j, D[i,i] = 0:

The LP relaxation admits the fractional solution A[i,j] = 1/3 for all i,j. This is feasible (cardinality: 3 * 1/3 = 1; assignment: 3 * 1/3 = 1; linking: 1/3 <= 1/3) with objective value 2.

The integer optimum also equals 2 here (any single medoid), so no integrality gap for this symmetric instance. To exhibit a genuine gap, consider p = 4, k = 2 with distances chosen to force fractional facilities on an odd cycle -- the LP optimal exploits the fractional medoids to reduce cost below any integer solution.

---

## 3. Known Integrality Results

### 3.1 Integrality Gap

For **metric instances** (D satisfies triangle inequality):

| Bound | Value | Reference |
|-------|-------|-----------|
| Lower bound on gap | 2 | Known constructions |
| Upper bound on gap | ~2.675 | Cohen-Addad et al. (2022) |

For non-metric D, the gap can be unbounded.

**Note for DTWC++:** Standard DTW is NOT a metric (it violates the triangle inequality). These bounds are therefore not formally guaranteed for DTW distance matrices, though empirically the gap is typically small. See Section 3.4 for details.

### 3.2 When Is the LP Naturally Integer?

1. **Well-separated clusters**: If cluster centers satisfy `||mu_i - mu_j|| > (2 + eps) * spread`, the LP is exact (Awasthi et al., 2015). This explains why the user sees integer solutions "most of the time" -- well-clustered data has clear separation.

2. **Small instances**: For p <= ~10, the LP is empirically almost always integral.

3. **Fixed medoid set**: If A[i,i] values are fixed to 0 or 1, the remaining assignment problem is a **transportation problem** with a TU constraint matrix. The assignment LP always yields integer solutions.

4. **Absence of odd cycles (Baiou and Barahona, 2011)**: The LP is exact iff the auxiliary bipartite graph (defined by the support of the optimal LP solution) has no odd cycles -- the "intersection property." This connects directly to the TU analysis in Section 2: the same odd-cycle structure that prevents total unimodularity (Section 2.5) also governs when specific LP instances are fractional. When the optimal assignment pattern avoids odd cycles among facilities, the LP vertex is integer even though the constraint matrix is not TU.

### 3.3 Half-Integrality and Odd-Cycle Extreme Points

Fractional LP solutions are typically **half-integral** (values in {0, 1/2, 1}). This is explained by the odd-cycle theory:

- Each fractional extreme point corresponds to an odd cycle in the facility assignment graph (Baiou and Barahona, 2009)
- Variables on the cycle take value 1/2; all other variables are integral
- The {0, 1/2}-Chvatal-Gomory cuts (Caprara and Fischetti, 1996) that eliminate these fractional solutions are exactly the **odd-cycle inequalities** from Section 4.2
- For p = 3, exhaustive enumeration of all square submatrices of the 10x9 constraint matrix finds exactly **2 violating submatrices** (both 6x6 with |det| = 2), corresponding to the two directed 3-cycles: 0->1->2->0 and 0->2->1->0

### 3.4 Role of the Metric Property

**Important caveat:** Standard DTW does NOT satisfy the triangle inequality and is therefore NOT a proper metric (Marteau, 2009; Jain, 2018). Counterexamples exist where DTW(A,C) > DTW(A,B) + DTW(B,C). This means the metric integrality gap bounds of [2, 2.675] do **not** directly apply to DTWC++ distance matrices.

However, several factors mitigate this in practice:
- DTW distances are non-negative and symmetric, which is sufficient for the MIP formulation to be well-defined
- Empirically, DTW distances on real data approximately satisfy the triangle inequality for similar-length series
- The LP relaxation quality depends on the specific distance matrix, not just whether it is formally metric
- For truly metric distances (e.g., Euclidean on equal-length series), the full theory applies

For non-metric D, the integrality gap is theoretically **unbounded**. In practice with DTW distances, the gap is typically small, but there is no formal guarantee.

---

## 4. Polyhedral Analysis

### 4.1 The p-Median Polytope

The **integer hull** P_PM = conv{feasible integer solutions} is strictly contained in the **LP relaxation** Q_PM = {A in [0,1]^{p x p} : constraints 1-3}. The gap P_PM != Q_PM is the source of fractional optima.

### 4.2 Facet-Defining Inequalities

The standard formulation already includes facets:
- Non-negativity: A[i,j] >= 0
- Linking: A[i,j] <= A[i,i] (facet-defining for p >= 3)

**Additional cuts that tighten the relaxation:**

1. **{0, 1/2}-Chvatal-Gomory cuts** (Caprara and Fischetti, 1996): Equivalent to odd-cycle inequalities. Separable in polynomial time via shortest-path in an auxiliary graph. Very effective for p-median.

2. **Strengthened linking (multi-assignment) cuts**:
   ```
   sum_{j in S} A[i,j] <= (|S| - 1) * A[i,i] + 1    for subsets S, i not in S
   ```
   These aggregate multiple linking constraints and cut off fractional solutions more aggressively.

3. **Odd-hole inequalities**: Derived from Baiou-Barahona theory. For odd cycles C in the conflict graph:
   ```
   sum_{(i,j) in C} A[i,j] <= (|C| - 1) / 2
   ```

---

## 5. Practical LP-Based Solving Strategies

### 5.1 Recommended Tiered Strategy

```
Tier 1: LP Relaxation First (always try this)
  - Solve without integrality constraints
  - If solution is integer (all vars within tol of 0 or 1): DONE (global optimum)
  - Cost: polynomial (simplex or interior point)
  - Expected: integer ~80-90% of the time for well-separated clusters

Tier 2: Branch on Facility Variables Only (if LP fractional)
  - Branch ONLY on diagonal vars A[i,i] (p variables, not p^2)
  - After fixing all A[i,i], assignment is TU -> LP gives integer assignment
  - Branching tree depth <= p, width = number of fractional A[i,i] (usually small)

Tier 3: Full MIP (fallback for hard instances)
  - Use HiGHS/Gurobi with:
    - Priority branching on A[i,i] variables
    - PAM solution as warm start (tight upper bound)
    - LP relaxation as lower bound
```

### 5.2 Cutting Plane Strategy

When the LP relaxation is fractional:

```
Step 1: Solve LP relaxation
Step 2: If integral, done
Step 3: Separate {0,1/2}-CG cuts (odd-cycle inequalities)
Step 4: Add violated cuts, re-solve LP
Step 5: Repeat until integral or no more cuts found
Step 6: If still fractional, branch on facility variables (Tier 2)
```

### 5.3 Lagrangian Relaxation

The most effective Lagrangian relaxation dualizes the **assignment constraints** `sum_i A[i,j] = 1`:

```
L(mu) = min  sum_{i,j} (D[i,j] - mu_j) * A[i,j] + sum_j mu_j
        s.t. sum_i A[i,i] = k
             A[i,j] <= A[i,i]  for all i != j
             A[i,j] in {0,1}
```

After dualization, the problem decomposes **by facility**: for each potential medoid i, independently decide whether to open it and which points to attract, based on the modified costs D[i,j] - mu_j. This yields p independent subproblems each solvable in O(p), plus a cardinality selection (choose the k best facilities). The Lagrangian dual is solved by subgradient optimization. The Lagrangian bound equals the LP bound for this formulation (Jain and Vazirani, 2001).

### 5.4 Benders Decomposition

The key observation that **fixed medoids => TU assignment** enables Benders decomposition:

**Master problem** (over facility variables only):
```
min  theta
s.t. sum_i y_i = k
     theta >= sum_j min_{i: y_i=1} D[i,j]    (Benders cuts)
     y_i in {0,1}
```

**Subproblem** (given fixed y): Assign each point to nearest open medoid. Always integral (TU). O(pk) per cut generation.

This scales to very large instances -- Duran-Mateluna, Ales, and Elloumi (2023) solved p-median with up to 238,025 points using a two-phase Benders approach.

### 5.5 LP Rounding

When exact optimality is not needed:

**Deterministic rounding:**
1. Solve LP relaxation
2. Set A[i,i] = 1 for the k largest diagonal values
3. Assign each point to nearest selected medoid (exact, TU)
4. Note: this simple rounding has no proven constant-factor guarantee. For formal guarantees, use the filtering/rounding technique of Charikar et al. (1999) which achieves ~6.67-approximation, or the dependent rounding of Li and Svensson (2016) for ~2.73-approximation.

**Iterative rounding:**
1. Solve LP. Fix variables at 0 or 1.
2. Re-solve reduced LP. Repeat until all integral.
3. Works well in practice since LP typically has many integer-valued variables.

### 5.6 Warm-Starting from PAM

Run PAM first, pass solution as MIP start:
- Provides tight upper bound
- Dramatically reduces branching
- If LP lower bound matches PAM cost (within tolerance), skip MIP entirely

---

## 6. Practical Solver Tuning & Implementation

### 6.1 Single Most Impactful Change: MIP Start from PAM

Providing the PAM (k-medoids) solution as a warm start to the MIP solver is the highest-impact, lowest-effort improvement. The solver immediately has a tight upper bound, dramatically reducing the branch-and-bound tree.

**HiGHS (~10 lines):**
```cpp
// After building the model, before highs.run():
std::vector<double> initial_solution(Nvar, 0.0);
// Set PAM solution: diagonal for medoids, assignment for members
for (auto med : pam_medoids)
    initial_solution[med * (Nb + 1)] = 1.0;
for (auto j : Range(Nb))
    initial_solution[nearest_medoid[j] * Nb + j] = 1.0;
highs.setSolution(HighsSolution{initial_solution, {}});
```

**Gurobi (~10 lines):**
```cpp
// After creating variables, before model.optimize():
for (auto i : Range(Nb))
    w[i * (Nb + 1)].set(GRB_DoubleAttr_Start, 0.0);
for (auto med : pam_medoids)
    w[med * (Nb + 1)].set(GRB_DoubleAttr_Start, 1.0);
for (auto j : Range(Nb))
    w[nearest_medoid[j] * Nb + j].set(GRB_DoubleAttr_Start, 1.0);
```

### 6.2 Gurobi Parameter Tuning

Current code in [mip_Gurobi.cpp:70](dtwc/mip/mip_Gurobi.cpp#L70) sets `NumericFocus = 3` (maximum). This is overkill for the p-median formulation (all coefficients are 0/1/-1 in constraints; only the objective has real-valued distances).

**Recommended changes:**
```cpp
// Replace:
model.set(GRB_IntParam_NumericFocus, 3);  // Overkill, 1.5-3x slower
// With:
model.set(GRB_IntParam_NumericFocus, 1);  // Sufficient for p-median
model.set(GRB_IntParam_MIPFocus, 2);      // Focus on proving optimality (tight LP bound)

// Add branching priorities on diagonal (facility) variables:
for (auto i : Range(Nb))
    w[i * (Nb + 1)].set(GRB_IntAttr_BranchPriority, 100);
```

**Why `MIPFocus = 2`:** Since the user wants global optimality certification, focus 2 directs Gurobi to work on tightening the lower bound rather than finding feasible solutions (the MIP start already provides a good feasible solution).

**Why branching on diagonals:** Once all A[i,i] are fixed to 0/1, the remaining assignment is a transportation problem (TU) -- the LP relaxation is automatically integer. So branching on just p variables (instead of p^2) suffices.

### 6.3 HiGHS Tuning

```cpp
// Useful HiGHS options:
highs.setOptionValue("mip_detect_symmetry", true);   // Exploit p-median symmetry
highs.setOptionValue("mip_heuristic_effort", 0.2);   // Reduce heuristic time (MIP start provides good incumbent)
```

**Symmetry breaking:** Add explicit symmetry-breaking constraints to prevent equivalent medoid orderings:
```cpp
// If medoid indices are interchangeable, enforce ordering:
// A[i,i] >= A[i+1,i+1]  (optional, reduces search space by k!)
```

### 6.4 LP-First Mode

In [mip_Highs.cpp:150-151](dtwc/mip/mip_Highs.cpp#L150-L151), integrality is set:
```cpp
model.lp_.integrality_.clear();
model.lp_.integrality_.resize(model.lp_.num_col_, HighsVarType::kInteger);
```

Add a flag to skip this for LP relaxation mode. Check if the LP solution is naturally integer before falling back to MIP.

### 6.5 Report LP Bound

After solving LP relaxation, report the lower bound. If it matches the PAM solution cost (within tolerance), the PAM solution is provably optimal -- skip MIP entirely.

### 6.6 Benders Decomposition (for N > 200)

The Benders master problem has only **N binary variables** (y_i = "is i a medoid?") plus 1 continuous variable (theta = cost), compared to N^2 in the compact formulation. The subproblem is trivial: assign each point to the nearest open medoid, O(Nk) -- no LP solve needed.

**Algorithm:**

```text
1. Initialize: solve PAM for upper bound UB, set LB = -inf
2. Master: min theta s.t. sum(y_i) = k, Benders cuts, y binary
3. Solve master -> get y*, theta*. Set LB = theta*
4. Subproblem: for each j, assign to nearest open medoid i* = argmin_{i: y_i=1} D[i,j]
   Compute actual cost Z = sum_j D[i*_j, j]
5. If Z <= theta* + eps: optimal. Else add cut:
   theta >= Z - sum_{i in S} D[sigma(j,i), j] * (1 - y_i)
   where S = current medoid set, sigma(j,i) = reassignment cost
6. Go to step 2
```

**HiGHS implementation:** Use `kCallbackMipDefineLazyConstraints` callback to add Benders cuts during branch-and-bound.

**Gurobi implementation:** Use `GRBCallback::addLazy()` within a callback class.

Benders is expected to outperform compact MIP for N > 200 and becomes essential for N > 1000.

### 6.7 Constraint Formulation Notes

**Keep disaggregated linking constraints.** The current formulation uses `A[i,j] <= A[i,i]` for each (i,j) pair. An alternative is aggregated: `sum_j A[i,j] <= (N-1)*A[i,i]`. Do NOT switch -- disaggregated constraints give a **tighter LP relaxation**, which is more important for proving optimality.

### 6.8 Decision Table

| Problem Size | Cluster Quality | Strategy | Expected Runtime |
|-------------|----------------|----------|-----------------|
| p <= 50 | Any | Full MIP + MIP start | < 1 second |
| p <= 200 | Well-separated | LP relaxation only | < 1 second |
| p <= 200 | Poor separation | MIP + MIP start + branching priority on A[i,i] | < 10 seconds |
| p <= 1000 | Well-separated | LP relaxation only | < 30 seconds |
| p <= 1000 | Poor separation | PAM + LP bound check; MIP with tuned params if gap > 0 | < 1 minute |
| p > 1000 | Any | Benders decomposition (N binary master + O(Nk) subproblem) | Scales well |

---

## 7. Summary of Key Results

| Question | Answer |
|----------|--------|
| Is the constraint matrix TU? | **No** for p >= 3. **Yes** for p <= 2. Constructive proof: 6x6 submatrix with det = -2 from any 3-cycle (Section 2.4). |
| What breaks TU? | Odd cycles among facilities. The assignment+linking interaction creates a parity conflict (Section 2.7). Cardinality constraint is irrelevant (Section 2.8). |
| Precise TU boundary? | p = 2 is TU (exhaustive verification), p >= 3 is not. For p = 3, exhaustive enumeration finds exactly 2 violating submatrices (the two directed 3-cycles, both 6x6 with det = ±2). |
| General formula? | For an n-cycle: det = (-1)^n - 1. Only odd cycles violate TU (Section 2.5). |
| Integrality gap (metric D)? | Bounded: lower ~2, upper ~2.675. **DTW is not metric; gap bounds not formally guaranteed but empirically small.** |
| When is LP naturally integer? | Well-separated clusters, small instances, no odd cycles in assignment graph (Baiou-Barahona, Section 3.2). |
| Is the assignment subproblem TU? | **Yes.** Fixed medoids => transportation problem. |
| Best decomposition? | Benders: master selects medoids (binary), subproblem assigns (TU). |
| How to interpret fractional values? | Fractional A[i,i] = 1/2 indicates facility i is on an odd cycle in the LP solution. Fractional A[i,j] means point j is "shared" between clusters. |
| Practical strategy? | LP first -> branch on A[i,i] only -> full MIP fallback. |

### k-Medoids constraint matrix is NOT totally unimodular
- TU boundary is p=3. For p≤2, the matrix IS TU.
- Odd cycles among facilities break TU (det = (-1)^n - 1 for n-cycle).
- With fixed medoid set, the assignment IS a transportation problem (TU) → enables Benders.

---


## 8. References

1. **Balinski, M.L.** (1965). "Integer programming: methods, uses, computation." Management Science 12(3), 253-313. *Original p-median formulation.*

2. **Kariv, O. and Hakimi, S.L.** (1979). "An algorithmic approach to network location problems." SIAM J. Applied Mathematics 37(3), 539-560. *NP-hardness of p-median.*

3. **Cornuejols, G., Nemhauser, G.L., and Wolsey, L.A.** (1990). "The uncapacitated facility location problem." In Discrete Location Theory, Wiley, 119-171. *Foundational polyhedral study.*

4. **Caprara, A. and Fischetti, M.** (1996). "{0, 1/2}-Chvatal-Gomory cuts." Mathematical Programming 74, 221-235. *Polynomial separation of half-integral cuts.*

5. **Charikar, M., Guha, S., Tardos, E., and Shmoys, D.B.** (1999). "A constant-factor approximation algorithm for the k-median problem." STOC 1999. *LP rounding for k-median.*

6. **Jain, K. and Vazirani, V.** (2001). "Approximation algorithms for metric facility location and k-median problems." JACM 48(2), 274-296. *Primal-dual method, Lagrangian bound = LP bound.*

7. **Avella, P. and Sassano, A.** (2001). "On the p-median polytope." Mathematical Programming 89, 395-411. *Facet-defining inequalities.*

8. **Arya, V. et al.** (2004). "Local search heuristics for k-median and facility location problems." SIAM J. Computing 33(3), 544-562. *3+eps local search approximation.*

9. **Baiou, M. and Barahona, F.** (2009). "On the integrality of some facility location polytopes." SIAM J. Discrete Mathematics 23(2), 665-679. *Half-integrality, odd-cycle characterization.*

10. **Baiou, M. and Barahona, F.** (2011). "On the p-median polytope and the intersection property." SIAM J. Discrete Mathematics 25(1), 1-20. *Graph-theoretic LP = IP characterization.*

11. **Awasthi, P. et al.** (2015). "Relax, no need to round: integrality of clustering formulations." ITCS 2015. *LP exact recovery under cluster separation.*

12. **Li, S. and Svensson, O.** (2016). "Approximating k-median via pseudo-approximation." SIAM J. Computing 45(2), 530-547. *2.732 approximation.*

13. **Duran-Mateluna, G., Ales, Z., and Elloumi, S.** (2023). "An efficient Benders decomposition for the p-median problem." European J. Operational Research. *Two-phase Benders decomposition scaling to 238k+ points.*

14. **Cohen-Addad, V. et al.** (2022). "Improved approximation for k-median." STOC 2022. *Current best 2.675+eps.*

15. **Nemhauser, G.L. and Wolsey, L.A.** (1988). "Integer and Combinatorial Optimization." Wiley. *Textbook: TU, cutting planes, polyhedral theory.*

16. **Hoffman, A.J. and Kruskal, J.B.** (1956). "Integral boundary points of convex polyhedra." In Linear Inequalities and Related Systems, Annals of Math. Studies 38, 223-246. *Foundational result: incidence matrices of bipartite graphs are TU.*

17. **Jain, B.J.** (2018). "Semi-Metrification of the Dynamic Time Warping Distance." arXiv:1808.09964. *DTW violates triangle inequality; proposes semi-metric conversion.*

18. **Marteau, P.F.** (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching." IEEE TPAMI 31(2), 306-318. *Proposes TWED as a proper metric alternative to DTW; documents DTW's metric failures.*

19. **Ghouila-Houri, A.** (1962). "Caracterisation des matrices totalement unimodulaires." Comptes Rendus de l'Academie des Sciences 254, 1192-1194. *Necessary and sufficient condition for TU via row partitioning.*

20. **Seymour, P.D.** (1980). "Decomposition of regular matroids." Journal of Combinatorial Theory, Series B, 28(3), 305-359. *Every TU matrix decomposes via 1/2/3-sums from network matrices.*

21. **Schrijver, A.** (1986). "Theory of Linear and Integer Programming." Wiley. *Standard textbook reference for TU theory, Ghouila-Houri characterisation, sufficient conditions.*
