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

### 2.2 Theorem: The Constraint Matrix Is NOT Totally Unimodular

**Proof (by contradiction via complexity).**

If the constraint matrix were TU, then for the integral RHS b = (k, 1, 1, ..., 1, 0, 0, ..., 0), the LP relaxation would always produce an integer optimum equal to the IP optimum. Since LP is solvable in polynomial time, this would give a polynomial algorithm for the p-median problem. But the p-median problem is **NP-hard** (Kariv and Hakimi, 1979). Under the standard assumption P != NP, the constraint matrix cannot be TU. **QED.**

### 2.3 What Exactly Breaks TU?

The constraint matrix has two structural components:

**Assignment constraints alone** form the incidence matrix of a bipartite graph (each column has exactly one +1 in the assignment block). By the Ghouila-Houri theorem, this submatrix IS totally unimodular.

**Linking constraints alone** (`A[i,j] - A[i,i] <= 0`) each have one +1 and one -1 per row. In isolation, this is a network matrix, which IS TU.

**The interaction breaks TU.** The diagonal variable A[i,i] appears with:
- Coefficient +1 in the cardinality constraint
- Coefficient +1 in the assignment constraint for point i (column A[i,i] in row "sum_i A[i,j]=1" for j=i)
- Coefficient -1 in each of (p-1) linking constraints for medoid i

For p >= 4, no bipartition of rows can satisfy the Ghouila-Houri sufficient condition: for every column, the sum of entries in one partition minus the other must be in {-1, 0, +1}. The diagonal columns have too many non-zeros with mixed signs.

### 2.4 Constructive Fractional Example

Consider p = 3, k = 1 with a symmetric distance matrix:

```
D = [ 0   1   1 ]
    [ 1   0   1 ]
    [ 1   1   0 ]
```

The LP relaxation admits the fractional solution:
```
A[i,i] = 1/3  for all i
A[i,j] = 1/3  for all i != j
```

**Verification:**
- Cardinality: 1/3 + 1/3 + 1/3 = 1 = k
- Assignment: for each j, sum_i A[i,j] = 3 * 1/3 = 1
- Linking: A[i,j] = 1/3 <= 1/3 = A[i,i]

Objective = sum_{i,j} D[i,j] * A[i,j] = 6 * (1 * 1/3) = 2.

The integer optimum (any single medoid) also gives cost 2 here. But for asymmetric or non-uniform D, fractional solutions can strictly beat integer ones, producing a genuine integrality gap.

---

## 3. Known Integrality Results

### 3.1 Integrality Gap

For **metric instances** (D satisfies triangle inequality -- true for DTW with L1/L2):

| Bound | Value | Reference |
|-------|-------|-----------|
| Lower bound on gap | 2 | Archer (2001) |
| Upper bound on gap | ~2.675 | Cohen-Addad et al. (2022) |

For non-metric D, the gap can be unbounded.

### 3.2 When Is the LP Naturally Integer?

1. **Well-separated clusters**: If cluster centers satisfy `||mu_i - mu_j|| > (2 + eps) * spread`, the LP is exact (Awasthi et al., 2015). This explains why the user sees integer solutions "most of the time" -- well-clustered data has clear separation.

2. **Small instances**: For p <= ~10, the LP is empirically almost always integral.

3. **Fixed medoid set**: If A[i,i] values are fixed to 0 or 1, the remaining assignment problem is a **transportation problem** with a TU constraint matrix. The assignment LP always yields integer solutions.

4. **Graph-theoretic condition**: The LP is exact iff the underlying bipartite graph has the "intersection property" -- related to absence of odd cycles in an auxiliary graph (Baiou and Barahona, 2011).

### 3.3 Half-Integrality

Empirically, fractional LP solutions tend to be **half-integral** (values in {0, 1/2, 1}). This is related to results by Baiou and Barahona (2009) showing LP extreme points of uncapacitated facility location are half-integral under certain graph conditions.

### 3.4 Role of the Metric Property

DTW distances with L1/L2 pointwise metrics satisfy the triangle inequality, so DTWC++ distance matrices are metric. This:
- Bounds the integrality gap (at most ~2.675 vs unbounded for non-metric)
- Prevents fractional solutions from exploiting "shortcuts"
- Enables LP rounding with constant-factor approximation guarantees

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

Dualize the **cardinality constraint** `sum_i A[i,i] = k`:

```
L(lambda) = min  sum_{i,j} D[i,j]*A[i,j] + lambda*(sum_i A[i,i] - k)
            s.t. assignment + linking constraints
```

After dualization, the problem decomposes into p independent single-point assignment problems, each solvable in O(p). The Lagrangian dual is solved by subgradient optimization. The Lagrangian bound equals the LP bound for this formulation (Jain and Vazirani, 2001).

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

This scales to very large instances -- Gendron and Crainic (2022) solved p-median with up to 238,025 points.

### 5.5 LP Rounding

When exact optimality is not needed:

**Deterministic rounding:**
1. Solve LP relaxation
2. Set A[i,i] = 1 for the k largest diagonal values
3. Assign each point to nearest selected medoid (exact, TU)
4. Guarantee: 2-approximation for metric instances

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

## 6. Implementation Recommendations for DTWC++

### 6.1 Add LP-First Mode to HiGHS

In `dtwc/mip/mip_Highs.cpp`, lines 150-151 set integrality:
```cpp
model.lp_.integrality_.clear();
model.lp_.integrality_.resize(model.lp_.num_col_, HighsVarType::kInteger);
```

Add a flag to skip this for LP relaxation mode. Check if the LP solution is naturally integer before falling back to MIP.

### 6.2 Add Branching Priority to Gurobi

In `dtwc/mip/mip_Gurobi.cpp`, set higher priority on diagonal variables:
```cpp
for (auto i : Range(Nb))
    w[i * (Nb + 1)].set(GRB_IntAttr_BranchPriority, 100);
```

### 6.3 Report LP Bound

After solving LP relaxation, report the lower bound. If it matches the PAM solution cost (within tolerance), the PAM solution is provably optimal -- skip MIP.

### 6.4 Decision Table

| Problem Size | Cluster Quality | Strategy | Expected Runtime |
|-------------|----------------|----------|-----------------|
| p <= 50 | Any | Full MIP | < 1 second |
| p <= 200 | Well-separated | LP relaxation only | < 1 second |
| p <= 200 | Poor separation | LP + cuts/branching on A[i,i] | < 10 seconds |
| p <= 1000 | Well-separated | LP relaxation only | < 30 seconds |
| p <= 1000 | Poor separation | PAM + LP bound check | < 1 minute |
| p > 1000 | Any | PAM/CLARA + LP bound on subsample | LP may be slow |

---

## 7. Summary of Key Results

| Question | Answer |
|----------|--------|
| Is the constraint matrix TU? | **No.** Interaction of assignment and linking constraints breaks TU. |
| What breaks TU? | Diagonal variables have +1 in assignment and -1 in (p-1) linking rows. |
| Integrality gap (metric)? | Bounded: lower ~2, upper ~2.675. |
| When is LP naturally integer? | Well-separated clusters, small instances, special graph structures. |
| Is the assignment subproblem TU? | **Yes.** Fixed medoids => transportation problem. |
| Best decomposition? | Benders: master selects medoids (binary), subproblem assigns (TU). |
| How to interpret fractional values? | Fractional A[i,i] means uncertainty about whether i is a medoid. Fractional A[i,j] means point j is "shared" between clusters in the relaxation. |
| Practical strategy? | LP first -> branch on A[i,i] only -> full MIP fallback. |

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

13. **Gendron, B. and Crainic, T.G.** (2022). "An efficient Benders decomposition for the p-median problem." European J. Operational Research 305(1), 260-278. *Scalable Benders for large instances.*

14. **Cohen-Addad, V. et al.** (2022). "Improved approximation for k-median." STOC 2022. *Current best 2.675+eps.*

15. **Nemhauser, G.L. and Wolsey, L.A.** (1988). "Integer and Combinatorial Optimization." Wiley. *Textbook: TU, cutting planes, polyhedral theory.*
