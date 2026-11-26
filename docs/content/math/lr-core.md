---
title: "LR-Core Exact k-Medoids Solver: Polyhedral Structure & Derivation"
weight: 10
description: "Total unimodularity characterization, Geoffrion's theorem, Lagrangian dual ascent, reduced-cost fixing, and y-space branch-and-bound."
---

# Exact $k$-Medoids (`p`-Median) Optimization in DTWC++

DTWC++ provides two exact $k$-medoids clustering paths:

1. **`Method::LRCore` (`--method lrcore`)** — a solver-free Lagrangian relaxation, subgradient dual-ascent engine, Beasley reduced-cost variable fixer, and core-restricted $y$-space branch-and-bound solver. `LRCore` requires zero external LP/MIP libraries.
2. **`Method::MIP` (`--method mip`)** — solves the compact integer linear program directly with HiGHS or Gurobi, warm-started from FasterPAM.

---

## 1. Integer Linear Programming Formulations

Let $N$ be the number of time series, $k \in \{1, \dots, N\}$ the target number of medoids, and $D \in \mathbb{R}^{N \times N}$ the pairwise elastic distance matrix (symmetric and non-negative, not required to satisfy the triangle inequality; under Soft-DTW it holds the Soft-DTW divergence).

### 1.1 Cornuéjols–Fisher–Nemhauser Formulation (`LRCore`)

Introduce facility-opening binaries $y_i \in \{0, 1\}$ ("series $i$ is chosen as a medoid") and assignment binaries $x_{ij} \in \{0, 1\}$ ("series $j$ is assigned to medoid $i$"):

$$
\begin{aligned}
\min_{x, y} \quad & \sum_{i=1}^{N} \sum_{j=1}^{N} D_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i=1}^{N} x_{ij} = 1 && \forall j \in \{1, \dots, N\} \quad \text{(Assignment)} \\
& \sum_{i=1}^{N} y_i = k && \text{(Cardinality)} \\
& x_{ij} \le y_i && \forall i, j \in \{1, \dots, N\} \quad \text{(Linking)} \\
& x_{ij} \in \{0, 1\}, \; y_i \in \{0, 1\}
\end{aligned}
$$

When $D_{ii} = 0$ and $D_{ij} \ge 0$ (every distance a `Problem` computes: DTW, DDTW, WDTW, ADTW, MSM, TWE and the Soft-DTW divergence), every open medoid $i$ optimally assigns to itself ($x_{ii} = y_i$), reducing directly to Balinski's (1965) $N^2$-variable formulation implemented in `mip_Highs.cpp` and `mip_Gurobi.cpp`. At $N = 10^4$, the compact formulation has $10^8$ binary variables and $\approx 3 \times 10^8$ nonzeros, motivating an implicit $y$-space exact method that never materializes the $N^2$-column LP.

---

## 2. Total Unimodularity Characterization

### 2.1 When Is the Full Constraint Matrix Totally Unimodular?

**Proposition 2.1.** *The constraint matrix of the $k$-medoids LP relaxation is totally unimodular (TU) if and only if $N \le 2$.*

*Proof sketch.* For $N \le 2$, exhaustive cofactor expansion confirms all square submatrices have determinant in $\{-1, 0, +1\}$. For any $N \ge 3$, consider a directed facility odd cycle of length $n = 2m + 1 \ge 3$ (specifically $n = 3$ on indices $\{0, 1, 2\}$). Selecting the $2n$ rows $\{\text{Link}(0,1), \text{Assign}(1), \text{Link}(1,2), \text{Assign}(2), \dots, \text{Link}(n-1,0), \text{Assign}(0)\}$ and columns $\{x_{00}, x_{01}, x_{11}, x_{12}, \dots, x_{n-1,n-1}, x_{n-1,0}\}$ yields a $(2n) \times (2n)$ two-cyclic bidiagonal submatrix whose Laplace expansion down the first column gives

$$
\det(M_n) = (-1)^n - 1 = \begin{cases} 0 & \text{if } n \text{ is even}, \\ -2 & \text{if } n \text{ is odd}. \end{cases}
$$

At $N = 3$, the two directed $3$-cycles have $\det = \pm 2$. Moreover, under the cardinality row $\sum_{i=1}^N y_i = k$, fractional LP extreme points on non-metric distance matrices are **not** restricted to half-integers $\{0, \tfrac{1}{2}, 1\}$ (components such as $\tfrac{1}{4}, \tfrac{1}{3}, \tfrac{3}{4}$ arise when multiple odd-cycle structures share the cardinality budget), so $\{0, \tfrac{1}{2}\}$-Chvátal–Gomory cuts alone cannot close all integrality gaps; branching on $y$ is required in the general worst case.

### 2.2 Pairwise Totally Unimodular Substructures (Valid for All $N$)

Although the full system is non-TU for $N \ge 3$, **any two of the three constraint families form a totally unimodular matrix for every $N \ge 1$**:

1. **(Cardinality + Assignment) is TU for all $N$.** Every matrix entry is in $\{0, 1\}$. Diagonal columns $x_{ii}$ contain two $1$s (Cardinality and $\text{Assign}(i)$), while off-diagonal columns $x_{ij}$ ($i \ne j$) contain a single $1$ ($\text{Assign}(j)$). By Hoffman–Kruskal, a $\{0, 1\}$-matrix with at most two $1$s per column is TU iff the row incidence graph is bipartite; here every edge joins the Cardinality row to an $\text{Assign}(i)$ row, forming a **star graph**, which is bipartite. $\square$
2. **(Cardinality + Linking) is TU for all $N$.** By Ghouila–Houri's criterion, let $R$ be any subset of rows. Assign sign $+1$ to the Cardinality row (if in $R$). For each facility $i$, suppose $m_i$ linking rows $x_{ij} - y_i \le 0$ belong to $R$; assign $+1$ to $\lceil m_i / 2 \rceil$ of them and $-1$ to $\lfloor m_i / 2 \rfloor$ of them (flipped if Cardinality $\notin R$). Then column $y_i$'s signed sum lies in $\{0, +1\}$ (or $\{-1, 0\}$), while each column $x_{ij}$ appears in at most one linking row and has signed sum in $\{-1, 0, +1\}$. Hence every subset of rows admits an equitable $\{ \pm 1 \}$ signing. $\square$

Consequently, **all polyhedral hardness resides in the interaction between Assignment and Linking across the $N$ binary medoid decisions $y \in \{0, 1\}^N$**; once $y$ is fixed, optimal client assignments decouple into a trivial $O(Nk)$ nearest-medoid scan.

---

## 3. Lagrangian Relaxation & Geoffrion's Theorem

Dualizing the $N$ assignment equalities $1 - \sum_{i=1}^N x_{ij} = 0$ with unconstrained multipliers $\mu \in \mathbb{R}^N$ yields the Lagrangian dual function

$$
L(\mu) = \sum_{j=1}^{N} \mu_j + \min_{x, y} \left\{ \sum_{i=1}^{N} \sum_{j=1}^{N} (D_{ij} - \mu_j) x_{ij} \;:\; \sum_{i=1}^{N} y_i = k,\; 0 \le x_{ij} \le y_i \le 1 \right\}.
$$

For each candidate medoid $i$, setting $y_i = 1$ allows setting $x_{ij} = 1$ precisely where $D_{ij} < \mu_j$, yielding the decoupled **facility reduced benefit**

$$
\rho_i(\mu) = \sum_{j=1}^{N} \min(0,\; D_{ij} - \mu_j) \le 0.
$$

Sorting or selecting the $k$ most negative scores $S_k(\mu) = \operatorname{argmin}^{(k)}_i \rho_i(\mu)$ solves the Lagrangian subproblem in closed form:

$$
L(\mu) = \sum_{j=1}^{N} \mu_j + \sum_{i \in S_k(\mu)} \rho_i(\mu).
$$

**Theorem 3.1 (Zero Lagrangian Duality Gap over LP; Cornuéjols et al. 1977, Geoffrion 1974).** *Because the remaining constraint system (Cardinality + Linking) is totally unimodular for all $N$, the Lagrangian subproblem possesses the integrality property, and therefore*

$$
\max_{\mu \in \mathbb{R}^N} L(\mu) = z_{\text{LP}},
$$

*attaining the exact LP relaxation bound without ever constructing the $N^2$-variable LP.*

On well-separated time-series clusters (Awasthi et al. 2015), $z_{\text{LP}}$ equals the integer optimum $z_{\text{IP}}$, allowing `LRCore` to certify FasterPAM's solution as globally optimal right at the root node (`0` branch-and-bound nodes).

---

## 4. Warm Start, Dual Ascent, Variable Fixing & Branch-and-Bound

1. **Second-Nearest Warm Start & Subgradient Dual Ascent:** The multipliers start at $\mu_j^{(0)} = d_2(j)$, the second-smallest entry of column $j$ of $D$ (the distance from series $j$ to its nearest other series: the smallest entry, $D_{jj} = 0$, is the series itself; $d_1(j)$ is used when $k = N$), which immediately places $L(\mu^{(0)})$ near $z_{\text{LP}}$. The root then ascends $L$ by damped Polyak subgradient steps along $g_j = 1 - \sum_{i \in S_k(\mu)} \mathbb{I}[D_{ij} < \mu_j]$, deflected against zig-zag, towards the best primal cost found so far (FasterPAM's incumbent seeds it). Each iteration repairs a primal solution from $S_k(\mu)$, and a local-search polish runs periodically.
2. **Beasley Reduced-Cost Variable Fixing:** Let $\rho_{(k)}$ and $\rho_{(k+1)}$ denote the $k$-th and $(k+1)$-st smallest values of $\rho_i(\mu^*)$, and $U$ be the best primal incumbent cost:
   - Any candidate $i \notin S_k(\mu^*)$ with $L(\mu^*) + (\rho_i(\mu^*) - \rho_{(k)}) > U$ cannot appear in any optimal medoid set and is permanently **fixed closed ($y_i = 0$)**.
   - Any candidate $i \in S_k(\mu^*)$ with $L(\mu^*) + (\rho_{(k+1)} - \rho_i(\mu^*)) > U$ must appear in every optimal medoid set and is permanently **fixed open ($y_i = 1$)**.
3. **Core-Restricted $y$-Space Branch-and-Bound:** If $(U - L(\mu^*)) / |U| > \varepsilon_{\text{gap}}$ (the gap taken as 0 when $U = L(\mu^*)$), `LRCore` branches only on surviving core candidates $i \in \mathcal{C}$. At a tree node with forced-open medoids $F_1$ ($d_{F_1}(j) = \min_{i \in F_1} D_{ij}$) and remaining candidates $\mathcal{C}_{\text{rem}}$ (client floor $m_j(A) = \min_{i \in F_1 \cup \mathcal{C}_{\text{rem}}} D_{ij}$), clamping multipliers onto $\tilde{\mu}_j = \max(m_j(A), \min(\mu_j^*, d_{F_1}(j)))$ strictly dominates frozen root multipliers by eliminating double-counted client surpluses across $F_1$ and remaining candidates while preserving client-floor tightening.
