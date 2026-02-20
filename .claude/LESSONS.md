# DTWC++ Lessons Learned

Critical knowledge discovered during development, to avoid repeating mistakes.

---

## Mathematical / Theoretical

### DTW is NOT a metric
Standard DTW violates the triangle inequality. This means:
- Integrality gap bounds from p-median theory (which assume metric D) do NOT formally apply
- LP relaxation quality has no theoretical guarantee for DTW distances
- In practice, the gap is small, but there is no proof
- References: Marteau (2009, IEEE TPAMI), Jain (2018, arXiv:1808.09964)

### k-Medoids constraint matrix is NOT totally unimodular
- **TU boundary is p = 3** (not p = 4 as initially claimed). For p <= 2, the matrix IS TU.
- **Constructive proof:** a 6x6 submatrix from any 3-cycle among facilities has det = -2
- **General formula:** for an n-cycle, det = (-1)^n - 1. Only **odd cycles** violate TU.
- **Cardinality constraint is irrelevant** to the TU failure. The violation comes purely from assignment+linking interaction.
- The Ghouila-Houri partition failure follows an elegant parity argument: the odd cycle creates a chain of forced sign assignments that wraps around with a sign flip.
- For p = 3, exactly 2 violating submatrices exist (the two directed 3-cycles). All verified computationally.
- BUT: with fixed medoid set, the remaining assignment IS a transportation problem (TU!)
- This enables Benders decomposition and the tiered LP strategy

### DTW-AROW != simple zero-cost DTW
- DTW-AROW (Yurtman et al. 2023) constrains each missing value to one-to-one alignment
- Simple zero-cost DTW is less restrictive and may underestimate distances more
- They are NOT equivalent despite superficial similarity

## C++ Implementation

### NaN representation for missing data
- Use `std::numeric_limits<data_t>::quiet_NaN()` as sentinel
- `constexpr` NaN is only guaranteed in C++23; use `inline const` for C++17
- Union type-punning for bitwise NaN check is **undefined behavior** in C++; use `std::memcpy` instead
- Add `static_assert(sizeof(data_t) == 8)` guard if using 64-bit bit patterns
- The `is_missing()` check MUST happen BEFORE calling the distance function, otherwise NaN propagates silently through the entire cost matrix

### HiGHS vs Gurobi indexing conventions
- HiGHS uses a transposed indexing convention compared to the document notation
- Both produce equivalent results because D is symmetric for DTW
- For non-symmetric D, the implementations would give different results

## Research Process

### Always verify citations with a separate review agent
- Author names, venues, and volume numbers can be hallucinated
- Found: wrong authors for Jiang et al. (was Jang/Kpotufe, should be Arias-Castro)
- Found: wrong attribution for arXiv paper (was Herrmann, should be Jain)
- Found: wrong venue for Marteau (was ICPR, should be TPAMI)
- Found: wrong volume for Yurtman (was LNAI 14169, should be LNCS 14173)
- Found: wrong authors for Benders p-median paper (was "Gendron and Crainic (2022)", should be "Duran-Mateluna, Ales, and Elloumi (2023)")

### Claims about approximation guarantees need careful sourcing
- "2-approximation for metric instances" was unsupported for naive rounding
- The actual LP-rounding literature gives larger constants (~6.67 from Charikar et al., ~2.73 from Li & Svensson)
- Always cite the specific paper that proves the claimed bound

### Lagrangian relaxation: which constraint to dualize matters
- Dualizing cardinality does NOT decompose into independent subproblems (linking constraints still couple variables)
- Dualizing assignment constraints DOES decompose by facility -- this is the standard approach in the literature
