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

## Cross-Language Bindings (Python / MATLAB)

### mexErrMsgIdAndTxt uses longjmp — skips C++ destructors
- **CRITICAL**: Calling `mexErrMsgIdAndTxt` inside a try/catch block leaks all RAII objects in scope
- The function calls `longjmp` internally, which bypasses C++ stack unwinding
- **Fix**: Capture the error string into a local `std::string`, exit the try/catch scope (allowing destructors to run), THEN call `mexErrMsgIdAndTxt`
- This applies to ALL `mexErr*` functions, not just `mexErrMsgIdAndTxt`

### nanobind chosen over pybind11
- The adversarial review recommended pybind11 based on "670 lines of institutional knowledge" — but that was a Claude skill prompt file, not production code. Actual sunk cost: 68 lines.
- nanobind advantages that matter for DTWC++: native `nb::ndarray<T, nb::device::cuda>` for GPU Phase 3 (avoids dual-framework), stable ABI (one wheel per platform), 5-10x smaller binaries.
- Migration is trivial: 7 mechanical search-and-replace changes. Same author (Wenzel Jakob), same API philosophy.
- GIL release, dangling ref, and lifetime patterns transfer directly: `nb::call_guard<nb::gil_scoped_release>()`, `nb::rv_policy::reference_internal`.
- **Lesson**: Evaluate sunk cost by measuring actual production code, not supporting documentation.

### GIL release is essential for expensive C++ methods
- Any method running longer than ~10ms should release the GIL via `nb::call_guard<nb::gil_scoped_release>()`
- Without GIL release, `fillDistanceMatrix()` and `cluster()` freeze ALL Python threads
- nanobind's `std::function` wrapper automatically re-acquires the GIL when calling back into Python

### Reference-returning methods create dangling pointers in bindings
- Methods like `p_vec(size_t i)` returning `const std::vector<double>&` can dangle if the parent C++ object is GC'd
- Use `py::return_value_policy::reference_internal` to tie the returned reference's lifetime to the parent
- When in doubt, return by value (copy) — slight overhead but always safe

### MATLAB handles stored as double lose precision above 2^53
- MATLAB represents everything as double; uint64 handles above 2^53 (~9e15) silently corrupt
- In practice the counter never reaches 2^53, but add a runtime guard

### Armadillo ↔ MATLAB is zero-copy but dangerous for writes
- Both use column-major storage, so sharing memory via `const_cast<double*>` works for read-only
- But any Armadillo operation that triggers reallocation writes into MATLAB memory — undefined behavior
- Always use the copy version (`mxArray_to_arma`) for any mutable operations

### CasADi's cross-language philosophy works well
- Same class names (PascalCase), same method names (camelCase) across C++/Python/MATLAB
- Python gets snake_case aliases as a bonus, but camelCase is the primary for consistency
- CasADi uses SWIG; for just 2 target languages, hand-written pybind11 + MEX is simpler and more maintainable

### dtwBanded template default is float, not double
- `dtwBanded<data_t = float>` — must explicitly instantiate `dtwBanded<double>` for the project's `data_t = double`
- Easy to miss; always check template defaults against the project's type aliases

### dtwBanded allocates full N x N matrix even for banded computation
- **CRITICAL PERFORMANCE BUG**: `C.resize(m_long, m_short)` allocates the full matrix then only fills the band
- For 8K x 8K series: 512 MB per thread. With 8 threads: 4 GB of scratch memory
- **Fix**: Use a rolling buffer of width `2*band+1`. For band=50: 808 bytes vs 512 MB (>600,000x reduction)
- This single fix gives more speedup than all SIMD work combined for 8K series

### "PAM" implementation is actually Lloyd iteration
- The code in `cluster_by_kMedoidsPAM` does alternating assign-then-update-medoid-within-cluster
- True PAM SWAP considers swapping any medoid with ANY non-medoid point across all clusters
- Lloyd iteration restricts medoid updates to within each cluster, getting stuck in worse optima
- FastPAM (Schubert & Rousseeuw 2021) is the state-of-the-art with O(k) speedup over naive PAM

### DTW is latency-bound on the recurrence chain, not memory-bound
- Original analysis (pre-optimization): 0.125 FLOP/byte with full matrix — appeared memory-bound
- After `std::min` initializer_list fix: DTW uses only 3% of L1 bandwidth. The bottleneck is the 10-cycle loop-carried dependency chain (`min(diag, min(left, below)) + dist`)
- At 2.5 GHz, this gives ~250M cells/sec — matches measured performance exactly
- **The recurrence cannot be shortened** — it is fundamental to DTW's dynamic programming structure
- **Multi-pair SIMD** (4 pairs in AVX2) can hide latency by processing 4 independent recurrences in parallel — expected 3-3.5x
- **BUT**: scatter/gather overhead kills multi-pair DTW. Each inner-loop iteration needs 4 scalar reads to fill an aligned array (28 ops/cell vs 9 scalar). Max theoretical speedup: 1.29x. Fix: pre-interleave series data before batching.
- **Highway dispatch overhead**: ~10-15ns per call (atomic load + BitScan + indirect call). For 40ns LB_Keogh, that's 25-35% overhead. For trivial loops, compiler auto-vectorization beats Highway.
- **MSVC `/O2` auto-vectorizes** simple reduction loops (LB_Keogh, z_normalize) to SSE2. Measured 1.07 cycles/element for LB_Keogh (theoretical scalar min is 2 cycles) — proves auto-vectorization. Explicit Highway adds dispatch overhead and prevents inlining. Use `#pragma omp simd` instead.
- **SIMD fillDistanceMatrix must respect band and DTW variant**. The multi-pair DTW function only implements full L1 DTW — using it when band>0 or DDTW/WDTW/ADTW is selected produces WRONG results (different distance values → different clusters).

### std::min with initializer_list is catastrophically slow in hot loops
- `std::min({a, b, c})` constructs a temporary `std::initializer_list` on every call
- On MSVC this compiles to a function call + stack allocation, not two `vminsd` instructions
- Replacing with `std::min(a, std::min(b, c))` gave **2.5-3x speedup** on all DTW functions
- **Rule**: Never use `std::min` with initializer_list in inner loops. Two nested calls are equally readable and dramatically faster.

### Lambda capture-by-value creates stale parameter bugs
- `rebind_dtw_fn()` originally captured `[b = band]` — the lambda froze `band` at creation time
- Setting `prob.band = 50` after construction silently had no effect — banding was ignored
- **Fix**: Capture `[this]` and read `this->band` at invocation time
- **Rule**: For lambdas stored as `std::function` members, capture `this` for parameters that may change. Only capture by value for truly immutable data.
- **Implication**: Fix memory access patterns BEFORE doing SIMD work. SIMD on a bandwidth-bound kernel gives minimal gains.
- The rolling buffer fix (fitting in L1 cache) is worth more than vectorization

### Virtual dispatch in hot inner loops kills performance
- `IDistanceMatrix::get(i,j)` with virtual dispatch costs ~3ns per call via vtable indirection
- At N=10K, PAM does ~100M distance lookups per iteration → 300ms pure dispatch overhead
- Use CRTP or template parameters for hot-path abstractions; virtual dispatch only at outer API boundary

### Template explosion is rarely justified for DTW
- DTWPolicy<Constraint, Metric, Missing> with 3×6×3 = 54+ instantiations
- Runtime metric dispatch overhead: ~3ns per DTW call. DTW computation: 1-100ms.
- Overhead ratio: 0.003% — templates for metric selection are unjustified
- Template on constraint type only (2-3 variants); pass metric as runtime callable

### LB_Keogh is only valid for L1 and squared L2 metrics
- The lower bound proof assumes envelope-based properties: `d(x, U) = 0` when `L <= x <= U`
- This holds for L1 and squared L2 but NOT for cosine distance (defined between vectors, not scalars)
- Not for Huber (piecewise, LB_Keogh proof requires squared norm)
- Using LB_Keogh with invalid metrics silently prunes valid nearest neighbors → wrong clustering results
- Maintain a compile-time compatibility matrix; disable LB pruning for unvalidated metrics

### Soft-DTW cannot be implemented via autodiff on std::min
- `std::min` gives hard subgradients (1 for the min argument, 0 for others) — zero gradient almost everywhere
- Soft-DTW requires replacing `min` with `softmin_gamma = -gamma * log(sum exp(-a/gamma))` in the DP recurrence
- Must implement log-sum-exp trick for numerical stability
- Soft-DTW is a separate algorithm, NOT a metric substitution or autodiff trick

### WDTW and ADTW cannot be implemented via metric abstraction
- WDTW weights depend on position (i,j) in the cost matrix, not on values x[i], y[j]
- ADTW penalizes non-diagonal steps — a recurrence change, not a distance change
- DDTW requires preprocessing (derivative computation) before DTW, not a metric swap
- Architecture needs Transform pipeline + parameterizable recurrence, not just swappable metrics

### OpenMP inside MEX needs careful management
- MATLAB manages its own thread pool; OpenMP threads can conflict
- On macOS, MATLAB ships its own `libomp.dylib` which conflicts with system/Homebrew copies
- Always limit `omp_set_num_threads` conservatively in MEX; never enable nested parallelism

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
