/**
 * @file pdlp_lp.hpp
 * @brief First-order (PDLP) solve of the p-median LP relaxation — an independent
 *        cross-check of the LR-core Lagrangian bound (PLAN.md Phase 4, Task 4.5).
 *
 * @details The LR-core bound engine (`lagrangian_root*`) reaches the p-median LP
 * relaxation optimum by MAXIMIZING the Lagrangian dual max_μ L(μ) — matrix-free,
 * never forming the N²-column LP (Geoffrion's theorem: the dualized subproblem is
 * totally unimodular, so the Lagrangian dual value equals the LP relaxation value).
 *
 * This routine reaches the SAME number by the opposite route: it forms the explicit
 * p-median LP relaxation and solves it with HiGHS's first-order primal-dual method
 * PDLP (`solver = "pdlp"`, the cuPDLP-C port, or `"hipdlp"`, HiGHS's own PDHG),
 * optionally on the GPU. Two computations from different mathematics converging on
 * the same LP optimum is an independent arbiter (CLAUDE.md §4) that validates the
 * clever matrix-free bound at N far beyond the brute-force IP oracle's reach.
 *
 * PDLP is an LP solver: it returns the LP-RELAXATION optimum (a lower bound on the
 * integer p-median cost), NOT an integer clustering. It does not replace the LR-core
 * or the compact MIP — the p-median LP still has an integrality gap that only
 * branching closes. See LESSONS.md "PDLP / first-order LP arbiter".
 *
 * The LP built here (from a dense D):
 *   min  Σ_{i,j} D_ij x_ij
 *   s.t. Σ_i x_ii = k                       (cardinality: open k medoids)
 *        Σ_i x_ij = 1        ∀ j            (assignment: every point served once)
 *        x_ij ≤ x_ii         ∀ i ≠ j       (linking: serve from an open medoid)
 *        0 ≤ x_ij ≤ 1
 * with the diagonal x_ii acting as facility-open indicator y_i.
 *
 * @author Volkan Kumtepeli
 * @date 08 Jul 2026
 */

#pragma once

#include <string>

namespace dtwc::mip {

/// @brief Tuning for the PDLP LP relaxation solve (safe defaults).
struct PdlpParams
{
  std::string variant = "pdlp"; ///< HiGHS first-order LP: "pdlp" (cuPDLP-C) or "hipdlp" (HiGHS PDHG).
  double tol = 1e-8;            ///< PDLP KKT / optimality tolerance (first-order — see registered arbiter band).
  long iteration_limit = 0;    ///< PDLP iteration cap; 0 ⇒ leave the HiGHS default.
  bool use_gpu = false;        ///< Ask for the GPU backend. The device is a COMPILE-TIME property of the HiGHS
                               ///< build (CUPDLP_GPU), not a per-call toggle: on a GPU build solver="pdlp" always
                               ///< runs on the GPU regardless of this flag; on a CPU build this flag=true only
                               ///< WARNS to stderr (never a silent downgrade). See gpu_used / pdlp_gpu_available().
  bool verbose = false;        ///< Let HiGHS print its solver log.
};

/// @brief Result of the PDLP LP-relaxation solve. `lp_bound` is in RAW distance
///        units (same as LagrangianResult::lower_bound), so it compares directly.
struct PdlpResult
{
  double lp_bound = 0.0; ///< p-median LP-relaxation optimum (raw units); a valid lower bound on the integer cost.
  bool solved = false;   ///< true iff HiGHS reported the LP optimal within tolerance.
  long iterations = 0;   ///< PDLP iterations run.
  bool gpu_used = false; ///< true iff the solve actually ran on the GPU — i.e. a CUPDLP_GPU build AND
                         ///< variant=="pdlp". Reflects the build+variant, not the use_gpu request flag.
};

/**
 * @brief Solve the p-median LP relaxation on a dense distance matrix with PDLP.
 *
 * @param D       Row-major N×N distances; symmetric, `D[i*N+i] == 0`, `D ≥ 0`.
 * @param N       Number of points.
 * @param k       Number of medoids, 1 ≤ k ≤ N.
 * @param params  PDLP tuning.
 * @return LP-relaxation optimum (raw units), solve status, iterations, GPU flag.
 * @throws dtwc::InvalidInput if N ≤ 0 or k ∉ [1, N].
 * @throws dtwc::SolverError if HiGHS is not compiled in, or the solve fails.
 */
PdlpResult pdlp_lp_bound(const double *D, int N, int k, const PdlpParams &params = {});

/**
 * @brief Whether this build's HiGHS was compiled with the CUDA/cuPDLP GPU backend
 *        (DTWC_HIGHS_GPU, forwarded to HiGHS as CUPDLP_GPU=ON).
 *
 * A runtime capability query — the library reports its own build, since the
 * compile-time define does not propagate to consumer translation units. When
 * false, `pdlp_lp_bound` with `use_gpu=true` warns and runs on CPU; when true it
 * runs on the GPU and reports `gpu_used=true`.
 */
bool pdlp_gpu_available();

} // namespace dtwc::mip
