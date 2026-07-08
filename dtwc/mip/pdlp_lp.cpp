/**
 * @file pdlp_lp.cpp
 * @brief PDLP first-order solve of the p-median LP relaxation (PLAN.md Task 4.5).
 *
 * Builds the explicit p-median LP relaxation from a dense distance matrix and
 * solves it with HiGHS's first-order primal-dual method (`solver = "pdlp"` or
 * `"hipdlp"`), optionally on the GPU. The optimum is an independent cross-check
 * of the LR-core Lagrangian bound (both equal the LP-relaxation value by
 * Geoffrion; see pdlp_lp.hpp). LP-only: returns a lower bound, not a clustering.
 */

#include "pdlp_lp.hpp"
#include "../error.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace dtwc::mip {

bool pdlp_gpu_available()
{
#ifdef DTWC_HIGHS_GPU
  return true;
#else
  return false;
#endif
}

PdlpResult pdlp_lp_bound(const double *D, int N, int k, const PdlpParams &params)
{
#ifndef DTWC_ENABLE_HIGHS
  (void)D;
  (void)N;
  (void)k;
  (void)params;
  throw SolverError(
    "pdlp_lp_bound: the PDLP LP solver requires HiGHS. Rebuild with -DDTWC_ENABLE_HIGHS=ON.");
#else
  if (N <= 0) throw InvalidInput("pdlp_lp_bound: N must be positive");
  if (k < 1 || k > N)
    throw InvalidInput("pdlp_lp_bound: require 1 <= k <= N (k=" + std::to_string(k)
                       + ", N=" + std::to_string(N) + ")");

  const std::size_t Nz = static_cast<std::size_t>(N);

  // Scale costs to O(1) for the first-order solver's conditioning; unscale the
  // objective on the way out so lp_bound is reported in RAW distance units.
  double maxD = 0.0;
  for (std::size_t t = 0; t < Nz * Nz; ++t) maxD = std::max(maxD, D[t]);
  const double scaling = std::max(maxD, 1.0);

  // ---- Build the p-median LP relaxation --------------------------------------
  // Columns: x_ij at index i*N + j (i = candidate medoid, j = point), x ∈ [0,1];
  // the diagonal x_ii doubles as the facility-open indicator y_i.
  const std::size_t nvar = Nz * Nz;
  // Rows: 0 = cardinality; 1..N = assignment (row 1+j serves point j);
  //       N+1.. = linking, one per ordered pair (i,j), i≠j.
  const std::size_t n_link = Nz * (Nz - 1);
  const std::size_t nrow = 1 + Nz + n_link;

  HighsModel model;
  model.lp_.num_col_ = static_cast<HighsInt>(nvar);
  model.lp_.num_row_ = static_cast<HighsInt>(nrow);
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.offset_ = 0.0;

  model.lp_.col_cost_.assign(nvar, 0.0);
  model.lp_.col_lower_.assign(nvar, 0.0);
  model.lp_.col_upper_.assign(nvar, 1.0);
  for (std::size_t i = 0; i < Nz; ++i)
    for (std::size_t j = 0; j < Nz; ++j)
      model.lp_.col_cost_[i * Nz + j] = D[i * Nz + j] / scaling;

  model.lp_.row_lower_.assign(nrow, 0.0);
  model.lp_.row_upper_.assign(nrow, 0.0);
  model.lp_.row_lower_[0] = model.lp_.row_upper_[0] = static_cast<double>(k); // Σ_i x_ii = k
  for (std::size_t j = 0; j < Nz; ++j)
    model.lp_.row_lower_[1 + j] = model.lp_.row_upper_[1 + j] = 1.0;          // Σ_i x_ij = 1
  for (std::size_t r = 1 + Nz; r < nrow; ++r) {
    model.lp_.row_lower_[r] = -kHighsInf;                                     // x_ij − x_ii ≤ 0
    model.lp_.row_upper_[r] = 0.0;
  }

  // Sparse matrix, row-wise (CSR): emit nonzeros row by row so start_ is sorted.
  auto &A = model.lp_.a_matrix_;
  A.format_ = MatrixFormat::kRowwise;
  A.num_col_ = static_cast<HighsInt>(nvar);
  A.num_row_ = static_cast<HighsInt>(nrow);
  const std::size_t nnz = Nz + nvar + 2 * n_link; // cardinality + assignment + linking
  A.start_.clear();
  A.start_.reserve(nrow + 1);
  A.index_.clear();
  A.index_.reserve(nnz);
  A.value_.clear();
  A.value_.reserve(nnz);

  A.start_.push_back(0);
  // Row 0: cardinality — +1 on each diagonal x_ii.
  for (std::size_t i = 0; i < Nz; ++i) {
    A.index_.push_back(static_cast<HighsInt>(i * Nz + i));
    A.value_.push_back(1.0);
  }
  A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
  // Rows 1..N: assignment for point j — +1 on x_ij for every medoid i.
  for (std::size_t j = 0; j < Nz; ++j) {
    for (std::size_t i = 0; i < Nz; ++i) {
      A.index_.push_back(static_cast<HighsInt>(i * Nz + j));
      A.value_.push_back(1.0);
    }
    A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
  }
  // Linking rows: x_ij − x_ii ≤ 0 for each ordered pair (i,j), i≠j. Column
  // indices must be ascending within a row: for fixed i, x_ii (=i*N+i) precedes
  // x_ij (=i*N+j) iff i<j.
  for (std::size_t i = 0; i < Nz; ++i) {
    for (std::size_t j = 0; j < Nz; ++j) {
      if (i == j) continue;
      const HighsInt c_ij = static_cast<HighsInt>(i * Nz + j);
      const HighsInt c_ii = static_cast<HighsInt>(i * Nz + i);
      if (i < j) {
        A.index_.push_back(c_ii);
        A.value_.push_back(-1.0);
        A.index_.push_back(c_ij);
        A.value_.push_back(1.0);
      } else {
        A.index_.push_back(c_ij);
        A.value_.push_back(1.0);
        A.index_.push_back(c_ii);
        A.value_.push_back(-1.0);
      }
      A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
    }
  }

  // ---- Solve with PDLP -------------------------------------------------------
  Highs highs;
  if (!params.verbose) highs.setOptionValue("output_flag", false);
  highs.setOptionValue("solver", params.variant); // "pdlp" (cuPDLP-C) | "hipdlp" (HiGHS PDHG)
  highs.setOptionValue("kkt_tolerance", params.tol);
  if (params.iteration_limit > 0)
    highs.setOptionValue("pdlp_iteration_limit", static_cast<HighsInt>(params.iteration_limit));

  // The compute device is a COMPILE-TIME property of the HiGHS build
  // (CUPDLP_GPU), not a per-solve toggle: on a CUPDLP_GPU build, solver="pdlp"
  // (cuPDLP-C) ALWAYS runs on the GPU — there is no per-call CPU path — while
  // "hipdlp" (HiGHS's own PDHG) stays on the CPU. So gpu_used reflects the build
  // and the chosen variant, NOT the request flag: reporting gpu_used=false for a
  // solve that actually ran on the GPU would be a false report (CLAUDE.md §1).
  // use_gpu only governs the warning when the GPU is asked for but not built in.
  const bool gpu_used = pdlp_gpu_available() && (params.variant == "pdlp");
  if (params.use_gpu && !pdlp_gpu_available())
    std::fprintf(stderr,
      "pdlp_lp_bound: GPU PDLP requested but HiGHS was built without CUPDLP_GPU; "
      "running on CPU. Rebuild with -DDTWC_HIGHS_GPU=ON to enable the GPU backend.\n");

  if (highs.passModel(model) == HighsStatus::kError)
    throw SolverError("pdlp_lp_bound: HiGHS rejected the LP model (passModel returned error).");

  if (highs.run() == HighsStatus::kError)
    throw SolverError("pdlp_lp_bound: HiGHS PDLP run returned an error status.");

  const HighsModelStatus st = highs.getModelStatus();
  const HighsInfo &info = highs.getInfo();

  PdlpResult r;
  r.solved = (st == HighsModelStatus::kOptimal);
  r.lp_bound = info.objective_function_value * scaling; // back to raw distance units
  r.iterations = static_cast<long>(info.pdlp_iteration_count);
  r.gpu_used = gpu_used;

  if (!r.solved)
    throw SolverError("pdlp_lp_bound: PDLP did not reach optimality. Model status: "
                      + highs.modelStatusToString(st));
  return r;
#endif
}

} // namespace dtwc::mip
