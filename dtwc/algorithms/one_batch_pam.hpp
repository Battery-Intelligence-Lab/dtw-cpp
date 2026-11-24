/**
 * @file one_batch_pam.hpp
 * @brief OneBatchPAM: O(Nm)-distance approximate k-medoids.
 *
 * Implements de Mathelin et al., "OneBatchPAM: A Fast and Frugal
 * K-Medoids Algorithm" (AAAI 2025, doi:10.1609/aaai.v39i15.33776).
 * @author Volkan Kumtepeli
 */

#pragma once

#include "../core/clustering_result.hpp"
#include "../base/settings.hpp"

#include <cstddef>
#include <cstdint>

namespace dtwc {

class Problem;

namespace algorithms {

struct OneBatchPAMOptions {
  index_t n_clusters = 3;
  index_t batch_size = -1;   ///< -1: max(k, min(N, max(64, 20*ceil(log2(N+1))))); an explicit value must be >= k.
  int max_iter = 100;        ///< Maximum eager-swap sweeps.
  std::uint64_t random_seed = settings::DEFAULT_RANDOM_SEED;
  double relative_tolerance = 1e-9; ///< Accept gain > tolerance * current estimate.
};

/** Observable work and approximation diagnostics for a OneBatchPAM run. */
struct OneBatchPAMStats {
  std::size_t batch_size = 0;
  std::uint64_t distance_evaluations = 0; ///< Actual DTW calls (self-pairs excluded).
  double full_matrix_fraction = 0.0;      ///< distance_evaluations / N^2.
  double estimated_objective = 0.0;      ///< Weighted fixed-batch objective.
  int accepted_swaps = 0;
};

/**
 * Run OneBatchPAM while keeping all N points eligible as medoids.
 *
 * Exactly one N-by-m table is materialised. Final labels require at most N*k
 * extra distances when a selected medoid is not itself in the batch; this is
 * still O(Nm) for the required m >= k setting. The full N-by-N Problem distance
 * matrix is never allocated or populated.
 */
core::ClusteringResult one_batch_pam(Problem& prob,
                                     const OneBatchPAMOptions& options,
                                     OneBatchPAMStats* stats = nullptr);

namespace detail {

/// The batch size for `n` points and `k` clusters, `requested` being OneBatchPAMOptions::batch_size: a positive one
/// (at most n), else min(n, max(64, 20 ceil(log2(n + 1)))); never below k. Plain values: the CLI's memory plan
/// asks before any series is read.
[[nodiscard]] std::size_t resolve_batch_size(std::size_t n, index_t k, index_t requested);

} // namespace detail
} // namespace algorithms
} // namespace dtwc
