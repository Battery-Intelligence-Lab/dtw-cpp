/**
 * @file pruned_distance_matrix.hpp
 * @brief Exact distance-matrix construction with LB-guided cutoff attempts.
 *
 * @details Builds an exact distance matrix using cascading lower bounds
 * (LB_Kim -> configured envelope bound(s) -> early-abandon DTW). Enhanced
 * evaluates max(LB_Keogh, LB_Enhanced); the bound cascade is L1-valued.
 *
 * The strategy for all-pairs distance matrix:
 * - Precompute summaries and Lemire envelopes once in O(sum_i n_i), or O(N*n)
 *   for N equal-length series of length n.
 * - For each pair (i,j), evaluate the configured Kim/envelope-bound cascade.
 * - Track per-row nearest-neighbor distance as it's discovered.
 * - If that bound clears the current nearest-neighbor threshold, attempt DTW
 *   with the threshold as its cutoff. A cutoff sentinel triggers a retry
 *   without early abandon because every exact entry is required.
 *
 * This route does not skip an exact matrix entry or a required full result.
 * An abandoned attempt adds work before recomputation, so its counters are
 * branch-accounting diagnostics rather than evidence of work saved. With
 * band=-1, LB_Keogh is disabled; constructing its current radius-zero envelope
 * would not be admissible for full DTW.
 *
 * References:
 *   - E. Keogh, C.A. Ratanamahatana, "Exact indexing of dynamic time warping",
 *     Knowledge and Information Systems, 7(3), 358-386, 2005.
 *   - S.-W. Kim, S. Park, W.W. Chu, "An Index-Based Approach for Similarity
 *     Search Supporting Time Warping in Large Sequence Databases", ICDE 2001.
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 28 Mar 2026
 */

#pragma once

#include "../warping.hpp"
#include "../Problem.hpp"
#include "../enums/LowerBoundStrategy.hpp"
#include "lower_bounds.hpp"
#include "lower_bound_impl.hpp"
#include "../settings.hpp"

#include <cstddef>
#include <vector>

namespace dtwc::core {

/// Statistics from pruned distance matrix construction.
struct PruningStats {
  size_t total_pairs = 0;         ///< Total unique pairs (upper triangle)
  size_t pruned_by_lb_kim = 0;    ///< Pairs where LB_Kim selected the cutoff branch
  size_t pruned_by_lb_keogh = 0;  ///< Pairs where an envelope LB selected that branch
  size_t early_abandoned = 0;     ///< Cutoff attempts followed by exact recomputation
  size_t computed_full_dtw = 0;   ///< Pairs computed directly without a cutoff attempt

  /// Fraction of pairs whose first attempt abandoned; not a work-saved ratio.
  double pruning_ratio() const
  {
    return total_pairs > 0
             ? static_cast<double>(early_abandoned) / total_pairs
             : 0.0;
  }
};

/// Fill a Problem's distance matrix with LB-guided early-abandon DTW.
///
/// @param prob      Problem with data loaded
/// @param band      Sakoe-Chiba band width (-1 for full DTW; disables Keogh)
/// @param lb_strat  Which lower bound(s) to apply (Auto -> Kim+Keogh cascade).
///                  None short-circuits to brute-force within the pruned fill.
/// @return Pruning statistics
PruningStats fill_distance_matrix_pruned(
    dtwc::Problem &prob,
    int band,
    dtwc::LowerBoundStrategy lb_strat = dtwc::LowerBoundStrategy::Auto);

/// Compute NxN pairwise DTW distance matrix with LB-guided early-abandon.
///
/// Standalone function for use by Python bindings (no Problem dependency).
/// Output is written to a row-major N*N array.
///
/// @param series  Vector of time series
/// @param output  Pre-allocated N*N output array (row-major)
/// @param band    Sakoe-Chiba band width (-1 for full DTW)
/// @param metric  Pointwise metric. L1/scalar L2 select the LB-guided route;
///                SquaredL2 is computed directly.
/// @return Pruning statistics
PruningStats compute_distance_matrix_pruned(
  const std::vector<std::vector<double>> &series,
  double *output,
  int band,
  MetricType metric = MetricType::L1);

} // namespace dtwc::core
