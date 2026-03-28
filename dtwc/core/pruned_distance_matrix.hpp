/**
 * @file pruned_distance_matrix.hpp
 * @brief Distance matrix construction with cascading lower bound pruning.
 *
 * @details Builds a distance matrix using cascading lower bounds
 * (LB_Kim -> LB_Keogh -> full DTW) to skip unnecessary DTW computations.
 * Only valid for L1 metric (which is the current default).
 *
 * References:
 *   - E. Keogh, C.A. Ratanamahatana, "Exact indexing of dynamic time warping",
 *     Knowledge and Information Systems, 7(3), 358-386, 2005.
 *   - S.-W. Kim, S. Park, W.W. Chu, "An Index-Based Approach for Similarity
 *     Search Supporting Time Warping in Large Sequence Databases", ICDE 2001.
 *
 * @author Claude Code
 * @date 2026-03-28
 */

#pragma once

#include "../warping.hpp"
#include "../Problem.hpp"
#include "lower_bounds.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstddef>

namespace dtwc::core {

/// Statistics from pruned distance matrix construction.
struct PruningStats {
  size_t total_pairs = 0;         ///< Total unique pairs (upper triangle)
  size_t pruned_by_lb_kim = 0;    ///< Pairs pruned by LB_Kim
  size_t pruned_by_lb_keogh = 0;  ///< Pairs pruned by LB_Keogh
  size_t computed_full_dtw = 0;   ///< Pairs that required full DTW

  /// Fraction of pairs that were pruned (0.0 to 1.0).
  double pruning_ratio() const
  {
    return total_pairs > 0
             ? 1.0 - static_cast<double>(computed_full_dtw) / total_pairs
             : 0.0;
  }
};

/// Fill a distance matrix with LB pruning.
///
/// Precomputes envelopes for all series, then for each pair:
///   1. Check LB_Kim (O(1)) -- skip if LB > best_so_far
///   2. Check LB_Keogh (O(n)) -- skip if LB > best_so_far
///   3. Compute full DTW only if both LBs are below threshold
///
/// Note: "best_so_far" for pruning in all-pairs mode is the maximum distance
/// seen so far for the nearest neighbor of the query. This is less effective
/// than for 1-NN search (where we have a tight per-query bound).
/// Expected pruning: 40-70% for correlated time series, less for random data.
///
/// Important: LB pruning only works for L1 metric (the project's default).
/// The compatibility matrix in lower_bounds.hpp must be respected.
///
/// @param prob  Problem with data loaded
/// @param band  Sakoe-Chiba band width (-1 for full DTW, no LB_Keogh)
/// @return Pruning statistics
PruningStats fill_distance_matrix_pruned(dtwc::Problem &prob, int band);

} // namespace dtwc::core
