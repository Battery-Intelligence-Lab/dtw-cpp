/**
 * @file multi_pair_dtw.hpp
 * @brief Multi-pair DTW: process 4 independent DTW computations in SIMD lanes.
 *
 * @details DTW's inner recurrence is latency-bound (10 cycles/cell) because
 *          each cell depends on its left, below, and diagonal neighbors. SIMD
 *          within a single pair is limited. But when computing a distance matrix,
 *          we have N*(N-1)/2 independent pairs. By processing 4 pairs at once
 *          (one per AVX2 lane), we hide the recurrence latency.
 *
 *          The rolling buffer becomes 4-wide: each position holds a Vec<double,4>
 *          with one element per pair. The recurrence is identical across lanes.
 *
 * @date 29 Mar 2026
 */

#pragma once

#include <cstddef>
#include <vector>

#ifdef DTWC_HAS_HIGHWAY

namespace dtwc::simd {

/// Batch size for multi-pair DTW (matches AVX2 lane count for double).
constexpr std::size_t kDtwBatchSize = 4;

/// Result structure for a batch of DTW computations.
struct MultiPairResult {
  double distances[kDtwBatchSize];
};

/// Compute DTW for up to 4 pairs simultaneously using SIMD.
///
/// @param x_ptrs  Array of 4 pointers to first series in each pair.
/// @param y_ptrs  Array of 4 pointers to second series in each pair.
/// @param x_lens  Array of 4 lengths for first series.
/// @param y_lens  Array of 4 lengths for second series.
/// @param n_pairs Number of valid pairs (1-4). Unused lanes are ignored.
/// @return MultiPairResult with distances for each valid pair.
///
/// @note All series in a batch should have similar lengths for best efficiency.
///       Pairs with different lengths are handled correctly but may waste SIMD
///       lanes on padding iterations.
MultiPairResult dtw_multi_pair(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs);

}  // namespace dtwc::simd

#endif  // DTWC_HAS_HIGHWAY
