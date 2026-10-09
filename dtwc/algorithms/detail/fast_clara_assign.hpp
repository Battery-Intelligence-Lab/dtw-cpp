/**
 * @file fast_clara_assign.hpp
 * @brief FastCLARA's nearest-medoid assignment, shared by fast_clara.cpp (series in RAM)
 *        and fast_clara_parquet.cpp (series streamed from Parquet, dtwc_io).
 * @author Volkan Kumtepeli
 */

#pragma once

#include "../fast_clara.hpp"
#include "../../base/parallelisation.hpp"
#include "../../core/dtw_dispatch.hpp" // resolve_dtw_block_fn
#include "../../core/dtw_kernel.hpp"   // dtw_lanes, dtw_lanes_packed
#include "../../core/dtw_options.hpp"  // DistanceConfig
#include "../../core/medoid_assignment_policy.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace dtwc::algorithms::detail {

/// Invocation-local PAM seed for one CLARA subsample.
inline std::uint64_t clara_pam_seed(const CLARAOptions &opts, int sample_index)
{
  return opts.random_seed + static_cast<std::uint64_t>(sample_index);
}

/**
 * Point p to its nearest medoid: labels[p] and `best`. distance(m) is p's
 * distance to medoid slot m, whichever route computed it; a point is at 0
 * from itself.
 */
template <typename Distance>
void assign_point(index_t p, double &best, const std::vector<index_t> &medoid_indices,
                  std::vector<index_t> &labels, const Distance &distance)
{
  const auto k = static_cast<index_t>(medoid_indices.size());
  double best_dist = std::numeric_limits<double>::max();
  index_t best_label = 0;
  bool has_best = false;

  for (index_t m = 0; m < k; ++m) {
    const index_t medoid = medoid_indices[m];
    const double d = core::detail::require_finite_medoid_distance(
      p == medoid ? 0.0 : distance(m),
      "fast_clara", static_cast<std::size_t>(p), m, medoid);
    // A medoid tied with another medoid (a duplicate series) serves
    // itself, or its own cluster would be published empty.
    if (!has_best || d < best_dist || (d == best_dist && medoid == p)) {
      best_dist = d;
      best_label = m;
      has_best = true;
    }
  }

  labels[static_cast<std::size_t>(p)] = best_label;
  best = best_dist;
}

/**
 * Points first .. first + best.size() - 1 to their nearest medoid: labels[p]
 * and best[p - first]. row_of(p)(m) is point p's distance to medoid slot m,
 * whichever device computed it. Each point writes its own slots.
 */
template <typename RowOf>
void assign_points(
  index_t first, std::span<double> best, const std::vector<index_t> &medoid_indices,
  std::vector<index_t> &labels, const RowOf &row_of)
{
  auto assign = [&](std::size_t index) {
    const index_t p = first + static_cast<index_t>(index);
    assign_point(p, best[index], medoid_indices, labels, row_of(p));
  };
  run_openmp(assign, best.size(), best.size() > 64);
}

/**
 * assign_points on the CPU for series of T: point(p) and medoid(m) are the
 * series of point p and of medoid slot m, `pair` the per-pair DTW function of
 * `distance`. Where `distance` has the fill's lane function
 * (core::resolve_dtw_block_fn), the points go W = core::dtw_lanes<T> at a time,
 * the last repeated in the lanes past the end: each medoid is x against the W in
 * one call, into a table that assign_point reads. Without a band the lanes take
 * any lengths, but only where they are packed (core::dtw_lanes_packed); under a
 * band, W points as long as the medoid. A pair they do not take, an empty
 * point's included, is pair(point, medoid), as without lanes. Each lane is that
 * pair's distance as dtw_dispatch.hpp states it (the recurrence transposes
 * exactly: the medoid is x here, where `pair` puts the shorter series, or on a
 * tie the point, first), so labels and costs are the per-pair route's.
 */
template <typename T, typename Pair, typename PointOf, typename MedoidOf>
void assign_points_cpu(
  index_t first, std::span<double> best, const std::vector<index_t> &medoid_indices,
  std::vector<index_t> &labels, const core::DistanceConfig &distance, const Pair &pair,
  const PointOf &point, const MedoidOf &medoid)
{
  const auto block = core::resolve_dtw_block_fn<T>(distance);
  const bool unbanded = distance.band < 0;
  if (!block || (unbanded && !core::dtw_lanes_packed)) {
    assign_points(first, best, medoid_indices, labels, [&](index_t p) {
      return [&, x = point(p)](index_t m) { return pair(x, medoid(m)); };
    });
    return;
  }

  // static: GCC 13 asks the inner lambda below to capture a local it reads.
  static constexpr std::size_t W = core::dtw_lanes<T>;
  const std::size_t n = best.size(), k = medoid_indices.size();
  auto assign_block = [&](std::size_t b) {
    const std::size_t i0 = b * W, count = std::min(W, n - i0);
    std::array<std::span<const T>, W> ys;
    std::size_t shortest = std::numeric_limits<std::size_t>::max(), longest = 0;
    for (std::size_t w = 0; w < W; ++w) {
      ys[w] = point(first + static_cast<index_t>(i0 + std::min(w, count - 1)));
      shortest = std::min(shortest, ys[w].size());
      longest = std::max(longest, ys[w].size());
    }
    thread_local std::vector<double> table; // [m * W + w]: point i0 + w to medoid slot m
    if (table.size() < k * W) table.resize(k * W);
    double *const t = table.data();
    for (std::size_t m = 0; m < k; ++m) {
      const auto x = medoid(static_cast<index_t>(m));
      if (shortest > 0 && (unbanded || (shortest == x.size() && longest == x.size())))
        block(x, ys, std::span<double>(t + m * W, W));
      else
        for (std::size_t w = 0; w < count; ++w) t[m * W + w] = pair(ys[w], x);
    }
    for (std::size_t w = 0; w < count; ++w)
      assign_point(first + static_cast<index_t>(i0 + w), best[i0 + w], medoid_indices, labels,
                   [t, w](index_t m) { return t[static_cast<std::size_t>(m) * W + w]; });
  };
  run_openmp(assign_block, (n + W - 1) / W, n > 64);
}

#ifdef DTWC_HAS_CUDA
/**
 * Points first .. of `series` against the k `medoids` (slot m is
 * medoid_indices[m]) on `prob`'s GPU, with its band, metric and precision;
 * each block of distances is assigned as it arrives.
 */
void assign_on_gpu(
  const Problem &prob, const std::vector<std::vector<double>> &series, index_t first,
  const std::vector<std::vector<double>> &medoids, const std::vector<index_t> &medoid_indices,
  std::vector<index_t> &labels, std::span<double> best);
#else
/// Metal has no kernel for the assignment, so on a GPU device it runs on the
/// CPU (the sample matrices still fill on the GPU); verbose says so.
void say_cpu_assignment(const Problem &prob, std::size_t n_points, std::size_t k);
#endif

} // namespace dtwc::algorithms::detail
