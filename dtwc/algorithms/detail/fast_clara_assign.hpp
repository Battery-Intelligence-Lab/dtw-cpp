/**
 * @file fast_clara_assign.hpp
 * @brief FastCLARA's nearest-medoid assignment, shared by fast_clara.cpp (series in RAM)
 *        and fast_clara_parquet.cpp (series streamed from Parquet, dtwc_io).
 */

#pragma once

#include "../fast_clara.hpp"
#include "../../Data.hpp"
#include "../../base/parallelisation.hpp"
#include "../../core/dtw_dispatch.hpp" // resolve_dtw_block_fn
#include "../../core/dtw_kernel.hpp"   // dtw_lanes, dtw_lanes_packed
#include "../../core/dtw_options.hpp"  // DistanceConfig
#include "../../core/medoid_assignment_policy.hpp"
#include "../../core/portable_random.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace dtwc::algorithms::detail {

/// Invocation-local PAM seed for one CLARA subsample.
inline std::uint64_t clara_pam_seed(const CLARAOptions &opts, int sample_index)
{
  return opts.random_seed + static_cast<std::uint64_t>(sample_index);
}

/// The n_samples sorted samples of sample_size indices from [0, N), drawn in turn from one engine seeded with
/// opts.random_seed. The draws are the engine's only reader (each PAM seeds its own, clara_pam_seed), so they are
/// those of a loop that ran each sample's PAM and assignment before drawing the next.
inline std::vector<std::vector<index_t>> draw_clara_samples(index_t N, index_t sample_size, const CLARAOptions &opts)
{
  std::mt19937_64 rng(opts.random_seed);
  std::vector<std::vector<index_t>> samples(static_cast<std::size_t>(opts.n_samples));
  for (auto &sample : samples) sample = core::portable_sample_indices<index_t>(N, sample_size, rng);
  return samples;
}

/**
 * Point p to its nearest medoid of one sample: `label` and `best`. medoids[m] is the sample's medoid in slot m and
 * distance(m) p's distance to it, whichever route computed it; a point is at 0 from itself.
 */
template <typename Distance>
void assign_point(index_t p, std::span<const index_t> medoids, index_t &label, double &best,
                  const Distance &distance)
{
  const auto k = static_cast<index_t>(medoids.size());
  double best_dist = std::numeric_limits<double>::max();
  index_t best_label = 0;
  bool has_best = false;

  for (index_t m = 0; m < k; ++m) {
    const index_t medoid = medoids[static_cast<std::size_t>(m)];
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

  label = best_label;
  best = best_dist;
}

/**
 * Points first .. first + n - 1 to their nearest medoid in each sample s: labels[s][p] and best[s * n + p - first], by
 * assign_point over the sample's k medoids, slots s * k .. s * k + k - 1 of `medoids` (parent indices). row_of(p)(j)
 * is point p's distance to medoid slot j, whichever device computed it. Each point writes its own slots; a failure
 * is the lowest point's, at its lowest sample.
 */
template <typename RowOf>
void assign_points(index_t first, std::size_t n, std::span<double> best, std::span<const index_t> medoids,
                   std::vector<std::vector<index_t>> &labels, const RowOf &row_of)
{
  const std::size_t samples = labels.size(), k = medoids.size() / samples;
  auto assign = [&](std::size_t i) {
    const index_t p = first + static_cast<index_t>(i);
    const auto distance = row_of(p);
    for (std::size_t s = 0; s < samples; ++s) {
      const auto slot = static_cast<index_t>(s * k); // the sample's first medoid slot
      assign_point(p, medoids.subspan(s * k, k), labels[s][static_cast<std::size_t>(p)], best[s * n + i],
                   [&distance, slot](index_t m) { return distance(slot + m); });
    }
  };
  run_openmp(assign, n, n > 64);
}

/**
 * assign_points on the CPU for series of T: point(p) and medoid(j) are the series of point p and of medoid slot j,
 * `pair` the per-pair DTW function of `distance`. Where `distance` has the fill's lane function
 * (core::resolve_dtw_block_fn), the points go W = core::dtw_lanes<T> at a time, the last repeated in the lanes past
 * the end: each medoid of every sample is x against the W in one call, into a table that assign_point reads. Without
 * a band the lanes take any lengths, but only where they are packed (core::dtw_lanes_packed); under a band, W points
 * as long as the medoid. A pair they do not take, an empty point's included, is pair(point, medoid), as without
 * lanes. Each lane is that pair's distance as dtw_dispatch.hpp states it (the recurrence transposes exactly: the
 * medoid is x here, where `pair` puts the shorter series, or on a tie the point, first), so labels and costs are the
 * per-pair route's.
 */
template <typename T, typename Pair, typename PointOf, typename MedoidOf>
void assign_points_cpu(index_t first, std::size_t n, std::span<double> best, std::span<const index_t> medoids,
                       std::vector<std::vector<index_t>> &labels, const core::DistanceConfig &distance,
                       const Pair &pair, const PointOf &point, const MedoidOf &medoid)
{
  const auto block = core::resolve_dtw_block_fn<T>(distance);
  const bool unbanded = distance.band < 0;
  if (!block || (unbanded && !core::dtw_lanes_packed)) {
    assign_points(first, n, best, medoids, labels, [&](index_t p) {
      return [&, x = point(p)](index_t j) { return pair(x, medoid(j)); };
    });
    return;
  }

  // static: GCC 13 asks the inner lambda below to capture a local it reads.
  static constexpr std::size_t W = core::dtw_lanes<T>;
  const std::size_t slots = medoids.size(), samples = labels.size(), k = slots / samples;
  auto assign_block = [&](std::size_t b) {
    const std::size_t i0 = b * W, count = std::min(W, n - i0);
    std::array<std::span<const T>, W> ys;
    std::size_t shortest = std::numeric_limits<std::size_t>::max(), longest = 0;
    for (std::size_t w = 0; w < W; ++w) {
      ys[w] = point(first + static_cast<index_t>(i0 + std::min(w, count - 1)));
      shortest = std::min(shortest, ys[w].size());
      longest = std::max(longest, ys[w].size());
    }
    thread_local std::vector<double> table; // [j * W + w]: point i0 + w to medoid slot j
    if (table.size() < slots * W) table.resize(slots * W);
    double *const t = table.data();
    for (std::size_t j = 0; j < slots; ++j) {
      const auto x = medoid(static_cast<index_t>(j));
      if (shortest > 0 && (unbanded || (shortest == x.size() && longest == x.size())))
        block(x, ys, std::span<double>(t + j * W, W));
      else
        for (std::size_t w = 0; w < count; ++w) t[j * W + w] = pair(ys[w], x);
    }
    for (std::size_t w = 0; w < count; ++w) {
      const index_t p = first + static_cast<index_t>(i0 + w);
      for (std::size_t s = 0; s < samples; ++s) {
        const double *const column = t + s * k * W + w; // sample s's slots against point i0 + w
        assign_point(p, medoids.subspan(s * k, k), labels[s][static_cast<std::size_t>(p)], best[s * n + i0 + w],
                     [column](index_t m) { return column[static_cast<std::size_t>(m) * W]; });
      }
    }
  };
  run_openmp(assign_block, (n + W - 1) / W, n > 64);
}

/// Points per call of assign_in_blocks' `assign`: their nearest distances, 8 bytes a point and sample, are its scratch.
inline constexpr std::size_t clara_block_points = 65536;

/**
 * Points first .. first + n - 1 of every sample, clara_block_points a call of assign(first, count, best), which writes
 * best[s * count + i], sample s's nearest distance of point first + i; after each call each sample's distances go into
 * its objective, costs[s], in point order: the sum one call over all the points would add.
 */
template <typename Assign>
void assign_in_blocks(index_t first, std::size_t n, std::vector<core::detail::OrderedMedoidObjective> &costs,
                      const Assign &assign)
{
  const std::size_t samples = costs.size();
  std::vector<double> best(samples * std::min(clara_block_points, n));
  for (std::size_t i0 = 0; i0 < n; i0 += clara_block_points) {
    const std::size_t count = std::min(clara_block_points, n - i0);
    const std::span<double> values(best.data(), samples * count);
    assign(first + static_cast<index_t>(i0), count, values);
    for (std::size_t s = 0; s < samples; ++s) costs[s].add(values.subspan(s * count, count));
  }
}

/**
 * The first sample with the least objective (strict <; each checked finite, in sample order): its labels move into
 * the result, with its medoids (slots s * k .. s * k + k - 1 of `medoids`) and its PAM's iterations and converged flag.
 */
inline core::ClusteringResult clara_winner(const std::vector<core::ClusteringResult> &pams,
                                           std::span<const index_t> medoids,
                                           std::vector<std::vector<index_t>> &labels,
                                           const std::vector<core::detail::OrderedMedoidObjective> &costs)
{
  std::size_t winner = 0;
  double least = 0.0;
  for (std::size_t s = 0; s < costs.size(); ++s) {
    const double cost = core::detail::finite_objective(costs[s].value(), "fast_clara");
    if (s == 0 || cost < least) {
      winner = s;
      least = cost;
    }
  }
  const std::size_t k = medoids.size() / costs.size();
  core::ClusteringResult result;
  result.labels = std::move(labels[winner]);
  result.medoid_indices.assign(medoids.begin() + static_cast<std::ptrdiff_t>(winner * k),
                               medoids.begin() + static_cast<std::ptrdiff_t>((winner + 1) * k));
  result.total_cost = least;
  result.iterations = pams[winner].iterations;
  result.converged = pams[winner].converged;
  return result;
}

/// The series of `source` at `positions`, in that order, with their names: an owned copy, or a view of them.
Data clara_sample(const Data &source, const std::vector<index_t> &positions, bool owned);

/**
 * FastPAM on each sample in turn (seed + s): sample s is a Problem with `prob`'s distance settings holding
 * clara_sample(source, positions[s], owned) (owned: a GPU uploads owned series). Its medoids, parent indices through
 * samples[s], are slots s * k .. s * k + k - 1 of `medoids`.
 */
std::vector<core::ClusteringResult> clara_pams(const Problem &prob, const CLARAOptions &opts, const Data &source,
                                               const std::vector<std::vector<index_t>> &samples,
                                               const std::vector<std::vector<index_t>> &positions, bool owned,
                                               std::vector<index_t> &medoids);

#ifdef DTWC_HAS_CUDA
/**
 * Points first .. of `series` against the medoid series `medoids` (slot j is medoid_indices[j]; sample s's are slots
 * s * k .. s * k + k - 1) on `prob`'s GPU, with its band, metric and precision, in one call: each block of distances
 * is assigned as it arrives, each sample's nearest distances going into costs[s] in point order.
 */
void assign_on_gpu(
  const Problem &prob, const std::vector<std::vector<double>> &series, index_t first,
  const std::vector<std::vector<double>> &medoids, std::span<const index_t> medoid_indices,
  std::vector<std::vector<index_t>> &labels, std::vector<core::detail::OrderedMedoidObjective> &costs);
#else
/// Metal has no kernel for the assignment, so on a GPU device it runs on the
/// CPU (the sample matrices still fill on the GPU); verbose says so, once a pass.
void say_cpu_assignment(const Problem &prob, std::size_t n_points, std::size_t n_samples, std::size_t k);
#endif

} // namespace dtwc::algorithms::detail
