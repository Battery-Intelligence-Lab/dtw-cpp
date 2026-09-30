/**
 * @file one_batch_pam.cpp
 * @brief Fixed-batch eager PAM with O(Nm) distance storage.
 */

#include "one_batch_pam.hpp"

#include "../Problem.hpp"
#include "../core/dtw_dispatch.hpp"
#include "../core/dtw_kernel.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../core/portable_random.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace dtwc::algorithms {
namespace {

std::size_t automatic_batch_size(std::size_t n)
{
  if (n <= 1) return n;
  const auto logarithmic = static_cast<std::size_t>(
    20.0 * std::ceil(std::log2(static_cast<double>(n) + 1.0)));
  return std::min(n, std::max<std::size_t>(64, logarithmic));
}

void validate_options(std::size_t n, const OneBatchPAMOptions& options)
{
  if (n == 0)
    throw InvalidInput("one_batch_pam: Problem has no data points.");
  if (options.n_clusters <= 0 || static_cast<std::size_t>(options.n_clusters) > n) {
    throw InvalidInput(
      "one_batch_pam: n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(options.n_clusters) + ", N=" + std::to_string(n) + ".");
  }
  if (options.batch_size == 0 || options.batch_size < -1)
    throw InvalidInput("one_batch_pam: batch_size must be -1 or a positive integer.");
  // The O(Nm) promise assumes m >= k; an explicit batch below k is refused, not
  // silently raised.
  if (options.batch_size > 0 && options.batch_size < options.n_clusters)
    throw InvalidInput("one_batch_pam: batch_size must be at least n_clusters. Got batch_size="
                       + std::to_string(options.batch_size) + ", n_clusters="
                       + std::to_string(options.n_clusters) + ".");
  if (options.max_iter <= 0)
    throw InvalidInput("one_batch_pam: max_iter must be positive.");
  if (!std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.0)
    throw InvalidInput("one_batch_pam: relative_tolerance must be finite and non-negative.");
}

struct FixedBatchDistances {
  Problem& prob;
  // The dispatcher for the stored precision, resolved once and serially before
  // any OpenMP region: a legacy raw semantic mutation may require the mutable
  // getter to rebind once, and its first call validates the request (band
  // feasibility, non-finite values) with a typed error that must not be thrown
  // inside a parallel region. Workers read the stable function object; none calls
  // the accessor. Only the getter for the stored precision: the Float32 one also
  // rejects variant parameters Float64 data never narrows.
  const Problem::dtw_fn_f32_t *dtw_f32;
  const Problem::dtw_fn_t *dtw_f64;
  std::size_t n;
  std::size_t m;
  std::vector<index_t> sample;
  std::vector<index_t> sample_position;
  std::vector<double> raw;
  std::vector<double> weights;
  double scale = 1.0;
  std::uint64_t evaluations = 0;

  FixedBatchDistances(Problem& problem, std::vector<index_t> batch)
    : prob(problem),
      dtw_f32(problem.data().is_f32() ? &problem.dtw_function_f32() : nullptr),
      dtw_f64(problem.data().is_f32() ? nullptr : &problem.dtw_function()),
      n(problem.size()), m(batch.size()), sample(std::move(batch)),
      sample_position(n, -1), raw(n * m, 0.0), weights(m, 0.0)
  {
    for (std::size_t j = 0; j < m; ++j)
      sample_position[static_cast<std::size_t>(sample[j])] = static_cast<index_t>(j);

    // Each worker writes a disjoint row. Row i takes its batch columns W at a
    // time through the lane function (the entry Problem's fill uses, resolved
    // once here; each of its distances is bitwise the per-pair one) where the
    // block's series are as long as series i; every other column, and every
    // request the lanes do not cover, goes pair by pair. The block at the end of
    // the batch repeats its last column in the lanes past it, whose results are
    // dropped, and a block holding series i computes that self-pair and drops it.
    std::vector<std::uint64_t> row_evaluations(n, 0);
    std::vector<double> row_maxima(n, 0.0);
    const auto block_f32 = core::resolve_dtw_block_fn<float>(problem.distance());
    const auto block_f64 = core::resolve_dtw_block_fn<data_t>(problem.distance());
    auto fill_row = [&](std::size_t i) {
      double row_max = 0.0;
      std::uint64_t calls = 0;
      auto fill_blocks = [&](auto x, auto column, const auto &block) {
        using T = typename decltype(x)::value_type;
        constexpr std::size_t W = core::dtw_lanes<T>;
        std::array<std::span<const T>, W> ys;
        std::array<double, W> lane;
        for (std::size_t j0 = 0; j0 < m; j0 += W) {
          const std::size_t count = std::min(W, m - j0);
          bool lanes = static_cast<bool>(block);
          for (std::size_t w = 0; lanes && w < W; ++w) {
            ys[w] = column(static_cast<std::size_t>(sample[j0 + std::min(w, count - 1)]));
            lanes = ys[w].size() == x.size();
          }
          if (lanes) block(x, ys, lane);
          for (std::size_t w = 0; w < count; ++w) {
            const auto column_series = static_cast<std::size_t>(sample[j0 + w]);
            double d = 0.0;
            if (i != column_series) {
              d = lanes ? lane[w] : dtw(i, column_series);
              ++calls;
            }
            if (!std::isfinite(d) || d < 0.0)
              throw InvalidInput("one_batch_pam: distance function returned a non-finite or negative value.");
            raw[i * m + j0 + w] = d;
            row_max = std::max(row_max, d);
          }
        }
      };
      if (prob.data().is_f32())
        fill_blocks(prob.data().series_f32(i),
                    [&](std::size_t c) { return prob.data().series_f32(c); }, block_f32);
      else
        fill_blocks(prob.series(i), [&](std::size_t c) { return prob.series(c); }, block_f64);
      row_evaluations[i] = calls;
      row_maxima[i] = row_max;
    };
    run_openmp(fill_row, n, n > 64);
    const double table_max = *std::max_element(row_maxima.begin(), row_maxima.end());
    scale = table_max > 0.0 ? table_max : 1.0;
    evaluations = std::accumulate(row_evaluations.begin(), row_evaluations.end(),
                                  std::uint64_t{0});

    // Count/mean NNIW (Loog 2012): the fixed table already contains everything
    // needed to estimate each sampled point's Voronoi-cell mass. This is
    // deliberately combined with the obpam experiment code's
    // finite-table-maximum diagonal correction in estimate(). The paper's
    // literal +infinity and maintained OneBatchPAM v0.1.0 do not describe this
    // exact hybrid estimator.
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t nearest = 0;
      double best = raw[i * m];
      for (std::size_t j = 1; j < m; ++j) {
        const double d = raw[i * m + j];
        if (d < best) { best = d; nearest = j; }
      }
      weights[nearest] += 1.0;
    }
    const double mean = static_cast<double>(n) / static_cast<double>(m);
    for (double& weight : weights) weight /= mean;
  }

  double dtw(std::size_t a, std::size_t b) const
  {
    if (dtw_f32)
      return (*dtw_f32)(prob.data().series_f32(a), prob.data().series_f32(b));
    return (*dtw_f64)(prob.series(a), prob.series(b));
  }

  double estimate(std::size_t candidate, std::size_t batch_column) const
  {
    // The authors' obpam experiment code replaces d(x,x)=0 by the actual finite
    // table maximum, i.e. exactly 1 after normalization. Do not use the
    // zero-table fallback scale as an unnormalized replacement value.
    if (candidate == static_cast<std::size_t>(sample[batch_column]))
      return weights[batch_column];
    return (raw[candidate * m + batch_column] / scale) * weights[batch_column];
  }

  /// Distance of a point to a medoid: the table entry when the medoid is in the
  /// batch, else one DTW call, added to the caller's own tally `calls`. Const,
  /// so every worker may call it on its own point.
  double exact(std::size_t point, index_t medoid, std::uint64_t& calls) const
  {
    const index_t column = sample_position[static_cast<std::size_t>(medoid)];
    if (column >= 0)
      return raw[point * m + static_cast<std::size_t>(column)];
    if (point == static_cast<std::size_t>(medoid)) return 0.0;
    ++calls;
    return dtw(point, static_cast<std::size_t>(medoid));
  }
};

void nearest_two(const FixedBatchDistances& distances,
                 const std::vector<index_t>& medoids,
                 std::vector<index_t>& nearest,
                 std::vector<double>& nearest_distance,
                 std::vector<double>& second_distance)
{
  const std::size_t m = distances.m;
  const auto k = static_cast<index_t>(medoids.size());
  nearest.assign(m, 0);
  nearest_distance.assign(m, std::numeric_limits<double>::infinity());
  second_distance.assign(m, std::numeric_limits<double>::infinity());
  for (std::size_t j = 0; j < m; ++j) {
    for (index_t slot = 0; slot < k; ++slot) {
      const double d = distances.estimate(static_cast<std::size_t>(medoids[slot]), j);
      if (d < nearest_distance[j]) {
        second_distance[j] = nearest_distance[j];
        nearest_distance[j] = d;
        nearest[j] = slot;
      } else if (d < second_distance[j]) {
        second_distance[j] = d;
      }
    }
  }
}

double estimated_cost(const std::vector<double>& nearest_distance)
{
  return std::accumulate(nearest_distance.begin(), nearest_distance.end(), 0.0);
}

} // namespace

core::ClusteringResult one_batch_pam(Problem& prob,
                                     const OneBatchPAMOptions& options,
                                     OneBatchPAMStats* stats)
{
  const std::size_t n = prob.size();
  validate_options(n, options);
  const index_t k = options.n_clusters;

  if (k == static_cast<index_t>(n)) {
    core::ClusteringResult result;
    result.labels.resize(n);
    result.medoid_indices.resize(n);
    std::iota(result.labels.begin(), result.labels.end(), index_t{ 0 });
    std::iota(result.medoid_indices.begin(), result.medoid_indices.end(), index_t{ 0 });
    result.converged = true;
    prob.set_result(result);
    if (stats) *stats = OneBatchPAMStats{};
    return result;
  }

  std::size_t m = options.batch_size < 0
                    ? automatic_batch_size(n)
                    : std::min(n, static_cast<std::size_t>(options.batch_size));
  // The automatic size is an internal policy: it rises to k silently (an
  // explicit batch below k was refused above).
  m = std::max(m, static_cast<std::size_t>(k));

  std::mt19937_64 rng(options.random_seed);
  std::vector<index_t> permutation(n);
  std::iota(permutation.begin(), permutation.end(), index_t{ 0 });
  core::portable_shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<index_t> sample(permutation.begin(), permutation.begin() + static_cast<std::ptrdiff_t>(m));
  // The paper draws the candidate initialization independently of the fixed
  // batch: sampled points remain eligible, just like every other point.
  core::portable_shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<index_t> medoids(permutation.begin(), permutation.begin() + k);

  FixedBatchDistances distances(prob, std::move(sample));
  std::vector<bool> is_medoid(n, false);
  for (index_t medoid : medoids) is_medoid[static_cast<std::size_t>(medoid)] = true;

  std::vector<index_t> nearest;
  std::vector<double> nearest_distance;
  std::vector<double> second_distance;
  nearest_two(distances, medoids, nearest, nearest_distance, second_distance);

  int accepted_swaps = 0;
  int sweeps = 0;
  bool converged = false;

  if (k == 1) {
    index_t best = medoids[0];
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0; candidate < n; ++candidate) {
      double cost = 0.0;
      for (std::size_t j = 0; j < m; ++j) cost += distances.estimate(candidate, j);
      if (cost < best_cost || (cost == best_cost && static_cast<index_t>(candidate) < best)) {
        best = static_cast<index_t>(candidate);
        best_cost = cost;
      }
    }
    medoids[0] = best;
    nearest_two(distances, medoids, nearest, nearest_distance, second_distance);
    converged = true;
    sweeps = 1;
  } else {
    // C1: `base_removal_gain` and `tolerance` depend only on (nearest,
    // nearest_distance, second_distance), so they change exactly when a swap is
    // accepted — not once per candidate. Hoisting them out of the candidate loop
    // removes one heap allocation and one O(m) pass per candidate (N of each per
    // sweep); both scratch vectors are allocated once and reused. The
    // accumulation order over j is unchanged, so the result is digit-identical.
    std::vector<double> base_removal_gain(static_cast<std::size_t>(k), 0.0);
    std::vector<double> removal_gain(static_cast<std::size_t>(k), 0.0);
    double tolerance = 0.0;
    const auto refresh_swap_state = [&] {
      std::fill(base_removal_gain.begin(), base_removal_gain.end(), 0.0);
      for (std::size_t j = 0; j < m; ++j)
        base_removal_gain[static_cast<std::size_t>(nearest[j])]
          += nearest_distance[j] - second_distance[j];
      tolerance = options.relative_tolerance * estimated_cost(nearest_distance);
    };
    refresh_swap_state();

    for (; sweeps < options.max_iter; ++sweeps) {
      bool changed = false;
      for (std::size_t candidate = 0; candidate < n; ++candidate) {
        if (is_medoid[candidate]) continue;

        removal_gain = base_removal_gain;

        double add_gain = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
          const double d = distances.estimate(candidate, j);
          const index_t slot = nearest[j];
          if (d < nearest_distance[j]) {
            add_gain += nearest_distance[j] - d;
            removal_gain[static_cast<std::size_t>(slot)]
              += second_distance[j] - nearest_distance[j];
          } else if (d < second_distance[j]) {
            removal_gain[static_cast<std::size_t>(slot)] += second_distance[j] - d;
          }
        }

        const auto best_it = std::max_element(removal_gain.begin(), removal_gain.end());
        const auto slot = static_cast<index_t>(std::distance(removal_gain.begin(), best_it));
        const double gain = add_gain + *best_it;
        if (gain > tolerance) {
          is_medoid[static_cast<std::size_t>(medoids[slot])] = false;
          medoids[slot] = static_cast<index_t>(candidate);
          is_medoid[candidate] = true;
          nearest_two(distances, medoids, nearest, nearest_distance, second_distance);
          refresh_swap_state();
          ++accepted_swaps;
          changed = true;
        }
      }
      if (!changed) { converged = true; ++sweeps; break; }
    }
  }

  core::ClusteringResult result;
  result.medoid_indices = medoids;
  result.labels.resize(n);
  std::vector<double> point_cost(n, 0.0);
  std::vector<std::uint64_t> point_evaluations(n, 0);

  // Race-free by design: point p alone writes labels[p], point_cost[p] and
  // point_evaluations[p]; the batch table, the medoids and the series are
  // read-only here. The DTW call count and the objective are combined serially
  // after the region, in point order, so neither depends on the thread count.
  //
  // exact() is the one distance read here that the fixed-batch table's
  // finiteness check cannot cover: it is reached precisely when a selected
  // medoid is NOT in the batch. Unguarded, a non-finite d makes `d < best` false
  // in every slot, the point silently keeps label 0, and the run publishes a
  // wrong partition where fast_pam and fast_clara throw. run_openmp rethrows the
  // failure of the lowest point, the one a serial scan meets first.
  auto assign_point = [&](std::size_t point) {
    double best = std::numeric_limits<double>::infinity();
    index_t label = 0;
    std::uint64_t calls = 0;
    for (index_t slot = 0; slot < k; ++slot) {
      const double d = core::detail::require_finite_medoid_distance(
        distances.exact(point, medoids[slot], calls), "one_batch_pam", point, slot,
        medoids[slot]);
      if (d < best) { best = d; label = slot; }
    }
    result.labels[point] = label;
    point_cost[point] = best;
    point_evaluations[point] = calls;
  };
  run_openmp(assign_point, n, n > 64);
  distances.evaluations += std::accumulate(point_evaluations.begin(), point_evaluations.end(),
                                           std::uint64_t{0});
  // A medoid tied with another medoid (a duplicate series) serves itself, or its
  // own cluster would be published empty. Its own distance is exactly 0, so a
  // best of 0 is that tie. After the scan, so the scan's min stays branch-free.
  for (index_t slot = 0; slot < k; ++slot) {
    const auto medoid = static_cast<std::size_t>(medoids[slot]);
    if (point_cost[medoid] == 0.0) result.labels[medoid] = slot;
  }
  // Point-ordered accumulation: the published objective is a cross-route byte
  // contract, so it uses the same reassociation-proof accumulator as fast_pam
  // and fast_clara rather than a plain std::accumulate.
  result.total_cost = core::detail::ordered_medoid_objective(point_cost, "one_batch_pam");
  result.iterations = sweeps;
  result.converged = converged;

  prob.set_result(result);

  if (stats) {
    stats->batch_size = m;
    stats->distance_evaluations = distances.evaluations;
    const long double denominator = static_cast<long double>(n) * static_cast<long double>(n);
    stats->full_matrix_fraction = denominator > 0.0L
      ? static_cast<double>(static_cast<long double>(distances.evaluations) / denominator)
      : 0.0;
    stats->estimated_objective = estimated_cost(nearest_distance);
    stats->accepted_swaps = accepted_swaps;
  }
  return result;
}

} // namespace dtwc::algorithms
