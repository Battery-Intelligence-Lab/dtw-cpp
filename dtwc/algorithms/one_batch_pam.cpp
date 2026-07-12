/**
 * @file one_batch_pam.cpp
 * @brief Fixed-batch eager PAM with O(Nm) distance storage.
 */

#include "one_batch_pam.hpp"

#include "../Problem.hpp"
#include "../core/portable_random.hpp"
#include "../error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
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

void warn_batch_size_adjustment(int requested, std::size_t effective)
{
  // This is intentionally per invocation, not process-once: each call can
  // request a different invalid value, and hiding later corrections would
  // make the effective configuration silent again. Serialize the complete
  // line so concurrent OneBatchPAM calls cannot interleave their diagnostics.
  static std::mutex warning_mutex;
  const std::lock_guard<std::mutex> lock(warning_mutex);
  std::cerr
    << "[dtwc] warning: one_batch_pam requested batch_size=" << requested
    << ", but n_clusters=" << effective
    << " requires batch_size >= " << effective
    << "; using effective batch_size=" << effective
    << ". Set batch_size to at least n_clusters to avoid this adjustment.\n";
}

void validate_options(std::size_t n, const OneBatchPAMOptions& options)
{
  if (n == 0)
    throw InvalidInput("one_batch_pam: Problem has no data points.");
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw InvalidInput("one_batch_pam: N exceeds the int-indexed result API limit.");
  if (options.n_clusters <= 0 || static_cast<std::size_t>(options.n_clusters) > n) {
    throw InvalidInput(
      "one_batch_pam: n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(options.n_clusters) + ", N=" + std::to_string(n) + ".");
  }
  if (options.batch_size == 0 || options.batch_size < -1)
    throw InvalidInput("one_batch_pam: batch_size must be -1 or a positive integer.");
  if (options.max_iter <= 0)
    throw InvalidInput("one_batch_pam: max_iter must be positive.");
  if (!std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.0)
    throw InvalidInput("one_batch_pam: relative_tolerance must be finite and non-negative.");
}

struct FixedBatchDistances {
  Problem& prob;
  std::size_t n;
  std::size_t m;
  std::vector<int> sample;
  std::vector<int> sample_position;
  std::vector<double> raw;
  std::vector<double> weights;
  double scale = 1.0;
  OneBatchWeighting weighting;
  std::uint64_t evaluations = 0;

  FixedBatchDistances(Problem& problem, std::vector<int> batch,
                      OneBatchWeighting weighting_)
    : prob(problem), n(problem.size()), m(batch.size()), sample(std::move(batch)),
      sample_position(n, -1), raw(n * m, 0.0), weights(m, 1.0),
      weighting(weighting_)
  {
    for (std::size_t j = 0; j < m; ++j)
      sample_position[static_cast<std::size_t>(sample[j])] = static_cast<int>(j);

    // Resolve both getters serially before entering OpenMP: a legacy raw
    // semantic mutation may require the mutable getter to rebind once. Each
    // worker then reads stable function objects and writes a disjoint row.
    const auto &dtw_f32 = prob.dtw_function_f32();
    const auto &dtw_f64 = prob.dtw_function();
    std::vector<std::uint64_t> row_evaluations(n, 0);
    std::vector<double> row_maxima(n, 0.0);
    std::exception_ptr failure;
    #pragma omp parallel for schedule(dynamic) if(n > 64)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
      try {
        double row_max = 0.0;
        std::uint64_t calls = 0;
        for (std::size_t j = 0; j < m; ++j) {
          double d = 0.0;
          if (i != sample[j]) {
            if (prob.data.is_f32())
              d = dtw_f32(prob.data.series_f32(static_cast<std::size_t>(i)),
                          prob.data.series_f32(static_cast<std::size_t>(sample[j])));
            else
              d = dtw_f64(prob.series(static_cast<std::size_t>(i)),
                          prob.series(static_cast<std::size_t>(sample[j])));
            ++calls;
          }
          if (!std::isfinite(d) || d < 0.0)
            throw InvalidInput("one_batch_pam: distance function returned a non-finite or negative value.");
          raw[static_cast<std::size_t>(i) * m + j] = d;
          row_max = std::max(row_max, d);
        }
        row_evaluations[static_cast<std::size_t>(i)] = calls;
        row_maxima[static_cast<std::size_t>(i)] = row_max;
      } catch (...) {
        #pragma omp critical
        { if (!failure) failure = std::current_exception(); }
      }
    }
    if (failure) std::rethrow_exception(failure);
    const double table_max = *std::max_element(row_maxima.begin(), row_maxima.end());
    scale = table_max > 0.0 ? table_max : 1.0;
    evaluations = std::accumulate(row_evaluations.begin(), row_evaluations.end(),
                                  std::uint64_t{0});

    switch (weighting) {
    case OneBatchWeighting::Uniform:
    case OneBatchWeighting::Debiased:
      break;
    case OneBatchWeighting::NearestNeighbor: {
      // Count/mean NNIW (Loog 2012): the fixed table already contains
      // everything needed to estimate each sampled point's Voronoi-cell mass.
      // This is deliberately combined with the obpam experiment code's
      // finite-table-maximum diagonal correction in estimate().  The paper's
      // literal +infinity and maintained OneBatchPAM v0.1.0 do not describe
      // this exact hybrid estimator.
      std::fill(weights.begin(), weights.end(), 0.0);
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
      break;
    }
    default:
      throw std::logic_error(
        "FixedBatchDistances: unreachable OneBatchWeighting");
    }
  }

  double estimate(std::size_t candidate, std::size_t batch_column) const
  {
    switch (weighting) {
    case OneBatchWeighting::Uniform:
      break;
    case OneBatchWeighting::Debiased:
    case OneBatchWeighting::NearestNeighbor:
      if (candidate != static_cast<std::size_t>(sample[batch_column])) break;
      // The authors' obpam experiment code replaces d(x,x)=0 by the actual
      // finite table maximum, i.e. exactly 1 after normalization.  Do not use
      // the zero-table fallback scale as an unnormalized replacement value.
      return weights[batch_column];
    default:
      throw std::logic_error(
        "FixedBatchDistances::estimate: unreachable OneBatchWeighting");
    }
    return (raw[candidate * m + batch_column] / scale) * weights[batch_column];
  }

  double exact(std::size_t point, int medoid)
  {
    const int column = sample_position[static_cast<std::size_t>(medoid)];
    if (column >= 0)
      return raw[point * m + static_cast<std::size_t>(column)];
    if (point == static_cast<std::size_t>(medoid)) return 0.0;
    ++evaluations;
    if (prob.data.is_f32())
      return prob.dtw_function_f32()(prob.data.series_f32(point),
                                     prob.data.series_f32(static_cast<std::size_t>(medoid)));
    return prob.dtw_function()(prob.series(point),
                               prob.series(static_cast<std::size_t>(medoid)));
  }
};

void nearest_two(const FixedBatchDistances& distances,
                 const std::vector<int>& medoids,
                 std::vector<int>& nearest,
                 std::vector<double>& nearest_distance,
                 std::vector<double>& second_distance)
{
  const std::size_t m = distances.m;
  const int k = static_cast<int>(medoids.size());
  nearest.assign(m, 0);
  nearest_distance.assign(m, std::numeric_limits<double>::infinity());
  second_distance.assign(m, std::numeric_limits<double>::infinity());
  for (std::size_t j = 0; j < m; ++j) {
    for (int slot = 0; slot < k; ++slot) {
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
  validate_one_batch_weighting(options.weighting);
  const std::size_t n = prob.size();
  validate_options(n, options);
  const int k = options.n_clusters;

  if (k == static_cast<int>(n)) {
    core::ClusteringResult result;
    result.labels.resize(n);
    result.medoid_indices.resize(n);
    std::iota(result.labels.begin(), result.labels.end(), 0);
    std::iota(result.medoid_indices.begin(), result.medoid_indices.end(), 0);
    result.converged = true;
    prob.set_n_clusters(k);
    prob.clusters_ind = result.labels;
    prob.centroids_ind = result.medoid_indices;
    if (stats) *stats = OneBatchPAMStats{};
    return result;
  }

  std::size_t m = options.batch_size < 0
                    ? automatic_batch_size(n)
                    : std::min(n, static_cast<std::size_t>(options.batch_size));
  // The O(Nm) promise assumes m >= k. Explicitly requesting fewer evaluation
  // points than clusters is statistically weak and usually accidental, so the
  // correction must be visible. Automatic selection remains an internal policy
  // and intentionally stays silent.
  if (options.batch_size > 0 && m < static_cast<std::size_t>(k))
    warn_batch_size_adjustment(options.batch_size, static_cast<std::size_t>(k));
  m = std::max(m, static_cast<std::size_t>(k));

  std::mt19937_64 rng(options.random_seed);
  std::vector<int> permutation(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  core::portable_shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<int> sample(permutation.begin(), permutation.begin() + static_cast<std::ptrdiff_t>(m));
  // The paper draws the candidate initialization independently of the fixed
  // batch: sampled points remain eligible, just like every other point.
  core::portable_shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<int> medoids(permutation.begin(), permutation.begin() + k);

  FixedBatchDistances distances(prob, std::move(sample), options.weighting);
  std::vector<bool> is_medoid(n, false);
  for (int medoid : medoids) is_medoid[static_cast<std::size_t>(medoid)] = true;

  std::vector<int> nearest;
  std::vector<double> nearest_distance;
  std::vector<double> second_distance;
  nearest_two(distances, medoids, nearest, nearest_distance, second_distance);

  int accepted_swaps = 0;
  int sweeps = 0;
  bool converged = false;

  if (k == 1) {
    int best = medoids[0];
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0; candidate < n; ++candidate) {
      double cost = 0.0;
      for (std::size_t j = 0; j < m; ++j) cost += distances.estimate(candidate, j);
      if (cost < best_cost || (cost == best_cost && static_cast<int>(candidate) < best)) {
        best = static_cast<int>(candidate);
        best_cost = cost;
      }
    }
    medoids[0] = best;
    nearest_two(distances, medoids, nearest, nearest_distance, second_distance);
    converged = true;
    sweeps = 1;
  } else {
    for (; sweeps < options.max_iter; ++sweeps) {
      bool changed = false;
      for (std::size_t candidate = 0; candidate < n; ++candidate) {
        if (is_medoid[candidate]) continue;

        std::vector<double> removal_gain(static_cast<std::size_t>(k), 0.0);
        for (std::size_t j = 0; j < m; ++j)
          removal_gain[static_cast<std::size_t>(nearest[j])]
            += nearest_distance[j] - second_distance[j];

        double add_gain = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
          const double d = distances.estimate(candidate, j);
          const int slot = nearest[j];
          if (d < nearest_distance[j]) {
            add_gain += nearest_distance[j] - d;
            removal_gain[static_cast<std::size_t>(slot)]
              += second_distance[j] - nearest_distance[j];
          } else if (d < second_distance[j]) {
            removal_gain[static_cast<std::size_t>(slot)] += second_distance[j] - d;
          }
        }

        const auto best_it = std::max_element(removal_gain.begin(), removal_gain.end());
        const int slot = static_cast<int>(std::distance(removal_gain.begin(), best_it));
        const double gain = add_gain + *best_it;
        const double tolerance = options.relative_tolerance
          * estimated_cost(nearest_distance);
        if (gain > tolerance) {
          is_medoid[static_cast<std::size_t>(medoids[slot])] = false;
          medoids[slot] = static_cast<int>(candidate);
          is_medoid[candidate] = true;
          nearest_two(distances, medoids, nearest, nearest_distance, second_distance);
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

  // exact() updates the evaluation counter, so keep this loop serial. DTW work
  // dominates and selected medoids are frequently in the batch; correctness
  // and an exact observable count are preferable to an atomic hot path here.
  for (std::size_t point = 0; point < n; ++point) {
    double best = std::numeric_limits<double>::infinity();
    int label = 0;
    for (int slot = 0; slot < k; ++slot) {
      const double d = distances.exact(point, medoids[slot]);
      if (d < best) { best = d; label = slot; }
    }
    result.labels[point] = label;
    point_cost[point] = best;
  }
  result.total_cost = std::accumulate(point_cost.begin(), point_cost.end(), 0.0);
  result.iterations = sweeps;
  result.converged = converged;

  prob.set_n_clusters(k);
  prob.centroids_ind = result.medoid_indices;
  prob.clusters_ind = result.labels;

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
