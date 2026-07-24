/**
 * @file fast_clara.cpp
 * @brief Implementation of FastCLARA: scalable k-medoids via subsampling + FastPAM.
 *
 * @details For each of n_samples subsamples:
 *   1. Draw sample_size random indices from [0, N).
 *   2. Create a sub-Problem containing only the sampled series.
 *   3. Run FastPAM on the sub-Problem to find medoids.
 *   4. Map sub-Problem medoid indices back to original dataset indices.
 *   5. Assign ALL N points to the nearest medoid (computing only N*k distances).
 *   6. Track the result with the lowest total cost across all subsamples.
 *
 * References:
 *   - Kaufman & Rousseeuw (1990), "Finding Groups in Data."
 *   - Schubert & Rousseeuw (2021), JMLR 22(1), 4653-4688.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @author Claude 4.6
 * @date 29 Mar 2026
 */

#include "fast_clara.hpp"
#include "detail/fast_clara_plan.hpp"
#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../core/portable_random.hpp"
#include "../error.hpp"

#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dtwc::algorithms {

namespace detail {

  void validate_clara_controls(
    const CLARAOptions &options, std::string_view caller)
  {
    const std::string prefix(caller);
    if (options.n_samples <= 0)
      throw InvalidInput(prefix + ": n_samples must be positive.");
    if (options.sample_size == 0 || options.sample_size < -1)
      throw InvalidInput(
        prefix + ": sample_size must be -1 or a positive integer.");
    if (options.max_iter <= 0)
      throw InvalidInput(prefix + ": max_iter must be positive.");
  }

  ClaraPlan resolve_clara_plan(
    std::int64_t n_points, const CLARAOptions &options,
    std::string_view caller)
  {
    validate_clara_controls(options, caller);
    const std::string prefix(caller);
    if (n_points <= 0)
      throw InvalidInput(prefix + ": Problem has no data points.");
    if (n_points > std::numeric_limits<int>::max())
      throw InvalidInput(
        prefix + ": N exceeds the int-indexed clustering result limit.");
    if (options.n_clusters <= 0 || options.n_clusters > n_points) {
      throw InvalidInput(
        prefix + ": n_clusters must be in [1, N]. Got n_clusters="
        + std::to_string(options.n_clusters) + ", N="
        + std::to_string(n_points) + ".");
    }

    const auto n = static_cast<int>(n_points);
    std::int64_t requested = options.sample_size;
    if (requested == -1) {
      const std::int64_t k = options.n_clusters;
      const std::int64_t classical = 40 + 2 * k;
      const std::int64_t improved = std::min<std::int64_t>(n_points, 10 * k + 100);
      requested = std::max(classical, improved);
    }
    requested = std::max<std::int64_t>(requested, options.n_clusters);
    requested = std::min(requested, n_points);
    return { n, static_cast<int>(requested) };
  }

  void validate_streaming_clara_plan(
    const ClaraPlan &plan, std::string_view caller)
  {
    if (plan.sample_size == plan.n_points) {
      const std::string prefix(caller);
      throw InvalidInput(
        prefix +
        ": sample_size resolves to N, but the Parquet dataset exceeds "
        "ram_limit_bytes; use sample_size < N or raise the RAM limit for the "
        "single full-data PAM fallback.");
    }
  }

} // namespace detail

namespace {

  void capture_assignment_failure(
    std::exception_ptr candidate, std::int64_t point,
    std::exception_ptr &failure, std::int64_t &failure_point)
  {
#pragma omp critical(dtwc_medoid_assignment_failure)
    {
      if (point < failure_point) {
        failure_point = point;
        failure = std::move(candidate);
      }
    }
  }

#ifdef DTWC_HAS_PARQUET
  size_t resident_data_bytes(const Data &data, size_t element_bytes)
  {
    const size_t object_bytes = data.is_f32()
      ? sizeof(std::vector<float>) + sizeof(std::string)
      : sizeof(std::vector<data_t>) + sizeof(std::string);
    if (data.size() > std::numeric_limits<size_t>::max() / object_bytes)
      return std::numeric_limits<size_t>::max();
    size_t total = data.size() * object_bytes;
    for (size_t i = 0; i < data.size(); ++i) {
      const size_t count = data.series_flat_size(i);
      if (count > (std::numeric_limits<size_t>::max() - total) / element_bytes)
        return std::numeric_limits<size_t>::max();
      total += count * element_bytes;
    }
    return total;
  }
#endif

  /// Invocation-local PAM seed for one CLARA subsample. CLARAOptions uses an
  /// unsigned base and n_samples is a positive int, so widening both operands to
  /// uint64_t makes the addition overflow-safe for every representable input.
  std::uint64_t clara_pam_seed(const CLARAOptions &opts, int sample_index)
  {
    return static_cast<std::uint64_t>(opts.random_seed)
           + static_cast<std::uint64_t>(sample_index);
  }

  template <typename Distance, typename SeriesAt>
  double assign_all_points_direct(
    int n_points, const std::vector<int> &medoid_indices,
    std::vector<int> &labels, const Distance &distance, SeriesAt series_at)
  {
    const int k = static_cast<int>(medoid_indices.size());
    labels.resize(static_cast<std::size_t>(n_points));
    std::vector<double> best_dists(static_cast<std::size_t>(n_points));
    std::exception_ptr failure;
    std::int64_t failure_point = n_points;

#pragma omp parallel for schedule(static) if (n_points > 64)
    for (int p = 0; p < n_points; ++p) {
      try {
        double best_dist = std::numeric_limits<double>::max();
        int best_label = 0;
        bool has_best = false;
        const auto point = series_at(p);

        for (int m = 0; m < k; ++m) {
          const int medoid = medoid_indices[m];
          const double d = core::detail::require_finite_medoid_distance(
            p == medoid ? 0.0 : distance(point, series_at(medoid)),
            "fast_clara", p, m, medoid);
          if (!has_best || d < best_dist) {
            best_dist = d;
            best_label = m;
            has_best = true;
          }
        }

        labels[p] = best_label;
        best_dists[static_cast<std::size_t>(p)] = best_dist;
      } catch (...) {
        capture_assignment_failure(
          std::current_exception(), p, failure, failure_point);
      }
    }

    if (failure) std::rethrow_exception(failure);
    return core::detail::ordered_medoid_objective(
      best_dists, "fast_clara");
  }

  /** Assign through the bound DTW function without allocating the parent cache. */
  double assign_all_points(
    Problem &prob, const std::vector<int> &medoid_indices,
    std::vector<int> &labels)
  {
    if (prob.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw InvalidInput(
        "fast_clara: N exceeds the int-indexed clustering result limit.");
    const int n_points = static_cast<int>(prob.size());
    if (prob.data.is_f32()) {
      const auto &distance = prob.dtw_function_f32();
      return assign_all_points_direct(
        n_points, medoid_indices, labels, distance, [&prob](int index) { return prob.data.series_f32(index); });
    }
    const auto &distance = prob.dtw_function();
    return assign_all_points_direct(
      n_points, medoid_indices, labels, distance, [&prob](int index) { return prob.series(index); });
  }

#ifdef DTWC_HAS_PARQUET
  /**
   * @brief Chunked assignment: stream Parquet row groups, compute DTW to medoids.
   *
   * Loads one batch of row groups at a time within the RAM budget.
   * Each chunk's DTW distances to medoids are computed and discarded.
   *
   * @param dtw_fn       Bound DTW function (float64).
   * @param medoid_data  Data containing only the k medoid series.
   * @param[out] labels  Cluster assignment per point [0, k) for all N points.
   * @param reader       Parquet chunk reader (already opened).
   * @param ram_budget   Available bytes for chunk data.
   * @return Total cost (sum of distances to nearest medoid).
   */
  double assign_all_points_chunked(
    const Problem::dtw_fn_t &dtw_fn,
    const Data &medoid_data,
    const std::vector<int> &medoid_indices,
    std::vector<int> &labels,
    const io::ParquetChunkReader &reader,
    size_t ram_budget)
  {
    const auto N = reader.logical_series_count();
    const int k = static_cast<int>(medoid_data.size());
    labels.resize(static_cast<size_t>(N));

    const size_t medoid_bytes = resident_data_bytes(medoid_data, sizeof(data_t));
    if (medoid_bytes >= ram_budget)
      throw InvalidInput(
        "fast_clara: ram_limit_bytes is too small to retain the selected "
        "medoid series during chunked assignment.");
    const size_t chunk_budget = ram_budget - medoid_bytes;

    int rg_per_batch = reader.row_groups_per_batch(chunk_budget, false);
    int total_rg = reader.num_row_groups();

    core::detail::OrderedMedoidObjective total_cost("fast_clara");
    std::vector<double> best_dists;
    int64_t global_offset = 0;

    for (int rg = 0; rg < total_rg; rg += rg_per_batch) {
      int batch_count = std::min(rg_per_batch, total_rg - rg);
      Data chunk = reader.read_row_groups(rg, batch_count);

      const int chunk_size = static_cast<int>(chunk.size());
      best_dists.resize(static_cast<size_t>(chunk_size));
      std::exception_ptr failure;
      std::int64_t failure_point = global_offset + chunk_size;

// Inner loop is embarrassingly parallel: each point's DTW is independent.
// Reader is NOT called here (chunk already loaded), so this is thread-safe.
#pragma omp parallel for schedule(dynamic) if (chunk_size > 64)
      for (int p = 0; p < chunk_size; ++p) {
        const auto global_index = global_offset + p;
        try {
          double best_dist = std::numeric_limits<double>::max();
          int best_label = 0;
          bool has_best = false;
          auto series_p = chunk.series(p);

          for (int m = 0; m < k; ++m) {
            const double d = core::detail::require_finite_medoid_distance(
              global_index == medoid_indices[m]
                ? 0.0 : dtw_fn(series_p, medoid_data.series(m)),
              "fast_clara", static_cast<std::size_t>(global_index),
              m, medoid_indices[m]);
            if (!has_best || d < best_dist) {
              best_dist = d;
              best_label = m;
              has_best = true;
            }
          }

          labels[static_cast<size_t>(global_index)] = best_label;
          best_dists[static_cast<size_t>(p)] = best_dist;
        } catch (...) {
          capture_assignment_failure(
            std::current_exception(), global_index, failure, failure_point);
        }
      }
      if (failure) std::rethrow_exception(failure);
      total_cost.add(best_dists);
      global_offset += chunk_size;
    }

    return total_cost.value();
  }

  /// Float32 variant: loads chunks as float32 (2x memory saving per chunk).
  double assign_all_points_chunked_f32(
    const Problem::dtw_fn_f32_t &dtw_fn_f32,
    const Data &medoid_data,
    const std::vector<int> &medoid_indices,
    std::vector<int> &labels,
    const io::ParquetChunkReader &reader,
    size_t ram_budget)
  {
    const auto N = reader.logical_series_count();
    const int k = static_cast<int>(medoid_data.size());
    labels.resize(static_cast<size_t>(N));

    const size_t medoid_bytes = resident_data_bytes(medoid_data, sizeof(float));
    if (medoid_bytes >= ram_budget)
      throw InvalidInput(
        "fast_clara: ram_limit_bytes is too small to retain the selected "
        "Float32 medoid series during chunked assignment.");
    const size_t chunk_budget = ram_budget - medoid_bytes;

    int rg_per_batch = reader.row_groups_per_batch(chunk_budget, true);
    int total_rg = reader.num_row_groups();

    core::detail::OrderedMedoidObjective total_cost("fast_clara");
    std::vector<double> best_dists;
    int64_t global_offset = 0;

    for (int rg = 0; rg < total_rg; rg += rg_per_batch) {
      int batch_count = std::min(rg_per_batch, total_rg - rg);
      Data chunk = reader.read_row_groups_f32(rg, batch_count);

      const int chunk_size = static_cast<int>(chunk.size());
      best_dists.resize(static_cast<size_t>(chunk_size));
      std::exception_ptr failure;
      std::int64_t failure_point = global_offset + chunk_size;

#pragma omp parallel for schedule(dynamic) if (chunk_size > 64)
      for (int p = 0; p < chunk_size; ++p) {
        const auto global_index = global_offset + p;
        try {
          double best_dist = std::numeric_limits<double>::max();
          int best_label = 0;
          bool has_best = false;
          auto series_p = chunk.series_f32(p);

          for (int m = 0; m < k; ++m) {
            const double d = core::detail::require_finite_medoid_distance(
              global_index == medoid_indices[m]
                ? 0.0 : dtw_fn_f32(series_p, medoid_data.series_f32(m)),
              "fast_clara", static_cast<std::size_t>(global_index),
              m, medoid_indices[m]);
            if (!has_best || d < best_dist) {
              best_dist = d;
              best_label = m;
              has_best = true;
            }
          }

          labels[static_cast<size_t>(global_index)] = best_label;
          best_dists[static_cast<size_t>(p)] = best_dist;
        } catch (...) {
          capture_assignment_failure(
            std::current_exception(), global_index, failure, failure_point);
        }
      }
      if (failure) std::rethrow_exception(failure);
      total_cost.add(best_dists);
      global_offset += chunk_size;
    }

    return total_cost.value();
  }

  /**
   * @brief Chunked CLARA: all data streamed from Parquet, nothing held in RAM.
   *
   * The main Problem holds settings only (band, variant, etc.).
   * Subsamples and medoid series are loaded from Parquet on demand.
   */
  core::ClusteringResult fast_clara_chunked(
    Problem &prob_template,
    const CLARAOptions &opts,
    io::ParquetChunkReader &reader,
    const detail::ClaraPlan &plan)
  {
    const int N = plan.n_points;
    const int sample_size = plan.sample_size;
    detail::validate_streaming_clara_plan(plan, "fast_clara");

    std::mt19937_64 rng(opts.random_seed);

    core::ClusteringResult best_result;
    best_result.total_cost = std::numeric_limits<double>::max();
    bool best_present = false;

    for (int s = 0; s < opts.n_samples; ++s) {
      // 1. Stable O(N)-time selection with O(sample_size) sampling scratch.  The
      // result still necessarily owns O(N) labels, but the old 8*N-byte index
      // pool was avoidable in the streaming path.
      auto sample_indices = core::portable_sample_indices<int64_t>(
        N, sample_size, rng);

      // 2. Load subsample from Parquet (small — always fits in RAM)
      core::ClusteringResult sub_result;
      {
        // Release the sample series and O(s^2) PAM cache before medoid and
        // assignment chunks are materialized; otherwise streaming peaks add.
        std::vector<int64_t> sample_rows(
          sample_indices.begin(), sample_indices.end());
        Data sample_data = opts.use_float32
          ? reader.read_rows_f32(std::move(sample_rows), opts.ram_limit_bytes)
          : reader.read_rows(std::move(sample_rows), opts.ram_limit_bytes);

        Problem sub_prob("clara_chunked_" + std::to_string(s));
        sub_prob.band = prob_template.band;
        sub_prob.variant_params = prob_template.variant_params;
        sub_prob.missing_strategy = prob_template.missing_strategy;
        sub_prob.distance_strategy = prob_template.distance_strategy;
        sub_prob.verbose = false;
        sub_prob.set_data(std::move(sample_data));
        sub_result = fast_pam_seeded(
          sub_prob, opts.n_clusters, clara_pam_seed(opts, s), opts.max_iter);
      }

      // 5. Map medoid indices back to global dataset indices
      std::vector<int> full_medoids(opts.n_clusters);
      std::vector<int64_t> medoid_rows(opts.n_clusters);
      for (int m = 0; m < opts.n_clusters; ++m) {
        int64_t global_idx = sample_indices[sub_result.medoid_indices[m]];
        full_medoids[m] = static_cast<int>(global_idx); // ClusteringResult uses int
        medoid_rows[m] = global_idx;
      }

      // 6. Load medoid series from Parquet (k series — tiny)
      Data medoid_data = opts.use_float32
        ? reader.read_rows_f32(std::move(medoid_rows), opts.ram_limit_bytes)
        : reader.read_rows(std::move(medoid_rows), opts.ram_limit_bytes);

      // 7. Chunked assignment: stream row groups, compute DTW to medoids
      std::vector<int> labels;
      double total_cost;
      if (opts.use_float32) {
        total_cost = assign_all_points_chunked_f32(
          prob_template.dtw_function_f32(), medoid_data, full_medoids, labels,
          reader, opts.ram_limit_bytes);
      } else {
        total_cost = assign_all_points_chunked(
          prob_template.dtw_function(), medoid_data, full_medoids, labels,
          reader, opts.ram_limit_bytes);
      }

      // 8. Track best result
      if (!best_present || total_cost < best_result.total_cost) {
        best_result.labels = std::move(labels);
        best_result.medoid_indices = std::move(full_medoids);
        best_result.total_cost = total_cost;
        best_result.iterations = sub_result.iterations;
        best_result.converged = sub_result.converged;
        best_present = true;
      }
    }

    // 2.0 result write-back (Task 1.6): mirror the in-RAM fast_clara path so the
    // Parquet-streamed path also populates prob's labels/medoids/k (Phase 2 deletes
    // the binding auto-wire at _dtwcpp_core.cpp:615-619).
    prob_template.set_n_clusters(opts.n_clusters);
    prob_template.centroids_ind = best_result.medoid_indices;
    prob_template.clusters_ind = best_result.labels;

    return best_result;
  }
#endif // DTWC_HAS_PARQUET

} // anonymous namespace


core::ClusteringResult fast_clara(Problem &prob, const CLARAOptions &opts)
{
  // Validate caller-controlled fields before opening a Parquet reader or doing
  // any other I/O. Dataset-size validation follows against the selected source.
  detail::validate_clara_controls(opts, "fast_clara");
  if (opts.force_parquet_streaming
      && (opts.ram_limit_bytes == 0 || opts.parquet_path.empty()))
    throw InvalidInput(
      "fast_clara: force_parquet_streaming requires ram_limit_bytes and "
      "parquet_path.");
  if (opts.force_parquet_streaming && prob.size() != 0)
    throw InvalidInput(
      "fast_clara: force_parquet_streaming requires a settings-only Problem "
      "without resident series.");
#ifndef DTWC_HAS_PARQUET
  if (opts.force_parquet_streaming)
    throw InvalidInput(
      "fast_clara: force_parquet_streaming requires a build with Parquet support.");
#endif
#ifdef DTWC_HAS_PARQUET
  // Chunked mode: stream from Parquet when ram_limit is set
  if (opts.ram_limit_bytes > 0 && !opts.parquet_path.empty()) {
    io::ParquetChunkReader reader(opts.parquet_path, opts.parquet_column);
    const auto resident_bytes =
      reader.estimated_resident_bytes(opts.use_float32);

    // If data fits in RAM, skip chunked mode
    if (opts.force_parquet_streaming
        || resident_bytes > opts.ram_limit_bytes) {
      if (prob.size() != 0)
        throw InvalidInput(
          "fast_clara: Parquet streaming was selected, but Problem still "
          "contains resident series; use a settings-only Problem to avoid "
          "resident-plus-chunk memory.");
      if (!reader.is_list_layout()) {
        throw InvalidInput(
          "fast_clara: RAM-limited Parquet streaming requires list-per-row "
          "input; a scalar column is one time series and cannot be streamed "
          "as independent clustering points.");
      }
      const auto stream_plan = detail::resolve_clara_plan(
        reader.logical_series_count(), opts, "fast_clara");
      if (prob.verbose)
        std::cout << "FastCLARA: streaming from Parquet ("
                  << reader.logical_series_count()
                  << " rows, " << reader.num_row_groups() << " row groups, ~"
                  << resident_bytes / (1ULL << 20) << " MB resident estimate)\n";
      return fast_clara_chunked(prob, opts, reader, stream_plan);
    }
  }
#endif

  if (prob.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw InvalidInput(
      "fast_clara: N exceeds the int-indexed clustering result limit.");
  const auto plan = detail::resolve_clara_plan(
    static_cast<std::int64_t>(prob.size()), opts, "fast_clara");
  const int N = plan.n_points;
  const int sample_size = plan.sample_size;

  // If sample_size >= N, just run FastPAM on the full dataset.
  if (sample_size >= N) {
    return fast_pam_seeded(
      prob, opts.n_clusters, clara_pam_seed(opts, 0), opts.max_iter);
  }

  // One portable map and stable selection scan keep the in-RAM and streaming
  // paths bit-identical across standard-library implementations.
  std::mt19937_64 rng(opts.random_seed);

  core::ClusteringResult best_result;
  best_result.total_cost = std::numeric_limits<double>::max();
  bool best_present = false;

  for (int s = 0; s < opts.n_samples; ++s) {
    // 1. Draw a sorted sample using the same map as the chunked path.
    auto sample_indices = core::portable_sample_indices<int>(
      static_cast<int>(N), sample_size, rng);

    // 2. Create a sub-Problem with zero-copy span views into parent data.
    std::vector<std::string_view> sub_names;
    sub_names.reserve(sample_size);
    for (int idx : sample_indices)
      sub_names.push_back(prob.series_name(idx)); // O(1), no string copy

    Problem sub_prob("clara_subsample_" + std::to_string(s));
    // Copy all relevant settings from the original problem.
    sub_prob.band = prob.band;
    sub_prob.variant_params = prob.variant_params;
    sub_prob.missing_strategy = prob.missing_strategy;
    sub_prob.distance_strategy = prob.distance_strategy;
    sub_prob.verbose = prob.verbose;

    if (prob.data.is_f32()) {
      std::vector<std::span<const float>> sub_spans;
      sub_spans.reserve(sample_size);
      for (int idx : sample_indices)
        sub_spans.push_back(prob.data.series_f32(idx));
      sub_prob.set_view_data(Data(std::move(sub_spans), std::move(sub_names), prob.data.ndim));
    } else {
      std::vector<std::span<const data_t>> sub_spans;
      sub_spans.reserve(sample_size);
      for (int idx : sample_indices)
        sub_spans.push_back(prob.series(idx)); // O(1), no data copy
      sub_prob.set_view_data(Data(std::move(sub_spans), std::move(sub_names), prob.data.ndim));
    }

    // 3. Run FastPAM on the sub-Problem.
    auto sub_result = fast_pam_seeded(
      sub_prob, opts.n_clusters, clara_pam_seed(opts, s), opts.max_iter);

    // 4. Map sub-Problem medoid indices back to full dataset indices.
    std::vector<int> full_medoids(opts.n_clusters);
    for (int m = 0; m < opts.n_clusters; ++m) {
      full_medoids[m] = sample_indices[sub_result.medoid_indices[m]];
    }

    // 5. Assign ALL N points to the nearest medoid.
    std::vector<int> labels;
    double total_cost = assign_all_points(prob, full_medoids, labels);

    // 6. Track the best result.
    if (!best_present || total_cost < best_result.total_cost) {
      best_result.labels = std::move(labels);
      best_result.medoid_indices = std::move(full_medoids);
      best_result.total_cost = total_cost;
      best_result.iterations = sub_result.iterations;
      best_result.converged = sub_result.converged;
      best_present = true;
    }
  }

  // 2.0 result write-back (Task 1.6): store labels/medoids/k into `prob` so
  // scores::silhouette(prob) etc. work with NO manual wiring (mirrors the binding
  // auto-wire at _dtwcpp_core.cpp:615-619, which Phase 2 deletes). The
  // sample_size>=N branch above delegates to fast_pam, which already writes back.
  prob.set_n_clusters(opts.n_clusters);
  prob.centroids_ind = best_result.medoid_indices;
  prob.clusters_ind = best_result.labels;

  return best_result;
}

} // namespace dtwc::algorithms
