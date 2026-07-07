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
#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../settings.hpp"

#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dtwc::algorithms {

namespace {

/**
 * @brief Compute effective CLARA sample size per Schubert & Rousseeuw (2021) recommendation.
 *
 * @param k Number of clusters.
 * @param N Total number of data points.
 * @return Effective sample size: max(40 + 2*k, min(N, 10*k + 100)).
 */
int clara_sample_size(int k, int N)
{
  return std::max(40 + 2 * k, std::min(N, 10 * k + 100));
}

/**
 * @brief Assign all N points to the nearest medoid, computing only N*k distances.
 *
 * @param prob          Original Problem with all N series.
 * @param medoid_indices Indices (in full dataset) of the k medoids.
 * @param[out] labels   Cluster assignment per point [0, k).
 * @return Total cost (sum of distances to nearest medoid).
 */
double assign_all_points(
  Problem& prob,
  const std::vector<int>& medoid_indices,
  std::vector<int>& labels)
{
  // 64-bit size: the old `int N = static_cast<int>(prob.size())` silently
  // truncated for size() > INT_MAX (audit handoff-2026-06-01:24). Loop counters
  // stay `int` (signed-vs-signed vs N -> no signed/unsigned warning); the one
  // int-typed anchor index below narrows N explicitly.
  const int64_t N = static_cast<int64_t>(prob.size());
  static_assert(sizeof(N) >= 8, "N must stay 64-bit so static_cast<int>(size()) cannot truncate (audit R4).");
  const int k = static_cast<int>(medoid_indices.size());
  labels.resize(N);

  // ---------------------------------------------------------------------------
  // Thread-safety pre-warm (Task 0.11: parallelise the in-RAM assignment).
  //
  // distByInd() caches into prob's dense distance matrix. Its lazy
  // allocate-and-compute path is NOT thread-safe — Problem::distByInd documents
  // "call fillDistanceMatrix() before any parallel region; afterwards all calls
  // are read-only lookups". CLARA must never fill the full N*N matrix (that is
  // the whole point), so instead we serially prime exactly the cells that could
  // otherwise race:
  //   (1) an anchor pair forces the one-time matrix allocation + dtw_fn rebind
  //       (needed even when k == 1), and
  //   (2) every medoid<->medoid distance.
  // After this, the only cell two distinct loop points p != q could both target
  // is {p, q} when BOTH are medoids — and those are now already computed, so the
  // parallel loop only READS them. Every remaining write is to a distinct
  // {non-medoid, medoid} cell owned by the single thread handling that point's
  // row, matching DenseDistanceMatrix's "disjoint (i,j) pairs -> lock-free"
  // contract. No net DTW work is added: the primed cells are ones the loop needs.
  if (N > 1) {
    const int anchor_j = (medoid_indices[0] == N - 1) ? 0 : static_cast<int>(N - 1);
    prob.distByInd(medoid_indices[0], anchor_j);
    for (int a = 0; a < k; ++a)
      for (int b = a + 1; b < k; ++b)
        prob.distByInd(medoid_indices[a], medoid_indices[b]);
  }

  // Per-point nearest-medoid scan. Lock-free: thread p writes only labels[p] and
  // best_dists[p] (its own indices); shared distance-matrix writes are disjoint
  // by the pre-warm above. Only touches N*k distance entries, not the full N^2.
  std::vector<double> best_dists(static_cast<size_t>(N));

  #pragma omp parallel for schedule(static) if (N > 64)
  for (int p = 0; p < N; ++p) {
    double best_dist = std::numeric_limits<double>::max();
    int best_label = 0;

    for (int m = 0; m < k; ++m) {
      double d = prob.distByInd(p, medoid_indices[m]);
      if (d < best_dist) {
        best_dist = d;
        best_label = m;
      }
    }

    labels[p] = best_label;
    best_dists[static_cast<size_t>(p)] = best_dist;
  }

  // Serial, index-ordered reduction keeps total_cost bit-identical to the
  // original serial loop regardless of thread count (order-independent sum).
  return std::accumulate(best_dists.begin(), best_dists.end(), 0.0);
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
  std::vector<int> &labels,
  const io::ParquetChunkReader &reader,
  size_t ram_budget)
{
  const auto N = reader.total_rows();
  const int k = static_cast<int>(medoid_data.size());
  labels.resize(static_cast<size_t>(N));

  size_t medoid_bytes = 0;
  for (int m = 0; m < k; ++m)
    medoid_bytes += medoid_data.series_flat_size(m) * sizeof(data_t);
  size_t chunk_budget = (ram_budget > medoid_bytes) ? ram_budget - medoid_bytes : ram_budget / 2;

  int rg_per_batch = reader.row_groups_per_batch(chunk_budget);
  int total_rg = reader.num_row_groups();

  double total_cost = 0.0;
  int64_t global_offset = 0;

  for (int rg = 0; rg < total_rg; rg += rg_per_batch) {
    int batch_count = std::min(rg_per_batch, total_rg - rg);
    Data chunk = reader.read_row_groups(rg, batch_count);

    const int chunk_size = static_cast<int>(chunk.size());
    double chunk_cost = 0.0;

    // Inner loop is embarrassingly parallel: each point's DTW is independent.
    // Reader is NOT called here (chunk already loaded), so this is thread-safe.
    #pragma omp parallel for schedule(dynamic) reduction(+:chunk_cost) if(chunk_size > 64)
    for (int p = 0; p < chunk_size; ++p) {
      double best_dist = std::numeric_limits<double>::max();
      int best_label = 0;
      auto series_p = chunk.series(p);

      for (int m = 0; m < k; ++m) {
        double d = dtw_fn(series_p, medoid_data.series(m));
        if (d < best_dist) {
          best_dist = d;
          best_label = m;
        }
      }

      labels[static_cast<size_t>(global_offset + p)] = best_label;
      chunk_cost += best_dist;
    }
    total_cost += chunk_cost;
    global_offset += chunk_size;
  }

  return total_cost;
}

/// Float32 variant: loads chunks as float32 (2x memory saving per chunk).
double assign_all_points_chunked_f32(
  const Problem::dtw_fn_f32_t &dtw_fn_f32,
  const Data &medoid_data,
  std::vector<int> &labels,
  const io::ParquetChunkReader &reader,
  size_t ram_budget)
{
  const auto N = reader.total_rows();
  const int k = static_cast<int>(medoid_data.size());
  labels.resize(static_cast<size_t>(N));

  size_t medoid_bytes = 0;
  for (int m = 0; m < k; ++m)
    medoid_bytes += medoid_data.series_flat_size(m) * sizeof(float);
  size_t chunk_budget = (ram_budget > medoid_bytes) ? ram_budget - medoid_bytes : ram_budget / 2;

  int rg_per_batch = reader.row_groups_per_batch(chunk_budget);
  int total_rg = reader.num_row_groups();

  double total_cost = 0.0;
  int64_t global_offset = 0;

  for (int rg = 0; rg < total_rg; rg += rg_per_batch) {
    int batch_count = std::min(rg_per_batch, total_rg - rg);
    Data chunk = reader.read_row_groups_f32(rg, batch_count);

    const int chunk_size = static_cast<int>(chunk.size());
    double chunk_cost = 0.0;

    #pragma omp parallel for schedule(dynamic) reduction(+:chunk_cost) if(chunk_size > 64)
    for (int p = 0; p < chunk_size; ++p) {
      double best_dist = std::numeric_limits<double>::max();
      int best_label = 0;
      auto series_p = chunk.series_f32(p);

      for (int m = 0; m < k; ++m) {
        double d = dtw_fn_f32(series_p, medoid_data.series_f32(m));
        if (d < best_dist) {
          best_dist = d;
          best_label = m;
        }
      }

      labels[static_cast<size_t>(global_offset + p)] = best_label;
      chunk_cost += best_dist;
    }
    total_cost += chunk_cost;
    global_offset += chunk_size;
  }

  return total_cost;
}

/**
 * @brief Chunked CLARA: all data streamed from Parquet, nothing held in RAM.
 *
 * The main Problem holds settings only (band, variant, etc.).
 * Subsamples and medoid series are loaded from Parquet on demand.
 */
core::ClusteringResult fast_clara_chunked(
  Problem& prob_template,
  const CLARAOptions& opts,
  io::ParquetChunkReader& reader)
{
  const auto N = reader.total_rows();

  if (N <= 0)
    throw std::runtime_error("fast_clara_chunked: Parquet file has no rows.");
  if (opts.n_clusters <= 0 || opts.n_clusters > static_cast<int>(N))
    throw std::runtime_error("fast_clara_chunked: n_clusters out of range.");

  int64_t sample_size_64 = opts.sample_size;
  if (sample_size_64 < 0)
    sample_size_64 = clara_sample_size(opts.n_clusters, static_cast<int>(std::min(N, int64_t{INT_MAX})));
  sample_size_64 = std::max(sample_size_64, static_cast<int64_t>(opts.n_clusters));
  sample_size_64 = std::min(sample_size_64, N);
  const auto sample_size = static_cast<size_t>(sample_size_64);

  std::mt19937_64 rng(opts.random_seed);

  // Use int64_t indices for >2B row support
  // Build index pool once; use std::sample to avoid copying N elements per subsample
  std::vector<int64_t> all_indices(static_cast<size_t>(N));
  std::iota(all_indices.begin(), all_indices.end(), int64_t{0});

  core::ClusteringResult best_result;
  best_result.total_cost = std::numeric_limits<double>::max();

  for (int s = 0; s < opts.n_samples; ++s) {
    // 1. Draw random subsample indices (O(sample_size), not O(N))
    std::vector<int64_t> sample_indices;
    sample_indices.reserve(sample_size);
    std::sample(all_indices.begin(), all_indices.end(),
                std::back_inserter(sample_indices), sample_size, rng);

    // 2. Load subsample from Parquet (small — always fits in RAM)
    std::vector<int64_t> sample_rows(sample_indices.begin(), sample_indices.end());
    Data sample_data = reader.read_rows(std::move(sample_rows));

    // 3. Create sub-Problem with loaded sample data
    Problem sub_prob("clara_chunked_" + std::to_string(s));
    sub_prob.band = prob_template.band;
    sub_prob.variant_params = prob_template.variant_params;
    sub_prob.missing_strategy = prob_template.missing_strategy;
    sub_prob.distance_strategy = prob_template.distance_strategy;
    sub_prob.verbose = false; // suppress subsample verbosity
    sub_prob.set_data(std::move(sample_data));

    // 4. Run FastPAM on subsample
    auto sub_result = fast_pam(sub_prob, opts.n_clusters, opts.max_iter);

    // 5. Map medoid indices back to global dataset indices
    std::vector<int> full_medoids(opts.n_clusters);
    std::vector<int64_t> medoid_rows(opts.n_clusters);
    for (int m = 0; m < opts.n_clusters; ++m) {
      int64_t global_idx = sample_indices[sub_result.medoid_indices[m]];
      full_medoids[m] = static_cast<int>(global_idx); // ClusteringResult uses int
      medoid_rows[m] = global_idx;
    }

    // 6. Load medoid series from Parquet (k series — tiny)
    Data medoid_data = reader.read_rows(std::move(medoid_rows));

    // 7. Chunked assignment: stream row groups, compute DTW to medoids
    std::vector<int> labels;
    double total_cost;
    if (opts.use_float32) {
      // Convert medoid data to f32 (k series — tiny, negligible cost)
      const size_t n_med = medoid_data.size();
      std::vector<std::vector<float>> med_f32(n_med);
      for (size_t i = 0; i < n_med; ++i) {
        auto s = medoid_data.series(i);
        med_f32[i].resize(s.size());
        for (size_t j = 0; j < s.size(); ++j)
          med_f32[i][j] = static_cast<float>(s[j]);
      }
      std::vector<std::string> med_names(n_med);
      for (size_t i = 0; i < n_med; ++i)
        med_names[i] = std::string(medoid_data.name(i));
      Data medoid_f32(std::move(med_f32), std::move(med_names));

      total_cost = assign_all_points_chunked_f32(
        prob_template.dtw_function_f32(), medoid_f32, labels, reader, opts.ram_limit_bytes);
    } else {
      total_cost = assign_all_points_chunked(
        prob_template.dtw_function(), medoid_data, labels, reader, opts.ram_limit_bytes);
    }

    // 8. Track best result
    if (total_cost < best_result.total_cost) {
      best_result.labels = std::move(labels);
      best_result.medoid_indices = std::move(full_medoids);
      best_result.total_cost = total_cost;
      best_result.iterations = sub_result.iterations;
      best_result.converged = sub_result.converged;
    }
  }

  // 2.0 result write-back (Task 1.6): mirror the in-RAM fast_clara path so the
  // Parquet-streamed path also populates prob's labels/medoids/k (Phase 2 deletes
  // the binding auto-wire at _dtwcpp_core.cpp:615-619).
  prob_template.set_n_clusters(opts.n_clusters);
  prob_template.centroids_ind = best_result.medoid_indices;
  prob_template.clusters_ind  = best_result.labels;

  return best_result;
}
#endif // DTWC_HAS_PARQUET

} // anonymous namespace


core::ClusteringResult fast_clara(Problem& prob, const CLARAOptions& opts)
{
#ifdef DTWC_HAS_PARQUET
  // Chunked mode: stream from Parquet when ram_limit is set
  if (opts.ram_limit_bytes > 0 && !opts.parquet_path.empty()) {
    io::ParquetChunkReader reader(opts.parquet_path, opts.parquet_column);

    // If data fits in RAM, skip chunked mode
    if (reader.estimated_total_bytes() > opts.ram_limit_bytes) {
      if (prob.verbose)
        std::cout << "FastCLARA: streaming from Parquet (" << reader.total_rows()
                  << " rows, " << reader.num_row_groups() << " row groups, ~"
                  << reader.estimated_total_bytes() / (1ULL << 20) << " MB)\n";
      return fast_clara_chunked(prob, opts, reader);
    }
  }
#endif

  // 64-bit size: the old `int N = static_cast<int>(prob.size())` silently
  // truncated for size() > INT_MAX (audit handoff-2026-06-01:24). This in-RAM
  // path is int-bound (vector<int> all_indices, int sample_size), so where N
  // feeds int-typed APIs it is narrowed explicitly, clamped to INT_MAX to match
  // the chunked path's idiom; the chunked path (above) already handles N>INT_MAX.
  const int64_t N = static_cast<int64_t>(prob.size());
  static_assert(sizeof(N) >= 8, "N must stay 64-bit so static_cast<int>(size()) cannot truncate (audit R4).");

  if (N <= 0) {
    throw std::runtime_error("fast_clara: Problem has no data points.");
  }
  if (opts.n_clusters <= 0 || opts.n_clusters > N) {
    throw std::runtime_error(
      "fast_clara: n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(opts.n_clusters) + ", N=" + std::to_string(N) + ".");
  }
  if (opts.n_samples <= 0) {
    throw std::runtime_error("fast_clara: n_samples must be > 0.");
  }

  // Determine effective sample size.
  int sample_size = opts.sample_size;
  if (sample_size < 0) {
    // clara_sample_size takes int N; clamp to INT_MAX first (as the chunked path
    // does) so a >INT_MAX dataset yields min(INT_MAX, 10k+100) rather than garbage.
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    sample_size = clara_sample_size(opts.n_clusters, static_cast<int>(std::min<int64_t>(N, kIntMax))); // Schubert & Rousseeuw 2021
  }
  // Clamp to [k, N]. sample_size is small, so the int64 min result fits in int.
  sample_size = std::max(sample_size, opts.n_clusters);
  sample_size = static_cast<int>(std::min<int64_t>(sample_size, N));

  // If sample_size >= N, just run FastPAM on the full dataset.
  if (sample_size >= N) {
    return fast_pam(prob, opts.n_clusters, opts.max_iter);
  }

  // Task 0.11: use mt19937_64 + std::sample to match the chunked path exactly, so
  // the in-RAM and Parquet-streaming code paths draw the SAME subsample for a
  // given seed. Previously this path used mt19937 + std::shuffle while the chunked
  // path (above) used mt19937_64 + std::sample, so the same seed silently diverged.
  std::mt19937_64 rng(opts.random_seed);

  // All indices [0, N).
  std::vector<int> all_indices(N);
  std::iota(all_indices.begin(), all_indices.end(), 0);

  core::ClusteringResult best_result;
  best_result.total_cost = std::numeric_limits<double>::max();

  for (int s = 0; s < opts.n_samples; ++s) {
    // 1. Draw a random subsample of indices. std::sample keeps them sorted and
    //    matches the chunked path's selection bit-for-bit for the same rng.
    std::vector<int> sample_indices;
    sample_indices.reserve(static_cast<size_t>(sample_size));
    std::sample(all_indices.begin(), all_indices.end(),
                std::back_inserter(sample_indices), sample_size, rng);

    // 2. Create a sub-Problem with zero-copy span views into parent data.
    std::vector<std::string_view> sub_names;
    sub_names.reserve(sample_size);
    for (int idx : sample_indices)
      sub_names.push_back(prob.series_name(idx));   // O(1), no string copy

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
        sub_spans.push_back(prob.series(idx));        // O(1), no data copy
      sub_prob.set_view_data(Data(std::move(sub_spans), std::move(sub_names), prob.data.ndim));
    }

    // 3. Run FastPAM on the sub-Problem.
    auto sub_result = fast_pam(sub_prob, opts.n_clusters, opts.max_iter);

    // 4. Map sub-Problem medoid indices back to full dataset indices.
    std::vector<int> full_medoids(opts.n_clusters);
    for (int m = 0; m < opts.n_clusters; ++m) {
      full_medoids[m] = sample_indices[sub_result.medoid_indices[m]];
    }

    // 5. Assign ALL N points to the nearest medoid.
    std::vector<int> labels;
    double total_cost = assign_all_points(prob, full_medoids, labels);

    // 6. Track the best result.
    if (total_cost < best_result.total_cost) {
      best_result.labels = std::move(labels);
      best_result.medoid_indices = std::move(full_medoids);
      best_result.total_cost = total_cost;
      best_result.iterations = sub_result.iterations;
      best_result.converged = sub_result.converged;
    }
  }

  // 2.0 result write-back (Task 1.6): store labels/medoids/k into `prob` so
  // scores::silhouette(prob) etc. work with NO manual wiring (mirrors the binding
  // auto-wire at _dtwcpp_core.cpp:615-619, which Phase 2 deletes). The
  // sample_size>=N branch above delegates to fast_pam, which already writes back.
  prob.set_n_clusters(opts.n_clusters);
  prob.centroids_ind = best_result.medoid_indices;
  prob.clusters_ind  = best_result.labels;

  return best_result;
}

} // namespace dtwc::algorithms
