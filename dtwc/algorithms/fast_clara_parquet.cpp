/**
 * @file fast_clara_parquet.cpp
 * @brief FastCLARA over series streamed from Parquet: dtwc_io, in a build with Parquet.
 *
 * @details The RAM-limited route of dtwc_cl --ram-limit. The main Problem holds settings only; each
 *   sample, the medoids and the assignment's row groups are read from the file as they are needed.
 *   It moved out of fast_clara.cpp so that the core, which the Python module and the MEX link, holds
 *   no Parquet; the rest of FastCLARA is fast_clara.cpp's, which this file calls.
 */

#ifdef DTWC_HAS_PARQUET

#include "fast_clara.hpp"
#include "detail/fast_clara_assign.hpp"
#include "detail/fast_clara_plan.hpp"
#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../base/error.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../core/portable_random.hpp"
#include "../io/parquet_chunk_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace dtwc::algorithms {

namespace {

  size_t resident_data_bytes(const Data &data, size_t element_bytes)
  {
    const size_t object_bytes = data.is_f32()
      ? sizeof(std::vector<float>) + sizeof(std::string)
      : sizeof(std::vector<data_t>) + sizeof(std::string);
    size_t total = data.size() * object_bytes;
    for (size_t i = 0; i < data.size(); ++i)
      total += data.series_flat_size(i) * element_bytes;
    return total;
  }

  /**
   * @brief Chunked assignment: stream Parquet row groups, compute DTW to medoids.
   *
   * Loads one batch of row groups at a time within the RAM budget.
   * Each chunk's DTW distances to medoids are computed and discarded.
   *
   * @param prob         Settings-only Problem: its device computes the distances.
   * @param dtw_fn       Bound DTW function (float64).
   * @param medoid_data  Data containing only the k medoid series.
   * @param[out] labels  Cluster assignment per point [0, k) for all N points.
   * @param reader       Parquet chunk reader (already opened).
   * @param ram_budget   Available bytes for chunk data.
   * @return Total cost (sum of distances to nearest medoid).
   */
  /**
   * @brief Chunked nearest-medoid assignment streamed from Parquet.
   *
   * One body for both precisions. `F32` selects the reader entry point, the
   * per-series accessor and the resident-byte size at COMPILE time, so each
   * instantiation emits exactly the loop the two hand-written copies emitted —
   * no runtime branch enters the per-element inner loop.
   */
  template <bool F32, typename DtwFn>
  double assign_all_points_chunked(
    const Problem &prob,
    const DtwFn &dtw_fn,
    const Data &medoid_data,
    const std::vector<index_t> &medoid_indices,
    std::vector<index_t> &labels,
    const io::ParquetChunkReader &reader,
    size_t ram_budget)
  {
    const auto series_at = [](const Data &d, index_t index) {
      if constexpr (F32)
        return d.series_f32(static_cast<size_t>(index));
      else
        return d.series(static_cast<size_t>(index));
    };

    const auto N = reader.logical_series_count();
    labels.resize(static_cast<size_t>(N));
#ifndef DTWC_HAS_CUDA
    detail::say_cpu_assignment(prob, static_cast<std::size_t>(N), medoid_indices.size());
#endif

    const size_t medoid_bytes =
      resident_data_bytes(medoid_data, F32 ? sizeof(float) : sizeof(data_t));
    if (medoid_bytes >= ram_budget) {
      if constexpr (F32)
        throw InvalidInput(
          "fast_clara: ram_limit_bytes is too small to retain the selected "
          "Float32 medoid series during chunked assignment.");
      else
        throw InvalidInput(
          "fast_clara: ram_limit_bytes is too small to retain the selected "
          "medoid series during chunked assignment.");
    }
    const size_t chunk_budget = ram_budget - medoid_bytes;

    int rg_per_batch = reader.row_groups_per_batch(chunk_budget, F32);
    int total_rg = reader.num_row_groups();

    core::detail::OrderedMedoidObjective total_cost;
    std::vector<double> best_dists;
    int64_t global_offset = 0;

    for (int rg = 0; rg < total_rg; rg += rg_per_batch) {
      int batch_count = std::min(rg_per_batch, total_rg - rg);
      Data chunk = F32 ? reader.read_row_groups<float>(rg, batch_count)
                       : reader.read_row_groups(rg, batch_count);

      const index_t chunk_size = chunk.size();
      best_dists.resize(static_cast<size_t>(chunk_size));

      // The chunk is loaded: the assignment reads it, never the reader.
#ifdef DTWC_HAS_CUDA
      bool on_gpu = false;
      if constexpr (!F32) // Float32 series never reach a GPU: the sample fill refuses them first
        on_gpu = prob.device().first == Device::GPU;
      if (on_gpu)
        detail::assign_on_gpu(prob, chunk.p_vec, global_offset, medoid_data.p_vec, medoid_indices,
                              labels, best_dists);
      else
#endif
        detail::assign_points(global_offset, best_dists, medoid_indices, labels, [&](index_t p) {
          return [&, point = series_at(chunk, p - global_offset)](index_t m) {
            return dtw_fn(point, series_at(medoid_data, m));
          };
        });
      total_cost.add(best_dists);
      global_offset += chunk_size;
    }

    return core::detail::finite_objective(total_cost.value(), "fast_clara");
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
    const index_t N = plan.n_points;
    const index_t sample_size = plan.sample_size;
    detail::validate_streaming_clara_plan(plan, "fast_clara");

    std::mt19937_64 rng(opts.random_seed);

    core::ClusteringResult best_result;
    best_result.total_cost = std::numeric_limits<double>::max();
    bool best_present = false;

    for (int s = 0; s < opts.n_samples; ++s) {
      // 1. Stable O(N)-time selection with O(sample_size) sampling scratch.  The
      // result still necessarily owns O(N) labels, but the old 8*N-byte index
      // pool was avoidable in the streaming path.
      auto sample_indices = core::portable_sample_indices<index_t>(
        N, sample_size, rng);

      // 2. Load subsample from Parquet (small — always fits in RAM)
      core::ClusteringResult sub_result;
      {
        // Release the sample series and O(s^2) PAM cache before medoid and
        // assignment chunks are materialized; otherwise streaming peaks add.
        std::vector<int64_t> sample_rows(
          sample_indices.begin(), sample_indices.end());
        Data sample_data = opts.use_float32
          ? reader.read_rows<float>(std::move(sample_rows), opts.ram_limit_bytes)
          : reader.read_rows(std::move(sample_rows), opts.ram_limit_bytes);

        Problem sub_prob("clara_chunked_" + std::to_string(s));
        sub_prob.set_distance(prob_template.distance());
        // Device, GPU index and precision: the sample fill honours them, or
        // validate_fill_request refuses them (e.g. a Float32 sample on a GPU).
        const auto [device, index] = prob_template.device();
        sub_prob.set_device(device, index);
        sub_prob.set_gpu_precision(prob_template.gpu_precision());
        sub_prob.set_verbose(prob_template.verbose());
        sub_prob.set_data(std::move(sample_data));
        sub_result = fast_pam_seeded(
          sub_prob, opts.n_clusters, detail::clara_pam_seed(opts, s), opts.max_iter);
      }

      // 5. Map medoid indices back to global dataset indices
      std::vector<index_t> full_medoids(opts.n_clusters);
      std::vector<int64_t> medoid_rows(opts.n_clusters);
      for (index_t m = 0; m < opts.n_clusters; ++m) {
        const index_t global_idx = sample_indices[sub_result.medoid_indices[m]];
        full_medoids[m] = global_idx;
        medoid_rows[m] = global_idx;
      }

      // 6. Load medoid series from Parquet (k series — tiny)
      Data medoid_data = opts.use_float32
        ? reader.read_rows<float>(std::move(medoid_rows), opts.ram_limit_bytes)
        : reader.read_rows(std::move(medoid_rows), opts.ram_limit_bytes);

      // 7. Chunked assignment: stream row groups, compute DTW to medoids
      std::vector<index_t> labels;
      double total_cost;
      if (opts.use_float32) {
        total_cost = assign_all_points_chunked<true>(
          prob_template, prob_template.dtw_function_f32(), medoid_data, full_medoids, labels,
          reader, opts.ram_limit_bytes);
      } else {
        total_cost = assign_all_points_chunked<false>(
          prob_template, prob_template.dtw_function(), medoid_data, full_medoids, labels,
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

    prob_template.set_result(best_result);

    return best_result;
  }

} // anonymous namespace

core::ClusteringResult fast_clara_parquet(Problem &prob, const CLARAOptions &opts)
{
  detail::validate_clara_request(prob, opts);
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
      if (prob.verbose())
        std::cout << "FastCLARA: streaming from Parquet ("
                  << reader.logical_series_count()
                  << " rows, " << reader.num_row_groups() << " row groups, ~"
                  << resident_bytes / (1ULL << 20) << " MB resident estimate)\n";
      return fast_clara_chunked(prob, opts, reader, stream_plan);
    }
  }
  return fast_clara(prob, opts); // the series fit the limit, or no stream was asked for
}

} // namespace dtwc::algorithms

#endif // DTWC_HAS_PARQUET
