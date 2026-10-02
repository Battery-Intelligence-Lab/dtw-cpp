/**
 * @file fast_clara.cpp
 * @brief Implementation of FastCLARA: scalable k-medoids via subsampling + FastPAM.
 *
 * @details For each of n_samples subsamples:
 *   1. Draw sample_size random indices from [0, N).
 *   2. Create a sub-Problem containing only the sampled series.
 *   3. Run FastPAM on the sub-Problem to find medoids.
 *   4. Map sub-Problem medoid indices back to original dataset indices.
 *   5. Assign ALL N points to the nearest medoid (computing only N*k distances,
 *      on the GPU when the device is CUDA's).
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
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"

#ifdef DTWC_HAS_CUDA
#include "../cuda/cuda_dtw.cuh"
#endif
#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
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
    index_t n_points, const CLARAOptions &options,
    std::string_view caller)
  {
    validate_clara_controls(options, caller);
    const std::string prefix(caller);
    if (n_points <= 0)
      throw InvalidInput(prefix + ": Problem has no data points.");
    if (options.n_clusters <= 0 || options.n_clusters > n_points) {
      throw InvalidInput(
        prefix + ": n_clusters must be in [1, N]. Got n_clusters="
        + std::to_string(options.n_clusters) + ", N="
        + std::to_string(n_points) + ".");
    }

    index_t requested = options.sample_size;
    if (requested == -1) {
      const index_t k = options.n_clusters;
      const index_t classical = 40 + 2 * k;
      const index_t improved = std::min<index_t>(n_points, 10 * k + 100);
      requested = std::max(classical, improved);
    }
    requested = std::max(requested, options.n_clusters);
    requested = std::min(requested, n_points);
    return { n_points, requested };
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

#ifdef DTWC_HAS_PARQUET
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
#endif

  /// Invocation-local PAM seed for one CLARA subsample.
  std::uint64_t clara_pam_seed(const CLARAOptions &opts, int sample_index)
  {
    return opts.random_seed + static_cast<std::uint64_t>(sample_index);
  }

  /**
   * Points first .. first + best.size() - 1 to their nearest medoid: labels[p]
   * and best[p - first]. row_of(p)(m) is point p's distance to medoid slot m,
   * whichever device computed it; a point is at 0 from itself. Each point
   * writes its own slots.
   */
  template <typename RowOf>
  void assign_points(
    index_t first, std::span<double> best, const std::vector<index_t> &medoid_indices,
    std::vector<index_t> &labels, const RowOf &row_of)
  {
    const auto k = static_cast<index_t>(medoid_indices.size());
    auto assign_point = [&](std::size_t index) {
      const index_t p = first + static_cast<index_t>(index);
      const auto distance = row_of(p);
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
      best[index] = best_dist;
    };
    run_openmp(assign_point, best.size(), best.size() > 64);
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
    std::vector<index_t> &labels, std::span<double> best)
  {
    const auto distance = prob.distance();
    cuda::CUDADistMatOptions opts;
    opts.device_id = prob.device().second;
    opts.precision = prob.gpu_precision();
    opts.band = distance.band;
    opts.use_squared_l2 = distance.metric == core::MetricType::SquaredL2;
    opts.verbose = prob.verbose();
    const std::size_t k = medoids.size();
    (void)cuda::compute_medoid_distances_cuda(
      series, medoids, opts,
      [&](std::size_t block_first, std::size_t count, std::span<const double> distances) {
        const index_t block_point = first + static_cast<index_t>(block_first);
        assign_points(
          block_point, best.subspan(block_first, count), medoid_indices, labels, [&](index_t p) {
            const double *row = distances.data() + static_cast<std::size_t>(p - block_point) * k;
            return [row](index_t m) { return row[m]; };
          });
      });
  }
#else
  /// Metal has no kernel for the assignment, so on a GPU device it runs on the
  /// CPU (the sample matrices still fill on the GPU); verbose says so.
  void say_cpu_assignment(const Problem &prob, std::size_t n_points, std::size_t k)
  {
    if (prob.verbose() && prob.device().first == Device::GPU)
      std::cout << "FastCLARA: assigning " << n_points << " series to " << k
                << " medoids on the CPU (Metal has no kernel for the assignment)\n";
  }
#endif

  /**
   * Assign through the bound DTW function without allocating the parent
   * cache: on the GPU when the Problem's device is CUDA's, else on the CPU.
   */
  double assign_all_points(
    Problem &prob, const std::vector<index_t> &medoid_indices,
    std::vector<index_t> &labels)
  {
    const auto n_points = static_cast<std::size_t>(prob.size());
    labels.resize(n_points);
    std::vector<double> best(n_points);
    if (prob.data().is_f32()) {
      const auto &distance = prob.dtw_function_f32();
      assign_points(0, best, medoid_indices, labels, [&](index_t p) {
        return [&, point = prob.data().series_f32(static_cast<std::size_t>(p))](index_t m) {
          return distance(
            point, prob.data().series_f32(static_cast<std::size_t>(medoid_indices[m])));
        };
      });
    } else {
      // The request's check, on every device: band, values, and what a GPU takes.
      const auto &distance = prob.dtw_function();
#ifdef DTWC_HAS_CUDA
      if (prob.device().first == Device::GPU) {
        std::vector<std::vector<double>> medoids;
        for (const index_t medoid : medoid_indices)
          medoids.push_back(prob.data().p_vec[static_cast<std::size_t>(medoid)]);
        assign_on_gpu(prob, prob.data().p_vec, 0, medoids, medoid_indices, labels, best);
        return core::detail::ordered_medoid_objective(best, "fast_clara");
      }
#else
      say_cpu_assignment(prob, n_points, medoid_indices.size());
#endif
      assign_points(0, best, medoid_indices, labels, [&](index_t p) {
        return [&, point = prob.series(static_cast<std::size_t>(p))](index_t m) {
          return distance(point, prob.series(static_cast<std::size_t>(medoid_indices[m])));
        };
      });
    }
    return core::detail::ordered_medoid_objective(best, "fast_clara");
  }

#ifdef DTWC_HAS_PARQUET
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
    say_cpu_assignment(prob, static_cast<std::size_t>(N), medoid_indices.size());
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
        assign_on_gpu(prob, chunk.p_vec, global_offset, medoid_data.p_vec, medoid_indices,
                      labels, best_dists);
      else
#endif
        assign_points(global_offset, best_dists, medoid_indices, labels, [&](index_t p) {
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
          sub_prob, opts.n_clusters, clara_pam_seed(opts, s), opts.max_iter);
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
  if (opts.force_parquet_streaming) // this build cannot read the format
    throw IOError(
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
      if (prob.verbose())
        std::cout << "FastCLARA: streaming from Parquet ("
                  << reader.logical_series_count()
                  << " rows, " << reader.num_row_groups() << " row groups, ~"
                  << resident_bytes / (1ULL << 20) << " MB resident estimate)\n";
      return fast_clara_chunked(prob, opts, reader, stream_plan);
    }
  }
#endif

  const auto plan = detail::resolve_clara_plan(prob.size(), opts, "fast_clara");
  const index_t N = plan.n_points;
  const index_t sample_size = plan.sample_size;

  // If sample_size >= N, just run FastPAM on the full dataset.
  if (sample_size >= N) {
    return fast_pam_seeded(
      prob, opts.n_clusters, clara_pam_seed(opts, 0), opts.max_iter);
  }

  // A GPU fills copies of the samples whatever the parent holds, but the
  // assignment uploads the parent's own series: check those (a view, Float32,
  // a band, the values) once, before any sample runs.
  if (prob.device().first == Device::GPU) (void)prob.dtw_function();

  // One portable map and stable selection scan keep the in-RAM and streaming
  // paths bit-identical across standard-library implementations.
  std::mt19937_64 rng(opts.random_seed);

  core::ClusteringResult best_result;
  best_result.total_cost = std::numeric_limits<double>::max();
  bool best_present = false;

  for (int s = 0; s < opts.n_samples; ++s) {
    // 1. Draw a sorted sample using the same map as the chunked path.
    auto sample_indices = core::portable_sample_indices<index_t>(N, sample_size, rng);

    // 2. Create a sub-Problem with zero-copy span views into parent data; a GPU
    // uploads owned series (Data::p_vec), so there the sample is a copy.
    std::vector<std::string_view> sub_names;
    sub_names.reserve(sample_size);
    for (index_t idx : sample_indices)
      sub_names.push_back(prob.series_name(static_cast<std::size_t>(idx))); // O(1), no string copy

    Problem sub_prob("clara_subsample_" + std::to_string(s));
    // Copy all relevant settings from the original problem.
    sub_prob.set_distance(prob.distance());
    // Device, GPU index and precision: the sample fill honours them, or
    // validate_fill_request refuses them (e.g. Float32 series, a view, on a GPU).
    const auto [device, index] = prob.device();
    sub_prob.set_device(device, index);
    sub_prob.set_gpu_precision(prob.gpu_precision());
    sub_prob.set_verbose(prob.verbose());

    if (prob.data().is_f32()) {
      std::vector<std::span<const float>> sub_spans;
      sub_spans.reserve(sample_size);
      for (index_t idx : sample_indices)
        sub_spans.push_back(prob.data().series_f32(static_cast<std::size_t>(idx)));
      sub_prob.set_view_data(
        Data(std::move(sub_spans), std::move(sub_names), prob.data().ndim));
    } else if (device == Device::GPU) {
      std::vector<std::vector<data_t>> sub_series;
      sub_series.reserve(sample_size);
      for (index_t idx : sample_indices) {
        const auto series = prob.series(static_cast<std::size_t>(idx));
        sub_series.emplace_back(series.begin(), series.end());
      }
      sub_prob.set_data(Data(std::move(sub_series),
                             std::vector<std::string>(sub_names.begin(), sub_names.end()),
                             prob.data().ndim));
    } else {
      std::vector<std::span<const data_t>> sub_spans;
      sub_spans.reserve(sample_size);
      for (index_t idx : sample_indices)
        sub_spans.push_back(prob.series(static_cast<std::size_t>(idx))); // O(1), no data copy
      sub_prob.set_view_data(
        Data(std::move(sub_spans), std::move(sub_names), prob.data().ndim));
    }

    // 3. Run FastPAM on the sub-Problem.
    auto sub_result = fast_pam_seeded(
      sub_prob, opts.n_clusters, clara_pam_seed(opts, s), opts.max_iter);

    // 4. Map sub-Problem medoid indices back to full dataset indices.
    std::vector<index_t> full_medoids(opts.n_clusters);
    for (index_t m = 0; m < opts.n_clusters; ++m) {
      full_medoids[m] = sample_indices[sub_result.medoid_indices[m]];
    }

    // 5. Assign ALL N points to the nearest medoid.
    std::vector<index_t> labels;
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

  // The sample_size >= N branch above delegates to fast_pam, which publishes too.
  prob.set_result(best_result);

  return best_result;
}

} // namespace dtwc::algorithms
