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
#include "detail/fast_clara_assign.hpp"
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

  void validate_clara_request(const Problem &prob, const CLARAOptions &opts)
  {
    // Caller-controlled fields, before a Parquet reader is opened or any other
    // I/O. Dataset-size validation follows against the selected source.
    validate_clara_controls(opts, "fast_clara");
    if (opts.force_parquet_streaming
        && (opts.ram_limit_bytes == 0 || opts.parquet_path.empty()))
      throw InvalidInput(
        "fast_clara: force_parquet_streaming requires ram_limit_bytes and "
        "parquet_path.");
    if (opts.force_parquet_streaming && prob.size() != 0)
      throw InvalidInput(
        "fast_clara: force_parquet_streaming requires a settings-only Problem "
        "without resident series.");
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

#ifdef DTWC_HAS_CUDA
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
  void say_cpu_assignment(const Problem &prob, std::size_t n_points, std::size_t k)
  {
    if (prob.verbose() && prob.device().first == Device::GPU)
      std::cout << "FastCLARA: assigning " << n_points << " series to " << k
                << " medoids on the CPU (Metal has no kernel for the assignment)\n";
  }
#endif

} // namespace detail

namespace {

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
      detail::assign_points(0, best, medoid_indices, labels, [&](index_t p) {
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
        detail::assign_on_gpu(prob, prob.data().p_vec, 0, medoids, medoid_indices, labels, best);
        return core::detail::ordered_medoid_objective(best, "fast_clara");
      }
#else
      detail::say_cpu_assignment(prob, n_points, medoid_indices.size());
#endif
      detail::assign_points(0, best, medoid_indices, labels, [&](index_t p) {
        return [&, point = prob.series(static_cast<std::size_t>(p))](index_t m) {
          return distance(point, prob.series(static_cast<std::size_t>(medoid_indices[m])));
        };
      });
    }
    return core::detail::ordered_medoid_objective(best, "fast_clara");
  }

} // anonymous namespace


core::ClusteringResult fast_clara(Problem &prob, const CLARAOptions &opts)
{
  detail::validate_clara_request(prob, opts);
  if (opts.force_parquet_streaming) // the core reads no Parquet: fast_clara_parquet (dtwc_io) streams it
    throw IOError(
      "fast_clara: force_parquet_streaming requires a build with Parquet support.");

  const auto plan = detail::resolve_clara_plan(prob.size(), opts, "fast_clara");
  const index_t N = plan.n_points;
  const index_t sample_size = plan.sample_size;

  // If sample_size >= N, just run FastPAM on the full dataset.
  if (sample_size >= N) {
    return fast_pam_seeded(
      prob, opts.n_clusters, detail::clara_pam_seed(opts, 0), opts.max_iter);
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
      sub_prob, opts.n_clusters, detail::clara_pam_seed(opts, s), opts.max_iter);

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
