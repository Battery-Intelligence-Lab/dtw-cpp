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
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"

#ifdef DTWC_HAS_CUDA
#include "../cuda/cuda_dtw.cuh"
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

  std::size_t clara_pass_bytes(const ClaraPlan &plan, std::size_t n_samples)
  {
    const auto n = static_cast<std::size_t>(plan.n_points);
    return n_samples * (n * sizeof(index_t) + std::min(clara_block_points, n) * sizeof(double));
  }

  Data clara_sample(const Data &source, const std::vector<index_t> &positions, bool owned)
  {
    const auto take = [&](const auto &series_of) {
      using T = typename std::remove_cvref_t<decltype(series_of(std::size_t{ 0 }))>::value_type;
      if (owned) {
        std::vector<std::vector<T>> series;
        std::vector<std::string> names;
        series.reserve(positions.size());
        names.reserve(positions.size());
        for (const index_t i : positions) {
          const auto values = series_of(static_cast<std::size_t>(i));
          series.emplace_back(values.begin(), values.end());
          names.emplace_back(source.name(static_cast<std::size_t>(i)));
        }
        return Data(std::move(series), std::move(names), source.ndim);
      }
      std::vector<std::span<const T>> series;
      std::vector<std::string_view> names;
      series.reserve(positions.size());
      names.reserve(positions.size());
      for (const index_t i : positions) {
        series.push_back(series_of(static_cast<std::size_t>(i)));
        names.push_back(source.name(static_cast<std::size_t>(i))); // O(1), no string copy
      }
      return Data(std::move(series), std::move(names), source.ndim);
    };
    if (source.is_f32()) return take([&](std::size_t i) { return source.series_f32(i); });
    return take([&](std::size_t i) { return source.series(i); });
  }

#ifdef DTWC_HAS_CUDA
  void assign_on_gpu(
    const Problem &prob, const std::vector<std::vector<double>> &series, index_t first,
    const std::vector<std::vector<double>> &medoids, std::span<const index_t> medoid_indices,
    std::vector<std::vector<index_t>> &labels, std::vector<core::detail::OrderedMedoidObjective> &costs)
  {
    const auto distance = prob.distance();
    cuda::CUDADistMatOptions opts;
    opts.device_id = prob.device().second;
    opts.precision = prob.gpu_precision();
    opts.band = distance.band;
    opts.use_squared_l2 = distance.metric == core::MetricType::SquaredL2;
    opts.verbose = prob.verbose();
    const std::size_t slots = medoids.size();
    (void)cuda::compute_medoid_distances_cuda(
      series, medoids, opts,
      [&](std::size_t block_first, std::size_t count, std::span<const double> distances) {
        const index_t block_point = first + static_cast<index_t>(block_first);
        assign_in_blocks(block_point, count, costs, [&](index_t sub_first, std::size_t n, std::span<double> best) {
          assign_points(sub_first, n, best, medoid_indices, labels, [&](index_t p) {
            const double *row = distances.data() + static_cast<std::size_t>(p - block_point) * slots;
            return [row](index_t j) { return row[j]; };
          });
        });
      });
  }
#else
  void say_cpu_assignment(const Problem &prob, std::size_t n_points, std::size_t n_samples, std::size_t k)
  {
    if (prob.verbose() && prob.device().first == Device::GPU)
      std::cout << "FastCLARA: assigning " << n_points << " series to " << n_samples << " samples' "
                << n_samples * k << " medoids on the CPU (Metal has no kernel for the assignment)\n";
  }
#endif

  std::vector<core::ClusteringResult> clara_pams(
    const Problem &prob, const CLARAOptions &opts, const Data &source,
    const std::vector<std::vector<index_t>> &samples, const std::vector<std::vector<index_t>> &positions,
    bool owned, std::vector<index_t> &medoids)
  {
    const auto k = static_cast<std::size_t>(opts.n_clusters);
    std::vector<core::ClusteringResult> pams(samples.size());
    medoids.resize(samples.size() * k);
    for (std::size_t s = 0; s < samples.size(); ++s) {
      Problem sub_prob("clara_subsample_" + std::to_string(s));
      sub_prob.copy_distance_settings_from(prob);
      if (owned)
        sub_prob.set_data(clara_sample(source, positions[s], true));
      else
        sub_prob.set_view_data(clara_sample(source, positions[s], false));
      pams[s] = fast_pam(sub_prob, opts.n_clusters, opts.max_iter, clara_pam_seed(opts, static_cast<int>(s)));
      for (std::size_t m = 0; m < k; ++m)
        medoids[s * k + m] = samples[s][static_cast<std::size_t>(pams[s].medoid_indices[m])];
    }
    return pams;
  }

} // namespace detail


core::ClusteringResult fast_clara(Problem &prob, const CLARAOptions &opts)
{
  detail::validate_clara_request(prob, opts);
  // The core reads no Parquet: fast_clara_parquet (dtwc_io) streams it.
  if (opts.force_parquet_streaming)
    throw IOError(
      "fast_clara: force_parquet_streaming requires a build with Parquet support.");
  if (opts.ram_limit_bytes > 0 && !opts.parquet_path.empty())
    throw InvalidInput(
      "fast_clara: parquet_path with ram_limit_bytes asks to stream the series from "
      "Parquet, which algorithms::fast_clara_parquet does (dtwc_cl --ram-limit, in a "
      "build with Parquet); fast_clara clusters the Problem's own series. Clear "
      "parquet_path, or call fast_clara_parquet.");

  const auto plan = detail::resolve_clara_plan(prob.size(), opts, "fast_clara");
  const index_t N = plan.n_points;
  const index_t sample_size = plan.sample_size;

  // If sample_size >= N, just run FastPAM on the full dataset.
  if (sample_size >= N) {
    return fast_pam(
      prob, opts.n_clusters, opts.max_iter, detail::clara_pam_seed(opts, 0));
  }

  // The parent's series are checked once, before any sample: the band, the values and, on a GPU, what it takes (a
  // GPU fills copies of the samples whatever the parent holds, but the assignment uploads the parent's own series).
  // Every sample views or copies them, and the pass reads them all.
  const bool gpu = prob.device().first == Device::GPU;
  const bool f32 = prob.data().is_f32() && !gpu; // a GPU refuses Float32 series in that check
  const auto *const pair = f32 ? nullptr : &prob.dtw_function();
  const auto *const pair_f32 = f32 ? &prob.dtw_function_f32() : nullptr;

  // All samples first, from one engine; each PAM on a view of its series (a copy on a GPU, which uploads owned ones).
  const auto samples = detail::draw_clara_samples(N, sample_size, opts);
  std::vector<index_t> medoids; // sample s's medoid m is slot s * k + m
  const auto pams = detail::clara_pams(prob, opts, prob.data(), samples, samples, gpu, medoids);

  // One pass: every point against the medoids of every sample.
  const auto n_points = static_cast<std::size_t>(N);
  std::vector<std::vector<index_t>> labels(samples.size(), std::vector<index_t>(n_points));
  std::vector<core::detail::OrderedMedoidObjective> costs(samples.size());
  const auto &data = prob.data();
  const auto in_blocks = [&](const auto &assign) { detail::assign_in_blocks(0, n_points, costs, assign); };
  if (pair_f32) {
    in_blocks([&](index_t first, std::size_t n, std::span<double> best) {
      detail::assign_points_cpu<float>(
        first, n, best, medoids, labels, prob.distance(), *pair_f32,
        [&](index_t p) { return data.series_f32(static_cast<std::size_t>(p)); },
        [&](index_t j) { return data.series_f32(static_cast<std::size_t>(medoids[static_cast<std::size_t>(j)])); });
    });
  } else {
#ifdef DTWC_HAS_CUDA
    if (gpu) {
      std::vector<std::vector<double>> medoid_series;
      medoid_series.reserve(medoids.size());
      for (const index_t medoid : medoids) medoid_series.push_back(data.p_vec[static_cast<std::size_t>(medoid)]);
      detail::assign_on_gpu(prob, data.p_vec, 0, medoid_series, medoids, labels, costs);
    } else
#else
    detail::say_cpu_assignment(prob, n_points, samples.size(), static_cast<std::size_t>(opts.n_clusters));
#endif
      in_blocks([&](index_t first, std::size_t n, std::span<double> best) {
        detail::assign_points_cpu<data_t>(
          first, n, best, medoids, labels, prob.distance(), *pair,
          [&](index_t p) { return data.series(static_cast<std::size_t>(p)); },
          [&](index_t j) { return data.series(static_cast<std::size_t>(medoids[static_cast<std::size_t>(j)])); });
      });
  }

  auto result = detail::clara_winner(pams, medoids, labels, costs);
  // The sample_size >= N branch above delegates to fast_pam, which publishes too.
  prob.set_result(result);
  return result;
}

} // namespace dtwc::algorithms
