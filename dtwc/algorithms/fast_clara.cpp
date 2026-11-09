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
#include "../core/thread_scratch.hpp" // clara_assign_scratch, held_bytes

#ifdef DTWC_HAS_CUDA
#include "../cuda/cuda_dtw.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
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
    if (options.gpu_share != -1.0 && !(options.gpu_share >= 0.0 && options.gpu_share <= 1.0))
      throw InvalidInput(prefix + ": gpu_share must be -1 (measured) or in [0, 1].");
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
    if (opts.gpu_share != -1.0 && prob.device().first != Device::CPU_GPU)
      throw InvalidInput("fast_clara: gpu_share splits the assignment pass of device cpu+gpu, but the device is "
                         + to_string(prob.device().first) + ", which would ignore it; leave gpu_share at -1.");
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

  std::size_t clara_pass_bytes(const ClaraPlan &plan, std::size_t n_samples, bool hybrid)
  {
    const auto n = static_cast<std::size_t>(plan.n_points);
    const std::size_t nearest = std::min(clara_block_points, n) + (hybrid ? n : 0);
    return n_samples * (n * sizeof(index_t) + nearest * sizeof(double));
  }

  Data clara_sample(const Data &source, const std::vector<index_t> &positions, bool owned)
  {
    const auto view = [&](const auto &series_of) {
      using T = typename std::remove_cvref_t<decltype(series_of(std::size_t{ 0 }))>::value_type;
      std::vector<std::span<const T>> series;
      std::vector<std::string_view> names;
      series.reserve(positions.size());
      names.reserve(positions.size());
      for (const index_t i : positions) {
        series.push_back(series_of(static_cast<std::size_t>(i)));
        names.push_back(source.name(static_cast<std::size_t>(i))); // O(1), no string copy
      }
      return Data(std::move(series), std::move(names), source.ndim, source.owner());
    };
    Data sample = source.is_f32() ? view([&](std::size_t i) { return source.series_f32(i); })
                                  : view([&](std::size_t i) { return source.series(i); });
    if (owned) return dtwc::detail::owned_copy(sample, source.precision);
    return sample;
  }

#ifdef DTWC_HAS_CUDA
  void assign_on_gpu(
    const Problem &prob, const Data &points, index_t first, const std::vector<std::span<const double>> &medoids,
    std::span<const index_t> medoid_indices, std::vector<std::vector<index_t>> &labels,
    std::vector<core::detail::OrderedMedoidObjective> &costs, std::size_t staging_bytes)
  {
    assign_on_gpu(
      prob, static_cast<std::size_t>(points.size()), [&points](std::size_t i) { return points.series(i); }, first,
      medoids, medoid_indices, labels, costs, staging_bytes, true);
  }

  void assign_on_gpu(
    const Problem &prob, std::size_t count, const std::function<std::span<const double>(std::size_t)> &point,
    index_t first, const std::vector<std::span<const double>> &medoids, std::span<const index_t> medoid_indices,
    std::vector<std::vector<index_t>> &labels, std::vector<core::detail::OrderedMedoidObjective> &costs,
    std::size_t staging_bytes, bool team)
  {
    const auto distance = prob.distance();
    cuda::CUDADistMatOptions opts;
    opts.device_id = prob.device().second;
    opts.precision = prob.gpu_precision();
    opts.band = distance.band;
    opts.use_squared_l2 = distance.metric == core::MetricType::SquaredL2;
    opts.verbose = prob.verbose();
    opts.host_staging_bytes = staging_bytes;
    const std::size_t slots = medoids.size();
    // Each padded row is one thread's (disjoint writes); a block's rows go to the team only where they are many.
    const auto parallel = [team](std::size_t rows, const std::function<void(std::size_t)> &task) {
      run_openmp(task, rows, team && rows > 64);
    };
    (void)cuda::compute_medoid_distances_cuda(
      count, point, medoids, opts, parallel,
      [&](std::size_t block_first, std::size_t block_count, std::span<const double> distances) {
        const index_t block_point = first + static_cast<index_t>(block_first);
        assign_in_blocks(block_point, block_count, costs, [&](index_t sub_first, std::size_t n, std::span<double> best) {
          assign_points(
            sub_first, n, best, medoid_indices, labels,
            [&](index_t p) {
              const double *row = distances.data() + static_cast<std::size_t>(p - block_point) * slots;
              return [row](index_t j) { return row[j]; };
            },
            team);
        });
      });
  }

  /// cpu+gpu's share of the points for the GPU where no calibration block is measured (one block would be the whole
  /// input): the design's predicted rates at 4.19 M series of 128 steps, band 2, 25 medoid slots, FP64, on an RTX 4000
  /// Ada beside 24 threads (2026-10-11, inferred): the CPU's lanes 55 Gcell/s, the GPU 26 Gcell/s where one thread pads
  /// its blocks from a mapped file.
  constexpr double recorded_gpu_share = 26.0 / (55.0 + 26.0);

  /**
   * FastCLARA's pass on cpu+gpu, as assign_on_gpu and the CPU pass give it (each sample's labels, and its nearest
   * distances into costs[s] in point order): the GPU takes points 0 .. n_gpu - 1, the CPU's team the rest. n_gpu is
   * opts.gpu_share of the points, rounded; with -1 it is measured, on the last clara_block_points points (fewer points
   * take recorded_gpu_share): the block assigned on the GPU, its blocks padded and assigned by this thread alone as in
   * the pass, then on the CPU's team, whose labels and distances stand; the CPU's rate is scaled to the team less the
   * GPU's driving thread. Where the block lies in the CPU's share the pass keeps it, else assigns it again. Each point's
   * distance is its device's whatever the split (the CPU's lanes are its per-pair function's bits), so a forced share
   * repeats a run exactly.
   */
  template <typename Pair>
  core::HybridSplit assign_hybrid(
    const Problem &prob, const Data &data, const std::vector<std::span<const double>> &medoid_series,
    std::span<const index_t> medoids, std::vector<std::vector<index_t>> &labels,
    std::vector<core::detail::OrderedMedoidObjective> &costs, const CLARAOptions &opts, const Pair &pair)
  {
    const auto n = static_cast<std::size_t>(data.size());
    const std::size_t samples = costs.size();
    const auto series = [&data](std::size_t i) { return data.series(i); };
    const auto on_cpu = [&](index_t first, std::size_t count, std::span<double> best, bool team) {
      assign_points_cpu<data_t>(
        first, count, best, medoids, labels, prob.distance(), pair,
        [&data](index_t p) { return data.series(static_cast<std::size_t>(p)); },
        [&medoid_series](index_t j) { return medoid_series[static_cast<std::size_t>(j)]; }, team);
    };
    const int threads = get_max_threads();

    core::HybridSplit split;
    split.gpu_share = opts.gpu_share;
    const std::size_t block = std::min(clara_block_points, n), tail = n - block; // the calibration block
    std::vector<double> calibrated; // its nearest distances on the CPU, [s * block + i]
    if (opts.gpu_share < 0 && block == n) {
      split.gpu_share = recorded_gpu_share;
      split.source = "recorded";
    } else if (opts.gpu_share < 0) {
      using clock = std::chrono::steady_clock;
      std::vector<core::detail::OrderedMedoidObjective> unused(samples);
      const auto start = clock::now();
      assign_on_gpu(
        prob, block, [&data, tail](std::size_t i) { return data.series(tail + i); }, static_cast<index_t>(tail),
        medoid_series, medoids, labels, unused, opts.gpu_staging_bytes, false);
      const auto gpu_done = clock::now();
      calibrated.resize(samples * block);
      on_cpu(static_cast<index_t>(tail), block, calibrated, true); // its labels replace the GPU's
      const std::chrono::duration<double> on_gpu = gpu_done - start, on_team = clock::now() - gpu_done;
      split.gpu_rate = static_cast<double>(block) / on_gpu.count();
      split.cpu_rate = static_cast<double>(block) / on_team.count() * (threads - 1) / threads;
      split.gpu_share = split.gpu_rate / (split.gpu_rate + split.cpu_rate);
      split.source = "calibrated";
    }
    const auto n_gpu = static_cast<std::size_t>(std::llround(split.gpu_share * static_cast<double>(n)));
    split.gpu_share = static_cast<double>(n_gpu) / static_cast<double>(n);
    const bool keep = !calibrated.empty() && n_gpu <= tail;
    if (!keep) std::vector<double>().swap(calibrated);
    const std::size_t cpu_end = keep ? tail : n;

    if (n_gpu > 0)
      assign_on_gpu(prob, n_gpu, series, 0, medoid_series, medoids, labels, costs, opts.gpu_staging_bytes, true);
    if (cpu_end > n_gpu)
      assign_in_blocks(static_cast<index_t>(n_gpu), cpu_end - n_gpu, costs,
                       [&](index_t first, std::size_t count, std::span<double> best) { on_cpu(first, count, best, true); });
    if (keep)
      for (std::size_t s = 0; s < samples; ++s)
        costs[s].add(std::span<const double>(calibrated).subspan(s * block, block));
    return split;
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
      if (prob.device().first == Device::CPU_GPU) sub_prob.set_device(Device::GPU, prob.device().second); // one fill
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

std::string describe(const core::HybridSplit &split, std::size_t n_points)
{
  std::ostringstream text;
  text << "cpu+gpu: the GPU assigned the first " << std::llround(split.gpu_share * static_cast<double>(n_points))
       << " of " << n_points << " series (share " << std::setprecision(17) << split.gpu_share << ", " << split.source;
  if (split.gpu_rate > 0)
    text << std::setprecision(4) << ": the CPU's team " << split.cpu_rate << " and the GPU " << split.gpu_rate
         << " series a second";
  text << "), the CPU's team the rest";
  return text.str();
}

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

  // The parent's series are checked as the pass reads them, on the CPU (Problem::SeriesCheck): the empty series and
  // the band first, the samples' rows before the PAMs, each block of points just before it is assigned, so a mapped
  // file is read once; the verdict, type and text are dtw_function()'s, only later. A GPU checks them all first, as
  // dtw_function() does: CUDA's assignment packs and uploads the parent's series inside its call, where no block can
  // be checked (and a GPU fills copies of the samples whatever the parent holds).
  const Device device = prob.device().first;
  const bool gpu = uses_gpu(device); // cpu+gpu checks as the GPU does: its GPU share reads the series inside its call
  const bool f32 = prob.data().is_f32() && !gpu; // a GPU refuses Float32 series in that check
  std::optional<Problem::SeriesCheck> check;
  if (!gpu) check.emplace(prob.series_check(f32 ? "Problem::dtw_function_f32" : "Problem::dtw_function", f32));
  const auto *const pair = f32 ? nullptr : check ? &prob.dtw_function(*check) : &prob.dtw_function();
  const auto *const pair_f32 = f32 ? &prob.dtw_function_f32(*check) : nullptr;

  // All samples first, from one engine; each PAM on a view of its series (a copy on a GPU, which uploads owned ones).
  const auto samples = detail::draw_clara_samples(N, sample_size, opts);
  const auto n_points = static_cast<std::size_t>(N);
  // The pass stops at the first entirely-NaN series marked (Interpolate), whose distances throw, and at an error the
  // PAMs or a block's assignment raise on values the check takes (a distance that overflows): the series from
  // `checked` on are checked and finish() runs, so a refusal of dtw_function()'s check comes before that error.
  const auto check_rest = [&](std::size_t checked) {
    if (!check) return;
    check->check(checked, n_points - checked);
    check->finish();
  };
  if (check) {
    check->check_rows(detail::sample_union(samples));
    if (check->marked()) check_rest(0); // before the PAMs, whose fills would refuse it first
  }
  std::vector<index_t> medoids; // sample s's medoid m is slot s * k + m
  std::vector<core::ClusteringResult> pams;
  try {
    pams = detail::clara_pams(prob, opts, prob.data(), samples, samples, gpu, medoids);
  } catch (...) {
    check_rest(0);
    throw;
  }

  // One pass: every point against the medoids of every sample.
  std::vector<std::vector<index_t>> labels(samples.size(), std::vector<index_t>(n_points));
  std::vector<core::detail::OrderedMedoidObjective> costs(samples.size());
  std::optional<core::HybridSplit> split;
  const auto &data = prob.data();
  const auto in_blocks = [&](const auto &assign) {
    detail::assign_in_blocks(0, n_points, costs, [&](index_t first, std::size_t n, std::span<double> best) {
      const std::size_t next = static_cast<std::size_t>(first) + n;
      if (check) {
        check->check(static_cast<std::size_t>(first), n);
        if (check->marked()) check_rest(next);
      }
      try {
        assign(first, n, best);
      } catch (...) {
        check_rest(next);
        throw;
      }
    });
  };
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
      // The series and the medoids are read where they lie, a view's or owned: no copy of them is made.
      std::vector<std::span<const double>> medoid_series;
      medoid_series.reserve(medoids.size());
      for (const index_t medoid : medoids) medoid_series.push_back(data.series(static_cast<std::size_t>(medoid)));
      if (device == Device::CPU_GPU) {
        split = detail::assign_hybrid(prob, data, medoid_series, medoids, labels, costs, opts, *pair);
        if (prob.verbose()) std::cout << "FastCLARA " << describe(*split, n_points) << '\n';
      } else
        detail::assign_on_gpu(prob, data, 0, medoid_series, medoids, labels, costs, opts.gpu_staging_bytes);
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

  if (check) check->finish();
  auto result = detail::clara_winner(pams, medoids, labels, costs);
  result.hybrid = split;
  // The sample_size >= N branch above delegates to fast_pam, which publishes too.
  prob.set_result(result);
  return result;
}

} // namespace dtwc::algorithms

std::size_t dtwc::core::detail::clara_assign_scratch(bool release) noexcept
{
  return held_bytes(release, dtwc::algorithms::detail::assign_table());
}
