/**
 * @file cdist.cpp
 * @brief Cross-collection DTW distance matrix implementation across CPU SIMD lanes and GPU.
 * @author Volkan Kumtepeli
 */

#include "cdist.hpp"

#include "base/error.hpp"
#include "base/missing_utils.hpp"
#include "base/parallelisation.hpp"
#include "config.hpp"
#include "core/dtw_dispatch.hpp"
#include "core/dtw_kernel.hpp"
#include "warping.hpp"

#ifdef DTWC_HAS_CUDA
#include "cuda/cuda_dtw.cuh"
#endif
#ifdef DTWC_HAS_METAL
#include "metal/metal_dtw.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dtwc {
namespace {

const char *variant_label(core::DTWVariant v)
{
  switch (v) {
  case core::DTWVariant::Standard: return "Standard";
  case core::DTWVariant::DDTW: return "DDTW";
  case core::DTWVariant::WDTW: return "WDTW";
  case core::DTWVariant::ADTW: return "ADTW";
  case core::DTWVariant::SoftDTW: return "SoftDTW";
  case core::DTWVariant::MSM: return "MSM";
  case core::DTWVariant::TWE: return "TWE";
  }
  return "?";
}

const char *missing_label(core::MissingStrategy m)
{
  switch (m) {
  case core::MissingStrategy::Error: return "Error";
  case core::MissingStrategy::ZeroCost: return "ZeroCost";
  case core::MissingStrategy::AROW: return "AROW";
  case core::MissingStrategy::Interpolate: return "Interpolate";
  }
  return "?";
}

#ifdef DTWC_HAS_CUDA
constexpr std::string_view kGpuBackend = "CUDA";
#else
constexpr std::string_view kGpuBackend = "Metal";
#endif

[[noreturn]] void reject_cdist_gpu(const std::string &reason, const std::string &fix)
{
  throw GpuRefusal("cdist: " + std::string(kGpuBackend) + " " + reason
                     + "; no backend call or CPU fallback was attempted. " + fix,
                   std::string(kGpuBackend) + " " + reason);
}

void validate_cdist_device(const core::DistanceConfig &cfg, bool f32,
                           Device device, int device_index, GpuPrecision gpu_precision)
{
  if (!uses_gpu(device)) return;

#if !defined(DTWC_HAS_CUDA) && !defined(DTWC_HAS_METAL)
  (void)cfg;
  (void)f32;
  (void)device_index;
  (void)gpu_precision;
  if (device == Device::CPU_GPU)
    throw detail::hybrid_refusal("cdist", detail::hybrid_not_built_reason(), false);
  throw detail::gpu_not_built_refusal();
#else
  if (device == Device::CPU_GPU) {
    const std::string reason = detail::hybrid_not_built_reason();
    if (!reason.empty())
      throw detail::hybrid_refusal("cdist", reason);
  }
  if (f32)
    reject_cdist_gpu("computes from Float64 series, but precision = Float32 was requested",
                     "Load the series as Float64, or use device cpu.");
  if (cfg.ndim > 1)
    reject_cdist_gpu("is univariate only, but ndim = " + std::to_string(cfg.ndim) + " was requested",
                     "Use device cpu for multivariate series.");
  if (cfg.variant.variant != core::DTWVariant::Standard)
    reject_cdist_gpu(std::string("implements Standard DTW only, but variant = ")
                       + variant_label(cfg.variant.variant) + " was requested",
                     "Use device cpu for this variant.");
  if (cfg.missing != core::MissingStrategy::Error)
    reject_cdist_gpu(std::string("has no missing-data strategy, but missing_strategy = ")
                       + missing_label(cfg.missing) + " was requested",
                     "Use device cpu, or missing_strategy Error for data without NaN.");
#if !defined(DTWC_HAS_CUDA) && defined(DTWC_HAS_METAL)
  (void)device_index; // Metal runs on the system default GPU, and has no cross-collection kernel for any index
  metal::validate_metal_precision(gpu_precision);
  reject_cdist_gpu("cross-collection rectangular cdist is not implemented on Metal",
                   "Use device cpu.");
#else
  (void)gpu_precision;
  require_gpu_device("cdist", device_index); // the rules above need no device: it is asked last, before any series is read
#endif
#endif
}

enum SeriesVerdict : unsigned char { kUsable, kEntirelyNaN, kRefused };

void check_collection_series(const Data &data, std::string_view side, core::MissingStrategy missing)
{
  const auto n = static_cast<std::size_t>(data.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (data.series_flat_size(i) == 0) {
      throw InvalidInput("cdist: " + std::string(side) + " series '" + std::string(data.name(i))
                         + "' (index " + std::to_string(i) + ") is empty.");
    }
  }
  if (n == 0) return;

  const bool nan_is_missing = (missing != core::MissingStrategy::Error);
  std::vector<SeriesVerdict> verdicts(n, kUsable);
  auto check_one = [&](std::size_t i) {
    auto eval = [&](const auto values) {
      if (!detail::all_usable(values, nan_is_missing)) return kRefused;
      return (missing == core::MissingStrategy::Interpolate && all_missing(values)) ? kEntirelyNaN : kUsable;
    };
    verdicts[i] = data.is_f32() ? eval(data.series_f32(i)) : eval(data.series(i));
  };
  run_openmp(check_one, n, n > 64, 16);

  for (std::size_t i = 0; i < n; ++i) {
    if (verdicts[i] == kRefused) {
      const std::string label = std::string(side) + " series '" + std::string(data.name(i))
                              + "' (index " + std::to_string(i) + ")";
      if (data.is_f32()) detail::refuse_unusable(data.series_f32(i), label, "cdist", nan_is_missing);
      detail::refuse_unusable(data.series(i), label, "cdist", nan_is_missing);
    }
    if (verdicts[i] == kEntirelyNaN) {
      throw InvalidInput("cdist: " + std::string(side) + " series '" + std::string(data.name(i))
                         + "' (index " + std::to_string(i)
                         + ") is all NaN, so Interpolate has no value to interpolate from.");
    }
  }
}

void check_cross_band(const Data &X, const Data &Y, const core::DistanceConfig &cfg)
{
  const auto n_x = static_cast<std::size_t>(X.size());
  const auto n_y = static_cast<std::size_t>(Y.size());
  if (n_x == 0 || n_y == 0 || cfg.band < 0) return;
  const auto variant = cfg.variant.variant;
  if (variant == core::DTWVariant::SoftDTW || variant == core::DTWVariant::MSM
      || variant == core::DTWVariant::TWE) {
    return;
  }

  std::size_t min_x_idx = 0, max_x_idx = 0;
  std::size_t min_x_len = X.series_length(0), max_x_len = min_x_len;
  for (std::size_t i = 1; i < n_x; ++i) {
    const std::size_t len = X.series_length(i);
    if (len < min_x_len) { min_x_len = len; min_x_idx = i; }
    if (len > max_x_len) { max_x_len = len; max_x_idx = i; }
  }

  std::size_t min_y_idx = 0, max_y_idx = 0;
  std::size_t min_y_len = Y.series_length(0), max_y_len = min_y_len;
  for (std::size_t j = 1; j < n_y; ++j) {
    const std::size_t len = Y.series_length(j);
    if (len < min_y_len) { min_y_len = len; min_y_idx = j; }
    if (len > max_y_len) { max_y_len = len; max_y_idx = j; }
  }

  const std::size_t gap_xy = max_x_len > min_y_len ? max_x_len - min_y_len : 0;
  const std::size_t gap_yx = max_y_len > min_x_len ? max_y_len - min_x_len : 0;
  const std::size_t gap = std::max(gap_xy, gap_yx);
  if (gap > static_cast<std::size_t>(cfg.band)) {
    const std::size_t xi = (gap_xy >= gap_yx) ? max_x_idx : min_x_idx;
    const std::size_t yj = (gap_xy >= gap_yx) ? min_y_idx : max_y_idx;
    throw InvalidInput(
      "cdist: band = " + std::to_string(cfg.band)
      + " is narrower than the length difference between X series '" + std::string(X.name(xi))
      + "' (index " + std::to_string(xi) + ", length " + std::to_string(X.series_length(xi))
      + ") and Y series '" + std::string(Y.name(yj))
      + "' (index " + std::to_string(yj) + ", length " + std::to_string(Y.series_length(yj))
      + "), so no warping path fits that pair. The smallest feasible band is "
      + std::to_string(gap) + "; pass band >= " + std::to_string(gap)
      + ", or band = -1 for full DTW.");
  }
}

/// Upper bound on temporary pre-packed Y lane panels (~64 MiB); larger or
/// memory-mapped collections stream W-lane panels on the fly in thread scratch.
constexpr std::size_t kMaxPrepackBytes = std::size_t{ 64 } << 20;

template <typename T>
bool can_prepack_panels(const Data &Y, const auto &y_at) noexcept
{
  if (Y.is_mapped()) return false;
  const auto n_y = static_cast<std::size_t>(Y.size());
  constexpr std::size_t W = core::dtw_lanes<T>;
  const std::size_t n_panels = (n_y + W - 1) / W;
  std::size_t total_bytes = 0;
  for (std::size_t p = 0; p < n_panels; ++p) {
    const std::size_t b0 = p * W;
    const std::size_t cnt = std::min(W, n_y - b0);
    std::size_t max_len = 0;
    for (std::size_t w = 0; w < cnt; ++w)
      max_len = std::max(max_len, y_at(b0 + w).size());
    const std::size_t row_bytes = W * sizeof(T);
    if (max_len > (kMaxPrepackBytes - total_bytes) / row_bytes) return false;
    total_bytes += max_len * row_bytes;
  }
  return true;
}

template <typename T>
void compute_cdist_cpu_rows(const Data &X, const Data &Y,
                            std::size_t x_begin, std::size_t x_end,
                            const core::DistanceConfig &cfg,
                            const auto &x_at, const auto &y_at,
                            double *out)
{
  const std::size_t n_rows = x_end - x_begin;
  const auto n_y = static_cast<std::size_t>(Y.size());
  if (n_rows == 0 || n_y == 0) return;

  const auto packed_fn = core::resolve_dtw_packed_block_fn<T>(cfg);
  if (packed_fn && can_prepack_panels<T>(Y, y_at)) {
    constexpr std::size_t W = core::dtw_lanes<T>;
    const std::size_t n_panels = (n_y + W - 1) / W;
    std::vector<core::PackedLanesPanel<T>> panels(n_panels);
    auto pack_one = [&](std::size_t p) {
      const std::size_t b0 = p * W;
      const std::size_t cnt = std::min(W, n_y - b0);
      std::array<std::span<const T>, W> ys;
      for (std::size_t w = 0; w < W; ++w) ys[w] = y_at(b0 + std::min(w, cnt - 1));
      core::pack_dtw_panel<T>(cfg, ys.data(), panels[p]);
    };
    run_openmp(pack_one, n_panels, n_panels > 8);

    const auto threads = static_cast<std::size_t>(std::max(1, get_max_threads()));
    if (n_rows >= 4 * threads || n_panels <= 1) {
      auto row_task = [&](std::size_t r) {
        const std::size_t i = x_begin + r;
        const auto x_prep = core::prepare_dtw_query<T>(cfg, x_at(i));
        double *row_out = out + i * n_y;
        std::array<double, W> d;
        for (std::size_t p = 0; p < n_panels; ++p) {
          const std::size_t b0 = p * W;
          const std::size_t cnt = std::min(W, n_y - b0);
          packed_fn(x_prep, panels[p].view(), d);
          for (std::size_t w = 0; w < cnt; ++w) row_out[b0 + w] = d[w];
        }
      };
      run_openmp(row_task, n_rows, (n_rows * n_panels) > 4);
    } else {
      const std::size_t total_tiles = n_rows * n_panels;
      auto tile_task = [&](std::size_t idx) {
        const std::size_t i = x_begin + (idx / n_panels);
        const std::size_t p = idx % n_panels;
        const std::size_t b0 = p * W;
        const std::size_t cnt = std::min(W, n_y - b0);
        const auto x_prep = core::prepare_dtw_query<T>(cfg, x_at(i));
        std::array<double, W> d;
        packed_fn(x_prep, panels[p].view(), d);
        double *row_out = out + i * n_y + b0;
        for (std::size_t w = 0; w < cnt; ++w) row_out[w] = d[w];
      };
      run_openmp(tile_task, total_tiles, total_tiles > 4);
    }
    return;
  }

  const auto block_fn = core::resolve_dtw_block_fn<T>(cfg);
  if (block_fn) {
    constexpr std::size_t W = core::dtw_lanes<T>;
    const std::size_t n_panels = (n_y + W - 1) / W;
    const std::size_t total_tiles = n_rows * n_panels;
    auto stream_tile = [&](std::size_t idx) {
      const std::size_t i = x_begin + (idx / n_panels);
      const std::size_t p = idx % n_panels;
      const std::size_t b0 = p * W;
      const std::size_t cnt = std::min(W, n_y - b0);
      std::array<std::span<const T>, W> ys;
      for (std::size_t w = 0; w < W; ++w) ys[w] = y_at(b0 + std::min(w, cnt - 1));
      std::array<double, W> d;
      block_fn(x_at(i), ys, d);
      double *row_out = out + i * n_y + b0;
      for (std::size_t w = 0; w < cnt; ++w) row_out[w] = d[w];
    };
    run_openmp(stream_tile, total_tiles, total_tiles > 4);
    return;
  }

  const auto pair_fn = core::resolve_dtw_fn<T>(cfg, X, &Y);
  const std::size_t total_pairs = n_rows * n_y;
  auto pair_task = [&](std::size_t idx) {
    const std::size_t i = x_begin + (idx / n_y);
    const std::size_t j = idx % n_y;
    out[i * n_y + j] = pair_fn(x_at(i), y_at(j));
  };
  run_openmp(pair_task, total_pairs, total_pairs > 8);
}

#ifdef DTWC_HAS_CUDA
template <typename CpuT>
void run_hybrid_cdist_split(
  const Data &X_gpu, const std::vector<std::span<const double>> &y_gpu_spans,
  const Data &X_cpu, const Data &Y_cpu,
  std::size_t gpu_rows, std::size_t n_x, std::size_t n_y,
  const core::DistanceConfig &cfg, const cuda::CUDADistMatOptions &opts,
  int threads, const auto &x_cpu_at, const auto &y_cpu_at, double *out)
{
  constexpr std::size_t W = core::dtw_lanes<CpuT>;
  const std::size_t n_panels = (n_y + W - 1) / W;
  const auto packed_fn = core::resolve_dtw_packed_block_fn<CpuT>(cfg);
  const auto block_fn = core::resolve_dtw_block_fn<CpuT>(cfg);
  std::vector<core::PackedLanesPanel<CpuT>> panels;
  std::function<double(std::span<const CpuT>, std::span<const CpuT>)> pair_fn;
  const bool use_panels = packed_fn && can_prepack_panels<CpuT>(Y_cpu, y_cpu_at);
  if (use_panels) {
    panels.resize(n_panels);
    auto pack_one = [&](std::size_t p) {
      const std::size_t b0 = p * W;
      const std::size_t cnt = std::min(W, n_y - b0);
      std::array<std::span<const CpuT>, W> ys;
      for (std::size_t w = 0; w < W; ++w) ys[w] = y_cpu_at(b0 + std::min(w, cnt - 1));
      core::pack_dtw_panel<CpuT>(cfg, ys.data(), panels[p]);
    };
    run_openmp(pack_one, n_panels, n_panels > 8);
  } else if (!block_fn) {
    pair_fn = core::resolve_dtw_fn<CpuT>(cfg, X_cpu, &Y_cpu);
  }

  std::exception_ptr gpu_exc;
  std::exception_ptr cpu_exc;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num();
    if (tid == 0) {
      try {
        const auto serial_pad = [](std::size_t count, const std::function<void(std::size_t)> &task) {
          for (std::size_t k = 0; k < count; ++k) task(k);
        };
        (void)cuda::compute_medoid_distances_cuda(
          gpu_rows, [&](std::size_t i) { return X_gpu.series(i); }, y_gpu_spans, opts, serial_pad,
          [&](std::size_t first, [[maybe_unused]] std::size_t count, std::span<const double> block) {
            std::copy(block.begin(), block.end(), out + first * n_y);
          });
      } catch (...) {
        gpu_exc = std::current_exception();
      }
    }
#pragma omp for schedule(dynamic, 1) nowait
    for (std::ptrdiff_t r = static_cast<std::ptrdiff_t>(gpu_rows);
         r < static_cast<std::ptrdiff_t>(n_x); ++r) {
      const auto i = static_cast<std::size_t>(r);
      try {
        double *row_out = out + i * n_y;
        if (use_panels) {
          const auto x_prep = core::prepare_dtw_query<CpuT>(cfg, x_cpu_at(i));
          std::array<double, W> d;
          for (std::size_t p = 0; p < n_panels; ++p) {
            const std::size_t b0 = p * W;
            const std::size_t cnt = std::min(W, n_y - b0);
            packed_fn(x_prep, panels[p].view(), d);
            for (std::size_t w = 0; w < cnt; ++w) row_out[b0 + w] = d[w];
          }
        } else if (block_fn) {
          std::array<std::span<const CpuT>, W> ys;
          std::array<double, W> d;
          for (std::size_t p = 0; p < n_panels; ++p) {
            const std::size_t b0 = p * W;
            const std::size_t cnt = std::min(W, n_y - b0);
            for (std::size_t w = 0; w < W; ++w) ys[w] = y_cpu_at(b0 + std::min(w, cnt - 1));
            block_fn(x_cpu_at(i), ys, d);
            for (std::size_t w = 0; w < cnt; ++w) row_out[b0 + w] = d[w];
          }
        } else {
          for (std::size_t j = 0; j < n_y; ++j) row_out[j] = pair_fn(x_cpu_at(i), y_cpu_at(j));
        }
      } catch (...) {
#pragma omp critical(dtwc_cdist_cpu_exc)
        {
          if (!cpu_exc) cpu_exc = std::current_exception();
        }
      }
    }
  }
#else
  (void)threads;
  const auto serial_pad = [](std::size_t count, const std::function<void(std::size_t)> &task) {
    for (std::size_t k = 0; k < count; ++k) task(k);
  };
  (void)cuda::compute_medoid_distances_cuda(
    gpu_rows, [&](std::size_t i) { return X_gpu.series(i); }, y_gpu_spans, opts, serial_pad,
    [&](std::size_t first, [[maybe_unused]] std::size_t count, std::span<const double> block) {
      std::copy(block.begin(), block.end(), out + first * n_y);
    });
  compute_cdist_cpu_rows<CpuT>(X_cpu, Y_cpu, gpu_rows, n_x, cfg, x_cpu_at, y_cpu_at, out);
#endif
  if (gpu_exc) std::rethrow_exception(gpu_exc);
  if (cpu_exc) std::rethrow_exception(cpu_exc);
}
#endif

} // namespace

std::vector<double> cdist(const Data &X_in, const Data &Y_in,
                          const core::DistanceConfig &distance,
                          Device device,
                          int device_index,
                          GpuPrecision gpu_precision,
                          double gpu_share,
                          bool fallback)
{
  // The settings no device can honour, refused before the device is asked and whatever `fallback` says, as
  // Problem::set_device and Problem::set_distance refuse them.
  if (device_index < 0)
    throw InvalidInput("cdist: the GPU index must be >= 0; got " + std::to_string(device_index) + ".");
  if (distance.band < -1)
    throw InvalidInput("cdist: band must be -1 (full DTW) or at least 0; got " + std::to_string(distance.band) + ".");
  if (gpu_share != -1.0 && (std::isnan(gpu_share) || gpu_share < 0.0 || gpu_share > 1.0)) {
    throw InvalidInput("cdist: gpu_share must be -1 (automatic) or in [0, 1]; got "
                       + std::to_string(gpu_share) + ".");
  }
  if (gpu_share != -1.0 && device != Device::CPU_GPU) {
    throw InvalidInput(detail::share_ignored_message("cdist", "cpu+gpu cdist", device));
  }

  X_in.validate_ndim();
  Y_in.validate_ndim();
  if (X_in.ndim != Y_in.ndim) {
    throw InvalidInput("cdist: X and Y must have the same number of dimensions (X.ndim = "
                       + std::to_string(X_in.ndim) + ", Y.ndim = " + std::to_string(Y_in.ndim) + ").");
  }

  const bool both_f32 = X_in.is_f32() && Y_in.is_f32();
  core::DistanceConfig cfg = distance;
  cfg.ndim = X_in.ndim;
  core::validate(cfg, both_f32);
  // The GPU request is judged before any series is read; with `fallback` what it cannot honour computes on the CPU.
  try {
    validate_cdist_device(cfg, both_f32, device, device_index, gpu_precision);
  } catch (const GpuRefusal &refusal) {
    if (!fallback) throw;
    detail::say_fallback(refusal.reason());
    device = Device::CPU;
    device_index = 0;
    gpu_share = -1.0;
  }

  check_collection_series(X_in, "X", cfg.missing);
  check_collection_series(Y_in, "Y", cfg.missing);
  check_cross_band(X_in, Y_in, cfg);

  const auto n_x = static_cast<std::size_t>(X_in.size());
  const auto n_y = static_cast<std::size_t>(Y_in.size());
  if (n_x == 0 || n_y == 0) return {};

  std::optional<Data> X_widened;
  std::optional<Data> Y_widened;
  if (!both_f32 && X_in.is_f32()) X_widened.emplace(detail::owned_copy(X_in, core::Precision::Float64));
  if (!both_f32 && Y_in.is_f32()) Y_widened.emplace(detail::owned_copy(Y_in, core::Precision::Float64));
  const Data &X = X_widened ? *X_widened : X_in;
  const Data &Y = Y_widened ? *Y_widened : Y_in;

  std::vector<double> out(n_x * n_y);

#ifdef DTWC_HAS_CUDA
  if (device == Device::GPU || device == Device::CPU_GPU) {
    // On cpu+gpu, Auto resolves to FP64 so GPU and CPU rows share uniform precision;
    // explicit FP32 runs both GPU and CPU rows in single precision.
    const GpuPrecision eff_gpu_precision =
      (device == Device::CPU_GPU && gpu_precision == GpuPrecision::Auto) ? GpuPrecision::FP64 : gpu_precision;

    if (device == Device::CPU_GPU && gpu_share == 0.0) {
      if (eff_gpu_precision == GpuPrecision::FP32) {
        const Data X32 = detail::owned_copy(X, core::Precision::Float32);
        const Data Y32 = detail::owned_copy(Y, core::Precision::Float32);
        compute_cdist_cpu_rows<float>(
          X32, Y32, 0, n_x, cfg,
          [&](std::size_t i) { return X32.series_f32(i); },
          [&](std::size_t j) { return Y32.series_f32(j); },
          out.data());
      } else {
        compute_cdist_cpu_rows<data_t>(
          X, Y, 0, n_x, cfg,
          [&](std::size_t i) { return X.series(i); },
          [&](std::size_t j) { return Y.series(j); },
          out.data());
      }
      return out;
    }

    cuda::CUDADistMatOptions opts;
    opts.device_id = device_index;
    opts.band = cfg.band;
    opts.precision = eff_gpu_precision;
    opts.use_squared_l2 = (cfg.metric == core::MetricType::SquaredL2);

    std::vector<std::span<const double>> y_spans(n_y);
    for (std::size_t j = 0; j < n_y; ++j) y_spans[j] = Y.series(j);

    const int threads = std::max(1, get_max_threads());
    const bool all_gpu = (device == Device::GPU) || (gpu_share == 1.0)
                         || (gpu_share < 0.0 && (n_x < 64 || threads <= 1)) || n_x == 1;
    if (all_gpu) {
      const auto parallel = [](std::size_t count, const std::function<void(std::size_t)> &task) {
        run_openmp(task, count, count > 64);
      };
      (void)cuda::compute_medoid_distances_cuda(
        n_x, [&](std::size_t i) { return X.series(i); }, y_spans, opts, parallel,
        [&](std::size_t first, [[maybe_unused]] std::size_t count, std::span<const double> block) {
          std::copy(block.begin(), block.end(), out.data() + first * n_y);
        });
      return out;
    }

    // Device::CPU_GPU: split X's rows concurrently between CUDA (thread 0) and CPU SIMD lanes (threads 1..P-1).
    const bool use_fp32 = cuda::cuda_uses_fp32(eff_gpu_precision, device_index);
    const double default_share = use_fp32 ? (13.0 * 28.5 / (13.0 * 28.5 + 55.0)) : (28.5 / (28.5 + 55.0));
    const double share = (gpu_share >= 0.0) ? gpu_share : default_share;
    const auto gpu_rows = std::clamp<std::size_t>(
      static_cast<std::size_t>(static_cast<double>(n_x) * share + 0.5), 1, n_x - 1);

    if (use_fp32) {
      const Data X32 = detail::owned_copy(X, core::Precision::Float32);
      const Data Y32 = detail::owned_copy(Y, core::Precision::Float32);
      run_hybrid_cdist_split<float>(
        X, y_spans, X32, Y32, gpu_rows, n_x, n_y, cfg, opts, threads,
        [&](std::size_t i) { return X32.series_f32(i); },
        [&](std::size_t j) { return Y32.series_f32(j); },
        out.data());
    } else {
      run_hybrid_cdist_split<data_t>(
        X, y_spans, X, Y, gpu_rows, n_x, n_y, cfg, opts, threads,
        [&](std::size_t i) { return X.series(i); },
        [&](std::size_t j) { return Y.series(j); },
        out.data());
    }
    return out;
  }
#endif

  if (both_f32) {
    compute_cdist_cpu_rows<float>(
      X, Y, 0, n_x, cfg,
      [&](std::size_t i) { return X.series_f32(i); },
      [&](std::size_t j) { return Y.series_f32(j); },
      out.data());
  } else {
    compute_cdist_cpu_rows<data_t>(
      X, Y, 0, n_x, cfg,
      [&](std::size_t i) { return X.series(i); },
      [&](std::size_t j) { return Y.series(j); },
      out.data());
  }
  return out;
}

std::vector<double> cdist(const Data &X, const Data &Y, const Config &config)
{
  return cdist(X, Y,
               { .variant = config.variant, .metric = config.metric, .missing = config.missing, .band = config.band },
               config.device, config.device_index, config.gpu_precision, config.gpu_share, config.fallback);
}

} // namespace dtwc
