/**
 * @file cdist.cpp
 * @brief Cross-collection DTW distance matrix implementation across CPU SIMD lanes and GPU.
 *
 * @author Volkan Kumtepeli
 * @date 13 Nov 2026
 */

#include "cdist.hpp"

#include "base/error.hpp"
#include "base/missing_utils.hpp"
#include "base/parallelisation.hpp"
#include "config.hpp"
#include "core/dtw_dispatch.hpp"
#include "core/dtw_kernel.hpp"
#include "core/gpu_policy.hpp"
#include "core/storage.hpp"
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
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dtwc {
namespace {

// The refusals of a GPU cdist, which validate_cdist_device below reaches only in a build with a GPU backend.
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
[[noreturn]] void reject_cdist_gpu(const std::string &reason, const std::string &fix)
{
  core::detail::reject_gpu("cdist", reason, fix);
}
#endif

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
  core::detail::validate_gpu_distance_config("cdist", f32, cfg.ndim, cfg.variant.variant, cfg.missing);
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

  std::vector<detail::SeriesVerdict> verdicts(n, detail::SeriesVerdict::Usable);
  auto check_one = [&](std::size_t i) { verdicts[i] = detail::series_verdict(data, i, missing); };
  run_openmp(check_one, n, n > 64, 16);

  const bool nan_is_missing = (missing != core::MissingStrategy::Error);
  for (std::size_t i = 0; i < n; ++i) {
    if (verdicts[i] == detail::SeriesVerdict::Refused) {
      const std::string label = std::string(side) + " series '" + std::string(data.name(i))
                              + "' (index " + std::to_string(i) + ")";
      detail::refuse_series_unusable(data, i, label, "cdist", nan_is_missing);
    }
    if (verdicts[i] == detail::SeriesVerdict::EntirelyNaN) {
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
  if (n_x == 0 || n_y == 0 || cfg.band < 0 || !core::variant_uses_band(cfg.variant.variant)) return;

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
inline std::span<const T> series_at(const Data &data, std::size_t i) noexcept
{
  if constexpr (std::is_same_v<T, float>)
    return data.series_f32(i);
  else
    return data.series(i);
}

template <typename T>
bool can_prepack_panels(const Data &Y) noexcept
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
      max_len = std::max(max_len, series_at<T>(Y, b0 + w).size());
    const std::size_t row_bytes = W * sizeof(T);
    if (max_len > (kMaxPrepackBytes - total_bytes) / row_bytes) return false;
    total_bytes += max_len * row_bytes;
  }
  return true;
}

template <typename OutT>
inline void store_cdist_value(OutT *dst, std::size_t row, std::size_t col, double v)
{
  if constexpr (std::is_same_v<OutT, float>) {
    const float vf = static_cast<float>(v);
    if (!std::isfinite(vf) && std::isfinite(v))
      core::refuse_float32_range("cdist", row, col, v);
    *dst = vf;
  } else {
    *dst = v;
  }
}

template <typename CpuT>
class CpuCdistPlan
{
public:
  CpuCdistPlan(const Data &X, const Data &Y, const core::DistanceConfig &cfg)
    : X_(X), Y_(Y), cfg_(cfg), n_y_(static_cast<std::size_t>(Y.size())),
      n_panels_((n_y_ + W - 1) / W),
      packed_fn_(core::resolve_dtw_packed_block_fn<CpuT>(cfg)),
      block_fn_(core::resolve_dtw_block_fn<CpuT>(cfg)),
      use_panels_(packed_fn_ && can_prepack_panels<CpuT>(Y))
  {
    if (use_panels_) {
      panels_ = core::pack_series_panels<CpuT>(
        cfg_, n_y_, [&Y](std::size_t j) { return series_at<CpuT>(Y, j); }, true);
    } else if (!block_fn_) {
      pair_fn_ = core::resolve_dtw_fn<CpuT>(cfg_, X_, &Y_);
    }
  }

  template <typename OutT>
  void compute_row(std::size_t i, OutT *out) const
  {
    OutT *const row_out = out + i * n_y_;
    if (use_panels_) {
      const auto x_prep = core::prepare_dtw_query<CpuT>(cfg_, series_at<CpuT>(X_, i));
      std::array<double, W> d;
      for (std::size_t p = 0; p < n_panels_; ++p) {
        const std::size_t b0 = p * W;
        const std::size_t cnt = std::min(W, n_y_ - b0);
        packed_fn_(x_prep, panels_[p].view(), d);
        for (std::size_t w = 0; w < cnt; ++w) store_cdist_value(row_out + b0 + w, i, b0 + w, d[w]);
      }
    } else if (block_fn_) {
      std::array<std::span<const CpuT>, W> ys;
      std::array<double, W> d;
      for (std::size_t p = 0; p < n_panels_; ++p) {
        const std::size_t b0 = p * W;
        const std::size_t cnt = std::min(W, n_y_ - b0);
        for (std::size_t w = 0; w < W; ++w) ys[w] = series_at<CpuT>(Y_, b0 + std::min(w, cnt - 1));
        block_fn_(series_at<CpuT>(X_, i), ys, d);
        for (std::size_t w = 0; w < cnt; ++w) store_cdist_value(row_out + b0 + w, i, b0 + w, d[w]);
      }
    } else {
      for (std::size_t j = 0; j < n_y_; ++j)
        store_cdist_value(row_out + j, i, j, pair_fn_(series_at<CpuT>(X_, i), series_at<CpuT>(Y_, j)));
    }
  }

  template <typename OutT>
  void compute_rows(std::size_t x_begin, std::size_t x_end, OutT *out) const
  {
    const std::size_t n_rows = x_end - x_begin;
    if (n_rows == 0 || n_y_ == 0) return;

    if (use_panels_) {
      const auto threads = static_cast<std::size_t>(std::max(1, get_max_threads()));
      if (n_rows >= 4 * threads || n_panels_ <= 1) {
        auto row_task = [&](std::size_t r) { compute_row(x_begin + r, out); };
        run_openmp(row_task, n_rows, (n_rows * n_panels_) > 4);
      } else {
        const std::size_t total_tiles = n_rows * n_panels_;
        auto tile_task = [&](std::size_t idx) {
          const std::size_t i = x_begin + (idx / n_panels_);
          const std::size_t p = idx % n_panels_;
          const std::size_t b0 = p * W;
          const std::size_t cnt = std::min(W, n_y_ - b0);
          const auto x_prep = core::prepare_dtw_query<CpuT>(cfg_, series_at<CpuT>(X_, i));
          std::array<double, W> d;
          packed_fn_(x_prep, panels_[p].view(), d);
          OutT *const row_out = out + i * n_y_ + b0;
          for (std::size_t w = 0; w < cnt; ++w) store_cdist_value(row_out + w, i, b0 + w, d[w]);
        };
        run_openmp(tile_task, total_tiles, total_tiles > 4);
      }
      return;
    }

    if (block_fn_) {
      const std::size_t total_tiles = n_rows * n_panels_;
      auto stream_tile = [&](std::size_t idx) {
        const std::size_t i = x_begin + (idx / n_panels_);
        const std::size_t p = idx % n_panels_;
        const std::size_t b0 = p * W;
        const std::size_t cnt = std::min(W, n_y_ - b0);
        std::array<std::span<const CpuT>, W> ys;
        for (std::size_t w = 0; w < W; ++w) ys[w] = series_at<CpuT>(Y_, b0 + std::min(w, cnt - 1));
        std::array<double, W> d;
        block_fn_(series_at<CpuT>(X_, i), ys, d);
        OutT *const row_out = out + i * n_y_ + b0;
        for (std::size_t w = 0; w < cnt; ++w) store_cdist_value(row_out + w, i, b0 + w, d[w]);
      };
      run_openmp(stream_tile, total_tiles, total_tiles > 4);
      return;
    }

    const std::size_t total_pairs = n_rows * n_y_;
    auto pair_task = [&](std::size_t idx) {
      const std::size_t i = x_begin + (idx / n_y_);
      const std::size_t j = idx % n_y_;
      store_cdist_value(out + i * n_y_ + j, i, j, pair_fn_(series_at<CpuT>(X_, i), series_at<CpuT>(Y_, j)));
    };
    run_openmp(pair_task, total_pairs, total_pairs > 8);
  }

private:
  static constexpr std::size_t W = core::dtw_lanes<CpuT>;
  const Data &X_;
  const Data &Y_;
  const core::DistanceConfig &cfg_;
  std::size_t n_y_;
  std::size_t n_panels_;
  std::function<void(std::span<const CpuT>, const core::PackedLanesView<CpuT> &, std::span<double>)> packed_fn_;
  std::function<void(std::span<const CpuT>, std::span<const std::span<const CpuT>>, std::span<double>)> block_fn_;
  std::function<double(std::span<const CpuT>, std::span<const CpuT>)> pair_fn_;
  bool use_panels_;
  std::vector<core::PackedLanesPanel<CpuT>> panels_;
};

#ifdef DTWC_HAS_CUDA
template <typename OutT>
void copy_cuda_block(OutT *out, std::size_t row_offset, std::size_t count, std::size_t n_y,
                     std::span<const double> block)
{
  for (std::size_t r = 0; r < count; ++r) {
    const std::size_t row = row_offset + r;
    const double *src = block.data() + r * n_y;
    OutT *dst = out + row * n_y;
    for (std::size_t j = 0; j < n_y; ++j) store_cdist_value(dst + j, row, j, src[j]);
  }
}

template <typename CpuT, typename OutT>
void run_hybrid_cdist_split(
  const Data &X_gpu, const std::vector<std::span<const double>> &y_gpu_spans,
  const Data &X_cpu, const Data &Y_cpu,
  double gpu_share, std::size_t n_x, std::size_t n_y,
  const core::DistanceConfig &cfg, const cuda::CUDADistMatOptions &opts,
  int threads, OutT *out)
{
  const CpuCdistPlan<CpuT> cpu_plan(X_cpu, Y_cpu, cfg);

  auto run_gpu_range = [&](std::size_t begin, std::size_t count) {
    const auto serial_pad = [](std::size_t rows, const std::function<void(std::size_t)> &task) {
      for (std::size_t k = 0; k < rows; ++k) task(k);
    };
    (void)cuda::compute_medoid_distances_cuda(
      count, [&](std::size_t i) { return X_gpu.series(begin + i); }, y_gpu_spans, opts, serial_pad,
      [&](std::size_t first, std::size_t blk_count, std::span<const double> block) {
        copy_cuda_block(out, begin + first, blk_count, n_y, block);
      });
  };

  detail::FailureSlots<std::size_t> failures(static_cast<std::size_t>(threads), n_x);

  if (gpu_share >= 0.0) {
    const auto gpu_rows = std::clamp<std::size_t>(
      static_cast<std::size_t>(static_cast<double>(n_x) * gpu_share + 0.5), 1, n_x - 1);
#  ifdef _OPENMP
#  pragma omp parallel num_threads(threads)
    {
      auto &slot = failures.current();
      const int tid = omp_get_thread_num();
      if (tid == 0) {
        try {
          run_gpu_range(0, gpu_rows);
        } catch (...) {
          detail::FailureSlots<std::size_t>::record(slot, 0, std::current_exception());
        }
      }
#  pragma omp for schedule(dynamic, 1) nowait
      for (std::ptrdiff_t r = static_cast<std::ptrdiff_t>(gpu_rows);
           r < static_cast<std::ptrdiff_t>(n_x); ++r) {
        const auto row = static_cast<std::size_t>(r);
        if (row > slot.failed_at) continue;
        try {
          cpu_plan.compute_row(row, out);
        } catch (...) {
          detail::FailureSlots<std::size_t>::record(slot, row, std::current_exception());
        }
      }
    }
#  else
    (void)threads;
    run_gpu_range(0, gpu_rows);
    cpu_plan.compute_rows(gpu_rows, n_x, out);
#  endif
  } else {
    // Dynamic 2-ended work queue: GPU claims coarse batches from head, CPU threads claim fine batches from tail.
    std::mutex q_mtx;
    std::size_t head = 0;
    std::size_t tail = n_x;
    const std::size_t gpu_batch = std::clamp<std::size_t>(n_x / 4, 64, 512);
    const std::size_t cpu_batch = std::clamp<std::size_t>(n_x / (4 * static_cast<std::size_t>(threads)), 1, 16);
    auto claim_head = [&](std::size_t &b, std::size_t &e) {
      std::lock_guard<std::mutex> lk(q_mtx);
      if (head >= tail) return false;
      b = head;
      e = std::min(tail, head + gpu_batch);
      head = e;
      return true;
    };
    auto claim_tail = [&](std::size_t &b, std::size_t &e) {
      std::lock_guard<std::mutex> lk(q_mtx);
      if (head >= tail) return false;
      e = tail;
      b = (tail - head > cpu_batch) ? (tail - cpu_batch) : head;
      tail = b;
      return true;
    };
#  ifdef _OPENMP
#  pragma omp parallel num_threads(threads)
    {
      auto &slot = failures.current();
      const int tid = omp_get_thread_num();
      if (tid == 0) {
        std::size_t b = 0, e = 0;
        try {
          while (claim_head(b, e)) run_gpu_range(b, e - b);
        } catch (...) {
          detail::FailureSlots<std::size_t>::record(slot, b, std::current_exception());
        }
      }
      std::size_t b = 0, e = 0;
      std::size_t r = n_x;
      try {
        while (claim_tail(b, e)) {
          for (r = b; r < e; ++r) cpu_plan.compute_row(r, out);
        }
      } catch (...) {
        detail::FailureSlots<std::size_t>::record(slot, r, std::current_exception());
      }
    }
#  else
    (void)threads;
    (void)claim_tail;
    (void)cpu_batch;
    (void)claim_head;
    run_gpu_range(0, n_x);
#  endif
  }

  failures.rethrow_lowest();
}
#endif

template <typename OutT>
std::vector<OutT> cdist_impl(const Data &X_in, const Data &Y_in,
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
    detail::say_fallback(device, device_index, refusal.reason());
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

  std::vector<OutT> out(n_x * n_y);

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
        CpuCdistPlan<float>(X32, Y32, cfg).compute_rows(0, n_x, out.data());
      } else {
        CpuCdistPlan<data_t>(X, Y, cfg).compute_rows(0, n_x, out.data());
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
        [&](std::size_t first, std::size_t count, std::span<const double> block) {
          copy_cuda_block(out.data(), first, count, n_y, block);
        });
      return out;
    }

    const bool use_fp32 = cuda::cuda_uses_fp32(eff_gpu_precision, device_index);
    if (use_fp32) {
      const Data X32 = detail::owned_copy(X, core::Precision::Float32);
      const Data Y32 = detail::owned_copy(Y, core::Precision::Float32);
      run_hybrid_cdist_split<float, OutT>(
        X, y_spans, X32, Y32, gpu_share, n_x, n_y, cfg, opts, threads, out.data());
    } else {
      run_hybrid_cdist_split<data_t, OutT>(
        X, y_spans, X, Y, gpu_share, n_x, n_y, cfg, opts, threads, out.data());
    }
    return out;
  }
#endif

  if (both_f32)
    CpuCdistPlan<float>(X, Y, cfg).compute_rows(0, n_x, out.data());
  else
    CpuCdistPlan<data_t>(X, Y, cfg).compute_rows(0, n_x, out.data());
  return out;
}

} // namespace

std::vector<double> cdist(const Data &X_in, const Data &Y_in,
                          const core::DistanceConfig &distance,
                          Device device,
                          int device_index,
                          GpuPrecision gpu_precision,
                          double gpu_share,
                          bool fallback)
{
  return cdist_impl<double>(X_in, Y_in, distance, device, device_index, gpu_precision, gpu_share, fallback);
}

std::vector<double> cdist(const Data &X, const Data &Y, const Config &config)
{
  if (config.matrix_dtype == core::Precision::Float32) {
    throw InvalidInput("cdist(Data, Data, Config) returns double; call cdist_f32(...) for matrix_dtype = float32.");
  }
  return cdist(X, Y,
               { .variant = config.variant, .metric = config.metric, .missing = config.missing, .band = config.band },
               config.device, config.device_index, config.gpu_precision, config.gpu_share, config.fallback);
}

std::vector<float> cdist_f32(const Data &X_in, const Data &Y_in,
                             const core::DistanceConfig &distance,
                             Device device,
                             int device_index,
                             GpuPrecision gpu_precision,
                             double gpu_share,
                             bool fallback)
{
  return cdist_impl<float>(X_in, Y_in, distance, device, device_index, gpu_precision, gpu_share, fallback);
}

std::vector<float> cdist_f32(const Data &X, const Data &Y, const Config &config)
{
  return cdist_f32(X, Y,
                   { .variant = config.variant, .metric = config.metric, .missing = config.missing, .band = config.band },
                   config.device, config.device_index, config.gpu_precision, config.gpu_share, config.fallback);
}

} // namespace dtwc
