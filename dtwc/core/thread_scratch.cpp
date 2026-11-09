/**
 * @file thread_scratch.cpp
 * @brief The closed list of the threads' scratch, its release, and thread_scratch_bytes (see thread_scratch.hpp).
 * @author Volkan Kumtepeli
 */

#include "thread_scratch.hpp"

#include "distance_matrix.hpp"          // release_free_memory
#include "dtw_kernel.hpp"               // dtw_lanes
#include "../base/parallelisation.hpp"  // full_team_threads
#include "../base/settings.hpp"         // data_t

#ifdef DTWC_HAS_CUDA
#include "../cuda/cuda_dtw.cuh" // release_thread_workspace
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc::core {

namespace {

/// The closed list: every function, each beside the code that uses its buffers (thread_scratch.hpp says why).
constexpr std::size_t (*closed_list[])(bool) noexcept = {
  detail::dispatch_scratch,
  detail::lanes_scratch,
  detail::fast_pam_scratch,
  detail::clara_assign_scratch,
};

/// What the closed list holds on the calling thread; each buffer emptied when `release`, and CUDA's launch workspaces
/// freed (which thread_scratch_held() does not count: cuda::thread_workspace_bytes() does).
std::size_t this_thread(bool release) noexcept
{
  std::size_t bytes = 0;
  for (const auto scratch : closed_list) bytes += scratch(release);
#ifdef DTWC_HAS_CUDA
  if (release) dtwc::cuda::release_thread_workspace();
#endif
  return bytes;
}

} // namespace

void release_thread_scratch() noexcept
{
#ifdef _OPENMP
#pragma omp parallel num_threads(full_team_threads())
  (void)this_thread(true);
#else
  (void)this_thread(true);
#endif
  release_free_memory();
}

std::vector<std::size_t> thread_scratch_held()
{
  std::vector<std::size_t> bytes(static_cast<std::size_t>(full_team_threads()), 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(bytes.size()))
  bytes[static_cast<std::size_t>(omp_get_thread_num())] = this_thread(false);
#else
  bytes[0] = this_thread(false);
#endif
  return bytes;
}

// ----------------------------------------------------------------------------
// thread_scratch_bytes: each buffer's sizing rule, the order resolve_dtw_fn takes
// ----------------------------------------------------------------------------

namespace {

template <typename T>
std::size_t scratch_bytes(const DistanceConfig &c, std::size_t length)
{
  const std::size_t value = sizeof(T), values = length * c.ndim;
  // run_dtw (dtw_kernel.hpp): without a band dtw_linear's column of the shorter series; under one dtw_linear's only for
  // a pair it covers whole (the longer series at most band + 1 long, or one of length 1), dtw_banded's of the longer.
  std::size_t dp = length;
  if (c.band >= 0 && length > static_cast<std::size_t>(c.band) + 1) dp = static_cast<std::size_t>(c.band) + 1 + length;
  dp *= value;
  if (c.variant.mv_mode == MVMode::Independent && c.ndim > 1) return 2 * length * value + dp; // cx, cy (warping.hpp)
  switch (c.missing) {
  case MissingStrategy::ZeroCost:
  case MissingStrategy::AROW: return dp;
  case MissingStrategy::Interpolate: return 2 * length * value + dp; // x_buffer, y_buffer (make_interpolate)
  case MissingStrategy::Error: break;
  }
  switch (c.variant.variant) {
  case DTWVariant::DDTW: return 2 * values * value + dp; // dx, dy (warping_ddtw.hpp; make_ddtw's, multivariate)
  case DTWVariant::WDTW: return length * value + dp;     // cached_w: max_dev + 1 weights (warping_wdtw.hpp)
  case DTWVariant::ADTW: return dp;
  case DTWVariant::SoftDTW: return length * value;       // dtw_kernel_linear's column, no band
  case DTWVariant::MSM: return 2 * length * value;       // prev_buf, curr_buf (msm.hpp)
  case DTWVariant::TWE: return length == 0 ? 0 : 2 * (length + 1) * value; // prev_buf, curr_buf, padded (twe.hpp)
  case DTWVariant::Standard: break;
  }
  // The lanes (resolve_dtw_block_fn): y_buf and s_buf, a row of dtw_lanes<T> values a time step each.
  return (c.ndim == 1 ? 2 * length * dtw_lanes<T> * value : 0) + dp;
}

} // namespace

std::size_t thread_scratch_bytes(const DistanceConfig &config, std::size_t length, Precision precision)
{
  return precision == Precision::Float32 ? scratch_bytes<float>(config, length) : scratch_bytes<data_t>(config, length);
}

} // namespace dtwc::core
