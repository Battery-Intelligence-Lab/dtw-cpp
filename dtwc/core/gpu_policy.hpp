/**
 * @file gpu_policy.hpp
 * @brief Shared GPU capability checks and error wording for Problem and cdist.
 *
 * @author Volkan Kumtepeli
 * @date 26 Nov 2026
 */

#pragma once

#include "../base/env.hpp"
#include "../base/error.hpp"
#include "../core/dtw_options.hpp"
#include "../core/storage.hpp"

#include <string>
#include <string_view>

namespace dtwc::core::detail {

inline const char *gpu_variant_name(DTWVariant v)
{
  switch (v) {
  case DTWVariant::Standard: return "Standard";
  case DTWVariant::DDTW: return "DDTW";
  case DTWVariant::WDTW: return "WDTW";
  case DTWVariant::ADTW: return "ADTW";
  case DTWVariant::SoftDTW: return "SoftDTW";
  case DTWVariant::MSM: return "MSM";
  case DTWVariant::TWE: return "TWE";
  }
  return "?";
}

inline const char *gpu_missing_name(MissingStrategy m)
{
  switch (m) {
  case MissingStrategy::Error: return "Error";
  case MissingStrategy::ZeroCost: return "ZeroCost";
  case MissingStrategy::AROW: return "AROW";
  case MissingStrategy::Interpolate: return "Interpolate";
  }
  return "?";
}

#ifdef DTWC_HAS_CUDA
inline constexpr std::string_view kGpuBackendName = "CUDA";
#elif defined(DTWC_HAS_METAL)
inline constexpr std::string_view kGpuBackendName = "Metal";
#else
inline constexpr std::string_view kGpuBackendName = "GPU";
#endif

[[noreturn]] inline void reject_gpu(std::string_view where, const std::string &request, const std::string &fix)
{
  throw GpuRefusal(std::string(where) + ": " + std::string(kGpuBackendName) + " " + request
                     + "; no backend call or CPU fallback was attempted. " + fix,
                   std::string(kGpuBackendName) + " " + request);
}

inline void validate_gpu_distance_config(std::string_view where, bool series_f32, std::size_t ndim,
                                         DTWVariant variant, MissingStrategy missing)
{
  if (series_f32)
    reject_gpu(where, "computes from Float64 series, but precision = Float32 was requested",
               "Load the series as Float64, or use device cpu.");
  if (variant != DTWVariant::Standard)
    reject_gpu(where,
               std::string("implements Standard DTW only, but variant = ") + gpu_variant_name(variant)
                 + " was requested",
               "Use device cpu for this variant.");
  if (missing != MissingStrategy::Error)
    reject_gpu(where,
               std::string("has no missing-data strategy, but missing_strategy = ") + gpu_missing_name(missing)
                 + " was requested",
               "Use device cpu, or missing_strategy Error for data without NaN.");
  if (ndim > 1)
    reject_gpu(where, "is univariate only, but ndim = " + std::to_string(ndim) + " was requested",
               "Use device cpu for multivariate series.");
}

} // namespace dtwc::core::detail
