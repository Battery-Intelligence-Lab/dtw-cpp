/**
 * @file tier1_method_resolution.hpp
 * @brief Pure device-aware method selection for the Tier-1 clustering API.
 */

#pragma once

#include <cstddef>
#include <string_view>

namespace dtwc::detail {

enum class Tier1ExecutionTarget {
  CPU,
  GPU,
  HPC
};

inline constexpr std::size_t kAutoPamSeriesLimit = 5000;

/**
 * Resolve the special ``auto`` method without changing explicit requests.
 *
 * HPC deliberately preserves ``auto`` because the remote process owns data
 * materialisation and therefore the final size-dependent choice.
 */
constexpr std::string_view resolve_tier1_method(
  std::string_view method, std::size_t n_series,
  Tier1ExecutionTarget target) noexcept
{
  if (method != "auto" || target == Tier1ExecutionTarget::HPC) return method;
  if (target == Tier1ExecutionTarget::GPU) return "pam";
  return n_series <= kAutoPamSeriesLimit ? "pam" : "clara";
}

} // namespace dtwc::detail
