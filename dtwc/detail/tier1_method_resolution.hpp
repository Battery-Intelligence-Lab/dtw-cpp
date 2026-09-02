/**
 * @file tier1_method_resolution.hpp
 * @brief Pure device-aware method selection for the Tier-1 clustering API.
 */

#pragma once

#include "../core/storage.hpp" //!< For core::StoragePolicy

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

/**
 * Series-storage policy the Tier-1 route must install BEFORE Problem::set_data.
 *
 * A GPU distance schedule cannot read mmap-backed series
 * (Problem::fill_distance_matrix throws DeviceError), and StoragePolicy::Auto
 * spills a dataset larger than the free-RAM threshold to mmap. So the GPU target
 * pins Heap; CPU and HPC keep the Auto default. Explicit Tier-2 use is unchanged
 * — the DeviceError still guards a deliberately mapped Problem.
 */
constexpr core::StoragePolicy tier1_storage_policy(
  Tier1ExecutionTarget target) noexcept
{
  return target == Tier1ExecutionTarget::GPU ? core::StoragePolicy::Heap
                                             : core::StoragePolicy::Auto;
}

} // namespace dtwc::detail
