/**
 * @file thread_scratch.hpp
 * @brief The thread_local scratch of the DTW kernels and the algorithms, given back when a top-level call returns.
 *
 * @details Per-thread scratch buffers for DTW kernels and clustering/barycenter algorithms are kept as block-scope
 *          `thread_local` vectors inside the translation units that use them (`dtw_dispatch.cpp`, `dtw_lanes.cpp`,
 *          `fast_pam.cpp`, `fast_clara.cpp`, `barycenter.cpp`, plus CUDA's launch workspace in `cuda_dtw.cu`), keeping
 *          TLS accesses local to each compilation unit under LTO. Buffers grow monotonically within a call so warmed
 *          threads do not reallocate. `release_thread_scratch()` empties every registered buffer across the OpenMP
 *          team (`ScratchRelease` invokes it at entry and exit of `dtwc::run`, while Tier-2 calls retain scratch across
 *          sweeps). Silhouette per-cluster accumulators (`scores.cpp`, `k * 64` bytes/thread) remain inline.
 *
 * @author Volkan Kumtepeli
 * @date 01 Nov 2026
 */

#pragma once

#include "dtw_options.hpp" // DistanceConfig
#include "storage.hpp"     // Precision

#include <cstddef>
#include <vector>

namespace dtwc::core {

/// Per-thread scratch budget (bytes) for `config`'s bound distance (including univariate lane buffers when active) on
/// series of up to `length` time steps at `precision`. Counts kernel rolling buffers, preprocessing/channel copies, and
/// WDTW's cached weight table (vector capacities may exceed these sizes by growth factor); Soft-DTW counts its value's
/// one column, never its gradient's matrices. Planners add algorithm tables.
std::size_t thread_scratch_bytes(const DistanceConfig &config, std::size_t length, Precision precision);

namespace detail {

/// The bytes `buffers` (std::vectors) hold, their capacities, each emptied, its block freed, when `release`: the body
/// of a function on the closed list.
template <typename... V>
std::size_t held_bytes(bool release, V &...buffers) noexcept
{
  const std::size_t bytes = (std::size_t{ 0 } + ... + (buffers.capacity() * sizeof(typename V::value_type)));
  if (release) (V().swap(buffers), ...);
  return bytes;
}

/// The closed list's functions: each returns what its buffers hold on the calling thread and, when `release`, empties
/// them.
std::size_t dispatch_scratch(bool release) noexcept;     ///< dtw_dispatch.cpp: the buffers its bound distances use
std::size_t lanes_scratch(bool release) noexcept;        ///< dtw_lanes.cpp: the lanes' rows
std::size_t envelope_scratch(bool release) noexcept;     ///< lower_bound_impl.hpp: compute_envelopes ring/prefix buffers
std::size_t fast_pam_scratch(bool release) noexcept;     ///< fast_pam.cpp: FasterPAM's per-slot losses
std::size_t clara_assign_scratch(bool release) noexcept; ///< fast_clara.cpp: FastCLARA's assignment table
std::size_t one_batch_pam_scratch(bool release) noexcept;///< one_batch_pam.cpp: OOB nearest-medoid buffers
std::size_t barycenter_scratch(bool release) noexcept;   ///< barycenter.cpp: DBA and Soft-DTW barycenter matrices

} // namespace detail

/// Empties every buffer on the closed list, and with CUDA the launch workspaces, on every thread of the OpenMP team,
/// the calling thread's too, in one parallel region, and returns the freed memory to the system
/// (release_free_memory): what a top-level call does as it starts and when it returns or throws. A team of the size
/// every region uses at most (full_team_threads()) holds every thread that ran one.
void release_thread_scratch() noexcept;

/// The bytes the closed list's buffers hold on each thread of the OpenMP team, by its thread number in one parallel
/// region (the calling thread's first): a test hook.
std::vector<std::size_t> thread_scratch_held();

/// release_thread_scratch() when it is made and when it goes out of scope, on return and on error: dtwc::run's guard,
/// so a run starts with none of the closed list's scratch an earlier call left (a Tier-2 k-sweep's) and leaves none.
struct ScratchRelease
{
  ScratchRelease() { release_thread_scratch(); }
  ScratchRelease(const ScratchRelease &) = delete;
  ScratchRelease &operator=(const ScratchRelease &) = delete;
  ~ScratchRelease() { release_thread_scratch(); }
};

} // namespace dtwc::core
