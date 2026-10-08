// count_lanes.hpp: the kit's lane-call counter (never shipped code). Force-included (cl /FI, clang -include) into a kit
// copy of the shipped dtw_lanes.cpp, it renames that file's one call of dtw_kernel_lanes to counted_lanes, which counts
// the call and forwards it unchanged; rprobe.cpp built with -DRL_COUNT prints the count after its fills. A probe so
// built links this object before the version's dtwc_core library, whose own dtw_lanes object is then not pulled in.
#pragma once
#include "core/dtw_kernel.hpp"

#include <atomic>
#include <utility>

namespace rl_count {
inline std::atomic<long long> lane_calls{ 0 };
}

namespace dtwc::core {
// Whatever the version's signature (base: x, ys, n, band, dist, cell; RL: xs, ys, band, dist, cell).
template <typename T, typename... Args>
auto counted_lanes(Args &&...args)
{
  rl_count::lane_calls.fetch_add(1, std::memory_order_relaxed);
  return dtw_kernel_lanes<T>(std::forward<Args>(args)...);
}
} // namespace dtwc::core

#define dtw_kernel_lanes counted_lanes
