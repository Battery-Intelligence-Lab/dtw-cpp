/** @file warm_start.hpp
 *  @brief Shared deterministic FastPAM warm start for exact MIP backends.
 */

#pragma once

#include "../core/clustering_result.hpp"

#include <cstdint>

namespace dtwc {
class Problem;

namespace mip {

/// Build one invocation-local FastPAM incumbent for a MIP backend without
/// publishing its heuristic medoids or labels into the caller's Problem.
[[nodiscard]] core::ClusteringResult make_warm_start(
  Problem &prob, std::uint64_t random_seed);

} // namespace mip
} // namespace dtwc
