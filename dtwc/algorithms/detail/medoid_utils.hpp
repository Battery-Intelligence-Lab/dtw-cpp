/**
 * @file medoid_utils.hpp
 * @brief Medoid-list validation shared by the medoid-based clustering algorithms.
 *
 * @details D1: the former shared `assign_to_nearest`, `compute_nearest_and_second`
 * and `find_cluster_medoid` helpers were removed from this header (fast_pam.cpp
 * keeps its own file-local `compute_nearest_and_second`). They had no caller and
 * their semantics had drifted from every shipping assignment loop (no
 * `require_finite_medoid_distance`, plain `+=` instead of the ordered published
 * objective, a `DBL_MAX` sentinel). The shipping copies differ by parallel vs
 * serial, index space, distance signature and per-element side effects, so a
 * single helper could only absorb them via runtime switches in the library's
 * hottest loops. `validate_medoids` stays: it is shared and is pure validation.
 *
 * @author Volkan Kumtepeli
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::algorithms::detail {

/// Validate medoid indices: must be non-empty, unique, and in range [0, N).
/// @throws std::runtime_error on any violation.
inline void validate_medoids(const std::vector<int>& medoids, int N)
{
    if (medoids.empty())
        throw std::runtime_error("validate_medoids: empty medoid list");
    for (int m : medoids) {
        if (m < 0 || m >= N)
            throw std::runtime_error("validate_medoids: medoid index " + std::to_string(m)
                                     + " out of range [0, " + std::to_string(N) + ")");
    }
    auto sorted = medoids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::runtime_error("validate_medoids: duplicate medoid indices");
}

} // namespace dtwc::algorithms::detail
