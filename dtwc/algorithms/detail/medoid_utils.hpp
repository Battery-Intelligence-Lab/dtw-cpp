/**
 * @file medoid_utils.hpp
 * @brief Internal helper utilities for medoid-based clustering algorithms.
 *
 * @details Provides reusable medoid-assignment logic decoupled from any
 * specific clustering algorithm or Problem type. All utilities accept a
 * distance function of signature `(int, int) -> double` so they work with
 * dense distance matrices, lazy-evaluation wrappers, or any other backend.
 *
 * Used by FastPAM, CLARANS, and improved CLARA.
 *
 * @author Volkan Kumtepeli
 */

#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::algorithms::detail {

/// Assign each of N points to the nearest medoid.
/// @param dist_fn  Distance function: dist_fn(point, medoid) -> double
/// @param medoids  Medoid indices
/// @param N        Total number of points
/// @param labels   [out] Cluster label per point [0, k)
/// @return Total cost (sum of distances to nearest medoid)
template <typename DistFn>
double assign_to_nearest(DistFn dist_fn, const std::vector<int>& medoids, int N,
                         std::vector<int>& labels)
{
    const int k = static_cast<int>(medoids.size());
    labels.resize(N);
    double total_cost = 0.0;

    for (int p = 0; p < N; ++p) {
        double best_dist = std::numeric_limits<double>::max();
        int best_label = 0;
        for (int m = 0; m < k; ++m) {
            double d = dist_fn(p, medoids[m]);
            if (d < best_dist) {
                best_dist = d;
                best_label = m;
            }
        }
        labels[p] = best_label;
        total_cost += best_dist;
    }
    return total_cost;
}

/// For each point, find nearest and second-nearest medoid.
/// @param dist_fn      Distance function: dist_fn(point, medoid) -> double
/// @param medoids      Medoid indices
/// @param N            Total number of points
/// @param nearest      [out] Index into medoids array of nearest medoid per point
/// @param nearest_dist [out] Distance to nearest medoid per point
/// @param second_dist  [out] Distance to second-nearest medoid per point
template <typename DistFn>
void compute_nearest_and_second(DistFn dist_fn, const std::vector<int>& medoids, int N,
                                std::vector<int>& nearest,
                                std::vector<double>& nearest_dist,
                                std::vector<double>& second_dist)
{
    const int k = static_cast<int>(medoids.size());
    nearest.resize(N);
    nearest_dist.resize(N);
    second_dist.resize(N);

    for (int p = 0; p < N; ++p) {
        double best = std::numeric_limits<double>::max();
        double second = std::numeric_limits<double>::max();
        int best_idx = 0;

        for (int m = 0; m < k; ++m) {
            double d = dist_fn(p, medoids[m]);
            if (d < best) {
                second = best;
                best = d;
                best_idx = m;
            } else if (d < second) {
                second = d;
            }
        }
        nearest[p] = best_idx;
        nearest_dist[p] = best;
        second_dist[p] = second;
    }
}

/// Find the medoid within a cluster: the point minimising sum-of-distances to
/// other cluster members.
/// @param dist_fn    Distance function: dist_fn(i, j) -> double
/// @param labels     Cluster label per point
/// @param cluster_id Which cluster to find medoid for
/// @param N          Total number of points
/// @return Index of the medoid point; on tie the smallest index wins.
///         Returns -1 if no point belongs to cluster_id.
template <typename DistFn>
int find_cluster_medoid(DistFn dist_fn, const std::vector<int>& labels, int cluster_id, int N)
{
    int best_point = -1;
    double best_cost = std::numeric_limits<double>::max();

    for (int p = 0; p < N; ++p) {
        if (labels[p] != cluster_id) continue;
        double cost = 0.0;
        for (int q = 0; q < N; ++q) {
            if (labels[q] == cluster_id)
                cost += dist_fn(p, q);
        }
        if (cost < best_cost || (cost == best_cost && (best_point < 0 || p < best_point))) {
            best_cost = cost;
            best_point = p;
        }
    }
    return best_point;
}

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
