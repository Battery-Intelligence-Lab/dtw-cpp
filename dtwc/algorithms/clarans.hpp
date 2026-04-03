/**
 * @file clarans.hpp
 * @brief CLARANS: Clustering Large Applications based on RANdomized Search.
 *
 * @details Experimental bounded mid-ground randomized k-medoids algorithm.
 *   Implements the original CLARANS neighborhood search with:
 *   - Hard budget controls (max_dtw_evals, max_neighbor)
 *   - Strictly improving swaps only (no neutral moves)
 *   - Deterministic per-restart RNG seeding
 *
 *   Reference:
 *   Ng, R.T. & Han, J. (2002). "CLARANS: A method for clustering objects for
 *   spatial data mining." IEEE TKDE, 14(5), 1003–1016.
 *
 * @warning Experimental — not exposed in CLI. Use FastCLARA for large N.
 *          Promote only after benchmarks justify.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

#include "../core/clustering_result.hpp"
#include <cstdint>

namespace dtwc {
class Problem;
namespace algorithms {

/// Configuration options for the CLARANS algorithm.
struct CLARANSOptions {
    int n_clusters   = 3;    ///< Number of clusters (k).
    int num_local    = 2;    ///< Number of random restarts.
    int max_neighbor = -1;   ///< Max non-improving swaps per restart (-1 = auto).
    int64_t max_dtw_evals = -1; ///< Hard budget on total DTW computations (-1 = no limit).
    unsigned random_seed = 42;  ///< RNG seed for determinism.
};

/**
 * @brief CLARANS: randomized k-medoids via neighborhood search.
 *
 * @details Experimental bounded mid-ground algorithm. For each restart, picks
 * random initial medoids, then iteratively tests random (medoid_out, x_in)
 * swaps, accepting only strictly improving ones. Stops when max_neighbor
 * consecutive non-improving swaps are seen or the DTW evaluation budget is
 * exhausted. Best result across all restarts is returned.
 *
 * Auto max_neighbor formula: max(250, (int)(0.0125 * k * (N - k)))
 *
 * @param prob  Problem instance with data loaded.
 * @param opts  CLARANS configuration options.
 * @return core::ClusteringResult with the best labels, medoid indices, and
 *         total cost found across all restarts.
 *
 * @throws std::runtime_error if prob has no data or n_clusters is invalid.
 *
 * @warning Use FastCLARA for large N. CLARANS is O(N) per swap evaluation,
 *          which is acceptable for moderate datasets.
 */
core::ClusteringResult clarans(Problem& prob, const CLARANSOptions& opts);

} // namespace algorithms
} // namespace dtwc
