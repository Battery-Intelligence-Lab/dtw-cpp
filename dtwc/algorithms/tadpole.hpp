/**
 * @file tadpole.hpp
 * @brief TADPole: density-peaks clustering with conditionally admissible DTW pruning.
 *
 * @details Density-peaks clustering (Rodriguez & Laio, *Science* 344:1492-1496,
 *   2014) accelerated by the admissible lower/upper-bound pruning of Begum,
 *   Ulanova, Wang & Keogh, "Accelerating Dynamic Time Warping Clustering with a
 *   Novel Admissible Pruning Strategy," KDD 2015 (extended: arXiv:1612.00637).
 *
 *   Why this is the one method that need NOT materialise the full N×N matrix
 *   (contrast k-medoids / MIP / LR-core, which read essentially every entry):
 *   the local density uses the CUTOFF kernel ρ_i = |{ j≠i : d(i,j) < dc }| — a
 *   BINARY per-pair test. For equal-length Standard-L1 univariate series, a
 *   lower bound LB and upper bound UB decide neighbours without exact DTW:
 *       LB(i,j) ≥ dc  ⇒  d(i,j) ≥ dc  ⇒  NOT a neighbour   (skip DTW)
 *       UB(i,j) <  dc  ⇒  d(i,j) <  dc ⇒  IS  a neighbour   (skip DTW)
 *       otherwise                       compute exact DTW.
 *   These hold in exact arithmetic; agreement with the brute-force path when a
 *   rounded bound lands at dc (or at the δ stage's best) is not proved.
 *   The Gaussian kernel Σ exp(-(d/dc)²) is deliberately not used because it
 *   requires every exact distance.
 *
 *   LB = symmetric LB_Keogh (reusing each series' cached envelope across pairs).
 *   UB = the no-warp diagonal cost Σ_t |x_t − y_t| for equal-length series
 *   (valid because the diagonal always satisfies any Sakoe-Chiba band ≥ 0).
 *
 * @author Volkan Kumtepeli
 * @date 08 Jul 2026
 */

#pragma once

#include "../core/clustering_result.hpp"
#include "../core/dtw_options.hpp"
#include "../core/storage.hpp"

#include <cstddef>

namespace dtwc {

class Problem; // Forward declaration

namespace algorithms {

/// Pruning ledger for a TADPole run.
///
/// The comparable baseline is brute-force density-peaks, which computes every one
/// of the N(N-1)/2 unordered pairs exactly. `dtw_calls` counts the UNIQUE pairs
/// this run computed exactly across ALL stages (density + δ), deduplicated — so
/// `pruned_fraction() = 1 - dtw_calls / total_pairs` is exactly the fraction of
/// the brute-force DTW work avoided. `pruned_by_lb`/`pruned_by_ub` break down the
/// density stage for insight (a δ-stage pair can still be computed later).
struct TADPoleStats {
  std::size_t total_pairs = 0;   ///< N(N-1)/2 — the brute-force DTW count.
  std::size_t dtw_calls = 0;     ///< Unique pairs computed exactly (all stages, deduplicated).
  std::size_t pruned_by_lb = 0;  ///< Density-stage NOT-neighbour decisions by LB ≥ dc.
  std::size_t pruned_by_ub = 0;  ///< Density-stage neighbour decisions by UB < dc.
  double dc = 0.0;               ///< Cutoff distance used.
  bool pruning_enabled = false;  ///< Whether the LB/UB path ran at all (see @p prune).

  /// Fraction of the brute-force DTW work avoided. 0 on the brute path.
  double pruned_fraction() const
  {
    return total_pairs == 0 ? 0.0
                            : 1.0 - static_cast<double>(dtw_calls) / static_cast<double>(total_pairs);
  }
};

/**
 * @brief TADPole density-peaks clustering with conditionally admissible LB/UB
 *        DTW pruning.
 *
 * @param prob        Problem with data loaded. Does NOT call fill_distance_matrix();
 *                    the whole point is to avoid the full matrix.
 * @param n_clusters  Number of clusters k (top-k points by γ = ρ·δ become centers).
 * @param dc          Cutoff distance for the density kernel (must be > 0).
 * @param prune       true  → apply LB/UB pruning when admissible (Standard L1
 *                    univariate float64 DTW with MissingStrategy::Error and an
 *                    unfilled float64 distance matrix; unequal-length pairs and
 *                    other configurations fall back to exact DTW, reported via
 *                    TADPoleStats::pruning_enabled);
 *                    false → compute every DTW (brute-force oracle).
 * @param stats       If non-null, receives the pruning ledger.
 * @return core::ClusteringResult: labels[i] ∈ [0,k), medoid_indices = the k
 *         density-peak centers, total_cost = Σ_i d(i, its center).
 * @throws InvalidInput on n_clusters ∉ [1, N] or dc ≤ 0.
 *
 * @note Writes the result back into `prob` (`clusters_ind`, `centroids_ind`,
 *       `n_clusters`) via `prob.set_result(...)`.
 */
core::ClusteringResult tadpole(Problem& prob, index_t n_clusters, double dc,
                               bool prune = true, TADPoleStats* stats = nullptr);

/**
 * @brief Choose dc as a percentile of a deterministic exact-DTW subsample.
 *
 * @details Rodriguez & Laio pick dc so the average neighbour count is ≈1–2% of
 *   N. Estimating that needs the distance distribution; we sample a deterministic,
 *   evenly spaced subset of `min(N, 64)` series across `[0, N)` (no RNG), compute
 *   their exact pairwise DTW distances, and return the `percentile`-th percentile.
 *   These subsample DTWs are counted honestly if a caller tracks the budget.
 *
 * @param prob        Problem with data loaded.
 * @param percentile  Target percentile in (0, 100). Default 2.0.
 * @return dc > 0. Deterministic: identical inputs → identical dc.
 */
double tadpole_auto_dc(Problem& prob, double percentile = 2.0);

namespace detail {

/// Whether the L1 bounds are admissible for a distance (Standard DTW under L1, MissingStrategy::Error) on univariate
/// float64 series (`series`) into a float64 matrix (`matrix`). `tadpole` also prunes only into a matrix not yet filled.
/// Plain values: the CLI's memory plan asks before any series is read.
[[nodiscard]] bool bounds_valid(core::DTWVariant variant, core::MetricType metric, std::size_t ndim,
                                core::Precision series, core::Precision matrix, core::MissingStrategy missing);

} // namespace detail
} // namespace algorithms
} // namespace dtwc
