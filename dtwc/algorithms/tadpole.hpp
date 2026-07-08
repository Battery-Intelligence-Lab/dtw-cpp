/**
 * @file tadpole.hpp
 * @brief TADPole: density-peaks clustering with admissible DTW pruning.
 *
 * @details Density-peaks clustering (Rodriguez & Laio, *Science* 344:1492-1496,
 *   2014) accelerated by the admissible lower/upper-bound pruning of Begum,
 *   Ulanova, Wang & Keogh, "Accelerating Dynamic Time Warping Clustering with a
 *   Novel Admissible Pruning Strategy," KDD 2015 (extended: arXiv:1612.00637).
 *
 *   Why this is the one method that need NOT materialise the full N×N matrix
 *   (contrast k-medoids / MIP / LR-core, which read essentially every entry):
 *   the local density uses the CUTOFF kernel ρ_i = |{ j≠i : d(i,j) < dc }| — a
 *   BINARY per-pair test. For that test a lower bound LB and upper bound UB give
 *   an admissible decision without the exact DTW:
 *       LB(i,j) ≥ dc  ⇒  d(i,j) ≥ dc  ⇒  NOT a neighbour   (skip DTW)
 *       UB(i,j) <  dc  ⇒  d(i,j) <  dc ⇒  IS  a neighbour   (skip DTW)
 *       otherwise                       compute exact DTW.
 *   With the Rodriguez-Laio dc (~1–2% average neighbours) the vast majority of
 *   pairs are far (LB ≥ dc) and prune. The Gaussian kernel Σ exp(-(d/dc)²) is
 *   deliberately NOT used: it needs every exact distance, so it cannot prune.
 *
 *   LB = symmetric LB_Keogh cascade (reuses each series' envelope across ALL
 *   pairs — Begum's "reuse cached envelope"); tighter cascade bounds (LB_Webb /
 *   LB_Enhanced, Task 5.2) prune STRICTLY MORE here (the opposite of their
 *   exact-matrix pessimisation, since more LB ≥ dc decisions fire).
 *   UB = the no-warp diagonal cost Σ_t metric(x_t, y_t) for equal-length series
 *   (a valid DTW upper bound: the diagonal always satisfies the Sakoe-Chiba band).
 *
 *   Pruning is valid only for plain L1/SquaredL2 univariate DTW; for any other
 *   variant, or unequal-length pairs, the exact DTW is computed (result stays
 *   correct, pruning simply degrades). The final clustering labels are PROVABLY
 *   IDENTICAL to the brute-force (all-exact) density-peaks result — the `prune`
 *   flag toggles only whether a DTW is skipped, never the decision it feeds.
 *
 * @author Volkan Kumtepeli
 * @date 8 Jul 2026
 */

#pragma once

#include "../core/clustering_result.hpp"

#include <cstddef>

namespace dtwc {

class Problem; // Forward declaration

namespace algorithms {

/// Pruning ledger for a TADPole run — the numbers behind the ≥50% pruning band.
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
  std::size_t pruned_by_lb = 0;  ///< Density-stage pairs decided NOT-neighbour by LB ≥ dc (no DTW).
  std::size_t pruned_by_ub = 0;  ///< Density-stage pairs decided neighbour by UB < dc (no DTW).
  double dc = 0.0;               ///< Cutoff distance used.

  /// Fraction of the brute-force DTW work avoided. 0 on the brute path.
  double pruned_fraction() const
  {
    return total_pairs == 0 ? 0.0
                            : 1.0 - static_cast<double>(dtw_calls) / static_cast<double>(total_pairs);
  }
};

/**
 * @brief TADPole density-peaks clustering with admissible LB/UB DTW pruning.
 *
 * @param prob        Problem with data loaded. Does NOT call fill_distance_matrix();
 *                    the whole point is to avoid the full matrix.
 * @param n_clusters  Number of clusters k (top-k points by γ = ρ·δ become centers).
 * @param dc          Cutoff distance for the density kernel (must be > 0).
 * @param prune       true  → apply LB/UB pruning (fast path);
 *                    false → compute every DTW (independent brute-force oracle).
 *                    Labels are identical either way — that identity is the test.
 * @param stats       If non-null, receives the pruning ledger.
 * @return core::ClusteringResult: labels[i] ∈ [0,k), medoid_indices = the k
 *         density-peak centers, total_cost = Σ_i d(i, its center).
 * @throws std::runtime_error on n_clusters ∉ [1, N] or dc ≤ 0.
 *
 * @note Writes the result back into `prob` (clusters_ind / centroids_ind /
 *       n_clusters), matching fast_pam's Task-1.6 write-back contract.
 */
core::ClusteringResult tadpole(Problem& prob, int n_clusters, double dc,
                               bool prune = true, TADPoleStats* stats = nullptr);

/**
 * @brief Choose dc as a percentile of a deterministic exact-DTW subsample.
 *
 * @details Rodriguez & Laio pick dc so the average neighbour count is ≈1–2% of
 *   N. Estimating that needs the distance distribution; we sample it from the
 *   pairs among the first `min(N, cap)` series (a fixed, reproducible subset —
 *   no RNG), compute their exact DTW, and return the `percentile`-th percentile.
 *   These subsample DTWs are counted honestly if a caller tracks the budget.
 *
 * @param prob        Problem with data loaded.
 * @param percentile  Target percentile in (0, 100). Default 2.0.
 * @return dc > 0. Deterministic: identical inputs → identical dc.
 */
double tadpole_auto_dc(Problem& prob, double percentile = 2.0);

} // namespace algorithms
} // namespace dtwc
