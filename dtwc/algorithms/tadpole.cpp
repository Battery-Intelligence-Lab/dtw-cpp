/**
 * @file tadpole.cpp
 * @brief TADPole density-peaks clustering with conditionally admissible DTW pruning.
 *
 * @details Implements dtwc::algorithms::tadpole (see tadpole.hpp). Algorithm and
 *   pruning follow Begum, Ulanova, Dau, Wang & Keogh, arXiv:1612.00637 (extended
 *   TADPole / KDD 2015), density-peaks core from Rodriguez & Laio, *Science* 2014.
 *
 *   ── Pruning admissibility inside the supported finite/nonempty domain ──
 *   For finite, nonempty, equal-length Standard-L1 univariate series whose
 *   length fits the integer band API, the following decisions preserve the
 *   brute-force result.
 *   ρ_i uses the CUTOFF kernel ρ_i = |{ j≠i : d(i,j) < dc }| — a binary "d<dc"
 *   test per pair. With LB ≤ d ≤ UB:
 *     • LB ≥ dc  ⇒ d ≥ dc ⇒ NOT (d<dc)              → not a neighbour, skip DTW
 *     • UB < dc  ⇒ d < dc                            → a neighbour,     skip DTW
 *     • else                                          compute the exact DTW
 *   `≥ dc` (not the paper's `> dc`) is used for the LB case: it is still admissible
 *   (d ≥ LB ≥ dc) and agrees with the brute path's strict `d < dc`. The δ step
 *   prunes a candidate q when LB(i,q) ≥ best-so-far (Begum Table 7): d(i,q) ≥ LB ≥
 *   best ⇒ q cannot lower the running minimum, so skipping it cannot change δ_i or
 *   the parent. In exact arithmetic those decisions preserve the brute-force
 *   result; the permanent exactly representable regression confirms that
 *   regime. Bit-level identity when a floating reduction lands near `dc` or
 *   `best` remains D17. Empty series are outside the guarantee (F48), and the
 *   configuration predicate does not yet validate finiteness or integer length
 *   representability (F46).
 *
 * @author Volkan Kumtepeli
 * @date 8 Jul 2026
 */

#include "tadpole.hpp"
#include "../Problem.hpp"
#include "../core/lower_bound_impl.hpp" // Envelope, compute_envelope, lb_keogh_symmetric
#include "../core/distance_matrix.hpp"  // tri_index, packed_size
#include "../core/dtw_options.hpp"      // DTWVariant, MissingStrategy

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc {
namespace algorithms {

namespace {

/// Configuration-only predicate for the regime where LB_Keogh + the diagonal L1
/// upper bound can be valid: plain Standard DTW, univariate, float64, no NaN
/// handling — the case where dispatch binds L1 `dtwBanded` (dtw_dispatch.cpp
/// make_standard). It does not inspect samples for finiteness or check that a
/// series length fits the integer envelope-radius API (F46).
///
/// Float32 is excluded because the prune would not be ADMISSIBLE: the bound path
/// reads Data::series() (float64 storage) while the exact side goes through
/// Problem::dist_by_ind, which branches on is_f32() — the two sides of the bound
/// would come from different data. Float32 therefore takes the exact path
/// everywhere, a slowdown TADPoleStats::pruning_enabled reports.
bool bounds_valid(const Problem &prob)
{
  return prob.variant_params.variant == core::DTWVariant::Standard
         && prob.data().ndim == 1
         && !prob.data().is_f32()
         && prob.missing_strategy == core::MissingStrategy::Error;
}

/// No-warp diagonal cost Σ_t |x_t − y_t| — a valid DTW upper bound for
/// equal-length series (the diagonal always satisfies the Sakoe-Chiba band). L1,
/// matching the Standard-DTW metric the dispatch uses.
double diagonal_ub_l1(std::span<const double> x, std::span<const double> y)
{
  double s = 0.0; // caller guarantees x.size() == y.size()
  for (std::size_t t = 0; t < x.size(); ++t) s += std::abs(x[t] - y[t]);
  return s;
}

/// Strict total order that makes "higher density" well-defined under ρ-ties:
/// a ranks above b iff ρ_a > ρ_b, or (ρ_a == ρ_b and a < b). Deterministic, so
/// pruned / brute / oracle agree on δ, parents, γ-ranking and labels.
inline bool higher_density(int a, int b, const std::vector<int> &rho)
{
  return rho[a] > rho[b] || (rho[a] == rho[b] && a < b);
}

} // anonymous namespace


double tadpole_auto_dc(Problem &prob, double percentile)
{
  const int N = static_cast<int>(prob.size());
  if (N < 2) return 1.0; // degenerate: any positive dc
  if (percentile <= 0.0 || percentile >= 100.0)
    throw std::runtime_error("tadpole_auto_dc: percentile must be in (0, 100).");

  // Deterministic subsample: all pairs among the first min(N, cap) series (no
  // RNG → reproducible). Rodriguez & Laio pick dc so avg neighbours ≈ 1–2% of N.
  const int cap = std::min(N, 64);
  std::vector<double> sample;
  sample.reserve(static_cast<std::size_t>(cap) * (cap - 1) / 2);
  for (int i = 0; i < cap; ++i)
    for (int j = i + 1; j < cap; ++j)
      sample.push_back(prob.dist_by_ind(i, j));

  std::size_t idx = static_cast<std::size_t>(percentile / 100.0 * (sample.size() - 1));
  if (idx >= sample.size()) idx = sample.size() - 1;
  std::nth_element(sample.begin(), sample.begin() + idx, sample.end());
  double dc = sample[idx];

  if (!(dc > 0.0)) { // percentile fell on zero (many identical series): use smallest positive
    double mn = std::numeric_limits<double>::max();
    for (double v : sample)
      if (v > 0.0 && v < mn) mn = v;
    dc = (mn == std::numeric_limits<double>::max()) ? 1.0 : mn;
  }
  return dc;
}


core::ClusteringResult tadpole(Problem &prob, int n_clusters, double dc, bool prune, TADPoleStats *stats)
{
  const int N = static_cast<int>(prob.size());
  const int k = n_clusters;
  if (N <= 0) throw std::runtime_error("tadpole: Problem has no data points.");
  if (k <= 0 || k > N)
    throw std::runtime_error("tadpole: n_clusters must be in [1, N]. Got k="
                             + std::to_string(k) + ", N=" + std::to_string(N) + ".");
  if (!(dc > 0.0)) throw std::runtime_error("tadpole: dc must be > 0.");

  const int band = prob.band;
  const bool can_prune = prune && bounds_valid(prob);

  const std::size_t M = core::packed_size(static_cast<std::size_t>(N)); // incl. diagonal slots
  std::vector<char> computed(M, 0); // first-touch flag per pair (dedup + exact-call count)

  // Exact DTW with dedup. computed[p] is written by exactly one thread per pair
  // (each unordered pair is visited by a single owner in every stage, and the
  // stages are barrier-separated), so no lock is needed. prob.dist_by_ind caches
  // internally, so a pair touched twice recomputes zero times.
  auto exact = [&](int i, int j) -> double {
    computed[core::tri_index(static_cast<std::size_t>(i), static_cast<std::size_t>(j))] = 1;
    return prob.dist_by_ind(i, j);
  };

  // Serial pre-trigger: allocate the lazy distance matrix and bind the DTW fn
  // ONCE, single-threaded, so the parallel regions never race the lazy-init /
  // rebind path inside dist_by_ind. Counted (conservative: if pruning would have
  // skipped this pair, it still shows as one real DTW — never over-claims pruning).
  if (N >= 2) exact(0, 1);

  // Per-series LB_Keogh envelopes, reused across ALL pairs (Begum's cached
  // envelope). For full DTW (band<0) the valid envelope is the GLOBAL min/max
  // (window ≥ n); compute_envelopes' band-0 default would equal the series and
  // over-estimate the LB (invalid), so pass the series length as the window then.
  std::vector<core::Envelope> envs;
  if (can_prune) {
    envs.resize(N);
    for (int i = 0; i < N; ++i) {
      auto s = prob.series(i);
      const int env_band = (band < 0) ? static_cast<int>(s.size()) : band;
      envs[i] = core::compute_envelope(s, env_band);
    }
  }

  // ── Stage 1 · local density ρ_i = |{ j≠i : d(i,j) < dc }| (cutoff kernel) ──
  std::vector<int> rho(N, 0);
  std::size_t plb = 0, pub = 0; // density-stage LB / UB pruning tallies

  #pragma omp parallel
  {
    std::vector<int> rho_local(N, 0);
    std::size_t loc_plb = 0, loc_pub = 0;

    // `can_prune` is loop-invariant, so it selects the whole i-body once rather
    // than being retested per pair, and keeps prob.series() — which throws under
    // Float32 — strictly inside the pruning branch.
    #pragma omp for schedule(dynamic, 8) nowait
    for (int i = 0; i < N; ++i) {
      if (can_prune) {
        const auto si = prob.series(i);
        for (int j = i + 1; j < N; ++j) {
          const auto sj = prob.series(j);
          bool neighbour;
          if (si.size() == sj.size()) {
            const double lb = core::lb_keogh_symmetric(si, envs[i], sj, envs[j]);
            if (lb >= dc) {                            // Case C: d ≥ LB ≥ dc ⇒ not a neighbour
              neighbour = false;
              ++loc_plb;
            } else {
              const double ub = diagonal_ub_l1(si, sj);
              if (ub < dc) {                           // Case B: d ≤ UB < dc ⇒ neighbour
                neighbour = true;
                ++loc_pub;
              } else {                                 // Case D: bounds straddle dc ⇒ exact
                neighbour = (exact(i, j) < dc);
              }
            }
          } else {
            neighbour = (exact(i, j) < dc);
          }
          if (neighbour) { ++rho_local[i]; ++rho_local[j]; }
        }
      } else {
        for (int j = i + 1; j < N; ++j)
          if (exact(i, j) < dc) { ++rho_local[i]; ++rho_local[j]; }
      }
    }
    #pragma omp critical(tadpole_density_reduce)
    {
      for (int i = 0; i < N; ++i) rho[i] += rho_local[i];
      plb += loc_plb;
      pub += loc_pub;
    }
  }

  // ── Stage 2 · separation δ_i = min distance to a higher-density point, and
  //    parent = that nearest higher-density neighbour (with LB pruning) ──
  constexpr double kInf = std::numeric_limits<double>::max();
  std::vector<double> delta(N, kInf);
  std::vector<int> parent(N, -1);

  #pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < N; ++i) {
    double best = kInf;
    int best_parent = -1;
    if (can_prune) { // prob.series() only on the pruning path — see above
      const auto si = prob.series(i);
      for (int q = 0; q < N; ++q) { // ascending index ⇒ ties resolve to the smallest index
        if (q == i || !higher_density(q, i, rho)) continue;
        const auto sq = prob.series(q);
        if (si.size() == sq.size()) {
          const double lb = core::lb_keogh_symmetric(si, envs[i], sq, envs[q]);
          if (lb >= best) continue; // d ≥ LB ≥ best ⇒ q cannot lower the min (nor tie-win)
        }
        const double d = exact(i, q);
        if (d < best) { best = d; best_parent = q; }
      }
    } else {
      for (int q = 0; q < N; ++q) {
        if (q == i || !higher_density(q, i, rho)) continue;
        const double d = exact(i, q);
        if (d < best) { best = d; best_parent = q; }
      }
    }
    delta[i] = best;
    parent[i] = best_parent;
  }

  // Global densest point (unique under the strict order: no higher-density
  // neighbour ⇒ parent == -1): δ = max of all other points' δ (Begum's
  // convention δ(sortIndex(1)) = max(δ(2:n))).
  int densest = -1;
  for (int i = 0; i < N; ++i)
    if (parent[i] < 0) { densest = i; break; }
  if (densest >= 0) {
    double mx = 0.0;
    for (int i = 0; i < N; ++i)
      if (i != densest && delta[i] < kInf) mx = std::max(mx, delta[i]);
    delta[densest] = mx;
  }

  // ── Cluster centers = top-k by γ_i = ρ_i · δ_i (descending, ties by index) ──
  std::vector<double> gamma(N);
  for (int i = 0; i < N; ++i) gamma[i] = static_cast<double>(rho[i]) * delta[i];

  std::vector<int> by_gamma(N);
  std::iota(by_gamma.begin(), by_gamma.end(), 0);
  std::sort(by_gamma.begin(), by_gamma.end(), [&](int a, int b) {
    return gamma[a] != gamma[b] ? gamma[a] > gamma[b] : a < b;
  });

  std::vector<char> is_center(N, 0);
  std::vector<int> label(N, -1);
  std::vector<int> center_of_label(k);
  for (int c = 0; c < k; ++c) {
    const int ci = by_gamma[c];
    is_center[ci] = 1;
    label[ci] = c;
    center_of_label[c] = ci;
  }

  // ── Assignment · descending density; each non-center inherits its parent's
  //    (higher-density, already-labelled) cluster (single non-iterative pass) ──
  std::vector<int> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return rho[a] != rho[b] ? rho[a] > rho[b] : a < b;
  });
  for (int r = 0; r < N; ++r) {
    const int p = order[r];
    if (is_center[p]) continue; // keeps its center label
    const int par = parent[p];
    // parent ranks strictly higher ⇒ already labelled; the densest point is always
    // a center (γ maximal), so a non-center always has a valid parent — guard anyway.
    label[p] = (par >= 0 && label[par] >= 0) ? label[par] : 0;
  }

  // Objective: Σ_i d(i, its cluster's center).
  double total_cost = 0.0;
  for (int i = 0; i < N; ++i) {
    const int c = center_of_label[label[i]];
    if (i != c) total_cost += exact(i, c);
  }

  if (stats) {
    std::size_t uniq = 0;
    for (char c : computed) uniq += (c != 0); // diagonal slots never set
    stats->total_pairs = static_cast<std::size_t>(N) * (N - 1) / 2;
    stats->dtw_calls = uniq;
    stats->pruned_by_lb = plb;
    stats->pruned_by_ub = pub;
    stats->dc = dc;
    stats->pruning_enabled = can_prune;
  }

  core::ClusteringResult result;
  result.labels = label;
  result.medoid_indices = center_of_label;
  result.total_cost = total_cost;
  result.iterations = 1; // single non-iterative pass
  result.converged = true;

  // Write-back (Task 1.6 contract): pure-C++ callers get the same state the
  // bindings wire by hand, so scores::silhouette(prob) etc. work with no wiring.
  prob.set_n_clusters(k);
  prob.centroids_ind = result.medoid_indices;
  prob.clusters_ind = result.labels;
  return result;
}

} // namespace algorithms
} // namespace dtwc
