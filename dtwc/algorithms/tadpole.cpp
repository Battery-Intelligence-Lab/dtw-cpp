/**
 * @file tadpole.cpp
 * @brief TADPole density-peaks clustering with conditionally admissible DTW pruning.
 *
 * @details Implements dtwc::algorithms::tadpole (see tadpole.hpp). Algorithm and
 *   pruning follow Begum, Ulanova, Dau, Wang & Keogh, arXiv:1612.00637 (KDD 2015),
 *   with the density-peaks core from Rodriguez & Laio, *Science* 2014.
 *
 *   Stage 1 uses `LB ≥ dc` (rather than the paper's `> dc`) because the cutoff
 *   kernel tests strict `d < dc`, so `d ≥ LB ≥ dc` already rules out a neighbour.
 *   Stage 2 prunes candidate q when `LB(i, q) ≥ best` (or `> best` when `q` would
 *   win a tie against `best_parent`; Begum Table 7).
 *
 * @author Volkan Kumtepeli
 * @date 8 Jul 2026
 */

#include "tadpole.hpp"
#include "../Problem.hpp"
#include "../base/env.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"
#include "../core/lower_bound_impl.hpp" // Envelope, compute_envelope, lb_keogh_symmetric
#include "../core/distance_matrix.hpp"  // refuse_float32_range
#include "../core/dtw_options.hpp"      // DTWVariant, MissingStrategy

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dtwc {
namespace algorithms {

  namespace detail {

    /// True when the L1 bounds (LB_Keogh and the diagonal UB; under another metric the
    /// LB can exceed the exact distance) are admissible against `exact_distance`:
    /// Standard L1 univariate float64 DTW with `MissingStrategy::Error` into a float64
    /// matrix. Float32 series/matrices are excluded so float64 bounds never prune
    /// float32-rounded distances. A pre-filled matrix is already O(1) lookup (and may
    /// hold user-supplied non-DTW distances), so `tadpole` prunes only into an unfilled one.
    bool bounds_valid(core::DTWVariant variant, core::MetricType metric, std::size_t ndim, core::Precision series,
                      core::Precision matrix, core::MissingStrategy missing)
    {
      return variant == core::DTWVariant::Standard
             && metric == core::MetricType::L1
             && ndim == 1
             && series != core::Precision::Float32
             && matrix != core::Precision::Float32
             && missing == core::MissingStrategy::Error;
    }

  } // namespace detail

  namespace {

    /// No-warp diagonal cost Σ_t |x_t − y_t| — a valid DTW upper bound for
    /// equal-length series (the diagonal always satisfies the Sakoe-Chiba band). L1,
    /// matching the Standard-DTW metric the dispatch uses.
    double diagonal_ub_l1(std::span<const double> x, std::span<const double> y)
    {
      double s = 0.0; // caller guarantees x.size() == y.size()
      const double *const xp = x.data();
      const double *const yp = y.data();
      const std::size_t n = x.size();
#if !defined(_MSC_VER)
#pragma omp simd reduction(+ : s)
#endif
      for (std::size_t t = 0; t < n; ++t) s += std::abs(xp[t] - yp[t]);
      return s;
    }

    /// O(1) LB_Kim lower bound for equal-length L1 DTW: endpoints must align.
    inline double lb_kim_l1(std::span<const double> x, std::span<const double> y) noexcept
    {
      const std::size_t len = x.size();
      if (len == 0) return 0.0;
      if (len == 1) return std::abs(x[0] - y[0]);
      return std::abs(x[0] - y[0]) + std::abs(x[len - 1] - y[len - 1]);
    }

    /// The exact distance of a pair: the Problem's matrix when it is filled (a
    /// matrix read from a file included), else Problem::pair_distance on the stored
    /// series, whose request the accessor of their precision checks here, once,
    /// serially. Neither writes, so the parallel stages may call it for disjoint pairs.
    auto exact_distance(Problem &prob)
    {
      const bool filled = prob.is_distance_matrix_filled();
      if (!filled && prob.data().is_f32())
        (void)prob.dtw_function_f32();
      else if (!filled)
        (void)prob.dtw_function();
      return [&prob, filled](index_t i, index_t j) -> double {
        if (filled) return prob.dist_by_ind(i, j);
        return prob.pair_distance(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      };
    }

    /// Strict total order that makes "higher density" well-defined under ρ-ties:
    /// a ranks above b iff ρ_a > ρ_b, or (ρ_a == ρ_b and a < b). Deterministic, so
    /// pruned / brute / oracle agree on δ, parents, γ-ranking and labels.
    inline bool higher_density(index_t a, index_t b, const std::vector<index_t> &rho)
    {
      return rho[a] > rho[b] || (rho[a] == rho[b] && a < b);
    }

    /// A thread's failure in a parallel stage, as run_openmp keeps it (parallelisation.hpp, which says why): an exception
    /// may not leave an OpenMP region, so each thread keeps the failure of the lowest index it saw fail and skips only the
    /// indices above it; `failed_at` (the stage's end: none) is what the loop tests.
    struct Failure
    {
      std::exception_ptr failure;
      index_t failed_at;
    };

    /// After a stage's join: the failure with the lowest index over every thread, rethrown unchanged, the one a serial
    /// run raises.
    void rethrow_lowest(const std::vector<Failure> &failures, index_t end)
    {
      const auto lowest = std::min_element(failures.begin(), failures.end(), [](const Failure &a, const Failure &b) { return a.failed_at < b.failed_at; });
      if (lowest->failed_at < end) std::rethrow_exception(lowest->failure);
    }

  } // anonymous namespace


  double tadpole_auto_dc(Problem &prob, double percentile)
  {
    const index_t N = prob.size();
    if (N < 2) return 1.0; // degenerate: any positive dc
    if (percentile <= 0.0 || percentile >= 100.0)
      throw InvalidInput("tadpole_auto_dc: percentile must be in (0, 100).");

    // Deterministic evenly spaced subsample of min(N, 64) series across [0, N): every
    // series when N <= 64 (s * N / N = s), and for larger N the whole dataset rather
    // than its first 64 rows, which may belong to one sorted group.
    const index_t cap = std::min<index_t>(N, 64);
    std::vector<index_t> chosen(static_cast<std::size_t>(cap));
    for (index_t s = 0; s < cap; ++s)
      chosen[static_cast<std::size_t>(s)] = static_cast<index_t>(
        (static_cast<std::size_t>(s) * static_cast<std::size_t>(N)) / static_cast<std::size_t>(cap));
    const auto distance = exact_distance(prob);
    std::vector<std::pair<index_t, index_t>> pairs;
    pairs.reserve(static_cast<std::size_t>(cap * (cap - 1) / 2));
    for (index_t a = 0; a < cap; ++a)
      for (index_t b = a + 1; b < cap; ++b)
        pairs.emplace_back(chosen[static_cast<std::size_t>(a)], chosen[static_cast<std::size_t>(b)]);
    // Up to 2016 exact DTWs, which tadpole() computes again (it keeps its own
    // pair cache): one slot per pair, in parallel.
    std::vector<double> sample(pairs.size());
    auto sample_pair = [&](std::size_t p) { sample[p] = distance(pairs[p].first, pairs[p].second); };
    run_openmp(sample_pair, pairs.size());

    for (double v : sample)
      if (!std::isfinite(v))
        throw InvalidInput("tadpole_auto_dc: distance function returned a non-finite value.");

    std::size_t idx = static_cast<std::size_t>(percentile / 100.0 * static_cast<double>(sample.size() - 1));
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


  core::ClusteringResult tadpole(Problem &prob, index_t n_clusters, double dc, bool prune, TADPoleStats *stats)
  {
    // A Problem on a GPU, or one with a forced share, asks for work this method has no part in (it computes its
    // distances on the CPU as it goes): refused here, as Problem::cluster refuses it, not run on the CPU without a word;
    // a Problem given `fallback` computes on the CPU, and says so. A forced share is refused first, whatever the
    // fallback; cpu+gpu's goes with its device, as no CPU run splits a share.
    if (prob.gpu_share() != -1.0 && prob.device().first != Device::CPU_GPU)
      throw InvalidInput(dtwc::detail::share_ignored_message(
        "tadpole", "the distance matrix's fill and FastCLARA's pass", prob.device().first));
    const dtwc::detail::DeviceScope device_scope(prob, "tadpole", Method::TADPole);
    const index_t N = prob.size();
    const index_t k = n_clusters;
    if (N <= 0) throw InvalidInput("tadpole: Problem has no data points.");
    if (k <= 0 || k > N)
      throw InvalidInput("tadpole: n_clusters must be in [1, N]. Got k="
                         + std::to_string(k) + ", N=" + std::to_string(N) + ".");
    if (!(dc > 0.0)) throw InvalidInput("tadpole: dc must be > 0.");

    const auto distance = exact_distance(prob);
    const int band = prob.distance().band;
    const bool filled = prob.is_distance_matrix_filled();
    const bool f32_matrix = !filled && prob.matrix_dtype() == core::Precision::Float32;
    const bool can_prune = prune && !filled
                           && detail::bounds_valid(prob.variant_params().variant, prob.metric(), prob.data().ndim,
                                                   prob.data().precision, prob.matrix_dtype(), prob.missing_strategy());

    // Sparse per-row exact-pair storage when the Problem's matrix is not pre-filled:
    // Stage 1 populates upper_cache[i] (j > i, sorted ascending) exclusively per row i;
    // after Stage 1, lower_cache[j] mirrors (i, d) with i < j, also sorted ascending.
    const auto points = static_cast<std::size_t>(N);
    std::vector<std::vector<std::pair<index_t, double>>> upper_cache(filled ? 0 : points);
    std::vector<std::vector<std::pair<index_t, double>>> lower_cache;
    constexpr std::size_t kMaxUbSeeds = 16;
    std::vector<std::vector<std::pair<index_t, double>>> upper_ub(can_prune ? points : 0);
    std::vector<std::vector<std::pair<index_t, double>>> lower_ub(can_prune ? points : 0);

    auto eval_pair = [&](index_t i, index_t j) -> double {
      double d = distance(i, j);
      if (f32_matrix) {
        const float f = static_cast<float>(d);
        if (!std::isfinite(f) && std::isfinite(d))
          core::refuse_float32_range("tadpole", static_cast<std::size_t>(std::min(i, j)),
                                     static_cast<std::size_t>(std::max(i, j)), d);
        d = static_cast<double>(f);
      }
      return d;
    };

    // Per-series LB_Keogh envelopes, reused across ALL pairs (Begum's cached
    // envelope). For full DTW (band<0) the valid envelope is the GLOBAL min/max
    // (window >= n), which the series length as the window gives (compute_envelopes
    // reads a negative band the same way).
    std::vector<core::Envelope> envs;
    if (can_prune) {
      envs.resize(points);
      auto build_env = [&](std::size_t idx_i) {
        auto s = prob.series(idx_i);
        const int env_band = (band < 0) ? static_cast<int>(s.size()) : band;
        envs[idx_i] = core::compute_envelope(s, env_band);
      };
      run_openmp(build_env, points, N >= 64);
    }

    // ── Stage 1 · local density ρ_i = |{ j≠i : d(i,j) < dc }| (cutoff kernel) ──
    const int n_threads = get_max_threads();
    const auto threads = static_cast<std::size_t>(n_threads);
    std::vector<index_t> rho_rows(threads * points, 0);
    std::vector<std::size_t> plb_by_thread(threads, 0), pub_by_thread(threads, 0);
    std::vector<Failure> failures(threads, Failure{ nullptr, N });

#pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
      const auto thread = static_cast<std::size_t>(omp_get_thread_num());
#else
      const std::size_t thread = 0;
#endif
      index_t *const rho_local = rho_rows.data() + thread * points;
      std::size_t loc_plb = 0, loc_pub = 0;
      Failure &slot = failures[thread];

#pragma omp for schedule(dynamic, 8) nowait
      for (index_t i = 0; i < N; ++i) {
        if (i > slot.failed_at) continue;
        try {
          if (can_prune) {
            const auto si = prob.series(static_cast<std::size_t>(i));
            auto &uc = upper_cache[static_cast<std::size_t>(i)];
            auto &uub = upper_ub[static_cast<std::size_t>(i)];
            for (index_t j = i + 1; j < N; ++j) {
              const auto sj = prob.series(static_cast<std::size_t>(j));
              bool neighbour;
              if (si.size() == sj.size()) {
                if (lb_kim_l1(si, sj) >= dc) {
                  neighbour = false;
                  ++loc_plb;
                } else if (core::lb_keogh_at_least(si, envs[i], sj, envs[j], dc)) { // Case C: d ≥ LB ≥ dc ⇒ not a neighbour
                  neighbour = false;
                  ++loc_plb;
                } else {
                  const double ub = diagonal_ub_l1(si, sj);
                  if (ub < dc) { // Case B: d ≤ UB < dc ⇒ neighbour
                    neighbour = true;
                    ++loc_pub;
                    if (uub.size() < kMaxUbSeeds) uub.emplace_back(j, ub);
                  } else { // Case D: bounds straddle dc ⇒ exact
                    const double d = eval_pair(i, j);
                    uc.emplace_back(j, d);
                    neighbour = (d < dc);
                  }
                }
              } else {
                const double d = eval_pair(i, j);
                uc.emplace_back(j, d);
                neighbour = (d < dc);
              }
              if (neighbour) {
                ++rho_local[i];
                ++rho_local[j];
              }
            }
          } else {
            for (index_t j = i + 1; j < N; ++j) {
              const double d = eval_pair(i, j);
              if (!filled) upper_cache[static_cast<std::size_t>(i)].emplace_back(j, d);
              if (d < dc) {
                ++rho_local[i];
                ++rho_local[j];
              }
            }
          }
        } catch (...) {
          slot.failure = std::current_exception();
          slot.failed_at = i;
        }
      }
      plb_by_thread[thread] = loc_plb;
      pub_by_thread[thread] = loc_pub;
    }
    rethrow_lowest(failures, N);

    std::vector<index_t> rho(N, 0);
    for (std::size_t t = 0; t < threads; ++t)
      for (std::size_t i = 0; i < points; ++i) rho[i] += rho_rows[t * points + i];
    const std::size_t plb = std::accumulate(plb_by_thread.begin(), plb_by_thread.end(), std::size_t{ 0 });
    const std::size_t pub = std::accumulate(pub_by_thread.begin(), pub_by_thread.end(), std::size_t{ 0 });

    std::size_t stage1_calls = 0;
    if (!filled) {
      lower_cache.resize(points);
      for (index_t i = 0; i < N; ++i) {
        const auto &uc = upper_cache[static_cast<std::size_t>(i)];
        stage1_calls += uc.size();
        for (const auto &[j, d] : uc)
          lower_cache[static_cast<std::size_t>(j)].emplace_back(i, d);
      }
      if (can_prune) {
        for (index_t i = 0; i < N; ++i) {
          for (const auto &[j, ub] : upper_ub[static_cast<std::size_t>(i)]) {
            auto &lub = lower_ub[static_cast<std::size_t>(j)];
            if (lub.size() < kMaxUbSeeds) lub.emplace_back(i, ub);
          }
        }
      }
    } else {
      stage1_calls = points * (points - 1) / 2;
    }

    // ── Stage 2 · separation δ_i = min distance to a higher-density point, and
    //    parent = that nearest higher-density neighbour (with LB pruning) ──
    constexpr double kInf = std::numeric_limits<double>::max();
    std::vector<double> delta(N, kInf);
    std::vector<index_t> parent(N, -1);
    std::vector<std::size_t> stage2_by_thread(threads, 0);

#pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
      const auto thread = static_cast<std::size_t>(omp_get_thread_num());
#else
      const std::size_t thread = 0;
#endif
      Failure &slot = failures[thread];
      std::size_t loc_stage2 = 0;

#pragma omp for schedule(dynamic, 8) nowait
      for (index_t i = 0; i < N; ++i) {
        if (i > slot.failed_at) continue;
        try {
          double best = kInf;
          index_t best_parent = -1;
          if (can_prune) { // prob.series() only on the pruning path — see above
            const auto si = prob.series(static_cast<std::size_t>(i));
            const auto &lc = lower_cache[static_cast<std::size_t>(i)];
            const auto &uc = upper_cache[static_cast<std::size_t>(i)];

            // Seed `best` from any higher-density pairs already evaluated in Stage 1.
            for (const auto &[q, d] : lc) {
              if (!higher_density(q, i, rho)) continue;
              if (d < best || (d == best && q < best_parent)) {
                best = d;
                best_parent = q;
              }
            }
            for (const auto &[q, d] : uc) {
              if (!higher_density(q, i, rho)) continue;
              if (d < best || (d == best && q < best_parent)) {
                best = d;
                best_parent = q;
              }
            }

            // Seed `best` from Stage 1 close neighbours (UB < dc) when helpful.
            index_t ub_seed_q = -1;
            double ub_seed_val = best;
            for (const auto &list : { &lower_ub[static_cast<std::size_t>(i)], &upper_ub[static_cast<std::size_t>(i)] }) {
              for (const auto &[q, ub] : *list) {
                if (!higher_density(q, i, rho)) continue;
                if (ub < ub_seed_val) {
                  ub_seed_val = ub;
                  ub_seed_q = q;
                }
              }
            }
            if (ub_seed_q >= 0) {
              const double d = eval_pair(i, ub_seed_q);
              ++loc_stage2;
              if (d < best || (d == best && ub_seed_q < best_parent)) {
                best = d;
                best_parent = ub_seed_q;
              }
            }

            std::size_t li = 0, ui = 0;
            for (index_t q = 0; q < N; ++q) { // ascending index ⇒ ties resolve to the smallest index
              while (li < lc.size() && lc[li].first < q) ++li;
              while (ui < uc.size() && uc[ui].first < q) ++ui;
              if (q == i || !higher_density(q, i, rho)) continue;
              if ((li < lc.size() && lc[li].first == q) || (ui < uc.size() && uc[ui].first == q)
                  || q == ub_seed_q)
                continue;
              const auto sq = prob.series(static_cast<std::size_t>(q));
              const bool wins_tie = (best_parent >= 0 && q < best_parent);
              if (si.size() == sq.size()) {
                const double kim = lb_kim_l1(si, sq);
                if (wins_tie ? (kim > best) : (kim >= best)) continue;
                const bool pruned = wins_tie ? core::lb_keogh_greater(si, envs[i], sq, envs[q], best)
                                             : core::lb_keogh_at_least(si, envs[i], sq, envs[q], best);
                if (pruned) continue;
              }
              const double d = eval_pair(i, q);
              ++loc_stage2;
              if (d < best || (wins_tie && d == best)) {
                best = d;
                best_parent = q;
              }
            }
          } else if (filled) {
            for (index_t q = 0; q < N; ++q) {
              if (q == i || !higher_density(q, i, rho)) continue;
              const double d = distance(i, q);
              if (d < best) {
                best = d;
                best_parent = q;
              }
            }
          } else {
            for (const auto &[q, d] : lower_cache[static_cast<std::size_t>(i)]) {
              if (!higher_density(q, i, rho)) continue;
              if (d < best || (d == best && q < best_parent)) {
                best = d;
                best_parent = q;
              }
            }
            for (const auto &[q, d] : upper_cache[static_cast<std::size_t>(i)]) {
              if (!higher_density(q, i, rho)) continue;
              if (d < best || (d == best && q < best_parent)) {
                best = d;
                best_parent = q;
              }
            }
          }
          delta[i] = best;
          parent[i] = best_parent;
        } catch (...) {
          slot.failure = std::current_exception();
          slot.failed_at = i;
        }
      }
      stage2_by_thread[thread] = loc_stage2;
    }
    rethrow_lowest(failures, N);
    const std::size_t stage2_calls = std::accumulate(stage2_by_thread.begin(), stage2_by_thread.end(), std::size_t{ 0 });

    // Global densest point (unique under the strict order: no higher-density
    // neighbour ⇒ parent == -1): δ = max of all other points' δ (Begum's
    // convention δ(sortIndex(1)) = max(δ(2:n))).
    index_t densest = -1;
    for (index_t i = 0; i < N; ++i)
      if (parent[i] < 0) {
        densest = i;
        break;
      }
    if (densest >= 0) {
      double mx = 0.0;
      for (index_t i = 0; i < N; ++i)
        if (i != densest && delta[i] < kInf) mx = std::max(mx, delta[i]);
      delta[densest] = mx;
    }

    // ── Cluster centers = top-k by γ_i = ρ_i · δ_i (descending, ties by index) ──
    std::vector<double> gamma(N);
    for (index_t i = 0; i < N; ++i) gamma[i] = static_cast<double>(rho[i]) * delta[i];

    std::vector<index_t> by_gamma(N);
    std::iota(by_gamma.begin(), by_gamma.end(), index_t{ 0 });
    std::sort(by_gamma.begin(), by_gamma.end(), [&](index_t a, index_t b) {
      return gamma[a] != gamma[b] ? gamma[a] > gamma[b] : a < b;
    });

    std::vector<char> is_center(N, 0);
    std::vector<index_t> label(N, -1);
    std::vector<index_t> center_of_label(k);
    for (index_t c = 0; c < k; ++c) {
      const index_t ci = by_gamma[c];
      is_center[ci] = 1;
      label[ci] = c;
      center_of_label[c] = ci;
    }

    // ── Assignment · descending density; each non-center inherits its parent's
    //    (higher-density, already-labelled) cluster (single non-iterative pass) ──
    std::vector<index_t> order(N);
    std::iota(order.begin(), order.end(), index_t{ 0 });
    std::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
      return rho[a] != rho[b] ? rho[a] > rho[b] : a < b;
    });
    for (index_t r = 0; r < N; ++r) {
      const index_t p = order[r];
      if (is_center[p]) continue; // keeps its center label
      const index_t par = parent[p];
      // parent ranks strictly higher ⇒ already labelled; the densest point is always
      // a center (γ maximal), so a non-center always has a valid parent — guard anyway.
      label[p] = (par >= 0 && label[par] >= 0) ? label[par] : 0;
    }

    // Objective: Σ_i d(i, its cluster's center) without polluting dtw_calls or caches.
    auto lookup_or_distance = [&](index_t i, index_t c) -> double {
      if (filled) return distance(i, c);
      const auto &row = (c > i) ? upper_cache[static_cast<std::size_t>(i)]
                                : lower_cache[static_cast<std::size_t>(i)];
      const auto it = std::lower_bound(row.begin(), row.end(), c,
                                       [](const std::pair<index_t, double> &entry, index_t target) {
                                         return entry.first < target;
                                       });
      if (it != row.end() && it->first == c) return it->second;
      return eval_pair(i, c);
    };

    double total_cost = 0.0;
    for (index_t i = 0; i < N; ++i) {
      const index_t c = center_of_label[label[i]];
      if (i != c) total_cost += lookup_or_distance(i, c);
    }

    if (stats) {
      stats->total_pairs = points * (points - 1) / 2;
      stats->dtw_calls = stage1_calls + stage2_calls;
      stats->pruned_by_lb = plb;
      stats->pruned_by_ub = pub;
      stats->dc = dc;
      stats->pruning_enabled = can_prune;
    }

    core::ClusteringResult result;
    result.labels = std::move(label);
    result.medoid_indices = std::move(center_of_label);
    result.total_cost = total_cost;
    result.iterations = 0; // non-iterative (matches MIP / LRCore in Result)
    result.converged = true;

    prob.set_result(result); // scores::silhouette(prob) etc. read it back
    return result;
  }

} // namespace algorithms
} // namespace dtwc
