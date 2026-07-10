/**
 * @file pruned_distance_matrix.cpp
 * @brief Implementation of pruned distance matrix construction.
 *
 * @details Fills a distance matrix using cascading lower bounds
 * (LB_Kim -> LB_Keogh) to guide early-abandon in DTW computations.
 * All pairs are computed exactly -- early-abandon makes individual
 * DTW computations terminate sooner when partial cost exceeds an
 * upper bound, saving 30-60% of inner-loop work for correlated data.
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 29 Mar 2026
 */

#include "pruned_distance_matrix.hpp"
#include "lower_bound_impl.hpp"
#include "../warping.hpp"
#include "../warping_adtw.hpp"
#include "../settings.hpp"
#include "../parallelisation.hpp"

#include <vector>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc::core {

// =========================================================================
//  C++17 lock-free atomic min for non-negative doubles.
//
//  Uses uint64_t CAS via compiler intrinsics. For non-negative IEEE-754
//  doubles, the bit representation preserves ordering: if a < b (and both
//  >= 0), then memcpy-to-uint64_t(a) < memcpy-to-uint64_t(b). This means
//  we can do atomic CAS on the uint64_t bits directly.
//
//  Relaxed memory order is sufficient: stale reads only reduce pruning
//  effectiveness, not correctness.
// =========================================================================

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static inline void atomic_min_double(double *addr, double val)
{
#ifdef _OPENMP
  static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64 bits");

  // Reinterpret as uint64_t pointer for atomic CAS.
  // This is technically UB per strict aliasing, but every major compiler
  // (GCC/Clang/MSVC/ICC) supports it, and the alternative (memcpy + CAS)
  // generates identical code.
  volatile uint64_t *iaddr = reinterpret_cast<volatile uint64_t *>(addr);

  uint64_t new_bits;
  std::memcpy(&new_bits, &val, sizeof(double));

  for (;;) {
    // Load current value
    uint64_t old_bits;
#if defined(_MSC_VER)
    old_bits = *iaddr;  // volatile read is sufficient on x86
#elif defined(__GNUC__) || defined(__clang__)
    old_bits = __atomic_load_n(reinterpret_cast<uint64_t *>(const_cast<uint64_t *>(iaddr)),
                               __ATOMIC_RELAXED);
#else
    old_bits = *iaddr;
#endif

    double old_val;
    std::memcpy(&old_val, &old_bits, sizeof(double));

    // If current value is already <= val, nothing to do
    if (old_val <= val) return;

    // Try to swap in the new (smaller) value
#if defined(_MSC_VER)
    uint64_t prev = static_cast<uint64_t>(
      _InterlockedCompareExchange64(
        reinterpret_cast<volatile long long *>(iaddr),
        static_cast<long long>(new_bits),
        static_cast<long long>(old_bits)));
    if (prev == old_bits) return;  // Success
#elif defined(__GNUC__) || defined(__clang__)
    uint64_t expected = old_bits;
    if (__atomic_compare_exchange_n(
          reinterpret_cast<uint64_t *>(const_cast<uint64_t *>(iaddr)),
          &expected, new_bits, /*weak=*/true,
          __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      return;  // Success
#else
    // Fallback: omp critical (should not be reached on major compilers)
    #pragma omp critical(nn_dist_update)
    {
      if (val < *addr) *addr = val;
    }
    return;
#endif
    // CAS failed (another thread updated concurrently) — retry
  }
#else
  // Serial fallback — no synchronisation needed.
  if (val < *addr) *addr = val;
#endif
}

// =========================================================================
//  Problem-based version (for C++ clustering) — PARALLEL
// =========================================================================

PruningStats fill_distance_matrix_pruned(
    dtwc::Problem &prob, int band, dtwc::LowerBoundStrategy lb_strat)
{
  // Pruned fill only operates on DenseDistanceMatrix (resize required).
  auto &dm = prob.dense_distance_matrix();

  PruningStats stats;
  const int N = static_cast<int>(prob.size());
  const bool is_adtw = (prob.variant_params.variant == dtwc::core::DTWVariant::ADTW);
  const double adtw_penalty = prob.variant_params.adtw_penalty;
  if (N <= 1) {
    if (N == 1) {
      dm.resize(1);
      dm.set(0, 0, 0.0);
    }
    return stats;
  }

  // Ensure matrix is sized
  dm.resize(static_cast<size_t>(N));

  // Resolve the effective lower-bound strategy. Auto keeps the historical
  // Kim+Keogh cascade; None disables both (equivalent to brute-force with
  // the pruned framework's bookkeeping); Kim/Keogh/KimKeogh flip individual
  // bounds on or off.
  bool use_lb_kim_flag = false;
  bool use_lb_keogh_flag = false;
  bool use_lb_enhanced_flag = false;
  bool use_lb_webb_flag = false;
  switch (lb_strat) {
    case dtwc::LowerBoundStrategy::None:
      break;
    case dtwc::LowerBoundStrategy::Kim:
      use_lb_kim_flag = true;
      break;
    case dtwc::LowerBoundStrategy::Keogh:
      use_lb_keogh_flag = true;
      break;
    case dtwc::LowerBoundStrategy::Enhanced:
      use_lb_kim_flag = true;
      use_lb_enhanced_flag = true;
      break;
    case dtwc::LowerBoundStrategy::Webb:
      use_lb_kim_flag = true;
      use_lb_webb_flag = true;
      break;
    case dtwc::LowerBoundStrategy::KimKeogh:
    case dtwc::LowerBoundStrategy::Auto:
    default:
      use_lb_kim_flag = true;
      use_lb_keogh_flag = true;
      break;
  }

  // Step 1: Precompute summaries for LB_Kim (O(N * n)) — parallel
  // Lock-free by design: each iteration writes only to summaries[i] at its own index.
  std::vector<SeriesSummary> summaries(N);
  if (use_lb_kim_flag) {
    auto compute_summary_at = [&](size_t i) {
      summaries[i] = compute_summary(prob.series(i));
    };
    run_openmp(compute_summary_at, static_cast<size_t>(N));
  }

  // Step 2: Precompute envelopes for LB_Keogh / LB_Enhanced (band >= 0) — parallel.
  // Lock-free by design: each iteration writes only to its own index.
  // LB_Enhanced reuses the same (upper,lower) Envelope as LB_Keogh; LB_Webb needs
  // the extended envelope set (adds the secondary L(U), U(L) arrays).
  const bool use_lb_keogh = use_lb_keogh_flag && (band >= 0);
  const bool use_lb_enhanced = use_lb_enhanced_flag && (band >= 0);
  const bool use_lb_webb = use_lb_webb_flag && (band >= 0);
  std::vector<Envelope> envelopes(N);
  if (use_lb_keogh || use_lb_enhanced) {
    auto compute_envelope_at = [&](size_t i) {
      envelopes[i] = compute_envelope(prob.series(i), band);
    };
    run_openmp(compute_envelope_at, static_cast<size_t>(N));
  }
  std::vector<WebbEnvelope> webb_envs(N);
  if (use_lb_webb) {
    auto compute_webb_envelope_at = [&](size_t i) {
      webb_envs[i] = compute_webb_envelope(prob.series(i), band);
    };
    run_openmp(compute_webb_envelope_at, static_cast<size_t>(N));
  }

  // Step 3: Per-row nearest-neighbor tracking (shared, updated atomically)
  constexpr double inf = std::numeric_limits<double>::max();
  std::vector<double> nn_dist(N, inf);

  // Step 4: Set diagonal to 0
  for (int i = 0; i < N; ++i)
    dm.set(static_cast<size_t>(i), static_cast<size_t>(i), 0.0);

  // Step 5: Compute total number of upper-triangle pairs
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  stats.total_pairs = static_cast<size_t>(num_pairs);

  // Step 6: Parallel loop over all upper-triangle pairs.
  // Each pair (i, j) is decoded from a linear index k.
  // nn_dist is shared: reads may be stale (relaxed consistency) but
  // this only reduces pruning effectiveness, not correctness —
  // every pair still gets the exact DTW distance.

  // Use contiguous pair-index blocks so each worker accumulates statistics and
  // reuses its Webb scratch without a shared critical section.  Blocks preserve
  // canonical pair order: run_openmp selects the lowest failing block, and the
  // loop below selects the first failing pair within that block.
  const size_t pair_count = static_cast<size_t>(num_pairs);
  const size_t worker_count = static_cast<size_t>(std::max(1, get_max_threads()));
  const size_t block_count = std::min(pair_count, worker_count * size_t{8});
  const size_t pairs_per_block = pair_count / block_count;
  const size_t larger_block_count = pair_count % block_count;

  std::atomic<size_t> global_pruned_kim{0};
  std::atomic<size_t> global_pruned_keogh{0};
  std::atomic<size_t> global_early_abandoned{0};
  std::atomic<size_t> global_full_dtw{0};

  auto compute_pair_block = [&](size_t block_index) {
    size_t local_pruned_kim = 0;
    size_t local_pruned_keogh = 0;   // envelope bound (Keogh/Enhanced/Webb) fired
    size_t local_early_abandoned = 0;
    size_t local_full_dtw = 0;
    std::vector<char> webb_scratch;  // per-block scratch, reused across pairs

    const size_t pair_begin = block_index * pairs_per_block
                            + std::min(block_index, larger_block_count);
    const size_t pair_end = pair_begin + pairs_per_block
                          + (block_index < larger_block_count ? 1 : 0);

    for (size_t k = pair_begin; k < pair_end; ++k) {
      // Decode linear pair index k -> (i, j) in the upper triangle.
      // Row i: using the quadratic formula on k = i*N - i*(i+1)/2 + (j - i - 1)
      const double Nd = static_cast<double>(N);
      const double kd = static_cast<double>(k);
      int i = static_cast<int>(Nd - 0.5 - std::sqrt((Nd - 0.5) * (Nd - 0.5) - 2.0 * kd));
      // Correct for floating-point imprecision
      int64_t row_start = static_cast<int64_t>(i) * N - static_cast<int64_t>(i) * (i + 1) / 2;
      if (static_cast<int64_t>(k) - row_start >= static_cast<int64_t>(N - i - 1)) {
        ++i;
        row_start = static_cast<int64_t>(i) * N - static_cast<int64_t>(i) * (i + 1) / 2;
      }
      int j = static_cast<int>(static_cast<int64_t>(k) - row_start) + i + 1;

      // Compute lower bound (cascading: LB_Kim, then LB_Keogh).
      // If Kim is disabled, start at 0 (no-op threshold); Keogh may still fire.
      double lb = use_lb_kim_flag ? lb_kim(summaries[i], summaries[j]) : 0.0;

      // Envelope-based bounds (Keogh / Enhanced / Webb) all require equal lengths.
      // Take the max: each is a valid lower bound, the tightest is best. Only one
      // of keogh/enhanced/webb is active per strategy, but the code is uniform.
      bool lb_keogh_used = false;
      const bool equal_len = prob.series(i).size() == prob.series(j).size();
      if (use_lb_keogh && equal_len) {
        const double lb_k = lb_keogh_symmetric(
          prob.series(i), envelopes[i],
          prob.series(j), envelopes[j]);
        if (lb_k > lb) { lb = lb_k; lb_keogh_used = true; }
      }
      if (use_lb_enhanced && equal_len) {
        const double lb_e = lb_enhanced_symmetric(
          prob.series(i), envelopes[i],
          prob.series(j), envelopes[j], band);
        if (lb_e > lb) { lb = lb_e; lb_keogh_used = true; }
      }
      if (use_lb_webb && equal_len) {
        const double lb_w = lb_webb_symmetric(
          prob.series(i), webb_envs[i],
          prob.series(j), webb_envs[j], band, &webb_scratch);
        if (lb_w > lb) { lb = lb_w; lb_keogh_used = true; }
      }

      // Early-abandon threshold: smallest NN distance for either endpoint.
      // Reads may be stale from other threads — this is benign.
      const double threshold = std::min(nn_dist[i], nn_dist[j]);

      // Helper lambdas to dispatch Standard vs ADTW, with or without early abandon.
      auto dtw_with_abandon = [&](double abandon) -> double {
        if (is_adtw)
          return (band >= 0)
            ? dtwc::adtwBanded<double>(prob.series(i), prob.series(j), band, adtw_penalty, abandon)
            : dtwc::adtwFull_L<double>(prob.series(i), prob.series(j), adtw_penalty, abandon);
        return (band >= 0)
          ? dtwc::dtwBanded<double>(prob.series(i), prob.series(j), band, abandon)
          : dtwc::dtwFull_L<double>(prob.series(i), prob.series(j), abandon);
      };

      double dist;
      if (lb > threshold && threshold < inf) {
        // LB exceeds NN threshold -- try early-abandon DTW
        if (lb_keogh_used)
          local_pruned_keogh++;
        else
          local_pruned_kim++;

        dist = dtw_with_abandon(threshold);

        if (dist >= inf * 0.5) {
          // Early abandon triggered -- recompute for exact distance
          local_early_abandoned++;
          dist = dtw_with_abandon(-1.0);
        }
      } else {
        // Pair may be close -- compute without early abandon
        local_full_dtw++;
        dist = dtw_with_abandon(-1.0);
      }

      // Lock-free by design: DenseDistanceMatrix::set() writes to two independent
      // memory locations: data_[i*N+j] and data_[j*N+i]. The pair-based
      // decomposition guarantees no two threads write the same (i,j) pair,
      // so this is safe without locks or atomics.
      dm.set(static_cast<size_t>(i), static_cast<size_t>(j), dist);

      // Update nearest-neighbor tracking (atomic min).
      atomic_min_double(&nn_dist[i], dist);
      atomic_min_double(&nn_dist[j], dist);
    }

    global_pruned_kim.fetch_add(local_pruned_kim, std::memory_order_relaxed);
    global_pruned_keogh.fetch_add(local_pruned_keogh, std::memory_order_relaxed);
    global_early_abandoned.fetch_add(local_early_abandoned, std::memory_order_relaxed);
    global_full_dtw.fetch_add(local_full_dtw, std::memory_order_relaxed);
  };
  run_openmp(compute_pair_block, block_count, true, 8);

  stats.pruned_by_lb_kim = global_pruned_kim.load(std::memory_order_relaxed);
  stats.pruned_by_lb_keogh = global_pruned_keogh.load(std::memory_order_relaxed);
  stats.early_abandoned = global_early_abandoned.load(std::memory_order_relaxed);
  stats.computed_full_dtw = global_full_dtw.load(std::memory_order_relaxed);

  return stats;
}

// =========================================================================
//  Standalone version (for Python binding)
// =========================================================================

PruningStats compute_distance_matrix_pruned(
  const std::vector<std::vector<double>> &series,
  double *output,
  int band,
  MetricType metric)
{
  PruningStats stats;
  const size_t N = series.size();
  if (N <= 1) {
    for (size_t i = 0; i < N; ++i)
      output[i * N + i] = 0.0;
    return stats;
  }

  // Zero-initialize output
  for (size_t i = 0; i < N * N; ++i)
    output[i] = 0.0;

  // LB pruning only valid for L1 (and L2 which is equivalent for scalars)
  const bool use_lb = (metric == MetricType::L1 || metric == MetricType::L2);

  // Step 1: Precompute summaries for LB_Kim
  std::vector<SeriesSummary> summaries;
  if (use_lb) {
    summaries.resize(N);
    for (size_t i = 0; i < N; ++i)
      summaries[i] = compute_summary(series[i]);
  }

  // Step 2: Precompute envelopes for LB_Keogh (only if band >= 0)
  const bool use_lb_keogh = use_lb && (band >= 0);
  std::vector<Envelope> envelopes;
  if (use_lb_keogh) {
    envelopes.resize(N);
    for (size_t i = 0; i < N; ++i)
      envelopes[i] = compute_envelope(series[i], band);
  }

  // Step 3: Per-row nearest-neighbor tracking
  constexpr double inf = std::numeric_limits<double>::max();
  std::vector<double> nn_dist(N, inf);

  // Step 4: Compute all upper-triangle pairs with OpenMP parallelism.
  // Each thread gets contiguous rows. nn_dist reads may be stale across
  // threads (relaxed consistency) but this only reduces pruning effectiveness,
  // not correctness -- every pair still gets the exact distance.
  std::vector<PruningStats> row_stats(N);
  auto compute_row = [&](size_t i) {
    auto &local = row_stats[i];

    // Thread-local stats
    for (size_t j = i + 1; j < N; ++j) {
      local.total_pairs++;

      double lb = 0.0;
      bool lb_keogh_used = false;

      if (use_lb) {
        // LB_Kim: O(1)
        lb = lb_kim(summaries[i], summaries[j]);

        // LB_Keogh: O(n), only for same-length series with band constraint
        if (use_lb_keogh && series[i].size() == series[j].size()) {
          const double lb_k = lb_keogh_symmetric(
            series[i], envelopes[i],
            series[j], envelopes[j]);
          if (lb_k > lb) {
            lb = lb_k;
            lb_keogh_used = true;
          }
        }
      }

      // nn_dist[i] and nn_dist[j] are both updated atomically (CAS) after each
      // pair (see lines 446-451), so either may be concurrently written by other
      // threads while read here. This is benign: a stale value only reduces
      // pruning effectiveness, never correctness (nn_dist feeds an early-abandon
      // threshold only).
      const double threshold = std::min(nn_dist[i], nn_dist[j]);

      double dist;
      if (use_lb && lb > threshold && threshold < inf) {
        // LB exceeds NN threshold -- try early-abandon DTW
        if (lb_keogh_used)
          local.pruned_by_lb_keogh++;
        else
          local.pruned_by_lb_kim++;

        dist = (band >= 0)
          ? dtwc::dtwBanded<double>(series[i], series[j], band, threshold, metric)
          : dtwc::dtwFull_L<double>(series[i], series[j], threshold, metric);

        if (dist >= inf * 0.5) {
          // Early abandon triggered -- recompute for exact distance
          local.early_abandoned++;
          dist = (band >= 0)
            ? dtwc::dtwBanded<double>(series[i], series[j], band, -1.0, metric)
            : dtwc::dtwFull_L<double>(series[i], series[j], -1.0, metric);
        }
      } else {
        // Compute without early abandon
        local.computed_full_dtw++;
        dist = (band >= 0)
          ? dtwc::dtwBanded<double>(series[i], series[j], band, -1.0, metric)
          : dtwc::dtwFull_L<double>(series[i], series[j], -1.0, metric);
      }

      // Store symmetrically
      output[i * N + j] = dist;
      output[j * N + i] = dist;

      // Update nearest-neighbor distances for pruning.
      // Use atomic min for both endpoints — matches the Problem-based version
      // (lines 256-257). The previous design only updated nn_dist[i], reducing
      // pruning effectiveness for later pairs involving series j.
      atomic_min_double(&nn_dist[i], dist);
      atomic_min_double(&nn_dist[j], dist);
    }
  };
  run_openmp(compute_row, N, true, 8);

  // Deterministic serial reduction; each parallel row owned exactly one slot.
  for (const auto &local : row_stats) {
    stats.total_pairs += local.total_pairs;
    stats.pruned_by_lb_kim += local.pruned_by_lb_kim;
    stats.pruned_by_lb_keogh += local.pruned_by_lb_keogh;
    stats.early_abandoned += local.early_abandoned;
    stats.computed_full_dtw += local.computed_full_dtw;
  }

  return stats;
}

} // namespace dtwc::core
