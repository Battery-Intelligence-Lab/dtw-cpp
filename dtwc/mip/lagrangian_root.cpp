/**
 * @file lagrangian_root.cpp
 * @brief Implementation of the Lagrangian root bound and exact LR-core B&B solver.
 *
 * @details See lagrangian_root.hpp for the derivation. Supports both row-major
 * dense `const double *D` (public C++ API) and zero-copy packed/memory-mapped
 * `core::DistanceMatrix` (`LR_core_clustering`), avoiding an O(N^2) dense copy
 * at large N or on `.dtwm` memory-mapped runs.
 *
 * Algorithmic highlights:
 *   1. Erlenkotter (1978) 2nd-nearest dual warm start μ_j^(0) = d_(2)(j) (or
 *      d_(1)(j) when k == N), starting L(μ^(0)) near the LP optimum on iter 0;
 *   2. Cache-line-blocked 8-row evaluation (`core::for_each_point`) on packed /
 *      `.dtwm` distance matrices;
 *   3. O(Σ_c |C_c|^2) cluster-grouped Lloyd local search (`pmedian_local_search`)
 *      plus Volume-style exponential primal averages ȳ_i for fractional recovery;
 *   4. Zero-allocation DFS branch-and-bound with precomputed suffix minima,
 *      additive Erlenkotter client-floor tightening, and d_{F_1}(j) dual clamping.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include "lagrangian_root.hpp"

#include "mip.hpp" // LR_core_clustering
#include "nearest_medoid.hpp"

#include "../core/clustering_result.hpp"
#include "../core/distance_matrix.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"
#include "../Problem.hpp"
#include "../algorithms/fast_pam.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// evaluate_dual's first loop is `#pragma omp` behind `#ifdef _OPENMP`. This file compiled without the flag in a build that has OpenMP would
// run the Lagrangian phase serial with no sign of it; dtwc/CMakeLists.txt gives mip-solvers the flag together with this definition.
// (cppcheck, a maintainer option, takes the target's -D but none of its compiler flags and so never sees _OPENMP: it stands aside.)
#if defined(DTWC_HAS_OPENMP) && !defined(_OPENMP) && !defined(__CPPCHECK__)
#error "DTWC_HAS_OPENMP is defined but lagrangian_root.cpp is compiled without the OpenMP flag: the Lagrangian dual would run serial."
#endif

namespace dtwc::mip {

namespace {
  /// A sum's rounding, per term and relative to the size of its terms. Times both, it floors a tolerance that compares
  /// sums which cancel: the floor scales with the distances and stays above the rounding where the cost is zero.
  constexpr double kRounding = 64.0 * std::numeric_limits<double>::epsilon();

  /// OpenMP threshold for the O(N^2) rho loops: below it the fork and join (one per subgradient iteration, about 50 us
  /// with 18 threads on an Apple M5 Pro) cost more than the loop. Measured break-even there: between N = 240 and 280.
  [[maybe_unused]] constexpr index_t kParallelMinN = 280;

  /// Maximum `(C + 1) * N` doubles (~32 MiB) for precomputing B&B candidate suffix minima.
  constexpr std::size_t kMaxSuffixMinCells = 4194304;

  /// Subgradient tuning (safe defaults).
  struct LagrangianParams
  {
    int max_iters = 4000;       ///< Subgradient iteration cap.
    double rel_gap_tol = 1e-6;  ///< Certify optimal when UB − LB ≤ this × |UB|.
    double lambda0 = 1.0;       ///< Initial Polyak step scale λ ∈ (0, 2] (1.0 = damped, well inside the convergent range).
    int stall_halve = 20;       ///< Halve λ after this many iters with no LB improvement.
    double lambda_min = 1e-4;   ///< Floor for λ — CLAMPED here (never frozen), so diminishing steps keep converging.
    double deflect = 1.5;       ///< CFM subgradient deflection γ ∈ [0,2) — steers the step off the previous direction to kill zig-zag (0 = plain subgradient).
    int polish_period = 16;     ///< Run the O(Σ|C_c|²) medoid polish every this many iters (a cheap O(Nk) assignment repair still runs EVERY iter).
  };
  constexpr LagrangianParams params{};

  /// Optional external primal seed (e.g. FastPAM warm start from LR_core_clustering).
  struct SeedIncumbent
  {
    bool has_ub = false;
    double ub = std::numeric_limits<double>::infinity();
    const std::vector<index_t> *medoids = nullptr;
  };

  /// Row-major dense N×N matrix adapter (public C++ API).
  struct DenseMatrixView
  {
    const double *D;
    index_t N;
    std::size_t Nz;
    bool nonneg = false;

    DenseMatrixView(const double *d, index_t n) noexcept
      : D(d), N(n), Nz(static_cast<std::size_t>(n))
    {
      nonneg = (min_distance() >= 0.0);
    }

    double operator()(index_t i, index_t j) const noexcept
    {
      return D[static_cast<std::size_t>(i) * Nz + static_cast<std::size_t>(j)];
    }

    double min_distance() const noexcept
    {
      double mn = std::numeric_limits<double>::infinity();
      const std::size_t total = Nz * Nz;
      for (std::size_t t = 0; t < total; ++t) {
        const double v = D[t];
        if (v < mn) mn = v;
      }
      return mn;
    }

    void compute_initial_mu(index_t k, std::vector<double> &mu) const
    {
      mu.assign(Nz, 0.0);
#ifdef _OPENMP
      const int nthreads = (N >= kParallelMinN) ? get_max_threads() : 1;
      const index_t chunk = omp_chunk_size_for(N, 4, nthreads);
#pragma omp parallel for schedule(dynamic, chunk) num_threads(nthreads) if (N >= kParallelMinN)
#endif
      for (index_t j = 0; j < N; ++j) {
        const double *Dj = D + static_cast<std::size_t>(j) * Nz;
        double d1 = std::numeric_limits<double>::infinity();
        double d2 = std::numeric_limits<double>::infinity();
        for (index_t i = 0; i < N; ++i) {
          const double v = Dj[i];
          if (v < d1) {
            d2 = d1;
            d1 = v;
          } else if (v < d2) {
            d2 = v;
          }
        }
        mu[static_cast<std::size_t>(j)] = (k < N && N >= 2) ? d2 : d1;
      }
    }

    void compute_rho(const std::vector<double> &mu, std::vector<double> &rho) const
    {
#ifdef _OPENMP
      const int nthreads = (N >= kParallelMinN) ? get_max_threads() : 1;
      const index_t chunk = omp_chunk_size_for(N, 4, nthreads);
#pragma omp parallel for schedule(dynamic, chunk) num_threads(nthreads) if (N >= kParallelMinN)
#endif
      for (index_t i = 0; i < N; ++i) {
        const double *Di = D + static_cast<std::size_t>(i) * Nz;
        double s = 0.0;
        for (index_t j = 0; j < N; ++j) {
          const double dd = Di[j] - mu[static_cast<std::size_t>(j)];
          if (dd < 0.0) s += dd;
        }
        rho[static_cast<std::size_t>(i)] = s;
      }
    }
  };

  /// Packed lower-triangular DistanceMatrix adapter (float64 / float32 / .dtwm mapped).
  template <typename T>
  struct PackedMatrixView
  {
    const T *packed;
    index_t N;
    std::size_t Nz;
    static constexpr bool nonneg = true;

    PackedMatrixView(const T *p, index_t n) noexcept
      : packed(p), N(n), Nz(static_cast<std::size_t>(n)) {}

    double operator()(index_t i, index_t j) const noexcept
    {
      return static_cast<double>(packed[core::tri_index(static_cast<std::size_t>(i), static_cast<std::size_t>(j))]);
    }

    void compute_initial_mu(index_t k, std::vector<double> &mu) const
    {
      mu.assign(Nz, 0.0);
      constexpr std::size_t W = core::group_rows;
      const index_t n_groups = static_cast<index_t>((Nz + W - 1) / W);
#ifdef _OPENMP
      const int nthreads = (N >= kParallelMinN) ? get_max_threads() : 1;
      const index_t chunk = omp_chunk_size_for(n_groups, 4, nthreads);
#pragma omp parallel for schedule(dynamic, chunk) num_threads(nthreads) if (N >= kParallelMinN)
#endif
      for (index_t q = 0; q < n_groups; ++q) {
        const index_t a = q * static_cast<index_t>(W);
        double d1[W];
        double d2[W];
        for (std::size_t g = 0; g < W; ++g) {
          d1[g] = std::numeric_limits<double>::infinity();
          d2[g] = std::numeric_limits<double>::infinity();
        }
        core::for_each_point<W>(packed, N, a, [&](index_t, const double *v) {
          for (std::size_t g = 0; g < W; ++g) {
            const double val = v[g];
            if (val < d1[g]) {
              d2[g] = d1[g];
              d1[g] = val;
            } else if (val < d2[g]) {
              d2[g] = val;
            }
          }
        });
        const index_t valid = std::min<index_t>(static_cast<index_t>(W), N - a);
        for (index_t g = 0; g < valid; ++g)
          mu[static_cast<std::size_t>(a + g)] = (k < N && N >= 2) ? d2[g] : d1[g];
      }
    }

    void compute_rho(const std::vector<double> &mu, std::vector<double> &rho) const
    {
      constexpr std::size_t W = core::group_rows;
      const index_t n_groups = static_cast<index_t>((Nz + W - 1) / W);
#ifdef _OPENMP
      const int nthreads = (N >= kParallelMinN) ? get_max_threads() : 1;
      const index_t chunk = omp_chunk_size_for(n_groups, 4, nthreads);
#pragma omp parallel for schedule(dynamic, chunk) num_threads(nthreads) if (N >= kParallelMinN)
#endif
      for (index_t q = 0; q < n_groups; ++q) {
        const index_t a = q * static_cast<index_t>(W);
        double s[W]{};
        core::for_each_point<W>(packed, N, a, [&](index_t j, const double *v) {
          const double muj = mu[static_cast<std::size_t>(j)];
          for (std::size_t g = 0; g < W; ++g) {
            const double dd = v[g] - muj;
            if (dd < 0.0) s[g] += dd;
          }
        });
        const index_t valid = std::min<index_t>(static_cast<index_t>(W), N - a);
        for (index_t g = 0; g < valid; ++g)
          rho[static_cast<std::size_t>(a + g)] = s[g];
      }
    }
  };

  /// Candidate facilities surviving Beasley reduced-cost fixing (`fixed_open ⊆ core`, ascending).
  struct FixingResult
  {
    std::vector<index_t> core;       ///< survivors (facilities NOT proven closed).
    std::vector<index_t> fixed_open; ///< facilities proven open in every optimum.
  };

  /**
   * Beasley (1993) reduced-cost fixing from the Lagrangian dual state: facility
   * scores ρ_i(μ*) = Σ_j min(0, D_ij − μ_j), a valid lower bound LB = L(μ*) and a
   * valid primal upper bound UB.
   */
  FixingResult reduced_cost_fixing(const std::vector<double> &rho, index_t k,
                                   double lower_bound, double upper_bound)
  {
    const auto N = static_cast<index_t>(rho.size());
    FixingResult out;

    if (!std::isfinite(lower_bound) || !std::isfinite(upper_bound)) {
      out.core.resize(static_cast<std::size_t>(N));
      std::iota(out.core.begin(), out.core.end(), index_t{ 0 });
      return out;
    }

    std::vector<index_t> order(static_cast<std::size_t>(N));
    std::iota(order.begin(), order.end(), index_t{ 0 });
    std::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
      const double ra = rho[static_cast<std::size_t>(a)], rb = rho[static_cast<std::size_t>(b)];
      return ra < rb || (ra == rb && a < b);
    });

    const double rho_k = rho[static_cast<std::size_t>(order[static_cast<std::size_t>(k - 1)])];
    const double rho_kp1 = (k < N)
                             ? rho[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])]
                             : std::numeric_limits<double>::infinity();

    const double tol = 1e-9 * std::max(std::abs(lower_bound), std::abs(upper_bound));
    const double fix_rhs = upper_bound + tol;

    std::vector<char> in_Sk(static_cast<std::size_t>(N), 0);
    for (index_t t = 0; t < k; ++t) in_Sk[static_cast<std::size_t>(order[static_cast<std::size_t>(t)])] = 1;

    for (index_t i = 0; i < N; ++i) {
      const double ri = rho[static_cast<std::size_t>(i)];
      if (in_Sk[static_cast<std::size_t>(i)]) {
        if (lower_bound + (rho_kp1 - ri) > fix_rhs) out.fixed_open.push_back(i);
        out.core.push_back(i);
      } else if (!(lower_bound + (ri - rho_k) > fix_rhs)) {
        out.core.push_back(i);
      }
    }
    return out;
  }

  /// Cluster-grouped O(Σ_c |C_c|^2) Lloyd local search followed by exact O(N^2)
  /// FastPAM 1-swap polish (enabled when `enable_1swap` is true and 2 <= k < N).
  template <typename Mat>
  double pmedian_local_search(const Mat &mat, index_t N, index_t k,
                              std::vector<index_t> &medoids, std::vector<index_t> &labels,
                              int max_sweeps = 32, bool enable_1swap = true)
  {
    const std::size_t Nz = static_cast<std::size_t>(N);
    const std::size_t kz = static_cast<std::size_t>(k);
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<index_t> cluster_of(Nz, 0);
    std::vector<std::size_t> counts(kz, 0), offsets(kz + 1, 0), cursor(kz, 0);
    std::vector<index_t> members(Nz, 0);
    labels.assign(Nz, medoids[0]);
    double cost = 0.0;

    auto deduplicate_medoids = [&]() {
      std::vector<char> used(Nz, 0);
      index_t next_free = 0;
      for (std::size_t c = 0; c < kz; ++c) {
        const index_t m = medoids[c];
        if (m >= 0 && m < N && !used[static_cast<std::size_t>(m)]) {
          used[static_cast<std::size_t>(m)] = 1;
        } else {
          while (next_free < N && used[static_cast<std::size_t>(next_free)]) ++next_free;
          if (next_free < N) {
            medoids[c] = next_free;
            used[static_cast<std::size_t>(next_free)] = 1;
          }
        }
      }
    };
    deduplicate_medoids();

    auto assign = [&]() {
      cost = 0.0;
      for (index_t j = 0; j < N; ++j) {
        const auto nearest = mip::nearest_medoid(k, [&](index_t c) {
          return mat(medoids[static_cast<std::size_t>(c)], j);
        });
        cluster_of[static_cast<std::size_t>(j)] = nearest.position;
        labels[static_cast<std::size_t>(j)] = medoids[static_cast<std::size_t>(nearest.position)];
        cost += nearest.distance;
      }
    };

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
      assign();
      std::fill(counts.begin(), counts.end(), 0);
      for (std::size_t j = 0; j < Nz; ++j)
        ++counts[static_cast<std::size_t>(cluster_of[j])];
      offsets[0] = 0;
      for (std::size_t c = 0; c < kz; ++c)
        offsets[c + 1] = offsets[c] + counts[c];
      std::copy(offsets.begin(), offsets.begin() + static_cast<std::ptrdiff_t>(kz), cursor.begin());
      for (index_t j = 0; j < N; ++j) {
        const auto c = static_cast<std::size_t>(cluster_of[static_cast<std::size_t>(j)]);
        members[cursor[c]++] = j;
      }

      bool changed = false;
      for (index_t c = 0; c < k; ++c) {
        const std::size_t start = offsets[static_cast<std::size_t>(c)];
        const std::size_t sz = counts[static_cast<std::size_t>(c)];
        if (sz == 0) continue;
        const index_t *const mem = members.data() + start;
        double best_sum = inf;
        index_t best_m = medoids[static_cast<std::size_t>(c)];
        for (std::size_t p = 0; p < sz; ++p) {
          const index_t cand = mem[p];
          double s = 0.0;
          for (std::size_t q = 0; q < sz; ++q)
            s += mat(cand, mem[q]);
          if (s < best_sum) {
            best_sum = s;
            best_m = cand;
          }
        }
        if (best_m != medoids[static_cast<std::size_t>(c)]) {
          medoids[static_cast<std::size_t>(c)] = best_m;
          changed = true;
        }
      }
      if (!changed) break;
    }

    deduplicate_medoids();

    // Exact FastPAM 1-swap polish (O(N^2) per pass): escapes within-cluster Lloyd
    // local minima when an improving swap moves a medoid across cluster boundaries.
    if (enable_1swap && k >= 2 && k < N) {
      std::vector<char> is_medoid(Nz, 0);
      for (std::size_t c = 0; c < kz; ++c)
        is_medoid[static_cast<std::size_t>(medoids[c])] = 1;

      std::vector<double> d1(Nz, inf), d2(Nz, inf);
      std::vector<index_t> m1(Nz, 0);
      std::vector<double> removal_loss(kz, 0.0), adjust(kz, 0.0);

      for (int pass = 0; pass < max_sweeps; ++pass) {
        double cur_cost = 0.0;
        std::fill(removal_loss.begin(), removal_loss.end(), 0.0);
        for (index_t j = 0; j < N; ++j) {
          double best1 = inf, best2 = inf;
          index_t owner = 0;
          for (index_t r = 0; r < k; ++r) {
            const double d = mat(medoids[static_cast<std::size_t>(r)], j);
            if (d < best1) {
              best2 = best1;
              best1 = d;
              owner = r;
            } else if (d < best2) {
              best2 = d;
            }
          }
          d1[static_cast<std::size_t>(j)] = best1;
          d2[static_cast<std::size_t>(j)] = best2;
          m1[static_cast<std::size_t>(j)] = owner;
          removal_loss[static_cast<std::size_t>(owner)] += (best2 - best1);
          cur_cost += best1;
        }

        double best_delta = 0.0;
        index_t best_r = -1, best_c = -1;
        for (index_t cand = 0; cand < N; ++cand) {
          if (is_medoid[static_cast<std::size_t>(cand)]) continue;
          std::copy(removal_loss.begin(), removal_loss.end(), adjust.begin());
          double gain_all = 0.0;
          for (index_t j = 0; j < N; ++j) {
            const std::size_t jz = static_cast<std::size_t>(j);
            const double d = mat(cand, j);
            if (d < d1[jz]) {
              gain_all += (d - d1[jz]);
              adjust[static_cast<std::size_t>(m1[jz])] += (d1[jz] - d2[jz]);
            } else if (d < d2[jz]) {
              adjust[static_cast<std::size_t>(m1[jz])] += (d - d2[jz]);
            }
          }
          for (index_t r = 0; r < k; ++r) {
            const double delta = gain_all + adjust[static_cast<std::size_t>(r)];
            if (delta < best_delta) {
              best_delta = delta;
              best_r = r;
              best_c = cand;
            }
          }
        }

        if (best_r < 0) break;
        // A swap's gain must exceed the rounding of the cost and of the removal loss the gain cancels against.
        const double tol = std::max(1e-12 * std::abs(cur_cost),
                                    kRounding * static_cast<double>(N) * removal_loss[static_cast<std::size_t>(best_r)]);
        if (!(best_delta < -tol)) break;
        is_medoid[static_cast<std::size_t>(medoids[static_cast<std::size_t>(best_r)])] = 0;
        is_medoid[static_cast<std::size_t>(best_c)] = 1;
        medoids[static_cast<std::size_t>(best_r)] = best_c;
      }
    }

    std::sort(medoids.begin(), medoids.end());
    assign();
    return cost;
  }

  /// Evaluate the Lagrangian dual at μ and fused primal nearest-medoid repair on S_k = idx[0..k-1]:
  /// fills @p rho, @p g, @p idx, @p cheap_lab, and @p cheap_cost; returns L(μ).
  template <typename Mat>
  double evaluate_dual_and_primal(const Mat &mat, index_t N, index_t k,
                                  const std::vector<double> &mu, std::vector<double> &rho,
                                  std::vector<double> &g, std::vector<index_t> &idx,
                                  std::vector<index_t> &cheap_lab, double &cheap_cost)
  {
    mat.compute_rho(mu, rho);

    std::iota(idx.begin(), idx.end(), index_t{ 0 });
    std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), [&](index_t a, index_t b) {
      const double ra = rho[static_cast<std::size_t>(a)];
      const double rb = rho[static_cast<std::size_t>(b)];
      return ra < rb || (ra == rb && a < b);
    });

    double sum_mu = 0.0;
    for (index_t j = 0; j < N; ++j) sum_mu += mu[static_cast<std::size_t>(j)];
    double sum_rho_S = 0.0;
    for (index_t t = 0; t < k; ++t) sum_rho_S += rho[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])];
    const double L = sum_mu + sum_rho_S;

    cheap_cost = 0.0;
    for (index_t j = 0; j < N; ++j) {
      index_t served = 0;
      const double muj = mu[static_cast<std::size_t>(j)];
      double best_d = std::numeric_limits<double>::infinity();
      index_t best_m = idx[0];
      for (index_t t = 0; t < k; ++t) {
        const index_t m = idx[static_cast<std::size_t>(t)];
        const double d = mat(m, j);
        if (d < muj) ++served;
        if (d < best_d) {
          best_d = d;
          best_m = m;
        }
      }
      g[static_cast<std::size_t>(j)] = 1.0 - static_cast<double>(served);
      cheap_lab[static_cast<std::size_t>(j)] = best_m;
      cheap_cost += best_d;
    }
    return L;
  }

  /// Update the primal incumbent from a pre-evaluated cheap primal repair (`cheap_cost`, `cheap_lab`),
  /// optionally followed by local-search polish.
  template <typename Mat>
  void update_primal(const Mat &mat, index_t N, index_t k,
                     const index_t *cand_k, double cheap_cost,
                     const std::vector<index_t> &cheap_lab, bool do_polish,
                     double &best_primal, std::vector<index_t> &best_medoids,
                     std::vector<index_t> &best_labels)
  {
    if (cheap_cost < best_primal) {
      best_primal = cheap_cost;
      best_medoids.assign(cand_k, cand_k + k);
      std::sort(best_medoids.begin(), best_medoids.end());
      best_labels = cheap_lab;
    }
    if (do_polish) {
      std::vector<index_t> rm(cand_k, cand_k + k), rl;
      const double rc = pmedian_local_search(mat, N, k, rm, rl, /*max_sweeps=*/32, /*enable_1swap=*/(N <= 256));
      if (rc < best_primal) {
        best_primal = rc;
        best_medoids = std::move(rm);
        best_labels = std::move(rl);
      }
    }
  }

  /// Evaluate cheap primal repair for candidate medoid prefix `cand_k[0..k-1]` and update incumbent.
  template <typename Mat>
  void try_primal(const Mat &mat, index_t N, index_t k,
                  const index_t *cand_k, bool do_polish, double &best_primal,
                  std::vector<index_t> &best_medoids, std::vector<index_t> &best_labels,
                  std::vector<index_t> &cheap_lab)
  {
    double cheap_cost = 0.0;
    for (index_t j = 0; j < N; ++j) {
      const auto nearest = mip::nearest_medoid(k, [&](index_t t) {
        return mat(cand_k[t], j);
      });
      cheap_lab[static_cast<std::size_t>(j)] = cand_k[nearest.position];
      cheap_cost += nearest.distance;
    }
    update_primal(mat, N, k, cand_k, cheap_cost, cheap_lab, do_polish,
                  best_primal, best_medoids, best_labels);
  }

  /// Resolve public `initial_ub` (default -1.0 means "unset" on non-negative matrices) into a SeedIncumbent.
  SeedIncumbent resolve_public_seed(double initial_ub, double minD) noexcept
  {
    SeedIncumbent s;
    if (!std::isfinite(initial_ub)) return s;
    if (minD >= 0.0) {
      if (initial_ub > 0.0) {
        s.has_ub = true;
        s.ub = initial_ub;
      }
    } else {
      if (initial_ub != -1.0) {
        s.has_ub = true;
        s.ub = initial_ub;
      }
    }
    return s;
  }

  /// Seed `best_primal`, `best_medoids`, `best_labels` from `seed.medoids` if supplied.
  template <typename Mat>
  void apply_seed_medoids(const Mat &mat, index_t N, index_t k, const SeedIncumbent &seed,
                          double &best_primal, std::vector<index_t> &best_medoids,
                          std::vector<index_t> &best_labels, std::vector<index_t> &cheap_lab)
  {
    if (seed.medoids != nullptr && static_cast<index_t>(seed.medoids->size()) == k) {
      try_primal(mat, N, k, seed.medoids->data(), /*do_polish=*/false, best_primal, best_medoids, best_labels, cheap_lab);
    }
  }

  struct RootSolveOutput
  {
    LagrangianResult result;
    std::vector<double> rho_at_best;
    FixingResult fix;
  };

  /// Tail of the subgradient root: final polish, Beasley reduced-cost fixing, and result assembly.
  template <typename Mat>
  RootSolveOutput finalize(const Mat &mat, index_t N, index_t k, double best_lb,
                           double best_primal, double seed_ub,
                           const std::vector<index_t> &sorted_seed_medoids,
                           std::vector<index_t> best_medoids, std::vector<index_t> best_labels,
                           std::vector<double> mu, std::vector<double> rho_at_best,
                           int iterations, double rel_gap_tol)
  {
    const double inf = std::numeric_limits<double>::infinity();
    const bool seed_unchanged = !sorted_seed_medoids.empty() && best_medoids == sorted_seed_medoids;
    if (!best_medoids.empty() && !seed_unchanged) {
      std::vector<index_t> fm = best_medoids, fl;
      const double fc = pmedian_local_search(mat, N, k, fm, fl, /*max_sweeps=*/32, /*enable_1swap=*/(N <= 2048));
      if (fc < best_primal) {
        best_primal = fc;
        best_medoids = std::move(fm);
        best_labels = std::move(fl);
      }
    }

    double lower = (best_lb == -inf) ? 0.0 : best_lb;
    if (mat.nonneg) lower = std::max(lower, 0.0);

    const double fix_ub = std::min(best_primal, seed_ub);
    FixingResult fix = reduced_cost_fixing(rho_at_best, k, lower, fix_ub);

    LagrangianResult r;
    r.lower_bound = lower;
    r.upper_bound = best_primal;
    if ((mat.nonneg && best_primal == 0.0) || best_primal == r.lower_bound) {
      r.gap = 0.0;
      r.certified_optimal = true;
    } else {
      const double denom = (best_primal == 0.0) ? 1e-12 : std::abs(best_primal);
      r.gap = (best_primal - r.lower_bound) / denom;
      r.certified_optimal = (r.gap <= rel_gap_tol);
    }
    r.medoids = std::move(best_medoids);
    r.labels = std::move(best_labels);
    r.multipliers = std::move(mu);
    r.iterations = iterations;
    r.core = fix.core;
    r.n_core = static_cast<index_t>(r.core.size());
    return { std::move(r), std::move(rho_at_best), std::move(fix) };
  }

  template <typename Mat>
  RootSolveOutput lagrangian_root_subgrad_internal(const Mat &mat, index_t N, index_t k,
                                                   const SeedIncumbent &seed)
  {
    const std::size_t Nz = static_cast<std::size_t>(N);
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> mu(Nz, 0.0);
    mat.compute_initial_mu(k, mu); // Erlenkotter d_(2)(j) warm start (d_(1)(j) when k == N).

    std::vector<double> rho(Nz, 0.0);
    std::vector<double> g(Nz, 0.0);
    std::vector<double> d(Nz, 0.0);
    std::vector<double> d_prev(Nz, 0.0);
    std::vector<double> y_bar(Nz, 0.0); // Volume-style exponential primal facility averages.
    std::vector<index_t> idx(Nz), vol_idx(Nz);
    std::vector<index_t> cheap_lab(Nz, 0);
    double dprev_norm2 = 0.0;
    bool have_dprev = false;

    double best_lb = -inf;
    double best_primal = inf;
    const double seed_ub = seed.has_ub ? seed.ub : inf;
    std::vector<index_t> best_medoids, best_labels;
    std::vector<index_t> sorted_seed_medoids;
    if (seed.medoids != nullptr && static_cast<index_t>(seed.medoids->size()) == k) {
      sorted_seed_medoids = *seed.medoids;
      std::sort(sorted_seed_medoids.begin(), sorted_seed_medoids.end());
    }
    apply_seed_medoids(mat, N, k, seed, best_primal, best_medoids, best_labels, cheap_lab);

    std::vector<double> mu_at_best = mu;
    std::vector<double> rho_at_best(Nz, 0.0);
    std::array<double, 100> lb_hist;
    lb_hist.fill(-inf);

    double lambda = params.lambda0;
    int stall = 0;
    int iter = 0;

    warn_if_single_threaded();

    for (iter = 0; iter < params.max_iters; ++iter) {
      double cheap_cost = inf;
      const double L = evaluate_dual_and_primal(mat, N, k, mu, rho, g, idx, cheap_lab, cheap_cost);

      if (L > best_lb) {
        best_lb = mat.nonneg ? std::max(L, 0.0) : L;
        mu_at_best = mu;
        rho_at_best = rho;
        stall = 0;
      } else {
        ++stall;
      }

      // Update Volume primal averages ȳ ← (1 − α) ȳ + α 1_{S_k}.
      const double alpha = (iter == 0) ? 1.0 : std::max(0.05, 2.0 / static_cast<double>(iter + 2));
      for (std::size_t i = 0; i < Nz; ++i) y_bar[i] *= (1.0 - alpha);
      for (index_t t = 0; t < k; ++t)
        y_bar[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])] += alpha;

      const bool do_polish = params.polish_period > 0 && (iter % params.polish_period) == 0;
      update_primal(mat, N, k, idx.data(), cheap_cost, cheap_lab, do_polish,
                    best_primal, best_medoids, best_labels);
      if (do_polish && iter > 0) {
        std::iota(vol_idx.begin(), vol_idx.end(), index_t{ 0 });
        std::nth_element(vol_idx.begin(), vol_idx.begin() + (k - 1), vol_idx.end(), [&](index_t a, index_t b) {
          const double ya = y_bar[static_cast<std::size_t>(a)];
          const double yb = y_bar[static_cast<std::size_t>(b)];
          return ya > yb || (ya == yb && a < b);
        });
        const bool vol_polish = (iter % (4 * params.polish_period)) == 0;
        try_primal(mat, N, k, vol_idx.data(), vol_polish, best_primal, best_medoids, best_labels, cheap_lab);
      }

      if ((mat.nonneg && best_primal == 0.0)
          || best_primal - best_lb <= params.rel_gap_tol * std::abs(best_primal)) {
        ++iter;
        break;
      }

      const double lb_100_ago = lb_hist[static_cast<std::size_t>(iter % 100)];
      lb_hist[static_cast<std::size_t>(iter % 100)] = best_lb;
      if (iter >= 100 && lambda <= params.lambda_min && std::isfinite(lb_100_ago) && std::isfinite(best_lb)) {
        const double lb_denom = std::max(std::abs(best_lb), std::abs(lb_100_ago));
        if ((best_lb - lb_100_ago) <= 1e-7 * lb_denom) {
          ++iter;
          break;
        }
      }

      double gnorm2 = 0.0;
      for (index_t j = 0; j < N; ++j) gnorm2 += g[static_cast<std::size_t>(j)] * g[static_cast<std::size_t>(j)];
      if (gnorm2 == 0.0) {
        ++iter;
        break;
      }

      if (stall >= params.stall_halve) {
        lambda = std::max(lambda * 0.5, params.lambda_min);
        stall = 0;
      }

      double beta = 0.0;
      if (have_dprev && params.deflect > 0.0 && dprev_norm2 > 0.0) {
        double dot = 0.0;
        for (index_t j = 0; j < N; ++j) dot += g[static_cast<std::size_t>(j)] * d_prev[static_cast<std::size_t>(j)];
        if (dot < 0.0) beta = -params.deflect * dot / dprev_norm2;
      }
      double dnorm2 = 0.0;
      for (index_t j = 0; j < N; ++j) {
        const double dj = g[static_cast<std::size_t>(j)] + beta * d_prev[static_cast<std::size_t>(j)];
        d[static_cast<std::size_t>(j)] = dj;
        dnorm2 += dj * dj;
      }
      if (dnorm2 <= 0.0) {
        d = g;
        dnorm2 = gnorm2;
      }

      const double ub_step = std::min(best_primal, seed_ub);
      if (ub_step <= L) {
        ++iter;
        break;
      }
      const double step = lambda * (ub_step - L) / dnorm2;
      for (index_t j = 0; j < N; ++j) mu[static_cast<std::size_t>(j)] += step * d[static_cast<std::size_t>(j)];

      d_prev = d;
      dprev_norm2 = dnorm2;
      have_dprev = true;
    }

    return finalize(mat, N, k, best_lb, best_primal, seed_ub, sorted_seed_medoids,
                    std::move(best_medoids), std::move(best_labels),
                    std::move(mu_at_best), std::move(rho_at_best), iter, params.rel_gap_tol);
  }

  template <typename Mat>
  LagrangianResult lagrangian_root_subgrad_impl(const Mat &mat, index_t N, index_t k,
                                                const SeedIncumbent &seed)
  {
    return lagrangian_root_subgrad_internal(mat, N, k, seed).result;
  }

  template <typename Mat>
  LagrangianResult lagrangian_root_exact_impl(const Mat &mat, index_t N, index_t k,
                                              const SeedIncumbent &seed,
                                              std::int64_t max_nodes)
  {
    RootSolveOutput sub = lagrangian_root_subgrad_internal(mat, N, k, seed);
    LagrangianResult &root = sub.result;
    if (root.certified_optimal) {
      root.nodes = 0;
      return root;
    }

    const std::size_t Nz = static_cast<std::size_t>(N);
    const double inf = std::numeric_limits<double>::infinity();

    const std::vector<double> &mu = root.multipliers;
    const std::vector<double> &rho = sub.rho_at_best;
    double sum_mu = 0.0;
    for (index_t j = 0; j < N; ++j) sum_mu += mu[static_cast<std::size_t>(j)];

    const FixingResult &fx = sub.fix;
    const std::vector<index_t> &forced = fx.fixed_open;
    std::vector<char> is_forced(Nz, 0);
    for (index_t f : forced) is_forced[static_cast<std::size_t>(f)] = 1;
    std::vector<index_t> cand;
    cand.reserve(fx.core.size());
    for (index_t c : fx.core)
      if (!is_forced[static_cast<std::size_t>(c)]) cand.push_back(c);

    std::sort(cand.begin(), cand.end(), [&](index_t a, index_t b) {
      const double ra = rho[static_cast<std::size_t>(a)], rb = rho[static_cast<std::size_t>(b)];
      return ra < rb || (ra == rb && a < b);
    });
    const auto C = static_cast<index_t>(cand.size());
    const index_t need = k - static_cast<index_t>(forced.size());

    double forced_rho = 0.0;
    for (index_t f : forced) forced_rho += rho[static_cast<std::size_t>(f)];
    std::vector<double> csum(static_cast<std::size_t>(C) + 1, 0.0);
    for (index_t t = 0; t < C; ++t)
      csum[static_cast<std::size_t>(t) + 1] = csum[static_cast<std::size_t>(t)]
                                              + rho[static_cast<std::size_t>(cand[static_cast<std::size_t>(t)])];

    // Precompute per-client suffix minima over cand[pos..C-1] and depth-indexed open_mins[0..need]
    // when within memory budget, eliminating per-node std::vector allocations and tightening node bounds
    // via exact Erlenkotter client-floor adjustments and d_{F_1}(j) dual clamping.
    const std::size_t suffix_cells = (static_cast<std::size_t>(C) + 1) * Nz;
    const std::size_t open_cells = (static_cast<std::size_t>(need) + 1) * Nz;
    const bool fast_tables = suffix_cells <= kMaxSuffixMinCells && open_cells <= kMaxSuffixMinCells;

    std::vector<double> suffix_min;
    std::vector<double> open_mins;
    std::vector<double> tilde_mu;
    std::vector<double> tilde_rho;
    if (fast_tables) {
      suffix_min.assign(suffix_cells, inf);
      for (index_t pos = C; pos-- > 0;) {
        const index_t fac = cand[static_cast<std::size_t>(pos)];
        const double *next_row = suffix_min.data() + static_cast<std::size_t>(pos + 1) * Nz;
        double *cur_row = suffix_min.data() + static_cast<std::size_t>(pos) * Nz;
        for (index_t j = 0; j < N; ++j)
          cur_row[j] = std::min(mat(fac, j), next_row[j]);
      }
      open_mins.assign(open_cells, inf);
      for (index_t f : forced) {
        for (index_t j = 0; j < N; ++j)
          open_mins[static_cast<std::size_t>(j)] = std::min(open_mins[static_cast<std::size_t>(j)], mat(f, j));
      }
      tilde_mu.resize(Nz);
      tilde_rho.resize(static_cast<std::size_t>(C));
    }

    auto fallback_cost_of = [&](const std::vector<index_t> &S) {
      const auto n_open = static_cast<index_t>(S.size());
      double c = 0.0;
      for (index_t j = 0; j < N; ++j)
        c += nearest_medoid(n_open, [&](index_t t) {
               return mat(S[static_cast<std::size_t>(t)], j);
             }).distance;
      return c;
    };

    double best_cost = root.upper_bound;
    std::vector<index_t> best_medoids = root.medoids;
    double mu_size = 0.0; // the size of the multipliers a node bound sums, whose rounding floors the tolerance
    for (const double m : mu) mu_size += std::abs(m);
    const double tol = std::max(1e-9 * std::abs(best_cost), kRounding * static_cast<double>(N) * mu_size);

    struct Frame
    {
      index_t pos;
      index_t opened;
      index_t added; ///< >= 0 when this frame opened `added`; -1 for root / SKIP.
      double opened_rho;
    };
    std::vector<Frame> stack;
    stack.reserve(static_cast<std::size_t>(2 * C + 2));
    stack.push_back({ 0, 0, -1, 0.0 });
    std::vector<index_t> chosen(static_cast<std::size_t>(need), 0);
    std::vector<index_t> leaf_S;
    leaf_S.reserve(static_cast<std::size_t>(k));

    std::int64_t nodes = 0;
    bool capped = false;

    while (!stack.empty()) {
      if (nodes >= max_nodes) {
        capped = true;
        break;
      }
      const Frame fr = stack.back();
      stack.pop_back();
      ++nodes;

      if (fr.added >= 0) {
        chosen[static_cast<std::size_t>(fr.opened - 1)] = fr.added;
        if (fast_tables) {
          const double *prev_open = open_mins.data() + static_cast<std::size_t>(fr.opened - 1) * Nz;
          double *cur_open = open_mins.data() + static_cast<std::size_t>(fr.opened) * Nz;
          for (index_t j = 0; j < N; ++j)
            cur_open[j] = std::min(prev_open[j], mat(fr.added, j));
        }
      }

      const index_t remaining_need = need - fr.opened;
      if (remaining_need == 0) {
        double c = 0.0;
        if (fast_tables) {
          const double *cur_open = open_mins.data() + static_cast<std::size_t>(fr.opened) * Nz;
          for (index_t j = 0; j < N; ++j) c += cur_open[j];
        } else {
          leaf_S = forced;
          leaf_S.insert(leaf_S.end(), chosen.begin(), chosen.end());
          c = fallback_cost_of(leaf_S);
        }
        if (c < best_cost) {
          best_cost = c;
          best_medoids = forced;
          best_medoids.insert(best_medoids.end(), chosen.begin(), chosen.end());
        }
        continue;
      }
      const index_t rem = C - fr.pos;
      if (rem < remaining_need) continue;

      double extra_lb = 0.0;
      if (fast_tables && fr.pos > 0) {
        const double *cur_open = open_mins.data() + static_cast<std::size_t>(fr.opened) * Nz;
        const double *rem_min = suffix_min.data() + static_cast<std::size_t>(fr.pos) * Nz;
        for (index_t j = 0; j < N; ++j) {
          const double mj = std::min(cur_open[j], rem_min[j]);
          const double delta_mu = mj - mu[static_cast<std::size_t>(j)];
          if (delta_mu > 0.0) extra_lb += delta_mu;
        }
      }

      const double node_lb = sum_mu + extra_lb + forced_rho + fr.opened_rho
                             + (csum[static_cast<std::size_t>(fr.pos) + static_cast<std::size_t>(remaining_need)]
                                - csum[static_cast<std::size_t>(fr.pos)]);
      if (node_lb >= best_cost - tol) continue;

      if (fast_tables && nodes >= 64 && (fr.opened > 0 || !forced.empty())) {
        const double *cur_open = open_mins.data() + static_cast<std::size_t>(fr.opened) * Nz;
        const double *rem_min = suffix_min.data() + static_cast<std::size_t>(fr.pos) * Nz;
        bool any_clamped = false;
        double sum_tilde_mu = 0.0;
        double tilde_mu_size = 0.0;
        for (index_t j = 0; j < N; ++j) {
          const double dj = cur_open[j];
          const double mj = std::min(dj, rem_min[j]);
          const double muj = mu[static_cast<std::size_t>(j)];
          if (dj < muj) any_clamped = true;
          const double tm = std::max(mj, std::min(muj, dj));
          tilde_mu[static_cast<std::size_t>(j)] = tm;
          sum_tilde_mu += tm;
          tilde_mu_size += std::abs(tm);
        }
        if (any_clamped) {
          for (index_t t = 0; t < rem; ++t) {
            const index_t fac = cand[static_cast<std::size_t>(fr.pos + t)];
            double s = 0.0;
            for (index_t j = 0; j < N; ++j) {
              const double dd = mat(fac, j) - tilde_mu[static_cast<std::size_t>(j)];
              if (dd < 0.0) s += dd;
            }
            tilde_rho[static_cast<std::size_t>(t)] = s;
          }
          std::nth_element(tilde_rho.begin(),
                           tilde_rho.begin() + (remaining_need - 1),
                           tilde_rho.begin() + rem);
          double clamped_lb = sum_tilde_mu;
          for (index_t t = 0; t < remaining_need; ++t)
            clamped_lb += tilde_rho[static_cast<std::size_t>(t)];
          const double clamped_tol = std::max(
            1e-9 * std::abs(best_cost),
            kRounding * static_cast<double>(N) * tilde_mu_size);
          if (clamped_lb >= best_cost - clamped_tol) continue;
        }
      }

      const index_t next_fac = cand[static_cast<std::size_t>(fr.pos)];
      stack.push_back({ fr.pos + 1, fr.opened, -1, fr.opened_rho });
      stack.push_back({ fr.pos + 1, fr.opened + 1, next_fac, fr.opened_rho + rho[static_cast<std::size_t>(next_fac)] });
    }

    std::sort(best_medoids.begin(), best_medoids.end());
    LagrangianResult r = root;
    r.medoids = best_medoids;
    r.labels.assign(Nz, best_medoids.empty() ? 0 : best_medoids[0]);
    const auto n_best = static_cast<index_t>(best_medoids.size());
    if (n_best > 0) {
      for (index_t j = 0; j < N; ++j) {
        const auto nearest = nearest_medoid(n_best, [&](index_t t) {
          return mat(best_medoids[static_cast<std::size_t>(t)], j);
        });
        r.labels[static_cast<std::size_t>(j)] = best_medoids[static_cast<std::size_t>(nearest.position)];
      }
    }
    r.upper_bound = best_cost;
    r.nodes = nodes;
    if (capped) {
      r.lower_bound = root.lower_bound;
      if ((mat.nonneg && best_cost == 0.0) || best_cost == r.lower_bound) {
        r.gap = 0.0;
        r.certified_optimal = (mat.nonneg && best_cost == 0.0);
      } else {
        const double denom = (best_cost == 0.0) ? 1e-12 : std::abs(best_cost);
        r.gap = (best_cost - r.lower_bound) / denom;
        r.certified_optimal = false;
      }
    } else {
      r.lower_bound = best_cost;
      r.gap = 0.0;
      r.certified_optimal = true;
    }
    return r;
  }

} // namespace

LagrangianResult lagrangian_root(const double *D, index_t N, index_t k, double initial_ub)
{
  if (N <= 0) throw InvalidInput("lagrangian_root: N must be positive");
  if (k < 1 || k > N)
    throw InvalidInput("lagrangian_root: require 1 <= k <= N (k=" + std::to_string(k)
                       + ", N=" + std::to_string(N) + ")");
  const DenseMatrixView mat(D, N);
  return lagrangian_root_subgrad_impl(mat, N, k, resolve_public_seed(initial_ub, mat.min_distance()));
}

LagrangianResult lagrangian_root_exact(const double *D, index_t N, index_t k,
                                       double initial_ub, std::int64_t max_nodes)
{
  if (N <= 0) throw InvalidInput("lagrangian_root_exact: N must be positive");
  if (k < 1 || k > N)
    throw InvalidInput("lagrangian_root_exact: require 1 <= k <= N (k=" + std::to_string(k)
                       + ", N=" + std::to_string(N) + ")");
  const DenseMatrixView mat(D, N);
  return lagrangian_root_exact_impl(mat, N, k, resolve_public_seed(initial_ub, mat.min_distance()), max_nodes);
}

} // namespace dtwc::mip

namespace dtwc {

void LR_core_clustering(Problem &prob)
{
  validate_mip_settings(prob.mip_settings);
  const index_t N = prob.size();
  const index_t k = prob.n_clusters();
  if (N <= 0) throw InvalidInput("LR-core: the Problem has no data.");

  // Run FasterPAM without mutating `prob` (transactional: if LR-core throws SolverError,
  // `prob`'s prior clustering state remains untouched), and feed both its cost and medoids
  // directly into the zero-copy packed DistanceMatrix exact solver.
  const auto pam = fast_pam_solve(prob, k, settings::DEFAULT_MAX_ITER, prob.random_seed());
  const mip::SeedIncumbent seed{ true, pam.total_cost, &pam.medoid_indices };

  const mip::LagrangianResult r = prob.distance_matrix().visit([&](const auto *packed) {
    const mip::PackedMatrixView<std::remove_cv_t<std::remove_pointer_t<decltype(packed)>>> mat(packed, N);
    return mip::lagrangian_root_exact_impl(mat, N, k, seed, prob.mip_settings.lr_max_nodes);
  });

  if (prob.verbose()) std::cout << "LR-core: subgradient root, " << r.nodes << " branch-and-bound nodes\n";

  if (!r.certified_optimal)
    throw SolverError("LR-core did not prove optimality (gap "
                      + std::to_string(r.gap) + " after " + std::to_string(r.nodes)
                      +
                      " branch-and-bound nodes). The incumbent is a heuristic, not a certificate; "
                      "raise Problem::mip_settings.lr_max_nodes (currently "
                      + std::to_string(prob.mip_settings.lr_max_nodes)
                      +
                      "), use Method::MIP for an exact MIP backend, or Method::Kmedoids for a "
                      "heuristic answer.");

  const auto n_medoids = static_cast<index_t>(r.medoids.size());
  std::vector<index_t> slot_of(static_cast<std::size_t>(N), -1);
  for (index_t c = 0; c < n_medoids; ++c)
    slot_of[static_cast<std::size_t>(r.medoids[static_cast<std::size_t>(c)])] = c;

  core::ClusteringResult result;
  result.medoid_indices = r.medoids;
  result.labels.resize(static_cast<std::size_t>(N));
  prob.distance_matrix().visit([&](const auto *packed) {
    for (index_t j = 0; j < N; ++j) {
      const auto jz = static_cast<std::size_t>(j);
      const index_t med_idx = r.labels[jz];
      index_t lab = slot_of[static_cast<std::size_t>(med_idx)];
      if (slot_of[jz] >= 0 && med_idx != j) {
        const auto d_self = static_cast<double>(packed[core::tri_index(jz, jz)]);
        const auto d_served = static_cast<double>(packed[core::tri_index(static_cast<std::size_t>(med_idx), jz)]);
        if (d_self <= d_served) lab = slot_of[jz];
      }
      result.labels[jz] = lab;
    }
  });
  result.total_cost = r.upper_bound;
  result.iterations = r.iterations;
  result.converged = true;
  prob.set_result(result);
}

} // namespace dtwc
