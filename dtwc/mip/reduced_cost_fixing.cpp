/**
 * @file reduced_cost_fixing.cpp
 * @brief Implementation of Beasley reduced-cost fixing (PLAN.md Phase 4, Task 4.2).
 *
 * @details See reduced_cost_fixing.hpp for the two conditional-bound tests. The
 * membership of a facility in S_k (the k opened by the dual) is resolved by a
 * full sort of the scores with an index tie-break, so exact ties at ρ_(k) are
 * handled deterministically and conservatively — a tie never causes an
 * incorrect fix (the arbitrary member kept in S_k gets the ≥ LB open bound; the
 * one pushed out gets the ≥ LB close bound; both are valid).
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include "reduced_cost_fixing.hpp"

#include "../error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace dtwc::mip {

FixingResult reduced_cost_fixing(const std::vector<double> &rho, int k,
                                 double lower_bound, double upper_bound)
{
  const int N = static_cast<int>(rho.size());
  if (N <= 0) throw InvalidInput("reduced_cost_fixing: rho must be non-empty");
  if (k < 1 || k > N)
    throw InvalidInput("reduced_cost_fixing: require 1 <= k <= N (k=" + std::to_string(k)
                       + ", N=" + std::to_string(N) + ")");

  FixingResult out;

  // No valid finite gap ⇒ nothing can be fixed; the whole set survives.
  if (!std::isfinite(lower_bound) || !std::isfinite(upper_bound)) {
    out.core.resize(static_cast<std::size_t>(N));
    std::iota(out.core.begin(), out.core.end(), 0);
    return out;
  }

  // Order facilities by ascending score, index tie-break ⇒ deterministic S_k.
  std::vector<int> order(static_cast<std::size_t>(N));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const double ra = rho[static_cast<std::size_t>(a)], rb = rho[static_cast<std::size_t>(b)];
    return ra < rb || (ra == rb && a < b);
  });

  const double rho_k = rho[static_cast<std::size_t>(order[static_cast<std::size_t>(k - 1)])];
  const double rho_kp1 = (k < N)
    ? rho[static_cast<std::size_t>(order[static_cast<std::size_t>(k)])]
    : std::numeric_limits<double>::infinity();

  // Conservative fixing margin. On a CERTIFIED instance (gap ≈ 0, LB ≈ UB) a
  // facility whose ρ merely TIES ρ_(k) — a genuine alternative optimal medoid —
  // has a conditional bound == UB in exact arithmetic, but rounding of the
  // independently-summed LB/UB/ρ can push it a few ULP above. An exact `>` then
  // fixes an OPTIMAL facility out. We only fix when the bound clears UB by `tol`,
  // scaled to the magnitude: safe (never removes an optimum) and lossless for real
  // eliminations, whose margin is O(problem scale) ≫ tol.
  const double tol = 1e-9 * (1.0 + std::max(std::abs(lower_bound), std::abs(upper_bound)));
  const double fix_rhs = upper_bound + tol;

  // Mark S_k = the k smallest (the facilities the dual opens).
  std::vector<char> in_Sk(static_cast<std::size_t>(N), 0);
  for (int t = 0; t < k; ++t) in_Sk[static_cast<std::size_t>(order[static_cast<std::size_t>(t)])] = 1;

  // Single ascending pass ⇒ every output vector is already sorted.
  for (int i = 0; i < N; ++i) {
    const double ri = rho[static_cast<std::size_t>(i)];
    if (in_Sk[static_cast<std::size_t>(i)]) {
      // Force i CLOSED: dual falls to LB + (ρ_(k+1) − ρ_i). > UB ⇒ i must be open.
      if (lower_bound + (rho_kp1 - ri) > fix_rhs) out.fixed_open.push_back(i);
      out.core.push_back(i); // an opened facility always survives.
    } else {
      // Force i OPEN: dual falls to LB + (ρ_i − ρ_(k)). > UB ⇒ i cannot be open.
      if (lower_bound + (ri - rho_k) > fix_rhs)
        out.fixed_closed.push_back(i);
      else
        out.core.push_back(i);
    }
  }
  return out;
}

} // namespace dtwc::mip
