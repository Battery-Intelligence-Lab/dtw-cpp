/**
 * @file lb_keogh_simd.cpp
 * @brief SIMD-accelerated LB_Keogh lower bound using Google Highway.
 *
 * @details Vectorizes the LB_Keogh reduction loop: three contiguous array reads,
 *          element-wise max(0, max(q-U, L-q)), horizontal sum. Processes 4 doubles
 *          per iteration on AVX2, 8 on AVX-512.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

// Highway multi-target compilation: this file is re-included per ISA target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dtwc/simd/lb_keogh_simd.cpp"
#include "dtwc/simd/highway_targets.hpp"

HWY_BEFORE_NAMESPACE();
namespace dtwc::simd::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

double LbKeoghSimd(const double* HWY_RESTRICT query,
                   const double* HWY_RESTRICT upper,
                   const double* HWY_RESTRICT lower,
                   std::size_t n)
{
  if (n == 0) return 0.0;

  const hn::ScalableTag<double> d;
  const std::size_t N = hn::Lanes(d);

  auto sum_vec = hn::Zero(d);
  const auto zero = hn::Zero(d);

  std::size_t i = 0;
  for (; i + N <= n; i += N) {
    const auto q = hn::LoadU(d, query + i);
    const auto u = hn::LoadU(d, upper + i);
    const auto l = hn::LoadU(d, lower + i);

    const auto excess_upper = hn::Sub(q, u);  // query[i] - upper[i]
    const auto excess_lower = hn::Sub(l, q);  // lower[i] - query[i]
    const auto excess = hn::Max(excess_upper, excess_lower);
    const auto clamped = hn::Max(zero, excess);
    sum_vec = hn::Add(sum_vec, clamped);
  }

  double sum = hn::ReduceSum(d, sum_vec);

  // Scalar tail
  for (; i < n; ++i) {
    double eu = query[i] - upper[i];
    double el = lower[i] - query[i];
    double e = eu > el ? eu : el;
    if (e > 0.0) sum += e;
  }

  return sum;
}

}  // namespace dtwc::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

// --- Single-compilation-unit dispatch table ---
#if HWY_ONCE
namespace dtwc::simd {

HWY_EXPORT(LbKeoghSimd);

double lb_keogh_highway(const double* query, const double* upper,
                        const double* lower, std::size_t n)
{
  return HWY_DYNAMIC_DISPATCH(LbKeoghSimd)(query, upper, lower, n);
}

}  // namespace dtwc::simd
#endif  // HWY_ONCE
