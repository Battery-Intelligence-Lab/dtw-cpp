/**
 * @file z_normalize_simd.cpp
 * @brief SIMD-accelerated z-normalization using Google Highway.
 *
 * @details Three embarrassingly parallel loops vectorized:
 *          1. Sum reduction for mean
 *          2. Squared-deviation sum for stddev
 *          3. Element-wise normalize: (x - mean) * inv_stddev
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/z_normalize_simd.cpp"
#include "simd/highway_targets.hpp"

#include <cmath>

HWY_BEFORE_NAMESPACE();
namespace dtwc::simd::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void ZNormalizeSimd(double* HWY_RESTRICT series, std::size_t n)
{
  if (n == 0) return;
  if (n == 1) { series[0] = 0.0; return; }

  const hn::ScalableTag<double> d;
  const std::size_t N = hn::Lanes(d);

  // Pass 1 (fused): compute sum(x) and sum(x²) in one memory sweep.
  // Gives mean and variance via var = E[x²] - mean² (König-Huygens).
  // Avoids a separate squared-deviation pass, halving pass-1+2 memory traffic.
  auto sum_vec = hn::Zero(d);
  auto sq_vec  = hn::Zero(d);
  std::size_t i = 0;
  for (; i + N <= n; i += N) {
    const auto val = hn::LoadU(d, series + i);
    sum_vec = hn::Add(sum_vec, val);
    sq_vec  = hn::MulAdd(val, val, sq_vec);  // FMA: sq_vec += val²
  }
  double sum    = hn::ReduceSum(d, sum_vec);
  double sq_sum = hn::ReduceSum(d, sq_vec);
  for (; i < n; ++i) {
    sum    += series[i];
    sq_sum += series[i] * series[i];
  }

  const double inv_n   = 1.0 / static_cast<double>(n);
  const double mean    = sum * inv_n;
  // Clamp to 0 to guard against tiny negative values from floating-point rounding.
  const double variance = std::max(0.0, sq_sum * inv_n - mean * mean);
  const double stddev   = std::sqrt(variance);

  // Pass 2: normalize in place
  const auto mean_vec = hn::Set(d, mean);
  if (stddev > 1e-10) {
    const auto inv_sd_vec = hn::Set(d, 1.0 / stddev);
    i = 0;
    for (; i + N <= n; i += N) {
      const auto val    = hn::LoadU(d, series + i);
      const auto normed = hn::Mul(hn::Sub(val, mean_vec), inv_sd_vec);
      hn::StoreU(normed, d, series + i);
    }
    const double inv_sd = 1.0 / stddev;
    for (; i < n; ++i)
      series[i] = (series[i] - mean) * inv_sd;
  } else {
    const auto zero = hn::Zero(d);
    i = 0;
    for (; i + N <= n; i += N)
      hn::StoreU(zero, d, series + i);
    for (; i < n; ++i)
      series[i] = 0.0;
  }
}

}  // namespace dtwc::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace dtwc::simd {

HWY_EXPORT(ZNormalizeSimd);

void z_normalize_highway(double* series, std::size_t n)
{
  HWY_DYNAMIC_DISPATCH(ZNormalizeSimd)(series, n);
}

}  // namespace dtwc::simd
#endif
