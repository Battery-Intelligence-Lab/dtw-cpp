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
#define HWY_TARGET_INCLUDE "dtwc/simd/z_normalize_simd.cpp"
#include "dtwc/simd/highway_targets.hpp"

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

  // Pass 1: compute sum for mean
  auto sum_vec = hn::Zero(d);
  std::size_t i = 0;
  for (; i + N <= n; i += N) {
    sum_vec = hn::Add(sum_vec, hn::LoadU(d, series + i));
  }
  double sum = hn::ReduceSum(d, sum_vec);
  for (; i < n; ++i) sum += series[i];

  const double mean = sum / static_cast<double>(n);

  // Pass 2: compute squared deviation sum
  const auto mean_vec = hn::Set(d, mean);
  auto sq_vec = hn::Zero(d);
  i = 0;
  for (; i + N <= n; i += N) {
    const auto diff = hn::Sub(hn::LoadU(d, series + i), mean_vec);
    sq_vec = hn::MulAdd(diff, diff, sq_vec);
  }
  double sq_sum = hn::ReduceSum(d, sq_vec);
  for (; i < n; ++i) {
    double diff = series[i] - mean;
    sq_sum += diff * diff;
  }

  const double stddev = std::sqrt(sq_sum / static_cast<double>(n));

  // Pass 3: normalize in place
  if (stddev > 1e-10) {
    const double inv_sd = 1.0 / stddev;
    const auto inv_sd_vec = hn::Set(d, inv_sd);
    i = 0;
    for (; i + N <= n; i += N) {
      const auto val = hn::LoadU(d, series + i);
      const auto normed = hn::Mul(hn::Sub(val, mean_vec), inv_sd_vec);
      hn::StoreU(normed, d, series + i);
    }
    for (; i < n; ++i) {
      series[i] = (series[i] - mean) * inv_sd;
    }
  } else {
    const auto zero = hn::Zero(d);
    i = 0;
    for (; i + N <= n; i += N) {
      hn::StoreU(zero, d, series + i);
    }
    for (; i < n; ++i) {
      series[i] = 0.0;
    }
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
