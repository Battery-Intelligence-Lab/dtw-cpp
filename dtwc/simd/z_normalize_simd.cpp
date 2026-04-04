/**
 * @file z_normalize_simd.cpp
 * @brief SIMD-accelerated z-normalization using Google Highway.
 *
 * @details Two-pass fused approach (König-Huygens):
 *          Pass 1: compute sum(x) and sum(x²) in a single memory sweep using
 *                  two accumulator vectors. Avoids the extra pass needed by the
 *                  naive mean → squared-deviation → normalize scheme.
 *                  Variance is recovered via: var = E[x²] - mean²  (König-Huygens).
 *                  FMA instruction (MulAdd) accumulates x² in one op.
 *          Pass 2: normalize in-place as x * inv_sd + bias (FMA-friendly form),
 *                  where bias = -mean * inv_sd is precomputed once.
 *
 *          Processes N doubles per iteration: N=4 on AVX2, N=8 on AVX-512 (ScalableTag).
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

  // Pass 2: normalize in place using FMA form: x * inv_sd + bias
  // where bias = -mean * inv_sd (precomputed once).
  // Avoids a separate Sub per element vs (x - mean) * inv_sd.
  if (stddev > 1e-10) {
    const double inv_sd    = 1.0 / stddev;
    const double bias      = -mean * inv_sd;
    const auto inv_sd_vec  = hn::Set(d, inv_sd);
    const auto bias_vec    = hn::Set(d, bias);
    i = 0;
    for (; i + N <= n; i += N) {
      const auto val    = hn::LoadU(d, series + i);
      const auto normed = hn::MulAdd(val, inv_sd_vec, bias_vec);  // val*inv_sd + bias
      hn::StoreU(normed, d, series + i);
    }
    for (; i < n; ++i)
      series[i] = series[i] * inv_sd + bias;
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
