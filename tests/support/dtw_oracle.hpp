/**
 * @file dtw_oracle.hpp
 * @brief Independent DTW oracle: plain O(nm) full-matrix dynamic programs written
 *        from the definitions, sharing nothing with dtwc/.
 *
 * @details Definitions (c = local cost, D = accumulated cost, window |i-j| <= band):
 *   - Standard, DDTW, WDTW: D(i,j) = c(i,j) + min(D(i-1,j-1), D(i-1,j), D(i,j-1)),
 *     docs/derivations/01-dtw-recurrence-sakoe-chiba.md. DDTW runs it on the
 *     derivative series, WDTW scales c(i,j) by w(|i-j|) = 1 / (1 + exp(-g (|i-j| - m/2))),
 *     m = max(nx, ny) - 1 (docs/content/method/dtw-variants.md).
 *   - ADTW: the two non-diagonal predecessors carry the penalty.
 *   - Soft-DTW: min becomes the gamma-softmin (softmin below), with the L1 cost, as DTWC++ defines it.
 *   - MSM (Stefan et al., IEEE TKDE 2013) and TWE (Marteau, IEEE TPAMI 2009): their
 *     own recurrences, written out below.
 *   - Multivariate: dependent mode sums the per-channel costs inside one cell;
 *     independent mode sums one univariate DTW per channel
 *     (docs/content/method/multivariate.md). Series are interleaved, x[t*ndim + c].
 * SoftDTW, MSM and TWE take no band, and the multivariate modes apply to the
 * variants DTWC++ gives them; the caller asks only for what is defined.
 * A pair with no admissible path (an empty series, or a band narrower than the
 * length difference) is kOracleNoPath, as in the derivation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace dtwc::test_support {

enum class OracleVariant { Standard, DDTW, WDTW, ADTW, SoftDTW, MSM, TWE };
enum class OracleMetric { L1, L2, SquaredL2 };

/// What to compute; parameters the variant does not use are ignored.
struct OracleSpec
{
  OracleVariant variant = OracleVariant::Standard;
  OracleMetric metric = OracleMetric::L1;
  int band = -1; ///< Sakoe-Chiba half-width in steps; negative means none
  std::size_t ndim = 1;
  bool independent = false;
  double g = 0.05, penalty = 1.0, gamma = 1.0, c = 1.0, nu = 0.001, lambda = 1.0;
};

inline constexpr double kOracleNoPath = std::numeric_limits<double>::infinity();

namespace oracle_detail {

inline constexpr double kInf = kOracleNoPath;

/// Cost of pairing one step of x with one step of y (ndim interleaved channels).
inline double step_cost(OracleMetric metric, const double *a, const double *b, std::size_t ndim)
{
  double sum = 0;
  for (std::size_t d = 0; d < ndim; ++d) {
    const double diff = a[d] - b[d];
    sum += metric == OracleMetric::L1 ? std::abs(diff) : diff * diff;
  }
  return metric == OracleMetric::L2 ? std::sqrt(sum) : sum;
}

/// D(i,j) = step(cost(i,j), D(i-1,j-1), D(i-1,j), D(i,j-1)) over the window. Row 0 and
/// column 0 of the padded table are +inf except D(0,0) = 0 ("before the first cell"),
/// so D(1,1) = cost(1,1) + min(0, inf, inf).
template <class Cost, class Step>
double full_matrix_dp(std::size_t nx, std::size_t ny, int band, Cost cost, Step step)
{
  if (nx == 0 || ny == 0) return kInf;
  std::vector<double> table((nx + 1) * (ny + 1), kInf);
  const auto at = [&](std::size_t i, std::size_t j) -> double & { return table[i * (ny + 1) + j]; };
  at(0, 0) = 0;
  for (std::size_t i = 1; i <= nx; ++i)
    for (std::size_t j = 1; j <= ny; ++j) {
      const std::size_t offset = i > j ? i - j : j - i;
      if (band >= 0 && offset > static_cast<std::size_t>(band)) continue;
      at(i, j) = step(cost(i - 1, j - 1), at(i - 1, j - 1), at(i - 1, j), at(i, j - 1));
    }
  return at(nx, ny);
}

/// Keogh and Pazzani's slope per channel: forward and backward differences at the
/// ends, ((x[i]-x[i-1]) + (x[i+1]-x[i-1])/2)/2 inside. A one-step series has slope 0.
inline std::vector<double> derivative(std::span<const double> x, std::size_t ndim)
{
  const std::size_t n = x.size() / ndim;
  std::vector<double> out(x.size(), 0.0);
  for (std::size_t c = 0; c < ndim && n > 1; ++c) {
    const auto v = [&](std::size_t t) { return x[t * ndim + c]; };
    for (std::size_t t = 0; t < n; ++t)
      out[t * ndim + c] = t == 0       ? v(1) - v(0)
                          : t + 1 == n ? v(n - 1) - v(n - 2)
                                       : ((v(t) - v(t - 1)) + (v(t + 1) - v(t - 1)) / 2) / 2;
  }
  return out;
}

inline double softmin(double a, double b, double c, double gamma)
{
  const double lowest = std::min({ a, b, c });
  if (std::isinf(lowest)) return kInf;
  double sum = 0;
  for (const double v : { a, b, c }) sum += std::exp(-(v - lowest) / gamma); // exp(-inf) = 0
  return lowest - gamma * std::log(sum);
}

/// MSM: M(0,0) = |x0-y0|; a first-row or first-column cell and each of the two
/// non-diagonal moves of an inner cell pay split_merge(moved, neighbour, other).
inline double msm(std::span<const double> x, std::span<const double> y, double c)
{
  const std::size_t nx = x.size(), ny = y.size();
  if (nx == 0 || ny == 0) return kInf;
  const auto split_merge = [c](double moved, double a, double b) {
    const bool between = (a <= moved && moved <= b) || (b <= moved && moved <= a);
    return between ? c : c + std::min(std::abs(moved - a), std::abs(moved - b));
  };
  std::vector<double> m(nx * ny);
  const auto at = [&](std::size_t i, std::size_t j) -> double & { return m[i * ny + j]; };
  at(0, 0) = std::abs(x[0] - y[0]);
  for (std::size_t i = 1; i < nx; ++i) at(i, 0) = at(i - 1, 0) + split_merge(x[i], x[i - 1], y[0]);
  for (std::size_t j = 1; j < ny; ++j) at(0, j) = at(0, j - 1) + split_merge(y[j], x[0], y[j - 1]);
  for (std::size_t i = 1; i < nx; ++i)
    for (std::size_t j = 1; j < ny; ++j)
      at(i, j) = std::min({ at(i - 1, j - 1) + std::abs(x[i] - y[j]),
                            at(i - 1, j) + split_merge(x[i], x[i - 1], y[j]),
                            at(i, j - 1) + split_merge(y[j], x[i], y[j - 1]) });
  return at(nx - 1, ny - 1);
}

/// TWE: both series gain a leading 0 (time stamps 0..n); delete and match as in Marteau.
inline double twe(std::span<const double> x, std::span<const double> y, double nu, double lambda)
{
  const std::size_t nx = x.size(), ny = y.size();
  if (nx == 0 || ny == 0) return kInf;
  const auto xp = [&](std::size_t i) { return i == 0 ? 0.0 : x[i - 1]; };
  const auto yp = [&](std::size_t j) { return j == 0 ? 0.0 : y[j - 1]; };
  std::vector<double> d((nx + 1) * (ny + 1), kInf);
  const auto at = [&](std::size_t i, std::size_t j) -> double & { return d[i * (ny + 1) + j]; };
  at(0, 0) = 0;
  for (std::size_t i = 1; i <= nx; ++i)
    for (std::size_t j = 1; j <= ny; ++j) {
      const double gap = static_cast<double>(i > j ? i - j : j - i);
      at(i, j) = std::min({ at(i - 1, j) + std::abs(xp(i - 1) - xp(i)) + nu + lambda,
                            at(i, j - 1) + std::abs(yp(j - 1) - yp(j)) + nu + lambda,
                            at(i - 1, j - 1) + std::abs(xp(i) - yp(j)) + std::abs(xp(i - 1) - yp(j - 1))
                              + 2 * nu * gap });
    }
  return at(nx, ny);
}

} // namespace oracle_detail

/// The distance between x and y (interleaved, spec.ndim channels) under `spec`.
inline double dtw_oracle(const OracleSpec &spec, std::span<const double> x, std::span<const double> y)
{
  using namespace oracle_detail;
  using V = OracleVariant;
  const std::size_t nx = x.size() / spec.ndim, ny = y.size() / spec.ndim;
  const auto plain = [](double c, double d, double u, double l) { return c + std::min({ d, u, l }); };
  const auto local = [&](std::span<const double> a, std::span<const double> b) {
    return [&, a, b](std::size_t i, std::size_t j) {
      return step_cost(spec.metric, &a[i * spec.ndim], &b[j * spec.ndim], spec.ndim);
    };
  };

  if (spec.independent) {
    double total = 0;
    for (std::size_t c = 0; c < spec.ndim; ++c) {
      std::vector<double> xc(nx), yc(ny);
      for (std::size_t t = 0; t < nx; ++t) xc[t] = x[t * spec.ndim + c];
      for (std::size_t t = 0; t < ny; ++t) yc[t] = y[t * spec.ndim + c];
      OracleSpec channel = spec;
      channel.independent = false;
      channel.ndim = 1;
      total += dtw_oracle(channel, xc, yc);
    }
    return total;
  }
  switch (spec.variant) {
  case V::Standard: return full_matrix_dp(nx, ny, spec.band, local(x, y), plain);
  case V::DDTW: {
    const auto dx = derivative(x, spec.ndim), dy = derivative(y, spec.ndim);
    return full_matrix_dp(nx, ny, spec.band, local(dx, dy), plain);
  }
  case V::WDTW: {
    const double m = static_cast<double>(std::max(nx, ny)) - 1;
    const auto cost = local(x, y);
    const auto weighted = [&](std::size_t i, std::size_t j) {
      const double offset = static_cast<double>(i > j ? i - j : j - i);
      const double weight = 1 / (1 + std::exp(-spec.g * (offset - m / 2)));
      return weight * cost(i, j);
    };
    return full_matrix_dp(nx, ny, spec.band, weighted, plain);
  }
  case V::ADTW:
    return full_matrix_dp(nx, ny, spec.band, local(x, y), [&](double c, double d, double u, double l) {
      return c + std::min({ d, u + spec.penalty, l + spec.penalty });
    });
  case V::SoftDTW:
    return full_matrix_dp(nx, ny, -1, local(x, y), [&](double c, double d, double u, double l) {
      return c + softmin(d, u, l, spec.gamma);
    });
  case V::MSM: return msm(x, y, spec.c);
  case V::TWE: return twe(x, y, spec.nu, spec.lambda);
  }
  return kOracleNoPath;
}

} // namespace dtwc::test_support
