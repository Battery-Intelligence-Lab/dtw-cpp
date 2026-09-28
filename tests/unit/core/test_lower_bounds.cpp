/**
 * @file test_lower_bounds.cpp
 * @brief Envelopes and LB_Keogh beyond the sizes the exhaustive derivation reaches.
 *
 * @details test_lb_keogh_derivation enumerates every warping path for series of
 *          length <= 5, the strongest oracle for LB_Keogh <= DTW. This file
 *          covers what that enumeration cannot: the O(n) ring-buffer envelope
 *          against a naive window scan at lengths up to 100 and bands up to 50,
 *          hand-computed bound values, admissibility against an independent
 *          full-matrix DP on longer random series, and the prefix / ragged
 *          envelope contract of the Envelope entry points.
 */

#include <core/lower_bound_impl.hpp>
#include <warping.hpp>

#include "../gpu_fixed_band_oracle.hpp"
#include "../../support/deterministic_series.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <span>
#include <sstream>
#include <vector>

namespace {

using Series = std::vector<double>;

Series random_series(std::mt19937 &rng, std::size_t n, double lo = -10.0, double hi = 10.0)
{
  std::uniform_real_distribution<double> dist(lo, hi);
  Series s(n);
  for (auto &v : s) v = dist(rng);
  return s;
}

/// Naive O(n*w) window scan; a negative band is the global window.
void naive_envelopes(const Series &s, int band, Series &upper, Series &lower)
{
  const std::size_t n = s.size();
  const std::size_t w = band < 0 ? n : static_cast<std::size_t>(band);
  upper.resize(n);
  lower.resize(n);
  for (std::size_t p = 0; p < n; ++p) {
    const std::size_t lo = p > w ? p - w : 0;
    const std::size_t hi = std::min(p + w + 1, n);
    upper[p] = *std::max_element(s.begin() + static_cast<std::ptrdiff_t>(lo),
                                 s.begin() + static_cast<std::ptrdiff_t>(hi));
    lower[p] = *std::min_element(s.begin() + static_cast<std::ptrdiff_t>(lo),
                                 s.begin() + static_cast<std::ptrdiff_t>(hi));
  }
}

double lb_keogh_of(const Series &query, const Series &candidate, int band)
{
  Series upper, lower;
  dtwc::core::compute_envelopes(candidate, band, upper, lower);
  return dtwc::core::lb_keogh(query, upper, lower);
}

} // namespace

TEST_CASE("compute_envelopes on a hand-computed series", "[lower_bounds][envelopes]")
{
  // band 1: window [i-1, i+1] over {1, 3, 2, 4, 1}.
  const Series s = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  Series upper, lower;
  dtwc::core::compute_envelopes(s, 1, upper, lower);
  CHECK(upper == Series{ 3.0, 3.0, 4.0, 4.0, 4.0 });
  CHECK(lower == Series{ 1.0, 1.0, 2.0, 1.0, 1.0 });

  // A negative band is full DTW: every position holds the global min and max.
  const auto env = dtwc::core::compute_envelope(Series{ 3.0, 1.0, 4.0, 1.0, 5.0 }, -1);
  CHECK(env.upper == Series(5, 5.0));
  CHECK(env.lower == Series(5, 1.0));
}

TEST_CASE("compute_envelopes equals a naive window scan", "[lower_bounds][envelopes]")
{
  // The ring buffer holds w + 2 indices; overflow would corrupt a live index
  // only at lengths and bands beyond the derivation's n <= 5.
  std::mt19937 rng(31415); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::size_t checked = 0;
  for (const std::size_t n : { 1u, 2u, 10u, 100u }) {
    const auto s = random_series(rng, n);
    for (const int band : { -1, 0, 1, 5, 50, static_cast<int>(n), static_cast<int>(n + 5) }) {
      Series eu, el, u, l;
      naive_envelopes(s, band, eu, el);
      dtwc::core::compute_envelopes(s, band, u, l);
      INFO("n=" << n << " band=" << band);
      REQUIRE(u == eu);
      REQUIRE(l == el);
      ++checked;
    }
  }
  for (int trial = 0; trial < 50; ++trial) {
    const std::size_t n = 20 + rng() % 80;
    const int band = static_cast<int>(rng() % 51);
    const auto s = random_series(rng, n);
    Series eu, el, u, l;
    naive_envelopes(s, band, eu, el);
    dtwc::core::compute_envelopes(s, band, u, l);
    INFO("trial=" << trial << " n=" << n << " band=" << band);
    REQUIRE(u == eu);
    REQUIRE(l == el);
    ++checked;
  }
  REQUIRE(checked == 78);
}

TEST_CASE("lb_keogh on hand-computed envelopes", "[lower_bounds][lb_keogh]")
{
  // Envelope [1, 5]: query {0, 3, 7} is 1 below, inside, 2 above.
  const double q3[] = { 0.0, 3.0, 7.0 };
  const double u3[] = { 5.0, 5.0, 5.0 };
  const double l3[] = { 1.0, 1.0, 1.0 };
  CHECK(dtwc::core::lb_keogh(q3, std::size_t{ 3 }, u3, l3) == 3.0);

  // Odd lengths exercise the vector loop's tail.
  const std::size_t n = 37;
  const Series upper(n, 10.0), lower(n, 0.0);
  CHECK(dtwc::core::lb_keogh(Series(n, 100.0), upper, lower) == 37.0 * 90.0);
  CHECK(dtwc::core::lb_keogh(Series(n, -5.0), upper, lower) == 37.0 * 5.0);

  // A series lies inside its own envelope, so the bound is zero.
  const Series s = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  CHECK(lb_keogh_of(s, s, 1) == 0.0);
  CHECK(lb_keogh_of(Series(50, 0.0), Series(50, 0.0), 5) == 0.0);
}

TEST_CASE("lb_keogh <= banded DTW on long random series", "[lower_bounds][lb_keogh][property]")
{
  std::mt19937 rng(12345); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_int_distribution<std::size_t> length(30, 150);
  for (int p = 0; p < 30; ++p) {
    const std::size_t n = length(rng);
    const auto x = random_series(rng, n);
    const auto y = random_series(rng, n);
    for (const int band : { 0, 5, 50 }) {
      const auto ex = dtwc::core::compute_envelope(x, band);
      const auto ey = dtwc::core::compute_envelope(y, band);
      const double xy = dtwc::core::lb_keogh(x, ey);
      const double yx = dtwc::core::lb_keogh(y, ex);
      const double sym = dtwc::core::lb_keogh_symmetric(x, ex, y, ey);
      const double dtw = dtwc::dtwBanded<double>(x, y, band);
      INFO("pair " << p << " n=" << n << " band=" << band);
      REQUIRE(sym == std::max(xy, yx));
      // At band 0 the bound equals the diagonal DTW in exact arithmetic; the two
      // sums round differently, so allow a relative 1e-12.
      REQUIRE(sym <= dtw * (1.0 + 1e-12));
    }
    // A wider window can only lower the bound.
    REQUIRE(lb_keogh_of(x, y, 15) <= lb_keogh_of(x, y, 3));
  }
}

TEST_CASE("lb_keogh at extreme magnitudes", "[lower_bounds][lb_keogh]")
{
  const Series big = { 1e15, -1e15, 1e15, -1e15, 1e15 };
  const Series tiny = { 1e-15, -1e-15, 1e-15, -1e-15, 1e-15 };
  const double lb = lb_keogh_of(big, tiny, 2);
  REQUIRE(lb >= 0.0);
  REQUIRE(lb <= dtwc::dtwBanded<double>(big, tiny, 2) * (1.0 + 1e-12));
}

TEST_CASE("lb_keogh takes the covered envelope prefix", "[lower_bounds][envelopes][sizes]")
{
  // B = {4, 3, 2, 1}, band 1: upper {4, 4, 3, 2}, lower {3, 2, 1, 1}.
  const Series A = { 1.0, 2.0, 3.0, 4.0 };
  const Series B = { 4.0, 3.0, 2.0, 1.0 };

  auto short_upper = dtwc::core::compute_envelope(B, 1);
  short_upper.upper.pop_back();
  // n = min(4, 3) = 3: 1 is 2 below lower[0]; 2 and 3 are inside.
  CHECK(dtwc::core::lb_keogh(std::span<const double>(A), short_upper) == 2.0);
  CHECK(dtwc::core::lb_keogh(A, short_upper) == 2.0);

  // A lower array shorter than the prefix would be read out of bounds, so the
  // trivially admissible 0 is returned.
  auto ragged = dtwc::core::compute_envelope(B, 1);
  ragged.lower.pop_back();
  CHECK(dtwc::core::lb_keogh(std::span<const double>(A), ragged) == 0.0);
  CHECK(dtwc::core::lb_keogh(A, ragged) == 0.0);
}

TEST_CASE("symmetric LB_Keogh <= exact DTW, full and banded, unequal lengths",
          "[lower_bounds][lb_keogh][property]")
{
  // Oracle: the plain full-matrix DP under the literal |i-j| <= band predicate
  // (every cell for band < 0); it shares no production DTW or envelope code.
  // For full DTW a radius-zero envelope overshoots; the witness count proves the
  // inputs can see that defect, so a green run is not vacuous.
  namespace oracle = dtwc::test::gpu_fixed_band;
  using dtwc::test_support::benchmark_series;

  std::size_t checked = 0, violations = 0, radius_zero_witnesses = 0;
  std::ostringstream first_violation;
  for (const int band : { -1, 0, 2 }) {
    for (unsigned seed = 1; seed <= 200; ++seed) {
      const auto x = benchmark_series(1 + seed % 9, seed);
      const auto y = benchmark_series(1 + (seed / 9) % 9, seed + 7919U);
      const double exact = oracle::full_matrix_oracle(x, y, band, false);
      const std::size_t n = std::min(x.size(), y.size());
      const auto bound = [&](int radius) {
        Series ux, lx, uy, ly;
        dtwc::core::compute_envelopes(x, radius, ux, lx);
        dtwc::core::compute_envelopes(y, radius, uy, ly);
        return std::max(dtwc::core::lb_keogh(x.data(), n, uy.data(), ly.data()),
                        dtwc::core::lb_keogh(y.data(), n, ux.data(), lx.data()));
      };
      const double lb = bound(band);
      ++checked;
      if (!(lb <= exact * (1.0 + 1e-12) + 1e-12) && violations++ == 0)
        first_violation << "band=" << band << " seed=" << seed << " bound=" << lb
                        << " exact=" << exact;
      if (band < 0 && bound(0) > exact) ++radius_zero_witnesses;
    }
  }
  INFO("first violation: " << first_violation.str());
  CHECK(violations == 0);
  REQUIRE(checked == 600);
  REQUIRE(radius_zero_witnesses > 0);
}
