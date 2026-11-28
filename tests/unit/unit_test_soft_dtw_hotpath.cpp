/**
 * @file unit_test_soft_dtw_hotpath.cpp
 * @brief Scratch reuse of the Soft-DTW gradient.
 *
 * @author Volkan Kumtepeli
 * @date 10 Jul 2026
 */

#include <soft_dtw.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <vector>

TEST_CASE("warmed Soft-DTW gradient reuses its large DP allocations",
          "[soft_dtw][gradient][allocation]")
{
  std::vector<double> x(257);
  std::vector<double> y(255);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<double>((i * 7) % 29) / 29.0;
  for (std::size_t i = 0; i < y.size(); ++i)
    y[i] = static_cast<double>((i * 11) % 31) / 31.0;

  // The gradient keeps its forward and backward matrices in the thread's two
  // buffers, grown to the largest pair seen and never shrunk: after a warm-up
  // call, one of the same size and one of a smaller size keep their storage.
  const auto warm = dtwc::soft_dtw_gradient<double>(x, y, 0.7);
  REQUIRE(warm.size() == x.size());

  auto [c_buf, e_buf] = dtwc::detail::soft_dtw_grad_bufs<double>();
  const double *const c_ptr = c_buf.data();
  const double *const e_ptr = e_buf.data();
  const std::size_t c_cap = c_buf.capacity();
  const std::size_t e_cap = e_buf.capacity();
  REQUIRE(c_cap >= x.size() * y.size());
  REQUIRE(e_cap >= x.size() * y.size());

  const auto gradient = dtwc::soft_dtw_gradient<double>(x, y, 0.7);
  REQUIRE(gradient == warm);
  CHECK(c_buf.data() == c_ptr);
  CHECK(e_buf.data() == e_ptr);
  CHECK(c_buf.capacity() == c_cap);
  CHECK(e_buf.capacity() == e_cap);

  const std::span<const double> x_short{ x.data(), 64 }, y_short{ y.data(), 64 };
  (void)dtwc::soft_dtw_gradient<double>(x_short, y_short, 0.7);
  CHECK(c_buf.data() == c_ptr);
  CHECK(e_buf.data() == e_ptr);
  CHECK(c_buf.capacity() == c_cap);
  CHECK(e_buf.capacity() == e_cap);
}
