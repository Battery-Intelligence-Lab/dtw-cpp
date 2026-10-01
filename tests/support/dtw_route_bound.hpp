/**
 * @file dtw_route_bound.hpp
 * @brief When two routes that compute one DTW distance with different code agree.
 *
 * @details The SIMD-lanes kernel and the per-pair kernel run the same recurrence
 *          but are not the same code, and a compiler may contract `d * d + m`
 *          into one FMA in one and not in the other (GCC does by default), so
 *          their results can differ in the last bits. Nothing promises bitwise
 *          equality across routes or compilers; the clustering must not change.
 *
 *          The bound is |a - b| <= c * path_length * eps(T) * max(|a|, |b|),
 *          with c = 2, T the precision the DTW ran in and path_length at most
 *          nx + ny - 1. A cell adds a non-negative cost to the minimum of three
 *          earlier values, so a route's relative error grows at most linearly
 *          with the path. The routes share the subtraction and differ only in
 *          whether the product is rounded before the sum: at most three
 *          roundings of eps / 2 per cell, 1.5 eps, which c = 2 covers.
 *
 *          A check whose two sides run the same code stays bitwise.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace dtwc::test_support {

/// True when `a` and `b`, the DTW of series of nx and ny samples computed in
/// precision T by two routes, agree within the bound above.
template <typename T>
bool dtw_routes_agree(double a, double b, std::size_t nx, std::size_t ny)
{
  constexpr double c = 2.0;
  const double path_length = static_cast<double>(nx + ny - 1);
  return std::abs(a - b)
         <= c * path_length * static_cast<double>(std::numeric_limits<T>::epsilon())
              * std::max(std::abs(a), std::abs(b));
}

} // namespace dtwc::test_support
