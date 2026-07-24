/**
 * @file gpu_fixed_band_oracle.hpp
 * @brief Independent exact oracle for cross-backend fixed-band DTW tests.
 *
 * @details This test-only oracle deliberately shares no production band-bound,
 *          rolling-buffer, CUDA, or Metal helper.  It evaluates a full dynamic
 *          programming matrix under the literal |i-j| <= band predicate.  A
 *          second arbiter enumerates every admissible monotone path without
 *          dynamic programming.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace dtwc::test::gpu_fixed_band {

inline const std::vector<double> &principal_x()
{
  static const std::vector<double> value{0.0, 1.0, 0.0, 2.0, 0.0};
  return value;
}

inline const std::vector<double> &principal_y()
{
  static const std::vector<double> value{
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0
  };
  return value;
}

inline const std::vector<double> &singleton_x()
{
  static const std::vector<double> value{0.0};
  return value;
}

inline const std::vector<double> &singleton_y()
{
  static const std::vector<double> value{1.0, 2.0, 3.0};
  return value;
}

/// Non-degenerate route-selection payload; it is never an oracle operand.
inline const std::vector<double> &filler_129()
{
  static const std::vector<double> value = [] {
    std::vector<double> series(129);
    for (std::size_t i = 0; i < series.size(); ++i) {
      const auto centred = static_cast<int>((i * 17U) % 23U) - 11;
      series[i] = static_cast<double>(centred);
    }
    return series;
  }();
  return value;
}

inline constexpr double public_no_path_sentinel =
    std::numeric_limits<double>::max();

struct LedgerRow {
  int band;
  std::size_t path_count;
  bool has_path;
  double l1;
  double squared_l2;
};

inline constexpr std::array<LedgerRow, 4> ledger{{
    {1, 0, false, 0.0, 0.0},
    {2, 696, true, 5.0, 9.0},
    {3, 1143, true, 3.0, 5.0},
    {std::numeric_limits<int>::max(), 1289, true, 3.0, 5.0}
}};

inline constexpr std::array<LedgerRow, 3> singleton_ledger{{
    {1, 0, false, 0.0, 0.0},
    {2, 1, true, 6.0, 14.0},
    {std::numeric_limits<int>::max(), 1, true, 6.0, 14.0}
}};

inline bool cell_is_allowed(
    std::size_t row, std::size_t column, int band) noexcept
{
  if (band < 0) return true;
  const auto offset =
      (row > column) ? (row - column) : (column - row);
  return offset <= static_cast<std::size_t>(band);
}

/// Full-matrix canonical DTW; mathematical no-path is positive infinity.
inline double full_matrix_oracle(
    const std::vector<double> &x,
    const std::vector<double> &y,
    int band,
    bool squared)
{
  const auto n = x.size();
  const auto m = y.size();
  const auto stride = m + 1;
  const auto infinity = std::numeric_limits<double>::infinity();
  std::vector<double> matrix((n + 1) * (m + 1), infinity);
  matrix[0] = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      if (!cell_is_allowed(i, j, band)) continue;

      auto local = std::abs(x[i] - y[j]);
      if (squared) local *= local;
      const auto diagonal = matrix[i * stride + j];
      const auto above = matrix[i * stride + (j + 1)];
      const auto left = matrix[(i + 1) * stride + j];
      matrix[(i + 1) * stride + (j + 1)] =
          local + std::min(diagonal, std::min(above, left));
    }
  }

  return matrix[n * stride + m];
}

struct ExhaustiveResult {
  std::size_t path_count{0};
  double min_l1{std::numeric_limits<double>::infinity()};
  double min_squared_l2{std::numeric_limits<double>::infinity()};
};

/// Explicit monotone-path enumeration: no DP state and no production helper.
inline ExhaustiveResult enumerate_paths(
    const std::vector<double> &x,
    const std::vector<double> &y,
    int band)
{
  ExhaustiveResult result;
  if (x.empty() || y.empty()) return result;

  const auto visit = [&](auto &&self, std::size_t i, std::size_t j,
                         double l1, double squared_l2) -> void {
    if (!cell_is_allowed(i, j, band)) return;

    const auto delta = std::abs(x[i] - y[j]);
    l1 += delta;
    squared_l2 += delta * delta;

    if (i + 1 == x.size() && j + 1 == y.size()) {
      ++result.path_count;
      result.min_l1 = std::min(result.min_l1, l1);
      result.min_squared_l2 =
          std::min(result.min_squared_l2, squared_l2);
      return;
    }

    if (i + 1 < x.size()) self(self, i + 1, j, l1, squared_l2);
    if (j + 1 < y.size()) self(self, i, j + 1, l1, squared_l2);
    if (i + 1 < x.size() && j + 1 < y.size()) {
      self(self, i + 1, j + 1, l1, squared_l2);
    }
  };

  visit(visit, 0, 0, 0.0, 0.0);
  return result;
}

} // namespace dtwc::test::gpu_fixed_band
