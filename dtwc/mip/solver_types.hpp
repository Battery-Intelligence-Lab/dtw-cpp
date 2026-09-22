/**
 * @file solver_types.hpp
 * @brief Sparse-matrix helpers and tolerances for the MIP solvers.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 29 Oct 2023
 *
 * @details These live in `mip/` because `mip/` is their only consumer. They
 * used to sit in `dtwc/types/` as `element_types.hpp` + `types_util.hpp`, which
 * `utility.hpp` pulls in, so every translation unit reaching the umbrella
 * header compiled them — 97 of 297 in the default configuration (C-12). No
 * forwarding header is left at the old paths: `dtwc/types/` ranks `base` and
 * `mip/` is the top layer, so a forwarder there would be an upward include
 * edge, which is exactly the coupling this move removes.
 */

#pragma once

#include <cmath>

namespace dtwc::solver {

constexpr double epsilon = 1e-8;

bool inline isAround(double x, double y = 0.0, double tolerance = epsilon)
{
  return std::abs(x - y) <= tolerance;
}
bool inline isFractional(double x) { return std::abs(x - std::round(x)) > epsilon; }

struct Element
{
  int index{};
  double value{};

  Element() = default;
  Element(int index_, double value_) : index(index_), value(value_) {}
};

struct Coordinate
{
  int row{}, col{}; // Row and column of the value
};

struct Triplet
{
  int row{}, col{}; // Row and column of the value
  double val{};

  Triplet() = default;
  Triplet(int row_, int col_, double val_) : row(row_), col(col_), val(val_) {}
};

struct CompElementIndices
{
  bool operator()(const Element &c1, const Element &c2) const
  {
    return (c1.index < c2.index);
  }
};

struct CompElementValuesAndIndices
{
  // To move the "zero" values to the end of the vector to be removed.
  bool operator()(const Element &c1, const Element &c2) const
  {
    const auto c1_zero = isAround(c1.value);
    const auto c2_zero = isAround(c2.value);

    return (c1_zero < c2_zero) || (c1_zero == c2_zero && c1.index < c2.index);
  }
};

struct RowMajor
{
  template <typename T>
  bool operator()(const T &c1, const T &c2) const
  {
    return (c1.col < c2.col) || (c1.col == c2.col && c1.row < c2.row);
  }
};

struct ColumnMajor
{
  template <typename T>
  bool operator()(const T &c1, const T &c2) const
  {
    return (c1.row < c2.row) || (c1.row == c2.row && c1.col < c2.col);
  }
};

} // namespace dtwc::solver
