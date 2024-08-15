/**
 * @file element_types.hpp
 * @brief Helper classes for sparse matrix utilities.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 29 Oct 2023
 */

#pragma once

#include "types_util.hpp"

namespace dtwc::solver {

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