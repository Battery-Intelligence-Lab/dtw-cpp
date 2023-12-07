/*
 * element_types.hpp
 *
 * Helper class for sparse matrix utilities.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
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
  bool operator()(const auto &c1, const auto &c2) const
  {
    return (c1.col < c2.col) || (c1.col == c2.col && c1.row < c2.row);
  }
};

struct ColumnMajor
{
  bool operator()(const auto &c1, const auto &c2) const
  {
    return (c1.row < c2.row) || (c1.row == c2.row && c1.col < c2.col);
  }
};

} // namespace dtwc::solver