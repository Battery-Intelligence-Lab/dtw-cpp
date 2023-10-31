/*
 * sparse_util.hpp
 *
 * Helper class for sparse matrix utilities.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "solver_util.hpp"

namespace dtwc::solver {

struct Element
{
  int index;
  double value;
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


struct Coordinate
{
  int row{}, col{}; // Row and column of the value
};

struct RowMajor
{
  bool operator()(const Coordinate &c1, const Coordinate &c2) const
  {
    return (c1.col < c2.col) || (c1.col == c2.col && c1.row < c2.row);
  }
};

struct ColumnMajor
{
  bool operator()(const Coordinate &c1, const Coordinate &c2) const
  {
    return (c1.row < c2.row) || (c1.row == c2.row && c1.col < c2.col);
  }
};

} // namespace dtwc::solver