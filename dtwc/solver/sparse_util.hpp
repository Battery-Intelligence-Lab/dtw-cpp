/*
 * sparse_util.hpp
 *
 * Helper class for sparse matrix utilities.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

namespace dtwc::solver {

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

}