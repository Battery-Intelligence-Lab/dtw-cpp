/*
 * SparseMatrix.hpp
 *
 * A map based implementation of a Sparse Matrix.

 *  Created on: 22 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "sparse_util.hpp"

#include <map>
#include <algorithm>
#include <iostream>
#include <span>
#include <iomanip>

namespace dtwc::solver {

template <typename Major = ColumnMajor>
struct SparseMatrix
{
  std::map<Coordinate, double, Major> data;
  int m{}, n{}; // rows and columns

  SparseMatrix() = default;
  SparseMatrix(int m_, int n_) : m{ m_ }, n{ n_ } {}

  double operator()(int x, int y) const
  {
    auto it = data.find(Coordinate{ x, y });
    return (it != data.end()) ? (it->second) : 0.0;
  }

  double &operator()(int x, int y) { return data[Coordinate{ x, y }]; }

  void compress()
  {
    std::erase_if(data, [](const auto &item) {
      auto const &[key, value] = item;
      return isAround(value, 0.0);
    });
  }

  void print() const
  {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (j != 0) std::cout << ',';
        std::cout << std::setw(3) << (*this)(i, j);
      }
      std::cout << '\n';
    }
  }

  int rows() const { return m; }
  int cols() const { return n; }

  void expand(int m_, int n_)
  {
    m = m_;
    n = n_;
  }

  auto row_begin(int i) { return data.lower_bound({ i, 0 }); }
  auto row_end(int i) { return data.lower_bound({ i + 1, 0 }); }
};
} // namespace dtwc::solver