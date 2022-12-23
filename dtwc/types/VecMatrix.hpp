#pragma once

#include <vector>
#include <iostream>
#include <limits>
#include <string>
#include <cmath>

namespace dtwc {

template <typename data_t>
class VecMatrix
{
  using VarType = int;
  using data_type = data_t;
  VarType m{}, n{};

public:
  std::vector<data_t> data;
  VecMatrix() = default;
  VecMatrix(VarType m_) : m(m_), n(m_), data(m_ * m_) {} // Sequare matrix
  VecMatrix(VarType m_, VarType n_) : m(m_), n(n_), data(m_ * n_) {}
  VecMatrix(VarType m_, VarType n_, data_t x) : m(m_), n(n_), data(m_ * n_, x) {}
  VecMatrix(VarType m_, VarType n_, std::vector<data_t> &&vec) : m(m_), n(n_), data(std::move(vec)) {}

  inline void resize(VarType m_, VarType n_, data_t x = 0)
  {
    m = m_;
    n = n_;
    data.resize(m_ * n_, x);
  }

  inline auto rows() const { return m; }
  inline auto cols() const { return n; }
  inline auto size() const { return m * n; }

  inline auto &operator()(VarType i, VarType j) { return data[i + j * m]; }

  void print()
  {
    for (VarType i = 0; i < m; i++) {
      for (VarType j = 0; j < n; j++)
        std::cout << this->operator()(i, j) << "\t\t";

      std::cout << '\n';
    }
  }
};
} // namespace dtwc
