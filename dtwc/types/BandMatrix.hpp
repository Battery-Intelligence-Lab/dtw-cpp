#pragma once

#include "types_util.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

namespace dtwc {

template <typename data_t>
class BandMatrix
{
public:
  VecMatrix<data_t> CompactMat;

private:
  using VarType = int;
  VarType m{}, ku{}, kl{};

  data_t fixedVal = maxValue<data_t>;

public:
  BandMatrix(VarType m_, VarType n_, VarType ku_, VarType kl_) : CompactMat(kl_ + ku_ + 1, n_), m(m_), ku(ku_), kl(kl_) {}

  inline auto &operator()(VarType i, VarType j) { return CompactMat(ku + i - j, j); }

  auto &at(VarType i, VarType j) // checked version of operator.
  {

    const VarType lowerBound_i = std::max(0, j - ku);
    const VarType upperBound_i = std::min(m - 1, j + kl); // Only for square matrices otherwise false bounds.

    if (i > upperBound_i || i < lowerBound_i)
      return fixedVal;

    return CompactMat(ku + i - j, j);
  }


  void resize(VarType m_, VarType n_, VarType ku_, VarType kl_, data_t x = 0)
  {
    m = m_;
    ku = ku_;
    kl = kl_;
    CompactMat.resize(kl_ + ku_ + 1, n_, x);
  }

  void print() { CompactMat.print(); }
};

}