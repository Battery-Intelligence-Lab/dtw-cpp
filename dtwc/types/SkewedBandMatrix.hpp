#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

namespace dtwc {


template <typename data_t>
class SkewedBandMatrix
{
private:
  using VarType = int;
  VarType m{}, ku{}, kl{};
  double m_n; // m/n
  std::vector<std::array<int, 3>> access;

  data_t outBoundsVal = std::numeric_limits<data_t>::max();

public:
  VecMatrix<data_t> CompactMat;
  SkewedBandMatrix(VarType m_, VarType n_, VarType ku_, VarType kl_)
    : m(m_), ku(ku_), kl(kl_), m_n(static_cast<double>(m_) / static_cast<double>(n_)),
      CompactMat(kl_ + ku_ + 1, n_) {}

  inline auto &operator()(VarType i, VarType j)
  {
    const VarType j_mod = std::round(m_n * j);
    return CompactMat(ku + i - j_mod, j);
  }

  auto &at(VarType i, VarType j) // checked version of operator.
  {
    const VarType j_mod = std::round(m_n * j);

    const VarType lowerBound_i = std::max(0, j_mod - ku);
    const VarType upperBound_i = std::min(m - 1, j_mod + kl); // Only for square matrices otherwise false bounds.

    if (i > upperBound_i || i < lowerBound_i)
      return outBoundsVal;

    return this->operator()(i, j);
  }

  auto &aat(VarType i, VarType j) // checked version of operator. Use for assignment
  {
    const VarType j_mod = std::round(m_n * j);

    const VarType lowerBound_i = std::max(0, j_mod - ku);
    const VarType upperBound_i = std::min(m - 1, j_mod + kl); // Only for square matrices otherwise false bounds.

    if (i > upperBound_i || i < lowerBound_i)
      throw 1002;

    const auto val = this->operator()(i, j);

    if (val < std::numeric_limits<data_t>::max() / 2)
      throw 1003; // Do not assign to an already assigned place.

    return this->operator()(i, j);
  }

  void resize(VarType m_, VarType n_, VarType ku_, VarType kl_, data_t x = 0)
  {
    m = m_;
    ku = ku_;
    kl = kl_;
    m_n = static_cast<double>(m_) / static_cast<double>(n_);
    CompactMat.resize(kl_ + ku_ + 1, n_, x);
  }

  void print() { CompactMat.print(); }
};

} // namespace dtwc