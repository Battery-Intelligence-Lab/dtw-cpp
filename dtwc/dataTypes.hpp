#pragma once

#include <vector>
#include <iostream>
#include <limits>
#include <string>
#include <cmath>

namespace dtwc {

template <typename data_t>
constexpr data_t maxValue = std::numeric_limits<data_t>::max();

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


template <typename data_t>
class SkewedBandMatrix
{

private:
  using VarType = int;
  VarType m{}, ku{}, kl{};
  double m_n; // m/n
  std::vector<std::array<int, 3>> access;

  data_t outBoundsVal = maxValue<data_t>;

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

    if (val < maxValue<data_t> / 2)
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