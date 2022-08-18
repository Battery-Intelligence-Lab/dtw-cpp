#pragma once

#include <vector>
#include <iostream>
#include <limits>
#include <string>

namespace dtwc {

template <typename Tdata>
constexpr Tdata maxValue = std::numeric_limits<Tdata>::max();

template <typename Tdata>
class VecMatrix
{
  using VarType = int;
  using data_type = Tdata;
  VarType m{}, n{};

public:
  std::vector<Tdata> data;
  VecMatrix(VarType m_) : m(m_), n(m_), data(m_ * m_) {} // Sequare matrix
  VecMatrix(VarType m_, VarType n_) : m(m_), n(n_), data(m_ * n_) {}
  VecMatrix(VarType m_, VarType n_, Tdata x) : m(m_), n(n_), data(m_ * n_, x) {}

  inline void resize(VarType m_, VarType n_, Tdata x = 0)
  {
    m = m_;
    n = n_;
    data.resize(m_ * n_, x);
  }

  inline auto rows() const { return m; }
  inline auto cols() const { return n; }
  inline auto size() const { return m * n; }

  inline auto &operator()(VarType i, VarType j)
  {
    return data[i + j * m];
  }

  void print()
  {
    for (VarType i = 0; i < m; i++) {
      for (VarType j = 0; j < n; j++)
        std::cout << this->operator()(i, j) << "\t\t";

      std::cout << '\n';
    }
  }

  // void initialise_from_file(std::string path)
  // {

  //     auto p = readFile<data_type>(path);

  //     if ((m * n) != p.size())
  //         std::cout << "Warning! Given file and sizes are not compatible.\n";

  //     std::swap(data, p);

  // }
};


// template <typename Tdata>
// class VecMatrix2
// {
//     using VarType = int;
//     VarType m{}, n{};
// public:

//     std::unique_ptr<Tdata[]> data_ptr{};

//     VecMatrix2(VarType m_, VarType n_)
//         : m(m_), n(n_), data_ptr(new Tdata[m_ * n_]) {}

//     inline void resize(VarType m_, VarType n_, Tdata x = 0)
//     {
//         if (m * n < m_ * n_) // Slower since we allocate exactly and vector allocates more than needed.
//             data_ptr.reset(new Tdata[static_cast<int>(m_ * n_)]);

//         m = m_;
//         n = n_;
//     }

//     inline auto rows() { return m; }
//     inline auto cols() { return n; }

//     inline auto& operator()(VarType i, VarType j)
//     {
//         return data_ptr[i + j * m];
//     }

//     void print()
//     {
//         for (VarType i = 0; i < m; i++)
//         {
//             for (VarType j = 0; j < n; j++)
//                 std::cout << this->operator()(i, j) << ' ';

//             std::cout << '\n';
//         }
//     }

// };

template <typename Tdata>
class BandMatrix
{
public:
  VecMatrix<Tdata> CompactMat;

private:
  using VarType = int;
  VarType m{}, ku{}, kl{};

  Tdata fixedVal = maxValue<Tdata>;

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


  void resize(VarType m_, VarType n_, VarType ku_, VarType kl_, Tdata x = 0)
  {
    m = m_;
    ku = ku_;
    kl = kl_;
    CompactMat.resize(kl_ + ku_ + 1, n_, x);
  }

  void print()
  {
    CompactMat.print();
  }
};


template <typename Tdata>
class SkewedBandMatrix
{

private:
  using VarType = int;
  VarType m{}, ku{}, kl{};
  double m_n; // m/n
  std::vector<std::array<int, 3>> access;

  Tdata outBoundsVal = maxValue<Tdata>;

public:
  VecMatrix<Tdata> CompactMat;
  SkewedBandMatrix(VarType m_, VarType n_, VarType ku_, VarType kl_) : m(m_), ku(ku_), kl(kl_), m_n(static_cast<double>(m_) / static_cast<double>(n_)),
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

    if (val < maxValue<Tdata> / 2)
      throw 1003; // Do not assign to an already assigned place.

    return this->operator()(i, j);
  }


  void resize(VarType m_, VarType n_, VarType ku_, VarType kl_, Tdata x = 0)
  {
    m = m_;
    ku = ku_;
    kl = kl_;
    m_n = static_cast<double>(m_) / static_cast<double>(n_);
    CompactMat.resize(kl_ + ku_ + 1, n_, x);
  }

  void print()
  {
    CompactMat.print();
  }
};

} // namespace dtwc
