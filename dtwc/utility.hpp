/*
 * utility.hpp
 *
 * utility functions

 * Created on: 15 Dec 2021
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"
#include "dataTypes.hpp"
#include "fileOperations.hpp"
#include "Range.hpp"
#include "parallelisation.hpp"


#include <iostream>
#include <vector>
#include <array>

#include <numeric>
#include <fstream>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <iterator>
#include <memory>
#include <tuple>
#include <iomanip>

static std::mt19937 randGenerator(5); // std::mt19937{ std::random_device{}() }

namespace dtwc {
// namespace stdr = std::ranges;
// namespace stdv = std::views;

template <typename data_t>
data_t dtwFun2(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  thread_local VecMatrix<data_t> C(x.size(), y.size()); //
  data_t z = maxValue<data_t>;


  if (&x == &y) return 0; // If they are the same data then distance is 0.

  int mx = x.size();
  int my = y.size();

  C.resize(mx, my);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((mx != 0) && (my != 0)) {
    C(0, 0) = distance(x[0], y[0]);

    for (int i = 1; i < mx; i++)
      C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]); // j = 0

    for (int j = 1; j < my; j++)
      C(0, j) = C(0, j - 1) + distance(x[0], y[j]); // i = 0


    for (int j = 1; j < my; j++) {
      for (int i = 1; i < mx; i++) {
        const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
        C(i, j) = minimum + distance(x[i], y[j]);
      }
    }

    return C(mx - 1, my - 1);
  }

  return z;
}


template <typename data_t>
data_t dtwFun_L(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  // This function uses L shaped method to compute distance but cannot backtrack.
  if (&x == &y) return 0; // If they are the same data then distance is 0.

  thread_local std::vector<data_t> short_side(10e3);

  const auto mx = x.size();
  const auto my = y.size();

  const auto &short_vec = (mx < my) ? x : y;
  const auto &long_vec = (mx < my) ? y : x;


  const auto m_short = short_vec.size();
  const auto m_long = long_vec.size();

  short_side.resize(m_short);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((m_short != 0) && (m_long != 0)) {

    short_side[0] = distance(short_vec[0], long_vec[0]);

    for (size_t i = 1; i < m_short; i++)
      short_side[i] = short_side[i - 1] + distance(short_vec[i], long_vec[0]);


    for (size_t j = 1; j < m_long; j++) {
      auto diag = short_side[0];
      short_side[0] += distance(short_vec[0], long_vec[j]);

      for (size_t i = 1; i < m_short; i++) {
        const auto next = std::min({ diag, short_side[i - 1], short_side[i] }) + distance(short_vec[i], long_vec[j]);

        diag = short_side[i];
        short_side[i] = next;
      }
    }

    return short_side[m_short - 1];
  }
  return maxValue<data_t>;
}


template <typename data_t>
data_t dtwFun_short(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  thread_local VecMatrix<data_t> C(x.size(), y.size()); //
  data_t z = maxValue<data_t>;


  if (&x == &y) return 0; // If they are the same data then distance is 0.

  int mx = x.size();
  int my = y.size();

  C.resize(mx, my);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((mx != 0) && (my != 0)) {
    C(0, 0) = distance(x[0], y[0]);

    for (int i = 1; i < mx; i++)
      C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]); // j = 0

    for (int j = 1; j < my; j++)
      C(0, j) = C(0, j - 1) + distance(x[0], y[j]); // i = 0


    for (int j = 1; j < my; j++) {
      for (int i = 1; i < mx; i++) {
        const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
        C(i, j) = minimum + distance(x[i], y[j]);
      }
    }

    int i{ mx - 1 }, j{ my - 1 };
    int ord{ 0 };
    do {
      std::array<data_t, 3> distTemp{ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) };
      ord = std::distance(distTemp.begin(), std::min_element(distTemp.begin(), distTemp.end()));

      i--;
    } while (ord == 0);

    do {
      std::array<data_t, 3> distTemp{ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) };
      ord = std::distance(distTemp.begin(), std::min_element(distTemp.begin(), distTemp.end()));

      j--;
    } while (ord == 1);

    // Backstep if we are going over the boundaries. Just not to use corners as boundary conditions.
    return C(i + 1, j + 1);
  }

  return z;
}


template <typename data_t = float>
data_t dtwFunBanded_Act_L(const std::vector<data_t> &x, const std::vector<data_t> &y, size_t band = 100)
{
  // Actual banding with skewness. New and light technique:
  // This function uses L shaped method to compute distance but cannot backtrack.
  // Be careful this is one sided band.

  band *= 2; // Since it is one sided band
  // Also puts the band to the short side. Not the long side. Long side could be beneficial but also not very good.


  if (&x == &y) return 0; // If they are the same data then distance is 0.

  const auto mx = x.size();
  const auto my = y.size();

  const auto &short_vec = (mx < my) ? x : y;
  const auto &long_vec = (mx < my) ? y : x;


  const auto m_short = short_vec.size();
  const auto m_long = long_vec.size();

  const auto band_size = std::min(band, m_short);

  thread_local std::vector<data_t> short_side(band_size);

  short_side.resize(band_size);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((m_short != 0) && (m_long != 0)) {
    const auto diff = m_short - band_size;                                            // So we need to move this much.
    const auto slope = static_cast<double>(diff) / (static_cast<double>(m_long) - 1); // This wont work with long size 1

    double acc = 0; // Accumulate slope.

    size_t shift = 0;

    short_side[0] = distance(short_vec[0], long_vec[0]);

    for (size_t i = 1; i < band_size; i++)
      short_side[i] = short_side[i - 1] + distance(short_vec[i], long_vec[0]);


    for (size_t j = 1; j < m_long; j++) {
      acc += slope;
      if (acc > 1.0) {
        acc--;
        shift++; // We need to shift one block now.

        short_side[0] = std::min({ short_side[0], short_side[1] }) + distance(short_vec[shift], long_vec[j]);

        for (size_t i = 1; i < (band_size - 1); i++)
          short_side[i] = std::min({ short_side[i - 1], short_side[i], short_side[i + 1] }) + distance(short_vec[shift + i], long_vec[j]);


        const size_t i = band_size - 1;
        short_side[i] = std::min({ short_side[i - 1], short_side[i] }) + distance(short_vec[shift + i], long_vec[j]);

      } else {
        auto diag = short_side[0];
        short_side[0] += distance(short_vec[shift], long_vec[j]);

        for (size_t i = 1; i < band_size; i++) {
          const auto next = std::min({ diag, short_side[i - 1], short_side[i] }) + distance(short_vec[shift + i], long_vec[j]);

          diag = short_side[i];
          short_side[i] = next;
        }
      }
    }
    return short_side[band_size - 1];
  }
  return maxValue<data_t>;
}


template <typename data_t = float>
data_t dtwFunBanded_Act(const std::vector<data_t> &x, const std::vector<data_t> &y, int band = 100)
{
  // Actual banding with skewness.
  static thread_local SkewedBandMatrix<data_t> C(x.size(), y.size(), band, band); //
  data_t z = maxValue<data_t>;

  const int mx = x.size();
  const int my = y.size();

  C.resize(mx, my, band, band);

  std::fill(C.CompactMat.data.begin(), C.CompactMat.data.end(), maxValue<data_t>);
  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((mx != 0) && (my != 0)) {
    auto slope = static_cast<double>(mx) / static_cast<double>(my);


    C(0, 0) = distance(x[0], y[0]);

    for (int i = 1; i < std::min(mx, band + 1); i++)
      C(i, 0) = C.at(i - 1, 0) + distance(x[i], y[0]);

    const int band_my = std::floor(band / slope);
    for (int j = 1; j < std::min(my, band_my + 1); j++)
      C(0, j) = C.at(0, j - 1) + distance(x[0], y[j]);

    for (int j = 1; j < my; j++) {
      int j_mod = std::round(j * slope);

      int i = std::max(1, j_mod - band);
      {
        const auto minimum = std::min({ C.at(i, j - 1), C.at(i - 1, j - 1) });
        C(i, j) = minimum + distance(x[i], y[j]);
      }

      for (i++; i < std::min(mx - 1, j_mod + band); i++) {
        const auto minimum = std::min({ C.at(i - 1, j), C.at(i, j - 1), C.at(i - 1, j - 1) });
        C(i, j) = minimum + distance(x[i], y[j]);
      }


      {
        const auto minimum = std::min({ C.at(i - 1, j), C.at(i - 1, j - 1) });
        C(i, j) = minimum + distance(x[i], y[j]);
      }
    }


    return C.at(mx - 1, my - 1);
  }

  return z;
}

template <typename Tfun>
void fillDistanceMatrix(Tfun &DTWdistByInd, size_t N)
{
  auto oneTask = [&, N = N](size_t i_linear) {
    size_t i{ i_linear / N }, j{ i_linear % N };
    if (i <= j)
      DTWdistByInd(i, j);
  };

  dtwc::run(oneTask, N * N);
}

struct TestNumberOfThreads
{
  TestNumberOfThreads() { std::cout << "A thread is created.\n"; }
};


}; // namespace dtwc