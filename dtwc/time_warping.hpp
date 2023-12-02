/*
 * time_warping.hpp
 *
 * This file contains time warping functions

 *  Created on: 08 Dec 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"
#include "utility.hpp"
#include "fileOperations.hpp"
#include "parallelisation.hpp"

#include <vector>

namespace dtwc {

template <typename data_t>
data_t dtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  thread_local VecMatrix<data_t> C(x.size(), y.size()); //
  data_t z = maxValue<data_t>;

  if (&x == &y) return 0; // If they are the same data then distance is 0.

  const int mx = x.size();
  const int my = y.size();

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
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  // This function uses L shaped method to compute distance but cannot backtrack.
  if (&x == &y) return 0; // If they are the same data then distance is 0.

  thread_local std::vector<data_t> short_side(data_t(10e3));

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

template <typename data_t = float>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y, int band = 100)
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
    const auto slope = static_cast<double>(mx) / static_cast<double>(my);

    C(0, 0) = distance(x[0], y[0]);

    for (int i = 1; i < std::min(mx, band + 1); i++)
      C(i, 0) = C.at(i - 1, 0) + distance(x[i], y[0]);

    const int band_my = std::floor(band / slope);
    for (int j = 1; j < std::min(my, band_my + 1); j++)
      C(0, j) = C.at(0, j - 1) + distance(x[0], y[j]);

    for (int j = 1; j < my; j++) {
      const int j_mod = std::round(j * slope);

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
} // namespace dtwc