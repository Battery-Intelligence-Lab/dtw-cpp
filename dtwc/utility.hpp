// Vk 2021.12.15

#pragma once

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
#include <thread>
#include <iterator>
#include <memory>
#include <ranges>
#include <execution>
#include <tuple>
#include <iomanip>

#include "settings.hpp"
#include "dataTypes.hpp"
#include "fileOperations.hpp"

//#include <Eigen/Dense>

static std::mt19937 randGenerator(5); // std::mt19937{ std::random_device{}() }


namespace dtwc {
namespace stdr = std::ranges;
namespace stdv = std::views;
namespace ex = std::execution;


template <typename Tfun>
void run(Tfun task_indv, int i_end, unsigned int numMaxParallelWorkers = settings::numMaxParallelWorkers)
{

  auto task_par = [&](int i_begin, int i_end, int Nth) {
    while (i_begin < i_end) {
      task_indv(i_begin);
      i_begin += Nth;
    }
  };

  if constexpr (settings::isParallel) {
    if (numMaxParallelWorkers <= 1) {
      task_par(0, i_end, 1);
    } else {
      const unsigned int N_th_max = std::min(numMaxParallelWorkers, std::thread::hardware_concurrency());
      std::vector<std::thread> threads;
      threads.reserve(N_th_max);

      for (unsigned int i_begin = 0; i_begin < N_th_max; i_begin++) // indices for the threads
      {
        // Multi threaded simul:

        threads.emplace_back(task_par, i_begin, i_end, N_th_max);
      }

      for (auto &th : threads) {
        if (th.joinable())
          th.join();
      }
    }
  } else {
    task_par(0, i_end, 1);
  }
}


template <typename Tdata, typename Tsequence>
void updateDBA(std::vector<Tdata> &mean, const std::vector<Tsequence> &sequences)
{
  std::vector<Tdata> newMean(mean.size());
  std::vector<unsigned short> Nmean(mean.size());
}


template <typename Tdata>
Tdata dtwFun2(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
{
  thread_local VecMatrix<Tdata> C(x.size(), y.size()); //
  Tdata z = maxValue<Tdata>;


  if (&x == &y) return 0; // If they are the same data then distance is 0.

  int mx = x.size();
  int my = y.size();

  C.resize(mx, my);

  auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

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


template <typename Tdata>
Tdata dtwFun_L(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
{
  // This function uses L shaped method to compute distance but cannot backtrack.
  if (&x == &y) return 0; // If they are the same data then distance is 0.

  thread_local std::vector<Tdata> short_side(10e3);

  const auto mx = x.size();
  const auto my = y.size();

  const auto &short_vec = (mx < my) ? x : y;
  const auto &long_vec = (mx < my) ? y : x;


  const auto m_short = short_vec.size();
  const auto m_long = long_vec.size();

  short_side.resize(m_short);

  auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

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
  return maxValue<Tdata>;
}


template <typename Tdata>
Tdata dtwFun_short(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
{
  thread_local VecMatrix<Tdata> C(x.size(), y.size()); //
  Tdata z = maxValue<Tdata>;


  if (&x == &y) return 0; // If they are the same data then distance is 0.

  int mx = x.size();
  int my = y.size();

  C.resize(mx, my);

  auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

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
      std::array<Tdata, 3> distTemp{ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) };
      ord = std::distance(distTemp.begin(), std::min_element(distTemp.begin(), distTemp.end()));

      i--;
    } while (ord == 0);

    do {
      std::array<Tdata, 3> distTemp{ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) };
      ord = std::distance(distTemp.begin(), std::min_element(distTemp.begin(), distTemp.end()));

      j--;
    } while (ord == 1);

    // Backstep if we are going over the boundaries. Just not to use corners as boundary conditions.
    return C(i + 1, j + 1);
  }

  return z;
}


template <typename Tdata = float>
Tdata dtwFunBanded_Act(const std::vector<Tdata> &x, const std::vector<Tdata> &y, int band = 100)
{
  // Actual banding with skewness.
  static thread_local SkewedBandMatrix<Tdata> C(x.size(), y.size(), band, band); //
  Tdata z = maxValue<Tdata>;

  const int mx = x.size();
  const int my = y.size();

  C.resize(mx, my, band, band);

  std::fill(C.CompactMat.data.begin(), C.CompactMat.data.end(), maxValue<Tdata>);
  auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

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


void fillDistanceMatrix(auto &DTWdistByInd, size_t N)
{
  auto distanceAllTask = [&](int i_p) {
    for (size_t i = 0; i <= i_p; i++)
      DTWdistByInd(i_p, i);

    auto i_p_p = N - i_p - 1;
    for (size_t i = 0; i <= i_p_p; i++)
      DTWdistByInd(i_p_p, i);
  };


  const size_t N_2 = (N + 1) / 2;

  dtwc::run(distanceAllTask, N_2);
}


void fillDistanceMatrix_new(auto &DTWdistByInd, size_t N)
{
  auto distanceAllTask = [&](int i_p) {
    for (int i = 0; i <= i_p; i++)
      DTWdistByInd(i_p, i);

    auto i_p_p = N - i_p - 1;
    for (int i = 0; i <= i_p_p; i++)
      DTWdistByInd(i_p_p, i);
  };


  const int N_2 = (N + 1) / 2;

  // auto range = stdv::iota(0, N_2);

  std::vector<size_t> range(N_2);
  std::iota(range.begin(), range.end(), 0);

  std::for_each(ex::par_unseq, std::begin(range), std::end(range), distanceAllTask);


  // dtwc::run(distanceAllTask, N_2);
}

}; // namespace dtwc