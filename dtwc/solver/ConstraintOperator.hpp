/*
 * op_A.hpp
 *
 * operators for A matrix

 *  Created on: 10 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "../settings.hpp"
#include "../utility.hpp"


#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

namespace dtwc::solver {

struct ConstraintOperator
{
  size_t N{ 0 }; // Number of time-series data.
  std::vector<std::array<size_t, 2>> fixed_variables;

  ConstraintOperator() = default;
  explicit ConstraintOperator(size_t N_) : N(N_) {}


  auto get_Nm() const { return 2 * N * N + N + 1; }
  auto get_Nx() const { return N * N; }

  auto A(std::vector<data_t> &x_out, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be N^2;
    const auto Nout = get_Nm(); // x_out size;
    x_out.resize(Nout);

    data_t sum = 0;
    size_t i_sum = 0;
    for (size_t i = 0; i < N * N; i++) {
      x_out[i] = x_in[i];
      x_out[N * N + i] = -x_in[i] + x_in[(i % N) * (N + 1)];

      sum += x_in[i];

      if (((i + 1) % N) == 0) {
        x_out[2 * N * N + i_sum] = sum;
        sum = 0;
        i_sum++;
      }
    }

    sum = 0;
    for (size_t i = 0; i < N * N; i += N + 1)
      sum += x_in[i];

    x_out[2 * N * N + N] = sum;
  }


  double A(size_t i, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be N^2;
    const auto Nout = get_Nm(); // x_out size;

    // Slower if used in V.
    if (i < N * N)
      return x_in[i];
    else if (i < 2 * N * N)
      return -x_in[i - N * N] + x_in[(i % N) * (N + 1)];
    else if (i < 2 * N * N + N) {
      const auto i_basic = i - (2 * N * N);
      return std::accumulate(x_in.begin() + i_basic * N, x_in.begin() + (i_basic + 1) * N, 0.0);
    } else if (i == (2 * N * N + N)) {
      data_t sum = 0;
      for (size_t j = 0; j < N * N; j += N + 1) sum += x_in[j];
      return sum;
    }

    throw 12345;
  }

  auto At(std::vector<data_t> &x_out, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    const auto Nout = get_Nx(); // x_out size;
    x_out.resize(Nout);

    size_t i_repeat = 0;
    for (size_t i = 0; i < N * N; i++) {
      if (i % N == 0 && i != 0) i_repeat++;
      x_out[i] = x_in[i] - x_in[N * N + i] + x_in[2 * N * N + i_repeat];
    }

    for (size_t i = 0; i < N; i++) {
      data_t sum = 0;
      for (size_t j = 0; j < N * N; j += N)
        sum += x_in[N * N + i + j];

      x_out[(i % N) * (N + 1)] += x_in[2 * N * N + N] + sum;
    }
  }


  data_t At(size_t i, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    // i/N should be size_t.
    data_t x_out_i = x_in[i] - x_in[N * N + i] + x_in[2 * N * N + i / N];

    if (i % (N + 1) == 0) {
      x_out_i += x_in[2 * N * N + N];

      for (size_t j = 0; j < N * N; j += N)
        x_out_i += x_in[N * N + i / (N + 1) + j];
    }

    return x_out_i;
  }

  template <typename Tfun>
  data_t At(size_t i, Tfun &&x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    // i/N should be size_t.
    data_t x_out_i = x_in(i) - x_in(N * N + i) + x_in(2 * N * N + i / N);

    if (i % (N + 1) == 0) {
      x_out_i += x_in(2 * N * N + N);

      for (size_t j = 0; j < N * N; j += N)
        x_out_i += x_in(N * N + i / (N + 1) + j);
    }

    return x_out_i;
  }


  // auto AtA(std::vector<data_t> &x_out, std::vector<data_t> &x_in)
  // {


  //   if (i < N * N)
  //     return x_in[i];
  //   else if (i < 2 * N * N)
  //     return -x_in[i - N * N] + x_in[(i % N) * (N + 1)];
  //   else if (i < 2 * N * N + N) {
  //     const auto i_basic = i - (2 * N * N);
  //     return std::accumulate(x_in.begin() + i_basic * N, x_in.begin() + (i_basic + 1) * N, 0.0);
  //   } else if (i == (2 * N * N + N)) {
  //     data_t sum = 0;
  //     for (size_t j = 0; j < N * N; j += N + 1) sum += x_in[j];
  //     return sum;
  //   }

  //   - x_in[N * N + i]

  //    x_out.resize(Nout);

  //   size_t i_repeat = 0;
  //   for (size_t i = 0; i < N * N; i++) {
  //     if (i % N == 0 && i != 0) i_repeat++;
  //     x_out[i] = x_in[i] +x_in[i] - x_in[(i % N) * (N + 1)] + std::accumulate(x_in.begin() + i_repeat * N, x_in.begin() + (i_repeat + 1) * N, 0.0);;
  //   }

  //   for (size_t i = 0; i < N; i++) {
  //     data_t sum = 0;
  //     for (size_t j = 0; j < N * N; j += N)
  //       sum += x_in[N * N + i + j];

  //     x_out[(i % N) * (N + 1)] += x_in[2 * N * N + N] + sum;
  //   }
  // }


  template <typename Tfun> // Takes a temporary object.
  auto At(std::vector<data_t> &x_out, Tfun &&x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    const auto Nout = get_Nx(); // x_out size;
    x_out.resize(Nout);

    size_t i_repeat = 0;
    for (size_t i = 0; i < N * N; i++) {
      if (i % N == 0 && i != 0) i_repeat++;
      x_out[i] = x_in(i) - x_in(N * N + i) + x_in(2 * N * N + i_repeat);
    }

    for (size_t i = 0; i < N; i++) {
      data_t sum = 0;
      for (size_t j = 0; j < N * N; j += N)
        sum += x_in(N * N + i + j);

      x_out[(i % N) * (N + 1)] += x_in(2 * N * N + N) + sum;
    }
  }

  auto V(std::vector<data_t> &x_out, std::vector<data_t> &x_in, double rho, double sigma)
  {
    // x_in size is assumed to be N^2;
    const auto Nout = get_Nx();   // x_out size;
    const auto Ninter = get_Nm(); // intermediate size.

    thread_local std::vector<data_t> x_inter(Ninter); // Intermediate state.

    A(x_inter, x_in);
    At(x_out, x_inter);

    for (size_t i = 0; i < Nout; i++)
      x_out[i] = rho * x_out[i] + sigma * x_in[i];
  }

  auto clamp(std::vector<data_t> &x_out, int Nc)
  {
    for (size_t i = 0; i != 2 * N * N; i++)
      x_out[i] = std::clamp(x_out[i], 0.0, 1.0);


    // Equality constraints:
    for (size_t i = 0; i != N; i++)
      x_out[2 * N * N + i] = 1.0;

    x_out[2 * N * N + N] = Nc;

    for (auto ind_val : fixed_variables) {
      // This loop is for fixing any variables to a certain values.
      const auto ind = ind_val[0];
      const auto val = ind_val[1];

      x_out[ind] = val;
    }
  }

  data_t clamp(data_t x_out_i, int Nc, size_t i)
  {
    if (i < 2 * N * N)
      x_out_i = std::clamp(x_out_i, 0.0, 1.0);
    else if (i < (2 * N * N + N)) // Equality constraints:
      return 1.0;
    else
      return Nc;

    for (auto ind_val : fixed_variables) {
      // This loop is for fixing any variables to a certain values.
      if (ind_val[0] == i) {
        x_out_i = ind_val[1];
        break;
      }
    }
    return x_out_i;
  }
};
} // namespace dtwc::solver