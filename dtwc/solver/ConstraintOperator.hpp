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

namespace dtwc::solver {

struct ConstraintOperator
{
  size_t N{}; // Number of time-series data.

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
  }
};
} // namespace dtwc::solver