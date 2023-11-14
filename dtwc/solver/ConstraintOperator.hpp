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
#include "solver_util.hpp"
#include "sparse_util.hpp"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

namespace dtwc::solver {

struct ConstraintOperator
{
  size_t N{ 0 };                          // Number of time-series data.
  std::vector<std::vector<Element>> Amat; // Each is a column
  std::vector<double> b;

  int Nb{}, Nc{}; // Number of time-series data.

  ConstraintOperator() = default;
  explicit ConstraintOperator(int Nb_, int Nc_) : Nb(Nb_), Nc(Nc_)
  {
    const auto Neq = Nb + 1;
    const auto Nineq = Nb * (Nb - 1);
    const auto Nconstraints = Neq + Nineq;

    const auto Nvar_original = Nb * Nb;
    const auto N_slack = Nineq;
    const auto Nvar = Nvar_original + N_slack; // x1--xN^2  + s_slack

    b.resize(Nconstraints);
    Amat.resize(Nvar);

    // Create b matrix:
    b[0] = Nc;
    for (int i = 0; i < Nb; ++i)
      b[i + 1] = 1;

    // Create A matrix:
    for (int i = 0; i < Nineq; i++)
      Amat[Nvar_original + i].emplace_back(Neq + i, 1.0);

    for (int i = 0; i < Nb; ++i) {
      Amat[i * (Nb + 1)].emplace_back(0, 1.0); // Sum of diagonals is Nc

      for (int j = 0; j < Nb; j++)
        Amat[Nb * i + j].emplace_back(1 + j, 1.0); // Every element belongs to one cluster.

      // ---------------
      int shift = 0;
      for (int j = 0; j < Nb; j++) {
        const int block_begin_row = Nb + 1 + (Nb - 1) * i;
        const int block_begin_col = Nb * i;
        if (i == j) {
          for (int k = 0; k < (Nb - 1); k++)
            Amat[block_begin_col + j].emplace_back(block_begin_row + k, -1.0);
          shift = 1;
        } else
          Amat[block_begin_col + j].emplace_back(block_begin_row + j - shift, 1.0);
      }
    }

    std::for_each(Amat.begin(), Amat.end(), [](auto &elemVec) {
      std::sort(elemVec.begin(), elemVec.end(), CompElementIndices{});
    });
  }

  auto get_Nx() const { return Amat.size(); }
  auto get_Nm() const { return b.size() + get_Nx(); }

  auto A(std::vector<data_t> &x_out, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be N^2;
    const auto Nout = get_Nm(); // x_out size;
    x_out.resize(Nout);

    std::fill_n(x_out.begin(), Nout, 0.0); // Zero out the vector.

    for (size_t i{}; i < x_in.size(); i++)
      for (const auto [row, val] : Amat[i])
        x_out[row] += val * x_in[i];

    std::copy(x_in.begin(), x_in.end(), x_out.begin() + b.size());
  }


  data_t A(size_t i, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be N^2;
    data_t x_out_i = 0;

    if (i >= b.size()) return x_in[i - b.size()];

    for (size_t i{}; i < x_in.size(); i++)
      for (const auto [row, val] : Amat[i])
        if (row == i)
          x_out_i += val * x_in[i];

    return x_out_i;
  }

  auto At(std::vector<data_t> &x_out, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    const auto Nout = get_Nx(); // x_out size;
    x_out.resize(Nout);
    std::fill_n(x_out.begin(), Nout, 0.0); // Zero out the vector.

    for (size_t i{}; i < x_out.size(); i++) {
      x_out[i] += x_in[i + b.size()];
      for (const auto [col, val] : Amat[i])
        x_out[i] += val * x_in[col];
    }
  }


  data_t At(size_t i, std::vector<data_t> &x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    // i/N should be size_t.
    data_t x_out_i = x_in[i + b.size()];
    for (const auto [col, val] : Amat[i])
      x_out_i += val * x_in[col];

    return x_out_i;
  }

  template <typename Tfun>
  data_t At(size_t i, Tfun &&x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    // i/N should be size_t.
    data_t x_out_i = x_in(i + b.size());
    for (const auto [col, val] : Amat[i])
      x_out_i += val * x_in(col);

    return x_out_i;
  }


  template <typename Tfun> // Takes a temporary object.
  auto At(std::vector<data_t> &x_out, Tfun &&x_in)
  {
    // x_in size is assumed to be 2*N^2 + N + 1;
    const auto Nout = get_Nx(); // x_out size;
    x_out.resize(Nout);
    std::fill_n(x_out.begin(), Nout, 0.0); // Zero out the vector.

    for (size_t i{}; i < x_out.size(); i++) {
      x_out[i] += x_in(i + b.size());
      for (const auto [col, val] : Amat[i])
        x_out[i] += val * x_in(col);
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

  void clamp(std::vector<data_t> &x_out, int Nc)
  {
    std::copy(b.begin(), b.end(), x_out.begin());

    std::transform(x_out.begin() + b.size(), x_out.end(), x_out.begin() + b.size(), [](double value) {
      return std::clamp(value, 0.0, 1.0);
    });
  }

  data_t clamp(data_t x_out_i, int Nc, size_t i)
  {
    return (i < b.size()) ? b[i] : std::clamp(x_out_i, 0.0, 1.0);
  }
};
} // namespace dtwc::solver