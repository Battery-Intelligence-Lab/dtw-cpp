/*
 * cg.hpp
 *
 * Conjugate gradients method

 *  Created on: 10 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "ConstraintOperator.hpp"
#include "../settings.hpp"
#include "../utility.hpp"


#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

namespace dtwc::solver {

auto cg_lp(std::vector<data_t> &xkp1, std::vector<data_t> &xk, std::vector<data_t> &zk, std::vector<data_t> &yk,
           std::vector<data_t> &q, ConstraintOperator &op, double rho, double sigma)
{

  const auto Nx = xk.size();
  const auto N = op.N;
  const auto Nm = 2 * N * N + N + 1;

  thread_local std::vector<data_t> r_prev, r_next, p_now;
  thread_local std::vector<data_t> temp_Nm(Nm), temp_Nx(Nx); // Temporary matrices in size Nm and Nx;
  // Make sure everything is in right size.
  xkp1.resize(Nx);
  r_prev.resize(Nx);
  r_next.resize(Nx);
  p_now.resize(Nx);

  temp_Nm.resize(Nm);
  temp_Nx.resize(Nx);

  // Initialise xk+1 with zeros:
  std::fill_n(xkp1.begin(), Nx, 0.0);

  for (size_t i = 0; i < Nm; i++)
    temp_Nm[i] = rho * zk[i] - yk[i]; // assuming x0 = 0; r0 = b;

  op.At(r_prev, temp_Nm);


  for (size_t i = 0; i < Nx; i++) {
    r_prev[i] += sigma * xk[i] - q[i]; // r_prev = sigma*xk - q  + op_At(rho*zk - yk,N);
    p_now[i] = r_prev[i];
  }

  size_t Niter{ 500 };
  bool flag = false;

  for (size_t i_iter = 0; i_iter < Niter; i_iter++) {
    const auto r_prev_sqr = std::inner_product(r_prev.begin(), r_prev.end(), r_prev.begin(), 0.0);

    if (r_prev_sqr <= 1e-28) {
      flag = true;
      break;
    }

    op.V(temp_Nx, p_now, rho, sigma); // op_V(p_now,rho,sigma,N)

    const auto p_norm_sqr = std::inner_product(p_now.begin(), p_now.end(), temp_Nx.begin(), 0.0);
    const auto alpha = r_prev_sqr / p_norm_sqr;


    for (size_t i = 0; i < Nx; i++)
      xkp1[i] += alpha * p_now[i];

    for (size_t i = 0; i < Nx; i++)
      r_next[i] = r_prev[i] - alpha * temp_Nx[i];

    const auto r_next_sqr = std::inner_product(r_next.begin(), r_next.end(), r_next.begin(), 0.0);
    const auto beta = r_next_sqr / r_prev_sqr;

    for (size_t i = 0; i < Nx; i++)
      p_now[i] = r_next[i] + alpha * p_now[i];


    std::copy(r_next.begin(), r_next.end(), r_prev.begin()); // r_prev = r_next;
  }

  if (!flag)
    std::cout << "Warning! cg_lp has reached maximum iteration!\n";
}

} // namespace dtwc::solver