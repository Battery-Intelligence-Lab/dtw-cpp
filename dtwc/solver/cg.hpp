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

inline auto cg_lp(std::vector<data_t> &xkp1, std::vector<data_t> &xk, std::vector<data_t> &zk, std::vector<data_t> &yk,
                  std::vector<data_t> &q, ConstraintOperator &op, double rho, double sigma)
{

  const auto Nx = xk.size();
  const auto N = op.N;
  const auto Nm = 2 * N * N + N + 1;

  thread_local std::vector<data_t> r_now, p_now;
  thread_local std::vector<data_t> temp_Nx(Nx); // Temporary matrices in size Nm and Nx;
  // Make sure everything is in right size.
  xkp1.resize(Nx);
  r_now.resize(Nx);
  p_now.resize(Nx);

  temp_Nx.resize(Nx);

  // Initialise xk+1 with zeros:
  std::fill_n(xkp1.begin(), Nx, 0.0);
  op.At(r_now, [rho, &zk, &yk](size_t i) { return rho * zk[i] - yk[i]; });


  for (size_t i = 0; i < Nx; i++) {
    r_now[i] += sigma * xk[i] - q[i]; // r_prev = sigma*xk - q  + op_At(rho*zk - yk,N);
    p_now[i] = r_now[i];
  }

  size_t Niter{ 10000 };
  bool flag = false;

  auto r_prev_sqr = std::inner_product(r_now.begin(), r_now.end(), r_now.begin(), 0.0);
  for (size_t i_iter = 0; i_iter < Niter; i_iter++) {
    // std::cout << "r_prev_sqr : " << r_prev_sqr << '\n';
    if (r_prev_sqr <= 1e-12) {
      flag = true;
      break;
    }

    op.V(temp_Nx, p_now, rho, sigma); // op_V(p_now,rho,sigma,N)

    const auto p_norm_sqr = std::inner_product(p_now.begin(), p_now.end(), temp_Nx.begin(), 0.0);
    const auto alpha = r_prev_sqr / p_norm_sqr;

    for (size_t i = 0; i < Nx; i++)
      xkp1[i] += alpha * p_now[i];

    double r_next_sqr = 0.0;
    for (size_t i = 0; i < Nx; i++) {
      r_now[i] -= alpha * temp_Nx[i];
      r_next_sqr += r_now[i] * r_now[i];
    }

    const auto beta = r_next_sqr / r_prev_sqr;
    r_prev_sqr = r_next_sqr;

    for (size_t i = 0; i < Nx; i++)
      p_now[i] = r_now[i] + beta * p_now[i];
  }

  if (!flag)
    std::cout << "Warning! cg_lp has reached maximum iteration!\n";
}

} // namespace dtwc::solver