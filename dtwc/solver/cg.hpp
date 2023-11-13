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

inline auto cg_lp(std::vector<data_t> &xkp1, std::vector<data_t> &r_now,
                  ConstraintOperator &op, double rho, double sigma)
{

  const auto Nx = op.get_Nx();

  thread_local std::vector<data_t> p_now;
  thread_local std::vector<data_t> temp_Nx(Nx); // Temporary matrices in size Nm and Nx;
  // Make sure everything is in right size.
  xkp1.resize(Nx);
  temp_Nx.resize(Nx);

  // Initialise xk+1 with zeros:
  std::fill_n(xkp1.begin(), Nx, 0.0);

  p_now = r_now;

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

    double r_next_sqr = 0.0;
    for (size_t i = 0; i < Nx; i++) {
      r_now[i] -= alpha * temp_Nx[i];
      r_next_sqr += r_now[i] * r_now[i];
    }

    for (size_t i = 0; i < Nx; i++)
      xkp1[i] += alpha * p_now[i];

    const auto beta = r_next_sqr / r_prev_sqr;
    r_prev_sqr = r_next_sqr;

    for (size_t i = 0; i < Nx; i++)
      p_now[i] = r_now[i] + beta * p_now[i];
  }

  if (!flag)
    std::cout << "Warning! cg_lp has reached maximum iteration!\n";
}

} // namespace dtwc::solver