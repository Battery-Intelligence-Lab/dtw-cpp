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

inline auto cg_lp(Eigen::VectorXd &xkp1, Eigen::VectorXd &r_now,
                  ConstraintOperator &op, double rho, double sigma)
{

  const auto Nx = op.get_Nx();

  thread_local Eigen::VectorXd p_now;
  thread_local Eigen::VectorXd temp_Nx(Nx); // Temporary matrices in size Nm and Nx;

  xkp1.setZero(); // Initialise xk+1 with zeros:

  p_now = r_now;

  size_t Niter{ 10000 };
  bool flag = false;

  auto r_prev_sqr = r_now.squaredNorm();
  for (size_t i_iter = 0; i_iter < Niter; i_iter++) {
    if (r_prev_sqr <= 1e-12) {
      flag = true;
      break;
    }

    temp_Nx = (rho * (op.A.transpose() * (op.A * p_now))).eval() + sigma * p_now; // op.V

    const auto p_norm_sqr = p_now.dot(temp_Nx);
    const auto alpha = r_prev_sqr / p_norm_sqr;

    double r_next_sqr = 0.0;
    for (size_t i = 0; i < Nx; i++) {
      r_now[i] -= alpha * temp_Nx[i];
      r_next_sqr += r_now[i] * r_now[i];
    }

    xkp1 += alpha * p_now;

    const auto beta = r_next_sqr / r_prev_sqr;
    r_prev_sqr = r_next_sqr;
    p_now = r_now + beta * p_now;
  }

  if (!flag)
    std::cout << "Warning! cg_lp has reached maximum iteration!\n";
}

} // namespace dtwc::solver