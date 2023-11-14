/*
 * solve_lp.hpp
 *
 * LP solution

 *  Created on: 11 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "cg.hpp"
#include "ConstraintOperator.hpp"
#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace dtwc::solver {

class LP
{
  double EPS_ADMM_FACTOR{ 1e-2 };
  double MIN_VAL_RHO{ 1e-6 };
  double MAX_VAL_RHO{ 1e6 };

  size_t N{ 0 }, Nc{ 0 };

  std::vector<data_t> vXX, vX;
  std::vector<data_t> vZ, vY; //
  std::vector<data_t> r_now;  // Residuals now.

  std::vector<data_t> q; // q vector;

  ConstraintOperator op;

public:
  size_t maxIterations{ 15000 }; // Default 4000
  size_t numItrConv{ 50 };       // Check convergence every 200 iteration.

  double epsAbs{ 1e-4 };
  double epsRel{ 1e-4 };
  double rho{ 0.1 };
  double alpha{ 1.6 };
  double delta{ 1e-6 };
  double sigma{ 1e-6 };
  double factor_rho{ 10 };
  bool adapt_rho{ true };

  auto Nx() { return op.get_Nx(); }
  auto Nm() { return op.get_Nm(); }

  void setSize(int Nb_, int Nc_)
  {
    N = Nb_;
    Nc = Nc_;

    op = ConstraintOperator(N, Nc);

    q.resize(Nx());
    vX.resize(Nx());
    vXX.resize(Nx());
    r_now.resize(Nx());

    vZ.resize(Nm());
    vY.resize(Nm());
  }

  auto &getSolution() { return vX; }
  auto &getQvec() { return q; }

  data_t cost() { return std::inner_product(q.begin(), q.end(), vX.begin(), 0.0); }

  inline bool isSolutionInteger() { return std::all_of(vX.cbegin(), vX.cend(), is_integer<data_t>); }

  ConvergenceFlag solve()
  {
    if (N == 0 || Nc == 0) {
      std::cout << "Size is not set!\n";
      return ConvergenceFlag::error_sizeNotSet;
    }

    const auto epsAdmm = std::min(epsAbs, epsRel) * EPS_ADMM_FACTOR;
    bool flag_ADMM = true;
    for (size_t i_iter = 0; i_iter < maxIterations; i_iter++) {
      op.At(r_now, [this](size_t i) { return rho * vZ[i] - vY[i]; });

      for (size_t i = 0; i != Nx(); i++)
        r_now[i] += sigma * vX[i] - q[i]; // r_prev = sigma*xk - q  + op_At(rho*zk - yk,N);

      cg_lp(vXX, r_now, op, rho, sigma);

      for (auto &vXX_i : vXX)
        vXX_i = std::max(0.0, vXX_i);


      flag_ADMM = true; // Also check ADMM convergence not to store previous X and Z values:
      for (size_t i = 0; i != Nx(); i++) {
        const auto dvX = (vXX[i] - vX[i]);
        vX[i] += alpha * dvX;
        flag_ADMM &= std::abs(dvX) < epsAdmm;
      }

      for (size_t i = 0; i != Nm(); i++) {
        const auto ZPi = vZ[i];
        const auto temp = alpha * (op.A(i, vXX) - ZPi);
        vZ[i] = op.clamp(vZ[i] + temp + vY[i] / rho, Nc, i);

        const auto dvZ = ZPi - vZ[i];
        vY[i] += rho * (temp + dvZ);
        flag_ADMM &= std::abs(dvZ) < epsAdmm;
      }

      if (flag_ADMM) return ConvergenceFlag::conv_admm;

      if (i_iter % numItrConv == 0) // Check convergence every time to time.
      {
        double normResPrim{ 0 }, normResDual{ 0 }, maxNormPrim{ 0 }, maxNormDual{ 0 };

        for (size_t i = 0; i != Nm(); i++) {
          const auto A_vX_i = op.A(i, vX);
          normResPrim = std::max(normResPrim, std::abs(A_vX_i - vZ[i]));
          maxNormPrim = std::max({ maxNormPrim, std::abs(A_vX_i), std::abs(vZ[i]) });
        }

        for (size_t i = 0; i != Nx(); i++) {
          const auto At_vY_i = op.At(i, vY);
          normResDual = std::max(normResDual, std::abs(At_vY_i + q[i]));
          maxNormDual = std::max({ maxNormDual, std::abs(At_vY_i), q[i] }); // std::abs(q[i])
        }

        std::cout << "Iter: " << i_iter << " normResPrim: "
                  << normResPrim << " normResDual: " << normResDual << '\n';

        // Adaptive rho:
        if (adapt_rho) {
          const auto numeratorVal = normResPrim * maxNormDual;
          const auto denominatorVal = normResDual * maxNormPrim;
          const auto rhoA = std::clamp(rho * std::sqrt(numeratorVal / denominatorVal), MIN_VAL_RHO, MAX_VAL_RHO);

          if ((rhoA * factor_rho < rho) || (rhoA > factor_rho * rho))
            rho = rhoA;
        }

        // Check termination:
        const auto epsPrim = epsAbs + epsRel * maxNormPrim;
        const auto epsDual = epsAbs + epsRel * maxNormDual;

        if ((normResPrim < epsPrim) && (normResDual < epsDual))
          return ConvergenceFlag::conv_problem; // Problem converged yay!
      }
    }

    return ConvergenceFlag::conv_fail;
  }


public:
  void round()
  {
    //<! Round numbers to make them integer:
    for (auto &x : vX)
      if (is_one(x))
        x = 1;
      else if (is_zero(x))
        x = 0;
  }


  // ConvergenceFlag int_solve()
  // {
  //   // solve ensuring that decision variables are integer.
  //   auto flag = solve(); // First run solve

  //   round();

  //   if (flag != ConvergenceFlag::conv_problem || isSolutionInteger())
  //     return flag; // Solve didn't converge...  or it converged and it is integer.

  //   // Solution converged but it is not integer now!
  //   // #TODO figure out why this happens for totally unimodular matrices...
  //   // Probably not exactly totally unimodular. Especially Nc constraint is dangerous.

  //   IntSolution bestSolution;
  //   bestSolution.cost = std::numeric_limits<data_t>::max();

  //   recursive_solve(bestSolution);

  //   vX = bestSolution.vX_opt;
  //   return ConvergenceFlag::conv_problem;
  // }
};


} // namespace dtwc::solver
