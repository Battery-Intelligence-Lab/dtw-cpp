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
#include "../dataTypes.hpp"


#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace dtwc::solver {

enum class ConvergenceFlag {
  conv_problem = 0, //<! Problem converged into feasible solution
  conv_admm = 1,    //<! ADMM converged, problem not
  conv_fail = 2     //<! No convergence
};

class LP
{
  double EPS_ADMM_FACTOR{ 1e-2 };
  double MIN_VAL_RHO{ 1e-3 };
  double MAX_VAL_RHO{ 1e6 };

public:
  size_t maxIterations{ 15000 }; // Default 4000
  size_t numItrConv{ 50 };       // Check convergence every 200 iteration.

  double epsAbs{ 1e-4 };
  double epsRel{ 1e-4 };
  double rho{ 1 };
  double alpha{ 1.6 };
  double delta{ 1e-6 };
  double sigma{ 1e-6 };
  double factor_rho{ 10 };
  bool adapt_rho{ true };

  ConvergenceFlag solve(std::vector<data_t> &vX, std::vector<data_t> &q, int N, int Nc)
  {
    ConstraintOperator op{ N };

    const auto Nx = op.get_Nx();
    const auto Nm = op.get_Nm();


    const auto epsAdmm = std::min(epsAbs, epsRel) * EPS_ADMM_FACTOR;

    thread_local std::vector<data_t> vXX{ vX };
    thread_local std::vector<data_t> vXP;                              //<! Previous iteration of vX
    thread_local std::vector<data_t> vZ(Nm), vY(Nm), vZZ(Nm), vZP(Nm); // #TODO, vZZ last N+1 term is known and constant.

    vXP.resize(Nx);
    vXX.resize(Nx);
    vZ.resize(Nm);
    vY.resize(Nm);
    vZZ.resize(Nm);
    vZP.resize(Nm);

    auto rhoA = rho; // adaptive rho

    for (size_t i_iter = 0; i_iter < maxIterations; i_iter++) {

      if (adapt_rho && ((rhoA * factor_rho < rho) || (rhoA > factor_rho * rho)))
        rho = rhoA;


      cg_lp(vXX, vX, vZ, vY, q, op, rho, sigma);
      op.A(vZZ, vXX);

      std::copy(vX.begin(), vX.end(), vXP.begin()); // vXP(:) = vX;
      std::copy(vZ.begin(), vZ.end(), vZP.begin()); // vZP(:) = vZ;

      for (size_t i = 0; i < Nx; i++)
        vX[i] = alpha * vXX[i] + (1.0 - alpha) * vX[i];

      for (size_t i = 0; i < Nm; i++)
        vZ[i] = alpha * vZZ[i] + (1.0 - alpha) * vZ[i] + (1 / rho) * vY[i];

      op.clamp(vZ, Nc);

      for (size_t i = 0; i < Nm; i++)
        vY[i] += rho * (alpha * vZZ[i] + (1.0 - alpha) * vZP[i] - vZ[i]);


      thread_local std::vector<data_t> temp_Nm, temp_Nx; // Temporary matrices.
      temp_Nm.resize(Nm);
      temp_Nx.resize(Nx);

      if (i_iter % numItrConv == 0) // Check convergence every time to time.
      {
        double normResPrim{ 0 }, normResDual{ 0 }, maxNormPrim{ 0 }, maxNormDual{ 0 };

        op.A(temp_Nm, vX);
        for (size_t i = 0; i != Nm; i++) {
          normResPrim = std::max(normResPrim, std::abs(temp_Nm[i] - vZ[i]));
          maxNormPrim = std::max({ maxNormPrim, std::abs(temp_Nm[i]), std::abs(vZ[i]) });
        }

        op.At(temp_Nx, vY);
        for (size_t i = 0; i != Nx; i++) {
          normResDual = std::max(normResDual, std::abs(temp_Nx[i] + q[i]));
          maxNormDual = std::max({ maxNormDual, std::abs(temp_Nx[i]), std::abs(q[i]) });
        }

        std::cout << "Iter: " << i_iter << " normResPrim: " << normResPrim << " normResDual: " << normResDual << '\n';

        // Adaptive rho:
        if (adapt_rho) {
          const auto numeratorVal = normResPrim * maxNormDual;
          const auto denominatorVal = normResDual * maxNormPrim;
          rhoA = std::clamp(rho * std::sqrt(numeratorVal / denominatorVal), MIN_VAL_RHO, MAX_VAL_RHO);
        }

        // Check termination:
        const auto epsPrim = epsAbs + epsRel * maxNormPrim;
        const auto epsDual = epsAbs + epsRel * maxNormDual;

        if ((normResPrim < epsPrim) && (normResDual < epsDual))
          return ConvergenceFlag::conv_problem; // Problem converged yay!

        bool flag_ADMM = true;

        for (size_t i = 0; i < Nx; i++)
          if (std::abs(vX[i] - vXP[i]) > epsAdmm) {
            flag_ADMM = false;
            break;
          }

        if (flag_ADMM)
          for (size_t i = 0; i < Nm; i++)
            if (std::abs(vZ[i] - vZP[i]) > epsAdmm) {
              flag_ADMM = false;
              break;
            }

        if (flag_ADMM) return ConvergenceFlag::conv_admm;
      }
    }

    return ConvergenceFlag::conv_fail;
  }
};


} // namespace dtwc::solver
