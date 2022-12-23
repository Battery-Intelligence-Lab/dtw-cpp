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
#include "solver_util.hpp"


#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace dtwc::solver {

class LP
{
  double EPS_ADMM_FACTOR{ 1e-2 };
  double MIN_VAL_RHO{ 1e-3 };
  double MAX_VAL_RHO{ 1e6 };

  size_t N{ 0 }, Nc{ 0 };

  std::vector<data_t> vXX, vX;
  std::vector<data_t> vXP;              //<! Previous iteration of vX
  std::vector<data_t> vZ, vY, vZZ, vZP; // #TODO, vZZ last N+1 term is known and constant.

  std::vector<data_t> q; // q vector;

  ConstraintOperator op{ 0 };

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

  auto Nx() { return op.get_Nx(); }
  auto Nm() { return op.get_Nm(); }

  void setSize(int Nb_, int Nc_)
  {
    N = Nb_;
    Nc = Nc_;

    op = ConstraintOperator(N);

    q.resize(Nx());
    vX.resize(Nx());
    vXP.resize(Nx());
    vXX.resize(Nx());
    vZ.resize(Nm());
    vY.resize(Nm());
    vZZ.resize(Nm());
    vZP.resize(Nm());
  }

  auto &getSolution() { return vX; }
  auto &getQvec() { return q; }

  data_t cost() { return std::inner_product(q.begin(), q.end(), vX.begin(), 0.0); }

  bool isSolutionInteger()
  {
    return std::all_of(vX.cbegin(), vX.cend(), is_integer<data_t>);
  }

  ConvergenceFlag solve()
  {
    if (N == 0 || Nc == 0) {
      std::cout << "Size is not set!\n";
      return ConvergenceFlag::error_sizeNotSet;
    }
    const auto epsAdmm = std::min(epsAbs, epsRel) * EPS_ADMM_FACTOR;
    auto rhoA = rho; // adaptive rho

    for (size_t i_iter = 0; i_iter < maxIterations; i_iter++) {

      if (adapt_rho && ((rhoA * factor_rho < rho) || (rhoA > factor_rho * rho)))
        rho = rhoA;

      cg_lp(vXX, vX, vZ, vY, q, op, rho, sigma);
      op.A(vZZ, vXX);

      std::copy(vX.begin(), vX.end(), vXP.begin()); // vXP(:) = vX;
      std::copy(vZ.begin(), vZ.end(), vZP.begin()); // vZP(:) = vZ;

      for (size_t i = 0; i != vXX.size(); i++)
        vX[i] = alpha * vXX[i] + (1.0 - alpha) * vX[i];

      for (size_t i = 0; i != vZZ.size(); i++)
        vZ[i] = alpha * vZZ[i] + (1.0 - alpha) * vZ[i] + (1.0 / rho) * vY[i];

      op.clamp(vZ, Nc);

      for (size_t i = 0; i < vZZ.size(); i++)
        vY[i] += rho * (alpha * vZZ[i] + (1.0 - alpha) * vZP[i] - vZ[i]);


      thread_local std::vector<data_t> temp_Nm, temp_Nx; // Temporary matrices.
      temp_Nm.resize(Nm());
      temp_Nx.resize(Nx());

      if (i_iter % numItrConv == 0) // Check convergence every time to time.
      {
        double normResPrim{ 0 }, normResDual{ 0 }, maxNormPrim{ 0 }, maxNormDual{ 0 };

        op.A(temp_Nm, vX);
        for (size_t i = 0; i != Nm(); i++) {
          normResPrim = std::max(normResPrim, std::abs(temp_Nm[i] - vZ[i]));
          maxNormPrim = std::max({ maxNormPrim, std::abs(temp_Nm[i]), std::abs(vZ[i]) });
        }

        op.At(temp_Nx, vY);
        for (size_t i = 0; i != Nx(); i++) {
          normResDual = std::max(normResDual, std::abs(temp_Nx[i] + q[i]));
          maxNormDual = std::max({ maxNormDual, std::abs(temp_Nx[i]), q[i] }); // std::abs(q[i])
        }

        std::cout << "Iter: " << i_iter << " normResPrim: "
                  << normResPrim << " normResDual: " << normResDual << '\n';

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

        for (size_t i = 0; i != vX.size(); i++)
          if (std::abs(vX[i] - vXP[i]) > epsAdmm) {
            flag_ADMM = false;
            break;
          }

        if (flag_ADMM)
          for (size_t i = 0; i != vZ.size(); i++)
            if (std::abs(vZ[i] - vZP[i]) > epsAdmm) {
              flag_ADMM = false;
              break;
            }

        if (flag_ADMM) return ConvergenceFlag::conv_admm;
      }
    }

    return ConvergenceFlag::conv_fail;
  }


private:
  void recursive_solve(IntSolution &bestSolution, size_t i_pos, size_t Nc_remaining_now)
  {

    if (i_pos == N) {
      std::cerr << "No more possibilities but integer solver failed!\n";
      throw 10101010;
    }

    std::vector<ind_t> possibilities;
    for (size_t i = i_pos * N; i < ((i_pos + 1) * N); i++)
      if (vX[i] > 1e-4)
        possibilities.push_back(i);


    if (possibilities.empty()) {
      recursive_solve(bestSolution, i_pos + 1, Nc_remaining_now);
      return;
    }


    for (size_t i = 0; i < possibilities.size(); i++) {
      // Only i'th element is one others are zero.

      if (possibilities[i] % (N + 1) == 0 && Nc_remaining_now == 0) // It is a diagonal.
        continue;


      for (size_t j = 0; j < possibilities.size(); j++) {
        if (i == j)
          op.fixed_variables.push_back({ possibilities[j], 1 });
        else
          op.fixed_variables.push_back({ possibilities[j], 0 });
      }

      solve();
      const auto cost_now = cost();

      if (cost_now < bestSolution.cost)
        if (isSolutionInteger()) {
          bestSolution.cost = cost_now;
          bestSolution.fix_var = op.fixed_variables;
          bestSolution.vX_opt = vX;
        } else {
          if (possibilities[i] % (N + 1) == 0) // It is a diagonal.
            recursive_solve(bestSolution, i_pos + 1, Nc_remaining_now - 1);
          else
            recursive_solve(bestSolution, i_pos + 1, Nc_remaining_now);
        }

      for (size_t j = 0; j < possibilities.size(); j++)
        op.fixed_variables.pop_back(); // Remove the variables added by this step!
    }
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


  ConvergenceFlag int_solve()
  {
    // solve ensuring that decision variables are integer.
    auto flag = solve(); // First run solve

    if (flag != ConvergenceFlag::conv_problem || isSolutionInteger())
      return flag; // Solve didn't converge...  or it converged and it is integer.

    // Solution converged but it is not integer now!
    // #TODO figure out why this happens for totally unimodular matrices...
    // Probably not exactly totally unimodular. Especially Nc constraint is dangerous.

    auto Nc_remaining = Nc;
    for (size_t i = 0; i < N * N; i += N + 1)
      if (is_one(vX[i]))
        Nc_remaining--; // Okay this one is probably 1.


    IntSolution bestSolution;
    bestSolution.cost = std::numeric_limits<data_t>::max();

    size_t i_pos = 0;
    recursive_solve(bestSolution, i_pos, Nc_remaining);

    vX = bestSolution.vX_opt;
    return ConvergenceFlag::conv_problem;
  }
};


} // namespace dtwc::solver
