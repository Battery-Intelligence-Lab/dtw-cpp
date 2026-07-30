/**
 * @file Method.hpp
 * @brief Method enum for classification method.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

#include "../error.hpp"

namespace dtwc {

enum class Method {
  Kmedoids, //<! Kmedoids (Lloyd/PAM) heuristic classification
  MIP,      //<! Mixed integer programming — solver-backed exact (Gurobi/HiGHS, optional Benders)
  LRCore,   //<! LR-core exact: Lagrangian-dual bound + reduced-cost fixing + y-branching (Phase 4). Regime: dense D held in RAM (N·N doubles is the memory budget).
  /**
   * Density peaks with conditionally admissible LB/UB pruning.
   * Exact-arithmetic identity covers finite, nonempty, equal-length Standard-L1
   * series with integer-representable lengths; floating thresholds remain D17
   * and empty series remain F48.
   */
  TADPole
};

inline void validate_method(Method value)
{
  switch (value) {
  case Method::Kmedoids:
  case Method::MIP:
  case Method::LRCore:
  case Method::TADPole:
    return;
  }
  throw InvalidInput("Invalid Method value.");
}

}
