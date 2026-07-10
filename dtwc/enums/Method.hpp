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
  TADPole   //<! Density-peaks clustering with admissible LB/UB DTW pruning (Begum KDD 2015). The ONLY method that does NOT materialise the full N×N matrix: a cutoff-kernel density needs only the binary "d<dc" per pair, so pairs with LB≥dc (or UB<dc) skip the exact DTW. Result is provably identical to brute-force density-peaks. Regime: equal-length series, plain L1/SquaredL2 DTW.
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
