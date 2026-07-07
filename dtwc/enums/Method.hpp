/**
 * @file Method.hpp
 * @brief Method enum for classification method.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

namespace dtwc {

enum class Method {
  Kmedoids, //<! Kmedoids (Lloyd/PAM) heuristic classification
  MIP,      //<! Mixed integer programming — solver-backed exact (Gurobi/HiGHS, optional Benders)
  LRCore    //<! LR-core exact: Lagrangian-dual bound + reduced-cost fixing + y-branching (Phase 4). Regime: dense D held in RAM (N·N doubles is the memory budget).
};

}