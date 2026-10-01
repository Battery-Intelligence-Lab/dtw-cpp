/**
 * @file Method.hpp
 * @brief Method enum for classification method.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

#include "../base/names.hpp"

namespace dtwc {

/// The clustering method, as Problem::cluster(), dtwc::run and `--method` take it.
/// v1.0.0's values keep their numbers: new values go at the end.
enum class Method {
  Kmedoids, //<! Kmedoids (Lloyd) heuristic classification
  MIP,      //<! Mixed integer programming — solver-backed exact (Gurobi/HiGHS)
  LRCore,   //<! LR-core exact: Lagrangian-dual bound + reduced-cost fixing + y-branching (Phase 4). Regime: dense D held in RAM (N·N doubles is the memory budget).
  /**
   * Density peaks with conditionally admissible LB/UB pruning.
   * Exact-arithmetic identity covers finite, nonempty, equal-length Standard-L1
   * series with integer-representable lengths; floating thresholds remain D17
   * and empty series remain F48.
   */
  TADPole,
  Auto,        //<! pam on a GPU and for up to 5000 series on the CPU, clara above
  PAM,         //<! FasterPAM over the full distance matrix
  OneBatch,    //<! OneBatchPAM: one N x m batch table, no N x N matrix
  CLARA,       //<! FastCLARA: FasterPAM on subsamples, every series assigned
  Hierarchical //<! Agglomerative clustering cut at k clusters
};

/// The one table of method names, in the order error messages list them.
inline constexpr Name<Method> method_names[]{
  { "auto", Method::Auto },
  { "pam", Method::PAM },
  { "onebatch", Method::OneBatch },
  { "obp", Method::OneBatch },
  { "clara", Method::CLARA },
  { "kmedoids", Method::Kmedoids },
  { "mip", Method::MIP },
  { "lrcore", Method::LRCore },
  { "lr", Method::LRCore },
  { "tadpole", Method::TADPole },
  { "hierarchical", Method::Hierarchical },
  { "hclust", Method::Hierarchical },
};

}
