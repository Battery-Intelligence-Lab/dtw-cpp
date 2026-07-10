/**
 * @file LowerBoundStrategy.hpp
 * @brief LowerBoundStrategy enum — selects which lower bound(s) feed the
 *        pruned distance matrix path.
 *
 * @details Only the pruned CPU path honours this setting. GPU backends
 *          (CUDA/Metal) carry their own `use_lb_keogh` switch on the options
 *          struct. Semantics:
 *            - Auto:     Kim+Keogh cascade when applicable (current default).
 *            - None:     Disable both; behaves like BruteForce within Pruned.
 *            - Kim:      LB_Kim only (cheapest, O(1) per pair).
 *            - Keogh:    LB_Keogh only (requires band >= 0).
 *            - KimKeogh: Cascade Kim -> Keogh (tightest non-GPU bound).
 *            - Enhanced: Cascade Kim -> LB_Enhanced (Tan et al. SDM 2019).
 *            - Webb:     Cascade Kim -> LB_Webb (Webb & Petitjean PR 2021;
 *                        always >= LB_Keogh). Requires band >= 0, equal lengths.
 *
 *          NOTE (Task 5.2): tighter bounds do NOT reduce DTW calls on an EXACT
 *          full-matrix build — every entry must be computed exactly, and the
 *          pruned path recomputes any early-abandoned pair. These strategies
 *          exist as SELECTABLE primitives for the online/NN and future
 *          density-pruning (TADPole) paths, and for cross-comparison; the
 *          resulting matrix is digit-identical regardless of strategy.
 *
 * @date 2026-04-12
 */

#pragma once

#include "../error.hpp"

namespace dtwc {

enum class LowerBoundStrategy {
  Auto,
  None,
  Kim,
  Keogh,
  KimKeogh,
  Enhanced,
  Webb
};

inline void validate_lower_bound_strategy(LowerBoundStrategy value)
{
  switch (value) {
  case LowerBoundStrategy::Auto:
  case LowerBoundStrategy::None:
  case LowerBoundStrategy::Kim:
  case LowerBoundStrategy::Keogh:
  case LowerBoundStrategy::KimKeogh:
  case LowerBoundStrategy::Enhanced:
  case LowerBoundStrategy::Webb:
    return;
  }
  throw InvalidInput("Invalid LowerBoundStrategy value.");
}

} // namespace dtwc
