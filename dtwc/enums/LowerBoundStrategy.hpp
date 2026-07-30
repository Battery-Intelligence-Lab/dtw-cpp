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
 *            - Kim:      LB_Kim only (O(1), currently L1-valued; F47).
 *            - Keogh:    LB_Keogh only (requires band >= 0).
 *            - KimKeogh: Cascade Kim -> Keogh.
 *            - Enhanced: Cascade Kim -> max(LB_Keogh, LB_Enhanced), because
 *                        neither envelope bound dominates for effective V>=2.
 *            - Webb:     Cascade Kim -> symmetric local LB_Webb_NoLR plus its
 *                        conservative tail cap. In this L1 route it dominates
 *                        symmetric Keogh; it is not full Algorithm 2. Requires
 *                        band >= 0 and equal lengths.
 *
 *          NOTE (Task 5.2): tighter bounds do NOT reduce DTW calls on an EXACT
 *          full-matrix build — every entry must be computed exactly, and the
 *          pruned path recomputes any early-abandoned pair. These selectors
 *          configure that legacy exact-matrix route for diagnostics and
 *          cross-comparison. They do not configure TADPole: TADPole owns its
 *          internal bound where a classified far-pair distance need not be
 *          materialised.
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
