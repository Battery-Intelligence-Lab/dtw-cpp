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
 *
 * @date 2026-04-12
 */

#pragma once

namespace dtwc {

enum class LowerBoundStrategy {
  Auto,
  None,
  Kim,
  Keogh,
  KimKeogh
};

} // namespace dtwc
