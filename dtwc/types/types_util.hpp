/**
 * @file types_util.hpp
 * @brief Utility functions for types
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 19 Dec 2022
 */

#pragma once

#include "../settings.hpp"

#include <limits>
#include <array>
#include <vector>

namespace dtwc::solver {

constexpr double epsilon = 1e-8;

bool inline isAround(double x, double y = 0.0, double tolerance = epsilon)
{
  return std::abs(x - y) <= tolerance;
}
bool inline isFractional(double x) { return std::abs(x - std::round(x)) > epsilon; }

} // namespace dtwc::solver
