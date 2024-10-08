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

constexpr data_t int_threshold = 0.01;
constexpr double epsilon = 1e-8;

bool inline isAround(double x, double y = 0.0, double tolerance = epsilon)
{
  return std::abs(x - y) <= tolerance;
}
bool inline isFractional(double x) { return std::abs(x - std::round(x)) > epsilon; }

template <typename T>
inline bool is_one(T x) { return x > (1 - int_threshold); }

template <typename T>
inline bool is_zero(T x) { return x < int_threshold; }

template <typename T>
inline bool is_integer(T x) { return is_one(x) || is_zero(x); }

} // namespace dtwc::solver
