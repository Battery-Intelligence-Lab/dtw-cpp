/**
 * @file Range.hpp
 * @brief A basic Range class not to create vector for Iota
 * @author Volkan Kumtepeli
 * @date 18 Aug 2022
 */

#pragma once

#include "Index.hpp"

#include <iterator>

namespace dtwc {

class Range
{
  size_t x0{}, xN{};

public:
  Range() = default; // Default constructor 0,0;
  explicit Range(size_t xN) : xN{ xN } {}
  Range(size_t x0, size_t xN) : x0{ x0 }, xN{ xN } {}

  auto begin() { return Index(x0); }
  auto end() { return Index(xN); }
};

} // namespace dtwc