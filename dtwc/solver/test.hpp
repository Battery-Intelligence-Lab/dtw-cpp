/*
 * test.hpp
 *
 * Test problems

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "EqualityConstraints.hpp"
#include <vector>
#include <utility>

namespace dtwc::solver {
auto get_prob_small()
{
  EqualityConstraints eq(2, 4);
  std::vector<double> c{ 1, -2, 0, 0 };

  eq.A(0, 0) = -4;
  eq.A(0, 1) = 6;
  eq.A(0, 2) = 1;

  eq.A(1, 0) = 1;
  eq.A(1, 1) = 1;
  eq.A(1, 3) = 1;

  eq.b[0] = 5;
  eq.b[1] = 5;

  return std::pair(eq, c);
}

} // namespace dtwc::solver