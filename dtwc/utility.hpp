/*
 * utility.hpp
 *
 * utility functions

 * Created on: 15 Dec 2021
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"
#include "types/types.hpp"
#include "fileOperations.hpp"
#include "parallelisation.hpp"

#include <iostream>
#include <vector>
#include <array>

#include <numeric>
#include <fstream>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <iterator>
#include <memory>
#include <tuple>
#include <iomanip>

namespace dtwc {

template <typename Tfun>
void fillDistanceMatrix(Tfun &distByInd, size_t N)
{
  auto oneTask = [&, N = N](size_t i_linear) {
    size_t i{ i_linear / N }, j{ i_linear % N };
    if (i <= j)
      distByInd(i, j);
  };

  dtwc::run(oneTask, N * N);
}

template <typename Tdata>
bool aremedoidsSame(const std::vector<Tdata> &m1, std::vector<Tdata> &m2)
{
  if (m1.size() != m2.size())
    return false;

  for (size_t i = 0; i != m1.size(); i++)
    if (m1[i] != m2[i])
      return false;

  return true;
}

}; // namespace dtwc