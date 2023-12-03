/*
 * Data.hpp
 *
 * Encapsulating DTWC data in a class.

 *  Created on: 04 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "settings.hpp"

#include <cstddef> // for size_t
#include <cassert> // for assert
#include <string>  // for string
#include <utility> // for move
#include <vector>  // for vector

namespace dtwc {

struct Data
{
  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  ssize_t Nb{ 0 }; // Number of data points
  auto size() const { return Nb; }

  Data() = default;
  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new)
  {
    assert(p_vec_new.size() == p_names_new.size());
    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    Nb = std::ssize(p_vec);
  }
};

} // namespace dtwc