/*
 * Data.hpp
 *
 * Encapsulating DTWC data in a class.

 *  Created on: 04 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "settings.hpp"
#include "utility.hpp"

#include <vector>
#include <string>

namespace dtwc {

struct Data
{
  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  ind_t Nb{ 0 }; // Number of data points
  auto size() const { return Nb; }

  Data() = default;
  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new)
  {
    assert(p_vec_new.size() == p_names_new.size());
    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    Nb = p_vec.size();
  }
};

} // namespace dtwc