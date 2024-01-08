/**
 * _Data.hpp_
 *
 * @brief Encapsulating DTWC data in a class.
 *
 * @author Volkan Kumtepeli, Becky Perriment
 * @date Created on: 04 Dec 2022
 */

#pragma once

#include "settings.hpp"

#include <cstddef> // for size_t
#include <cassert> // for assert
#include <string>  // for string
#include <utility> // for move
#include <vector>  // for vector

namespace dtwc {

/**
 * @brief Struct to encapsulate DTWC data
 */

struct Data
{
  std::vector<std::vector<data_t>> p_vec; //!< Vector of data vectors
  std::vector<std::string> p_names;       //!< Vector of data point names

  auto size() const { return static_cast<int>(p_vec.size()); } //!< Returns number of data points

  Data() = default;
  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new)
  {
    if (p_vec_new.size() != p_names_new.size())
      throw std::runtime_error("Data and name vectors should be of the same size");

    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
  }
};

} // namespace dtwc