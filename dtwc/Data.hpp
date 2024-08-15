/**
 * @file Data.hpp
 *
 * @brief Encapsulating DTWC data in a class.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 04 Dec 2022
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
 * @brief Struct to encapsulate DTWC data.
 *        This struct holds data vectors and their corresponding names, providing
 *        functionalities to manage and access the data.
 */
struct Data
{
  std::vector<std::vector<data_t>> p_vec; //!< Vector of data vectors
  std::vector<std::string> p_names;       //!< Vector of data point names

  /**
   * @brief Returns the number of data points.
   * @return Integer representing the size of the data vector.
   */
  auto size() const { return static_cast<int>(p_vec.size()); }

  Data() = default; //!< Default constructor

  /**
   * @brief Constructor that initializes data and name vectors.
   * @param p_vec_new Rvalue reference to a vector of data vectors.
   * @param p_names_new Rvalue reference to a vector of data point names.
   * @throws std::runtime_error if data and name vectors are not of the same size.
   */
  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new)
  {
    if (p_vec_new.size() != p_names_new.size())
      throw std::runtime_error("Data and name vectors should be of the same size");

    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
  }
};

} // namespace dtwc