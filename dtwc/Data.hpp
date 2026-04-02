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

#include <cstddef>    // for size_t
#include <cassert>    // for assert
#include <stdexcept>  // for runtime_error
#include <string>     // for string
#include <utility>    // for move
#include <vector>     // for vector

namespace dtwc {

/**
 * @brief Struct to encapsulate DTWC data.
 *        This struct holds data vectors and their corresponding names, providing
 *        functionalities to manage and access the data.
 */
struct Data
{
  std::vector<std::vector<data_t>> p_vec; //!< Vector of data vectors (flat: timesteps * ndim per series)
  std::vector<std::string> p_names;       //!< Vector of data point names
  size_t ndim = 1;                        //!< Number of features (dimensions) per timestep

  /**
   * @brief Returns the number of data points (series count).
   * @return The size of the data vector (size_t).
   */
  auto size() const { return p_vec.size(); }

  /**
   * @brief Returns the number of timesteps for series i.
   * @param i Index of the series.
   * @return Number of timesteps (flat size / ndim).
   */
  size_t series_length(size_t i) const { return p_vec[i].size() / ndim; }

  /**
   * @brief Validates that all series have flat sizes divisible by ndim.
   * @throws std::runtime_error if any series has an incompatible size.
   */
  void validate_ndim() const
  {
    for (size_t i = 0; i < p_vec.size(); ++i) {
      if (p_vec[i].size() % ndim != 0) {
        throw std::runtime_error(
          "Series " + std::to_string(i) + " has flat size " +
          std::to_string(p_vec[i].size()) + " which is not divisible by ndim=" +
          std::to_string(ndim));
      }
    }
  }

  Data() = default; //!< Default constructor

  /**
   * @brief Constructor that initializes data and name vectors.
   * @param p_vec_new Rvalue reference to a vector of data vectors.
   * @param p_names_new Rvalue reference to a vector of data point names.
   * @param ndim_ Number of features per timestep (default 1).
   * @throws std::runtime_error if data and name vectors are not of the same size,
   *         or if any series size is not divisible by ndim_.
   */
  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new,
       size_t ndim_ = 1)
    : ndim{ ndim_ }
  {
    if (p_vec_new.size() != p_names_new.size())
      throw std::runtime_error("Data and name vectors should be of the same size");

    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    validate_ndim();
  }
};

} // namespace dtwc