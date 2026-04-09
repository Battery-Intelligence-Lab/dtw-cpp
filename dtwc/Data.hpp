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
#include "core/storage.hpp"

#include <cassert>      // for assert
#include <cstddef>      // for size_t
#include <span>         // for span
#include <stdexcept>    // for runtime_error
#include <string>       // for string
#include <string_view>  // for string_view
#include <utility>      // for move
#include <vector>       // for vector

namespace dtwc {

/**
 * @brief Struct to encapsulate DTWC data.
 *        This struct holds data vectors and their corresponding names, providing
 *        functionalities to manage and access the data.
 */
struct Data
{
  std::vector<std::vector<data_t>> p_vec; //!< Float64 series (default)
  std::vector<std::vector<float>> p_vec_f32; //!< Float32 series (when precision == Float32)
  std::vector<std::string> p_names;       //!< Vector of data point names
  size_t ndim = 1;                        //!< Number of features (dimensions) per timestep
  core::Precision precision = core::Precision::Float64; //!< Active precision

  /// Returns the number of data points (series count).
  size_t size() const
  {
    if (is_view_) return is_f32() ? p_spans_f32_.size() : p_spans_.size();
    return (precision == core::Precision::Float32) ? p_vec_f32.size() : p_vec.size();
  }

  /// Returns the number of timesteps for series i.
  size_t series_length(size_t i) const { return series_flat_size(i) / ndim; }

  /// Returns a float64 span view of series i (only valid when precision == Float64 or view mode).
  std::span<const data_t> series(size_t i) const
  {
    if (is_view_) return p_spans_[i];
    return std::span<const data_t>(p_vec[i]);
  }

  /// Returns a float32 span view of series i (only valid when precision == Float32).
  std::span<const float> series_f32(size_t i) const
  {
    if (is_view_) return p_spans_f32_[i];
    return std::span<const float>(p_vec_f32[i]);
  }

  /// Returns the number of scalar values in series i.
  size_t series_flat_size(size_t i) const
  {
    if (is_view_) return is_f32() ? p_spans_f32_[i].size() : p_spans_[i].size();
    return (precision == core::Precision::Float32) ? p_vec_f32[i].size() : p_vec[i].size();
  }

  /// Returns the name of series i as a string_view.
  std::string_view name(size_t i) const
  {
    if (is_view_) return p_name_views_[i];
    return std::string_view(p_names[i]);
  }

  /// True if this Data is a non-owning view into another Data's storage.
  bool is_view() const { return is_view_; }

  /// True if data is stored as float32.
  bool is_f32() const { return precision == core::Precision::Float32; }

  /// Validates that all series have flat sizes divisible by ndim.
  void validate_ndim() const
  {
    const auto n = size();
    for (size_t i = 0; i < n; ++i) {
      if (series_flat_size(i) % ndim != 0) {
        throw std::runtime_error(
          "Series " + std::to_string(i) + " has flat size " +
          std::to_string(series_flat_size(i)) + " which is not divisible by ndim=" +
          std::to_string(ndim));
      }
    }
  }

  Data() = default; //!< Default constructor

  /// Float64 heap-mode constructor.
  Data(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new,
       size_t ndim_ = 1)
    : ndim{ ndim_ }, precision{ core::Precision::Float64 }
  {
    if (p_vec_new.size() != p_names_new.size())
      throw std::runtime_error("Data and name vectors should be of the same size");
    p_vec = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    validate_ndim();
  }

  /// Float32 heap-mode constructor.
  Data(std::vector<std::vector<float>> &&p_vec_new, std::vector<std::string> &&p_names_new,
       size_t ndim_ = 1)
    : ndim{ ndim_ }, precision{ core::Precision::Float32 }
  {
    if (p_vec_new.size() != p_names_new.size())
      throw std::runtime_error("Data and name vectors should be of the same size");
    p_vec_f32 = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    validate_ndim();
  }

  /// View-mode constructor: non-owning float64 spans into another Data's storage.
  Data(std::vector<std::span<const data_t>> &&spans,
       std::vector<std::string_view> &&name_views, size_t ndim_)
    : ndim{ ndim_ }, p_spans_(std::move(spans)),
      p_name_views_(std::move(name_views)), is_view_(true)
  {
    if (p_spans_.size() != p_name_views_.size())
      throw std::runtime_error("Data view: span and name vectors should be of the same size");
    validate_ndim();
  }

  /// View-mode constructor: non-owning float32 spans into another Data's storage.
  Data(std::vector<std::span<const float>> &&spans,
       std::vector<std::string_view> &&name_views, size_t ndim_)
    : ndim{ ndim_ }, precision{ core::Precision::Float32 },
      p_spans_f32_(std::move(spans)),
      p_name_views_(std::move(name_views)), is_view_(true)
  {
    if (p_spans_f32_.size() != p_name_views_.size())
      throw std::runtime_error("Data view: span and name vectors should be of the same size");
    validate_ndim();
  }

private:
  std::vector<std::span<const data_t>> p_spans_;    //!< View-mode: float64 spans into parent data
  std::vector<std::span<const float>> p_spans_f32_; //!< View-mode: float32 spans into parent data
  std::vector<std::string_view> p_name_views_;       //!< View-mode: names from parent
  bool is_view_ = false;                             //!< True when in view mode
};

} // namespace dtwc