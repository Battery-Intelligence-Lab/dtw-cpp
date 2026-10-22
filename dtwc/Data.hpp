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

#include "base/error.hpp"
#include "base/settings.hpp"
#include "core/storage.hpp"

#include <cstddef>      // for size_t
#include <memory>       // for shared_ptr
#include <span>         // for span
#include <string>       // for string
#include <string_view>  // for string_view
#include <utility>      // for move
#include <vector>       // for vector

// GCC's inline-unit-growth cap in Problem.cpp left series() and series_f32() out of line in the fill (2026-10-21).
#if defined(__GNUC__)
#define DTWC_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define DTWC_ALWAYS_INLINE inline
#endif

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
  index_t size() const
  {
    if (is_view_)
      return static_cast<index_t>(is_f32() ? p_spans_f32_.size() : p_spans_.size());
    return static_cast<index_t>(
      (precision == core::Precision::Float32) ? p_vec_f32.size() : p_vec.size());
  }

  /// Returns the number of timesteps for series i.
  size_t series_length(size_t i) const { return series_flat_size(i) / ndim; }

  /// Returns a float64 span view of series i. Requires precision == Float64.
  /// @details size() already branches on precision; without the same branch
  /// here, series() on Float32 data indexes the empty p_vec / p_spans_.
  DTWC_ALWAYS_INLINE std::span<const data_t> series(size_t i) const
  {
    if (is_f32()) throw_wrong_precision("series", "Float32", "series_f32");
    if (is_view_) return p_spans_[i];
    return std::span<const data_t>(p_vec[i]);
  }

  /// Returns a float32 span view of series i. Requires precision == Float32.
  DTWC_ALWAYS_INLINE std::span<const float> series_f32(size_t i) const
  {
    if (!is_f32()) throw_wrong_precision("series_f32", "Float64", "series");
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

  /// True if this Data is a non-owning view: of another Data's storage, or of bytes owner() keeps alive.
  bool is_view() const { return is_view_; }

  /// What keeps a view's bytes alive (a mapped file, a read buffer, decoded batches), shared by every copy of this
  /// Data; empty for an owning Data, and for a view into another Data's storage, which must then outlive it.
  const std::shared_ptr<const void> &owner() const { return owner_; }

  /// True if data is stored as float32.
  bool is_f32() const { return precision == core::Precision::Float32; }

  /// Validates that all series have flat sizes divisible by ndim.
  void validate_ndim() const
  {
    // series_length() and the modulo below both divide by ndim.
    if (ndim == 0)
      throw InvalidInput("Data: ndim must be at least 1");
    const auto n = size();
    for (size_t i = 0; i < n; ++i) {
      if (series_flat_size(i) % ndim != 0) {
        throw InvalidInput(
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
      throw InvalidInput("Data and name vectors should be of the same size");
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
      throw InvalidInput("Data and name vectors should be of the same size");
    p_vec_f32 = std::move(p_vec_new);
    p_names = std::move(p_names_new);
    validate_ndim();
  }

  /// View-mode constructor: non-owning float64 spans into storage `owner` keeps alive (see owner()).
  Data(std::vector<std::span<const data_t>> &&spans,
       std::vector<std::string_view> &&name_views, size_t ndim_,
       std::shared_ptr<const void> owner = {})
    : ndim{ ndim_ }, p_spans_(std::move(spans)),
      p_name_views_(std::move(name_views)), is_view_(true), owner_(std::move(owner))
  {
    if (p_spans_.size() != p_name_views_.size())
      throw InvalidInput("Data view: span and name vectors should be of the same size");
    validate_ndim();
  }

  /// View-mode constructor: non-owning float32 spans into storage `owner` keeps alive (see owner()).
  Data(std::vector<std::span<const float>> &&spans,
       std::vector<std::string_view> &&name_views, size_t ndim_,
       std::shared_ptr<const void> owner = {})
    : ndim{ ndim_ }, precision{ core::Precision::Float32 },
      p_spans_f32_(std::move(spans)),
      p_name_views_(std::move(name_views)), is_view_(true), owner_(std::move(owner))
  {
    if (p_spans_f32_.size() != p_name_views_.size())
      throw InvalidInput("Data view: span and name vectors should be of the same size");
    validate_ndim();
  }

private:
  std::vector<std::span<const data_t>> p_spans_;    //!< View-mode: float64 spans into parent data
  std::vector<std::span<const float>> p_spans_f32_; //!< View-mode: float32 spans into parent data
  std::vector<std::string_view> p_name_views_;       //!< View-mode: names from parent
  bool is_view_ = false;                             //!< True when in view mode
  std::shared_ptr<const void> owner_;                //!< View-mode: keeps the viewed bytes alive (owner())

  /// Common error for an accessor that does not match the active precision.
  [[noreturn]] static void throw_wrong_precision(const char *accessor,
                                                 const char *stored,
                                                 const char *correct)
  {
    throw InvalidInput(
      std::string("Data::") + accessor + ": data is stored as " + stored
      + "; use " + correct + "() instead.");
  }
};

namespace detail {

/// An owning copy of `data`, a view or not, in `precision`: each value cast once, the names copied.
inline Data owned_copy(const Data &data, core::Precision precision)
{
  const auto copy = [&data](auto zero) {
    using T = decltype(zero);
    std::vector<std::vector<T>> series(static_cast<size_t>(data.size()));
    std::vector<std::string> names(series.size());
    for (size_t i = 0; i < series.size(); ++i) {
      const auto cast = [&series, i](auto values) {
        series[i].reserve(values.size());
        for (const auto value : values) series[i].push_back(static_cast<T>(value));
      };
      if (data.is_f32())
        cast(data.series_f32(i));
      else
        cast(data.series(i));
      names[i] = data.name(i);
    }
    return Data(std::move(series), std::move(names), data.ndim);
  };
  return precision == core::Precision::Float32 ? copy(float{}) : copy(data_t{});
}

} // namespace detail

} // namespace dtwc