/**
 * @file time_series.hpp
 * @brief Non-owning view and owning container for time series data.
 *
 * @details TimeSeriesView provides a lightweight, non-owning reference to
 * contiguous time series data. TimeSeries owns its data and can implicitly
 * convert to a view.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "../settings.hpp"

#include <vector>
#include <string>
#include <cstddef>

namespace dtwc::core {

template <typename T = dtwc::settings::default_data_t>
struct TimeSeriesView {
  const T *data;
  size_t length;        //!< Number of timesteps
  size_t ndim = 1;      //!< Number of features (dimensions) per timestep

  /// Element access for univariate series (backward compat, accesses flat buffer).
  const T &operator[](size_t i) const { return data[i]; }

  /// Pointer to the start of timestep i (ndim elements).
  const T *at(size_t i) const { return data + i * ndim; }

  /// Total number of scalar values in the flat buffer (length * ndim).
  size_t flat_size() const { return length * ndim; }

  const T *begin() const { return data; }
  const T *end() const { return data + length * ndim; }
  bool empty() const { return length == 0; }
  size_t size() const { return length; }

  bool operator==(const TimeSeriesView &other) const {
    if (length != other.length || ndim != other.ndim) return false;
    const size_t n = flat_size();
    for (size_t i = 0; i < n; ++i)
      if (data[i] != other.data[i]) return false;
    return true;
  }
  bool operator!=(const TimeSeriesView &other) const { return !(*this == other); }
};

template <typename T = dtwc::settings::default_data_t>
struct TimeSeries {
  std::vector<T> data;
  std::string name;
  size_t ndim = 1;      //!< Number of features (dimensions) per timestep

  size_t size() const { return data.size(); }
  bool empty() const { return data.empty(); }
  const T &operator[](size_t i) const { return data[i]; }
  T &operator[](size_t i) { return data[i]; }
  /// Number of timesteps (flat length / ndim); ndim==0 guarded to avoid div-by-zero.
  size_t timesteps() const { return ndim ? data.size() / ndim : data.size(); }
  // Explicit conversion to prevent dangling from temporaries.
  // Use .view() for intentional conversion. Carries ndim so multivariate
  // series survive the round-trip (length is timesteps, not flat size).
  explicit operator TimeSeriesView<T>() const & { return { data.data(), timesteps(), ndim }; }
  operator TimeSeriesView<T>() const && = delete; // prevent dangling from temporaries
  TimeSeriesView<T> view() const { return { data.data(), timesteps(), ndim }; }
};

} // namespace dtwc::core

