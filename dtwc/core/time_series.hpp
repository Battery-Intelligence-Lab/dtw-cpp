/**
 * @file time_series.hpp
 * @brief Non-owning view and owning container for time series data.
 *
 * @details TimeSeriesView provides a lightweight, non-owning reference to
 * contiguous time series data. TimeSeries owns its data and can implicitly
 * convert to a view.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include <vector>
#include <string>
#include <cstddef>

namespace dtwc::core {

template <typename T = double>
struct TimeSeriesView {
  const T *data;
  size_t length;

  const T &operator[](size_t i) const { return data[i]; }
  const T *begin() const { return data; }
  const T *end() const { return data + length; }
  bool empty() const { return length == 0; }
  size_t size() const { return length; }

  bool operator==(const TimeSeriesView &other) const {
    if (length != other.length) return false;
    for (size_t i = 0; i < length; ++i)
      if (data[i] != other.data[i]) return false;
    return true;
  }
  bool operator!=(const TimeSeriesView &other) const { return !(*this == other); }
};

template <typename T = double>
struct TimeSeries {
  std::vector<T> data;
  std::string name;

  size_t size() const { return data.size(); }
  bool empty() const { return data.empty(); }
  const T &operator[](size_t i) const { return data[i]; }
  T &operator[](size_t i) { return data[i]; }
  // Explicit conversion to prevent dangling from temporaries.
  // Use .view() for intentional conversion.
  explicit operator TimeSeriesView<T>() const & { return { data.data(), data.size() }; }
  operator TimeSeriesView<T>() const && = delete; // prevent dangling from temporaries
  TimeSeriesView<T> view() const { return { data.data(), data.size() }; }
};

} // namespace dtwc::core
