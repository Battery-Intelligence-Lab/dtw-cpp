/**
 * @file distance_sampling_weights.hpp
 * @brief Stable D-sampling weights for signed dissimilarities.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::core {

struct DistanceSamplingWeights
{
  std::vector<double> values;
  double total = 0.0;
};

/**
 * Convert nearest-dissimilarity values to nonnegative D-sampling weights.
 * Nonnegative inputs are byte-for-byte unchanged. If an admissible variant
 * (notably raw Soft-DTW) yields negative off-diagonal values, every unselected
 * value is translated by the same `-min(0,d_min)`; ordering and differences
 * are preserved, while selected points remain weight zero.
 */
template <typename Distance>
DistanceSamplingWeights distance_sampling_weights(
  const std::vector<Distance> &distances,
  const std::vector<int> &selected,
  const std::string &caller)
{
  std::vector<bool> is_selected(distances.size(), false);
  for (const int index : selected) {
    if (index < 0 || static_cast<size_t>(index) >= distances.size())
      throw std::logic_error(caller + ": selected index is out of range");
    is_selected[static_cast<size_t>(index)] = true;
  }

  double minimum = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < distances.size(); ++i) {
    const double distance = static_cast<double>(distances[i]);
    if (!std::isfinite(distance))
      throw std::runtime_error(
        caller + ": initialization distance must be finite");
    if (is_selected[i]) continue;
    minimum = std::min(minimum, distance);
  }

  const double shift = std::min(0.0, minimum);
  DistanceSamplingWeights result;
  result.values.resize(distances.size());
  for (size_t i = 0; i < distances.size(); ++i) {
    if (is_selected[i]) continue;
    const double weight = static_cast<double>(distances[i]) - shift;
    if (!std::isfinite(weight) || weight < 0.0)
      throw std::runtime_error(
        caller + ": translated initialization weight is invalid");
    result.values[i] = weight;
    result.total += weight;
  }
  if (!std::isfinite(result.total))
    throw std::runtime_error(
      caller + ": initialization weight total is non-finite");
  return result;
}

} // namespace dtwc::core
