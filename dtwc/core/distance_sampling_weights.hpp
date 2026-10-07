/**
 * @file distance_sampling_weights.hpp
 * @brief k-medoids++ seeding: stable D-sampling weights for signed
 *        dissimilarities, and the one seeding loop that draws from them.
 */

#pragma once

#include "../base/error.hpp"
#include "../base/settings.hpp" // index_t
#include "portable_random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
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
  const std::vector<index_t> &selected,
  const std::string &caller)
{
  std::vector<bool> is_selected(distances.size(), false);
  for (const index_t index : selected) {
    if (index < 0 || static_cast<size_t>(index) >= distances.size()) // chosen by the caller's own seeding
      throw std::logic_error(caller + ": selected index is out of range");
    is_selected[static_cast<size_t>(index)] = true;
  }

  double minimum = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < distances.size(); ++i) {
    const double distance = static_cast<double>(distances[i]);
    if (!std::isfinite(distance))
      throw InvalidInput(
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
      throw InvalidInput(
        caller + ": translated initialization weight is invalid");
    result.values[i] = weight;
    result.total += weight;
  }
  if (!std::isfinite(result.total))
    throw InvalidInput(
      caller + ": initialization weight total is non-finite");
  return result;
}

/**
 * k-medoids++ seeding (D-sampling): `k` distinct indices of [0, N). The first is
 * uniform; each next one is drawn with probability proportional to its
 * distance_sampling_weights() weight, its dissimilarity to the nearest index
 * chosen so far. When every unchosen weight is zero (identical series) or the
 * draw lands on a chosen index, the smallest unchosen index is taken instead, so
 * the k indices are distinct whatever the data.
 *
 * `distance(c, i)` is point i's dissimilarity to the chosen index c. A Σd
 * objective passes d (k-median++: FastPAM, init::Kmeanspp); barycenter k-means
 * passes its squared alignment costs (D² sampling). The caller checks
 * 1 <= k <= N; `caller` names it in the errors.
 */
template <typename Distance>
[[nodiscard]] std::vector<index_t> kmedoids_pp(
  index_t N, index_t k, std::mt19937_64 &rng, Distance &&distance,
  const std::string &caller)
{
  std::vector<index_t> chosen{ static_cast<index_t>(
    portable_bounded(rng, static_cast<std::uint64_t>(N))) };
  chosen.reserve(static_cast<std::size_t>(k));
  const auto is_chosen = [&chosen](index_t i) {
    return std::find(chosen.begin(), chosen.end(), i) != chosen.end();
  };
  std::vector<double> nearest(static_cast<std::size_t>(N),
                              std::numeric_limits<double>::infinity());
  while (static_cast<index_t>(chosen.size()) < k) {
    for (index_t i = 0; i < N; ++i) {
      auto &d = nearest[static_cast<std::size_t>(i)];
      d = std::min(d, static_cast<double>(distance(chosen.back(), i)));
    }
    const auto weights = distance_sampling_weights(nearest, chosen, caller);
    index_t next = 0;
    if (weights.total > 0.0)
      next = static_cast<index_t>(portable_weighted_index(
        weights.values.begin(), weights.values.end(), weights.total, rng));
    if (weights.total <= 0.0 || is_chosen(next)) {
      next = 0;
      while (is_chosen(next)) ++next;
    }
    chosen.push_back(next);
  }
  return chosen;
}

} // namespace dtwc::core
