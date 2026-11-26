/**
 * @file distance_sampling_weights.hpp
 * @brief k-medoids++ seeding: D-sampling weights of non-negative
 *        dissimilarities, and the one seeding loop that draws from them.
 * @author Volkan Kumtepeli
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
 * The D-sampling weights of nearest-dissimilarity values: each unselected value as it is, each selected one zero.
 * Every caller's values are non-negative: fast_pam and init::Kmeanspp read a Problem's distances, which no variant
 * gives below zero (Soft-DTW's is the divergence) and no matrix intake accepts below zero, and barycenter k-means
 * its DTW of squared costs. portable_weighted_index refuses a negative weight.
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

  DistanceSamplingWeights result;
  result.values.resize(distances.size());
  for (size_t i = 0; i < distances.size(); ++i) {
    const double distance = static_cast<double>(distances[i]);
    if (!std::isfinite(distance))
      throw InvalidInput(
        caller + ": initialization distance must be finite");
    if (is_selected[i]) continue;
    result.values[i] = distance;
    result.total += distance;
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
  const auto n = static_cast<std::size_t>(N);
  std::vector<index_t> chosen{ static_cast<index_t>(
    portable_bounded(rng, static_cast<std::uint64_t>(N))) };
  chosen.reserve(static_cast<std::size_t>(k));

  std::vector<char> is_chosen(n, 0);
  is_chosen[static_cast<std::size_t>(chosen[0])] = 1;

  std::vector<double> nearest(n, std::numeric_limits<double>::infinity());
  std::vector<double> weights(n, 0.0);

  while (static_cast<index_t>(chosen.size()) < k) {
    const index_t last = chosen.back();
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = std::min(nearest[i], static_cast<double>(distance(last, static_cast<index_t>(i))));
      if (!std::isfinite(d))
        throw InvalidInput(caller + ": initialization distance must be finite");
      nearest[i] = d;
      if (is_chosen[i]) {
        weights[i] = 0.0;
      } else {
        weights[i] = d;
        total += d;
      }
    }
    if (!std::isfinite(total))
      throw InvalidInput(caller + ": initialization weight total is non-finite");

    index_t next = 0;
    if (total > 0.0)
      next = static_cast<index_t>(portable_weighted_index(
        weights.begin(), weights.end(), total, rng));
    if (total <= 0.0 || is_chosen[static_cast<std::size_t>(next)]) {
      next = 0;
      while (is_chosen[static_cast<std::size_t>(next)]) ++next;
    }
    is_chosen[static_cast<std::size_t>(next)] = 1;
    chosen.push_back(next);
  }
  return chosen;
}

} // namespace dtwc::core
