/**
 * @file fast_clara_assign.hpp
 * @brief FastCLARA's nearest-medoid assignment, shared by fast_clara.cpp (series in RAM)
 *        and fast_clara_parquet.cpp (series streamed from Parquet, dtwc_io).
 */

#pragma once

#include "../fast_clara.hpp"
#include "../../base/parallelisation.hpp"
#include "../../core/medoid_assignment_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace dtwc::algorithms::detail {

/// Invocation-local PAM seed for one CLARA subsample.
inline std::uint64_t clara_pam_seed(const CLARAOptions &opts, int sample_index)
{
  return opts.random_seed + static_cast<std::uint64_t>(sample_index);
}

/**
 * Points first .. first + best.size() - 1 to their nearest medoid: labels[p]
 * and best[p - first]. row_of(p)(m) is point p's distance to medoid slot m,
 * whichever device computed it; a point is at 0 from itself. Each point
 * writes its own slots.
 */
template <typename RowOf>
void assign_points(
  index_t first, std::span<double> best, const std::vector<index_t> &medoid_indices,
  std::vector<index_t> &labels, const RowOf &row_of)
{
  const auto k = static_cast<index_t>(medoid_indices.size());
  auto assign_point = [&](std::size_t index) {
    const index_t p = first + static_cast<index_t>(index);
    const auto distance = row_of(p);
    double best_dist = std::numeric_limits<double>::max();
    index_t best_label = 0;
    bool has_best = false;

    for (index_t m = 0; m < k; ++m) {
      const index_t medoid = medoid_indices[m];
      const double d = core::detail::require_finite_medoid_distance(
        p == medoid ? 0.0 : distance(m),
        "fast_clara", static_cast<std::size_t>(p), m, medoid);
      // A medoid tied with another medoid (a duplicate series) serves
      // itself, or its own cluster would be published empty.
      if (!has_best || d < best_dist || (d == best_dist && medoid == p)) {
        best_dist = d;
        best_label = m;
        has_best = true;
      }
    }

    labels[static_cast<std::size_t>(p)] = best_label;
    best[index] = best_dist;
  };
  run_openmp(assign_point, best.size(), best.size() > 64);
}

#ifdef DTWC_HAS_CUDA
/**
 * Points first .. of `series` against the k `medoids` (slot m is
 * medoid_indices[m]) on `prob`'s GPU, with its band, metric and precision;
 * each block of distances is assigned as it arrives.
 */
void assign_on_gpu(
  const Problem &prob, const std::vector<std::vector<double>> &series, index_t first,
  const std::vector<std::vector<double>> &medoids, const std::vector<index_t> &medoid_indices,
  std::vector<index_t> &labels, std::span<double> best);
#else
/// Metal has no kernel for the assignment, so on a GPU device it runs on the
/// CPU (the sample matrices still fill on the GPU); verbose says so.
void say_cpu_assignment(const Problem &prob, std::size_t n_points, std::size_t k);
#endif

} // namespace dtwc::algorithms::detail
