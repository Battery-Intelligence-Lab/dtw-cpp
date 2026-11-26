/**
 * @file decode_assignment.hpp
 * @brief Decode an optimal p-median assignment matrix into medoids and labels.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "nearest_medoid.hpp"
#include "../base/error.hpp"
#include "../core/clustering_result.hpp"
#include "../core/distance_matrix.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dtwc::mip {

namespace detail {

inline constexpr double kDecodeFeasTol = 1e-4;

/// Validate primal feasibility of a p-median solution vector `x` and extract the
/// `k` open medoids from the diagonal `y_f = x[f, f]`.
inline void validate_pmedian_primal(std::span<const double> x, std::size_t n, index_t k,
                                    bool point_major, std::string_view backend,
                                    core::ClusteringResult &result,
                                    std::vector<index_t> &slot_of)
{
  const auto fail = [&](const std::string &what) {
    throw SolverError(std::string(backend) + " returned an invalid p-median solution: " + what);
  };
  if (x.size() != n * n)
    fail("expected " + std::to_string(n * n) + " variables, got " + std::to_string(x.size()) + ".");

  const auto val = [&](std::size_t f, std::size_t p) {
    return x[point_major ? f + p * n : f * n + p];
  };

  std::vector<double> y(n, 0.0);
  slot_of.assign(n, -1);
  result.medoid_indices.clear();
  result.medoid_indices.reserve(static_cast<std::size_t>(k));

  for (std::size_t f = 0; f < n; ++f) {
    const double yf = val(f, f);
    if (!std::isfinite(yf))
      fail("non-finite diagonal indicator at facility " + std::to_string(f) + ".");
    if (yf >= 1.0 - kDecodeFeasTol && yf <= 1.0 + kDecodeFeasTol) {
      slot_of[f] = static_cast<index_t>(result.medoid_indices.size());
      result.medoid_indices.push_back(static_cast<index_t>(f));
    } else if (yf < -kDecodeFeasTol || yf > kDecodeFeasTol) {
      fail("non-binary diagonal indicator " + std::to_string(yf) + " at facility " + std::to_string(f) + ".");
    }
    y[f] = yf;
  }

  if (result.medoid_indices.size() != static_cast<std::size_t>(k))
    fail(std::to_string(result.medoid_indices.size()) + " medoids for k = " + std::to_string(k) + ".");

  for (std::size_t p = 0; p < n; ++p) {
    double sum_f = 0.0;
    for (std::size_t f = 0; f < n; ++f) {
      const double xfp = val(f, p);
      if (!std::isfinite(xfp))
        fail("non-finite assignment at (" + std::to_string(f) + ", " + std::to_string(p) + ").");
      if (xfp < -kDecodeFeasTol || xfp > y[f] + kDecodeFeasTol)
        fail("assignment at (" + std::to_string(f) + ", " + std::to_string(p) + ") violates linking/bounds.");
      sum_f += xfp;
    }
    if (std::abs(sum_f - 1.0) > kDecodeFeasTol)
      fail("point " + std::to_string(p) + " assignment sum is " + std::to_string(sum_f) + ", expected 1.");
  }
}

} // namespace detail

/// Validate primal feasibility of `x` and assign each point to its serving open medoid.
/// HiGHS stores facility-major (`f·n + p`), Gurobi point-major (`f + p·n`).
inline core::ClusteringResult decode_assignment(std::span<const double> x, std::size_t n, index_t k,
                                                bool point_major, std::string_view backend)
{
  core::ClusteringResult result;
  std::vector<index_t> slot_of;
  detail::validate_pmedian_primal(x, n, k, point_major, backend, result, slot_of);

  const auto val = [&](std::size_t f, std::size_t p) {
    return x[point_major ? f + p * n : f * n + p];
  };
  result.labels.assign(n, -1);
  for (std::size_t p = 0; p < n; ++p) {
    if (slot_of[p] >= 0 && val(p, p) > 0.5) {
      bool exclusive_self = true;
      for (const index_t med : result.medoid_indices) {
        if (static_cast<std::size_t>(med) != p && val(static_cast<std::size_t>(med), p) >= val(p, p)) {
          exclusive_self = false;
          break;
        }
      }
      if (exclusive_self) {
        result.labels[p] = slot_of[p];
        continue;
      }
    }
    double best_x = -1.0;
    index_t best_slot = -1;
    for (index_t t = 0; t < k; ++t) {
      const auto f = static_cast<std::size_t>(result.medoid_indices[static_cast<std::size_t>(t)]);
      const double xfp = val(f, p);
      if (xfp > best_x || (xfp == best_x && f == p)) {
        best_x = xfp;
        best_slot = t;
      }
    }
    if (best_slot < 0)
      throw SolverError(std::string(backend) + " returned an invalid p-median solution: point "
                        + std::to_string(p) + " is not assigned to an open medoid.");
    result.labels[p] = best_slot;
  }
  return result;
}

/// Validate primal feasibility of facility-major `x` and assign every point `p`
/// to its Voronoi-nearest open medoid in `dm` with self-tie break (`d == best && medoid == p`).
inline core::ClusteringResult decode_assignment_nearest(std::span<const double> x,
                                                        const core::DistanceMatrix &dm,
                                                        std::size_t n, index_t k,
                                                        std::string_view backend)
{
  core::ClusteringResult result;
  std::vector<index_t> slot_of;
  detail::validate_pmedian_primal(x, n, k, /*point_major=*/false, backend, result, slot_of);

  result.labels.assign(n, -1);
  dm.visit([&](const auto *packed) {
    for (std::size_t p = 0; p < n; ++p) {
      const auto best = mip::nearest_medoid(k, [&](index_t t) {
        const auto med = static_cast<std::size_t>(result.medoid_indices[static_cast<std::size_t>(t)]);
        return static_cast<double>(packed[core::tri_index(med, p)]);
      });
      index_t slot = best.position;
      if (slot_of[p] >= 0) {
        const double d_self = static_cast<double>(packed[core::tri_index(p, p)]);
        if (d_self <= best.distance) slot = slot_of[p];
      }
      result.labels[p] = slot;
    }
  });
  return result;
}

} // namespace dtwc::mip
