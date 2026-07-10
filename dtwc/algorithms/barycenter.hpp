/**
 * @file barycenter.hpp
 * @brief DTW barycenters and barycentric k-means clustering.
 */

#pragma once

#include "../core/clustering_result.hpp"
#include "../settings.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dtwc {
class Problem;

namespace algorithms {

enum class BarycenterMethod {
  SSG,      ///< Stochastic subgradient averaging (default; Schultz & Jain 2018).
  DBA,      ///< Classic majorize-minimize DTW Barycenter Averaging.
  SoftDTW  ///< Gradient descent on squared-cost soft-DTW (Cuturi & Blondel 2017).
};

struct BarycenterOptions {
  BarycenterMethod method = BarycenterMethod::SSG;
  int max_iter = 50;
  double learning_rate = 0.2;
  double learning_rate_decay = 0.01;
  double gamma = 1.0;
  double tolerance = 1e-6;
  std::uint64_t random_seed = 42;
};

struct BarycenterClusteringOptions {
  int n_clusters = 3;
  int max_iter = 50;
  int barycenter_max_iter = 30;
  int target_length = -1; ///< -1 keeps each initial centre's length.
  BarycenterMethod method = BarycenterMethod::SSG;
  double learning_rate = 0.2;
  double learning_rate_decay = 0.01;
  double gamma = 1.0;
  double tolerance = 1e-6;
  std::uint64_t random_seed = 42;
};

/** Barycentric clustering has sequence-valued centres, not medoid indices. */
struct BarycenterClusteringResult {
  std::vector<int> labels;
  std::vector<std::vector<data_t>> barycenters;
  double total_cost = 0.0; ///< Hard squared-DTW inertia for comparability.
  int iterations = 0;
  bool converged = false;
};

/** Compute one barycenter from the selected series in a Problem. */
std::vector<data_t> dtw_barycenter(const Problem& prob,
                                   const std::vector<int>& series_indices,
                                   std::size_t target_length,
                                   const BarycenterOptions& options = {});

/** Lloyd-style clustering whose centres are DTW barycenters. */
BarycenterClusteringResult barycenter_kmeans(
  const Problem& prob, const BarycenterClusteringOptions& options = {});

} // namespace algorithms
} // namespace dtwc

