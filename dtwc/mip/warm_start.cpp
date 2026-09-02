#include "warm_start.hpp"
#include "solution_transaction.hpp"

#include "../Problem.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../error.hpp"
#include "../settings.hpp"

#include <cstddef>
#include <string>

namespace dtwc::mip {

core::ClusteringResult make_warm_start(
  Problem &prob, std::uint64_t random_seed)
{
  const std::size_t n_points = prob.size();
  const int n_clusters = prob.n_clusters();

  core::ClusteringResult result;
  {
    ExactClusteringTransaction transaction(prob);
    result = fast_pam_seeded(
      prob, n_clusters, random_seed, settings::DEFAULT_MAX_ITER);
  }

  // The backends index an Nb*Nb start vector with medoid_indices[labels[j]], so
  // an incomplete or out-of-range heuristic result writes out of bounds. This is
  // the INPUT counterpart of extract_exact_clustering's output validation.
  if (result.medoid_indices.size() != static_cast<std::size_t>(n_clusters))
    throw SolverError("MIP warm start produced "
      + std::to_string(result.medoid_indices.size()) + " medoids but the problem asks for "
      + std::to_string(n_clusters) + ".");
  if (result.labels.size() != n_points)
    throw SolverError("MIP warm start produced " + std::to_string(result.labels.size())
      + " labels but the problem has " + std::to_string(n_points) + " points.");
  for (const int medoid : result.medoid_indices)
    if (medoid < 0 || static_cast<std::size_t>(medoid) >= n_points)
      throw SolverError("MIP warm start medoid index " + std::to_string(medoid)
        + " is outside [0, N).");
  for (const int label : result.labels)
    if (label < 0 || label >= n_clusters)
      throw SolverError("MIP warm start label " + std::to_string(label)
        + " is outside [0, k).");

  return result;
}

} // namespace dtwc::mip
