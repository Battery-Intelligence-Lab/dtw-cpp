#include "solution_transaction.hpp"

#include "../Problem.hpp"
#include "../error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace dtwc::mip {
namespace {

[[noreturn]] void invalid_solution(
  std::string_view backend, const std::string &detail)
{
  throw SolverError(
    std::string(backend) + " exact MIP solution is invalid: " + detail);
}

std::size_t assignment_index(
  std::size_t facility,
  std::size_t point,
  std::size_t n_points,
  AssignmentMatrixLayout layout)
{
  if (layout == AssignmentMatrixLayout::FacilityMajor)
    return facility * n_points + point;
  return facility + point * n_points;
}

void validate_result(
  const Problem &problem,
  const core::ClusteringResult &result,
  std::string_view backend)
{
  const auto n_points = problem.size();
  const int n_clusters = problem.n_clusters();
  if (n_clusters <= 0 || static_cast<std::size_t>(n_clusters) > n_points)
    invalid_solution(backend, "Problem cluster count is outside [1, N].");
  if (result.medoid_indices.size() != static_cast<std::size_t>(n_clusters))
    invalid_solution(backend, "expected " + std::to_string(n_clusters)
      + " medoids but decoded " + std::to_string(result.medoid_indices.size()) + ".");
  if (result.labels.size() != n_points)
    invalid_solution(backend, "expected " + std::to_string(n_points)
      + " labels but decoded " + std::to_string(result.labels.size()) + ".");

  std::vector<bool> seen(n_points, false);
  for (std::size_t cluster = 0; cluster < result.medoid_indices.size(); ++cluster) {
    const int medoid = result.medoid_indices[cluster];
    if (medoid < 0 || static_cast<std::size_t>(medoid) >= n_points)
      invalid_solution(backend, "medoid index " + std::to_string(medoid)
        + " is outside [0, N).");
    if (seen[static_cast<std::size_t>(medoid)])
      invalid_solution(backend, "medoid index " + std::to_string(medoid)
        + " is duplicated.");
    seen[static_cast<std::size_t>(medoid)] = true;
  }

  for (std::size_t point = 0; point < result.labels.size(); ++point) {
    const int label = result.labels[point];
    if (label < 0 || label >= n_clusters)
      invalid_solution(backend, "label " + std::to_string(label)
        + " for point " + std::to_string(point) + " is outside [0, k).");
  }

  for (std::size_t cluster = 0; cluster < result.medoid_indices.size(); ++cluster) {
    const auto medoid = static_cast<std::size_t>(result.medoid_indices[cluster]);
    if (result.labels[medoid] != static_cast<int>(cluster))
      invalid_solution(backend, "medoid " + std::to_string(medoid)
        + " is not assigned to its own cluster.");
  }
}

} // namespace

core::ClusteringResult extract_exact_clustering(
  std::span<const double> assignment_values,
  std::size_t n_points,
  int n_clusters,
  AssignmentMatrixLayout layout,
  std::string_view backend)
{
  if (n_points == 0)
    invalid_solution(backend, "the assignment matrix has no points.");
  if (n_points > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    invalid_solution(backend, "the point count exceeds the public int index range.");
  if (n_clusters <= 0 || static_cast<std::size_t>(n_clusters) > n_points)
    invalid_solution(backend, "cluster count is outside [1, N].");
  if (n_points > std::numeric_limits<std::size_t>::max() / n_points)
    invalid_solution(backend, "assignment matrix dimensions overflow size_t.");
  const std::size_t expected_values = n_points * n_points;
  if (assignment_values.size() != expected_values)
    invalid_solution(backend, "expected " + std::to_string(expected_values)
      + " assignment values but received "
      + std::to_string(assignment_values.size()) + ".");
  if (!std::all_of(assignment_values.begin(), assignment_values.end(),
                   [](double value) { return std::isfinite(value); }))
    invalid_solution(backend, "assignment values contain NaN or infinity.");

  core::ClusteringResult result;
  for (std::size_t point = 0; point < n_points; ++point) {
    const auto diagonal = assignment_index(point, point, n_points, layout);
    if (assignment_values[diagonal] > 0.5)
      result.medoid_indices.push_back(static_cast<int>(point));
  }
  if (result.medoid_indices.size() != static_cast<std::size_t>(n_clusters))
    invalid_solution(backend, "expected " + std::to_string(n_clusters)
      + " selected medoids but decoded "
      + std::to_string(result.medoid_indices.size()) + ".");

  std::vector<int> cluster_by_medoid(n_points, -1);
  for (std::size_t cluster = 0; cluster < result.medoid_indices.size(); ++cluster)
    cluster_by_medoid[static_cast<std::size_t>(result.medoid_indices[cluster])]
      = static_cast<int>(cluster);

  result.labels.resize(n_points);
  for (std::size_t point = 0; point < n_points; ++point) {
    int assigned_medoid = -1;
    int assignment_count = 0;
    for (std::size_t facility = 0; facility < n_points; ++facility) {
      const auto index = assignment_index(facility, point, n_points, layout);
      if (assignment_values[index] > 0.5) {
        assigned_medoid = static_cast<int>(facility);
        ++assignment_count;
      }
    }
    if (assignment_count != 1)
      invalid_solution(backend, "point " + std::to_string(point) + " has "
        + std::to_string(assignment_count) + " active assignments; expected one.");
    const int cluster = cluster_by_medoid[static_cast<std::size_t>(assigned_medoid)];
    if (cluster < 0)
      invalid_solution(backend, "point " + std::to_string(point)
        + " is assigned to an unselected medoid.");
    result.labels[point] = cluster;
  }

  return result;
}

ExactClusteringTransaction::ExactClusteringTransaction(Problem &problem)
  : problem_(problem),
    prior_medoids_(problem.centroids_ind),
    prior_labels_(problem.clusters_ind)
{}

ExactClusteringTransaction::~ExactClusteringTransaction() noexcept
{
  if (!published_) {
    problem_.centroids_ind.swap(prior_medoids_);
    problem_.clusters_ind.swap(prior_labels_);
  }
}

void ExactClusteringTransaction::publish(
  core::ClusteringResult result, std::string_view backend)
{
  if (published_)
    invalid_solution(backend, "the result transaction was already published.");
  validate_result(problem_, result, backend);
  problem_.centroids_ind.swap(result.medoid_indices);
  problem_.clusters_ind.swap(result.labels);
  published_ = true;
}

} // namespace dtwc::mip
