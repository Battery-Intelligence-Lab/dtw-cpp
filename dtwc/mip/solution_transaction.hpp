/**
 * @file solution_transaction.hpp
 * @brief Transactional validation and publication for exact MIP clustering.
 */

#pragma once

#include "../core/clustering_result.hpp"
#include "../error.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace dtwc {
class Problem;

namespace mip {

enum class AssignmentMatrixLayout {
  FacilityMajor, ///< A[facility, point] at facility * N + point (HiGHS).
  PointMajor     ///< A[facility, point] at facility + point * N (Gurobi).
};

inline void validate_assignment_matrix_layout(AssignmentMatrixLayout value)
{
  switch (value) {
  case AssignmentMatrixLayout::FacilityMajor:
  case AssignmentMatrixLayout::PointMajor:
    return;
  }
  throw InvalidInput("Invalid AssignmentMatrixLayout value.");
}

/**
 * Decode and validate an exact p-median assignment matrix without mutating a
 * Problem. The result contains exactly k unique medoids and one valid label per
 * point, or throws SolverError with backend context.
 */
[[nodiscard]] core::ClusteringResult extract_exact_clustering(
  std::span<const double> assignment_values,
  std::size_t n_points,
  int n_clusters,
  AssignmentMatrixLayout layout,
  std::string_view backend);

/**
 * Snapshot the caller-visible clustering state and publish a fully validated
 * exact result with no throwing operation between its two vector swaps.
 * Unpublished destruction restores the snapshot, including during unwind.
 */
class ExactClusteringTransaction {
public:
  explicit ExactClusteringTransaction(Problem &problem);
  ExactClusteringTransaction(const ExactClusteringTransaction &) = delete;
  ExactClusteringTransaction &operator=(const ExactClusteringTransaction &) = delete;
  ~ExactClusteringTransaction() noexcept;

  void publish(core::ClusteringResult result, std::string_view backend);

private:
  Problem &problem_;
  std::vector<int> prior_medoids_;
  std::vector<int> prior_labels_;
  bool published_{false};
};

} // namespace mip
} // namespace dtwc
