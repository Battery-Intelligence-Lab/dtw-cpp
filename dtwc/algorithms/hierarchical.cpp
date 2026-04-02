/**
 * @file hierarchical.cpp
 * @brief Agglomerative hierarchical clustering implementation.
 *
 * @details Generic O(N^3) agglomerative algorithm with Lance-Williams update
 * for single, complete, and average (UPGMA) linkage. Deterministic
 * tie-breaking: among equal-distance pairs the lexicographically smallest
 * (a, b) pair (with a < b) is chosen first.
 *
 * @date 02 Apr 2026
 */

#include "hierarchical.hpp"
#include "../Problem.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dtwc::algorithms {

// ---------------------------------------------------------------------------
// build_dendrogram
// ---------------------------------------------------------------------------

Dendrogram build_dendrogram(Problem &prob, const HierarchicalOptions &opts)
{
  const int N = static_cast<int>(prob.size());

  if (N > opts.max_points)
    throw std::runtime_error(
      "build_dendrogram: N=" + std::to_string(N) +
      " exceeds max_points=" + std::to_string(opts.max_points));

  if (!prob.isDistanceMatrixFilled())
    throw std::runtime_error(
      "build_dendrogram: distance matrix is not fully computed. "
      "Call prob.fillDistanceMatrix() first.");

  // -------------------------------------------------------------------------
  // Copy pairwise distances into a flat working array (indexed i*N+j).
  // We only maintain the upper-triangle in spirit but access both (i,j) and
  // (j,i) for simplicity; the matrix is symmetric.
  // -------------------------------------------------------------------------
  std::vector<double> work(static_cast<size_t>(N) * N, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      work[static_cast<size_t>(i) * N + j] = prob.distByInd(i, j);

  // active[i] == true  →  cluster i is still alive.
  std::vector<bool> active(N, true);

  // size[i] = current number of original points in cluster i.
  std::vector<int> sz(N, 1);

  Dendrogram dend;
  dend.n_points = N;
  dend.merges.reserve(static_cast<size_t>(N - 1));

  // -------------------------------------------------------------------------
  // N-1 merge steps
  // -------------------------------------------------------------------------
  for (int step = 0; step < N - 1; ++step) {
    // Find the minimum distance among all pairs of active clusters.
    // Tie-breaking: lexicographically smallest (a, b) with a < b.
    double best_dist = std::numeric_limits<double>::infinity();
    int best_a = -1, best_b = -1;

    for (int i = 0; i < N; ++i) {
      if (!active[i]) continue;
      for (int j = i + 1; j < N; ++j) {
        if (!active[j]) continue;
        double d = work[static_cast<size_t>(i) * N + j];
        if (d < best_dist ||
            (d == best_dist && (i < best_a || (i == best_a && j < best_b)))) {
          best_dist = d;
          best_a = i;
          best_b = j;
        }
      }
    }

    // Record the merge (a < b is guaranteed by construction).
    dend.merges.push_back({ best_a, best_b, best_dist, sz[best_a] + sz[best_b] });

    // Update working distances from every other active cluster c to the merged
    // cluster (kept under index best_a) using the Lance-Williams formula.
    const int sz_a = sz[best_a];
    const int sz_b = sz[best_b];

    for (int c = 0; c < N; ++c) {
      if (!active[c] || c == best_a || c == best_b) continue;

      double d_ac = work[static_cast<size_t>(best_a) * N + c];
      double d_bc = work[static_cast<size_t>(best_b) * N + c];
      double d_new{};

      switch (opts.linkage) {
        case Linkage::Single:
          d_new = std::min(d_ac, d_bc);
          break;
        case Linkage::Complete:
          d_new = std::max(d_ac, d_bc);
          break;
        case Linkage::Average:
          d_new = (static_cast<double>(sz_a) * d_ac +
                   static_cast<double>(sz_b) * d_bc) /
                  static_cast<double>(sz_a + sz_b);
          break;
      }

      // Write symmetrically so future lookups see the updated value.
      work[static_cast<size_t>(best_a) * N + c] = d_new;
      work[static_cast<size_t>(c) * N + best_a] = d_new;
    }

    // Update cluster size and deactivate the larger-index cluster.
    sz[best_a] += sz[best_b];
    active[best_b] = false;
  }

  return dend;
}

// ---------------------------------------------------------------------------
// Union-Find helpers (path compression + union-by-rank)
// ---------------------------------------------------------------------------
namespace {

struct UF {
  std::vector<int> parent, rank_;

  explicit UF(int n) : parent(n), rank_(n, 0)
  {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x)
  {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path compression (halving)
      x = parent[x];
    }
    return x;
  }

  void unite(int a, int b)
  {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
  }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// cut_dendrogram
// ---------------------------------------------------------------------------

core::ClusteringResult cut_dendrogram(const Dendrogram &dend, Problem &prob, int k)
{
  const int N = dend.n_points;
  if (k < 1 || k > N)
    throw std::runtime_error(
      "cut_dendrogram: k=" + std::to_string(k) +
      " out of range [1, " + std::to_string(N) + "]");

  // Replay only the first (N-k) merges — the last k-1 merges produce the
  // final k clusters (we skip those k-1 merges, keeping them separate).
  const int n_merges_to_apply = N - k;

  UF uf(N);
  for (int i = 0; i < n_merges_to_apply; ++i) {
    const auto &step = dend.merges[static_cast<size_t>(i)];
    uf.unite(step.cluster_a, step.cluster_b);
  }

  // Map canonical root → contiguous cluster label [0, k).
  // We assign labels in order of smallest root index for determinism.
  std::vector<int> root_to_label(N, -1);
  int next_label = 0;
  std::vector<int> labels(N);

  for (int i = 0; i < N; ++i) {
    int root = uf.find(i);
    if (root_to_label[root] == -1)
      root_to_label[root] = next_label++;
    labels[i] = root_to_label[root];
  }

  // Collect members per cluster.
  std::vector<std::vector<int>> members(k);
  for (int i = 0; i < N; ++i)
    members[static_cast<size_t>(labels[i])].push_back(i);

  // Find medoid per cluster: point minimising sum of distances to cluster peers.
  // Tie-breaking: smallest original index wins.
  std::vector<int> medoid_indices(k, -1);
  double total_cost = 0.0;

  for (int cl = 0; cl < k; ++cl) {
    const auto &mem = members[static_cast<size_t>(cl)];
    double best_cost = std::numeric_limits<double>::infinity();
    int best_idx = mem[0]; // fallback (smallest index in cluster)

    for (int cand : mem) {
      double cost = 0.0;
      for (int other : mem)
        cost += prob.distByInd(cand, other);

      if (cost < best_cost ||
          (cost == best_cost && cand < best_idx)) {
        best_cost = cost;
        best_idx = cand;
      }
    }

    medoid_indices[static_cast<size_t>(cl)] = best_idx;

    // Total cost = sum of distances of each point to its medoid.
    for (int pt : mem)
      total_cost += prob.distByInd(pt, best_idx);
  }

  core::ClusteringResult result;
  result.labels = std::move(labels);
  result.medoid_indices = std::move(medoid_indices);
  result.total_cost = total_cost;
  result.converged = true;
  result.iterations = N - 1; // dendrogram always completes in N-1 steps

  return result;
}

} // namespace dtwc::algorithms
