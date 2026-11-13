/**
 * @file hierarchical.cpp
 * @brief Agglomerative hierarchical clustering implementation.
 *
 * @details Generic O(N^3) agglomerative algorithm with Lance-Williams update
 * for single, complete, and average (UPGMA) linkage. Deterministic
 * tie-breaking: among equal-distance pairs the lexicographically smallest
 * (a, b) pair (with a < b) is chosen first.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include "hierarchical.hpp"
#include "../Problem.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

namespace dtwc::algorithms {

// ---------------------------------------------------------------------------
// build_dendrogram
// ---------------------------------------------------------------------------

Dendrogram build_dendrogram(Problem &prob, const HierarchicalOptions &opts)
{
  const index_t N = prob.size();

  if (N > opts.max_points)
    throw InvalidInput(
      "build_dendrogram: N=" + std::to_string(N) +
      " exceeds max_points=" + std::to_string(opts.max_points));

  prob.fill_distance_matrix();

  Dendrogram dend;
  dend.n_points = N;
  if (N <= 1) return dend;
  dend.merges.reserve(static_cast<size_t>(N - 1));

  // -------------------------------------------------------------------------
  // Copy pairwise distances into a packed lower-triangular working array
  // (half the memory of a dense N×N copy, widened once in a linear pass).
  // -------------------------------------------------------------------------
  const std::size_t n_packed = core::packed_size(static_cast<std::size_t>(N));
  std::vector<double> work(n_packed);
  prob.distance_matrix().visit([&](const auto *src) {
    for (std::size_t k = 0; k < n_packed; ++k)
      work[k] = static_cast<double>(src[k]);
  });

  auto at = [&](index_t a, index_t b) -> double & {
    return work[core::tri_index(static_cast<std::size_t>(a), static_cast<std::size_t>(b))];
  };

  // active[i] != 0  →  cluster i is still alive.
  std::vector<char> active(static_cast<std::size_t>(N), 1);

  // sz[i] = current number of original points in cluster i.
  std::vector<index_t> sz(static_cast<std::size_t>(N), 1);

  // Day & Edelsbrunner (1984) per-row nearest-neighbour cache over j > i:
  // reduces total agglomerative work from O(N^3) to O(N^2) while preserving
  // bit-identical merge distances and lexicographical (a, b) tie-breaking.
  constexpr double kInf = std::numeric_limits<double>::infinity();
  std::vector<double> nn_dist(static_cast<std::size_t>(N), kInf);
  std::vector<index_t> nn_idx(static_cast<std::size_t>(N), -1);

  auto recompute_row_nn = [&](index_t i) {
    double best_d = kInf;
    index_t best_j = -1;
    for (index_t j = i + 1; j < N; ++j) {
      if (!active[static_cast<std::size_t>(j)]) continue;
      const double d = work[static_cast<std::size_t>(j) * static_cast<std::size_t>(j + 1) / 2
                            + static_cast<std::size_t>(i)];
      if (d < best_d) {
        best_d = d;
        best_j = j;
      }
    }
    nn_dist[static_cast<std::size_t>(i)] = best_d;
    nn_idx[static_cast<std::size_t>(i)] = best_j;
  };

  for (index_t i = 0; i < N - 1; ++i)
    recompute_row_nn(i);

  // -------------------------------------------------------------------------
  // N-1 merge steps (O(N) per step -> O(N^2) total)
  // -------------------------------------------------------------------------
  for (index_t step = 0; step < N - 1; ++step) {
    double best_dist = kInf;
    index_t best_a = -1, best_b = -1;

    for (index_t i = 0; i < N - 1; ++i) {
      if (!active[static_cast<std::size_t>(i)]) continue;
      const index_t j = nn_idx[static_cast<std::size_t>(i)];
      if (j < 0) continue;
      const double d = nn_dist[static_cast<std::size_t>(i)];
      if (d < best_dist ||
          (d == best_dist && (i < best_a || (i == best_a && j < best_b)))) {
        best_dist = d;
        best_a = i;
        best_b = j;
      }
    }

    // Record the merge (best_a < best_b is guaranteed by construction).
    dend.merges.push_back({ best_a, best_b, best_dist,
                            sz[static_cast<std::size_t>(best_a)] + sz[static_cast<std::size_t>(best_b)] });

    const index_t sz_a = sz[static_cast<std::size_t>(best_a)];
    const index_t sz_b = sz[static_cast<std::size_t>(best_b)];
    const double wa = static_cast<double>(sz_a) / static_cast<double>(sz_a + sz_b);
    const double wb = static_cast<double>(sz_b) / static_cast<double>(sz_a + sz_b);

    // Deactivate best_b first so row rescans never select it.
    sz[static_cast<std::size_t>(best_a)] = sz_a + sz_b;
    active[static_cast<std::size_t>(best_b)] = 0;

    // Lance-Williams update on D(best_a, c) for all other active clusters c.
    for (index_t c = 0; c < N; ++c) {
      if (!active[static_cast<std::size_t>(c)] || c == best_a) continue;

      const double d_ac = at(best_a, c);
      const double d_bc = at(best_b, c);
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
          (void)wa; (void)wb;
          break;
      }

      at(best_a, c) = d_new;

      // Maintain nn cache for rows c < best_a where best_a > c is in c's upper scan.
      if (c < best_a) {
        const auto cz = static_cast<std::size_t>(c);
        if (d_new < nn_dist[cz] || (d_new == nn_dist[cz] && best_a <= nn_idx[cz])) {
          nn_dist[cz] = d_new;
          nn_idx[cz] = best_a;
        } else if (nn_idx[cz] == best_a || nn_idx[cz] == best_b) {
          recompute_row_nn(c);
        }
      } else if (c < best_b && nn_idx[static_cast<std::size_t>(c)] == best_b) {
        // For best_a < c < best_b, best_a is below c so only best_b's removal can invalidate c's NN.
        recompute_row_nn(c);
      }
    }

    if (best_a < N - 1)
      recompute_row_nn(best_a);
  }

  return dend;
}

// ---------------------------------------------------------------------------
// Union-Find helpers (path compression + union-by-rank)
// ---------------------------------------------------------------------------
namespace {

struct UF {
  std::vector<index_t> parent, rank_;

  explicit UF(index_t n) : parent(n), rank_(n, 0)
  {
    std::iota(parent.begin(), parent.end(), index_t{ 0 });
  }

  index_t find(index_t x)
  {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path compression (halving)
      x = parent[x];
    }
    return x;
  }

  void unite(index_t a, index_t b)
  {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
  }
};

/// Each cluster's medoid, the member whose distances to its peers sum least (the smallest index on a tie), into
/// `medoid_indices`; returns the cost, each point's distance to its medoid summed in cluster order. `matrix` holds
/// the packed distances at their width; every sum runs in double. The inner loops index the members, not a
/// range-for: so clang 21 vectorises the peer sums as it did the loop on dist_by_ind, in the same order (2026-10-11).
template <typename T>
double medoids_and_cost(const T *matrix, const std::vector<std::vector<index_t>> &members,
                        std::vector<index_t> &medoid_indices)
{
  double total_cost = 0.0;
  for (std::size_t cl = 0; cl < members.size(); ++cl) {
    const index_t *const mem = members[cl].data();
    const std::size_t size = members[cl].size();
    double best_cost = std::numeric_limits<double>::infinity();
    index_t best_idx = mem[0]; // fallback (smallest index in cluster)

    for (const index_t cand : members[cl]) {
      double cost = 0.0;
      for (std::size_t q = 0; q < size; ++q)
        cost += matrix[core::tri_index(static_cast<std::size_t>(cand), static_cast<std::size_t>(mem[q]))];

      if (cost < best_cost ||
          (cost == best_cost && cand < best_idx)) {
        best_cost = cost;
        best_idx = cand;
      }
    }

    medoid_indices[cl] = best_idx;

    // Total cost = sum of distances of each point to its medoid.
    for (std::size_t q = 0; q < size; ++q)
      total_cost += matrix[core::tri_index(static_cast<std::size_t>(mem[q]), static_cast<std::size_t>(best_idx))];
  }
  return total_cost;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// cut_dendrogram
// ---------------------------------------------------------------------------

core::ClusteringResult cut_dendrogram(const Dendrogram &dend, Problem &prob, index_t k)
{
  const index_t N = dend.n_points;

  // `Dendrogram` is a public, default-constructible aggregate exported to
  // Python, so a hand-built one must be validated, not trusted: otherwise it
  // reads dend.merges out of bounds and indexes UF::parent with arbitrary ids.
  if (N != prob.size())
    throw InvalidInput(
      "cut_dendrogram: dendrogram n_points=" + std::to_string(N) +
      " does not match Problem size=" + std::to_string(prob.size()) + ".");
  if (N < 1)
    throw InvalidInput("cut_dendrogram: dendrogram n_points must be >= 1.");
  if (dend.merges.size() != static_cast<size_t>(N - 1))
    throw InvalidInput(
      "cut_dendrogram: dendrogram has " + std::to_string(dend.merges.size()) +
      " merge steps; a dendrogram over " + std::to_string(N) + " points needs exactly "
      + std::to_string(N - 1) + ".");
  for (size_t i = 0; i < dend.merges.size(); ++i) {
    const auto &step = dend.merges[i];
    if (step.cluster_a < 0 || step.cluster_a >= N || step.cluster_b < 0 || step.cluster_b >= N)
      throw InvalidInput(
        "cut_dendrogram: merge step " + std::to_string(i) + " references cluster ids ("
        + std::to_string(step.cluster_a) + ", " + std::to_string(step.cluster_b)
        + ") outside [0, " + std::to_string(N) + ").");
  }

  if (k < 1 || k > N)
    throw InvalidInput(
      "cut_dendrogram: k=" + std::to_string(k) +
      " out of range [1, " + std::to_string(N) + "]");

  // Replay only the first (N-k) merges — the last k-1 merges produce the
  // final k clusters (we skip those k-1 merges, keeping them separate).
  const index_t n_merges_to_apply = N - k;

  UF uf(N);
  for (index_t i = 0; i < n_merges_to_apply; ++i) {
    const auto &step = dend.merges[static_cast<size_t>(i)];
    uf.unite(step.cluster_a, step.cluster_b);
  }

  // Map canonical root → contiguous cluster label [0, k).
  // We assign labels in order of smallest root index for determinism.
  std::vector<index_t> root_to_label(N, -1);
  index_t next_label = 0;
  std::vector<index_t> labels(N);

  for (index_t i = 0; i < N; ++i) {
    const index_t root = uf.find(i);
    if (root_to_label[root] == -1)
      root_to_label[root] = next_label++;
    labels[i] = root_to_label[root];
  }

  // A degenerate merge list (repeated or self-merges) unites fewer than N-k
  // distinct pairs and leaves more than k components — guard before sizing.
  if (next_label != k)
    throw InvalidInput(
      "cut_dendrogram: replaying the first " + std::to_string(n_merges_to_apply) +
      " merges left " + std::to_string(next_label) + " components, expected "
      + std::to_string(k) + "; the merge list is not a valid dendrogram.");

  // Collect members per cluster.
  std::vector<std::vector<index_t>> members(k);
  for (index_t i = 0; i < N; ++i)
    members[static_cast<size_t>(labels[i])].push_back(i);

  // Find medoid per cluster: point minimising sum of distances to cluster peers.
  // Tie-breaking: smallest original index wins. The pairs within each cluster
  // come from the matrix, which build_dendrogram filled.
  prob.fill_distance_matrix();
  std::vector<index_t> medoid_indices(k, -1);
  const double total_cost = prob.distance_matrix().visit(
    [&](const auto *matrix) { return medoids_and_cost(matrix, members, medoid_indices); });

  core::ClusteringResult result;
  result.labels = std::move(labels);
  result.medoid_indices = std::move(medoid_indices);
  result.total_cost = total_cost;
  result.converged = true;
  result.iterations = static_cast<int>(N - 1); // dendrogram always completes in N-1 steps

  prob.set_result(result); // scores::silhouette(prob) etc. read it back

  return result;
}

} // namespace dtwc::algorithms
