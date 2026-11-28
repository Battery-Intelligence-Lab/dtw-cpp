/**
 * @file one_batch_pam.cpp
 * @brief Fixed-batch eager PAM with O(Nm) distance storage.
 *
 * @author Volkan Kumtepeli
 * @date 10 Jul 2026
 */

#include "one_batch_pam.hpp"

#include "../Problem.hpp"
#include "../core/dtw_dispatch.hpp"
#include "../core/dtw_kernel.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../core/portable_random.hpp"
#include "../core/thread_scratch.hpp"
#include "../base/env.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace dtwc::algorithms {
namespace {

  std::vector<double> &oob_dists_buf() noexcept
  {
    thread_local std::vector<double> buffer;
    return buffer;
  }

  template <typename T>
  core::PackedLanesPanel<T> &oob_panel_buf() noexcept
  {
    thread_local core::PackedLanesPanel<T> panel;
    return panel;
  }

  std::vector<double> &oob_table_buf() noexcept
  {
    thread_local std::vector<double> buffer;
    return buffer;
  }

  std::size_t automatic_batch_size(std::size_t n)
  {
    if (n <= 1) return n;
    const auto logarithmic = static_cast<std::size_t>(
      20.0 * std::ceil(std::log2(static_cast<double>(n) + 1.0)));
    return std::min(n, std::max<std::size_t>(64, logarithmic));
  }

  void validate_options(std::size_t n, const OneBatchPAMOptions &options)
  {
    if (n == 0)
      throw InvalidInput("one_batch_pam: Problem has no data points.");
    if (options.n_clusters <= 0 || static_cast<std::size_t>(options.n_clusters) > n) {
      throw InvalidInput(
        "one_batch_pam: n_clusters must be in [1, N]. Got n_clusters="
        + std::to_string(options.n_clusters) + ", N=" + std::to_string(n) + ".");
    }
    if (options.batch_size == 0 || options.batch_size < -1)
      throw InvalidInput("one_batch_pam: batch_size must be -1 or a positive integer.");
    // The O(Nm) promise assumes m >= k; an explicit batch below k is refused, not
    // silently raised.
    if (options.batch_size > 0 && options.batch_size < options.n_clusters)
      throw InvalidInput("one_batch_pam: batch_size must be at least n_clusters. Got batch_size="
                         + std::to_string(options.batch_size) + ", n_clusters="
                         + std::to_string(options.n_clusters) + ".");
    if (options.max_iter <= 0)
      throw InvalidInput("one_batch_pam: max_iter must be positive.");
    if (!std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.0)
      throw InvalidInput("one_batch_pam: relative_tolerance must be finite and non-negative.");
  }

  // The accessor for the stored precision, called once and serially before any
  // OpenMP region: it applies a raw `band` write, validates the request (band
  // feasibility, non-finite values) with a typed error that must not be thrown
  // inside a parallel region, and under Soft-DTW computes the self terms
  // Problem::pair_distance reads, so workers only read. Not the Float32 getter
  // for Float64 data: it also rejects variant parameters Float64 never narrows.
  Problem &checked(Problem &problem)
  {
    if (problem.data().is_f32())
      (void)problem.dtw_function_f32();
    else
      (void)problem.dtw_function();
    return problem;
  }

  struct FixedBatchDistances
  {
    Problem &prob;
    std::size_t n;
    std::size_t m;
    std::vector<index_t> sample;
    std::vector<index_t> sample_position;
    std::vector<double> raw;
    std::vector<double> weights;
    double scale = 1.0;
    std::uint64_t evaluations = 0;

    FixedBatchDistances(Problem &problem, std::vector<index_t> batch)
      : prob(checked(problem)),
        n(problem.size()), m(batch.size()), sample(std::move(batch)),
        sample_position(n, -1), raw(n * m, 0.0), weights(m, 0.0)
    {
      for (std::size_t j = 0; j < m; ++j)
        sample_position[static_cast<std::size_t>(sample[j])] = static_cast<index_t>(j);

      // Each worker writes a disjoint row. Row i takes its batch columns W at a
      // time through the lane function (the entry Problem's fill uses, resolved
      // once here; each of its distances is bitwise the per-pair one) where there
      // is no band and the lanes are packed (core::dtw_lanes_packed), or the
      // block's series are as long as series i; every other column, and every
      // request the lanes do not cover, goes pair by pair. The columns stay in
      // batch order, so a block's lanes all run to its longest series. The block
      // at the end of the batch repeats its last column in the lanes past it, whose
      // results are dropped, and a block holding series i computes that self-pair
      // and drops it.
      std::vector<std::uint64_t> row_evaluations(n, 0);
      std::vector<double> row_maxima(n, 0.0);
      const auto dist_cfg = problem.distance();
      const auto block_f32 = core::resolve_dtw_block_fn<float>(dist_cfg);
      const auto block_f64 = core::resolve_dtw_block_fn<data_t>(dist_cfg);
      const auto packed_f32 = core::resolve_dtw_packed_block_fn<float>(dist_cfg);
      const auto packed_f64 = core::resolve_dtw_packed_block_fn<data_t>(dist_cfg);
      // Pre-pack the m batch columns into W-lane panels once when the data is in RAM,
      // ram_limit is unset, and the panel footprint fits 64 MiB; freed immediately
      // after the N rows are computed so peak swap memory is unchanged.
      constexpr std::size_t kMaxBatchPrepackBytes = std::size_t{ 64 } << 20;
      const std::size_t elem_bytes = problem.data().is_f32() ? sizeof(float) : sizeof(data_t);
      const std::size_t lane_w = problem.data().is_f32() ? core::dtw_lanes<float> : core::dtw_lanes<data_t>;
      std::size_t max_sample_len = 0;
      for (std::size_t j = 0; j < m; ++j)
        max_sample_len = std::max(max_sample_len, problem.data().series_flat_size(static_cast<std::size_t>(sample[j])));
      const std::size_t n_batch_panels = (m + lane_w - 1) / lane_w;
      const bool prepack = core::dtw_lanes_packed && problem.ram_limit() == 0 && !problem.data().is_mapped()
                           && n_batch_panels > 0
                           && max_sample_len <= kMaxBatchPrepackBytes / (n_batch_panels * lane_w * elem_bytes);
      std::vector<core::PackedLanesPanel<float>> panels_f32;
      std::vector<core::PackedLanesPanel<data_t>> panels_f64;
      if (prepack) {
        if (problem.data().is_f32() && packed_f32) {
          panels_f32 = core::pack_series_panels<float>(dist_cfg, m, [&](std::size_t j) {
            return problem.data().series_f32(static_cast<std::size_t>(sample[j]));
          });
        } else if (!problem.data().is_f32() && packed_f64) {
          panels_f64 = core::pack_series_panels<data_t>(dist_cfg, m, [&](std::size_t j) {
            return problem.series(static_cast<std::size_t>(sample[j]));
          });
        }
      }

      auto fill_row = [&](std::size_t i) {
        double row_max = 0.0;
        std::uint64_t calls = 0;
        auto fill_blocks = [&](auto x, auto column, const auto &block, const auto &packed_block, const auto &panels) {
          using T = typename decltype(x)::value_type;
          constexpr std::size_t W = core::dtw_lanes<T>;
          std::array<std::span<const T>, W> ys;
          std::array<double, W> lane;
          const bool use_panels = !panels.empty() && static_cast<bool>(packed_block);
          const std::span<const T> x_prep = use_panels ? core::prepare_dtw_query<T>(dist_cfg, x) : x;
          for (std::size_t j0 = 0, b = 0; j0 < m; j0 += W, ++b) {
            const std::size_t count = std::min(W, m - j0);
            bool lanes = false;
            if (use_panels) {
              packed_block(x_prep, panels[b].view(), lane);
              lanes = true;
            } else if (block) {
              lanes = true;
              for (std::size_t w = 0; lanes && w < W; ++w) {
                ys[w] = column(static_cast<std::size_t>(sample[j0 + std::min(w, count - 1)]));
                lanes = core::dtw_lanes_packed || ys[w].size() == x.size();
              }
              if (lanes) block(x, ys, lane);
            }
            for (std::size_t w = 0; w < count; ++w) {
              const auto column_series = static_cast<std::size_t>(sample[j0 + w]);
              double d = 0.0;
              if (i != column_series) {
                d = lanes ? lane[w] : dtw(i, column_series);
                ++calls;
              }
              if (!std::isfinite(d) || d < 0.0)
                throw InvalidInput("one_batch_pam: distance function returned a non-finite or negative value.");
              raw[i * m + j0 + w] = d;
              row_max = std::max(row_max, d);
            }
          }
        };
        if (prob.data().is_f32())
          fill_blocks(prob.data().series_f32(i), [&](std::size_t c) { return prob.data().series_f32(c); }, block_f32, packed_f32, panels_f32);
        else
          fill_blocks(prob.series(i), [&](std::size_t c) { return prob.series(c); }, block_f64, packed_f64, panels_f64);
        row_evaluations[i] = calls;
        row_maxima[i] = row_max;
      };
      run_openmp(fill_row, n, n > 64);
      panels_f32.clear();
      panels_f32.shrink_to_fit();
      panels_f64.clear();
      panels_f64.shrink_to_fit();
      const double table_max = *std::max_element(row_maxima.begin(), row_maxima.end());
      scale = table_max > 0.0 ? table_max : 1.0;
      evaluations = std::accumulate(row_evaluations.begin(), row_evaluations.end(), std::uint64_t{ 0 });

      // Count/mean NNIW (Loog 2012) Voronoi-cell mass weights, deliberately paired
      // with the obpam experiment code's finite-table-maximum diagonal correction
      // in estimate(): neither the paper's literal +infinity nor OneBatchPAM
      // v0.1.0 describes this hybrid.
      for (std::size_t i = 0; i < n; ++i) {
        std::size_t nearest = 0;
        double best = raw[i * m];
        for (std::size_t j = 1; j < m; ++j) {
          const double d = raw[i * m + j];
          if (d < best) {
            best = d;
            nearest = j;
          }
        }
        weights[nearest] += 1.0;
      }
      const double mean = static_cast<double>(n) / static_cast<double>(m);
      for (double &weight : weights) weight /= mean;
    }

    double dtw(std::size_t a, std::size_t b) const { return prob.pair_distance(a, b); }

    double estimate(std::size_t candidate, std::size_t batch_column) const
    {
      // The authors' obpam experiment code replaces d(x,x)=0 by the actual finite
      // table maximum, i.e. exactly 1 after normalization. Do not use the
      // zero-table fallback scale as an unnormalized replacement value.
      if (candidate == static_cast<std::size_t>(sample[batch_column]))
        return weights[batch_column];
      return (raw[candidate * m + batch_column] / scale) * weights[batch_column];
    }

    /// Distance of a point to a medoid: the table entry when the medoid is in the
    /// batch, else one DTW call, added to the caller's own tally `calls`. Const,
    /// so every worker may call it on its own point.
    double exact(std::size_t point, index_t medoid, std::uint64_t &calls) const
    {
      const index_t column = sample_position[static_cast<std::size_t>(medoid)];
      if (column >= 0)
        return raw[point * m + static_cast<std::size_t>(column)];
      if (point == static_cast<std::size_t>(medoid)) return 0.0;
      ++calls;
      return dtw(point, static_cast<std::size_t>(medoid));
    }
  };

  void nearest_two(const FixedBatchDistances &distances,
                   const std::vector<index_t> &medoids,
                   std::vector<index_t> &nearest,
                   std::vector<double> &nearest_distance,
                   std::vector<double> &second_distance)
  {
    const std::size_t m = distances.m;
    const auto k = static_cast<index_t>(medoids.size());
    nearest.assign(m, 0);
    nearest_distance.assign(m, std::numeric_limits<double>::infinity());
    second_distance.assign(m, std::numeric_limits<double>::infinity());
    for (std::size_t j = 0; j < m; ++j) {
      for (index_t slot = 0; slot < k; ++slot) {
        const double d = distances.estimate(static_cast<std::size_t>(medoids[slot]), j);
        if (d < nearest_distance[j]) {
          second_distance[j] = nearest_distance[j];
          nearest_distance[j] = d;
          nearest[j] = slot;
        } else if (d < second_distance[j]) {
          second_distance[j] = d;
        }
      }
    }
  }

  double estimated_cost(const std::vector<double> &nearest_distance)
  {
    return std::accumulate(nearest_distance.begin(), nearest_distance.end(), 0.0);
  }

} // namespace

namespace detail {

  std::size_t resolve_batch_size(std::size_t n, index_t k, index_t requested)
  {
    const std::size_t m = requested > 0 ? std::min(n, static_cast<std::size_t>(requested)) : automatic_batch_size(n);
    return std::max(m, static_cast<std::size_t>(k));
  }

} // namespace detail

core::ClusteringResult one_batch_pam(Problem &prob,
                                     const OneBatchPAMOptions &options,
                                     OneBatchPAMStats *stats)
{
  // A Problem on a GPU, or one with a forced share, asks for work this method has no part in (it computes its
  // distances on the CPU as it goes): refused here, as Problem::cluster refuses it, not run on the CPU without a word;
  // a Problem given `fallback` computes on the CPU, and says so. A forced share is refused first, whatever the
  // fallback; cpu+gpu's goes with its device, as no CPU run splits a share.
  if (prob.gpu_share() != -1.0 && prob.device().first != Device::CPU_GPU)
    throw InvalidInput(dtwc::detail::share_ignored_message(
      "one_batch_pam", "the distance matrix's fill and FastCLARA's pass", prob.device().first));
  const dtwc::detail::DeviceScope device_scope(prob, "one_batch_pam", Method::OneBatch);
  const std::size_t n = prob.size();
  validate_options(n, options);
  const index_t k = options.n_clusters;

  if (k == static_cast<index_t>(n)) {
    core::ClusteringResult result;
    result.labels.resize(n);
    result.medoid_indices.resize(n);
    std::iota(result.labels.begin(), result.labels.end(), index_t{ 0 });
    std::iota(result.medoid_indices.begin(), result.medoid_indices.end(), index_t{ 0 });
    result.converged = true;
    prob.set_result(result);
    if (stats) *stats = OneBatchPAMStats{};
    return result;
  }

  // The automatic size is an internal policy: it rises to k silently (an
  // explicit batch below k was refused above).
  const std::size_t m = detail::resolve_batch_size(n, k, options.batch_size);

  std::mt19937_64 rng(options.random_seed);
  std::vector<index_t> permutation(n);
  std::iota(permutation.begin(), permutation.end(), index_t{ 0 });
  core::portable_shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<index_t> sample(permutation.begin(), permutation.begin() + static_cast<std::ptrdiff_t>(m));
  // The paper draws the candidate initialization independently of the fixed
  // batch: sampled points remain eligible, just like every other point.
  core::portable_shuffle(permutation.begin(), permutation.end(), rng);
  std::vector<index_t> medoids(permutation.begin(), permutation.begin() + k);

  FixedBatchDistances distances(prob, std::move(sample));
  std::vector<bool> is_medoid(n, false);
  for (index_t medoid : medoids) is_medoid[static_cast<std::size_t>(medoid)] = true;

  std::vector<index_t> nearest;
  std::vector<double> nearest_distance;
  std::vector<double> second_distance;
  nearest_two(distances, medoids, nearest, nearest_distance, second_distance);

  int accepted_swaps = 0;
  int sweeps = 0;
  bool converged = false;

  if (k == 1) {
    index_t best = medoids[0];
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0; candidate < n; ++candidate) {
      double cost = 0.0;
      for (std::size_t j = 0; j < m; ++j) cost += distances.estimate(candidate, j);
      if (cost < best_cost || (cost == best_cost && static_cast<index_t>(candidate) < best)) {
        best = static_cast<index_t>(candidate);
        best_cost = cost;
      }
    }
    medoids[0] = best;
    nearest_two(distances, medoids, nearest, nearest_distance, second_distance);
    converged = true;
    sweeps = 1;
  } else {
    // Recompute removal gains and tolerance only after an accepted swap.
    std::vector<double> base_removal_gain(static_cast<std::size_t>(k), 0.0);
    std::vector<double> removal_gain(static_cast<std::size_t>(k), 0.0);
    std::vector<double> scaled_w(m);
    for (std::size_t j = 0; j < m; ++j) scaled_w[j] = distances.weights[j] / distances.scale;
    double tolerance = 0.0;
    const auto refresh_swap_state = [&] {
      std::fill(base_removal_gain.begin(), base_removal_gain.end(), 0.0);
      for (std::size_t j = 0; j < m; ++j)
        base_removal_gain[static_cast<std::size_t>(nearest[j])] += nearest_distance[j] - second_distance[j];
      tolerance = options.relative_tolerance * estimated_cost(nearest_distance);
    };
    refresh_swap_state();

    for (; sweeps < options.max_iter; ++sweeps) {
      bool changed = false;
      for (std::size_t candidate = 0; candidate < n; ++candidate) {
        if (is_medoid[candidate]) continue;

        removal_gain = base_removal_gain;

        double add_gain = 0.0;
        const double *const cand_row = distances.raw.data() + candidate * m;
        const double *const sw_col = scaled_w.data();
        const double *const w_col = distances.weights.data();
        const auto accumulate_col = [&](std::size_t j, double d) {
          const index_t slot = nearest[j];
          if (d < nearest_distance[j]) {
            add_gain += nearest_distance[j] - d;
            removal_gain[static_cast<std::size_t>(slot)] += second_distance[j] - nearest_distance[j];
          } else if (d < second_distance[j]) {
            removal_gain[static_cast<std::size_t>(slot)] += second_distance[j] - d;
          }
        };
        const index_t self_pos = distances.sample_position[candidate];
        if (self_pos < 0) {
          for (std::size_t j = 0; j < m; ++j)
            accumulate_col(j, cand_row[j] * sw_col[j]);
        } else {
          const auto self_j = static_cast<std::size_t>(self_pos);
          for (std::size_t j = 0; j < self_j; ++j)
            accumulate_col(j, cand_row[j] * sw_col[j]);
          accumulate_col(self_j, w_col[self_j]);
          for (std::size_t j = self_j + 1; j < m; ++j)
            accumulate_col(j, cand_row[j] * sw_col[j]);
        }

        const auto best_it = std::max_element(removal_gain.begin(), removal_gain.end());
        const index_t slot = std::distance(removal_gain.begin(), best_it);
        const double gain = add_gain + *best_it;
        if (gain > tolerance) {
          is_medoid[static_cast<std::size_t>(medoids[slot])] = false;
          medoids[slot] = static_cast<index_t>(candidate);
          is_medoid[candidate] = true;
          nearest_two(distances, medoids, nearest, nearest_distance, second_distance);
          refresh_swap_state();
          ++accepted_swaps;
          changed = true;
        }
      }
      if (!changed) {
        converged = true;
        ++sweeps;
        break;
      }
    }
  }

  core::ClusteringResult result;
  result.medoid_indices = medoids;
  result.labels.resize(n);
  std::vector<double> point_cost(n, 0.0);
  std::vector<std::uint64_t> point_evaluations(n, 0);

  // Race-free by design: point p alone writes labels[p], point_cost[p] and
  // point_evaluations[p]; the call count and the objective are combined serially
  // after the region, in point order, so neither depends on the thread count.
  // exact() reaches medoids outside the batch, which the table's finiteness check
  // never saw: unguarded, a non-finite d would leave the point silently on label 0
  // where fast_pam and fast_clara throw. run_openmp rethrows the lowest point's
  // failure, the one a serial scan meets first.
  std::vector<index_t> oob_slots;
  std::vector<index_t> oob_idx_of_slot(static_cast<std::size_t>(k), -1);
  for (index_t slot = 0; slot < k; ++slot) {
    if (distances.sample_position[static_cast<std::size_t>(medoids[slot])] < 0) {
      oob_idx_of_slot[static_cast<std::size_t>(slot)] = static_cast<index_t>(oob_slots.size());
      oob_slots.push_back(slot);
    }
  }

  const auto assign_scalar = [&] {
    auto assign_point = [&](std::size_t point) {
      double best = std::numeric_limits<double>::infinity();
      index_t label = 0;
      std::uint64_t calls = 0;
      for (index_t slot = 0; slot < k; ++slot) {
        const double d = core::detail::require_finite_medoid_distance(
          distances.exact(point, medoids[slot], calls), "one_batch_pam", point, slot, medoids[slot]);
        if (d < best) {
          best = d;
          label = slot;
        }
      }
      result.labels[point] = label;
      point_cost[point] = best;
      point_evaluations[point] = calls;
    };
    run_openmp(assign_point, n, n > 64);
  };

  const auto dist_cfg = prob.distance();
  const bool use_f32_lanes = prob.data().is_f32();
  const auto packed_f32 = use_f32_lanes ? core::resolve_dtw_packed_block_fn<float>(dist_cfg) : decltype(core::resolve_dtw_packed_block_fn<float>(dist_cfg)){};
  const auto packed_f64 = !use_f32_lanes ? core::resolve_dtw_packed_block_fn<data_t>(dist_cfg) : decltype(core::resolve_dtw_packed_block_fn<data_t>(dist_cfg)){};

  if (!oob_slots.empty() && core::dtw_lanes_packed && (packed_f32 || packed_f64)) {
    auto assign_simd = [&](auto series_of, const auto &packed_block) {
      using T = typename decltype(series_of(std::size_t{ 0 }))::value_type;
      constexpr std::size_t W = core::dtw_lanes<T>;
      const std::size_t n_oob = oob_slots.size();
      const index_t *const col_of_slot = oob_idx_of_slot.data();

      auto assign_from_oob = [&](std::size_t point, auto oob_dist) {
        double best = std::numeric_limits<double>::infinity();
        index_t label = 0;
        std::uint64_t calls = 0;
        for (index_t slot = 0; slot < k; ++slot) {
          const index_t med = medoids[slot];
          double raw_d = 0.0;
          const index_t col = distances.sample_position[static_cast<std::size_t>(med)];
          if (col >= 0) {
            raw_d = distances.raw[point * m + static_cast<std::size_t>(col)];
          } else if (point != static_cast<std::size_t>(med)) {
            raw_d = oob_dist(static_cast<std::size_t>(col_of_slot[slot]));
            ++calls;
          }
          const double d = core::detail::require_finite_medoid_distance(
            raw_d, "one_batch_pam", point, slot, med);
          if (d < best) {
            best = d;
            label = slot;
          }
        }
        result.labels[point] = label;
        point_cost[point] = best;
        point_evaluations[point] = calls;
      };

      if (n_oob >= W) {
        const auto oob_panels = core::pack_series_panels<T>(dist_cfg, n_oob, [&](std::size_t u) {
          return series_of(static_cast<std::size_t>(medoids[oob_slots[u]]));
        });
        const std::size_t n_panels = oob_panels.size();
        auto assign_point = [&](std::size_t point) {
          std::vector<double> &oob_dists = oob_dists_buf();
          if (oob_dists.size() < n_panels * W) oob_dists.resize(n_panels * W);
          const auto x_prep = core::prepare_dtw_query<T>(dist_cfg, series_of(point));
          for (std::size_t b = 0; b < n_panels; ++b)
            packed_block(x_prep, oob_panels[b].view(), std::span<double>(oob_dists.data() + b * W, W));
          assign_from_oob(point, [&oob_dists](std::size_t oob_idx) { return oob_dists[oob_idx]; });
        };
        run_openmp(assign_point, n, n > 64);
      } else {
        core::PreparedQueries<T> prep_oob;
        prep_oob.build(dist_cfg, n_oob, [&](std::size_t u) {
          return series_of(static_cast<std::size_t>(medoids[oob_slots[u]]));
        });

        const std::size_t n_blocks = (n + W - 1) / W;
        auto assign_block = [&](std::size_t b) {
          const std::size_t i0 = b * W, count = std::min(W, n - i0);
          std::array<std::span<const T>, W> ys;
          for (std::size_t w = 0; w < W; ++w)
            ys[w] = series_of(i0 + std::min(w, count - 1));
          core::PackedLanesPanel<T> &panel = oob_panel_buf<T>();
          core::pack_dtw_panel<T>(dist_cfg, ys.data(), panel);
          const auto panel_view = panel.view();
          std::vector<double> &oob_table = oob_table_buf();
          if (oob_table.size() < n_oob * W) oob_table.resize(n_oob * W);
          for (std::size_t u = 0; u < n_oob; ++u)
            packed_block(prep_oob.views[u], panel_view, std::span<double>(oob_table.data() + u * W, W));

          for (std::size_t w = 0; w < count; ++w)
            assign_from_oob(i0 + w, [&oob_table, w, W](std::size_t oob_idx) { return oob_table[oob_idx * W + w]; });
        };
        run_openmp(assign_block, n_blocks, n > 64);
      }
    };
    if (use_f32_lanes)
      assign_simd([&](std::size_t i) { return prob.data().series_f32(i); }, packed_f32);
    else
      assign_simd([&](std::size_t i) { return prob.series(i); }, packed_f64);
  } else {
    assign_scalar();
  }
  distances.evaluations += std::accumulate(point_evaluations.begin(), point_evaluations.end(), std::uint64_t{ 0 });
  // A medoid tied with another medoid (a duplicate series) serves itself, or its
  // own cluster would be published empty. Its own distance is exactly 0, so a
  // best of 0 is that tie. After the scan, so the scan's min stays branch-free.
  for (index_t slot = 0; slot < k; ++slot) {
    const auto medoid = static_cast<std::size_t>(medoids[slot]);
    if (point_cost[medoid] == 0.0) result.labels[medoid] = slot;
  }
  // Point-ordered accumulation: the published objective is a cross-route byte
  // contract, so it uses the same reassociation-proof accumulator as fast_pam
  // and fast_clara rather than a plain std::accumulate.
  result.total_cost = core::detail::ordered_medoid_objective(point_cost, "one_batch_pam");
  result.iterations = sweeps;
  result.converged = converged;

  prob.set_result(result);

  if (stats) {
    stats->batch_size = m;
    stats->distance_evaluations = distances.evaluations;
    stats->full_matrix_fraction = static_cast<double>(
      static_cast<long double>(distances.evaluations) / (static_cast<long double>(n) * static_cast<long double>(n)));
    stats->estimated_objective = estimated_cost(nearest_distance);
    stats->accepted_swaps = accepted_swaps;
  }
  return result;
}

} // namespace dtwc::algorithms

namespace dtwc::core::detail {

std::size_t one_batch_pam_scratch(bool release) noexcept
{
  return held_bytes(release,
                    dtwc::algorithms::oob_dists_buf(),
                    dtwc::algorithms::oob_panel_buf<float>().rows,
                    dtwc::algorithms::oob_panel_buf<data_t>().rows,
                    dtwc::algorithms::oob_table_buf());
}

} // namespace dtwc::core::detail
