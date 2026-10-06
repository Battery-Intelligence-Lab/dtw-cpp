/**
 * @file barycenter.cpp
 * @brief SSG/DBA/soft-DTW barycenters and a sequence-centroid k-means driver.
 */

#include "barycenter.hpp"

#include "../Problem.hpp"
#include "../core/distance_sampling_weights.hpp" // kmedoids_pp
#include "../core/portable_random.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <utility>

namespace dtwc::algorithms {
namespace {

using Series = std::vector<data_t>;
using SeriesView = std::span<const data_t>;

/// One squared-cost DTW alignment; `path` is empty unless it was asked for.
struct Alignment {
  double cost;
  std::span<const std::pair<std::size_t, std::size_t>> path;
};

void validate_series(std::span<const SeriesView> series)
{
  if (series.empty()) throw InvalidInput("dtw_barycenter: series set must not be empty.");
  for (const auto values : series) {
    if (values.empty()) throw InvalidInput("dtw_barycenter: series must not be empty.");
    for (double value : values)
      if (!std::isfinite(value))
        throw InvalidInput("dtw_barycenter: all values must be finite.");
  }
}

void validate_options(const BarycenterOptions& options)
{
  if (options.max_iter <= 0)
    throw InvalidInput("dtw_barycenter: max_iter must be positive.");
  if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0)
    throw InvalidInput("dtw_barycenter: learning_rate must be finite and positive.");
  if (!std::isfinite(options.learning_rate_decay) || options.learning_rate_decay < 0.0)
    throw InvalidInput("dtw_barycenter: learning_rate_decay must be finite and non-negative.");
  if (!std::isfinite(options.gamma) || options.gamma <= 0.0)
    throw InvalidInput("dtw_barycenter: gamma must be finite and positive.");
  if (!std::isfinite(options.tolerance) || options.tolerance < 0.0)
    throw InvalidInput("dtw_barycenter: tolerance must be finite and non-negative.");
}

[[noreturn]] void throw_nonfinite(const char* entry_point, const char* quantity)
{
  throw InvalidInput(
    std::string(entry_point) + ": computed " + quantity
    + " is non-finite; rescale input values to a smaller magnitude.");
}

void require_finite(double value, const char* entry_point, const char* quantity)
{
  if (!std::isfinite(value)) throw_nonfinite(entry_point, quantity);
}

void require_finite_series(
  const Series& values, const char* entry_point, const char* quantity)
{
  if (!std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); }))
    throw_nonfinite(entry_point, quantity);
}

void validate_problem_configuration(const Problem& prob, const char* entry_point)
{
  if (prob.variant_params().variant != core::DTWVariant::Standard)
    throw InvalidInput(
      std::string(entry_point) + ": only DTWVariant::Standard is supported; "
      "set the Problem variant to DTWVariant::Standard.");
  if (prob.band != settings::DEFAULT_BAND)
    throw InvalidInput(
      std::string(entry_point) +
      ": banded DTW is not supported; set the Problem band to -1.");
  if (prob.data().ndim != 1)
    throw InvalidInput(
      "dtw_barycenter: multivariate barycenters are not implemented; ndim must be 1.");
}

/// Series `indices` of `prob` as double views: Float64 storage is read in place,
/// Float32 storage converted once (the barycenter arithmetic is double).
struct SeriesSet {
  std::vector<Series> converted; ///< Float32 storage only; `views` point into it.
  std::vector<SeriesView> views;
};

SeriesSet problem_series(const Problem& prob, const std::vector<index_t>& indices)
{
  SeriesSet set;
  set.views.reserve(indices.size());
  if (prob.data().is_f32()) {
    set.converted.reserve(indices.size());
    for (const index_t index : indices) {
      const auto values = prob.data().series_f32(static_cast<std::size_t>(index));
      set.views.push_back(set.converted.emplace_back(values.begin(), values.end()));
    }
  } else {
    for (const index_t index : indices)
      set.views.push_back(prob.series(static_cast<std::size_t>(index)));
  }
  validate_series(set.views);
  return set;
}

Series resample_linear(SeriesView input, std::size_t length)
{
  if (length == 0) throw InvalidInput("dtw_barycenter: target_length must be positive.");
  if (input.size() == length) return Series(input.begin(), input.end());
  Series result(length, input.front());
  if (length == 1) {
    result[0] = std::accumulate(input.begin(), input.end(), 0.0)
                / static_cast<double>(input.size());
    return result;
  }
  if (input.size() == 1) return result;
  for (std::size_t i = 0; i < length; ++i) {
    const double position = static_cast<double>(i) * static_cast<double>(input.size() - 1)
                            / static_cast<double>(length - 1);
    const auto left = static_cast<std::size_t>(std::floor(position));
    const auto right = std::min(left + 1, input.size() - 1);
    const double alpha = position - static_cast<double>(left);
    result[i] = (1.0 - alpha) * input[left] + alpha * input[right];
  }
  return result;
}

/// Squared-cost DTW of x against y and, if asked, its optimal warping path. The
/// DP matrix and the path are this thread's scratch, grown and never shrunk as
/// the DTW kernels' is: a warmed thread allocates nothing. The path stays valid
/// until the thread's next call.
Alignment align_squared(SeriesView x, SeriesView y, bool need_path)
{
  thread_local std::vector<double> matrix_buffer;
  thread_local std::vector<std::pair<std::size_t, std::size_t>> path;
  const std::size_t nx = x.size();
  const std::size_t ny = y.size();
  if (matrix_buffer.size() < nx * ny) {
    matrix_buffer.reserve(nx * ny); // exact: resize alone may overshoot
    matrix_buffer.resize(nx * ny);
  }
  path.clear();
  if (need_path) path.reserve(nx + ny - 1);
  // The matrix pointer is hoisted and at(i, j - 1) carried in a register: without
  // TBAA (the MSVC target) each store forced a reload of both.
  double *const matrix = matrix_buffer.data();
  auto at = [matrix, ny](std::size_t i, std::size_t j) -> double& {
    return matrix[i * ny + j];
  };
  for (std::size_t i = 0; i < nx; ++i) {
    double left = std::numeric_limits<double>::infinity(); // at(i, j - 1)
    for (std::size_t j = 0; j < ny; ++j) {
      const double diff = x[i] - y[j];
      const double local = diff * diff;
      if (i == 0 && j == 0) left = local;
      else {
        const double up = i > 0 ? at(i - 1, j) : std::numeric_limits<double>::infinity();
        const double diagonal = (i > 0 && j > 0)
          ? at(i - 1, j - 1) : std::numeric_limits<double>::infinity();
        left = local + std::min(std::min(diagonal, up), left);
      }
      at(i, j) = left;
    }
  }
  const double cost = at(nx - 1, ny - 1);
  if (!need_path) return { cost, {} };

  std::size_t i = nx - 1;
  std::size_t j = ny - 1;
  path.emplace_back(i, j);
  while (i > 0 || j > 0) {
    const double diagonal = (i > 0 && j > 0)
      ? at(i - 1, j - 1) : std::numeric_limits<double>::infinity();
    const double up = i > 0 ? at(i - 1, j) : std::numeric_limits<double>::infinity();
    const double left = j > 0 ? at(i, j - 1) : std::numeric_limits<double>::infinity();
    // Deterministic tie order: diagonal, vertical, horizontal.
    if (diagonal <= up && diagonal <= left) { --i; --j; }
    else if (up <= left) { --i; }
    else { --j; }
    path.emplace_back(i, j);
  }
  std::reverse(path.begin(), path.end());
  return { cost, path };
}

double hard_objective(const Series& center, std::span<const SeriesView> series,
                      const char* entry_point)
{
  double objective = 0.0;
  for (const auto values : series) {
    const double cost = align_squared(center, values, false).cost;
    require_finite(cost, entry_point, "squared-DTW cost");
    objective += cost;
    require_finite(objective, entry_point, "squared-DTW cost");
  }
  return objective;
}

double relative_change(const Series& before, const Series& after)
{
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    const double delta = before[i] - after[i];
    numerator = std::hypot(numerator, delta);
    denominator = std::hypot(denominator, before[i]);
  }
  return numerator / std::max(1.0, denominator);
}

Series dba(std::span<const SeriesView> series, Series center,
           const BarycenterOptions& options, const char* entry_point)
{
  double previous = hard_objective(center, series, entry_point);
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    std::vector<double> sums(center.size(), 0.0);
    std::vector<std::size_t> counts(center.size(), 0);
    for (const auto values : series) {
      const auto [cost, path] = align_squared(center, values, true);
      require_finite(cost, entry_point, "squared-DTW cost");
      for (const auto [i, j] : path) {
        sums[i] += values[j];
        ++counts[i];
      }
    }
    Series next = center;
    for (std::size_t i = 0; i < next.size(); ++i)
      if (counts[i] > 0) next[i] = sums[i] / static_cast<double>(counts[i]);
    require_finite_series(next, entry_point, "barycenter update");
    const double objective = hard_objective(next, series, entry_point);
    const double change = relative_change(center, next);
    require_finite(change, entry_point, "convergence measure");
    center = std::move(next);
    if (change <= options.tolerance
        || std::abs(previous - objective) <= options.tolerance * std::max(1.0, previous))
      break;
    previous = objective;
  }
  return center;
}

Series ssg(std::span<const SeriesView> series, Series center,
           const BarycenterOptions& options, const char* entry_point)
{
  std::mt19937_64 rng(options.random_seed);
  std::vector<std::size_t> order(series.size());
  std::iota(order.begin(), order.end(), std::size_t{ 0 });
  std::uint64_t step = 0;
  double previous = hard_objective(center, series, entry_point);
  for (int epoch = 0; epoch < options.max_iter; ++epoch) {
    const Series before = center;
    core::portable_shuffle(order.begin(), order.end(), rng);
    for (std::size_t index : order) {
      const auto [cost, path] = align_squared(center, series[index], true);
      require_finite(cost, entry_point, "squared-DTW cost");
      std::vector<double> sums(center.size(), 0.0);
      std::vector<std::size_t> counts(center.size(), 0);
      for (const auto [i, j] : path) {
        sums[i] += series[index][j];
        ++counts[i];
      }
      const double eta = options.learning_rate
        / (1.0 + options.learning_rate_decay * static_cast<double>(step++));
      const double max_multiplicity = static_cast<double>(
        *std::max_element(counts.begin(), counts.end()));
      // The fixed-path component has Hessian 2*V and therefore gradient
      // Lipschitz constant L = 2*max(V_ii).  Capping the scalar step at 1/L
      // prevents high-valence paths from exploding while preserving the raw
      // stochastic-gradient direction (unlike dividing every coordinate by
      // its own count).
      const double effective_eta = std::min(eta, 0.5 / max_multiplicity);
      for (std::size_t i = 0; i < center.size(); ++i) {
        if (counts[i] == 0) continue;
        // For squared DTW and the selected optimal path, the stochastic
        // component gradient is 2 * (V*center - W*sample).  V_ii is exactly
        // counts[i]; dividing by it turns SSG into a coordinate-preconditioned
        // averaging heuristic and discards repeated alignments.
        const double multiplicity = static_cast<double>(counts[i]);
        center[i] += 2.0 * effective_eta * (sums[i] - multiplicity * center[i]);
      }
      require_finite_series(center, entry_point, "barycenter update");
    }
    const double objective = hard_objective(center, series, entry_point);
    const double change = relative_change(before, center);
    require_finite(change, entry_point, "convergence measure");
    if (change <= options.tolerance
        || std::abs(previous - objective) <= options.tolerance * std::max(1.0, previous))
      break;
    previous = objective;
  }
  return center;
}

double softmin3(double a, double b, double c, double gamma)
{
  const double minimum = std::min(std::min(a, b), c);
  return minimum - gamma * std::log(std::exp((minimum - a) / gamma)
                                   + std::exp((minimum - b) / gamma)
                                   + std::exp((minimum - c) / gamma));
}

} // namespace

namespace detail {

SoftDtwValueGradient soft_dtw_squared_value_gradient(
  std::span<const data_t> x, std::span<const data_t> y, double gamma)
{
  const std::size_t nx = x.size();
  const std::size_t ny = y.size();
  std::vector<double> accumulated(nx * ny, 0.0);
  auto cell = [&](std::size_t i, std::size_t j) -> double& {
    return accumulated[i * ny + j];
  };
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < ny; ++j) {
      const double diff = x[i] - y[j];
      const double local = diff * diff;
      if (i == 0 && j == 0) cell(i, j) = local;
      else if (i == 0) cell(i, j) = cell(i, j - 1) + local;
      else if (j == 0) cell(i, j) = cell(i - 1, j) + local;
      else cell(i, j) = local + softmin3(cell(i - 1, j), cell(i, j - 1),
                                         cell(i - 1, j - 1), gamma);
    }
  }

  std::vector<double> adjoint(nx * ny, 0.0);
  auto gradient_cell = [&](std::size_t i, std::size_t j) -> double& {
    return adjoint[i * ny + j];
  };
  gradient_cell(nx - 1, ny - 1) = 1.0;
  for (std::size_t reverse_i = nx; reverse_i-- > 0;) {
    for (std::size_t reverse_j = ny; reverse_j-- > 0;) {
      if (reverse_i == nx - 1 && reverse_j == ny - 1) continue;
      double value = 0.0;
      if (reverse_i + 1 < nx) {
        if (reverse_j == 0) value += gradient_cell(reverse_i + 1, reverse_j);
        else {
          const double diff = x[reverse_i + 1] - y[reverse_j];
          const double soft = cell(reverse_i + 1, reverse_j) - diff * diff;
          value += gradient_cell(reverse_i + 1, reverse_j)
                   * std::exp((soft - cell(reverse_i, reverse_j)) / gamma);
        }
      }
      if (reverse_j + 1 < ny) {
        if (reverse_i == 0) value += gradient_cell(reverse_i, reverse_j + 1);
        else {
          const double diff = x[reverse_i] - y[reverse_j + 1];
          const double soft = cell(reverse_i, reverse_j + 1) - diff * diff;
          value += gradient_cell(reverse_i, reverse_j + 1)
                   * std::exp((soft - cell(reverse_i, reverse_j)) / gamma);
        }
      }
      if (reverse_i + 1 < nx && reverse_j + 1 < ny) {
        const double diff = x[reverse_i + 1] - y[reverse_j + 1];
        const double soft = cell(reverse_i + 1, reverse_j + 1) - diff * diff;
        value += gradient_cell(reverse_i + 1, reverse_j + 1)
                 * std::exp((soft - cell(reverse_i, reverse_j)) / gamma);
      }
      gradient_cell(reverse_i, reverse_j) = value;
    }
  }

  SoftDtwValueGradient result;
  result.value = cell(nx - 1, ny - 1);
  result.gradient.assign(nx, 0.0);
  for (std::size_t i = 0; i < nx; ++i)
    for (std::size_t j = 0; j < ny; ++j)
      result.gradient[i] += gradient_cell(i, j) * 2.0 * (x[i] - y[j]);
  return result;
}

} // namespace detail

namespace {

detail::SoftDtwValueGradient soft_objective(const Series& center,
                                            std::span<const SeriesView> series,
                                            double gamma)
{
  detail::SoftDtwValueGradient total;
  total.gradient.assign(center.size(), 0.0);
  for (const auto values : series) {
    const auto current = detail::soft_dtw_squared_value_gradient(center, values, gamma);
    total.value += current.value;
    for (std::size_t i = 0; i < center.size(); ++i)
      total.gradient[i] += current.gradient[i];
  }
  const double inverse = 1.0 / static_cast<double>(series.size());
  total.value *= inverse;
  for (double& value : total.gradient) value *= inverse;
  return total;
}

bool soft_state_is_finite(const detail::SoftDtwValueGradient& state)
{
  return std::isfinite(state.value)
    && std::all_of(state.gradient.begin(), state.gradient.end(),
                   [](double value) { return std::isfinite(value); });
}

Series soft_barycenter(std::span<const SeriesView> series, Series center,
                       const BarycenterOptions& options, const char* entry_point)
{
  double learning_rate = options.learning_rate;
  auto current = soft_objective(center, series, options.gamma);
  if (!soft_state_is_finite(current))
    throw_nonfinite(entry_point, "soft-DTW value or gradient");
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    double norm_squared = 0.0;
    for (double value : current.gradient) norm_squared += value * value;
    require_finite(norm_squared, entry_point, "soft-DTW value or gradient");
    if (std::sqrt(norm_squared) <= options.tolerance) break;

    bool accepted = false;
    bool saw_finite_trial = false;
    Series candidate(center.size());
    detail::SoftDtwValueGradient trial;
    double step = learning_rate;
    for (int backtrack = 0; backtrack < 16; ++backtrack) {
      for (std::size_t i = 0; i < center.size(); ++i)
        candidate[i] = center[i] - step * current.gradient[i];
      if (!std::all_of(candidate.begin(), candidate.end(),
                       [](double value) { return std::isfinite(value); })) {
        step *= 0.5;
        continue;
      }
      trial = soft_objective(candidate, series, options.gamma);
      if (!soft_state_is_finite(trial)) {
        step *= 0.5;
        continue;
      }
      saw_finite_trial = true;
      if (trial.value < current.value) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      if (!saw_finite_trial)
        throw_nonfinite(entry_point, "soft-DTW value or gradient");
      break;
    }
    const double change = relative_change(center, candidate);
    require_finite(change, entry_point, "convergence measure");
    center = std::move(candidate);
    current = std::move(trial);
    learning_rate = step / (1.0 + options.learning_rate_decay);
    if (change <= options.tolerance) break;
  }
  return center;
}

Series compute_barycenter(std::span<const SeriesView> series, std::size_t target_length,
                          const BarycenterOptions& options, SeriesView initial,
                          const char* entry_point)
{
  validate_series(series);
  validate_options(options);
  Series center = resample_linear(initial, target_length);
  require_finite_series(center, entry_point, "barycenter initialization");
  switch (options.method) {
    case BarycenterMethod::SSG:
      return ssg(series, std::move(center), options, entry_point);
    case BarycenterMethod::DBA:
      return dba(series, std::move(center), options, entry_point);
    case BarycenterMethod::SoftDTW:
      return soft_barycenter(series, std::move(center), options, entry_point);
  }
  throw std::logic_error("compute_barycenter: unreachable BarycenterMethod");
}

double assign(std::span<const SeriesView> data, const std::vector<Series>& centers,
              std::vector<index_t>& labels, std::vector<double>* costs = nullptr)
{
  labels.resize(data.size());
  std::vector<double> local_costs(data.size(), 0.0);
  auto nearest_center = [&](std::size_t i) {
    double best = std::numeric_limits<double>::infinity();
    index_t label = 0;
    for (std::size_t c = 0; c < centers.size(); ++c) {
      const double distance = align_squared(data[i], centers[c], false).cost;
      if (distance < best) { best = distance; label = static_cast<index_t>(c); }
    }
    labels[i] = label;
    local_costs[i] = best;
  };
  // run_openmp carries a failure out of the region, a scratch bad_alloc included.
  run_openmp(nearest_center, data.size());
  for (double cost : local_costs)
    require_finite(cost, "barycenter_kmeans", "assignment cost");
  const double total = std::accumulate(local_costs.begin(), local_costs.end(), 0.0);
  require_finite(total, "barycenter_kmeans", "assignment cost");
  if (costs) *costs = local_costs;
  return total;
}

} // namespace

std::vector<data_t> dtw_barycenter(const Problem& prob,
                                   const std::vector<index_t>& series_indices,
                                   std::size_t target_length,
                                   const BarycenterOptions& options)
{
  validate_problem_configuration(prob, "dtw_barycenter");
  if (series_indices.empty())
    throw InvalidInput("dtw_barycenter: series_indices must not be empty.");
  for (index_t index : series_indices)
    if (index < 0 || index >= prob.size())
      throw InvalidInput("dtw_barycenter: series index out of range.");
  const auto selected = problem_series(prob, series_indices);
  return compute_barycenter(
    selected.views, target_length, options, selected.views.front(), "dtw_barycenter");
}

BarycenterClusteringResult barycenter_kmeans(
  const Problem& prob, const BarycenterClusteringOptions& options)
{
  validate_problem_configuration(prob, "barycenter_kmeans");
  std::vector<index_t> every(static_cast<std::size_t>(prob.size()));
  std::iota(every.begin(), every.end(), index_t{ 0 });
  const auto series = problem_series(prob, every);
  const auto& data = series.views;
  const std::size_t n = data.size();
  if (options.n_clusters <= 0 || static_cast<std::size_t>(options.n_clusters) > n)
    throw InvalidInput("barycenter_kmeans: n_clusters must be in [1, N].");
  if (options.max_iter <= 0 || options.barycenter.max_iter <= 0)
    throw InvalidInput("barycenter_kmeans: iteration limits must be positive.");
  if (options.target_length == 0 || options.target_length < -1)
    throw InvalidInput("barycenter_kmeans: target_length must be -1 or positive.");
  validate_options(options.barycenter);

  // D² sampling: align_squared is already the squared objective's cost.
  std::mt19937_64 rng(options.barycenter.random_seed);
  const auto initial_indices = core::kmedoids_pp(
    static_cast<index_t>(n), options.n_clusters, rng,
    [&data](index_t c, index_t i) {
      return align_squared(data[static_cast<std::size_t>(i)],
                           data[static_cast<std::size_t>(c)], false).cost;
    },
    "barycenter_kmeans");
  std::vector<Series> centers;
  centers.reserve(static_cast<std::size_t>(options.n_clusters));
  for (index_t index : initial_indices) {
    const std::size_t length = options.target_length > 0
      ? static_cast<std::size_t>(options.target_length)
      : data[static_cast<std::size_t>(index)].size();
    centers.push_back(resample_linear(data[static_cast<std::size_t>(index)], length));
  }

  BarycenterClusteringResult result;
  std::vector<index_t> previous_labels(n, -1);
  double previous_cost = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    std::vector<double> point_costs;
    const double cost = assign(data, centers, result.labels, &point_costs);
    const bool has_previous_assignment = iteration > 0
      && std::isfinite(previous_cost);
    const bool labels_stable = has_previous_assignment
      && result.labels == previous_labels;
    const bool cost_stable = has_previous_assignment
      && std::abs(previous_cost - cost)
           <= options.barycenter.tolerance * std::max(1.0, previous_cost);
    if (labels_stable || cost_stable) {
      result.converged = true;
      result.iterations = iteration;
      result.total_cost = cost;
      break;
    }
    previous_labels = result.labels;
    previous_cost = cost;

    std::vector<std::vector<SeriesView>> members(
      static_cast<std::size_t>(options.n_clusters));
    for (std::size_t i = 0; i < n; ++i)
      members[static_cast<std::size_t>(result.labels[i])].push_back(data[i]);

    // Empty-cluster repair is order-sensitive because each selected farthest
    // point is removed from consideration. Preserve the original cluster order
    // here, before independent non-empty updates enter OpenMP.
    std::vector<Series> next_centers = centers;
    std::vector<bool> needs_update(
      static_cast<std::size_t>(options.n_clusters), true);
    for (index_t cluster = 0; cluster < options.n_clusters; ++cluster) {
      const auto cluster_index = static_cast<std::size_t>(cluster);
      if (members[cluster_index].empty()) {
        // Deterministic empty-cluster repair: use the currently worst-represented
        // point, preserving k without silently returning a missing centre.
        const auto farthest = static_cast<std::size_t>(
          std::distance(point_costs.begin(),
                        std::max_element(point_costs.begin(), point_costs.end())));
        next_centers[cluster_index] = resample_linear(
          data[farthest], centers[cluster_index].size());
        point_costs[farthest] = -1.0;
        needs_update[cluster_index] = false;
      }
    }

    // run_openmp rethrows the lowest cluster's failure, the one a serial run raises.
    auto update_center = [&](std::size_t cluster) {
      if (!needs_update[cluster]) return;
      auto cluster_options = options.barycenter;
      cluster_options.random_seed = options.barycenter.random_seed
        + static_cast<std::uint64_t>(iteration)
          * static_cast<std::uint64_t>(options.n_clusters)
        + static_cast<std::uint64_t>(cluster);
      next_centers[cluster] = compute_barycenter(
        members[cluster], centers[cluster].size(), cluster_options,
        centers[cluster], "barycenter_kmeans");
    };
    run_openmp(update_center, static_cast<std::size_t>(options.n_clusters));
    centers = std::move(next_centers);
    result.iterations = iteration + 1;
  }
  // On the converged path the loop already stored `result.labels` and
  // `result.total_cost` from that iteration's assign(), and `centers` has not
  // changed since (the break precedes the centre update), so re-assigning would
  // repeat a full N*k DBA-DTW pass for an identical answer. Only the
  // max_iter-exhausted path needs it: there the last iteration did update
  // `centers` after its assign().
  if (!result.converged)
    result.total_cost = assign(data, centers, result.labels);
  result.barycenters = std::move(centers);
  return result;
}

} // namespace dtwc::algorithms
