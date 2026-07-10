/**
 * @file barycenter.cpp
 * @brief SSG/DBA/soft-DTW barycenters and a sequence-centroid k-means driver.
 */

#include "barycenter.hpp"

#include "../Problem.hpp"
#include "../error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>

namespace dtwc::algorithms {
namespace {

using Series = std::vector<data_t>;

struct Alignment {
  double cost = 0.0;
  std::vector<std::pair<std::size_t, std::size_t>> path;
};

void validate_series(const std::vector<Series>& series)
{
  if (series.empty()) throw InvalidInput("dtw_barycenter: series set must not be empty.");
  for (const auto& values : series) {
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

std::vector<Series> copy_problem_series(const Problem& prob)
{
  if (prob.data.ndim != 1)
    throw InvalidInput(
      "dtw_barycenter: multivariate barycenters are not implemented; ndim must be 1.");
  std::vector<Series> result(prob.size());
  for (std::size_t i = 0; i < prob.size(); ++i) {
    if (prob.data.is_f32()) {
      const auto values = prob.data.series_f32(i);
      result[i].assign(values.begin(), values.end());
    } else {
      const auto values = prob.series(i);
      result[i].assign(values.begin(), values.end());
    }
  }
  validate_series(result);
  return result;
}

Series resample_linear(const Series& input, std::size_t length)
{
  if (length == 0) throw InvalidInput("dtw_barycenter: target_length must be positive.");
  if (input.size() == length) return input;
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

Alignment align_squared(const Series& x, const Series& y, bool need_path)
{
  const std::size_t nx = x.size();
  const std::size_t ny = y.size();
  std::vector<double> matrix(nx * ny, std::numeric_limits<double>::infinity());
  auto at = [&](std::size_t i, std::size_t j) -> double& { return matrix[i * ny + j]; };
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < ny; ++j) {
      const double diff = x[i] - y[j];
      const double local = diff * diff;
      if (i == 0 && j == 0) at(i, j) = local;
      else {
        const double up = i > 0 ? at(i - 1, j) : std::numeric_limits<double>::infinity();
        const double left = j > 0 ? at(i, j - 1) : std::numeric_limits<double>::infinity();
        const double diagonal = (i > 0 && j > 0)
          ? at(i - 1, j - 1) : std::numeric_limits<double>::infinity();
        at(i, j) = local + std::min({diagonal, up, left});
      }
    }
  }
  Alignment result;
  result.cost = at(nx - 1, ny - 1);
  if (!need_path) return result;

  std::size_t i = nx - 1;
  std::size_t j = ny - 1;
  result.path.emplace_back(i, j);
  while (i > 0 || j > 0) {
    const double diagonal = (i > 0 && j > 0)
      ? at(i - 1, j - 1) : std::numeric_limits<double>::infinity();
    const double up = i > 0 ? at(i - 1, j) : std::numeric_limits<double>::infinity();
    const double left = j > 0 ? at(i, j - 1) : std::numeric_limits<double>::infinity();
    // Deterministic tie order: diagonal, vertical, horizontal.
    if (diagonal <= up && diagonal <= left) { --i; --j; }
    else if (up <= left) { --i; }
    else { --j; }
    result.path.emplace_back(i, j);
  }
  std::reverse(result.path.begin(), result.path.end());
  return result;
}

double hard_objective(const Series& center, const std::vector<Series>& series)
{
  double objective = 0.0;
  for (const auto& values : series) objective += align_squared(center, values, false).cost;
  return objective;
}

double relative_change(const Series& before, const Series& after)
{
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    const double delta = before[i] - after[i];
    numerator += delta * delta;
    denominator += before[i] * before[i];
  }
  return std::sqrt(numerator) / std::max(1.0, std::sqrt(denominator));
}

Series dba(const std::vector<Series>& series, Series center,
           const BarycenterOptions& options)
{
  double previous = hard_objective(center, series);
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    std::vector<double> sums(center.size(), 0.0);
    std::vector<std::size_t> counts(center.size(), 0);
    for (const auto& values : series) {
      const auto alignment = align_squared(center, values, true);
      for (const auto [i, j] : alignment.path) {
        sums[i] += values[j];
        ++counts[i];
      }
    }
    Series next = center;
    for (std::size_t i = 0; i < next.size(); ++i)
      if (counts[i] > 0) next[i] = sums[i] / static_cast<double>(counts[i]);
    const double objective = hard_objective(next, series);
    const double change = relative_change(center, next);
    center = std::move(next);
    if (change <= options.tolerance
        || std::abs(previous - objective) <= options.tolerance * std::max(1.0, previous))
      break;
    previous = objective;
  }
  return center;
}

Series ssg(const std::vector<Series>& series, Series center,
           const BarycenterOptions& options)
{
  std::mt19937_64 rng(options.random_seed);
  std::vector<std::size_t> order(series.size());
  std::iota(order.begin(), order.end(), 0);
  std::uint64_t step = 0;
  double previous = hard_objective(center, series);
  for (int epoch = 0; epoch < options.max_iter; ++epoch) {
    const Series before = center;
    std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t index : order) {
      const auto alignment = align_squared(center, series[index], true);
      std::vector<double> sums(center.size(), 0.0);
      std::vector<std::size_t> counts(center.size(), 0);
      for (const auto [i, j] : alignment.path) {
        sums[i] += series[index][j];
        ++counts[i];
      }
      const double eta = options.learning_rate
        / (1.0 + options.learning_rate_decay * static_cast<double>(step++));
      for (std::size_t i = 0; i < center.size(); ++i) {
        if (counts[i] == 0) continue;
        const double aligned_mean = sums[i] / static_cast<double>(counts[i]);
        center[i] += eta * (aligned_mean - center[i]);
      }
    }
    const double objective = hard_objective(center, series);
    if (relative_change(before, center) <= options.tolerance
        || std::abs(previous - objective) <= options.tolerance * std::max(1.0, previous))
      break;
    previous = objective;
  }
  return center;
}

double softmin3(double a, double b, double c, double gamma)
{
  const double minimum = std::min({a, b, c});
  return minimum - gamma * std::log(std::exp((minimum - a) / gamma)
                                   + std::exp((minimum - b) / gamma)
                                   + std::exp((minimum - c) / gamma));
}

} // namespace

namespace detail {

SoftDtwValueGradient soft_dtw_squared_value_gradient(
  const std::vector<data_t>& x, const std::vector<data_t>& y, double gamma)
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
                                            const std::vector<Series>& series,
                                            double gamma)
{
  detail::SoftDtwValueGradient total;
  total.gradient.assign(center.size(), 0.0);
  for (const auto& values : series) {
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

Series soft_barycenter(const std::vector<Series>& series, Series center,
                       const BarycenterOptions& options)
{
  double learning_rate = options.learning_rate;
  auto current = soft_objective(center, series, options.gamma);
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    double norm_squared = 0.0;
    for (double value : current.gradient) norm_squared += value * value;
    if (std::sqrt(norm_squared) <= options.tolerance) break;

    bool accepted = false;
    Series candidate(center.size());
    detail::SoftDtwValueGradient trial;
    double step = learning_rate;
    for (int backtrack = 0; backtrack < 16; ++backtrack) {
      for (std::size_t i = 0; i < center.size(); ++i)
        candidate[i] = center[i] - step * current.gradient[i];
      trial = soft_objective(candidate, series, options.gamma);
      if (std::isfinite(trial.value) && trial.value < current.value) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) break;
    const double change = relative_change(center, candidate);
    center = std::move(candidate);
    current = std::move(trial);
    learning_rate = step / (1.0 + options.learning_rate_decay);
    if (change <= options.tolerance) break;
  }
  return center;
}

Series compute_barycenter(const std::vector<Series>& series, std::size_t target_length,
                          const BarycenterOptions& options, const Series& initial)
{
  validate_series(series);
  validate_options(options);
  Series center = resample_linear(initial, target_length);
  switch (options.method) {
    case BarycenterMethod::SSG: return ssg(series, std::move(center), options);
    case BarycenterMethod::DBA: return dba(series, std::move(center), options);
    case BarycenterMethod::SoftDTW:
      return soft_barycenter(series, std::move(center), options);
  }
  throw InvalidInput("dtw_barycenter: unknown method.");
}

std::vector<int> kmeanspp(const std::vector<Series>& data, int k, std::mt19937_64& rng)
{
  std::uniform_int_distribution<std::size_t> first_distribution(0, data.size() - 1);
  std::vector<int> centers{static_cast<int>(first_distribution(rng))};
  std::vector<double> closest(data.size(), std::numeric_limits<double>::infinity());
  while (static_cast<int>(centers.size()) < k) {
    for (std::size_t i = 0; i < data.size(); ++i)
      closest[i] = std::min(closest[i],
        align_squared(data[i], data[static_cast<std::size_t>(centers.back())], false).cost);
    double total = std::accumulate(closest.begin(), closest.end(), 0.0);
    std::size_t chosen = 0;
    if (total <= 0.0) {
      while (std::find(centers.begin(), centers.end(), static_cast<int>(chosen)) != centers.end())
        ++chosen;
    } else {
      std::uniform_real_distribution<double> draw(0.0, total);
      double threshold = draw(rng);
      for (; chosen + 1 < data.size(); ++chosen) {
        threshold -= closest[chosen];
        if (threshold <= 0.0) break;
      }
      if (std::find(centers.begin(), centers.end(), static_cast<int>(chosen)) != centers.end()) {
        chosen = 0;
        while (std::find(centers.begin(), centers.end(), static_cast<int>(chosen)) != centers.end())
          ++chosen;
      }
    }
    centers.push_back(static_cast<int>(chosen));
  }
  return centers;
}

double assign(const std::vector<Series>& data, const std::vector<Series>& centers,
              std::vector<int>& labels, std::vector<double>* costs = nullptr)
{
  labels.resize(data.size());
  std::vector<double> local_costs(data.size(), 0.0);
  for (std::size_t i = 0; i < data.size(); ++i) {
    double best = std::numeric_limits<double>::infinity();
    int label = 0;
    for (std::size_t c = 0; c < centers.size(); ++c) {
      const double distance = align_squared(data[i], centers[c], false).cost;
      if (distance < best) { best = distance; label = static_cast<int>(c); }
    }
    labels[i] = label;
    local_costs[i] = best;
  }
  if (costs) *costs = local_costs;
  return std::accumulate(local_costs.begin(), local_costs.end(), 0.0);
}

} // namespace

std::vector<data_t> dtw_barycenter(const Problem& prob,
                                   const std::vector<int>& series_indices,
                                   std::size_t target_length,
                                   const BarycenterOptions& options)
{
  const auto all = copy_problem_series(prob);
  if (series_indices.empty())
    throw InvalidInput("dtw_barycenter: series_indices must not be empty.");
  std::vector<Series> selected;
  selected.reserve(series_indices.size());
  for (int index : series_indices) {
    if (index < 0 || static_cast<std::size_t>(index) >= all.size())
      throw InvalidInput("dtw_barycenter: series index out of range.");
    selected.push_back(all[static_cast<std::size_t>(index)]);
  }
  return compute_barycenter(selected, target_length, options, selected.front());
}

BarycenterClusteringResult barycenter_kmeans(
  const Problem& prob, const BarycenterClusteringOptions& options)
{
  const auto data = copy_problem_series(prob);
  const std::size_t n = data.size();
  if (options.n_clusters <= 0 || static_cast<std::size_t>(options.n_clusters) > n)
    throw InvalidInput("barycenter_kmeans: n_clusters must be in [1, N].");
  if (options.max_iter <= 0 || options.barycenter_max_iter <= 0)
    throw InvalidInput("barycenter_kmeans: iteration limits must be positive.");
  if (options.target_length == 0 || options.target_length < -1)
    throw InvalidInput("barycenter_kmeans: target_length must be -1 or positive.");

  BarycenterOptions barycenter_options;
  barycenter_options.method = options.method;
  barycenter_options.max_iter = options.barycenter_max_iter;
  barycenter_options.learning_rate = options.learning_rate;
  barycenter_options.learning_rate_decay = options.learning_rate_decay;
  barycenter_options.gamma = options.gamma;
  barycenter_options.tolerance = options.tolerance;
  barycenter_options.random_seed = options.random_seed;
  validate_options(barycenter_options);

  std::mt19937_64 rng(options.random_seed);
  const auto initial_indices = kmeanspp(data, options.n_clusters, rng);
  std::vector<Series> centers;
  centers.reserve(static_cast<std::size_t>(options.n_clusters));
  for (int index : initial_indices) {
    const std::size_t length = options.target_length > 0
      ? static_cast<std::size_t>(options.target_length)
      : data[static_cast<std::size_t>(index)].size();
    centers.push_back(resample_linear(data[static_cast<std::size_t>(index)], length));
  }

  BarycenterClusteringResult result;
  std::vector<int> previous_labels(n, -1);
  double previous_cost = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    std::vector<double> point_costs;
    const double cost = assign(data, centers, result.labels, &point_costs);
    if (result.labels == previous_labels
        || std::abs(previous_cost - cost) <= options.tolerance * std::max(1.0, previous_cost)) {
      result.converged = true;
      result.iterations = iteration;
      result.total_cost = cost;
      break;
    }
    previous_labels = result.labels;
    previous_cost = cost;

    for (int cluster = 0; cluster < options.n_clusters; ++cluster) {
      std::vector<Series> members;
      for (std::size_t i = 0; i < n; ++i)
        if (result.labels[i] == cluster) members.push_back(data[i]);
      if (members.empty()) {
        // Deterministic empty-cluster repair: use the currently worst-represented
        // point, preserving k without silently returning a missing centre.
        const auto farthest = static_cast<std::size_t>(
          std::distance(point_costs.begin(),
                        std::max_element(point_costs.begin(), point_costs.end())));
        centers[static_cast<std::size_t>(cluster)] = resample_linear(
          data[farthest], centers[static_cast<std::size_t>(cluster)].size());
        point_costs[farthest] = -1.0;
        continue;
      }
      barycenter_options.random_seed = options.random_seed
        + static_cast<std::uint64_t>(iteration * options.n_clusters + cluster);
      centers[static_cast<std::size_t>(cluster)] = compute_barycenter(
        members, centers[static_cast<std::size_t>(cluster)].size(), barycenter_options,
        centers[static_cast<std::size_t>(cluster)]);
    }
    result.iterations = iteration + 1;
  }
  result.total_cost = assign(data, centers, result.labels);
  result.barycenters = std::move(centers);
  return result;
}

} // namespace dtwc::algorithms
