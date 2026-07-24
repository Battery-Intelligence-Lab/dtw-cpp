/**
 * @file barycenter.cpp
 * @brief SSG/DBA/soft-DTW barycenters and a sequence-centroid k-means driver.
 */

#include "barycenter.hpp"

#include "../Problem.hpp"
#include "../core/portable_random.hpp"
#include "../error.hpp"
#include "../parallelisation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc::algorithms {
namespace {

using Series = std::vector<data_t>;

struct AlignmentWorkspace {
  std::vector<double> matrix;
  std::vector<std::pair<std::size_t, std::size_t>> path;

  static std::size_t checked_matrix_cells(std::size_t nx, std::size_t ny)
  {
    if (nx == 0 || ny == 0)
      throw InvalidInput("dtw_barycenter: series must not be empty.");
    if (nx > std::numeric_limits<std::size_t>::max() / ny)
      throw InvalidInput("dtw_barycenter: alignment matrix size overflows size_t.");
    return nx * ny;
  }

  static std::size_t checked_path_cells(std::size_t nx, std::size_t ny)
  {
    if (nx == 0 || ny == 0)
      throw InvalidInput("dtw_barycenter: series must not be empty.");
    if (nx > std::numeric_limits<std::size_t>::max() - (ny - 1))
      throw InvalidInput("dtw_barycenter: alignment path size overflows size_t.");
    return nx + ny - 1;
  }

  void reserve_for(std::size_t nx, std::size_t ny)
  {
    matrix.reserve(checked_matrix_cells(nx, ny));
    path.reserve(checked_path_cells(nx, ny));
  }

  void prepare(std::size_t nx, std::size_t ny, bool need_path)
  {
    matrix.resize(checked_matrix_cells(nx, ny));
    path.clear();
    if (need_path) path.reserve(checked_path_cells(nx, ny));
  }
};

std::size_t current_worker_index() noexcept
{
#ifdef _OPENMP
  return static_cast<std::size_t>(omp_get_thread_num());
#else
  return 0;
#endif
}

bool openmp_region_active() noexcept
{
#ifdef _OPENMP
  return omp_in_parallel() != 0;
#else
  return false;
#endif
}

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
  if (prob.variant_params.variant != core::DTWVariant::Standard)
    throw InvalidInput(
      std::string(entry_point) + ": only DTWVariant::Standard is supported; "
      "set the Problem variant to DTWVariant::Standard.");
  if (prob.band != settings::DEFAULT_BAND)
    throw InvalidInput(
      std::string(entry_point) +
      ": banded DTW is not supported; set the Problem band to -1.");
}

std::vector<Series> copy_problem_series(const Problem& prob)
{
  if (prob.data().ndim != 1)
    throw InvalidInput(
      "dtw_barycenter: multivariate barycenters are not implemented; ndim must be 1.");
  std::vector<Series> result(prob.size());
  for (std::size_t i = 0; i < prob.size(); ++i) {
    if (prob.data().is_f32()) {
      const auto values = prob.data().series_f32(i);
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

double align_squared(const Series& x, const Series& y, bool need_path,
                     AlignmentWorkspace& workspace)
{
  const std::size_t nx = x.size();
  const std::size_t ny = y.size();
  workspace.prepare(nx, ny, need_path);
  auto at = [&](std::size_t i, std::size_t j) -> double& {
    return workspace.matrix[i * ny + j];
  };
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
  const double cost = at(nx - 1, ny - 1);
  if (!need_path) return cost;

  std::size_t i = nx - 1;
  std::size_t j = ny - 1;
  workspace.path.emplace_back(i, j);
  while (i > 0 || j > 0) {
    const double diagonal = (i > 0 && j > 0)
      ? at(i - 1, j - 1) : std::numeric_limits<double>::infinity();
    const double up = i > 0 ? at(i - 1, j) : std::numeric_limits<double>::infinity();
    const double left = j > 0 ? at(i, j - 1) : std::numeric_limits<double>::infinity();
    // Deterministic tie order: diagonal, vertical, horizontal.
    if (diagonal <= up && diagonal <= left) { --i; --j; }
    else if (up <= left) { --i; }
    else { --j; }
    workspace.path.emplace_back(i, j);
  }
  std::reverse(workspace.path.begin(), workspace.path.end());
  return cost;
}

double hard_objective(const Series& center, const std::vector<Series>& series,
                      AlignmentWorkspace& workspace, const char* entry_point)
{
  double objective = 0.0;
  for (const auto& values : series) {
    const double cost = align_squared(center, values, false, workspace);
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

Series dba(const std::vector<Series>& series, Series center,
           const BarycenterOptions& options, AlignmentWorkspace& workspace,
           const char* entry_point)
{
  double previous = hard_objective(center, series, workspace, entry_point);
  for (int iteration = 0; iteration < options.max_iter; ++iteration) {
    std::vector<double> sums(center.size(), 0.0);
    std::vector<std::size_t> counts(center.size(), 0);
    for (const auto& values : series) {
      const double cost = align_squared(center, values, true, workspace);
      require_finite(cost, entry_point, "squared-DTW cost");
      for (const auto [i, j] : workspace.path) {
        sums[i] += values[j];
        ++counts[i];
      }
    }
    Series next = center;
    for (std::size_t i = 0; i < next.size(); ++i)
      if (counts[i] > 0) next[i] = sums[i] / static_cast<double>(counts[i]);
    require_finite_series(next, entry_point, "barycenter update");
    const double objective = hard_objective(next, series, workspace, entry_point);
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

Series ssg(const std::vector<Series>& series, Series center,
           const BarycenterOptions& options, AlignmentWorkspace& workspace,
           const char* entry_point)
{
  std::mt19937_64 rng(options.random_seed);
  std::vector<std::size_t> order(series.size());
  std::iota(order.begin(), order.end(), 0);
  std::uint64_t step = 0;
  double previous = hard_objective(center, series, workspace, entry_point);
  for (int epoch = 0; epoch < options.max_iter; ++epoch) {
    const Series before = center;
    core::portable_shuffle(order.begin(), order.end(), rng);
    for (std::size_t index : order) {
      const double cost = align_squared(center, series[index], true, workspace);
      require_finite(cost, entry_point, "squared-DTW cost");
      std::vector<double> sums(center.size(), 0.0);
      std::vector<std::size_t> counts(center.size(), 0);
      for (const auto [i, j] : workspace.path) {
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
    const double objective = hard_objective(center, series, workspace, entry_point);
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

bool soft_state_is_finite(const detail::SoftDtwValueGradient& state)
{
  return std::isfinite(state.value)
    && std::all_of(state.gradient.begin(), state.gradient.end(),
                   [](double value) { return std::isfinite(value); });
}

Series soft_barycenter(const std::vector<Series>& series, Series center,
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

Series compute_barycenter(const std::vector<Series>& series, std::size_t target_length,
                          const BarycenterOptions& options, const Series& initial,
                          AlignmentWorkspace& workspace, const char* entry_point)
{
  validate_barycenter_method(options.method);
  validate_series(series);
  validate_options(options);
  Series center = resample_linear(initial, target_length);
  require_finite_series(center, entry_point, "barycenter initialization");
  switch (options.method) {
    case BarycenterMethod::SSG:
      return ssg(series, std::move(center), options, workspace, entry_point);
    case BarycenterMethod::DBA:
      return dba(series, std::move(center), options, workspace, entry_point);
    case BarycenterMethod::SoftDTW:
      return soft_barycenter(series, std::move(center), options, entry_point);
  }
  throw std::logic_error("compute_barycenter: unreachable BarycenterMethod");
}

std::vector<int> kmeanspp(const std::vector<Series>& data, int k,
                          std::mt19937_64& rng, AlignmentWorkspace& workspace)
{
  std::vector<int> centers{static_cast<int>(
    core::portable_bounded(rng, static_cast<std::uint64_t>(data.size())))};
  std::vector<double> closest(data.size(), std::numeric_limits<double>::infinity());
  while (static_cast<int>(centers.size()) < k) {
    for (std::size_t i = 0; i < data.size(); ++i)
      closest[i] = std::min(closest[i],
        align_squared(data[i], data[static_cast<std::size_t>(centers.back())],
                      false, workspace));
    double total = std::accumulate(closest.begin(), closest.end(), 0.0);
    require_finite(
      total, "barycenter_kmeans", "initialization distance total");
    std::size_t chosen = 0;
    if (total <= 0.0) {
      while (std::find(centers.begin(), centers.end(), static_cast<int>(chosen)) != centers.end())
        ++chosen;
    } else {
      chosen = core::portable_weighted_index(
        closest.begin(), closest.end(), total, rng);
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
              std::vector<int>& labels, std::vector<AlignmentWorkspace>& workspaces,
              int worker_count, std::vector<double>* costs = nullptr)
{
  labels.resize(data.size());
  std::vector<double> local_costs(data.size(), 0.0);
  if (data.size() > static_cast<std::size_t>(
                      std::numeric_limits<std::int64_t>::max()))
    throw InvalidInput("barycenter_kmeans: series count exceeds int64 loop range.");
  const std::int64_t end = static_cast<std::int64_t>(data.size());
  const bool parallel_assignment = worker_count > 1 && !openmp_region_active();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(worker_count) if(parallel_assignment)
#endif
  for (std::int64_t raw_i = 0; raw_i < end; ++raw_i) {
    const auto i = static_cast<std::size_t>(raw_i);
    const auto worker = parallel_assignment ? current_worker_index() : 0;
    auto& workspace = workspaces[worker];
    double best = std::numeric_limits<double>::infinity();
    int label = 0;
    for (std::size_t c = 0; c < centers.size(); ++c) {
      const double distance = align_squared(data[i], centers[c], false, workspace);
      if (distance < best) { best = distance; label = static_cast<int>(c); }
    }
    labels[i] = label;
    local_costs[i] = best;
  }
  for (double cost : local_costs)
    require_finite(cost, "barycenter_kmeans", "assignment cost");
  const double total = std::accumulate(local_costs.begin(), local_costs.end(), 0.0);
  require_finite(total, "barycenter_kmeans", "assignment cost");
  if (costs) *costs = local_costs;
  return total;
}

} // namespace

std::vector<data_t> dtw_barycenter(const Problem& prob,
                                   const std::vector<int>& series_indices,
                                   std::size_t target_length,
                                   const BarycenterOptions& options)
{
  validate_barycenter_method(options.method);
  validate_problem_configuration(prob, "dtw_barycenter");
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
  AlignmentWorkspace workspace;
  return compute_barycenter(
    selected, target_length, options, selected.front(), workspace,
    "dtw_barycenter");
}

BarycenterClusteringResult barycenter_kmeans(
  const Problem& prob, const BarycenterClusteringOptions& options)
{
  validate_barycenter_method(options.method);
  validate_problem_configuration(prob, "barycenter_kmeans");
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

  // A caller may invoke clustering from its own OpenMP region. Do not create a
  // nested team: nested parallelism can oversubscribe the host and makes a
  // worker-indexed scratch pool unsafe.
  const auto max_workers = openmp_region_active()
    ? std::size_t{1} : static_cast<std::size_t>(get_max_threads());
  const int assignment_workers = static_cast<int>(
    std::max<std::size_t>(1, std::min(max_workers, n)));
  const int update_workers = static_cast<int>(std::min(
    max_workers, static_cast<std::size_t>(options.n_clusters)));
  const auto workspace_count = static_cast<std::size_t>(assignment_workers);
  // The pool is owned by the complete clustering call. Assignment borrows one
  // workspace per OpenMP worker, and the later cluster-update region reuses the
  // same worker-indexed pool. Its peak DP storage is bounded by
  // assignment_workers * max(data_length, target_length) * max(data_length)
  // doubles rather than growing with the number of clusters.
  std::vector<AlignmentWorkspace> workspaces(workspace_count);

  const auto max_data_length = std::max_element(
    data.begin(), data.end(),
    [](const Series& left, const Series& right) {
      return left.size() < right.size();
    })->size();
  const auto max_center_length = options.target_length > 0
    ? static_cast<std::size_t>(options.target_length) : max_data_length;
  const auto max_alignment_length = std::max(
    max_data_length, max_center_length);
  // All allocation and overflow failure happens on the caller thread. Once a
  // parallel region starts, resize/emplace stay within these capacities and no
  // exception can escape an OpenMP iteration.
  for (auto& workspace : workspaces)
    workspace.reserve_for(max_alignment_length, max_data_length);

  std::mt19937_64 rng(options.random_seed);
  const auto initial_indices = kmeanspp(
    data, options.n_clusters, rng, workspaces.front());
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
    const double cost = assign(
      data, centers, result.labels, workspaces, assignment_workers, &point_costs);
    const bool has_previous_assignment = iteration > 0
      && std::isfinite(previous_cost);
    const bool labels_stable = has_previous_assignment
      && result.labels == previous_labels;
    const bool cost_stable = has_previous_assignment
      && std::abs(previous_cost - cost)
           <= options.tolerance * std::max(1.0, previous_cost);
    if (labels_stable || cost_stable) {
      result.converged = true;
      result.iterations = iteration;
      result.total_cost = cost;
      break;
    }
    previous_labels = result.labels;
    previous_cost = cost;

    std::vector<std::vector<Series>> members(
      static_cast<std::size_t>(options.n_clusters));
    for (std::size_t i = 0; i < n; ++i)
      members[static_cast<std::size_t>(result.labels[i])].push_back(data[i]);

    // Empty-cluster repair is order-sensitive because each selected farthest
    // point is removed from consideration. Preserve the original cluster order
    // here, before independent non-empty updates enter OpenMP.
    std::vector<Series> next_centers = centers;
    std::vector<bool> needs_update(
      static_cast<std::size_t>(options.n_clusters), true);
    for (int cluster = 0; cluster < options.n_clusters; ++cluster) {
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

    // OpenMP cannot propagate C++ exceptions. Store one exception per cluster
    // and rethrow in cluster order after the region, matching serial priority.
    std::vector<std::exception_ptr> failures(
      static_cast<std::size_t>(options.n_clusters));
    const bool parallel_updates = update_workers > 1 && !openmp_region_active();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(update_workers) if(parallel_updates)
#endif
    for (int cluster = 0; cluster < options.n_clusters; ++cluster) {
      const auto cluster_index = static_cast<std::size_t>(cluster);
      if (!needs_update[cluster_index]) continue;
      try {
        const auto worker = parallel_updates ? current_worker_index() : 0;
        auto cluster_options = barycenter_options;
        cluster_options.random_seed = options.random_seed
          + static_cast<std::uint64_t>(iteration)
            * static_cast<std::uint64_t>(options.n_clusters)
          + static_cast<std::uint64_t>(cluster);
        next_centers[cluster_index] = compute_barycenter(
          members[cluster_index], centers[cluster_index].size(), cluster_options,
          centers[cluster_index], workspaces[worker], "barycenter_kmeans");
      } catch (...) {
        failures[cluster_index] = std::current_exception();
      }
    }
    for (const auto& failure : failures)
      if (failure) std::rethrow_exception(failure);
    centers = std::move(next_centers);
    result.iterations = iteration + 1;
  }
  result.total_cost = assign(
    data, centers, result.labels, workspaces, assignment_workers);
  result.barycenters = std::move(centers);
  return result;
}

} // namespace dtwc::algorithms
