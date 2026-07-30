/**
 * @file test_lb_keogh_derivation.cpp
 * @brief D2 executable oracle for scalar envelopes and LB_Keogh.
 *
 * The acceptance bands and exact inventories in this file were registered in
 * .claude/baselines/2026-07-30-d2-lb-keogh.md before the first execution.
 * Exact DTW costs are obtained by recursively enumerating every monotone path;
 * this oracle deliberately shares no DP recurrence or rolling storage with the
 * production kernel.
 */

#include <algorithms/tadpole.hpp>
#include <core/lower_bound_impl.hpp>
#include <core/pruned_distance_matrix.hpp>
#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace {

using Series = std::vector<double>;

struct PathCosts
{
  double l1 = std::numeric_limits<double>::max();
  double squared = std::numeric_limits<double>::max();
  std::size_t count = 0;
};

struct BoundValues
{
  double forward_l1 = 0;
  double reverse_l1 = 0;
  double symmetric_l1 = 0;
  double forward_squared = 0;
  double reverse_squared = 0;
  double symmetric_squared = 0;
};

struct Audit
{
  std::size_t violations = 0;
  std::string first_violation;

  void require(bool condition, const std::string &message)
  {
    if (condition) return;
    ++violations;
    if (first_violation.empty()) first_violation = message;
  }
};

#ifdef _OPENMP
struct SerialOpenMpScope
{
  int previous_threads = omp_get_max_threads();
  int previous_dynamic = omp_get_dynamic();

  SerialOpenMpScope()
  {
    omp_set_dynamic(0);
    omp_set_num_threads(1);
  }

  ~SerialOpenMpScope()
  {
    omp_set_dynamic(previous_dynamic);
    omp_set_num_threads(previous_threads);
  }
};
#else
struct SerialOpenMpScope
{};
#endif

std::string show(const Series &series)
{
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < series.size(); ++i) {
    if (i != 0) out << ',';
    out << series[i];
  }
  out << ']';
  return out.str();
}

void enumerate_words(
  std::size_t position,
  Series &word,
  const std::array<double, 3> &alphabet,
  std::vector<Series> &words)
{
  if (position == word.size()) {
    words.push_back(word);
    return;
  }
  for (const double value : alphabet) {
    word[position] = value;
    enumerate_words(position + 1, word, alphabet, words);
  }
}

std::vector<Series> words(std::size_t length, const std::array<double, 3> &alphabet)
{
  std::vector<Series> result;
  Series word(length);
  enumerate_words(0, word, alphabet, result);
  return result;
}

std::pair<Series, Series> naive_envelope(const Series &series, int radius)
{
  Series upper(series.size());
  Series lower(series.size());
  for (std::size_t i = 0; i < series.size(); ++i) {
    const std::size_t r = static_cast<std::size_t>(radius);
    const std::size_t first = i > r ? i - r : 0;
    const std::size_t last = std::min(series.size(), i + r + 1);
    const auto begin = series.begin() + static_cast<std::ptrdiff_t>(first);
    const auto end = series.begin() + static_cast<std::ptrdiff_t>(last);
    upper[i] = *std::max_element(begin, end);
    lower[i] = *std::min_element(begin, end);
  }
  return { std::move(upper), std::move(lower) };
}

bool inside_band(std::size_t i, std::size_t j, int radius)
{
  const auto delta = static_cast<std::ptrdiff_t>(i)
                   - static_cast<std::ptrdiff_t>(j);
  return std::abs(delta) <= static_cast<std::ptrdiff_t>(radius);
}

void enumerate_paths(
  const Series &x,
  const Series &y,
  int radius,
  std::size_t i,
  std::size_t j,
  double prefix_l1,
  double prefix_squared,
  PathCosts &best)
{
  if (!inside_band(i, j, radius)) return;

  const double delta = x[i] - y[j];
  const double next_l1 = prefix_l1 + std::abs(delta);
  const double next_squared = prefix_squared + delta * delta;
  if (i + 1 == x.size() && j + 1 == y.size()) {
    best.l1 = std::min(best.l1, next_l1);
    best.squared = std::min(best.squared, next_squared);
    ++best.count;
    return;
  }

  if (i + 1 < x.size())
    enumerate_paths(x, y, radius, i + 1, j, next_l1, next_squared, best);
  if (j + 1 < y.size())
    enumerate_paths(x, y, radius, i, j + 1, next_l1, next_squared, best);
  if (i + 1 < x.size() && j + 1 < y.size())
    enumerate_paths(x, y, radius, i + 1, j + 1, next_l1, next_squared, best);
}

PathCosts exact_path_costs(const Series &x, const Series &y, int radius)
{
  PathCosts result;
  enumerate_paths(x, y, radius, 0, 0, 0.0, 0.0, result);
  return result;
}

BoundValues production_bounds(
  const Series &x,
  const Series &y,
  int radius,
  std::size_t prefix_length)
{
  std::vector<double> upper_x;
  std::vector<double> lower_x;
  std::vector<double> upper_y;
  std::vector<double> lower_y;
  dtwc::core::compute_envelopes(x, radius, upper_x, lower_x);
  dtwc::core::compute_envelopes(y, radius, upper_y, lower_y);

  BoundValues result;
  result.forward_l1 = dtwc::core::lb_keogh(
    x.data(), prefix_length, upper_y.data(), lower_y.data());
  result.reverse_l1 = dtwc::core::lb_keogh(
    y.data(), prefix_length, upper_x.data(), lower_x.data());
  result.symmetric_l1 = std::max(result.forward_l1, result.reverse_l1);
  result.forward_squared = dtwc::core::lb_keogh_squared(
    x.data(), prefix_length, upper_y.data(), lower_y.data());
  result.reverse_squared = dtwc::core::lb_keogh_squared(
    y.data(), prefix_length, upper_x.data(), lower_x.data());
  result.symmetric_squared =
    std::max(result.forward_squared, result.reverse_squared);
  return result;
}

void audit_admissibility(
  const Series &x,
  const Series &y,
  int radius,
  std::size_t prefix_length,
  Audit &audit)
{
  const PathCosts exact = exact_path_costs(x, y, radius);
  if (exact.count == 0) {
    audit.require(
      false,
      "no path for registered feasible case x=" + show(x)
        + " y=" + show(y) + " radius=" + std::to_string(radius));
    return;
  }

  const BoundValues bound = production_bounds(x, y, radius, prefix_length);
  const std::string context =
    " x=" + show(x) + " y=" + show(y)
    + " radius=" + std::to_string(radius);
  audit.require(bound.forward_l1 <= exact.l1, "forward L1" + context);
  audit.require(bound.reverse_l1 <= exact.l1, "reverse L1" + context);
  audit.require(bound.symmetric_l1 <= exact.l1, "symmetric L1" + context);
  audit.require(
    bound.forward_squared <= exact.squared,
    "forward squared" + context);
  audit.require(
    bound.reverse_squared <= exact.squared,
    "reverse squared" + context);
  audit.require(
    bound.symmetric_squared <= exact.squared,
    "symmetric squared" + context);
}

dtwc::Problem make_problem(
  std::vector<Series> series,
  int band,
  const std::string &name)
{
  std::vector<std::string> names;
  names.reserve(series.size());
  for (std::size_t i = 0; i < series.size(); ++i)
    names.push_back("s" + std::to_string(i));

  dtwc::Problem problem(name);
  problem.set_data(dtwc::Data(std::move(series), std::move(names)));
  problem.set_band(band);
  return problem;
}

} // namespace

TEST_CASE(
  "D2 exhaustive envelopes and LB_Keogh derivation oracle",
  "[D2][envelope][lb_keogh][derivation]")
{
  SerialOpenMpScope serial_openmp;

  // Direct-window envelope arbiter: 2,004 registered cases.
  constexpr std::array<double, 3> envelope_alphabet = { -2.0, 0.0, 3.0 };
  std::size_t envelope_cases = 0;
  Audit envelope_audit;
  for (std::size_t n = 1; n <= 5; ++n) {
    for (const Series &series : words(n, envelope_alphabet)) {
      for (int radius = 0; radius <= static_cast<int>(n); ++radius) {
        const auto [expected_upper, expected_lower] =
          naive_envelope(series, radius);
        std::vector<double> actual_upper;
        std::vector<double> actual_lower;
        dtwc::core::compute_envelopes(
          series, radius, actual_upper, actual_lower);
        ++envelope_cases;
        envelope_audit.require(
          actual_upper == expected_upper,
          "upper envelope series=" + show(series)
            + " radius=" + std::to_string(radius));
        envelope_audit.require(
          actual_lower == expected_lower,
          "lower envelope series=" + show(series)
            + " radius=" + std::to_string(radius));
      }
    }
  }
  INFO("first envelope violation: " << envelope_audit.first_violation);
  REQUIRE(envelope_cases == 2004);
  REQUIRE(envelope_audit.violations == 0);

  // Explicit monotone-path arbiter: 28,602 equal-length cases.
  constexpr std::array<double, 3> bound_alphabet = { -1.0, 0.0, 2.0 };
  std::array<std::vector<Series>, 5> words_by_length;
  for (std::size_t n = 1; n <= 4; ++n)
    words_by_length[n] = words(n, bound_alphabet);

  std::size_t equal_cases = 0;
  Audit equal_audit;
  for (std::size_t n = 1; n <= 4; ++n) {
    for (const Series &x : words_by_length[n]) {
      for (const Series &y : words_by_length[n]) {
        for (int radius = 0; radius < static_cast<int>(n); ++radius) {
          audit_admissibility(x, y, radius, n, equal_audit);
          ++equal_cases;
        }
      }
    }
  }
  INFO("first equal-length violation: " << equal_audit.first_violation);
  REQUIRE(equal_cases == 28602);
  REQUIRE(equal_audit.violations == 0);

  // Explicit monotone-path arbiter: 17,712 feasible unequal-length cases.
  std::size_t unequal_cases = 0;
  Audit unequal_audit;
  for (std::size_t n = 1; n <= 4; ++n) {
    for (std::size_t m = 1; m <= 4; ++m) {
      if (n == m) continue;
      const int first_radius = std::abs(
        static_cast<int>(n) - static_cast<int>(m));
      const int last_radius = static_cast<int>(std::max(n, m)) - 1;
      for (const Series &x : words_by_length[n]) {
        for (const Series &y : words_by_length[m]) {
          for (int radius = first_radius; radius <= last_radius; ++radius) {
            audit_admissibility(
              x, y, radius, std::min(n, m), unequal_audit);
            ++unequal_cases;
          }
        }
      }
    }
  }
  INFO("first unequal-length violation: " << unequal_audit.first_violation);
  REQUIRE(unequal_cases == 17712);
  REQUIRE(unequal_audit.violations == 0);

  // Registered non-degenerate direction, symmetry, and metric discriminator.
  const Series q = { 5.0, -4.0, 1.0, 7.0, -2.0 };
  const Series c = { 0.0, 3.0, -6.0, 1.0, 4.0 };
  const BoundValues discriminator = production_bounds(q, c, 1, q.size());
  REQUIRE(discriminator.forward_l1 == 8.0);
  REQUIRE(discriminator.reverse_l1 == 2.0);
  REQUIRE(discriminator.symmetric_l1 == 8.0);
  REQUIRE(discriminator.forward_squared == 22.0);
  REQUIRE(discriminator.reverse_squared == 4.0);
  REQUIRE(discriminator.symmetric_squared == 22.0);

  const BoundValues singleton =
    production_bounds(Series{ 0.5 }, Series{ 0.0 }, 0, 1);
  REQUIRE(singleton.symmetric_l1 == 0.5);
  REQUIRE(singleton.symmetric_squared == 0.25);

  // Full-DTW gotcha: negative radius is currently coerced to radius zero.
  const Series x = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };
  const Series y = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 };
  REQUIRE(dtwc::dtwFull_L<double>(x, y) == 0.0);
  REQUIRE(production_bounds(x, y, 0, x.size()).symmetric_l1 == 2.0);
  REQUIRE(production_bounds(x, y, 1, x.size()).symmetric_l1 == 1.0);
  REQUIRE(
    production_bounds(x, y, static_cast<int>(x.size()), x.size())
      .symmetric_l1
    == 0.0);

  std::vector<double> upper_x_negative;
  std::vector<double> lower_x_negative;
  std::vector<double> upper_y_negative;
  std::vector<double> lower_y_negative;
  std::vector<double> upper_x_zero;
  std::vector<double> lower_x_zero;
  std::vector<double> upper_y_zero;
  std::vector<double> lower_y_zero;
  dtwc::core::compute_envelopes(
    x, -1, upper_x_negative, lower_x_negative);
  dtwc::core::compute_envelopes(
    y, -1, upper_y_negative, lower_y_negative);
  dtwc::core::compute_envelopes(x, 0, upper_x_zero, lower_x_zero);
  dtwc::core::compute_envelopes(y, 0, upper_y_zero, lower_y_zero);
  REQUIRE(upper_x_negative == upper_x_zero);
  REQUIRE(lower_x_negative == lower_x_zero);
  REQUIRE(upper_y_negative == upper_y_zero);
  REQUIRE(lower_y_negative == lower_y_zero);
  const double unsafe_negative_bound = std::max(
    dtwc::core::lb_keogh(
      x.data(), x.size(), upper_y_negative.data(), lower_y_negative.data()),
    dtwc::core::lb_keogh(
      y.data(), y.size(), upper_x_negative.data(), lower_x_negative.data()));
  REQUIRE(unsafe_negative_bound == 2.0);
  REQUIRE(unsafe_negative_bound > dtwc::dtwFull_L<double>(x, y));

  std::size_t call_sites = 0;

  // Exact-matrix full-DTW route must disable envelope bounds. The leading
  // series establishes threshold 0.5 for x and y; a radius-zero Keogh bound
  // would then fire on their true-zero pair.
  Series z = x;
  z.back() = 1.5;
  auto matrix_problem = make_problem({ z, x, y }, -1, "d2_full_matrix");
  const dtwc::core::PruningStats matrix_stats =
    dtwc::core::fill_distance_matrix_pruned(
      matrix_problem, -1, dtwc::LowerBoundStrategy::Keogh);
  REQUIRE(matrix_stats.total_pairs == 3);
  REQUIRE(matrix_stats.pruned_by_lb_kim == 0);
  REQUIRE(matrix_stats.pruned_by_lb_keogh == 0);
  REQUIRE(matrix_stats.early_abandoned == 0);
  REQUIRE(matrix_stats.computed_full_dtw == 3);
  REQUIRE(matrix_problem.dist_by_ind(1, 2) == 0.0);
  ++call_sites;

  // TADPole deliberately replaces a negative full-DTW request with global
  // envelopes. Its pruning ledger is the reachability proof.
  auto tadpole_pruned_problem =
    make_problem({ x, y }, -1, "d2_tadpole_pruned");
  auto tadpole_brute_problem =
    make_problem({ x, y }, -1, "d2_tadpole_brute");
  dtwc::algorithms::TADPoleStats pruned_stats;
  dtwc::algorithms::TADPoleStats brute_stats;
  const auto pruned = dtwc::algorithms::tadpole(
    tadpole_pruned_problem, 1, 0.5, true, &pruned_stats);
  const auto brute = dtwc::algorithms::tadpole(
    tadpole_brute_problem, 1, 0.5, false, &brute_stats);
  REQUIRE(pruned.labels == brute.labels);
  REQUIRE(pruned.medoid_indices == brute.medoid_indices);
  REQUIRE(pruned.total_cost == brute.total_cost);
  REQUIRE(pruned_stats.total_pairs == 1);
  REQUIRE(pruned_stats.dtw_calls == 1);
  REQUIRE(pruned_stats.pruned_by_lb == 0);
  REQUIRE(pruned_stats.pruned_by_ub == 0);
  REQUIRE(brute_stats.pruned_by_lb == 0);
  REQUIRE(brute_stats.pruned_by_ub == 0);

  // The complementary pair proves that the global-envelope stage actually
  // executes. Together with the zero-DTW pair above it distinguishes a
  // disabled LB (0,0), an unsafe radius-zero LB (1,1), and the correct global
  // full-window LB (0,1).
  const Series separated_a = { 0.0, 0.0 };
  const Series separated_b = { 2.0, 2.0 };
  REQUIRE(dtwc::dtwFull_L<double>(separated_a, separated_b) == 4.0);
  REQUIRE(
    production_bounds(separated_a, separated_b, 2, 2).symmetric_l1
    == 4.0);
  auto separated_pruned_problem = make_problem(
    { separated_a, separated_b }, -1, "d2_tadpole_reachable_pruned");
  auto separated_brute_problem = make_problem(
    { separated_a, separated_b }, -1, "d2_tadpole_reachable_brute");
  dtwc::algorithms::TADPoleStats separated_pruned_stats;
  dtwc::algorithms::TADPoleStats separated_brute_stats;
  const auto separated_pruned = dtwc::algorithms::tadpole(
    separated_pruned_problem, 1, 1.0, true, &separated_pruned_stats);
  const auto separated_brute = dtwc::algorithms::tadpole(
    separated_brute_problem, 1, 1.0, false, &separated_brute_stats);
  REQUIRE(separated_pruned.labels == separated_brute.labels);
  REQUIRE(
    separated_pruned.medoid_indices == separated_brute.medoid_indices);
  REQUIRE(separated_pruned.total_cost == separated_brute.total_cost);
  REQUIRE(separated_pruned_stats.total_pairs == 1);
  REQUIRE(separated_pruned_stats.dtw_calls == 1);
  REQUIRE(separated_pruned_stats.pruned_by_lb == 1);
  REQUIRE(separated_pruned_stats.pruned_by_ub == 0);
  REQUIRE(separated_brute_stats.pruned_by_lb == 0);
  REQUIRE(separated_brute_stats.pruned_by_ub == 0);
  ++call_sites;

  REQUIRE(call_sites == 2);
  std::cout
    << "D2_LB_KEOGH_GATE envelope_cases=" << envelope_cases
    << " equal_cases=" << equal_cases
    << " unequal_cases=" << unequal_cases
    << " call_sites=" << call_sites << "/2"
    << " skips=0 verdict=PASS\n";
}
