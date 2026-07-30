/**
 * @file test_lb_enhanced_webb_derivation.cpp
 * @brief D3 executable oracle for LB_Enhanced and local LB_Webb_NoLR.
 *
 * The exact inventories and marker in this file were registered in
 * .claude/baselines/2026-07-30-d3-lb-enhanced-webb.md before execution.
 * Small DTW instances are solved by explicit monotone-path enumeration.
 * Envelope, elastic-cut, Webb-predicate, and long V=5 references are
 * independent of the corresponding production implementations.
 */

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

struct Audit
{
  std::size_t violations = 0;
  std::string first;

  void fail(std::string message)
  {
    ++violations;
    if (first.empty()) first = std::move(message);
  }
};

struct PathCosts
{
  double l1 = std::numeric_limits<double>::max();
  double squared = std::numeric_limits<double>::max();
  std::size_t count = 0;
};

struct DirectEnvelope
{
  Series upper;
  Series lower;
  Series lu;
  Series ul;
};

struct BranchHits
{
  bool full_upper = false;
  bool full_lower = false;
  bool overlap_upper = false;
  bool overlap_lower = false;
};

struct FreeFlags
{
  bool above = true;
  bool below = true;
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
{
};
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
  std::vector<Series> &result)
{
  if (position == word.size()) {
    result.push_back(word);
    return;
  }
  for (const double value : alphabet) {
    word[position] = value;
    enumerate_words(position + 1, word, alphabet, result);
  }
}

std::vector<Series> words(
  std::size_t length,
  const std::array<double, 3> &alphabet)
{
  Series word(length);
  std::vector<Series> result;
  enumerate_words(0, word, alphabet, result);
  return result;
}

std::size_t effective_radius(int radius, std::size_t length)
{
  if (length == 0 || radius <= 0) return 0;
  return std::min(static_cast<std::size_t>(radius), length - 1);
}

std::pair<Series, Series> direct_envelope(
  const Series &series,
  std::size_t radius)
{
  Series upper(series.size());
  Series lower(series.size());
  for (std::size_t i = 0; i < series.size(); ++i) {
    const std::size_t first = i > radius ? i - radius : 0;
    const std::size_t last =
      std::min(series.size() - 1, i + std::min(radius, series.size() - 1 - i));
    double hi = series[first];
    double lo = series[first];
    for (std::size_t j = first + 1; j <= last; ++j) {
      hi = std::max(hi, series[j]);
      lo = std::min(lo, series[j]);
    }
    upper[i] = hi;
    lower[i] = lo;
  }
  return { std::move(upper), std::move(lower) };
}

DirectEnvelope direct_webb_envelope(const Series &series, std::size_t radius)
{
  auto [upper, lower] = direct_envelope(series, radius);
  auto [unused_upper, lu] = direct_envelope(upper, radius);
  auto [ul, unused_lower] = direct_envelope(lower, radius);
  (void)unused_upper;
  (void)unused_lower;
  return {
    std::move(upper), std::move(lower), std::move(lu), std::move(ul)
  };
}

bool inside_band(std::size_t i, std::size_t j, std::size_t radius)
{
  return i > j ? i - j <= radius : j - i <= radius;
}

void enumerate_paths(
  const Series &a,
  const Series &b,
  std::size_t radius,
  std::size_t i,
  std::size_t j,
  double prefix_l1,
  double prefix_squared,
  PathCosts &best)
{
  if (!inside_band(i, j, radius)) return;
  const double delta = a[i] - b[j];
  const double next_l1 = prefix_l1 + std::abs(delta);
  const double next_squared = prefix_squared + delta * delta;
  if (i + 1 == a.size() && j + 1 == b.size()) {
    best.l1 = std::min(best.l1, next_l1);
    best.squared = std::min(best.squared, next_squared);
    ++best.count;
    return;
  }
  if (i + 1 < a.size())
    enumerate_paths(
      a, b, radius, i + 1, j, next_l1, next_squared, best);
  if (j + 1 < b.size())
    enumerate_paths(
      a, b, radius, i, j + 1, next_l1, next_squared, best);
  if (i + 1 < a.size() && j + 1 < b.size())
    enumerate_paths(
      a, b, radius, i + 1, j + 1, next_l1, next_squared, best);
}

PathCosts exact_path_costs(
  const Series &a,
  const Series &b,
  std::size_t radius)
{
  PathCosts result;
  enumerate_paths(a, b, radius, 0, 0, 0.0, 0.0, result);
  return result;
}

void enumerate_cell_paths(
  std::size_t length,
  std::size_t radius,
  std::size_t i,
  std::size_t j,
  std::vector<std::size_t> &path,
  std::vector<std::vector<std::size_t>> &paths)
{
  if (!inside_band(i, j, radius)) return;
  path.push_back(i * length + j);
  if (i + 1 == length && j + 1 == length) {
    paths.push_back(path);
  } else {
    if (i + 1 < length)
      enumerate_cell_paths(length, radius, i + 1, j, path, paths);
    if (j + 1 < length)
      enumerate_cell_paths(length, radius, i, j + 1, path, paths);
    if (i + 1 < length && j + 1 < length)
      enumerate_cell_paths(length, radius, i + 1, j + 1, path, paths);
  }
  path.pop_back();
}

std::vector<std::vector<std::size_t>> enhanced_sets(
  std::size_t length,
  std::size_t radius,
  int requested_bands)
{
  if (length == 1) return { { 0 } };
  const std::size_t bands = std::min(
    static_cast<std::size_t>(std::max(requested_bands, 1)), length / 2);
  std::vector<std::vector<std::size_t>> sets;
  for (std::size_t cut = 0; cut < bands; ++cut) {
    const std::size_t first = cut > radius ? cut - radius : 0;
    std::vector<std::size_t> left;
    left.push_back(cut * length + cut);
    for (std::size_t j = first; j < cut; ++j) {
      left.push_back(cut * length + j);
      left.push_back(j * length + cut);
    }
    std::vector<std::size_t> right;
    right.reserve(left.size());
    for (const std::size_t cell : left) {
      const std::size_t i = cell / length;
      const std::size_t j = cell % length;
      right.push_back(
        (length - 1 - i) * length + (length - 1 - j));
    }
    sets.push_back(std::move(left));
    sets.push_back(std::move(right));
  }
  for (std::size_t i = bands; i < length - bands; ++i) {
    std::vector<std::size_t> middle;
    const std::size_t first = i > radius ? i - radius : 0;
    const std::size_t last =
      std::min(length - 1, i + std::min(radius, length - 1 - i));
    for (std::size_t j = first; j <= last; ++j)
      middle.push_back(i * length + j);
    sets.push_back(std::move(middle));
  }
  return sets;
}

void audit_enhanced_structure(
  std::size_t length,
  std::size_t radius,
  int requested_bands,
  Audit &audit)
{
  const auto sets = enhanced_sets(length, radius, requested_bands);
  std::vector<bool> claimed(length * length, false);
  for (const auto &set : sets) {
    if (set.empty()) {
      audit.fail("empty Enhanced set");
      continue;
    }
    for (const std::size_t cell : set) {
      if (claimed[cell])
        audit.fail(
          "overlapping Enhanced sets n=" + std::to_string(length)
          + " w=" + std::to_string(radius)
          + " v=" + std::to_string(requested_bands));
      claimed[cell] = true;
    }
  }

  std::vector<std::vector<std::size_t>> paths;
  std::vector<std::size_t> path;
  enumerate_cell_paths(length, radius, 0, 0, path, paths);
  if (paths.empty()) {
    audit.fail("no structural path");
    return;
  }
  for (const auto &candidate_path : paths) {
    for (const auto &set : sets) {
      bool crossed = false;
      for (const std::size_t cell : candidate_path)
        crossed = crossed
                  || std::find(set.begin(), set.end(), cell) != set.end();
      if (!crossed)
        audit.fail(
          "uncrossed Enhanced set n=" + std::to_string(length)
          + " w=" + std::to_string(radius)
          + " v=" + std::to_string(requested_bands));
    }
  }
}

double point_cost(double a, double b, bool squared)
{
  const double delta = std::abs(a - b);
  return squared ? delta * delta : delta;
}

double reference_enhanced(
  const Series &a,
  const Series &b,
  std::size_t radius,
  int requested_bands,
  bool squared)
{
  if (a.empty()) return 0.0;
  if (a.size() == 1) return point_cost(a[0], b[0], squared);

  const std::size_t bands = std::min(
    static_cast<std::size_t>(std::max(requested_bands, 1)),
    a.size() / 2);
  double result = 0.0;

  // Directly construct the mutually disjoint left elastic cuts and reflect
  // each coordinate through the lower-right corner for the right cuts.
  for (std::size_t cut = 0; cut < bands; ++cut) {
    const std::size_t first = cut > radius ? cut - radius : 0;
    double left_min = point_cost(a[cut], b[cut], squared);
    double right_min = point_cost(
      a[a.size() - 1 - cut], b[b.size() - 1 - cut], squared);
    for (std::size_t j = first; j < cut; ++j) {
      left_min = std::min(
        left_min,
        std::min(
          point_cost(a[cut], b[j], squared),
          point_cost(a[j], b[cut], squared)));
      const std::size_t reflected_cut = a.size() - 1 - cut;
      const std::size_t reflected_j = a.size() - 1 - j;
      right_min = std::min(
        right_min,
        std::min(
          point_cost(a[reflected_cut], b[reflected_j], squared),
          point_cost(a[reflected_j], b[reflected_cut], squared)));
    }
    result += left_min + right_min;
  }

  // Each middle contribution is the interval projection in Eq. 3.7. Build
  // the extrema by a direct donor scan, but do not minimize over the discrete
  // donors: a query between two extrema contributes zero even when neither
  // donor equals it.
  for (std::size_t i = bands; i < a.size() - bands; ++i) {
    const std::size_t first = i > radius ? i - radius : 0;
    const std::size_t last =
      std::min(b.size() - 1, i + std::min(radius, b.size() - 1 - i));
    double upper = b[first];
    double lower = b[first];
    for (std::size_t j = first + 1; j <= last; ++j) {
      upper = std::max(upper, b[j]);
      lower = std::min(lower, b[j]);
    }
    if (a[i] > upper)
      result += point_cost(a[i], upper, squared);
    else if (a[i] < lower)
      result += point_cost(a[i], lower, squared);
  }
  return result;
}

template <typename Metric>
double production_enhanced(
  const Series &a,
  const Series &b,
  int radius,
  int requested_bands)
{
  // D3 audits the canonical geometric window. F57 separately drives raw
  // n/INT_MAX API radii through production to pin saturation arithmetic.
  const int w = static_cast<int>(effective_radius(radius, a.size()));
  const auto envelope = dtwc::core::compute_envelope(b, w);
  return dtwc::core::lb_enhanced<double, Metric>(
    a.data(), b.data(), a.size(), envelope.upper.data(), envelope.lower.data(), w, requested_bands, Metric{});
}

bool safe_above(
  const Series &a,
  const DirectEnvelope &ea,
  const DirectEnvelope &eb,
  std::size_t i)
{
  if (a[i] > eb.upper[i]) return false;
  if (a[i] < eb.lower[i]) return eb.lower[i] <= ea.lu[i];
  return true;
}

bool safe_below(
  const Series &a,
  const DirectEnvelope &ea,
  const DirectEnvelope &eb,
  std::size_t i)
{
  if (a[i] < eb.lower[i]) return false;
  if (a[i] > eb.upper[i]) return eb.upper[i] >= ea.ul[i];
  return true;
}

FreeFlags direct_free_flags(
  const Series &a,
  const DirectEnvelope &ea,
  const DirectEnvelope &eb,
  std::size_t radius,
  std::size_t column,
  bool conservative_tail)
{
  std::size_t first;
  std::size_t last;
  if (conservative_tail) {
    last = std::min(
      column + std::min(radius, a.size() - 1 - column), a.size() - 1);
    const std::size_t width = std::min(2 * radius, last);
    first = last - width;
  } else {
    first = column > radius ? column - radius : 0;
    last = std::min(
      a.size() - 1,
      column + std::min(radius, a.size() - 1 - column));
  }

  FreeFlags result;
  for (std::size_t i = first; i <= last; ++i) {
    result.above = result.above && safe_above(a, ea, eb, i);
    result.below = result.below && safe_below(a, ea, eb, i);
  }
  return result;
}

double reference_webb(
  const Series &a,
  const DirectEnvelope &ea,
  const Series &b,
  const DirectEnvelope &eb,
  std::size_t radius,
  bool squared,
  bool conservative_tail,
  BranchHits *hits = nullptr)
{
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] > eb.upper[i])
      result += point_cost(a[i], eb.upper[i], squared);
    else if (a[i] < eb.lower[i])
      result += point_cost(a[i], eb.lower[i], squared);
  }

  for (std::size_t j = 0; j < b.size(); ++j) {
    const FreeFlags free =
      direct_free_flags(a, ea, eb, radius, j, conservative_tail);

    if (free.above && b[j] > ea.upper[j]) {
      result += point_cost(b[j], ea.upper[j], squared);
      if (hits != nullptr) hits->full_upper = true;
    } else if (free.below && b[j] < ea.lower[j]) {
      result += point_cost(b[j], ea.lower[j], squared);
      if (hits != nullptr) hits->full_lower = true;
    } else if (b[j] > eb.ul[j] && eb.ul[j] >= ea.upper[j]) {
      result += point_cost(b[j], ea.upper[j], squared)
                - point_cost(eb.ul[j], ea.upper[j], squared);
      if (hits != nullptr) hits->overlap_upper = true;
    } else if (b[j] < eb.lu[j] && eb.lu[j] <= ea.lower[j]) {
      result += point_cost(b[j], ea.lower[j], squared)
                - point_cost(eb.lu[j], ea.lower[j], squared);
      if (hits != nullptr) hits->overlap_lower = true;
    }
  }
  return result;
}

template <typename Metric>
double production_webb(const Series &a, const Series &b, int radius)
{
  // Keep the proof oracle independent of F57's raw-radius regression.
  const int w = static_cast<int>(effective_radius(radius, a.size()));
  const auto ea = dtwc::core::compute_webb_envelope(a, w);
  const auto eb = dtwc::core::compute_webb_envelope(b, w);
  return dtwc::core::lb_webb<Metric>(a, ea, b, eb, w);
}

double direct_keogh(
  const Series &query,
  const DirectEnvelope &candidate,
  bool squared)
{
  double result = 0.0;
  for (std::size_t i = 0; i < query.size(); ++i) {
    if (query[i] > candidate.upper[i])
      result += point_cost(query[i], candidate.upper[i], squared);
    else if (query[i] < candidate.lower[i])
      result += point_cost(query[i], candidate.lower[i], squared);
  }
  return result;
}

double full_matrix_dtw(
  const Series &a,
  const Series &b,
  std::size_t radius,
  bool squared)
{
  const double infinity = std::numeric_limits<double>::max();
  std::vector<double> matrix(a.size() * b.size(), infinity);
  auto at = [&](std::size_t i, std::size_t j) -> double & {
    return matrix[i * b.size() + j];
  };
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < b.size(); ++j) {
      if (!inside_band(i, j, radius)) continue;
      double predecessor = infinity;
      if (i == 0 && j == 0) predecessor = 0.0;
      if (i > 0) predecessor = std::min(predecessor, at(i - 1, j));
      if (j > 0) predecessor = std::min(predecessor, at(i, j - 1));
      if (i > 0 && j > 0)
        predecessor = std::min(predecessor, at(i - 1, j - 1));
      if (predecessor != infinity)
        at(i, j) = predecessor + point_cost(a[i], b[j], squared);
    }
  }
  return at(a.size() - 1, b.size() - 1);
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
  "D3 exhaustive LB_Enhanced and LB_Webb_NoLR derivation oracle",
  "[D3][lb_enhanced][lb_webb][derivation]")
{
  SerialOpenMpScope serial_openmp;

  constexpr std::array<double, 3> envelope_alphabet = { -2.0, 0.0, 3.0 };
  std::size_t envelope_cases = 0;
  Audit envelope_audit;
  for (std::size_t n = 1; n <= 5; ++n) {
    for (const Series &series : words(n, envelope_alphabet)) {
      for (int radius = 0; radius <= static_cast<int>(n); ++radius) {
        const std::size_t w = effective_radius(radius, n);
        const DirectEnvelope expected = direct_webb_envelope(series, w);
        const auto actual = dtwc::core::compute_webb_envelope(series, radius);
        if (actual.upper != expected.upper)
          envelope_audit.fail("U series=" + show(series));
        if (actual.lower != expected.lower)
          envelope_audit.fail("L series=" + show(series));
        if (actual.lu != expected.lu)
          envelope_audit.fail("L(U) series=" + show(series));
        if (actual.ul != expected.ul)
          envelope_audit.fail("U(L) series=" + show(series));
        ++envelope_cases;
      }
    }
  }
  INFO("first envelope violation: " << envelope_audit.first);
  REQUIRE(envelope_cases == 2004);
  REQUIRE(envelope_audit.violations == 0);

  constexpr std::array<double, 3> bound_alphabet = { -1.0, 0.0, 2.0 };
  std::array<std::vector<Series>, 5> words_by_length;
  for (std::size_t n = 1; n <= 4; ++n)
    words_by_length[n] = words(n, bound_alphabet);

  // Materialize the theorem's selected cell sets independently of values.
  // Every admissible path must cross every set, and no cell may be charged by
  // two selected sets.
  std::size_t structure_cases = 0;
  Audit structure_audit;
  for (std::size_t n = 1; n <= 4; ++n) {
    for (int radius = 0; radius <= static_cast<int>(n); ++radius) {
      const std::size_t w = effective_radius(radius, n);
      const int last_v = std::max(1, static_cast<int>(n / 2));
      for (int v = 1; v <= last_v; ++v) {
        audit_enhanced_structure(n, w, v, structure_audit);
        ++structure_cases;
      }
    }
  }
  INFO("first Enhanced structure violation: " << structure_audit.first);
  REQUIRE(structure_cases == 19);
  REQUIRE(structure_audit.violations == 0);

  std::size_t path_cases = 0;
  std::size_t full_cover_cases = 0;
  std::size_t enhanced_cases = 0;
  std::size_t webb_cases = 0;
  std::size_t tail_cases = 0;
  std::array<std::size_t, 5> full_path_counts{};
  Audit path_audit;
  Audit enhanced_audit;
  Audit webb_audit;
  Audit tail_audit;
  Audit predicate_audit;
  BranchHits branch_hits;

  for (std::size_t n = 1; n <= 4; ++n) {
    for (const Series &a : words_by_length[n]) {
      for (const Series &b : words_by_length[n]) {
        for (int radius = 0; radius <= static_cast<int>(n); ++radius) {
          const std::size_t w = effective_radius(radius, n);
          const PathCosts exact = exact_path_costs(a, b, w);
          if (exact.count == 0)
            path_audit.fail("no path a=" + show(a) + " b=" + show(b));
          if (radius == 0 && exact.count != 1)
            path_audit.fail("radius-zero path count n=" + std::to_string(n));
          ++path_cases;
          if (radius == static_cast<int>(n)) {
            ++full_cover_cases;
            if (full_path_counts[n] == 0) full_path_counts[n] = exact.count;
            if (full_path_counts[n] != exact.count)
              path_audit.fail("value-dependent path count n=" + std::to_string(n));
          }

          const DirectEnvelope ea = direct_webb_envelope(a, w);
          const DirectEnvelope eb = direct_webb_envelope(b, w);
          const int last_v = std::max(1, static_cast<int>(n / 2));
          for (int v = 1; v <= last_v; ++v) {
            const double ref_l1 = reference_enhanced(a, b, w, v, false);
            const double ref_sq = reference_enhanced(a, b, w, v, true);
            const double ref_rev_l1 =
              reference_enhanced(b, a, w, v, false);
            const double ref_rev_sq =
              reference_enhanced(b, a, w, v, true);
            const double prod_l1 =
              production_enhanced<dtwc::core::L1Metric>(a, b, radius, v);
            const double prod_sq =
              production_enhanced<dtwc::core::SquaredL2Metric>(
                a, b, radius, v);
            const double prod_rev_l1 =
              production_enhanced<dtwc::core::L1Metric>(b, a, radius, v);
            const double prod_rev_sq =
              production_enhanced<dtwc::core::SquaredL2Metric>(
                b, a, radius, v);
            if (prod_l1 != ref_l1 || prod_sq != ref_sq
                || prod_rev_l1 != ref_rev_l1 || prod_rev_sq != ref_rev_sq)
              enhanced_audit.fail(
                "formula a=" + show(a) + " b=" + show(b)
                + " r=" + std::to_string(radius)
                + " v=" + std::to_string(v));
            if (std::max(prod_l1, prod_rev_l1) > exact.l1
                || std::max(prod_sq, prod_rev_sq) > exact.squared)
              enhanced_audit.fail(
                "inadmissible a=" + show(a) + " b=" + show(b));
            if (v == 1
                && (prod_l1 < direct_keogh(a, eb, false)
                    || prod_sq < direct_keogh(a, eb, true)
                    || prod_rev_l1 < direct_keogh(b, ea, false)
                    || prod_rev_sq < direct_keogh(b, ea, true)))
              enhanced_audit.fail(
                "V1 below Keogh a=" + show(a) + " b=" + show(b));
            ++enhanced_cases;
          }

          const double prod_l1 =
            production_webb<dtwc::core::L1Metric>(a, b, radius);
          const double prod_sq =
            production_webb<dtwc::core::SquaredL2Metric>(a, b, radius);
          const double prod_rev_l1 =
            production_webb<dtwc::core::L1Metric>(b, a, radius);
          const double prod_rev_sq =
            production_webb<dtwc::core::SquaredL2Metric>(b, a, radius);
          const double capped_l1 =
            reference_webb(a, ea, b, eb, w, false, true, &branch_hits);
          const double capped_sq =
            reference_webb(a, ea, b, eb, w, true, true, &branch_hits);
          const double capped_rev_l1 =
            reference_webb(b, eb, a, ea, w, false, true, &branch_hits);
          const double capped_rev_sq =
            reference_webb(b, eb, a, ea, w, true, true, &branch_hits);
          if (prod_l1 != capped_l1 || prod_sq != capped_sq
              || prod_rev_l1 != capped_rev_l1
              || prod_rev_sq != capped_rev_sq)
            webb_audit.fail(
              "capped formula a=" + show(a) + " b=" + show(b)
              + " r=" + std::to_string(radius));
          if (prod_l1 < direct_keogh(a, eb, false)
              || prod_sq < direct_keogh(a, eb, true)
              || prod_rev_l1 < direct_keogh(b, ea, false)
              || prod_rev_sq < direct_keogh(b, ea, true))
            webb_audit.fail(
              "below Keogh a=" + show(a) + " b=" + show(b));
          if (prod_l1 > exact.l1 || prod_sq > exact.squared
              || prod_rev_l1 > exact.l1 || prod_rev_sq > exact.squared)
            webb_audit.fail(
              "inadmissible a=" + show(a) + " b=" + show(b));
          ++webb_cases;

          const double exact_nolr_l1 =
            reference_webb(a, ea, b, eb, w, false, false);
          const double exact_nolr_sq =
            reference_webb(a, ea, b, eb, w, true, false);
          const double exact_nolr_rev_l1 =
            reference_webb(b, eb, a, ea, w, false, false);
          const double exact_nolr_rev_sq =
            reference_webb(b, eb, a, ea, w, true, false);
          for (std::size_t j = 0; j < n; ++j) {
            const FreeFlags cap = direct_free_flags(a, ea, eb, w, j, true);
            const FreeFlags centered =
              direct_free_flags(a, ea, eb, w, j, false);
            const FreeFlags cap_reverse =
              direct_free_flags(b, eb, ea, w, j, true);
            const FreeFlags centered_reverse =
              direct_free_flags(b, eb, ea, w, j, false);
            if ((cap.above && !centered.above)
                || (cap.below && !centered.below)
                || (cap_reverse.above && !centered_reverse.above)
                || (cap_reverse.below && !centered_reverse.below))
              predicate_audit.fail(
                "tail predicate implication a=" + show(a)
                + " b=" + show(b) + " j=" + std::to_string(j));
          }
          if (capped_l1 > exact_nolr_l1
              || capped_sq > exact_nolr_sq
              || capped_rev_l1 > exact_nolr_rev_l1
              || capped_rev_sq > exact_nolr_rev_sq)
            tail_audit.fail(
              "cap increased NoLR a=" + show(a) + " b=" + show(b));
          if (exact_nolr_l1 > exact.l1
              || exact_nolr_sq > exact.squared
              || exact_nolr_rev_l1 > exact.l1
              || exact_nolr_rev_sq > exact.squared)
            tail_audit.fail(
              "NoLR inadmissible a=" + show(a) + " b=" + show(b));
          ++tail_cases;
        }
      }
    }
  }

  INFO("first path violation: " << path_audit.first);
  REQUIRE(path_cases == 35982);
  REQUIRE(full_cover_cases == 7380);
  REQUIRE(path_audit.violations == 0);
  REQUIRE(full_path_counts[1] == 1);
  REQUIRE(full_path_counts[2] == 3);
  REQUIRE(full_path_counts[3] == 13);
  REQUIRE(full_path_counts[4] == 63);
  const Series path_a = { 0, 0, 2 };
  const Series path_b = { 0, 2, 2 };
  const PathCosts non_diagonal = exact_path_costs(path_a, path_b, 1);
  REQUIRE(
    point_cost(path_a[0], path_b[0], false)
      + point_cost(path_a[1], path_b[1], false)
      + point_cost(path_a[2], path_b[2], false)
    == 2.0);
  REQUIRE(
    point_cost(path_a[0], path_b[0], true)
      + point_cost(path_a[1], path_b[1], true)
      + point_cost(path_a[2], path_b[2], true)
    == 4.0);
  REQUIRE(non_diagonal.count > 1);
  REQUIRE(non_diagonal.l1 == 0.0);
  REQUIRE(non_diagonal.squared == 0.0);

  INFO("first Enhanced violation: " << enhanced_audit.first);
  REQUIRE(enhanced_cases == 68787);
  REQUIRE(enhanced_audit.violations == 0);

  INFO("first Webb violation: " << webb_audit.first);
  REQUIRE(webb_cases == 35982);
  REQUIRE(webb_audit.violations == 0);
  REQUIRE(branch_hits.full_upper);
  REQUIRE(branch_hits.full_lower);
  REQUIRE(branch_hits.overlap_upper);
  REQUIRE(branch_hits.overlap_lower);

  INFO("first tail violation: " << tail_audit.first);
  REQUIRE(tail_cases == 35982);
  REQUIRE(tail_audit.violations == 0);
  INFO("first predicate violation: " << predicate_audit.first);
  REQUIRE(predicate_audit.violations == 0);

  // Webb's four-point metric condition on all 70 multisets of five
  // asymmetrically spaced values, for both metrics and both orientations.
  constexpr std::array<double, 5> metric_values = { -4.0, -1.0, 0.0, 2.0, 7.0 };
  std::size_t metric_cases = 0;
  Audit metric_audit;
  for (std::size_t ia = 0; ia < metric_values.size(); ++ia) {
    for (std::size_t ix = ia; ix < metric_values.size(); ++ix) {
      for (std::size_t iy = ix; iy < metric_values.size(); ++iy) {
        for (std::size_t ib = iy; ib < metric_values.size(); ++ib) {
          const double a = metric_values[ia];
          const double x = metric_values[ix];
          const double y = metric_values[iy];
          const double b = metric_values[ib];
          for (const bool squared : { false, true }) {
            const double lhs = point_cost(a, b, squared);
            const double rhs = point_cost(a, y, squared)
                               + point_cost(b, x, squared)
                               - point_cost(x, y, squared);
            const double reverse_lhs = point_cost(b, a, squared);
            const double reverse_rhs = point_cost(b, x, squared)
                                       + point_cost(a, y, squared)
                                       - point_cost(y, x, squared);
            if (lhs < rhs || reverse_lhs < reverse_rhs)
              metric_audit.fail("four-point condition");
            if (!squared && (lhs != rhs || reverse_lhs != reverse_rhs))
              metric_audit.fail("L1 four-point equality");
            const double expected_slack = 2.0 * (x - a) * (b - y);
            if (squared
                && (lhs - rhs != expected_slack
                    || reverse_lhs - reverse_rhs != expected_slack))
              metric_audit.fail("squared four-point slack");
            ++metric_cases;
          }
        }
      }
    }
  }
  REQUIRE(metric_cases == 140);
  REQUIRE(metric_audit.violations == 0);

  // Default V=5 is outside the exhaustive small-length domain. A separate
  // full-matrix recurrence arbitrates four exact metric/window values.
  const Series v5_a = { 0, 4, 1, 7, 2, 9, 3, 8, 5, 6 };
  const Series v5_b = { 1, 0, 5, 2, 8, 3, 9, 4, 7, 6 };
  std::size_t enhanced_v5 = 0;
  const double v5_w0_l1 =
    production_enhanced<dtwc::core::L1Metric>(v5_a, v5_b, 0, 5);
  const double v5_w0_sq =
    production_enhanced<dtwc::core::SquaredL2Metric>(v5_a, v5_b, 0, 5);
  const double v5_w1_l1 =
    production_enhanced<dtwc::core::L1Metric>(v5_a, v5_b, 1, 5);
  const double v5_w1_sq =
    production_enhanced<dtwc::core::SquaredL2Metric>(v5_a, v5_b, 1, 5);
  REQUIRE(v5_w0_l1 == 38.0);
  REQUIRE(full_matrix_dtw(v5_a, v5_b, 0, false) == 38.0);
  ++enhanced_v5;
  REQUIRE(v5_w0_sq == 186.0);
  REQUIRE(full_matrix_dtw(v5_a, v5_b, 0, true) == 186.0);
  ++enhanced_v5;
  REQUIRE(v5_w1_l1 == 6.0);
  REQUIRE(full_matrix_dtw(v5_a, v5_b, 1, false) == 8.0);
  ++enhanced_v5;
  REQUIRE(v5_w1_sq == 6.0);
  REQUIRE(full_matrix_dtw(v5_a, v5_b, 1, true) == 8.0);
  ++enhanced_v5;
  REQUIRE(
    v5_w1_l1 == reference_enhanced(v5_a, v5_b, 1, 5, false));
  REQUIRE(
    v5_w1_sq == reference_enhanced(v5_a, v5_b, 1, 5, true));
  Series v5_odd_a = v5_a;
  Series v5_odd_b = v5_b;
  v5_odd_a[5] = 6.0; // strictly inside [3,9], equal to no donor
  v5_odd_a.push_back(3.0);
  v5_odd_b.push_back(5.0);
  const double odd_l1 =
    production_enhanced<dtwc::core::L1Metric>(v5_odd_a, v5_odd_b, 1, 5);
  const double odd_sq =
    production_enhanced<dtwc::core::SquaredL2Metric>(
      v5_odd_a, v5_odd_b, 1, 5);
  REQUIRE(odd_l1 == reference_enhanced(v5_odd_a, v5_odd_b, 1, 5, false));
  REQUIRE(odd_sq == reference_enhanced(v5_odd_a, v5_odd_b, 1, 5, true));
  REQUIRE(odd_l1 <= full_matrix_dtw(v5_odd_a, v5_odd_b, 1, false));
  REQUIRE(odd_sq <= full_matrix_dtw(v5_odd_a, v5_odd_b, 1, true));
  REQUIRE(enhanced_v5 == 4);

  // Explicitly pin the paper's uniformly tighter V=1 endpoint replacement,
  // then count both strict order directions at effective V=2.
  {
    const Series a = { -1, 0 };
    const Series b = { 0, -1 };
    const auto ea = direct_webb_envelope(a, 1);
    const auto eb = direct_webb_envelope(b, 1);
    const double enhanced = std::max(
      production_enhanced<dtwc::core::L1Metric>(a, b, 1, 1),
      production_enhanced<dtwc::core::L1Metric>(b, a, 1, 1));
    const double keogh =
      std::max(direct_keogh(a, eb, false), direct_keogh(b, ea, false));
    REQUIRE(enhanced == 2.0);
    REQUIRE(keogh == 0.0);
  }

  std::size_t order_witnesses = 0;
  {
    const Series a = { 10, 0, 0, 0 };
    const Series b = { 0, 10, 0, 0 };
    const auto ea = direct_webb_envelope(a, 1);
    const auto eb = direct_webb_envelope(b, 1);
    const double enhanced = std::max(
      production_enhanced<dtwc::core::L1Metric>(a, b, 1, 2),
      production_enhanced<dtwc::core::L1Metric>(b, a, 1, 2));
    const double keogh =
      std::max(direct_keogh(a, eb, false), direct_keogh(b, ea, false));
    REQUIRE(enhanced == 10.0);
    REQUIRE(keogh == 0.0);
    REQUIRE(enhanced > keogh);
    ++order_witnesses;
  }
  {
    const Series a = { -1, -1, -1, -1 };
    const Series b = { -1, -1, 0, -1 };
    const auto ea = direct_webb_envelope(a, 1);
    const auto eb = direct_webb_envelope(b, 1);
    const double enhanced = std::max(
      production_enhanced<dtwc::core::L1Metric>(a, b, 1, 2),
      production_enhanced<dtwc::core::L1Metric>(b, a, 1, 2));
    const double keogh =
      std::max(direct_keogh(a, eb, false), direct_keogh(b, ea, false));
    REQUIRE(enhanced == 0.0);
    REQUIRE(keogh == 1.0);
    REQUIRE(keogh > enhanced);
    ++order_witnesses;
  }
  REQUIRE(order_witnesses == 2);

  // Webb strictly dominates matching-direction Keogh under both metrics.
  const Series webb_a = { -1, -1 };
  const Series webb_b = { -1, 2 };
  const auto webb_eb = direct_webb_envelope(webb_b, 1);
  std::size_t webb_strict = 0;
  const double strict_webb_l1 =
    production_webb<dtwc::core::L1Metric>(webb_a, webb_b, 1);
  const double strict_webb_sq =
    production_webb<dtwc::core::SquaredL2Metric>(webb_a, webb_b, 1);
  REQUIRE(direct_keogh(webb_a, webb_eb, false) == 0.0);
  REQUIRE(strict_webb_l1 == 3.0);
  REQUIRE(strict_webb_l1 > direct_keogh(webb_a, webb_eb, false));
  ++webb_strict;
  REQUIRE(direct_keogh(webb_a, webb_eb, true) == 0.0);
  REQUIRE(strict_webb_sq == 9.0);
  REQUIRE(strict_webb_sq > direct_keogh(webb_a, webb_eb, true));
  ++webb_strict;
  REQUIRE(webb_strict == 2);

  // Conservative tail flag is strictly lower than the exact clipped
  // predicate under both metrics, without using an over-wide radius.
  const Series tail_a = { 0, 0, 1, 1 };
  const Series tail_b = { 1, 1, 2, 0 };
  const auto tail_ea = direct_webb_envelope(tail_a, 1);
  const auto tail_eb = direct_webb_envelope(tail_b, 1);
  const double tail_prod_l1 =
    production_webb<dtwc::core::L1Metric>(tail_a, tail_b, 1);
  const double tail_prod_sq =
    production_webb<dtwc::core::SquaredL2Metric>(tail_a, tail_b, 1);
  const double tail_exact_l1 =
    reference_webb(tail_a, tail_ea, tail_b, tail_eb, 1, false, false);
  const double tail_exact_sq =
    reference_webb(tail_a, tail_ea, tail_b, tail_eb, 1, true, false);
  REQUIRE(tail_prod_l1 == 3.0);
  REQUIRE(tail_exact_l1 == 4.0);
  REQUIRE(tail_prod_l1 < tail_exact_l1);
  REQUIRE(tail_prod_sq == 3.0);
  REQUIRE(tail_exact_sq == 4.0);
  REQUIRE(tail_prod_sq < tail_exact_sq);

  // The registered counter represents upper and lower branch orientations.
  std::size_t tail_strict = 0;
  const Series upper_tail_a = { 0, 0, 20, 5, 5, 5, 5 };
  const Series upper_tail_b = { 0, 0, 0, 0, 0, 10, 0 };
  for (const double sign : { 1.0, -1.0 }) {
    Series a = upper_tail_a;
    Series b = upper_tail_b;
    for (double &value : a) value *= sign;
    for (double &value : b) value *= sign;
    const auto ea = direct_webb_envelope(a, 2);
    const auto eb = direct_webb_envelope(b, 2);
    const double capped_l1 =
      production_webb<dtwc::core::L1Metric>(a, b, 2);
    const double capped_sq =
      production_webb<dtwc::core::SquaredL2Metric>(a, b, 2);
    const double exact_l1 =
      reference_webb(a, ea, b, eb, 2, false, false);
    const double exact_sq =
      reference_webb(a, ea, b, eb, 2, true, false);
    REQUIRE(capped_l1 == 20.0);
    REQUIRE(capped_sq == 400.0);
    REQUIRE(exact_l1 == 30.0);
    REQUIRE(exact_sq == 450.0);
    REQUIRE(capped_l1 < exact_l1);
    REQUIRE(capped_sq < exact_sq);
    REQUIRE(exact_l1 <= full_matrix_dtw(a, b, 2, false));
    REQUIRE(exact_sq <= full_matrix_dtw(a, b, 2, true));
    ++tail_strict;
  }
  REQUIRE(tail_strict == 2);

  // F54: force the live Enhanced route to need Keogh's larger value.
  const Series cascade_a = { 0, 10, 0, 0, 0, 0 };
  const Series cascade_b = { 0, 0, 0, 10, 0, 0 };
  const Series cascade_c = { 0, 0, 10, 0, 0, 0 };
  const auto cascade_ea = direct_webb_envelope(cascade_a, 1);
  const auto cascade_eb = direct_webb_envelope(cascade_b, 1);
  const double cascade_enhanced = std::max(
    production_enhanced<dtwc::core::L1Metric>(
      cascade_a, cascade_b, 1, 5),
    production_enhanced<dtwc::core::L1Metric>(
      cascade_b, cascade_a, 1, 5));
  const double cascade_keogh = std::max(
    direct_keogh(cascade_a, cascade_eb, false),
    direct_keogh(cascade_b, cascade_ea, false));
  REQUIRE(
    dtwc::core::lb_kim(
      dtwc::core::compute_summary(cascade_a),
      dtwc::core::compute_summary(cascade_b))
    == 0.0);
  REQUIRE(cascade_enhanced == 0.0);
  REQUIRE(cascade_keogh == 10.0);

  constexpr std::array<std::array<double, 3>, 3> expected_matrix = { { { { 0.0, 0.0, 0.0 } },
                                                                       { { 0.0, 0.0, 20.0 } },
                                                                       { { 0.0, 20.0, 0.0 } } } };
  std::size_t cascade_routes = 0;
  auto direct_problem = make_problem(
    { cascade_c, cascade_a, cascade_b }, 1, "d3_direct");
  const dtwc::core::PruningStats stats =
    dtwc::core::fill_distance_matrix_pruned(
      direct_problem, 1, dtwc::LowerBoundStrategy::Enhanced);
  REQUIRE(stats.total_pairs == 3);
  REQUIRE(stats.pruned_by_lb_kim == 0);
  REQUIRE(stats.pruned_by_lb_keogh == 1);
  REQUIRE(stats.early_abandoned == 1);
  REQUIRE(stats.computed_full_dtw == 2);
  REQUIRE(
    stats.computed_full_dtw + stats.pruned_by_lb_kim
      + stats.pruned_by_lb_keogh
    == stats.total_pairs);
  REQUIRE(stats.early_abandoned <= stats.pruned_by_lb_keogh);
  REQUIRE(direct_problem.is_distance_matrix_filled());
  for (std::size_t i = 0; i < expected_matrix.size(); ++i)
    for (std::size_t j = 0; j < expected_matrix.size(); ++j)
      REQUIRE(
        direct_problem.dense_distance_matrix().get(i, j)
        == expected_matrix[i][j]);
  ++cascade_routes;

  auto public_problem = make_problem(
    { cascade_c, cascade_a, cascade_b }, 1, "d3_public");
  public_problem.set_distance_strategy(dtwc::DistanceMatrixStrategy::Pruned);
  public_problem.set_lb_strategy(dtwc::LowerBoundStrategy::Enhanced);
  public_problem.set_verbose(true);
  std::ostringstream public_output;
  {
    struct RestoreCout
    {
      std::streambuf *previous;
      ~RestoreCout() { std::cout.rdbuf(previous); }
    } restore{ std::cout.rdbuf(public_output.rdbuf()) };
    public_problem.fill_distance_matrix();
  }
  REQUIRE(public_problem.is_distance_matrix_filled());
  for (std::size_t i = 0; i < expected_matrix.size(); ++i)
    for (std::size_t j = 0; j < expected_matrix.size(); ++j)
      REQUIRE(
        public_problem.dense_distance_matrix().get(i, j)
        == expected_matrix[i][j]);
  REQUIRE(
    public_output.str()
    ==
    "Distance matrix is being filled!\n"
    "Pruned strategy: 3 pairs, 1 early-abandoned, pruning ratio: "
    "0.333333\n"
    "Distance matrix has been filled!\n");
  ++cascade_routes;
  REQUIRE(cascade_routes == 2);

  std::cout
    << "D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 "
       "full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 "
       "webb_cases=35982 webb_branches=4/4 webb_strict=2/2 "
       "tail_cases=35982 tail_strict=2/2 metric_cases=140 "
       "order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS\n";
}
