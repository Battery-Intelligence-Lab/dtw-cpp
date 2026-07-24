/**
 * @file unit_test_nearest_medoid_assignment.cpp
 * @brief F13 public-route contract for nearest-medoid assignment.
 *
 * The independent oracle below consumes literal distance tables. It shares no
 * DTW kernel, Problem cache, or production assignment helper with the routes it
 * judges.
 */

#include <dtwc.hpp>
#include <algorithms/clarans.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/fast_pam.hpp>
#include <error.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using dtwc::Problem;
using dtwc::core::ClusteringResult;

struct OracleAssignment
{
  std::vector<int> labels;
  std::vector<double> nearest;
  std::vector<double> second;
  double objective = 0.0;
};

std::uint64_t bits(double value)
{
  return std::bit_cast<std::uint64_t>(value);
}

OracleAssignment independent_assignment_oracle(
  const std::vector<std::vector<double>> &distances)
{
  if (distances.empty() || distances.front().empty())
    throw std::invalid_argument(
      "nearest-medoid oracle: distance table must be non-empty.");

  const std::size_t k = distances.front().size();
  OracleAssignment result;
  result.labels.resize(distances.size());
  result.nearest.resize(distances.size());
  result.second.resize(
    distances.size(), std::numeric_limits<double>::max());

  volatile double total = 0.0;
  for (std::size_t point = 0; point < distances.size(); ++point) {
    if (distances[point].size() != k)
      throw std::invalid_argument(
        "nearest-medoid oracle: distance table is ragged.");

    struct Candidate
    {
      double distance;
      std::size_t slot;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(k);
    for (std::size_t slot = 0; slot < k; ++slot) {
      const double distance = distances[point][slot];
      if (!std::isfinite(distance)) {
        throw std::runtime_error(
          "nearest-medoid oracle: non-finite nearest-medoid distance at point "
          + std::to_string(point) + ", medoid slot "
          + std::to_string(slot) + ".");
      }
      candidates.push_back({distance, slot});
    }
    std::stable_sort(
      candidates.begin(), candidates.end(),
      [](const Candidate &lhs, const Candidate &rhs) {
        if (lhs.distance < rhs.distance) return true;
        if (rhs.distance < lhs.distance) return false;
        return lhs.slot < rhs.slot;
      });

    result.labels[point] = static_cast<int>(candidates[0].slot);
    result.nearest[point] = candidates[0].distance;
    if (k > 1) result.second[point] = candidates[1].distance;

    const double next = total + candidates[0].distance;
    if (!std::isfinite(next)) {
      throw std::runtime_error(
        "nearest-medoid oracle: nearest-medoid objective became non-finite "
        "after point " + std::to_string(point) + ".");
    }
    total = next;
  }

  const double value = total;
  result.objective = value == 0.0 ? 0.0 : value;
  return result;
}

template <typename T>
Problem scalar_problem(
  std::vector<T> values, const std::string &name = "f13_scalar")
{
  std::vector<std::vector<T>> series;
  std::vector<std::string> names;
  series.reserve(values.size());
  names.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    series.push_back({values[i]});
    names.push_back("s" + std::to_string(i));
  }

  Problem problem(name);
  problem.set_data(dtwc::Data(std::move(series), std::move(names)));
  return problem;
}

template <typename T>
Problem no_path_problem(const std::string &name)
{
  std::vector<std::vector<T>> series{
    {T(0)},
    {T(0), T(0), T(0)}
  };
  std::vector<std::string> names{"short", "long"};
  Problem problem(name);
  problem.set_data(dtwc::Data(std::move(series), std::move(names)));
  problem.band = 0;
  return problem;
}

Problem poisoned_matrix_problem(double poison)
{
  auto problem = scalar_problem<double>({0.0, 1.0, 2.0}, "f13_poison");
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(3);
  matrix.set(0, 0, 0.0);
  matrix.set(0, 1, poison);
  matrix.set(0, 2, 2.0);
  matrix.set(1, 1, 0.0);
  matrix.set(1, 2, 1.0);
  matrix.set(2, 2, 0.0);
  return problem;
}

template <typename Fn>
void require_invalid_input(Fn &&fn, const std::string &expected)
{
  bool caught = false;
  try {
    std::forward<Fn>(fn)();
  } catch (const dtwc::InvalidInput &error) {
    caught = true;
    CHECK(std::string(error.what()) == expected);
  } catch (const std::exception &error) {
    FAIL("wrong exception type: " << error.what());
  }
  CHECK(caught);
}

void require_result(
  const ClusteringResult &result, const std::vector<int> &medoids,
  const std::vector<int> &labels, double objective)
{
  CHECK(result.medoid_indices == medoids);
  CHECK(result.labels == labels);
  CHECK(bits(result.total_cost) == bits(objective));
}

dtwc::algorithms::CLARAOptions clara_options(
  int k, int sample_size, unsigned seed)
{
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = k;
  options.sample_size = sample_size;
  options.n_samples = 1;
  options.random_seed = seed;
  return options;
}

dtwc::algorithms::CLARANSOptions clarans_options(
  int k, unsigned seed, int max_neighbor = 0)
{
  dtwc::algorithms::CLARANSOptions options;
  options.n_clusters = k;
  options.num_local = 1;
  options.max_neighbor = max_neighbor;
  options.random_seed = seed;
  return options;
}

} // namespace

TEST_CASE("F13 independent oracle pins ties, presence, and ordered bits",
          "[F13][medoid-assignment][oracle]")
{
  const auto midpoint = independent_assignment_oracle({
    {0.0, 2.0},
    {1.0, 1.0},
    {2.0, 0.0},
  });
  REQUIRE(midpoint.labels == std::vector<int>{0, 0, 1});
  REQUIRE(bits(midpoint.objective) == UINT64_C(0x3ff0000000000000));

  const auto reversed_slots = independent_assignment_oracle({
    {2.0, 0.0},
    {1.0, 1.0},
    {0.0, 2.0},
  });
  REQUIRE(reversed_slots.labels == std::vector<int>{1, 0, 0});
  REQUIRE(bits(reversed_slots.objective)
          == UINT64_C(0x3ff0000000000000));

  const auto mixed = independent_assignment_oracle({
    {0x1p53, 0x1p53},
    {4.0, 0.0},
    {1.0, 1.0},
    {-0.0, +0.0},
    {-0x1p53, -0x1p53},
    {2.0, 3.0},
  });
  REQUIRE(mixed.labels == std::vector<int>{0, 1, 0, 0, 0, 0});
  REQUIRE(mixed.objective == 2.0);
  REQUIRE(bits(mixed.objective) == UINT64_C(0x4000000000000000));

  const std::vector<std::uint64_t> expected_nearest{
    UINT64_C(0x4340000000000000),
    UINT64_C(0x0000000000000000),
    UINT64_C(0x3ff0000000000000),
    UINT64_C(0x8000000000000000),
    UINT64_C(0xc340000000000000),
    UINT64_C(0x4000000000000000),
  };
  REQUIRE(mixed.nearest.size() == expected_nearest.size());
  for (std::size_t i = 0; i < mixed.nearest.size(); ++i)
    CHECK(bits(mixed.nearest[i]) == expected_nearest[i]);

  const auto zero = independent_assignment_oracle({{-0.0, +0.0}});
  REQUIRE(zero.labels == std::vector<int>{0});
  REQUIRE(bits(zero.objective) == UINT64_C(0x0000000000000000));

  const double maximum = std::numeric_limits<double>::max();
  const auto sentinel = independent_assignment_oracle({{maximum, maximum}});
  REQUIRE(sentinel.labels == std::vector<int>{0});
  REQUIRE(bits(sentinel.nearest[0]) == UINT64_C(0x7fefffffffffffff));
  REQUIRE(bits(sentinel.second[0]) == UINT64_C(0x7fefffffffffffff));
  REQUIRE(bits(sentinel.objective) == UINT64_C(0x7fefffffffffffff));

  const double adjacent = std::nextafter(maximum, 0.0);
  const auto below = independent_assignment_oracle({{adjacent, maximum}});
  REQUIRE(below.labels == std::vector<int>{0});
  REQUIRE(bits(below.nearest[0]) == UINT64_C(0x7feffffffffffffe));

  const auto finite_huge =
    independent_assignment_oracle({{0x1p1021}, {0x1p1021}});
  REQUIRE(bits(finite_huge.objective) == UINT64_C(0x7fd0000000000000));

  constexpr double huge = 0x1.8p+1023;
  REQUIRE_THROWS_WITH(
    independent_assignment_oracle({
      {huge, huge},
      {huge, huge},
    }),
    "nearest-medoid oracle: nearest-medoid objective became non-finite "
    "after point 1.");
  REQUIRE_THROWS_WITH(
    independent_assignment_oracle({
      {0.0, 0.0},
      {huge, huge},
      {huge, huge},
      {0.0, 0.0},
    }),
    "nearest-medoid oracle: nearest-medoid objective became non-finite "
    "after point 2.");
}

TEST_CASE("F13 independent oracle rejects every non-finite coordinate",
          "[F13][medoid-assignment][oracle][nonfinite]")
{
  const std::vector<double> poisons{
    std::bit_cast<double>(UINT64_C(0x7ff8000000000f13)),
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
  };
  for (const double poison : poisons) {
    REQUIRE_THROWS_WITH(
      independent_assignment_oracle({{poison, 1.0}}),
      "nearest-medoid oracle: non-finite nearest-medoid distance at point 0, "
      "medoid slot 0.");
    REQUIRE_THROWS_WITH(
      independent_assignment_oracle({{1.0, poison}}),
      "nearest-medoid oracle: non-finite nearest-medoid distance at point 0, "
      "medoid slot 1.");
  }
}

TEST_CASE("F13 public assignment routes agree on an exact midpoint tie",
          "[F13][medoid-assignment][tie][public]")
{
  const std::vector<int> medoids{0, 2};
  const std::vector<int> labels{0, 0, 1};

  auto pam_problem = scalar_problem<double>({0.0, 1.0, 2.0});
  require_result(
    dtwc::fast_pam_swap(
      pam_problem, medoids, 0, dtwc::PAMVariant::FastPAM1),
    medoids, labels, 1.0);

  auto clarans_problem = scalar_problem<double>({0.0, 1.0, 2.0});
  require_result(
    dtwc::algorithms::clarans(
      clarans_problem, clarans_options(2, 4)),
    medoids, labels, 1.0);

  auto clara_f64 = scalar_problem<double>({0.0, 1.0, 2.0});
  require_result(
    dtwc::algorithms::fast_clara(
      clara_f64, clara_options(2, 2, 0)),
    medoids, labels, 1.0);
  CHECK(clara_f64.dense_distance_matrix().size() == 0);

  auto clara_f32 = scalar_problem<float>({0.0f, 1.0f, 2.0f});
  require_result(
    dtwc::algorithms::fast_clara(
      clara_f32, clara_options(2, 2, 0)),
    medoids, labels, 1.0);
  CHECK(clara_f32.dense_distance_matrix().size() == 0);

  auto lloyd = scalar_problem<double>({0.0, 1.0, 2.0});
  lloyd.centroids_ind = medoids;
  lloyd.assign_clusters();
  REQUIRE(lloyd.clusters_ind == labels);
}

TEST_CASE("F13 ties select the first slot, not the smallest global index",
          "[F13][medoid-assignment][tie][slot-order]")
{
  const std::vector<int> reversed_medoids{2, 0};
  const std::vector<int> reversed_labels{1, 0, 0};

  auto pam_problem = scalar_problem<double>({0.0, 1.0, 2.0});
  require_result(
    dtwc::fast_pam_swap(
      pam_problem, reversed_medoids, 0, dtwc::PAMVariant::FastPAM1),
    reversed_medoids, reversed_labels, 1.0);

  auto clara_problem = scalar_problem<double>({0.0, 1.0, 2.0});
  require_result(
    dtwc::algorithms::fast_clara(
      clara_problem, clara_options(2, 2, 10)),
    reversed_medoids, reversed_labels, 1.0);

  auto lloyd = scalar_problem<double>({0.0, 1.0, 2.0});
  lloyd.centroids_ind = reversed_medoids;
  lloyd.assign_clusters();
  REQUIRE(lloyd.clusters_ind == reversed_labels);

  auto swapped = scalar_problem<double>(
    {4.0, 0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 10.0});
  auto options = clarans_options(2, 317, 1);
  options.max_dtw_evals = 24;
  require_result(
    dtwc::algorithms::clarans(swapped, options),
    {6, 3}, {0, 0, 0, 1, 0, 0, 0, 1}, 9.0);
}

TEST_CASE("F13 published objectives use the point-ordered binary64 fold",
          "[F13][medoid-assignment][objective][order]")
{
  const std::vector<double> values{0.0, 0x1p53, 1.0, 1.0, 0.0};
  const std::vector<int> medoids{0, 4};
  const std::vector<int> labels(5, 0);
  constexpr double expected = 0x1p53;

  const auto oracle = independent_assignment_oracle({
    {0.0, 0.0},
    {0x1p53, 0x1p53},
    {1.0, 1.0},
    {1.0, 1.0},
    {0.0, 0.0},
  });
  REQUIRE(oracle.labels == labels);
  REQUIRE(bits(oracle.objective) == UINT64_C(0x4340000000000000));

  auto pam_problem = scalar_problem<double>(values);
  require_result(
    dtwc::fast_pam_swap(
      pam_problem, medoids, 0, dtwc::PAMVariant::FastPAM1),
    medoids, labels, expected);

  auto clarans_problem = scalar_problem<double>(values);
  require_result(
    dtwc::algorithms::clarans(
      clarans_problem, clarans_options(2, 20)),
    medoids, labels, expected);

  auto clara_f64 = scalar_problem<double>(values);
  require_result(
    dtwc::algorithms::fast_clara(
      clara_f64, clara_options(2, 2, 11)),
    medoids, labels, expected);

  auto clara_f32 = scalar_problem<float>(
    {0.0f, 0x1p53f, 1.0f, 1.0f, 0.0f});
  require_result(
    dtwc::algorithms::fast_clara(
      clara_f32, clara_options(2, 2, 11)),
    medoids, labels, expected);

  auto lloyd = scalar_problem<double>(values);
  lloyd.centroids_ind = medoids;
  lloyd.assign_clusters();
  REQUIRE(lloyd.clusters_ind == labels);
  REQUIRE(bits(lloyd.find_total_cost())
          == UINT64_C(0x4340000000000000));
}

TEST_CASE("F13 public assignments reject non-winning infinities",
          "[F13][medoid-assignment][nonfinite][distance]")
{
  for (const double poison : {
         std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()}) {
    auto pam_problem = poisoned_matrix_problem(poison);
    require_invalid_input(
      [&] {
        (void)dtwc::fast_pam_swap(
          pam_problem, {0, 2}, 0, dtwc::PAMVariant::FastPAM1);
      },
      "fast_pam: non-finite nearest-medoid distance at point 1, "
      "medoid slot 0 (index 0).");

    auto clarans_problem = poisoned_matrix_problem(poison);
    require_invalid_input(
      [&] {
        (void)dtwc::algorithms::clarans(
          clarans_problem, clarans_options(2, 4));
      },
      "clarans: non-finite nearest-medoid distance at point 1, "
      "medoid slot 0 (index 0).");

    auto lloyd = poisoned_matrix_problem(poison);
    lloyd.centroids_ind = {0, 2};
    require_invalid_input(
      [&] { lloyd.assign_clusters(); },
      "kmedoids_lloyd: non-finite nearest-medoid distance at point 1, "
      "medoid slot 0 (index 0).");
  }

  std::vector<double> f64_values(65, std::numeric_limits<double>::max());
  f64_values[0] = -std::numeric_limits<double>::max();
  auto clara_f64 = scalar_problem<double>(
    std::move(f64_values), "f13_clara_nonfinite_f64");
  require_invalid_input(
    [&] {
      (void)dtwc::algorithms::fast_clara(
        clara_f64, clara_options(1, 2, 0));
    },
    "fast_clara: non-finite nearest-medoid distance at point 0, "
    "medoid slot 0 (index 25).");

  std::vector<float> f32_values(65, std::numeric_limits<float>::max());
  f32_values[0] = -std::numeric_limits<float>::max();
  auto clara_f32 = scalar_problem<float>(
    std::move(f32_values), "f13_clara_nonfinite_f32");
  require_invalid_input(
    [&] {
      (void)dtwc::algorithms::fast_clara(
        clara_f32, clara_options(1, 2, 0));
    },
    "fast_clara: non-finite nearest-medoid distance at point 0, "
    "medoid slot 0 (index 25).");
}

TEST_CASE("F13 finite assignment distances cannot overflow the objective",
          "[F13][medoid-assignment][nonfinite][objective]")
{
  constexpr double huge = 0x1.8p+1023;
  const std::vector<double> values{0.0, huge, huge, 0.0};

  auto pam_problem = scalar_problem<double>(values);
  require_invalid_input(
    [&] {
      (void)dtwc::fast_pam_swap(
        pam_problem, {0, 3}, 0, dtwc::PAMVariant::FastPAM1);
    },
    "fast_pam: nearest-medoid objective became non-finite after point 2.");

  auto clarans_problem = scalar_problem<double>(values);
  require_invalid_input(
    [&] {
      (void)dtwc::algorithms::clarans(
        clarans_problem, clarans_options(2, 0));
    },
    "clarans: nearest-medoid objective became non-finite after point 2.");

  auto clara_problem = scalar_problem<double>(values);
  require_invalid_input(
    [&] {
      (void)dtwc::algorithms::fast_clara(
        clara_problem, clara_options(2, 2, 8));
    },
    "fast_clara: nearest-medoid objective became non-finite after point 2.");

  auto lloyd = scalar_problem<double>(values);
  lloyd.centroids_ind = {0, 3};
  lloyd.assign_clusters();
  require_invalid_input(
    [&] { (void)lloyd.find_total_cost(); },
    "kmedoids_lloyd: nearest-medoid objective became non-finite after point 2.");
}

TEST_CASE("F13 finite no-path objectives publish in both precisions",
          "[F13][medoid-assignment][sentinel][presence]")
{
  const double sentinel = std::numeric_limits<double>::max();

  SECTION("FastPAM")
  {
    auto f64 = no_path_problem<double>("f13_pam_no_path_f64");
    require_result(
      dtwc::fast_pam_swap(
        f64, {0}, 0, dtwc::PAMVariant::FastPAM1),
      {0}, {0, 0}, sentinel);

    auto f32 = no_path_problem<float>("f13_pam_no_path_f32");
    require_result(
      dtwc::fast_pam_swap(
        f32, {0}, 0, dtwc::PAMVariant::FastPAM1),
      {0}, {0, 0}, sentinel);
  }

  SECTION("CLARANS")
  {
    auto f64 = no_path_problem<double>("f13_clarans_no_path_f64");
    const auto result_f64 = dtwc::algorithms::clarans(
      f64, clarans_options(1, 0));
    CHECK(result_f64.medoid_indices.size() == 1);
    CHECK(result_f64.labels == std::vector<int>{0, 0});
    CHECK(bits(result_f64.total_cost) == bits(sentinel));

    auto f32 = no_path_problem<float>("f13_clarans_no_path_f32");
    const auto result_f32 = dtwc::algorithms::clarans(
      f32, clarans_options(1, 0));
    CHECK(result_f32.medoid_indices.size() == 1);
    CHECK(result_f32.labels == std::vector<int>{0, 0});
    CHECK(bits(result_f32.total_cost) == bits(sentinel));
  }

  SECTION("FastCLARA")
  {
    auto f64 = no_path_problem<double>("f13_clara_no_path_f64");
    const auto result_f64 = dtwc::algorithms::fast_clara(
      f64, clara_options(1, 1, 0));
    CHECK(result_f64.medoid_indices.size() == 1);
    CHECK(result_f64.labels == std::vector<int>{0, 0});
    CHECK(bits(result_f64.total_cost) == bits(sentinel));

    auto f32 = no_path_problem<float>("f13_clara_no_path_f32");
    const auto result_f32 = dtwc::algorithms::fast_clara(
      f32, clara_options(1, 1, 0));
    CHECK(result_f32.medoid_indices.size() == 1);
    CHECK(result_f32.labels == std::vector<int>{0, 0});
    CHECK(bits(result_f32.total_cost) == bits(sentinel));
  }
}
