/**
 * @file unit_test_nearest_medoid_assignment.cpp
 * @brief F13 public-route contract for nearest-medoid assignment: where a
 *        non-finite distance is refused, first-slot ties, the point-ordered
 *        objective and its one finite check.
 */

#include <dtwc.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/one_batch_pam.hpp>
#include <base/error.hpp>

#include "../../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <bit>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using dtwc::Problem;
using dtwc::core::ClusteringResult;

std::uint64_t bits(double value)
{
  return std::bit_cast<std::uint64_t>(value);
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
  const ClusteringResult &result, const std::vector<dtwc::index_t> &medoids,
  const std::vector<dtwc::index_t> &labels, double objective)
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

} // namespace

TEST_CASE("F13 ties select the first slot, not the smallest global index",
          "[F13][medoid-assignment][tie][slot-order]")
{
  const std::vector<dtwc::index_t> reversed_medoids{2, 0};
  const std::vector<dtwc::index_t> reversed_labels{1, 0, 0};

  auto clara_problem = scalar_problem<double>({0.0, 1.0, 2.0});
  require_result(
    dtwc::algorithms::fast_clara(
      clara_problem, clara_options(2, 2, 10)),
    reversed_medoids, reversed_labels, 1.0);

  auto lloyd = scalar_problem<double>({0.0, 1.0, 2.0});
  lloyd.centroids_ind = reversed_medoids;
  lloyd.assign_clusters();
  REQUIRE(lloyd.clusters_ind == reversed_labels);
}

TEST_CASE("F13 published objectives use the point-ordered binary64 fold",
          "[F13][medoid-assignment][objective][order]")
{
  const std::vector<double> values{0.0, 0x1p53, 1.0, 1.0, 0.0};
  const std::vector<dtwc::index_t> medoids{0, 4};
  // Series 4 duplicates series 0: every other point ties and takes the first
  // slot, but medoid 4 serves itself so its cluster is not published empty.
  const std::vector<dtwc::index_t> labels{0, 0, 0, 0, 1};
  // In point order every partial sum is 2^53: 2^53 + 1 is a tie that rounds to
  // the even 2^53. An order that adds the two 1s first gives 2^53 + 2.
  constexpr double expected = 0x1p53;
  REQUIRE(bits(expected) == UINT64_C(0x4340000000000000));

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
  lloyd.set_n_clusters(static_cast<int>(medoids.size())); // a clustering holds one medoid per cluster
  lloyd.centroids_ind = medoids;
  lloyd.assign_clusters();
  REQUIRE(lloyd.clusters_ind == labels);
  REQUIRE(bits(lloyd.find_total_cost())
          == UINT64_C(0x4340000000000000));
}

TEST_CASE("Every matrix intake refuses a distance that is not finite, naming the pair",
          "[medoid-assignment][nonfinite][intake]")
{
  // The clustering loops read a filled matrix unchecked, so every way a matrix
  // enters a Problem from outside its fill scans it once. The bad matrix holds a
  // hole (pair 0-1, NaN: computed later) before +inf (pair 1-2): the scan passes
  // the hole and names the infinity.
  constexpr double inf = std::numeric_limits<double>::infinity();
  const auto write_bad = [](dtwc::core::DistanceMatrix &matrix) {
    matrix.set(0, 0, 0.0);
    matrix.set(1, 1, 0.0);
    matrix.set(2, 2, 0.0);
    matrix.set(0, 2, 2.0);
    matrix.set(1, 2, inf);
  };
  const std::string bad =
    ": the distance between series 1 and 2 is +inf; a distance must be finite.";
  const dtwc::test_support::ScratchDirectory scratch("medoid_intake");
  const auto three = [] { return scalar_problem<double>({0.0, 1.0, 2.0}, "intake"); };

  struct Row
  {
    std::string expected;
    std::function<void(Problem &)> install;
  };
  const std::vector<Row> rows{
    { "Problem::fill_distance_matrix" + bad, // the commit point of writable_distance_matrix()
      [&](Problem &problem) {
        auto &matrix = problem.writable_distance_matrix();
        matrix.resize(3);
        write_bad(matrix);
        problem.fill_distance_matrix();
      } },
    { "Problem::read_distance_matrix" + bad,
      [&](Problem &problem) {
        const auto path = scratch.path / "bad.csv";
        std::ofstream(path) << "0,,2\n,0,inf\n2,inf,0\n";
        problem.read_distance_matrix(path);
      } },
    { "load_checkpoint" + bad,
      [&](Problem &problem) {
        auto writer = three();
        auto &matrix = writer.writable_distance_matrix();
        matrix.resize(3);
        write_bad(matrix);
        dtwc::save_checkpoint(writer, scratch.path.string());
        (void)dtwc::load_checkpoint(problem, scratch.path.string());
      } },
#ifdef DTWC_HAS_MMAP
    { "Problem::use_mmap_distance_matrix" + bad,
      [&](Problem &problem) {
        const auto path = scratch.path / "bad.dtwm";
        {
          auto writer = three();
          writer.use_mmap_distance_matrix(path);
          write_bad(writer.writable_distance_matrix());
        } // unmapped; the file keeps the values
        problem.use_mmap_distance_matrix(path);
      } },
#endif
  };
  for (const auto &row : rows) {
    CAPTURE(row.expected);
    auto problem = three();
    require_invalid_input([&] { row.install(problem); }, row.expected);
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }
}

TEST_CASE("F13 FastCLARA's assignment, which calls the DTW function, rejects a non-winning infinity",
          "[F13][medoid-assignment][nonfinite][distance]")
{
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

  auto clara_problem = scalar_problem<double>(values);
  require_invalid_input(
    [&] {
      (void)dtwc::algorithms::fast_clara(
        clara_problem, clara_options(2, 2, 8));
    },
    "fast_clara: the nearest-medoid objective is not finite (a distance or their sum overflowed).");

  auto lloyd = scalar_problem<double>(values);
  lloyd.set_n_clusters(2);
  lloyd.centroids_ind = {0, 3};
  lloyd.assign_clusters();
  require_invalid_input(
    [&] { (void)lloyd.find_total_cost(); },
    "kmedoids_lloyd: the nearest-medoid objective is not finite (a distance or their sum overflowed).");
}

TEST_CASE("F13 no-path requests are rejected on every route in both precisions",
          "[F13][medoid-assignment][sentinel][presence]")
{
  // FX-1: a band narrower than a length difference is rejected before any pair
  // is computed: on the fill, the lazy dist_by_ind path and the dtw_function
  // accessors. The kernels' finite max() sentinel never becomes an objective.
  const std::string no_path =
    ": band = 0 is narrower than the length difference between series 'short' "
    "(index 0, length 1) and series 'long' (index 1, length 3), so no warping "
    "path fits that pair. The smallest feasible band is 2; pass band >= 2, or "
    "band = -1 for full DTW.";

  SECTION("FastPAM")
  {
    auto f64 = no_path_problem<double>("f13_pam_no_path_f64");
    require_invalid_input(
      [&] { (void)dtwc::fast_pam(f64, 1); },
      "Problem::fill_distance_matrix" + no_path);

    auto f32 = no_path_problem<float>("f13_pam_no_path_f32");
    require_invalid_input(
      [&] { (void)dtwc::fast_pam(f32, 1); },
      "Problem::fill_distance_matrix" + no_path);
  }

  // FastCLARA's assignment and OneBatchPAM compute through the dtw_function
  // accessors, which validate the same request once, serially, before their
  // parallel loops. FastCLARA published the finite no-path objective here
  // (total_cost bits == DBL_MAX) until the accessors did.

  SECTION("FastCLARA")
  {
    auto f64 = no_path_problem<double>("f13_clara_no_path_f64");
    require_invalid_input(
      [&] { (void)dtwc::algorithms::fast_clara(f64, clara_options(1, 1, 0)); },
      "Problem::dtw_function" + no_path);
    CHECK(f64.labels().empty());

    auto f32 = no_path_problem<float>("f13_clara_no_path_f32");
    require_invalid_input(
      [&] { (void)dtwc::algorithms::fast_clara(f32, clara_options(1, 1, 0)); },
      "Problem::dtw_function_f32" + no_path);
    CHECK(f32.labels().empty());
  }

  SECTION("OneBatchPAM")
  {
    dtwc::algorithms::OneBatchPAMOptions options;
    options.n_clusters = 1;
    options.random_seed = 0;

    auto f64 = no_path_problem<double>("f13_onebatch_no_path_f64");
    require_invalid_input(
      [&] { (void)dtwc::algorithms::one_batch_pam(f64, options); },
      "Problem::dtw_function" + no_path);

    auto f32 = no_path_problem<float>("f13_onebatch_no_path_f32");
    require_invalid_input(
      [&] { (void)dtwc::algorithms::one_batch_pam(f32, options); },
      "Problem::dtw_function_f32" + no_path);
  }
}
