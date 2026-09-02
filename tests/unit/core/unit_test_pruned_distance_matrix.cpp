/**
 * @file unit_test_pruned_distance_matrix.cpp
 * @brief Unit tests for pruned distance matrix construction and lower bounds.
 *
 * @details Tests that the pruned distance matrix gives IDENTICAL results to
 * the unpruned version, that pruning statistics are valid, and that edge
 * cases are handled correctly.
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>
#include <core/lower_bounds.hpp>
#include <core/pruned_distance_matrix.hpp>
#include <detail/decode_pair.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <filesystem>
#include <system_error>

#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// Helper: create a Problem with synthetic data (no file I/O)
static Problem make_problem_with_data(
  std::vector<std::vector<double>> vecs,
  std::vector<std::string> names,
  int band_val = -1)
{
  Problem prob("test_pruned");
  prob.band = band_val;

  Data d(std::move(vecs), std::move(names));
  prob.set_data(std::move(d));

  return prob;
}

// Helper: load dummy data from the data/dummy/ folder
static Problem make_problem_from_dummy(int Ndata_max, int band_val = -1)
{
  std::filesystem::path data_dir = std::filesystem::path(DTWC_TEST_DATA_DIR) / "dummy";
  DataLoader dl{ data_dir, Ndata_max };
  dl.start_column(1).start_row(1);
  Problem prob("test_pruned_dummy", dl);
  prob.band = band_val;
  return prob;
}


// ======== Lower Bounds Unit Tests ========

TEST_CASE("LB_Kim basic properties", "[lower_bounds][lb_kim]")
{
  using namespace dtwc::core;

  SECTION("Identical series gives LB = 0")
  {
    std::vector<double> x{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    auto s = compute_summary(x);
    REQUIRE_THAT(lb_kim(s, s), WithinAbs(0.0, 1e-15));
  }

  SECTION("LB_Kim is a valid lower bound for full DTW")
  {
    std::vector<double> a{ 1.0, 3.0, 5.0, 2.0, 4.0 };
    std::vector<double> b{ 2.0, 4.0, 6.0, 3.0, 5.0 };
    auto sa = compute_summary(a);
    auto sb = compute_summary(b);

    double lb = lb_kim(sa, sb);
    double dtw_dist = dtwFull<double>(a, b);

    REQUIRE(lb <= dtw_dist + 1e-10);
    REQUIRE(lb >= 0.0);
  }

  SECTION("LB_Kim is non-negative")
  {
    std::vector<double> a{ -5.0, 0.0, 10.0 };
    std::vector<double> b{ -10.0, 5.0, 0.0 };
    auto sa = compute_summary(a);
    auto sb = compute_summary(b);

    REQUIRE(lb_kim(sa, sb) >= 0.0);
  }

  SECTION("Empty series")
  {
    std::vector<double> empty;
    auto s = compute_summary(empty);
    REQUIRE_THAT(lb_kim(s, s), WithinAbs(0.0, 1e-15));
  }
}

TEST_CASE("Envelope computation", "[lower_bounds][envelope]")
{
  using namespace dtwc::core;

  SECTION("Band = 0 means envelope equals the series itself")
  {
    std::vector<double> x{ 1.0, 5.0, 3.0, 7.0, 2.0 };
    auto env = compute_envelope(x, 0);

    REQUIRE(env.upper.size() == x.size());
    REQUIRE(env.lower.size() == x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      REQUIRE_THAT(env.upper[i], WithinAbs(x[i], 1e-15));
      REQUIRE_THAT(env.lower[i], WithinAbs(x[i], 1e-15));
    }
  }

  SECTION("Band = 1 looks at neighbors")
  {
    std::vector<double> x{ 1.0, 5.0, 3.0, 7.0, 2.0 };
    auto env = compute_envelope(x, 1);

    // i=0: window [0,1] -> upper=5, lower=1
    REQUIRE_THAT(env.upper[0], WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(env.lower[0], WithinAbs(1.0, 1e-15));

    // i=1: window [0,2] -> upper=5, lower=1
    REQUIRE_THAT(env.upper[1], WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(env.lower[1], WithinAbs(1.0, 1e-15));

    // i=2: window [1,3] -> upper=7, lower=3
    REQUIRE_THAT(env.upper[2], WithinAbs(7.0, 1e-15));
    REQUIRE_THAT(env.lower[2], WithinAbs(3.0, 1e-15));

    // i=4: window [3,4] -> upper=7, lower=2
    REQUIRE_THAT(env.upper[4], WithinAbs(7.0, 1e-15));
    REQUIRE_THAT(env.lower[4], WithinAbs(2.0, 1e-15));
  }

  SECTION("Large band covers entire series")
  {
    std::vector<double> x{ 1.0, 5.0, 3.0, 7.0, 2.0 };
    auto env = compute_envelope(x, 100);

    for (size_t i = 0; i < x.size(); ++i) {
      REQUIRE_THAT(env.upper[i], WithinAbs(7.0, 1e-15));
      REQUIRE_THAT(env.lower[i], WithinAbs(1.0, 1e-15));
    }
  }

  SECTION("Empty series")
  {
    std::vector<double> empty;
    auto env = compute_envelope(empty, 5);
    REQUIRE(env.upper.empty());
    REQUIRE(env.lower.empty());
  }
}

TEST_CASE("LB_Keogh basic properties", "[lower_bounds][lb_keogh]")
{
  using namespace dtwc::core;

  SECTION("Series inside envelope gives LB = 0")
  {
    std::vector<double> x{ 3.0, 4.0, 5.0, 4.0, 3.0 };
    auto env = compute_envelope(x, 2);

    // x is inside its own envelope
    REQUIRE_THAT(lb_keogh(x, env), WithinAbs(0.0, 1e-15));
  }

  SECTION("LB_Keogh is a valid lower bound for banded DTW")
  {
    std::vector<double> a{ 1.0, 3.0, 5.0, 2.0, 4.0 };
    std::vector<double> b{ 2.0, 4.0, 6.0, 3.0, 5.0 };
    int band = 1;

    auto env_b = compute_envelope(b, band);
    double lb = lb_keogh(a, env_b);
    double dtw_dist = dtwBanded<double>(a, b, band);

    REQUIRE(lb <= dtw_dist + 1e-10);
    REQUIRE(lb >= 0.0);
  }

  SECTION("LB_Keogh symmetric is tighter or equal")
  {
    std::vector<double> a{ 1.0, 3.0, 5.0, 2.0, 4.0 };
    std::vector<double> b{ 2.0, 4.0, 6.0, 3.0, 5.0 };
    int band = 1;

    auto env_a = compute_envelope(a, band);
    auto env_b = compute_envelope(b, band);

    double lb_ab = lb_keogh(a, env_b);
    double lb_ba = lb_keogh(b, env_a);
    double lb_sym = lb_keogh_symmetric(a, env_a, b, env_b);

    REQUIRE(lb_sym >= lb_ab - 1e-15);
    REQUIRE(lb_sym >= lb_ba - 1e-15);
    REQUIRE_THAT(lb_sym, WithinAbs(std::max(lb_ab, lb_ba), 1e-15));
  }

  SECTION("Mismatched sizes returns 0 (no pruning)")
  {
    std::vector<double> a{ 1.0, 2.0, 3.0 };
    std::vector<double> b{ 1.0, 2.0, 3.0, 4.0 };
    auto env_b = compute_envelope(b, 1);

    REQUIRE_THAT(lb_keogh(a, env_b), WithinAbs(0.0, 1e-15));
  }
}


// ======== Pruned Distance Matrix Tests (Problem-based) ========

TEST_CASE("Pruned distance matrix matches unpruned exactly - synthetic data",
          "[pruned_distance_matrix][correctness]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0 }
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e" };

  const int N = static_cast<int>(vecs.size());

  SECTION("Full DTW (band = -1)")
  {
    auto prob_ref = make_problem_with_data(vecs, names, -1);
    prob_ref.fill_distance_matrix();

    auto prob_pruned = make_problem_with_data(vecs, names, -1);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double ref_val = prob_ref.dist_by_ind(i, j);
        double pruned_val = prob_pruned.dist_by_ind(i, j);
        REQUIRE_THAT(pruned_val, WithinAbs(ref_val, 1e-10));
      }
    }
  }

  SECTION("Banded DTW (band = 1)")
  {
    auto prob_ref = make_problem_with_data(vecs, names, 1);
    prob_ref.fill_distance_matrix();

    auto prob_pruned = make_problem_with_data(vecs, names, 1);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, 1);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double ref_val = prob_ref.dist_by_ind(i, j);
        double pruned_val = prob_pruned.dist_by_ind(i, j);
        REQUIRE_THAT(pruned_val, WithinAbs(ref_val, 1e-10));
      }
    }
  }
}

TEST_CASE("Pruned distance matrix matches unpruned - dummy dataset",
          "[pruned_distance_matrix][correctness][dummy]")
{
  const int Ndata = 5;

  SECTION("Full DTW (band = -1)")
  {
    auto prob_ref = make_problem_from_dummy(Ndata, -1);
    prob_ref.fill_distance_matrix();

    auto prob_pruned = make_problem_from_dummy(Ndata, -1);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

    for (int i = 0; i < prob_ref.size(); ++i) {
      for (int j = 0; j < prob_ref.size(); ++j) {
        double ref_val = prob_ref.dist_by_ind(i, j);
        double pruned_val = prob_pruned.dist_by_ind(i, j);
        REQUIRE_THAT(pruned_val, WithinAbs(ref_val, 1e-10));
      }
    }

    // Check stats are consistent
    size_t expected_pairs = static_cast<size_t>(prob_ref.size()) * (prob_ref.size() - 1) / 2;
    REQUIRE(stats.total_pairs == expected_pairs);
    REQUIRE(stats.computed_full_dtw + stats.pruned_by_lb_kim + stats.pruned_by_lb_keogh == stats.total_pairs);
  }

  SECTION("Banded DTW (band = 2)")
  {
    auto prob_ref = make_problem_from_dummy(Ndata, 2);
    prob_ref.fill_distance_matrix();

    auto prob_pruned = make_problem_from_dummy(Ndata, 2);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, 2);

    for (int i = 0; i < prob_ref.size(); ++i) {
      for (int j = 0; j < prob_ref.size(); ++j) {
        double ref_val = prob_ref.dist_by_ind(i, j);
        double pruned_val = prob_pruned.dist_by_ind(i, j);
        REQUIRE_THAT(pruned_val, WithinAbs(ref_val, 1e-10));
      }
    }
  }
}

TEST_CASE("Pruned distance matrix with more dummy series",
          "[pruned_distance_matrix][correctness][dummy_more]")
{
  const int Ndata = 8;

  auto prob_ref = make_problem_from_dummy(Ndata, -1);
  prob_ref.fill_distance_matrix();

  auto prob_pruned = make_problem_from_dummy(Ndata, -1);
  auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

  for (int i = 0; i < prob_ref.size(); ++i) {
    for (int j = 0; j < prob_ref.size(); ++j) {
      double ref_val = prob_ref.dist_by_ind(i, j);
      double pruned_val = prob_pruned.dist_by_ind(i, j);
      REQUIRE_THAT(pruned_val, WithinAbs(ref_val, 1e-10));
    }
  }

  // Verify the stats add up
  size_t expected_pairs = static_cast<size_t>(prob_ref.size()) * (prob_ref.size() - 1) / 2;
  REQUIRE(stats.total_pairs == expected_pairs);
  REQUIRE(stats.computed_full_dtw + stats.pruned_by_lb_kim + stats.pruned_by_lb_keogh == expected_pairs);
}


// ======== Standalone Pruned Distance Matrix Tests ========

TEST_CASE("Standalone pruned matrix matches standard DTW - synthetic",
          "[pruned_distance_matrix][standalone][correctness]")
{
  std::vector<std::vector<double>> series = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0 }
  };
  const size_t N = series.size();

  SECTION("Full DTW (band = -1)")
  {
    // Reference: compute directly
    std::vector<double> ref(N * N, 0.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i + 1; j < N; ++j) {
        double d = dtwc::dtwFull_L<double>(series[i], series[j]);
        ref[i * N + j] = d;
        ref[j * N + i] = d;
      }

    // Pruned version
    std::vector<double> pruned(N * N, 0.0);
    auto stats = dtwc::core::compute_distance_matrix_pruned(
      series, pruned.data(), -1, dtwc::core::MetricType::L1);

    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        REQUIRE_THAT(pruned[i * N + j], WithinAbs(ref[i * N + j], 1e-10));

    // Stats consistency
    size_t expected = N * (N - 1) / 2;
    REQUIRE(stats.total_pairs == expected);
  }

  SECTION("Banded DTW (band = 1)")
  {
    const int band = 1;
    std::vector<double> ref(N * N, 0.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i + 1; j < N; ++j) {
        double d = dtwc::dtwBanded<double>(series[i], series[j], band);
        ref[i * N + j] = d;
        ref[j * N + i] = d;
      }

    std::vector<double> pruned(N * N, 0.0);
    auto stats = dtwc::core::compute_distance_matrix_pruned(
      series, pruned.data(), band, dtwc::core::MetricType::L1);

    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        REQUIRE_THAT(pruned[i * N + j], WithinAbs(ref[i * N + j], 1e-10));
  }

  SECTION("Banded DTW (band = 2)")
  {
    const int band = 2;
    std::vector<double> ref(N * N, 0.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i + 1; j < N; ++j) {
        double d = dtwc::dtwBanded<double>(series[i], series[j], band);
        ref[i * N + j] = d;
        ref[j * N + i] = d;
      }

    std::vector<double> pruned(N * N, 0.0);
    auto stats = dtwc::core::compute_distance_matrix_pruned(
      series, pruned.data(), band, dtwc::core::MetricType::L1);

    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        REQUIRE_THAT(pruned[i * N + j], WithinAbs(ref[i * N + j], 1e-10));
  }
}

TEST_CASE("Standalone pruned matrix - diverse series lengths",
          "[pruned_distance_matrix][standalone][variable_length]")
{
  // Variable-length series (LB_Keogh skipped for mismatched lengths)
  std::vector<std::vector<double>> series = {
    { 1.0, 2.0, 3.0 },
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 10.0, 20.0 },
    { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }
  };
  const size_t N = series.size();

  std::vector<double> ref(N * N, 0.0);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j) {
      double d = dtwc::dtwFull_L<double>(series[i], series[j]);
      ref[i * N + j] = d;
      ref[j * N + i] = d;
    }

  std::vector<double> pruned(N * N, 0.0);
  auto stats = dtwc::core::compute_distance_matrix_pruned(
    series, pruned.data(), -1, dtwc::core::MetricType::L1);

  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      REQUIRE_THAT(pruned[i * N + j], WithinAbs(ref[i * N + j], 1e-10));
}

TEST_CASE("Standalone pruned matrix - SquaredL2 metric (no LB pruning)",
          "[pruned_distance_matrix][standalone][sqeuclidean]")
{
  std::vector<std::vector<double>> series = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
  };
  const size_t N = series.size();

  std::vector<double> ref(N * N, 0.0);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j) {
      double d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                          dtwc::core::MetricType::SquaredL2);
      ref[i * N + j] = d;
      ref[j * N + i] = d;
    }

  std::vector<double> pruned(N * N, 0.0);
  auto stats = dtwc::core::compute_distance_matrix_pruned(
    series, pruned.data(), -1, dtwc::core::MetricType::SquaredL2);

  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      REQUIRE_THAT(pruned[i * N + j], WithinAbs(ref[i * N + j], 1e-10));

  // No LB pruning for SquaredL2
  REQUIRE(stats.pruned_by_lb_kim == 0);
  REQUIRE(stats.pruned_by_lb_keogh == 0);
  REQUIRE(stats.early_abandoned == 0);
}


// ======== Statistics Tests ========

TEST_CASE("Pruning ratio is valid", "[pruned_distance_matrix][stats]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0 }
  };
  std::vector<std::string> names = { "a", "b", "c", "d" };
  auto prob = make_problem_with_data(vecs, names, -1);
  auto stats = dtwc::core::fill_distance_matrix_pruned(prob, -1);

  REQUIRE(stats.pruning_ratio() >= 0.0);
  REQUIRE(stats.pruning_ratio() <= 1.0);
}

TEST_CASE("With band=-1, LB_Keogh pruning is skipped",
          "[pruned_distance_matrix][no_keogh]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0 }
  };
  std::vector<std::string> names = { "a", "b", "c", "d" };
  auto prob = make_problem_with_data(vecs, names, -1);
  auto stats = dtwc::core::fill_distance_matrix_pruned(prob, -1);

  REQUIRE(stats.pruned_by_lb_keogh == 0);
}

TEST_CASE("Statistics correctly computed",
          "[pruned_distance_matrix][stats_check]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0 }
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e" };
  auto prob = make_problem_with_data(vecs, names, 2);
  auto stats = dtwc::core::fill_distance_matrix_pruned(prob, 2);

  // Total pairs = N*(N-1)/2
  size_t expected = static_cast<size_t>(prob.size()) * (prob.size() - 1) / 2;
  REQUIRE(stats.total_pairs == expected);

  // Sum of categorized pairs equals total
  REQUIRE(stats.pruned_by_lb_kim + stats.pruned_by_lb_keogh + stats.computed_full_dtw == stats.total_pairs);

  // Full DTW count is valid
  REQUIRE(stats.computed_full_dtw >= 0);
  REQUIRE(stats.computed_full_dtw <= stats.total_pairs);
}

TEST_CASE("Empty problem", "[pruned_distance_matrix][edge]")
{
  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;
  auto prob = make_problem_with_data(vecs, names, -1);

  auto stats = dtwc::core::fill_distance_matrix_pruned(prob, -1);

  REQUIRE(stats.total_pairs == 0);
  REQUIRE(stats.computed_full_dtw == 0);
  REQUIRE(stats.pruned_by_lb_kim == 0);
  REQUIRE(stats.pruned_by_lb_keogh == 0);
  REQUIRE_THAT(stats.pruning_ratio(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("Single series problem", "[pruned_distance_matrix][edge]")
{
  std::vector<std::vector<double>> vecs = { { 1.0, 2.0, 3.0 } };
  std::vector<std::string> names = { "only" };
  auto prob = make_problem_with_data(vecs, names, -1);

  auto stats = dtwc::core::fill_distance_matrix_pruned(prob, -1);

  REQUIRE(stats.total_pairs == 0);
  REQUIRE(stats.computed_full_dtw == 0);
}

TEST_CASE("Standalone empty series list", "[pruned_distance_matrix][standalone][edge]")
{
  std::vector<std::vector<double>> series;
  std::vector<double> output;
  auto stats = dtwc::core::compute_distance_matrix_pruned(
    series, output.data(), -1, dtwc::core::MetricType::L1);
  REQUIRE(stats.total_pairs == 0);
}

TEST_CASE("Standalone single series", "[pruned_distance_matrix][standalone][edge]")
{
  std::vector<std::vector<double>> series = { { 1.0, 2.0, 3.0 } };
  std::vector<double> output(1, 999.0);  // should be set to 0
  auto stats = dtwc::core::compute_distance_matrix_pruned(
    series, output.data(), -1, dtwc::core::MetricType::L1);
  REQUIRE(stats.total_pairs == 0);
  REQUIRE_THAT(output[0], WithinAbs(0.0, 1e-15));
}

TEST_CASE("Pruned worker exceptions return to the caller with typed identity",
          "[pruned_distance_matrix][parallel][exception][m43]")
{
  constexpr auto expected =
    "Data::series: bulk time-series data is not resident locally (device='hpc'). "
    "Only shapes/counts/names are loaded on the client; the payload is streamed to "
    "the SLURM cluster at submit. Select device 'cpu' or 'gpu' for local data access.";

  auto run_metadata_failure = [] {
    std::vector<std::string> names;
    std::vector<size_t> sizes;
    for (size_t i = 0; i < 64; ++i) {
      names.push_back("series-" + std::to_string(i));
      sizes.push_back(8);
    }

    Problem prob("m43_metadata_failure");
    prob.set_data(Data::metadata_only(std::move(names), std::move(sizes), 1));
    try {
      (void)dtwc::core::fill_distance_matrix_pruned(prob, 2);
    } catch (const std::runtime_error &error) {
      REQUIRE_FALSE(prob.is_distance_matrix_filled());
      return std::string(error.what());
    } catch (...) {
      return std::string("<wrong exception type>");
    }
    return std::string("<no exception>");
  };

#ifdef _OPENMP
  const int previous_threads = omp_get_max_threads();
  const int previous_dynamic = omp_get_dynamic();
  struct RestoreOpenMP {
    int threads;
    int dynamic;
    ~RestoreOpenMP() { omp_set_dynamic(dynamic); omp_set_num_threads(threads); }
  } restore{previous_threads, previous_dynamic};

  omp_set_dynamic(0);
  omp_set_num_threads(1);
  const std::string serial_message = run_metadata_failure();
  omp_set_num_threads(2);
  const std::string parallel_message = run_metadata_failure();

  REQUIRE(serial_message == expected);
  REQUIRE(parallel_message == serial_message);
#else
  REQUIRE(run_metadata_failure() == expected);
#endif
}

TEST_CASE("Pruned pair blocks cover a non-divisible triangular range exactly",
          "[pruned_distance_matrix][parallel][partition][m43]")
{
  constexpr size_t N = 25;
  constexpr int band = 2;
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (size_t i = 0; i < N; ++i) {
    const double x = static_cast<double>(i);
    series.push_back({x, static_cast<double>((i * 7) % 13), x / 3.0, 24.0 - x});
    names.push_back("partition-" + std::to_string(i));
  }

#ifdef _OPENMP
  const int previous_threads = omp_get_max_threads();
  const int previous_dynamic = omp_get_dynamic();
  struct RestorePartitionOpenMP {
    int threads;
    int dynamic;
    ~RestorePartitionOpenMP() { omp_set_dynamic(dynamic); omp_set_num_threads(threads); }
  } restore{previous_threads, previous_dynamic};
  omp_set_dynamic(0);
  omp_set_num_threads(4); // 300 pairs / 32 blocks exposed the former tail overrun.
#endif

  auto prob = make_problem_with_data(series, names, band);
  const auto stats = dtwc::core::fill_distance_matrix_pruned(prob, band);

  REQUIRE(stats.total_pairs == N * (N - 1) / 2);
  REQUIRE(stats.computed_full_dtw + stats.pruned_by_lb_kim
          + stats.pruned_by_lb_keogh == stats.total_pairs);
  REQUIRE(stats.early_abandoned
          <= stats.pruned_by_lb_kim + stats.pruned_by_lb_keogh);
  for (size_t i = 0; i < N; ++i) {
    REQUIRE(prob.dist_by_ind(static_cast<int>(i), static_cast<int>(i)) == 0.0);
    for (size_t j = i + 1; j < N; ++j) {
      const double expected = dtwc::dtwBanded<double>(series[i], series[j], band);
      REQUIRE(prob.dist_by_ind(static_cast<int>(i), static_cast<int>(j)) == expected);
      REQUIRE(prob.dist_by_ind(static_cast<int>(j), static_cast<int>(i)) == expected);
    }
  }
}

TEST_CASE("Pruned thresholds preserve exact matrices across thread counts",
          "[pruned_distance_matrix][parallel][atomic][m44]")
{
  constexpr size_t N = 64;
  constexpr int band = 2;
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < N; ++i) {
    const double x = static_cast<double>(i);
    series.push_back({x / 5.0, static_cast<double>((i * 11) % 17),
                      7.0 - x / 9.0, static_cast<double>((i * i) % 23),
                      x / 3.0, 2.0 + static_cast<double>(i % 5)});
  }

  struct Run {
    std::vector<double> matrix;
    dtwc::core::PruningStats stats;
  };
  auto compute = [&](int threads) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    Run run{std::vector<double>(N * N, -1.0), {}};
    run.stats = dtwc::core::compute_distance_matrix_pruned(
      series, run.matrix.data(), band, dtwc::core::MetricType::L1);
    return run;
  };
  auto require_stats_contract = [](const dtwc::core::PruningStats &stats) {
    REQUIRE(stats.total_pairs == N * (N - 1) / 2);
    REQUIRE(stats.computed_full_dtw + stats.pruned_by_lb_kim
            + stats.pruned_by_lb_keogh == stats.total_pairs);
    REQUIRE(stats.early_abandoned
            <= stats.pruned_by_lb_kim + stats.pruned_by_lb_keogh);
  };

#ifdef _OPENMP
  const int previous_threads = omp_get_max_threads();
  const int previous_dynamic = omp_get_dynamic();
  struct RestoreAtomicOpenMP {
    int threads;
    int dynamic;
    ~RestoreAtomicOpenMP() { omp_set_dynamic(dynamic); omp_set_num_threads(threads); }
  } restore{previous_threads, previous_dynamic};
  omp_set_dynamic(0);
#else
  constexpr int previous_threads = 1;
#endif

  const Run serial = compute(1);
  const Run two_threads = compute(2);
  const Run max_threads = compute(std::max(1, previous_threads));
  require_stats_contract(serial.stats);
  require_stats_contract(two_threads.stats);
  require_stats_contract(max_threads.stats);
  REQUIRE(two_threads.matrix == serial.matrix);
  REQUIRE(max_threads.matrix == serial.matrix);

  for (size_t i = 0; i < N; ++i) {
    REQUIRE(serial.matrix[i * N + i] == 0.0);
    for (size_t j = i + 1; j < N; ++j) {
      const double expected = dtwc::dtwBanded<double>(series[i], series[j], band);
      REQUIRE(serial.matrix[i * N + j] == expected);
      REQUIRE(serial.matrix[j * N + i] == expected);
    }
  }
}

TEST_CASE("Pruned routing preserves configured missing-data semantics at N=63/64",
          "[pruned_distance_matrix][strategy][missing][m41]")
{
  {
    const std::vector<double> raw_a{
      0.0, 1.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 9.0};
    const std::vector<double> raw_b{0.25, 2.0, 1.0 / 3.0, 5.0, 8.875};
    const auto interpolated_a = dtwc::interpolate_linear(raw_a);
    const double raw = dtwc::dtwBanded<double>(raw_a, raw_b, 0);
    const double interpolated = dtwc::dtwBanded<double>(interpolated_a, raw_b, 0);
    INFO("raw=" << raw << " interpolated=" << interpolated);
    REQUIRE(raw != interpolated);
  }

  auto make_fixture = [](size_t n, bool with_missing) {
    std::vector<std::vector<double>> series;
    std::vector<std::string> names;
    for (size_t i = 0; i < n; ++i) {
      const double x = static_cast<double>(i);
      series.push_back({x / 4.0, 1.0 + static_cast<double>(i % 7),
                        x / 3.0, static_cast<double>((i * 5) % 11),
                        9.0 - x / 8.0});
      names.push_back("missing-route-" + std::to_string(i));
    }
    if (with_missing)
      series[0][2] = std::numeric_limits<double>::quiet_NaN();
    return std::pair{std::move(series), std::move(names)};
  };

  auto compute = [&](size_t n, bool with_missing,
                     dtwc::DistanceMatrixStrategy strategy,
                     dtwc::core::MissingStrategy missing) {
    auto [series, names] = make_fixture(n, with_missing);
    // band=0 forces the interior NaN onto the diagonal path. Interpolation
    // produces a finite midpoint; a raw Standard kernel cannot route around it.
    auto prob = make_problem_with_data(std::move(series), std::move(names), 0);
    prob.set_missing_strategy(missing);
    prob.set_distance_strategy(strategy);
    prob.fill_distance_matrix();
    REQUIRE(prob.is_distance_matrix_filled());
    std::vector<double> matrix(n * n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        matrix[i * n + j] = prob.dense_distance_matrix().get(i, j);
    return matrix;
  };

  for (const size_t n : {size_t{63}, size_t{64}}) {
    const auto reference = compute(n, true, dtwc::DistanceMatrixStrategy::BruteForce,
                                   dtwc::core::MissingStrategy::Interpolate);
    REQUIRE(std::isfinite(reference[1]));
    REQUIRE(reference[1] < std::numeric_limits<double>::max());
    const auto automatic = compute(n, true, dtwc::DistanceMatrixStrategy::Auto,
                                   dtwc::core::MissingStrategy::Interpolate);
    REQUIRE(automatic == reference);

    if (n == 64) {
      const auto explicit_pruned = compute(
        n, true, dtwc::DistanceMatrixStrategy::Pruned,
        dtwc::core::MissingStrategy::Interpolate);
      REQUIRE(explicit_pruned == reference);
    }
  }

  const auto finite_brute = compute(64, false, dtwc::DistanceMatrixStrategy::BruteForce,
                                    dtwc::core::MissingStrategy::Error);
  const auto finite_auto = compute(64, false, dtwc::DistanceMatrixStrategy::Auto,
                                   dtwc::core::MissingStrategy::Error);
  REQUIRE(finite_auto == finite_brute);

  auto [verbose_series, verbose_names] = make_fixture(64, true);
  auto verbose_prob = make_problem_with_data(
    std::move(verbose_series), std::move(verbose_names), 0);
  verbose_prob.set_missing_strategy(dtwc::core::MissingStrategy::Interpolate);
  verbose_prob.set_distance_strategy(dtwc::DistanceMatrixStrategy::Pruned);
  verbose_prob.set_verbose(true);
  std::ostringstream verbose_output;
  auto *previous_buffer = std::cout.rdbuf(verbose_output.rdbuf());
  struct RestoreCout {
    std::streambuf *buffer;
    ~RestoreCout() { std::cout.rdbuf(buffer); }
  } restore_cout{previous_buffer};
  verbose_prob.fill_distance_matrix();
  REQUIRE(verbose_output.str().find(
    "Pruned lower bounds support missing_strategy=Error only; using exact "
    "BruteForce to preserve the configured missing-data policy.")
    != std::string::npos);
}


// ======== Parallel Pruned + Strategy Integration Tests ========

TEST_CASE("fill_distance_matrix with Pruned strategy matches BruteForce exactly",
          "[pruned_distance_matrix][parallel][strategy]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0 }
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e" };
  const int N = static_cast<int>(vecs.size());

  SECTION("Default band (full DTW)")
  {
    auto prob_brute = make_problem_with_data(vecs, names, -1);
    prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob_brute.fill_distance_matrix();

    auto prob_pruned = make_problem_with_data(vecs, names, -1);
    prob_pruned.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
    prob_pruned.fill_distance_matrix();

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        REQUIRE_THAT(prob_pruned.dist_by_ind(i, j),
                     WithinAbs(prob_brute.dist_by_ind(i, j), 1e-10));
  }

  SECTION("Banded DTW (band = 2)")
  {
    auto prob_brute = make_problem_with_data(vecs, names, 2);
    prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob_brute.fill_distance_matrix();

    auto prob_pruned = make_problem_with_data(vecs, names, 2);
    prob_pruned.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
    prob_pruned.fill_distance_matrix();

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        REQUIRE_THAT(prob_pruned.dist_by_ind(i, j),
                     WithinAbs(prob_brute.dist_by_ind(i, j), 1e-10));
  }
}

TEST_CASE("Auto strategy selects Pruned for Standard DTW",
          "[pruned_distance_matrix][strategy][auto]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
  };
  std::vector<std::string> names = { "a", "b", "c" };
  const int N = static_cast<int>(vecs.size());

  // Auto with Standard DTW -> should use Pruned, results must be correct
  auto prob_auto = make_problem_with_data(vecs, names, -1);
  prob_auto.distance_strategy = dtwc::DistanceMatrixStrategy::Auto;
  prob_auto.fill_distance_matrix();

  // Reference using BruteForce
  auto prob_brute = make_problem_with_data(vecs, names, -1);
  prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_brute.fill_distance_matrix();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE_THAT(prob_auto.dist_by_ind(i, j),
                   WithinAbs(prob_brute.dist_by_ind(i, j), 1e-10));
}

TEST_CASE("Auto strategy falls back to BruteForce for non-Standard DTW",
          "[pruned_distance_matrix][strategy][auto_fallback]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0 },
  };
  std::vector<std::string> names = { "a", "b", "c" };
  const int N = static_cast<int>(vecs.size());

  // DDTW variant: Auto should fall back to BruteForce
  auto prob_ddtw_auto = make_problem_with_data(vecs, names, 2);
  prob_ddtw_auto.set_variant(dtwc::core::DTWVariant::DDTW);
  prob_ddtw_auto.distance_strategy = dtwc::DistanceMatrixStrategy::Auto;
  prob_ddtw_auto.fill_distance_matrix();

  auto prob_ddtw_brute = make_problem_with_data(vecs, names, 2);
  prob_ddtw_brute.set_variant(dtwc::core::DTWVariant::DDTW);
  prob_ddtw_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_ddtw_brute.fill_distance_matrix();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE_THAT(prob_ddtw_auto.dist_by_ind(i, j),
                   WithinAbs(prob_ddtw_brute.dist_by_ind(i, j), 1e-10));
}

TEST_CASE("Parallel pruned with larger dummy dataset matches brute-force",
          "[pruned_distance_matrix][parallel][dummy]")
{
  const int Ndata = 10;

  auto prob_brute = make_problem_from_dummy(Ndata, 3);
  prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_brute.fill_distance_matrix();

  auto prob_pruned = make_problem_from_dummy(Ndata, 3);
  prob_pruned.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
  prob_pruned.fill_distance_matrix();

  for (int i = 0; i < static_cast<int>(prob_brute.size()); ++i)
    for (int j = 0; j < static_cast<int>(prob_brute.size()); ++j)
      REQUIRE_THAT(prob_pruned.dist_by_ind(i, j),
                   WithinAbs(prob_brute.dist_by_ind(i, j), 1e-10));
}

TEST_CASE("DistanceMatrixStrategy enum values are distinct",
          "[pruned_distance_matrix][strategy][enum]")
{
  REQUIRE(dtwc::DistanceMatrixStrategy::Auto != dtwc::DistanceMatrixStrategy::BruteForce);
  REQUIRE(dtwc::DistanceMatrixStrategy::BruteForce != dtwc::DistanceMatrixStrategy::Pruned);
  REQUIRE(dtwc::DistanceMatrixStrategy::Pruned != dtwc::DistanceMatrixStrategy::CUDA);
}

// LB_Enhanced / LB_Webb strategies feed a tighter bound into the early-abandon
// path; the matrix must remain DIGIT-IDENTICAL to BruteForce (the bound only
// gates abandon; abandoned pairs are recomputed exactly). Task 5.2.
TEST_CASE("Pruned Enhanced/Webb lower-bound strategies match BruteForce exactly",
          "[pruned_distance_matrix][strategy][enhanced][webb]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0 },
    { 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0 },
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    { 10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 30.0 },
    { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0 }
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e", "f" };
  const int N = static_cast<int>(vecs.size());

  for (int band : { 1, 2, 3 }) {
    auto prob_brute = make_problem_with_data(vecs, names, band);
    prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob_brute.fill_distance_matrix();

    for (auto strat : { dtwc::LowerBoundStrategy::Enhanced,
                        dtwc::LowerBoundStrategy::Webb,
                        dtwc::LowerBoundStrategy::Keogh }) {
      auto prob = make_problem_with_data(vecs, names, band);
      prob.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
      prob.set_lb_strategy(strat);
      prob.fill_distance_matrix();

      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          INFO("band=" << band << " strat=" << static_cast<int>(strat)
               << " (" << i << "," << j << ")");
          REQUIRE_THAT(prob.dist_by_ind(i, j),
                       WithinAbs(prob_brute.dist_by_ind(i, j), 1e-10));
        }
    }
  }
}

// =========================================================================
//  A7: the pruned fill must honour entries that are already present.
//
//  fill_distance_matrix_pruned() called dm.resize(N) unconditionally, and
//  DenseDistanceMatrix::resize() re-fills every packed slot with NaN — so a
//  restored checkpoint was discarded the moment the fill started (Auto
//  resolves to Pruned for Standard DTW). The brute-force fill already skips
//  computed pairs; the pruned fill now does the same.
// =========================================================================

TEST_CASE("Pruned fill preserves pre-populated distance-matrix entries",
          "[pruned_distance_matrix][checkpoint][regression]")
{
  auto prob = make_problem_with_data(
    { { 1, 2, 3, 4, 5 }, { 5, 4, 3, 2, 1 }, { 1, 1, 1, 1, 1 }, { 2, 4, 6, 8, 10 } },
    { "a", "b", "c", "d" }, -1);

  auto &dm = std::get<core::DenseDistanceMatrix>(prob.distance_matrix());
  dm.resize(prob.size());

  // A sentinel no DTW of this data can produce, standing in for a restored
  // checkpoint entry. It must survive the fill untouched.
  constexpr double sentinel = 12345.0;
  dm.set(0, 1, sentinel);

  const auto stats = core::fill_distance_matrix_pruned(prob, -1,
                                                       LowerBoundStrategy::Auto);
  CAPTURE(stats.total_pairs);

  REQUIRE(dm.get(0, 1) == sentinel);
  // Every other pair is still filled exactly.
  for (size_t i = 0; i < prob.size(); ++i)
    for (size_t j = i; j < prob.size(); ++j)
      REQUIRE(dm.is_computed(i, j));
  REQUIRE_THAT(dm.get(0, 2), WithinAbs(dtwFull_L<double>(prob.series(0), prob.series(2)), 1e-12));
}

// =========================================================================
//  A12 follow-up (audit 2026-09-02, item 2). The pruned fill is f64-only: its
//  summaries, envelopes and kernels all read Problem::series(), which rejects
//  a Float32 store. Float32 + Standard DTW + band >= 0 + N >= 64 + dense used
//  to resolve Auto -> Pruned and then raise Data::series' precision error from
//  inside the parallel region. Auto must route f32 to the exact dense fill,
//  and an explicit Pruned request must be refused by name before any worker
//  starts.
// =========================================================================

namespace {

/// N deterministic length-16 series with distinct shapes, exactly
/// representable in neither precision so the f32/f64 comparison is real.
std::vector<std::vector<double>> f32_gate_series(size_t n)
{
  std::vector<std::vector<double>> vecs;
  vecs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<double> v(16);
    for (size_t t = 0; t < v.size(); ++t)
      v[t] = std::sin(0.25 * static_cast<double>(t) + 0.125 * static_cast<double>(i))
             * (1.0 + 0.03125 * static_cast<double>(i));
    vecs.push_back(std::move(v));
  }
  return vecs;
}

std::vector<std::string> f32_gate_names(size_t n)
{
  std::vector<std::string> names;
  names.reserve(n);
  for (size_t i = 0; i < n; ++i) names.push_back("s" + std::to_string(i));
  return names;
}

Problem make_f32_problem(const std::vector<std::vector<double>> &f64_series, int band_val)
{
  std::vector<std::vector<float>> f32_series;
  f32_series.reserve(f64_series.size());
  for (const auto &v : f64_series) {
    std::vector<float> converted;
    converted.reserve(v.size());
    for (const double x : v) converted.push_back(static_cast<float>(x));
    f32_series.push_back(std::move(converted));
  }
  Problem prob("pruned_f32_gate");
  prob.band = band_val;
  prob.set_data(Data(std::move(f32_series), f32_gate_names(f64_series.size())));
  return prob;
}

} // namespace

TEST_CASE("Float32 data never reaches the f64-only pruned fill",
          "[pruned_distance_matrix][strategy][f32][regression]")
{
  constexpr size_t N = 64; // pruned_strategy_applicable's size floor
  constexpr int band = 4;  // ... and its band >= 0 requirement
  const auto f64_series = f32_gate_series(N);

  SECTION("Auto falls back to the exact dense fill and matches the f64 matrix")
  {
    auto prob_f32 = make_f32_problem(f64_series, band);
    prob_f32.distance_strategy = dtwc::DistanceMatrixStrategy::Auto;
    REQUIRE_NOTHROW(prob_f32.fill_distance_matrix()); // threw Data::series unfixed
    REQUIRE(prob_f32.is_distance_matrix_filled());

    auto prob_f64 = make_problem_with_data(f64_series, f32_gate_names(N), band);
    prob_f64.distance_strategy = dtwc::DistanceMatrixStrategy::Auto;
    prob_f64.fill_distance_matrix();

    for (int i = 0; i < static_cast<int>(N); ++i)
      for (int j = 0; j < static_cast<int>(N); ++j)
        REQUIRE_THAT(prob_f32.dist_by_ind(i, j),
                     WithinAbs(prob_f64.dist_by_ind(i, j), 1e-4));
  }

  SECTION("An explicit Pruned request is a typed InvalidInput, not a worker throw")
  {
    auto prob_f32 = make_f32_problem(f64_series, band);
    prob_f32.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
    REQUIRE_THROWS_AS(prob_f32.fill_distance_matrix(), dtwc::InvalidInput);
    REQUIRE_FALSE(prob_f32.is_distance_matrix_filled());
  }

  SECTION("Explicit Pruned + f32 + mmap distance storage still fills")
  {
    // Audit 2026-09-02, D3: the f32 guard was placed ABOVE the
    // `Pruned && has_mmap_storage -> BruteForce` downgrade, so this
    // combination started throwing even though the downgrade routes it to the
    // exact generic row fill, which handles f32 correctly.
#ifndef DTWC_HAS_MMAP
    SKIP("mmap distance storage not compiled in (llfio)");
#else
    const auto scratch = std::filesystem::temp_directory_path() / "dtwc_pruned_f32_mmap";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);

    auto prob_f32 = make_f32_problem(f64_series, band);
    prob_f32.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
    prob_f32.use_mmap_distance_matrix(scratch / "pruned-f32.dtwm");
    REQUIRE_NOTHROW(prob_f32.fill_distance_matrix());
    REQUIRE(prob_f32.is_distance_matrix_filled());

    auto prob_f64 = make_problem_with_data(f64_series, f32_gate_names(N), band);
    prob_f64.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob_f64.fill_distance_matrix();
    for (int i = 0; i < static_cast<int>(N); ++i)
      for (int j = 0; j < static_cast<int>(N); ++j)
        REQUIRE_THAT(prob_f32.dist_by_ind(i, j),
                     WithinAbs(prob_f64.dist_by_ind(i, j), 1e-4));

    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
#endif
  }

  SECTION("Non-vacuity: the pruned builder itself still cannot take f32 data")
  {
    // Without the routing guard above, Auto would reach exactly this call and
    // surface a Data::series precision std::runtime_error from inside the
    // parallel region. If this ever stops throwing, the two sections above are
    // no longer testing anything and must be revisited.
    auto prob_f32 = make_f32_problem(f64_series, band);
    REQUIRE_THROWS_AS(dtwc::core::fill_distance_matrix_pruned(prob_f32, band),
                      std::runtime_error);
    REQUIRE(prob_f32.data().is_f32());
  }
}

// =========================================================================
//  A7 companion (audit 2026-09-02, item 10). The sentinel test above calls
//  fill_distance_matrix_pruned() directly at N = 4 / band = -1, which
//  pruned_strategy_applicable would never route to Pruned. This case builds
//  the configuration Auto actually resolves to Pruned for — Standard DTW,
//  dense storage, MissingStrategy::Error, band >= 0, N >= 64 — and drives it
//  through the public Problem::fill_distance_matrix().
// =========================================================================
TEST_CASE("Auto-resolved Pruned fill preserves pre-populated entries at N=64",
          "[pruned_distance_matrix][checkpoint][strategy][auto][regression]")
{
  constexpr size_t N = 64;
  constexpr int band = 4;
  const auto series = f32_gate_series(N);

  auto reference = make_problem_with_data(series, f32_gate_names(N), band);
  reference.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  reference.fill_distance_matrix();

  auto prob = make_problem_with_data(series, f32_gate_names(N), band);
  prob.distance_strategy = dtwc::DistanceMatrixStrategy::Auto;
  REQUIRE(prob.variant_params.variant == dtwc::core::DTWVariant::Standard);
  REQUIRE(prob.missing_strategy == dtwc::core::MissingStrategy::Error);

  auto &dm = std::get<core::DenseDistanceMatrix>(prob.distance_matrix());
  dm.resize(prob.size());
  constexpr double sentinel = 12345.0; // no DTW of this data can produce it
  dm.set(0, 1, sentinel);

  prob.fill_distance_matrix();

  REQUIRE(dm.get(0, 1) == sentinel);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      REQUIRE(dm.is_computed(i, j));
  for (int i = 0; i < static_cast<int>(N); ++i)
    for (int j = 0; j < static_cast<int>(N); ++j)
      if (!(i == 0 && j == 1) && !(i == 1 && j == 0))
        REQUIRE_THAT(prob.dist_by_ind(i, j),
                     WithinAbs(reference.dist_by_ind(i, j), 1e-10));
}

// =========================================================================
//  decode_pair SSOT: the pruned fill decoded pair indices with its own
//  single-`if` correction — the exact form decode_pair.hpp documents as
//  retired and broken. It now calls the SSOT.
//
//  The SSOT's host decode corrected the row in ONE direction only, unlike its
//  MSL sibling. An FP64 seed cannot be forced to overestimate at any reachable
//  N (every intermediate is exact below ~5.7e7), so the overestimate is
//  exercised on a replica of the algorithm with the seed deliberately raised
//  by one — the same white-box technique the MSL parity test uses.
// =========================================================================

namespace {

std::int64_t encode_pair_index(std::int64_t i, std::int64_t j, std::int64_t N)
{
  return i * (2 * N - i - 1) / 2 + (j - i - 1);
}

/// Replica of dtwc::detail::decode_pair with a caller-supplied seed bias, so a
/// seed that OVERESTIMATES the row can be constructed on purpose.
/// @param two_way  true  -> the shipped algorithm (correct down, then up);
///                 false -> the retired up-only form.
void decode_pair_with_seed_bias(std::int64_t k, std::int64_t N, std::int64_t bias,
                                bool two_way, std::int64_t &i, std::int64_t &j)
{
  const double Nd = static_cast<double>(N);
  const double kd = static_cast<double>(k);
  std::int64_t row = static_cast<std::int64_t>(
      std::floor(Nd - 0.5 - std::sqrt((Nd - 0.5) * (Nd - 0.5) - 2.0 * kd))) + bias;
  if (row > N - 2) row = N - 2;
  if (row < 0) row = 0;
  std::int64_t row_start = row * (2 * N - row - 1) / 2;
  if (two_way)
    while (row > 0 && row_start > k) {                          // correct down
      --row;
      row_start = row * (2 * N - row - 1) / 2;
    }
  while (row + 1 < N && row_start + (N - row - 1) <= k) {       // correct up
    row_start += (N - row - 1);
    ++row;
  }
  i = row;
  j = row + 1 + (k - row_start);
}

} // namespace

// The host decode's FP64 seed is exact for every reachable N (all intermediates
// stay below 2^53), so an overestimate cannot be produced through the public
// signature — the attempt is recorded here rather than claimed. What IS
// testable, white-box, is the correction algorithm the two decoders share: the
// MSL sibling seeds from a FLOORED integer isqrt, which genuinely overshoots,
// and the host copy corrected upward only. This pins that a down-correction is
// load-bearing, and that the shipped decode agrees with the two-way form.
TEST_CASE("decode_pair recovers from a row seed that is too high",
          "[pruned_distance_matrix][decode_pair][regression]")
{
  bool up_only_ever_wrong = false;

  for (const std::int64_t N : { std::int64_t(4), std::int64_t(63), std::int64_t(64),
                                std::int64_t(1000) }) {
    const std::int64_t num_pairs = N * (N - 1) / 2;
    const std::int64_t stride = std::max<std::int64_t>(1, num_pairs / 500);
    for (std::int64_t k = 0; k < num_pairs; k += stride) {
      std::int64_t ei = -1, ej = -1;
      dtwc::detail::decode_pair(k, N, ei, ej);
      REQUIRE(encode_pair_index(ei, ej, N) == k);

      for (const std::int64_t bias : { std::int64_t(1), std::int64_t(2) }) {
        std::int64_t bi = -1, bj = -1;
        decode_pair_with_seed_bias(k, N, bias, true, bi, bj);
        CAPTURE(N, k, bias, ei, ej, bi, bj);
        REQUIRE(bi == ei);   // two-way correction always recovers
        REQUIRE(bj == ej);

        std::int64_t ui = -1, uj = -1;
        decode_pair_with_seed_bias(k, N, bias, false, ui, uj);
        if (ui != ei || uj != ej) up_only_ever_wrong = true;
      }
    }
  }

  // The retired up-only form must be demonstrably wrong somewhere, otherwise
  // this test would pass for the wrong reason.
  CHECK(up_only_ever_wrong);
}
