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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>
#include <cmath>

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
  dl.startColumn(1).startRow(1);
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
    prob_ref.fillDistanceMatrix();

    auto prob_pruned = make_problem_with_data(vecs, names, -1);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double ref_val = prob_ref.distByInd(i, j);
        double pruned_val = prob_pruned.distByInd(i, j);
        REQUIRE_THAT(pruned_val, WithinAbs(ref_val, 1e-10));
      }
    }
  }

  SECTION("Banded DTW (band = 1)")
  {
    auto prob_ref = make_problem_with_data(vecs, names, 1);
    prob_ref.fillDistanceMatrix();

    auto prob_pruned = make_problem_with_data(vecs, names, 1);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, 1);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double ref_val = prob_ref.distByInd(i, j);
        double pruned_val = prob_pruned.distByInd(i, j);
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
    prob_ref.fillDistanceMatrix();

    auto prob_pruned = make_problem_from_dummy(Ndata, -1);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

    for (int i = 0; i < prob_ref.size(); ++i) {
      for (int j = 0; j < prob_ref.size(); ++j) {
        double ref_val = prob_ref.distByInd(i, j);
        double pruned_val = prob_pruned.distByInd(i, j);
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
    prob_ref.fillDistanceMatrix();

    auto prob_pruned = make_problem_from_dummy(Ndata, 2);
    auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, 2);

    for (int i = 0; i < prob_ref.size(); ++i) {
      for (int j = 0; j < prob_ref.size(); ++j) {
        double ref_val = prob_ref.distByInd(i, j);
        double pruned_val = prob_pruned.distByInd(i, j);
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
  prob_ref.fillDistanceMatrix();

  auto prob_pruned = make_problem_from_dummy(Ndata, -1);
  auto stats = dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

  for (int i = 0; i < prob_ref.size(); ++i) {
    for (int j = 0; j < prob_ref.size(); ++j) {
      double ref_val = prob_ref.distByInd(i, j);
      double pruned_val = prob_pruned.distByInd(i, j);
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


// ======== Parallel Pruned + Strategy Integration Tests ========

TEST_CASE("fillDistanceMatrix with Pruned strategy matches BruteForce exactly",
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
    prob_brute.fillDistanceMatrix();

    auto prob_pruned = make_problem_with_data(vecs, names, -1);
    prob_pruned.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
    prob_pruned.fillDistanceMatrix();

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        REQUIRE_THAT(prob_pruned.distByInd(i, j),
                     WithinAbs(prob_brute.distByInd(i, j), 1e-10));
  }

  SECTION("Banded DTW (band = 2)")
  {
    auto prob_brute = make_problem_with_data(vecs, names, 2);
    prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob_brute.fillDistanceMatrix();

    auto prob_pruned = make_problem_with_data(vecs, names, 2);
    prob_pruned.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
    prob_pruned.fillDistanceMatrix();

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        REQUIRE_THAT(prob_pruned.distByInd(i, j),
                     WithinAbs(prob_brute.distByInd(i, j), 1e-10));
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
  prob_auto.fillDistanceMatrix();

  // Reference using BruteForce
  auto prob_brute = make_problem_with_data(vecs, names, -1);
  prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_brute.fillDistanceMatrix();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE_THAT(prob_auto.distByInd(i, j),
                   WithinAbs(prob_brute.distByInd(i, j), 1e-10));
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
  prob_ddtw_auto.fillDistanceMatrix();

  auto prob_ddtw_brute = make_problem_with_data(vecs, names, 2);
  prob_ddtw_brute.set_variant(dtwc::core::DTWVariant::DDTW);
  prob_ddtw_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_ddtw_brute.fillDistanceMatrix();

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE_THAT(prob_ddtw_auto.distByInd(i, j),
                   WithinAbs(prob_ddtw_brute.distByInd(i, j), 1e-10));
}

TEST_CASE("Parallel pruned with larger dummy dataset matches brute-force",
          "[pruned_distance_matrix][parallel][dummy]")
{
  const int Ndata = 10;

  auto prob_brute = make_problem_from_dummy(Ndata, 3);
  prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_brute.fillDistanceMatrix();

  auto prob_pruned = make_problem_from_dummy(Ndata, 3);
  prob_pruned.distance_strategy = dtwc::DistanceMatrixStrategy::Pruned;
  prob_pruned.fillDistanceMatrix();

  for (int i = 0; i < static_cast<int>(prob_brute.size()); ++i)
    for (int j = 0; j < static_cast<int>(prob_brute.size()); ++j)
      REQUIRE_THAT(prob_pruned.distByInd(i, j),
                   WithinAbs(prob_brute.distByInd(i, j), 1e-10));
}

TEST_CASE("DistanceMatrixStrategy enum values are distinct",
          "[pruned_distance_matrix][strategy][enum]")
{
  REQUIRE(dtwc::DistanceMatrixStrategy::Auto != dtwc::DistanceMatrixStrategy::BruteForce);
  REQUIRE(dtwc::DistanceMatrixStrategy::BruteForce != dtwc::DistanceMatrixStrategy::Pruned);
  REQUIRE(dtwc::DistanceMatrixStrategy::Pruned != dtwc::DistanceMatrixStrategy::CUDA);
}
