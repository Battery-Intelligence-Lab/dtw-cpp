/**
 * @file test_problem_metric.cpp
 * @brief IF-2 S2: the pointwise metric is part of a Problem's distance
 *        semantics (Problem::set_metric), and FastCLARA's samples inherit it.
 *
 * @details Oracles are the checked free functions (distance::dtw), never the
 *          Problem under test: exact (==) where the same kernel runs, else within
 *          kCrossPathRel (the fill may take the SIMD lanes, which round differently).
 *          The fill-versus-oracle distance table is core/test_dtw.cpp. The Metal
 *          cases live in test_metal_mmap.cpp and the CUDA case in
 *          test_cuda_correctness.cpp, which may skip without a device.
 */

#include <dtwc.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/tadpole.hpp>
#include <base/error.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using dtwc::core::MetricType;
namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

// The Problem's fill and the checked free function reach the same squared-L2
// recurrence through different call paths; MSVC /fp:contract may fuse a
// multiply-add in one and not the other, which moves the last bit (observed:
// 1 ulp). 1e-14 relative is about 45 ulp: exact in intent, blind to contraction.
constexpr double kCrossPathRel = 1e-14;

namespace {

/// Interleaved series of lengths base_len, base_len + 1, base_len + 2, ...
std::vector<std::vector<double>> random_series(std::size_t n, std::size_t base_len,
                                               std::size_t ndim, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::vector<std::vector<double>> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i].resize((base_len + i % 3) * ndim);
    for (auto &v : out[i]) v = value(rng);
  }
  return out;
}

std::vector<std::string> names_for(std::size_t n)
{
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n; ++i) names.push_back("s" + std::to_string(i));
  return names;
}

dtwc::Problem make_problem(const std::vector<std::vector<double>> &series,
                           std::size_t ndim = 1)
{
  dtwc::Problem prob("metric");
  prob.set_data(dtwc::Data{ std::vector<std::vector<double>>(series),
                            names_for(series.size()), ndim });
  return prob;
}

} // namespace

TEST_CASE("set_metric: a metric the kernels cannot take is refused before any pair",
          "[problem][metric][errors][if2]")
{
  const auto series = random_series(4, 8, 1, 3);

  SECTION("another variant, or a missing-data strategy, keeps the Problem unchanged")
  {
    for (const auto variant : { dtwc::core::DTWVariant::DDTW, dtwc::core::DTWVariant::WDTW,
                                dtwc::core::DTWVariant::ADTW, dtwc::core::DTWVariant::SoftDTW,
                                dtwc::core::DTWVariant::MSM, dtwc::core::DTWVariant::TWE }) {
      CAPTURE(static_cast<int>(variant));
      auto prob = make_problem(series);
      prob.set_variant(variant);
      prob.fill_distance_matrix();
      REQUIRE_THROWS_MATCHES(prob.set_metric(MetricType::SquaredL2), dtwc::InvalidInput,
                             Catch::Matchers::MessageMatches(ContainsSubstring(
                               "metric SquaredL2 is implemented for Standard DTW")));
      CHECK(prob.metric() == MetricType::L1);
      CHECK(prob.is_distance_matrix_filled());

      auto squared = make_problem(series);
      squared.set_metric(MetricType::SquaredL2);
      REQUIRE_THROWS_AS(squared.set_variant(variant), dtwc::InvalidInput);
      CHECK(squared.variant_params().variant == dtwc::core::DTWVariant::Standard);
    }
    for (const auto missing : { dtwc::core::MissingStrategy::ZeroCost,
                                dtwc::core::MissingStrategy::AROW,
                                dtwc::core::MissingStrategy::Interpolate }) {
      CAPTURE(static_cast<int>(missing));
      auto squared = make_problem(series);
      squared.set_metric(MetricType::SquaredL2);
      REQUIRE_THROWS_AS(squared.set_missing_strategy(missing), dtwc::InvalidInput);
      CHECK(squared.missing_strategy() == dtwc::core::MissingStrategy::Error);
    }
  }
}

TEST_CASE("set_metric: the metric is part of refresh and of the checkpoint identity",
          "[problem][metric][checkpoint][if2]")
{
  const auto series = random_series(5, 9, 1, 13);
  auto squared = make_problem(series);
  squared.set_metric(MetricType::SquaredL2);
  squared.fill_distance_matrix();
  REQUIRE(squared.is_distance_matrix_filled());

  CHECK(squared.distance_checkpoint_identity()
        == squared.distance_checkpoint_identity(MetricType::SquaredL2));
  CHECK(squared.distance_checkpoint_identity()
        != squared.distance_checkpoint_identity(MetricType::L1));

  ScratchDirectory scratch("if2_metric_checkpoint");
  const std::string dir = (scratch.path / "ckpt").string();
  dtwc::save_checkpoint(squared, dir); // tagged with squared.metric()

  auto l1 = make_problem(series);
  CHECK_THROWS_AS(dtwc::load_checkpoint(l1, dir), dtwc::InvalidInput); // another metric's matrix
  // The three-argument form keeps its explicit tag: the CLI's own CUDA fill
  // (squared L2 into an L1 Problem) relies on it until IF-2 S3.
  CHECK(dtwc::load_checkpoint(l1, dir, MetricType::SquaredL2));

  auto resumed = make_problem(series);
  resumed.set_metric(MetricType::SquaredL2);
  REQUIRE(dtwc::load_checkpoint(resumed, dir));
  REQUIRE(resumed.is_distance_matrix_filled());
  CHECK(resumed.dist_by_ind(0, 1) == squared.dist_by_ind(0, 1));

  // A new metric is new semantics: the filled matrix is released.
  squared.set_metric(MetricType::L1);
  CHECK_FALSE(squared.is_distance_matrix_filled());
  squared.fill_distance_matrix();
  CHECK(squared.dist_by_ind(0, 1)
        == dtwc::distance::dtw<double>(series[0], series[1]));
}

TEST_CASE("set_metric: automatic checkpoints are tagged with the metric",
          "[problem][metric][checkpoint][if2]")
{
  const auto series = random_series(5, 9, 1, 17);
  ScratchDirectory scratch("if2_metric_autosave");
  auto prob = make_problem(series);
  prob.set_metric(MetricType::SquaredL2);
  prob.checkpoint.directory = (scratch.path / "auto").string();
  prob.checkpoint.save_interval = 2;
  prob.checkpoint.enabled = true;
  prob.fill_distance_matrix();

  auto l1 = make_problem(series);
  CHECK_THROWS_AS(dtwc::load_checkpoint(l1, prob.checkpoint.directory), dtwc::InvalidInput);
  auto resumed = make_problem(series);
  resumed.set_metric(MetricType::SquaredL2);
  REQUIRE(dtwc::load_checkpoint(resumed, prob.checkpoint.directory));
  CHECK_THAT(resumed.dist_by_ind(1, 2),
             WithinRel(dtwc::distance::dtw<double>(series[1], series[2], -1,
                                                   MetricType::SquaredL2),
                       kCrossPathRel));
}

#ifdef DTWC_HAS_MMAP
TEST_CASE("set_metric: an mmap cache takes the metric, and the CPU fills it",
          "[problem][metric][mmap][if2]")
{
  const auto series = random_series(6, 30, 1, 7);
  ScratchDirectory scratch("if2_metric_mmap");
  for (const int band : { -1, 3 }) {
    CAPTURE(band);
    const auto cache = scratch.path / ("sq_" + std::to_string(band) + ".cache");
    {
      // The two-argument form is set_metric + bind. On the synced base the
      // CPU fill refused it ("a non-L1 cache is external-fill-only").
      auto prob = make_problem(series);
      prob.set_band(band);
      prob.use_mmap_distance_matrix(cache, MetricType::SquaredL2);
      CHECK(prob.metric() == MetricType::SquaredL2);
      prob.fill_distance_matrix();
      for (std::size_t i = 0; i < series.size(); ++i)
        for (std::size_t j = i + 1; j < series.size(); ++j)
          CHECK_THAT(prob.dist_by_ind(int(i), int(j)),
                     WithinRel(dtwc::distance::dtw<double>(
                                 series[i], series[j], band, MetricType::SquaredL2),
                               kCrossPathRel));
    }
    // The one-argument form binds the Problem's metric: a squared-L2 Problem
    // reopens the filled cache, an L1 one does not match it.
    {
      auto reopened = make_problem(series);
      reopened.set_band(band);
      reopened.set_metric(MetricType::SquaredL2);
      reopened.use_mmap_distance_matrix(cache);
      CHECK(reopened.is_distance_matrix_filled());
    } // releases the cache's session lease

    auto l1 = make_problem(series);
    l1.set_band(band);
    CHECK_THROWS_WITH(l1.use_mmap_distance_matrix(cache),
                      ContainsSubstring("fingerprint mismatch"));
  }
}

TEST_CASE("use_mmap_distance_matrix(path, metric) changes nothing when the bind fails",
          "[problem][metric][mmap][if2]")
{
  const auto series = random_series(5, 9, 1, 19);
  ScratchDirectory scratch("if2_metric_bind");
  const auto cache = scratch.path / "l1.cache";
  {
    auto writer = make_problem(series);
    writer.use_mmap_distance_matrix(cache); // an L1 cache
  }
  auto prob = make_problem(series);
  prob.fill_distance_matrix();
  const double before = prob.dist_by_ind(0, 1);
  // The L1 file does not match a squared-L2 identity, so the bind throws.
  CHECK_THROWS_WITH(prob.use_mmap_distance_matrix(cache, MetricType::SquaredL2),
                    ContainsSubstring("fingerprint mismatch"));
  CHECK(prob.metric() == MetricType::L1);
  CHECK(prob.is_distance_matrix_filled());
  CHECK(prob.dist_by_ind(0, 1) == before);
}
#endif

TEST_CASE("FastCLARA samples compute with the parent's metric",
          "[fast_clara][metric][if2]")
{
  // Constant series: DTW is L * |a - b| (L1) or L * (a - b)^2 (squared L2), so
  // the 1-medoid of any 11-of-12 sample is the median (5 or 6) under L1 and
  // the value nearest the mean (8 or 9) under squared L2.
  const std::vector<double> values{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41 };
  std::vector<std::vector<double>> series;
  for (const double v : values) series.push_back(std::vector<double>(3, v));
  auto prob = make_problem(series);
  prob.set_metric(MetricType::SquaredL2);

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 1;
  opts.sample_size = 11;
  opts.n_samples = 1;
  const auto result = dtwc::algorithms::fast_clara(prob, opts);
  REQUIRE(result.medoid_indices.size() == 1);
  const int medoid = result.medoid_indices[0];
  CHECK((values[medoid] == 8 || values[medoid] == 9));

  double expected_cost = 0.0;
  for (const auto &s : series)
    expected_cost += dtwc::distance::dtw<double>(s, series[medoid], -1,
                                                 MetricType::SquaredL2);
  CHECK_THAT(result.total_cost, WithinRel(expected_cost, 1e-12));
}

TEST_CASE("TADPole under a squared-L2 metric takes the exact path",
          "[tadpole][metric][if2]")
{
  // TADPole's LB_Keogh and diagonal upper bound are L1. Under squared L2 with
  // |differences| < 1 the lower bound exceeds the exact distance, so pruning
  // would drop true neighbours; prune=false is the brute-force oracle.
  std::mt19937 rng(41);
  std::uniform_real_distribution<double> value(-0.5, 0.5);
  std::vector<std::vector<double>> series(30, std::vector<double>(16));
  for (auto &s : series)
    for (auto &v : s) v = value(rng);

  auto pruned = make_problem(series);
  pruned.set_metric(MetricType::SquaredL2);
  const double dc = dtwc::algorithms::tadpole_auto_dc(pruned);
  dtwc::algorithms::TADPoleStats stats;
  const auto fast = dtwc::algorithms::tadpole(pruned, 3, dc, true, &stats);
  CHECK_FALSE(stats.pruning_enabled);

  auto exact = make_problem(series);
  exact.set_metric(MetricType::SquaredL2);
  const auto brute = dtwc::algorithms::tadpole(exact, 3, dc, false);
  CHECK(fast.labels == brute.labels);
  CHECK(fast.medoid_indices == brute.medoid_indices);

  // Control: the L1 default still prunes.
  auto l1 = make_problem(series);
  dtwc::algorithms::TADPoleStats l1_stats;
  (void)dtwc::algorithms::tadpole(l1, 3, dtwc::algorithms::tadpole_auto_dc(l1), true,
                                  &l1_stats);
  CHECK(l1_stats.pruning_enabled);
}

TEST_CASE("Problem's default output folder is ./results/", "[problem][output][if2]")
{
  // It was settings::paths::results, a process-wide global (removed pre-tag);
  // on the synced base a Problem built after `settings::paths::results =
  // "elsewhere"` wrote to "elsewhere".
  CHECK(dtwc::Problem{}.output_folder() == fs::path{ "./results/" });
  CHECK(dtwc::Problem{ "named" }.output_folder() == fs::path{ "./results/" });
}
