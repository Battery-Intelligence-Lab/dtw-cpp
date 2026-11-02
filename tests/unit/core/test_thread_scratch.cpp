/**
 * @file test_thread_scratch.cpp
 * @brief The thread scratch of the DTW kernels and the algorithms (core/thread_scratch.hpp): dtwc::run gives every
 *        thread's back as it starts and when it returns or throws, a Tier-2 k-sweep keeps it, and WDTW's cached
 *        weights are built again after a release; beside a complete --dist-matrix the matrix stays in RAM within
 *        --ram-limit.
 *
 * @details Registered with OMP_NUM_THREADS=4: a team of 4. A thread's bytes are what the buffers on the closed list
 *          hold (core::thread_scratch_held(), by thread number).
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <base/parallelisation.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/thread_scratch.hpp>

#include "../../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

/// `n` series of `steps` time steps, each its own smooth walk.
std::vector<std::vector<double>> walks(std::size_t n, std::size_t steps)
{
  std::vector<std::vector<double>> out(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t v = 0; v < steps; ++v) out[i].push_back(std::sin(0.37 * double(i + 1) + 0.11 * double(v)));
  return out;
}

dtwc::Data data_of(std::vector<std::vector<double>> series)
{
  std::vector<std::string> names(series.size());
  for (std::size_t i = 0; i < names.size(); ++i) names[i] = "s" + std::to_string(i);
  return dtwc::Data(std::move(series), std::move(names), 1);
}

std::size_t all_threads_bytes()
{
  const auto bytes = dtwc::core::thread_scratch_held();
  return std::accumulate(bytes.begin(), bytes.end(), std::size_t{ 0 });
}

/// std::cout's text while `f` runs.
template <typename F>
std::string printed(F &&f)
{
  std::ostringstream out;
  auto *const previous = std::cout.rdbuf(out.rdbuf());
  try {
    f();
  } catch (...) {
    std::cout.rdbuf(previous);
    throw;
  }
  std::cout.rdbuf(previous);
  return out.str();
}

} // namespace

TEST_CASE("dtwc::run gives every thread's scratch back as it starts and when it returns or throws; a Tier-2 k-sweep "
          "keeps it",
          "[thread_scratch][run]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4); // OMP_NUM_THREADS=4: a team, not one thread
  dtwc::Problem prob{ "sweep" };
  prob.set_data(data_of(walks(64, 50)));
  prob.set_method(dtwc::Method::PAM);
  std::vector<std::size_t> kept;
  for (const dtwc::index_t k : { 4, 3, 2 }) { // the matrix filled once, FasterPAM's table at its largest first
    prob.set_n_clusters(k);
    (void)prob.cluster();
    kept.push_back(all_threads_bytes());
  }
  CHECK(kept.front() > 0);
  CHECK(kept == std::vector<std::size_t>(3, kept.front()));
  {
    const dtwc::core::ScratchRelease release; // what dtwc::run makes first: the sweep's scratch given back
    CHECK(all_threads_bytes() == 0);
  }

  dtwc::Config config;
  config.output.clear(); // nothing written: the matrix stays in RAM
  config.k = 3;
  config.method = dtwc::Method::PAM;
  config.band = -1;
  const auto result = dtwc::run(config, data_of(walks(64, 50)));
  CHECK(result.labels().size() == 64);
  CHECK(dtwc::core::thread_scratch_held() == std::vector<std::size_t>(4, 0));

  dtwc::Problem again{ "again" };
  again.set_data(data_of(walks(64, 50)));
  again.fill_distance_matrix();
  REQUIRE(all_threads_bytes() > 0);
  config.k = 0; // refused before any series is read: the scratch held before is given back too
  CHECK_THROWS_AS(dtwc::run(config, data_of(walks(64, 50))), dtwc::InvalidInput);
  CHECK(dtwc::core::thread_scratch_held() == std::vector<std::size_t>(4, 0));

  // WDTW's cached weights are emptied with the max_dev and g they were built for: the next call builds them again.
  const auto pair = walks(2, 40);
  const double wdtw = dtwc::wdtwBanded<double>(pair[0], pair[1], -1, 0.05);
  REQUIRE(all_threads_bytes() > 0);
  dtwc::core::release_thread_scratch();
  CHECK(all_threads_bytes() == 0);
  CHECK(dtwc::wdtwBanded<double>(pair[0], pair[1], -1, 0.05) == wdtw);
}

TEST_CASE("Beside a complete --dist-matrix, which no kernel adds to, the matrix stays in RAM within --ram-limit",
          "[thread_scratch][run][ram_limit]")
{
  // Reviewer R1's case: 4 series of 100,000 values, FastPAM under a band of 10, results written, --ram-limit 100M and
  // 10M: no kernel runs beside a complete matrix, and its 80 bytes stay in RAM, as 819e8703 kept them.
  const dtwc::test_support::ScratchDirectory scratch{ "scratch_dist_matrix" };
  const auto series = [] { return data_of(walks(4, 100000)); };
  dtwc::Config config;
  config.k = 2;
  config.method = dtwc::Method::PAM;
  config.band = 10;
  config.name = "long4";
  config.output = dtwc::path_to_utf8(scratch.path / "full");
  const auto full = dtwc::run(config, series()); // the matrix computed, and written
  const auto matrix = scratch.path / "full" / "long4_distance_matrix.csv";
  REQUIRE(std::filesystem::exists(matrix));
  config.dist_matrix = dtwc::path_to_utf8(matrix);
  config.verbose = true;
  for (const std::size_t mib : { 100, 10 }) {
    config.ram_limit = mib << 20;
    config.output = dtwc::path_to_utf8(scratch.path / ("read" + std::to_string(mib)));
    double cost = 0;
    const std::string out = printed([&] { cost = dtwc::run(config, series()).cost(); });
    INFO("--ram-limit " << mib << "M:\n" << out);
    CHECK_THAT(out, ContainsSubstring("Loaded distance matrix from"));
    CHECK_THAT(out, !ContainsSubstring("Using memory-mapped distance matrix"));
    CHECK_FALSE(std::filesystem::exists(scratch.path / ("read" + std::to_string(mib)) / "long4.dtwm"));
    CHECK(cost == full.cost());
  }
}
