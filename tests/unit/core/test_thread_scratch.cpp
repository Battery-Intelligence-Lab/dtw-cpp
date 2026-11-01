/**
 * @file test_thread_scratch.cpp
 * @brief The thread scratch of the DTW kernels and the algorithms (core/thread_scratch.hpp): what a thread holds after
 *        each variant's kernels ran is core::thread_scratch_bytes, to the byte; release_thread_scratch() and dtwc::run,
 *        returning or throwing, give every thread's back, a Tier-2 k-sweep keeps it; within an explicit --ram-limit a
 *        run's team is the threads whose scratch fits (core::KernelTeam), the same digits as the whole team's, and one
 *        thread that does not fit is refused, naming the bytes.
 *
 * @details Registered with OMP_NUM_THREADS=4, so a team of 4 shrinks to 2. A thread's bytes are its registered
 *          buffers' capacities (core::registered_scratch_bytes(), the calling thread's first), read after the kernels
 *          ran on it alone (TeamCap 1): the variant's fill through a Problem, the route a run takes, and its bound
 *          per-pair function on the pairs the fill does not take that reach a buffer's most (under a band, two series
 *          of band + 1, which dtw_linear takes whole). The oracle is each buffer's growth rule (dtw_dispatch.hpp), not
 *          the kernels' code: a buffer that grew past it, or one left unregistered, moves the count.
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <algorithms/detail/fast_clara_assign.hpp>
#include <algorithms/fast_pam.hpp>
#include <base/parallelisation.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/dtw_dispatch.hpp>
#include <core/thread_scratch.hpp>
#include <io/arrow_ipc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Message;
using dtwc::core::DistanceConfig;
using dtwc::core::DTWVariant;
using dtwc::core::MetricType;
using dtwc::core::MissingStrategy;
using dtwc::core::MVMode;
using dtwc::core::Precision;

namespace {

/// `n` series of `steps` time steps of `ndim` features, each its own smooth walk; every 7th value NaN where `gaps`.
template <typename T>
std::vector<std::vector<T>> walks(std::size_t n, std::size_t steps, std::size_t ndim, bool gaps)
{
  std::vector<std::vector<T>> out(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t v = 0; v < steps * ndim; ++v)
      out[i].push_back(gaps && v % 7 == 3 ? std::numeric_limits<T>::quiet_NaN()
                                           : static_cast<T>(std::sin(0.37 * double(i + 1) + 0.11 * double(v))));
  return out;
}

template <typename T>
dtwc::Data data_of(std::vector<std::vector<T>> series, std::size_t ndim)
{
  std::vector<std::string> names(series.size());
  for (std::size_t i = 0; i < names.size(); ++i) names[i] = "s" + std::to_string(i);
  return dtwc::Data(std::move(series), std::move(names), ndim);
}

std::size_t this_thread_bytes() { return dtwc::core::registered_scratch_bytes().front(); }

std::size_t all_threads_bytes()
{
  const auto bytes = dtwc::core::registered_scratch_bytes();
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

struct Case
{
  const char *name;
  DistanceConfig distance; // ndim: the series'
  bool gaps = false;       // NaN in the series (a missing-data strategy)
};

/// What one thread holds after `c`'s kernels ran on series of `steps` time steps of T, on it alone.
template <typename T>
std::size_t held_after(const Case &c, std::size_t steps)
{
  const std::size_t ndim = c.distance.ndim, band = static_cast<std::size_t>(c.distance.band);
  dtwc::Problem prob{ c.name };
  prob.set_data(data_of(walks<T>(3, steps, ndim, c.gaps), ndim));
  prob.set_distance(c.distance);
  dtwc::core::release_thread_scratch();
  {
    const dtwc::TeamCap one(1); // every region on this thread
    prob.fill_distance_matrix();
    // The bound per-pair function on two series of `steps` (the fill's lanes take them, where the binding has lanes)
    // and, under a band they exceed, two of band + 1, which dtw_linear takes whole. WDTW's weight table, the stream's
    // (its Problem holds no series), is built for each pair's length: the longest pair last.
    const dtwc::Data none = data_of(std::vector<std::vector<T>>{}, ndim);
    const auto pair = dtwc::core::resolve_dtw_fn<T>(prob.distance(), c.distance.variant.variant == DTWVariant::WDTW
                                                                       ? none
                                                                       : prob.data());
    const auto call = [&](std::size_t length) {
      const auto two = walks<T>(2, length, ndim, c.gaps);
      (void)pair(std::span<const T>(two[0]), std::span<const T>(two[1]));
    };
    if (c.distance.band >= 0 && steps > band + 1) call(band + 1);
    call(steps);
  }
  return this_thread_bytes();
}

DistanceConfig distance(DTWVariant variant, int band, std::size_t ndim = 1, MetricType metric = MetricType::L1,
                        MissingStrategy missing = MissingStrategy::Error, MVMode mode = MVMode::Dependent)
{
  DistanceConfig c;
  c.variant.variant = variant;
  c.variant.mv_mode = mode;
  c.metric = metric;
  c.missing = missing;
  c.band = band;
  c.ndim = ndim;
  return c;
}

} // namespace

TEST_CASE("A thread holds core::thread_scratch_bytes after each variant's kernels ran on it", "[thread_scratch]")
{
  const Case cases[] = {
    { "lanes and dtw_linear", distance(DTWVariant::Standard, -1) },
    { "lanes, banded and dtw_linear", distance(DTWVariant::Standard, 5) },
    { "lanes, squared L2", distance(DTWVariant::Standard, 5, 1, MetricType::SquaredL2) },
    { "a band over every pair", distance(DTWVariant::Standard, 60) },
    { "multivariate", distance(DTWVariant::Standard, 5, 2) },
    { "multivariate, independent", distance(DTWVariant::Standard, 5, 2, MetricType::L1, MissingStrategy::Error,
                                            MVMode::Independent) },
    { "zero cost", distance(DTWVariant::Standard, 5, 1, MetricType::L1, MissingStrategy::ZeroCost), true },
    { "zero cost, multivariate", distance(DTWVariant::Standard, -1, 2, MetricType::L1, MissingStrategy::ZeroCost),
      true },
    { "AROW", distance(DTWVariant::Standard, 5, 1, MetricType::L1, MissingStrategy::AROW), true },
    { "AROW, multivariate", distance(DTWVariant::Standard, 5, 2, MetricType::L1, MissingStrategy::AROW), true },
    { "interpolate", distance(DTWVariant::Standard, 5, 1, MetricType::L1, MissingStrategy::Interpolate), true },
    { "DDTW", distance(DTWVariant::DDTW, 5) },
    { "DDTW, multivariate", distance(DTWVariant::DDTW, -1, 2) },
    { "WDTW", distance(DTWVariant::WDTW, 5) },
    { "WDTW, multivariate", distance(DTWVariant::WDTW, -1, 2) },
    { "ADTW", distance(DTWVariant::ADTW, 5) },
    { "Soft-DTW", distance(DTWVariant::SoftDTW, 5) },
    { "MSM", distance(DTWVariant::MSM, -1) },
    { "TWE", distance(DTWVariant::TWE, -1) },
  };
  constexpr std::size_t steps = 40;
  for (const Case &c : cases) {
    INFO(c.name << ", band " << c.distance.band << ", ndim " << c.distance.ndim);
    CHECK(held_after<double>(c, steps) == dtwc::core::thread_scratch_bytes(c.distance, steps, Precision::Float64));
    CHECK(held_after<float>(c, steps) == dtwc::core::thread_scratch_bytes(c.distance, steps, Precision::Float32));
  }
  // The lanes' rows are dtw_lanes<T> values of either width: 2 x 40 rows of 128 bytes (64 under cl), and a column of
  // 40 values without a band.
  CHECK(dtwc::core::thread_scratch_bytes(distance(DTWVariant::Standard, -1), steps, Precision::Float64)
        == 2 * steps * dtwc::core::dtw_lanes<double> * sizeof(double) + steps * sizeof(double));
  dtwc::core::release_thread_scratch();
}

TEST_CASE("A thread holds the algorithms' tables a planner prices", "[thread_scratch]")
{
  // FasterPAM's swap table from 2,000 points (groups of core::group_rows candidates), on a filled matrix.
  dtwc::Problem prob{ "tables" };
  prob.set_data(data_of(walks<double>(2000, 2, 1, false), 1));
  prob.fill_distance_matrix();
  dtwc::core::release_thread_scratch();
  {
    const dtwc::TeamCap one(1);
    (void)dtwc::fast_pam(prob, 3);
  }
  CHECK(this_thread_bytes() == dtwc::detail::fast_pam_thread_bytes(3));
  // FastCLARA's assignment table beside the lanes': 2 samples of 3 medoids, every one against each lane block of the
  // 40 points; under a band, where every compiler's lanes take points as long as the medoid.
  constexpr std::size_t steps = 30, points = 40;
  const auto banded = distance(DTWVariant::Standard, 5);
  const auto series = walks<double>(points, steps, 1, false);
  const auto pair = dtwc::core::resolve_dtw_fn<double>(banded, data_of(walks<double>(0, 0, 1, false), 1));
  const std::vector<dtwc::index_t> medoids{ 0, 1, 2, 3, 4, 5 };
  std::vector<std::vector<dtwc::index_t>> labels(2, std::vector<dtwc::index_t>(points));
  std::vector<double> best(2 * points);
  dtwc::core::release_thread_scratch();
  {
    const dtwc::TeamCap one(1);
    dtwc::algorithms::detail::assign_points_cpu<double>(
      0, points, best, medoids, labels, banded, pair,
      [&](dtwc::index_t p) { return std::span<const double>(series[static_cast<std::size_t>(p)]); },
      [&](dtwc::index_t j) { return std::span<const double>(series[static_cast<std::size_t>(medoids[j])]); });
  }
  CHECK(this_thread_bytes()
        == 2 * steps * dtwc::core::dtw_lanes<double> * sizeof(double)
             + dtwc::algorithms::detail::clara_assign_thread_bytes(6, Precision::Float64));
  CHECK(dtwc::algorithms::detail::clara_assign_thread_bytes(6, Precision::Float64)
        == 6 * dtwc::core::dtw_lanes<double> * sizeof(double));
  dtwc::core::release_thread_scratch();
}

TEST_CASE("core::KernelTeam: a phase's team only shrinks, and the threads it leaves keep what they hold",
          "[thread_scratch]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  using dtwc::core::KernelTeam;
  KernelTeam team(1000, false);
  CHECK(team.fit("a", 100, 100, 1) == 4); // 4 x 100 beside 100
  CHECK(team.priced() == 400);
  CHECK(team.fit("b", 100, 300, 3) == 2); // 200 more each: (1000 - 100 - 400) / 200; the other 2 keep their 100
  CHECK(team.priced() == 800);
  CHECK(team.fit("c", 100, 200, 2) == 2); // nothing grows: what a thread holds stays, the team too
  CHECK(team.priced() == 800);
  CHECK_THROWS_MATCHES(team.fit("d", 300, 350, 4), dtwc::InvalidInput,
                       Message("d: --ram-limit = 1000 cannot hold one thread's DTW scratch, 350 bytes for series of up "
                               "to 4 time steps, beside the 300 bytes the run holds with it and the 800 its threads "
                               "hold already; raise the limit."));
  KernelTeam names(1000, false, 750); // what follows the phases: every phase leaves room for it
  CHECK(names.fit("e", 100, 100, 1) == 2);
  KernelTeam automatic(0, false); // the automatic budget: nothing shrinks, is refused or priced
  CHECK(automatic.fit("f", std::size_t{ 1 } << 40, std::size_t{ 1 } << 40, 1) == 4);
  CHECK(automatic.priced() == 0);
}

TEST_CASE("dtwc::run gives every thread's scratch back, returning or throwing; a Tier-2 k-sweep keeps it",
          "[thread_scratch][run]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4); // OMP_NUM_THREADS=4: a team, not one thread
  dtwc::Problem prob{ "sweep" };
  prob.set_data(data_of(walks<double>(64, 50, 1, false), 1));
  prob.set_method(dtwc::Method::PAM);
  std::vector<std::size_t> kept;
  for (const dtwc::index_t k : { 4, 3, 2 }) { // the matrix filled once, FasterPAM's table at its largest first
    prob.set_n_clusters(k);
    (void)prob.cluster();
    kept.push_back(all_threads_bytes());
  }
  CHECK(kept.front() > 0);
  CHECK(kept == std::vector<std::size_t>(3, kept.front()));

  dtwc::Config config;
  config.output.clear(); // nothing written: the matrix stays in RAM
  config.k = 3;
  config.method = dtwc::Method::PAM;
  config.band = -1;
  const auto result = dtwc::run(config, data_of(walks<double>(64, 50, 1, false), 1));
  CHECK(result.labels().size() == 64);
  CHECK(dtwc::core::registered_scratch_bytes() == std::vector<std::size_t>(4, 0));

  dtwc::Problem again{ "again" };
  again.set_data(data_of(walks<double>(64, 50, 1, false), 1));
  again.fill_distance_matrix();
  REQUIRE(all_threads_bytes() > 0);
  config.k = 0; // refused before any series is read: the scratch held before is given back too
  CHECK_THROWS_AS(dtwc::run(config, data_of(walks<double>(64, 50, 1, false), 1)), dtwc::InvalidInput);
  CHECK(dtwc::core::registered_scratch_bytes() == std::vector<std::size_t>(4, 0));
}

TEST_CASE("Within --ram-limit run keeps the threads whose scratch fits, with the whole team's digits; one that does not "
          "fit is refused",
          "[thread_scratch][run][ram_limit]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  constexpr std::size_t n = 24, steps = 200;
  dtwc::Config config;
  config.output.clear(); // nothing written: the matrix stays in RAM
  config.k = 3;
  config.method = dtwc::Method::PAM;
  config.band = -1;
  config.verbose = true;
  const auto series = [&] { return data_of(walks<double>(n, steps, 1, false), 1); };
  config.verbose = false;
  const auto whole = dtwc::run(config, series());
  config.verbose = true;
  // In RAM: the series' values and the packed matrix beside each thread's scratch, the lanes and a column for 200
  // values and FasterPAM's table for 3 medoids.
  const std::size_t rest = n * steps * sizeof(double) + dtwc::core::packed_size(n) * sizeof(double);
  const std::size_t per_thread = dtwc::core::thread_scratch_bytes(distance(DTWVariant::Standard, -1), steps,
                                                                  Precision::Float64)
                                 + dtwc::detail::fast_pam_thread_bytes(3);
  config.ram_limit = rest + 2 * per_thread + per_thread / 2;
  std::vector<dtwc::index_t> labels, medoids;
  double cost = 0;
  const std::string out = printed([&] {
    const auto two = dtwc::run(config, series());
    labels = two.labels();
    medoids = two.medoids();
    cost = two.cost();
  });
  CHECK_THAT(out, ContainsSubstring("FastPAM: 2 of 4 threads, each holding " + std::to_string(per_thread)
                                    + " bytes of DTW scratch for series of up to 200 time steps, as many as --ram-limit "
                                      "= " + std::to_string(config.ram_limit) + " holds beside the "
                                    + std::to_string(rest) + " bytes the run holds with it"));
  CHECK(labels == whole.labels());
  CHECK(medoids == whole.medoids());
  CHECK(cost == whole.cost());
  // One thread at the least limit; a byte below it the typed refusal, before any distance.
  config.ram_limit = rest + per_thread;
  CHECK_THAT(printed([&] { CHECK(dtwc::run(config, series()).cost() == whole.cost()); }),
             ContainsSubstring("FastPAM: 1 of 4 threads"));
  config.ram_limit = rest + per_thread - 1;
  CHECK_THROWS_MATCHES(dtwc::run(config, series()), dtwc::InvalidInput,
                       Message("FastPAM: --ram-limit = " + std::to_string(rest + per_thread - 1)
                               + " cannot hold one thread's DTW scratch, " + std::to_string(per_thread)
                               + " bytes for series of up to 200 time steps, beside the " + std::to_string(rest)
                               + " bytes the run holds with it; raise the limit."));
  // Under the automatic budget nothing shrinks: every thread.
  config.ram_limit = 0;
  CHECK_THAT(printed([&] { (void)dtwc::run(config, series()); }), !ContainsSubstring("of 4 threads"));
}

TEST_CASE("FastCLARA streamed within --ram-limit keeps the threads whose scratch fits, with the whole team's digits",
          "[thread_scratch][run][arrow][streaming]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  const auto file = std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader"
                    / "clara_256_lz4.arrow";
  dtwc::Config config;
  config.output.clear(); // nothing written: the matrix stays in RAM
  config.input = dtwc::path_to_utf8(file);
  config.method = dtwc::Method::CLARA;
  config.k = 3;
  config.sample_size = 8;
  config.n_samples = 2;
  config.seed = 42;
  const auto whole = dtwc::run(config); // in RAM
  config.verbose = true;
  // The stream at `limit`: its -v text, empty where it was refused.
  const auto streamed = [&](std::size_t limit) {
    config.ram_limit = limit;
    try {
      return printed([&] {
        const auto result = dtwc::run(config);
        CHECK(result.labels() == whole.labels());
        CHECK(result.medoids() == whole.medoids());
        CHECK(result.cost() == whole.cost());
      });
    } catch (const dtwc::InvalidInput &) {
      return std::string();
    }
  };
  // The least limit the stream keeps its 4 threads within, by bisection from the limit it starts at: a byte below,
  // the stream's team is smaller (the PAMs' first, as their price is the largest), not refused.
  std::size_t full = dtwc::io::IpcFile(file).resident_bytes(Precision::Float64, false), less = 0;
  const std::string at_full = streamed(full);
  REQUIRE_THAT(at_full, ContainsSubstring("Arrow IPC walk selected streaming: 256 series"));
  REQUIRE_THAT(at_full, !ContainsSubstring("of 4 threads"));
  while (full - less > 1) {
    const std::size_t mid = less + (full - less) / 2;
    const std::string out = streamed(mid);
    (out.empty() || out.find("of 4 threads") != std::string::npos ? less : full) = mid;
  }
  const std::string out = streamed(less);
  INFO(out);
  CHECK_THAT(out, ContainsSubstring("Arrow IPC walk selected streaming: 256 series"));
  CHECK(std::regex_search(out, std::regex("FastCLARA's PAMs: 3 of 4 threads, each holding [0-9]+ bytes of DTW "
                                          "scratch for series of up to 4 time steps, as many as --ram-limit = "
                                          + std::to_string(less) + " holds beside the [0-9]+ bytes the run holds "
                                          "with it")));
}
