/**
 * @file test_thread_scratch.cpp
 * @brief The thread scratch of the DTW kernels and the algorithms (core/thread_scratch.hpp): what a thread holds after
 *        each variant's kernels ran is core::thread_scratch_bytes, to the byte; release_thread_scratch() and dtwc::run,
 *        returning or throwing, give every thread's back, a Tier-2 k-sweep keeps it; within an explicit --ram-limit a
 *        run's team is the threads whose scratch fits (core::KernelTeam), the same digits as the whole team's, and one
 *        thread that does not fit is refused, naming the bytes; a stream gives its threads' back at its phase
 *        boundaries, the matrix stays in RAM while one thread fits beside it, and Arrow IPC's routing is monotone in
 *        the limit.
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
#include <algorithms/detail/fast_clara_plan.hpp>
#include <algorithms/fast_pam.hpp>
#include <base/parallelisation.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/dtw_dispatch.hpp>
#include <core/thread_scratch.hpp>
#include <io/arrow_ipc.hpp>
#include <scores.hpp>

#include "../../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Message;
using Catch::Matchers::MessageMatches;
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

TEST_CASE("core::KernelTeam: a phase's team only shrinks, and the threads it leaves keep what they hold, until "
          "release() gives every thread's back",
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
                       Message("d: --ram-limit = 1000 cannot hold one thread's scratch, 350 bytes (the DTW kernels' "
                               "for series of up to 4 time steps and the method's tables), 50 more than it holds, "
                               "beside the 300 bytes the run holds with it and the 800 its threads hold already; raise "
                               "the limit."));
  CHECK(team.fitting(100, 350) == 2); // the query: 2 of the team grow 50 each beside 100 and 800; nothing changes
  CHECK(team.priced() == 800);
  KernelTeam given_back = team;
  given_back.release(); // where no kernel runs: every thread's scratch given back, the next fit from the whole team
  CHECK(given_back.priced() == 0);
  CHECK(given_back.fit("r", 100, 200, 2) == 4); // kept, the 2 would stay at 300; from nothing, 4 x 200 beside 100
  CHECK(given_back.priced() == 800);
  // What the run holds alone exceeds the limit: said so, with what one thread would need beside it.
  CHECK_THROWS_MATCHES(team.fit("d", 1200, 350, 4, "the series"), dtwc::InvalidInput,
                       Message("d: --ram-limit = 1000 cannot hold the series alone, 1200 bytes, let alone one thread's "
                               "scratch, 350 bytes (the DTW kernels' for series of up to 4 time steps and the method's "
                               "tables), 50 more than it holds, beside the 1200 bytes the run holds with it and the "
                               "800 its threads hold already; raise the limit."));
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
                                    + " bytes of scratch (the DTW kernels' for series of up to 200 time steps and the "
                                      "method's tables), as many as --ram-limit = " + std::to_string(config.ram_limit)
                                    + " holds beside the " + std::to_string(rest) + " bytes the run holds with it"));
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
                               + " cannot hold one thread's scratch, " + std::to_string(per_thread)
                               + " bytes (the DTW kernels' for series of up to 200 time steps and the method's "
                                 "tables), beside the " + std::to_string(rest)
                               + " bytes the run holds with it; raise the limit."));
  // The series and the matrix alone beyond the limit: said so.
  config.ram_limit = rest - 1;
  CHECK_THROWS_MATCHES(dtwc::run(config, series()), dtwc::InvalidInput,
                       MessageMatches(Catch::Matchers::StartsWith(
                         "FastPAM: --ram-limit = " + std::to_string(rest - 1) + " cannot hold the series and the "
                         "distance matrix alone, " + std::to_string(rest) + " bytes, let alone one thread's scratch")));
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
  CHECK(std::regex_search(out, std::regex("FastCLARA's PAMs: 3 of 4 threads, each holding [0-9]+ bytes of scratch "
                                          "\\(the DTW kernels' for series of up to 4 time steps and the method's "
                                          "tables\\), as many as --ram-limit = " + std::to_string(less)
                                          + " holds beside the [0-9]+ bytes the run holds with it")));
  // When the PAMs finish no kernel runs until the pass: their threads give their scratch back and the pass's team is
  // fitted afresh from the whole one, so it says how many of the 4 it keeps (here the names read after the stream bind
  // both phases); a team that went on from the PAMs' 3 would say nothing.
  CHECK_THAT(out, ContainsSubstring("FastCLARA's assignment: 3 of 4 threads"));
}

TEST_CASE("dtwc::run's guard gives every thread's scratch back as a run starts, not only when it returns",
          "[thread_scratch][run]")
{
  dtwc::Problem prob{ "earlier" };
  prob.set_data(data_of(walks<double>(16, 200, 1, false), 1));
  prob.fill_distance_matrix(); // a Tier-2 call keeps its scratch
  REQUIRE(all_threads_bytes() > 0);
  const dtwc::core::ScratchRelease release; // what dtwc::run makes first
  CHECK(all_threads_bytes() == 0);
}

TEST_CASE("Within --ram-limit TADPole's threads hold their count rows, in-RAM FastCLARA's team fits beside its sample "
          "matrix and labels",
          "[thread_scratch][run][ram_limit]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  constexpr std::size_t n = 64, steps = 50;
  dtwc::Config config;
  config.output.clear(); // nothing written: the matrix stays in RAM
  config.k = 3;
  config.band = -1;
  config.verbose = true;
  const auto series = [&] { return data_of(walks<double>(n, steps, 1, false), 1); };
  const std::size_t values = n * steps * sizeof(double);
  const std::size_t kernels =
    dtwc::core::thread_scratch_bytes(distance(DTWVariant::Standard, -1), steps, Precision::Float64);

  // TADPole: each thread counts every point's neighbours in a row of its own, beside the kernels' scratch.
  config.method = dtwc::Method::TADPole;
  const std::size_t matrix = dtwc::core::packed_size(n) * sizeof(double);
  const std::size_t tadpole = kernels + n * sizeof(dtwc::index_t);
  config.ram_limit = values + matrix + 2 * tadpole + tadpole / 2;
  CHECK_THAT(printed([&] { (void)dtwc::run(config, series()); }),
             ContainsSubstring("TADPole clustering: 2 of 4 threads, each holding " + std::to_string(tadpole)
                               + " bytes"));

  // FastCLARA in RAM: a sample's PAM matrix, then every sample's labels and nearest distances, beside the series.
  config.method = dtwc::Method::CLARA;
  config.sample_size = 16;
  config.n_samples = 2;
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 3;
  options.sample_size = 16;
  options.n_samples = 2;
  const auto plan = dtwc::algorithms::detail::resolve_clara_plan(static_cast<std::int64_t>(n), options, "run");
  const std::size_t clara = std::max(dtwc::core::packed_size(16) * sizeof(double),
                                     dtwc::algorithms::detail::clara_pass_bytes(plan, 2));
  const std::size_t per_thread = kernels + dtwc::detail::fast_pam_thread_bytes(3)
                                 + dtwc::algorithms::detail::clara_assign_thread_bytes(6, Precision::Float64);
  config.ram_limit = values + clara + 2 * per_thread + per_thread / 2;
  CHECK_THAT(printed([&] { (void)dtwc::run(config, series()); }),
             ContainsSubstring("FastCLARA: 2 of 4 threads, each holding " + std::to_string(per_thread)
                               + " bytes of scratch (the DTW kernels' for series of up to 50 time steps and the "
                                 "method's tables), as many as --ram-limit = " + std::to_string(config.ram_limit)
                               + " holds beside the " + std::to_string(values + clara)
                               + " bytes the run holds with it"));
}

TEST_CASE("Beside a complete --dist-matrix, which no kernel adds to, the matrix stays in RAM within --ram-limit",
          "[thread_scratch][run][ram_limit]")
{
  // Reviewer R1's case: 4 series of 100,000 values, FastPAM under a band of 10, results written, --ram-limit 100M: one
  // thread's kernel scratch (the lanes' 25.6 MB) times the team is past the limit, but no kernel runs beside a complete
  // matrix: its 80 bytes stay in RAM, as 819e8703 kept them. At 10M one thread's kernel scratch alone is past it.
  const dtwc::test_support::ScratchDirectory scratch{ "scratch_dist_matrix" };
  const auto series = [] { return data_of(walks<double>(4, 100000, 1, false), 1); };
  dtwc::Config config;
  config.k = 2;
  config.method = dtwc::Method::PAM;
  config.band = 10;
  config.name = "long4";
  config.output = dtwc::path_to_utf8(scratch.path / "full");
  const auto full = dtwc::run(config, series()); // the matrix computed, and written
  const auto matrix = scratch.path / "full" / "long4_distance_matrix.csv";
  REQUIRE(std::filesystem::exists(matrix));
  REQUIRE(dtwc::core::thread_scratch_bytes(distance(DTWVariant::Standard, 10), 100000, Precision::Float64)
          > (std::size_t{ 10 } << 20));
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

TEST_CASE("Within --ram-limit the distance matrix stays in RAM while one thread's scratch fits beside it and the "
          "series and it takes at most half of what they and one thread leave; the team beside it shrinks",
          "[thread_scratch][run][ram_limit]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  // Reviewer R1's shape: 200 series of 100 values, FastPAM, results written. At the series, twice the matrix and one
  // thread's scratch (the kernels', FasterPAM's and the silhouettes' tables) the matrix takes half of what the series
  // and that thread leave: it stays in RAM, beside as many threads as fit (four); half of what the series and the four
  // threads' scratch leave would have mapped it, or without llfio refused the run.
  constexpr std::size_t n = 200, steps = 100;
  const dtwc::test_support::ScratchDirectory scratch{ "scratch_storage_one_thread" };
  const auto series = [&] { return data_of(walks<double>(n, steps, 1, false), 1); };
  dtwc::Config config;
  config.k = 3;
  config.method = dtwc::Method::PAM;
  config.band = -1;
  config.name = "s200";
  config.output = dtwc::path_to_utf8(scratch.path / "whole");
  const auto whole = dtwc::run(config, series());
  const std::size_t values = n * steps * sizeof(double), matrix = dtwc::core::packed_size(n) * sizeof(double);
  const std::size_t each = dtwc::core::thread_scratch_bytes(distance(DTWVariant::Standard, -1), steps,
                                                            Precision::Float64)
                           + dtwc::detail::fast_pam_thread_bytes(3) + dtwc::scores::detail::silhouette_thread_bytes(3);
  config.ram_limit = values + 2 * matrix + each;
  REQUIRE(config.ram_limit >= values + matrix + 4 * each); // the four fit beside it in RAM
  config.output = dtwc::path_to_utf8(scratch.path / "limited");
  config.verbose = true;
  double cost = 0;
  const std::string out = printed([&] { cost = dtwc::run(config, series()).cost(); });
  INFO(out);
  CHECK_THAT(out, !ContainsSubstring("Using memory-mapped distance matrix"));
  CHECK_FALSE(std::filesystem::exists(scratch.path / "limited" / "s200.dtwm"));
  CHECK(cost == whole.cost());
}

namespace {

/// What Arrow IPC's routing prices FastCLARA's in-RAM route at (k 3, two samples of 8, Float64, nothing written) on
/// `file` under `band` and `variant`, for series of up to `steps` time steps of the file's features: the file's
/// resident bytes, a sample PAM's matrix or every sample's labels and nearest distances, and one thread's scratch. With
/// the longest series' steps, the least limit that routes the series into RAM.
std::size_t ipc_in_ram_bytes(const std::filesystem::path &file, int band, std::size_t steps,
                             DTWVariant variant = DTWVariant::Standard)
{
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 3;
  options.sample_size = 8;
  options.n_samples = 2;
  const dtwc::io::IpcFile ipc(file);
  const auto plan = dtwc::algorithms::detail::resolve_clara_plan(ipc.num_rows(), options, "run");
  return ipc.resident_bytes(Precision::Float64, false)
         + std::max(dtwc::core::packed_size(8) * sizeof(double), dtwc::algorithms::detail::clara_pass_bytes(plan, 2))
         + dtwc::core::thread_scratch_bytes(distance(variant, band, ipc.ndim()), steps, Precision::Float64)
         + dtwc::detail::fast_pam_thread_bytes(3)
         + dtwc::algorithms::detail::clara_assign_thread_bytes(6, Precision::Float64);
}

} // namespace

TEST_CASE("FastCLARA streamed: a series longer than any sampled, read late, has its batch read again in smaller ones "
          "and its team fitted afresh, every thread's scratch given back first; the stream runs at every limit from its "
          "least up to where the in-RAM route runs, with the same results",
          "[thread_scratch][run][arrow][streaming]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  // Each file holds one series of 1,500 steps, read late, that the samples do not hold: clara_long_late.arrow's other
  // 2,047 are 8 steps (LZ4, record batches of 64: every lane width's plan reads the long one's batch with others, and so
  // again on its own), clara_late_wide.arrow's 511 are 48 (list<float>, batches of 32, the long one in the last), so
  // the scratch the PAMs' and the pass's 4 threads hold for them is not tiny.
  for (const auto &[name, sampled, rows, reread] :
       { std::tuple{ "clara_long_late.arrow", std::size_t{ 8 }, 2048, true },
         std::tuple{ "clara_late_wide.arrow", std::size_t{ 48 }, 512, false } }) {
    const auto file = std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader" / name;
    dtwc::Config config;
    config.output.clear();
    config.input = dtwc::path_to_utf8(file);
    config.method = dtwc::Method::CLARA;
    config.k = 3;
    config.sample_size = 8;
    config.n_samples = 2;
    config.seed = 42;
    const auto whole = dtwc::run(config); // in RAM
    config.verbose = true;
    struct Outcome
    {
      bool ran;
      std::string text; // -v's, or the refusal's
      bool streamed() const { return ran && text.find("Arrow IPC walk selected streaming") != std::string::npos; }
    };
    const auto at = [&](std::size_t limit) -> Outcome {
      config.ram_limit = limit;
      try {
        return { true, printed([&] {
                   const auto result = dtwc::run(config);
                   CHECK(result.labels() == whole.labels());
                   CHECK(result.medoids() == whole.medoids());
                   CHECK(result.cost() == whole.cost());
                 }) };
      } catch (const dtwc::InvalidInput &e) {
        return { false, e.what() };
      }
    };
    // The in-RAM route from where RAM holds the series and one thread's scratch for the longest, 1,500 steps (the walk
    // bounds it above, so its lengths are read); below it the stream, down to the least limit it runs at, by bisection.
    const dtwc::io::IpcFile ipc(file);
    REQUIRE(ipc.longest() == 1500);
    REQUIRE(ipc.longest_bound() > 1500);
    const std::size_t in_ram = ipc_in_ram_bytes(file, config.band, 1500);
    std::size_t refused = 1, streams = in_ram - 1;
    REQUIRE_THAT(at(streams).text, ContainsSubstring("Arrow IPC walk selected streaming: " + std::to_string(rows)));
    REQUIRE_FALSE(at(refused).ran);
    while (streams - refused > 1) {
      const std::size_t mid = refused + (streams - refused) / 2;
      (at(mid).ran ? streams : refused) = mid;
    }
    const Outcome least = at(streams), below = at(streams - 1);
    INFO(name << ": at the least limit " << streams << ":\n" << least.text << "\na byte below:\n" << below.text);
    REQUIRE(least.streamed());
    // Its record batch read again on its own (where the plan read it with others), and the team one thread for its
    // 1,500 steps.
    if (reread) CHECK_THAT(least.text, ContainsSubstring("FastCLARA's pass: 1 record batch a batch from record batch "));
    CHECK_THAT(least.text, ContainsSubstring("for series of up to 1500 time steps"));
    CHECK_THAT(least.text, ContainsSubstring("FastCLARA's assignment: 1 of 4 threads"));
    // A byte below, the typed refusal: one thread's scratch beside what the run holds, the threads having given theirs
    // back, is the limit.
    REQUIRE_FALSE(below.ran);
    std::smatch m;
    REQUIRE(std::regex_search(below.text, m,
                              std::regex("^FastCLARA's assignment: --ram-limit = ([0-9]+) cannot hold one thread's "
                                         "scratch, ([0-9]+) bytes \\(the DTW kernels' for series of up to 1500 time "
                                         "steps and the method's tables\\), beside the ([0-9]+) bytes the run holds "
                                         "with it; raise the limit\\.$")));
    CHECK(std::stoull(m[1].str()) == streams - 1);
    CHECK(std::stoull(m[2].str()) + std::stoull(m[3].str()) == streams);
    // Had the 4 threads kept their scratch for the sampled series, one growing, the limit would hold 3 x a thread's
    // more: the stream runs inside that window too (`kept`).
    const std::size_t kept = dtwc::core::thread_scratch_bytes(distance(DTWVariant::Standard, config.band), sampled,
                                                              Precision::Float64)
                             + dtwc::detail::fast_pam_thread_bytes(3)
                             + dtwc::algorithms::detail::clara_assign_thread_bytes(6, Precision::Float64);
    // Monotone: the stream at every limit from the least up to the in-RAM route's, and the in-RAM route from there on,
    // below the walk's bound's price too.
    for (const std::size_t limit : { streams + 3 * kept / 2, streams + (in_ram - streams) / 3,
                                     streams + 2 * (in_ram - streams) / 3, in_ram - 1 }) {
      if (limit < streams || limit >= in_ram) continue;
      INFO("--ram-limit " << limit);
      CHECK(at(limit).streamed());
    }
    const std::size_t bound = ipc_in_ram_bytes(file, config.band, ipc.longest_bound());
    for (const std::size_t limit : { in_ram, in_ram + (bound - in_ram) / 2, bound + bound / 4 }) {
      const Outcome o = at(limit);
      INFO("--ram-limit " << limit << ":\n" << o.text);
      CHECK(o.ran);
      CHECK_THAT(o.text, ContainsSubstring("Data loaded from Arrow IPC: " + std::to_string(rows) + " series"));
    }
  }
}

namespace {

/// OpenMP's team of 24 while it lives (the registration's 4 after it).
struct TwentyFour
{
  int previous = dtwc::get_max_threads();
#ifdef _OPENMP
  TwentyFour() { omp_set_num_threads(24); }
  ~TwentyFour() { omp_set_num_threads(previous); }
#endif
};

/// The lines of `text` that start with `prefix`.
std::string lines_of(const std::string &text, const std::string &prefix)
{
  std::istringstream in(text);
  std::string line, out;
  while (std::getline(in, line))
    if (line.rfind(prefix, 0) == 0) out += line + "\n";
  return out;
}

} // namespace

TEST_CASE("A team shrunk to --ram-limit computes what the whole team computes: a stream's batches and a mapped fill's "
          "tile-rows come from the full team",
          "[thread_scratch][run][arrow][streaming]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  const TwentyFour team_of_24;
  REQUIRE(dtwc::get_max_threads() == 24);
  // A stream of 512 series of 6 to 10 steps under a band of 4, read as Float32: a lane block takes series as long as
  // the medoid, so which points share a block is a result's. At a limit where its team shrinks, and within an outer
  // TeamCap of 2 (a team shrunk further), the same batches and the same digits as in RAM.
  const auto file = std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader"
                    / "clara_ragged.arrow";
  dtwc::Config config;
  config.output.clear();
  config.input = dtwc::path_to_utf8(file);
  config.method = dtwc::Method::CLARA;
  config.k = 3;
  config.sample_size = 8;
  config.n_samples = 2;
  config.seed = 42;
  config.band = 4;
  config.dtype = Precision::Float32; // a copy of the file's values: streamed whether llfio maps it or not
  const auto whole = dtwc::run(config); // in RAM
  config.verbose = true;
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
  // The least limit the stream runs at, by bisection (one thread); four threads' scratch above it the team is five of
  // 24, and a batch sized beside what those five hold would hold fewer record batches than one beside a team of 2.
  std::size_t runs = dtwc::io::IpcFile(file).resident_bytes(Precision::Float32, false), refused = 0;
  REQUIRE_THAT(streamed(runs), ContainsSubstring("Arrow IPC walk selected streaming: 512 series"));
  while (runs - refused > 1) {
    const std::size_t mid = refused + (runs - refused) / 2;
    (streamed(mid).empty() ? refused : runs) = mid;
  }
  std::smatch each;
  const std::string least = streamed(runs);
  REQUIRE(std::regex_search(least, each, std::regex("of 24 threads, each holding ([0-9]+) bytes")));
  const std::size_t limit = runs + 4 * std::stoull(each[1].str());
  const std::string shrunk = streamed(limit);
  std::string capped;
  {
    const dtwc::TeamCap two(2);
    capped = streamed(limit);
  }
  INFO("shrunk:\n" << shrunk << "\ncapped at 2:\n" << capped);
  REQUIRE_THAT(shrunk, ContainsSubstring("of 24 threads"));
  REQUIRE_THAT(capped, ContainsSubstring("Arrow IPC walk selected streaming: 512 series"));
  REQUIRE_FALSE(lines_of(shrunk, "FastCLARA's pass: ").empty());
  CHECK(lines_of(shrunk, "FastCLARA's pass: ") == lines_of(capped, "FastCLARA's pass: "));

#ifdef DTWC_HAS_MMAP
  // A mapped fill of 64 series of 990 to 1,000 steps under a band of 16, at a limit the whole team's scratch nearly
  // fills: the same tile-rows and the same matrix, bit for bit, at 24 threads and within a TeamCap of 2.
  const dtwc::test_support::ScratchDirectory scratch{ "scratch_mapped_team" };
  std::vector<std::vector<double>> ragged = walks<double>(64, 1000, 1, false);
  for (std::size_t i = 0; i < ragged.size(); ++i) ragged[i].resize(990 + i % 11);
  const DistanceConfig banded = distance(DTWVariant::Standard, 16);
  const std::size_t values = 64 * 1000 * sizeof(double);
  const std::size_t budget = values + 24 * dtwc::core::thread_scratch_bytes(banded, 1000, Precision::Float64)
                             + 64 * 64 * sizeof(double);
  const auto fill = [&](const char *name, std::vector<double> &matrix) {
    dtwc::Problem prob{ name };
    prob.set_data(data_of(ragged, 1));
    prob.set_distance(banded);
    prob.set_ram_limit(budget);
    prob.set_verbose(true);
    prob.use_mmap_distance_matrix(scratch.path / (std::string(name) + ".dtwm"));
    const std::string out = printed([&] { prob.fill_distance_matrix(); });
    for (dtwc::index_t i = 0; i < 64; ++i)
      for (dtwc::index_t j = 0; j < 64; ++j) matrix.push_back(prob.dist_by_ind(i, j));
    return lines_of(out, "Mapped distance matrix: ");
  };
  std::vector<double> at_24, at_2;
  const std::string tiles_24 = fill("team24", at_24);
  std::string tiles_2;
  {
    const dtwc::TeamCap two(2);
    tiles_2 = fill("team2", at_2);
  }
  INFO(tiles_24 << tiles_2);
  REQUIRE_FALSE(tiles_24.empty());
  CHECK(tiles_24 == tiles_2);
  CHECK(at_24 == at_2);
#endif
}

TEST_CASE("Arrow IPC's routing streams a file whose series and one thread's scratch for its longest series exceed "
          "--ram-limit",
          "[thread_scratch][run][arrow][streaming]")
{
  // The walk knows each record batch's rows and values, not each series' length: a series is at most its batch's time
  // steps less one for each other row (clara_256_lz4.arrow's 32 rows of 4 steps: 97). The in-RAM route, its team
  // fitted on the longest series once they are read, is priced with one thread's scratch for it beside what it holds
  // of the file and CLARA's terms: every series there is 4 steps, which the lengths read say where the bound cannot.
  // The thread is priced for the file's features a time step, which reach the Problem only with its series:
  // clara_256_ndim2.arrow's two (series of two or three time steps), where DDTW's derivatives hold two values a time
  // step, so a thread priced for one feature would route into RAM a limit that cannot hold it there.
  for (const auto &[name, ndim, variant, longest, bound] :
       { std::tuple{ "clara_256_lz4.arrow", std::size_t{ 1 }, DTWVariant::Standard, std::size_t{ 4 },
                     std::size_t{ 32 * 4 - 31 } },
         std::tuple{ "clara_256_ndim2.arrow", std::size_t{ 2 }, DTWVariant::DDTW, std::size_t{ 3 },
                     std::size_t{ 44 } } }) {
    INFO(name);
    const auto file = std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader" / name;
    const dtwc::io::IpcFile ipc(file);
    REQUIRE(ipc.ndim() == ndim);
    REQUIRE(ipc.longest_bound() == bound);
    REQUIRE(ipc.longest() == longest);
    dtwc::Config config;
    config.output.clear();
    config.input = dtwc::path_to_utf8(file);
    config.method = dtwc::Method::CLARA;
    config.k = 3;
    config.sample_size = 8;
    config.n_samples = 2;
    config.seed = 42;
    config.variant.variant = variant;
    config.verbose = true;
    const std::size_t in_ram = ipc_in_ram_bytes(file, config.band, longest, variant);
    config.ram_limit = in_ram - 1;
    CHECK_THAT(printed([&] { (void)dtwc::run(config); }), ContainsSubstring("Arrow IPC walk selected streaming"));
    config.ram_limit = in_ram;
    const std::string in_ram_text = printed([&] { (void)dtwc::run(config); });
    CHECK_THAT(in_ram_text, !ContainsSubstring("selected streaming"));
    CHECK_THAT(in_ram_text, ContainsSubstring("Data loaded from Arrow IPC: 256 series"));
  }
}

TEST_CASE("Arrow IPC's batch of record batches does not depend on a TeamCap: every thread of the full team holds its "
          "decoder's scratch",
          "[thread_scratch][arrow][streaming]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::full_team_threads() == 4);
  // An LZ4 file: each decoding thread but the first holds a reader's scratch beside its record batch. A budget of 4
  // record batches and 3 readers' scratch holds 4 decoded by the full team; a team capped to 1 would fit more.
  const dtwc::io::IpcFile file(std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader"
                               / "clara_256_lz4.arrow");
  const std::size_t batch = file.largest_row_group_bytes(false), scratch = file.reader_bytes();
  REQUIRE(scratch > batch); // so the cap would change the count
  const std::size_t budget = 4 * batch + 3 * scratch;
  const int whole = file.row_groups_per_batch(budget, false);
  CHECK(whole == 4);
  const dtwc::TeamCap one(1);
  CHECK(file.row_groups_per_batch(budget, false) == whole);
}
