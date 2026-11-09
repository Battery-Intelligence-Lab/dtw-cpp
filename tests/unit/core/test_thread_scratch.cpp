/**
 * @file test_thread_scratch.cpp
 * @brief The thread scratch of the DTW kernels and the algorithms (core/thread_scratch.hpp) and dtwc::run's memory
 *        plan, which reserves it: dtwc::run gives every thread's scratch back as it starts and when it returns or
 *        throws, a Tier-2 k-sweep keeps it, and WDTW's cached weights are built again after a release; beside a
 *        complete --dist-matrix the matrix stays in RAM within --ram-limit; a --ram-limit that runs still runs when
 *        raised, its route never stepping back, with the unlimited run's clustering; the clustering does not depend on
 *        the limit, the data cap or the threads; a refusal names its terms and, in RAM, the least limit, which runs;
 *        a stream's refusals say what their --ram-limit is; the plan holds the methods' own terms, OneBatchPAM's for
 *        the batch one_batch_pam takes, TADPole's envelopes exactly where tadpole prunes.
 *
 * @details Registered with OMP_NUM_THREADS=4: a team of 4. A thread's bytes are what the buffers on the closed list
 *          hold (core::thread_scratch_held(), by thread number). The plan's numbers are read from the run's own text:
 *          -v's "Memory plan: ..." line and the refusal's "at least N bytes".
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <algorithms/tadpole.hpp>
#include <base/parallelisation.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/lower_bound_impl.hpp> // Envelope, compute_envelope
#include <core/thread_scratch.hpp>

#include "../../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
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

using Runner = std::function<dtwc::Result(const dtwc::Config &)>;

/// A run at one --ram-limit, as its text tells it: -v's plan (the route, the threads, the reserve and the data cap),
/// whether it ran or was refused (InvalidInput), and its clustering.
struct Attempt
{
  bool ran = false;
  int rank = -1; ///< the route: 0 refused, 1 streamed, 2 in RAM with the matrix mapped, 3 in RAM; -1 no plan printed
  int threads = 0;
  std::size_t reserve = 0, data_cap = 0;
  std::string text, refusal; ///< -v's output; the refusal's
  std::vector<dtwc::index_t> labels, medoids;
  double cost = 0;
};

/// `config` run by `run` with -v at --ram-limit `limit`.
Attempt attempt(dtwc::Config config, std::size_t limit, const Runner &run)
{
  config.ram_limit = limit;
  config.verbose = true;
  Attempt a;
  std::ostringstream out;
  auto *const previous = std::cout.rdbuf(out.rdbuf());
  try {
    const auto result = run(config);
    a.ran = true;
    a.labels = result.labels();
    a.medoids = result.medoids();
    a.cost = result.cost();
  } catch (const dtwc::InvalidInput &refused) {
    a.refusal = refused.what();
  } catch (...) {
    std::cout.rdbuf(previous);
    throw;
  }
  std::cout.rdbuf(previous);
  a.text = out.str();
  static const std::regex plan("Memory plan: budget [0-9]+ bytes, reserve ([0-9]+) bytes \\(([0-9]+) of [0-9]+ threads, "
                               "[0-9]+ bytes each for series of up to [0-9]+ time steps\\), data cap ([0-9]+) bytes: "
                               "([^\r\n]+)");
  std::smatch m;
  if (std::regex_search(a.text, m, plan)) {
    a.reserve = std::stoull(m[1].str());
    a.threads = std::stoi(m[2].str());
    a.data_cap = std::stoull(m[3].str());
    const std::string route = m[4].str();
    a.rank = route == "refused" ? 0 : route == "streamed" ? 1 : route == "in RAM, the distance matrix mapped" ? 2 : 3;
  }
  return a;
}

/// The least limit in (lo, hi] at which `holds`, where it holds at `hi` and not at `lo`: bisected.
std::size_t least(std::size_t lo, std::size_t hi, const std::function<bool(std::size_t)> &holds)
{
  while (hi - lo > 1) {
    const std::size_t mid = lo + (hi - lo) / 2;
    (holds(mid) ? hi : lo) = mid;
  }
  return hi;
}

std::filesystem::path reader_fixture(const char *name)
{
  return std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader" / name;
}

/// FastCLARA's settings for the reader fixtures: 3 clusters, 2 samples of 40, no output.
dtwc::Config clara_on(const char *file)
{
  dtwc::Config config;
  config.input = dtwc::path_to_utf8(reader_fixture(file));
  config.method = dtwc::Method::CLARA;
  config.k = 3;
  config.sample_size = 40;
  config.n_samples = 2;
  config.output.clear();
  return config;
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
  // 4 series of 100,000 values, FastPAM under a band of 10, results written, --ram-limit 100M and 10M: no kernel runs
  // beside a complete matrix, and its 80 bytes stay in RAM.
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

TEST_CASE("A --ram-limit that runs still runs when raised, its route never stepping back, with the unlimited run's "
          "clustering",
          "[thread_scratch][run][ram_limit][plan]")
{
  // Arrow IPC files FastCLARA streams (a late long series LZ4-compressed, priced by the mean; one in a file of Float32
  // read as Float64, priced by the walk's offsets where the file is mapped; two features a time step under DDTW), and
  // 200 walks in memory as FastPAM with results written, its matrix (160,800 bytes) smaller than a thread's working
  // set (600 time steps under a band of 10: the lanes' rows, about 158,000 bytes), so that a matrix put in RAM beside a
  // fitted team, not the requested one, would step back to mapped where the next thread fits. For each, every limit
  // tried: the least that runs (bisected), the least in RAM, the least on each count of threads, and 12 from the least
  // to three times the least in RAM.
  const dtwc::test_support::ScratchDirectory scratch{ "scratch_monotone" };
  int outputs = 0;
  const Runner file = [](const dtwc::Config &config) { return dtwc::run(config); };
  const Runner walks_pam = [&](const dtwc::Config &config) {
    auto written = config;
    written.output = dtwc::path_to_utf8(scratch.path / ("run" + std::to_string(outputs++))); // a fresh .dtwm each
    return dtwc::run(written, data_of(walks(200, 600)));
  };
  dtwc::Config pam;
  pam.method = dtwc::Method::PAM;
  pam.k = 3;
  pam.band = 10;
  dtwc::Config ddtw = clara_on("clara_256_ndim2.arrow");
  ddtw.variant.variant = dtwc::core::DTWVariant::DDTW;
  struct Case
  {
    const char *name;
    dtwc::Config config;
    const Runner &run;
  };
  const Case cases[] = { { "clara_long_late.arrow", clara_on("clara_long_late.arrow"), file },
                         { "clara_late_wide.arrow", clara_on("clara_late_wide.arrow"), file },
                         { "clara_256_ndim2.arrow, DDTW", ddtw, file },
                         { "200 walks in memory, FastPAM, results written", pam, walks_pam } };
  constexpr std::size_t top = std::size_t{ 1 } << 30;
  for (const Case &c : cases) {
    std::map<std::size_t, Attempt> tried;
    const auto at = [&](std::size_t limit) -> const Attempt & {
      auto it = tried.find(limit);
      if (it == tried.end()) it = tried.emplace(limit, attempt(c.config, limit, c.run)).first;
      return it->second;
    };
    const Attempt unlimited = attempt(c.config, 0, c.run);
    INFO(c.name << ", unlimited:\n" << unlimited.text << unlimited.refusal);
    REQUIRE(unlimited.ran);
    REQUIRE(at(top).rank == 3);
    REQUIRE_FALSE(at(1).ran);
    const std::size_t lowest = least(1, top, [&](std::size_t limit) { return at(limit).ran; });
    const std::size_t in_ram = least(lowest - 1, top, [&](std::size_t limit) { return at(limit).rank == 3; });
    for (int threads = 2; threads <= 4; ++threads)
      (void)least(lowest - 1, in_ram, [&](std::size_t limit) { return at(limit).ran && at(limit).threads >= threads; });
    for (int i = 0; i < 12; ++i)
      (void)at(static_cast<std::size_t>(double(lowest) * std::pow(3.0 * double(in_ram) / double(lowest), i / 11.0)));
    bool ran = false;
    int rank = -1, threads = 0;
    for (const auto &[limit, a] : tried) {
      INFO(c.name << " at --ram-limit " << limit << ":\n" << a.text << a.refusal);
      CHECK((a.ran || !ran)); // a limit that runs still runs when raised
      REQUIRE(a.rank >= 0);
      CHECK(a.rank >= rank); // the route never steps back
      CHECK(a.threads >= threads);
      rank = a.rank;
      threads = a.threads;
      ran = ran || a.ran;
      if (!a.ran) continue;
      CHECK(a.labels == unlimited.labels);
      CHECK(a.medoids == unlimited.medoids);
      CHECK(a.cost == unlimited.cost);
    }
    INFO(c.name << ": " << tried.size() << " limits tried, the least that runs " << lowest << ", in RAM from "
                << in_ram);
    CHECK(tried.size() >= 12);
  }
}

TEST_CASE("The clustering depends on neither the limit nor the threads: a stream at one data cap on the caller's 1 "
          "and 4 threads, and in RAM; FastPAM on the 2 of 4 threads a limit fits",
          "[thread_scratch][run][ram_limit][plan]")
{
#ifndef _OPENMP
  SKIP("two OpenMP workers unavailable: a sequential build has no team");
#endif
  REQUIRE(dtwc::get_max_threads() == 4);
  constexpr std::size_t top = std::size_t{ 1 } << 30;
  // clara_ragged.arrow (512 walks of 6 to 10 time steps) under a band of 4 as Float32, a copy FastCLARA streams.
  dtwc::Config ragged = clara_on("clara_ragged.arrow");
  ragged.band = 4;
  ragged.dtype = dtwc::core::Precision::Float32;
  const Runner file = [](const dtwc::Config &config) { return dtwc::run(config); };
  const Attempt in_ram = attempt(ragged, 0, file);
  INFO("in RAM:\n" << in_ram.text << in_ram.refusal);
  REQUIRE(in_ram.ran);
  const std::size_t lowest = least(1, top, [&](std::size_t limit) { return attempt(ragged, limit, file).ran; });
  const Attempt four = attempt(ragged, lowest, file);
  INFO("4 threads at " << lowest << ":\n" << four.text);
  REQUIRE(four.rank == 1); // the least limit that runs streams
  REQUIRE(four.threads == 4);
  {
    const dtwc::TeamCap caller(1); // the caller's team: one thread
    const std::size_t one_reserve = attempt(ragged, lowest, file).reserve;
    const Attempt one = attempt(ragged, four.data_cap + one_reserve, file);
    INFO("1 thread at " << four.data_cap + one_reserve << ":\n" << one.text << one.refusal);
    REQUIRE(one.ran);
    CHECK(one.rank == 1);
    CHECK(one.threads == 1);
    CHECK(one.data_cap == four.data_cap);
    CHECK(one.labels == in_ram.labels);
    CHECK(one.medoids == in_ram.medoids);
    CHECK(one.cost == in_ram.cost);
  }
  CHECK(four.labels == in_ram.labels);
  CHECK(four.medoids == in_ram.medoids);
  CHECK(four.cost == in_ram.cost);

  // 64 walks in memory, `auto` (FastPAM for 64 series where its matrix is held), nothing written: the matrix in RAM.
  dtwc::Config automatic;
  automatic.k = 3;
  automatic.output.clear();
  const Runner memory = [](const dtwc::Config &config) { return dtwc::run(config, data_of(walks(64, 50))); };
  const Attempt all = attempt(automatic, 0, memory);
  REQUIRE(all.ran);
  const std::size_t two = least(1, top, [&](std::size_t limit) {
    const Attempt a = attempt(automatic, limit, memory);
    return a.ran && a.threads >= 2;
  });
  const Attempt fitted = attempt(automatic, two, memory);
  INFO("at " << two << ":\n" << fitted.text << fitted.refusal);
  REQUIRE(fitted.ran);
  CHECK_THAT(fitted.text, ContainsSubstring("(2 of 4 threads, "));
  CHECK_THAT(fitted.text, ContainsSubstring("Auto-selected method: pam"));
  CHECK(fitted.labels == all.labels);
  CHECK(fitted.medoids == all.medoids);
  CHECK(fitted.cost == all.cost);
}

TEST_CASE("A refused --ram-limit names the budget, what it holds, the working set and the least limit, which runs",
          "[thread_scratch][run][ram_limit][plan]")
{
  // 64 walks of 50 doubles in memory as FastPAM, nothing written: the series' 25,600 bytes and the matrix's 16,640
  // (64 x 65 / 2 doubles) held in RAM beside one thread's working set at the least.
  dtwc::Config config;
  config.method = dtwc::Method::PAM;
  config.k = 3;
  config.output.clear();
  const Runner memory = [](const dtwc::Config &c) { return dtwc::run(c, data_of(walks(64, 50))); };
  const Attempt unlimited = attempt(config, 0, memory);
  REQUIRE(unlimited.ran);
  const Attempt one_byte = attempt(config, 1, memory);
  REQUIRE_FALSE(one_byte.ran);
  std::smatch m;
  REQUIRE(std::regex_search(one_byte.refusal, m, std::regex("it needs at least ([0-9]+) bytes")));
  const std::size_t needs = std::stoull(m[1].str());

  const Attempt below = attempt(config, needs - 1, memory);
  INFO(below.text << below.refusal);
  REQUIRE_FALSE(below.ran);
  CHECK(below.rank == 0);
  CHECK_THAT(below.refusal, ContainsSubstring("--ram-limit = " + std::to_string(needs - 1)
                                              + " cannot hold the series and the distance matrix (42240 bytes: 25600 "
                                                "for the series, 16640 for the distance matrix, "));
  REQUIRE(std::regex_search(below.refusal, m,
                            std::regex("beside the working set of ([0-9]+) bytes \\(1 thread x ([0-9]+) bytes: the DTW "
                                       "kernels' scratch for series of up to 50 time steps and the method's tables, "
                                       "x1\\.1\\): it needs at least ([0-9]+) bytes\\. ")));
  const std::size_t working = std::stoull(m[1].str()), per_thread = std::stoull(m[2].str());
  CHECK(working == per_thread / 10 * 11);
  CHECK(std::stoull(m[3].str()) == needs);
  CHECK(42240 + working == needs);
  // The ways out that apply: one thread already, so fewer threads are none.
  for (const char *way : { "Raise --ram-limit", "(--dtype float32)", "use --method clara" })
    CHECK_THAT(below.refusal, ContainsSubstring(way));
  CHECK_THAT(below.refusal, !ContainsSubstring("OMP_NUM_THREADS"));

  const Attempt least_limit = attempt(config, needs, memory);
  INFO(least_limit.text << least_limit.refusal);
  REQUIRE(least_limit.ran);
  CHECK(least_limit.threads == 1);
  CHECK(least_limit.labels == unlimited.labels);
  CHECK(least_limit.medoids == unlimited.medoids);
  CHECK(least_limit.cost == unlimited.cost);
}

TEST_CASE("An in-RAM FastCLARA plan holds the winner's labels that Problem::set_result copies, as the stream's does",
          "[thread_scratch][run][ram_limit][plan]")
{
  // 200 walks of 20 doubles in memory, 5 samples of 50: when the pass ends every sample's labels (200 x 8 bytes each)
  // and a block of nearest distances (the block is all 200 points) are held, and set_result copies the winner's labels
  // (200 x 8 bytes) and marks its medoids (200 / 8) beside them. The sample PAM's matrix (50 x 51 / 2 doubles) is the
  // smaller term.
  dtwc::Config config;
  config.method = dtwc::Method::CLARA;
  config.k = 3;
  config.sample_size = 50;
  config.n_samples = 5;
  config.output.clear();
  const Runner memory = [](const dtwc::Config &c) { return dtwc::run(c, data_of(walks(200, 20))); };
  const Attempt one_byte = attempt(config, 1, memory);
  REQUIRE_FALSE(one_byte.ran);
  std::smatch m;
  REQUIRE(std::regex_search(one_byte.refusal, m, std::regex("([0-9]+) for FastCLARA's sample matrix and labels")));
  const std::size_t n = 200, samples = 5, index = sizeof(dtwc::index_t);
  CHECK(std::stoull(m[1].str()) == samples * (n * index + n * sizeof(double)) + n * index + n / 8);
}

TEST_CASE("A stream's refusal names no least limit, and the stream's own refusals say their --ram-limit is the data "
          "cap",
          "[thread_scratch][run][ram_limit][plan]")
{
  // clara_long_late.arrow, FastCLARA streamed on 4 threads: the plan's refusal states what the names and labels need
  // beside the working set, which the stream's own floor (the sampled rows, a sample's PAM, a record batch) exceeds.
  const dtwc::Config config = clara_on("clara_long_late.arrow");
  const Runner file = [](const dtwc::Config &c) { return dtwc::run(c); };
  const Attempt one_byte = attempt(config, 1, file);
  INFO(one_byte.refusal);
  REQUIRE_FALSE(one_byte.ran);
  CHECK_THAT(one_byte.refusal, !ContainsSubstring("at least"));
  CHECK_THAT(one_byte.refusal, ContainsSubstring("the stream's own floor"));
  CHECK_THAT(one_byte.refusal, ContainsSubstring("(OMP_NUM_THREADS)"));
  CHECK_THAT(one_byte.refusal, !ContainsSubstring("use --method clara"));
  std::smatch m;
  REQUIRE(std::regex_search(one_byte.refusal, m, std::regex("these alone need ([0-9]+) bytes")));
  const std::size_t alone = std::stoull(m[1].str());
  // The stream's refusal, within its data cap, says what that --ram-limit is: the limit less the reserve.
  const auto names_the_cap = [](const Attempt &a, std::size_t limit) {
    INFO("at " << limit << ":\n" << a.text << a.refusal);
    CHECK_FALSE(a.ran);
    CHECK(a.rank == 1); // streamed: the stream refused
    CHECK_THAT(a.refusal, ContainsSubstring("The --ram-limit it names is the stream's data cap: --ram-limit = "
                                            + std::to_string(limit) + " less the reserve of "
                                            + std::to_string(a.reserve) + " bytes for 4 threads"));
  };
  names_the_cap(attempt(config, alone, file), alone);
  const std::size_t lowest =
    least(alone, std::size_t{ 1 } << 30, [&](std::size_t limit) { return attempt(config, limit, file).ran; });
  INFO("these alone need " << alone << "; the least that runs " << lowest);
  CHECK(lowest > alone);
  names_the_cap(attempt(config, lowest - 1, file), lowest - 1);
}

TEST_CASE("The plan holds the method's own terms: TADPole's pair flags, cache and envelopes, the hierarchical "
          "clustering's dense copy, OneBatchPAM's batch table",
          "[thread_scratch][run][ram_limit][plan]")
{
  // 256 walks of 20 doubles in memory (40,960 bytes): the least limit each refusal names runs, a byte below it does
  // not, and it holds each method's own terms beside the series, from their sizes in the code: TADPole, its results
  // written (its matrix mapped where llfio maps it), a byte a pair and its cache, 8 bytes a pair (256 x 257 / 2
  // pairs), and as it prunes these series (Standard DTW, L1, univariate float64) each one's envelope as tadpole.cpp
  // builds it (core::compute_envelope, under full DTW over the series' length): its two vectors' capacity, and the N
  // Envelopes; the hierarchical clustering N x N doubles beside the matrix, nothing written (held); OneBatchPAM N x m
  // doubles for a batch of 32.
  constexpr std::size_t n = 256, series = n * 20 * sizeof(double), pairs = n * (n + 1) / 2;
  std::size_t envelopes = n * sizeof(dtwc::core::Envelope);
  for (const auto &walk : walks(n, 20)) {
    const auto envelope = dtwc::core::compute_envelope(walk, static_cast<int>(walk.size()));
    envelopes += (envelope.upper.capacity() + envelope.lower.capacity()) * sizeof(double);
  }
  const dtwc::test_support::ScratchDirectory scratch{ "scratch_fixed_terms" };
  int outputs = 0;
  const Runner memory = [&](const dtwc::Config &c) {
    auto config = c;
    if (!config.output.empty()) config.output = dtwc::path_to_utf8(scratch.path / ("run" + std::to_string(outputs++)));
    return dtwc::run(config, data_of(walks(256, 20)));
  };
  struct Case
  {
    dtwc::Method method;
    bool written;
    std::size_t holds;
  };
  for (const Case c : { Case{ dtwc::Method::TADPole, true, series + pairs + pairs * sizeof(double) + envelopes },
                        Case{ dtwc::Method::Hierarchical, false,
                              series + n * n * sizeof(double) + pairs * sizeof(double) },
                        Case{ dtwc::Method::OneBatch, false, series + n * 32 * sizeof(double) } }) {
    dtwc::Config config;
    config.method = c.method;
    config.k = 3;
    config.batch_size = 32;
    if (!c.written) config.output.clear();
    const Attempt one_byte = attempt(config, 1, memory);
    INFO(dtwc::name_of(dtwc::method_names, c.method) << ": " << one_byte.refusal);
    std::smatch m;
    REQUIRE(std::regex_search(one_byte.refusal, m, std::regex("it needs at least ([0-9]+) bytes")));
    const std::size_t needs = std::stoull(m[1].str());
    CHECK(needs >= c.holds);
    CHECK_FALSE(attempt(config, needs - 1, memory).ran);
    const Attempt least_limit = attempt(config, needs, memory);
    INFO(least_limit.text << least_limit.refusal);
    CHECK(least_limit.ran);
  }
}

TEST_CASE("The plan prices OneBatchPAM's table for the batch one_batch_pam takes", "[thread_scratch][run][plan]")
{
  // run.cpp copies one_batch_pam.cpp's batch size (automatic_batch_size, or --batch-size; then at least k) to price
  // the table, N x (m + 8) doubles: m read back from the refusal's term, against the m one_batch_pam reports
  // (OneBatchPAMStats::batch_size), for N below 64 (m = N), N = 5000 (m = 20 x ceil(log2(N + 1))), --batch-size 100,
  // and k above the automatic m (N = 200: m = 160, raised to k = 170).
  struct Case
  {
    std::size_t n;
    dtwc::index_t batch, k;
  };
  for (const Case c : { Case{ 40, -1, 3 }, Case{ 5000, -1, 3 }, Case{ 5000, 100, 3 }, Case{ 200, -1, 170 } }) {
    dtwc::Config config;
    config.method = dtwc::Method::OneBatch;
    config.k = c.k;
    config.batch_size = c.batch;
    config.output.clear();
    const Runner memory = [&](const dtwc::Config &cfg) { return dtwc::run(cfg, data_of(walks(c.n, 2))); };
    const Attempt one_byte = attempt(config, 1, memory);
    INFO("N = " << c.n << ", --batch-size " << c.batch << ", k = " << c.k << ": " << one_byte.refusal);
    std::smatch m;
    REQUIRE(std::regex_search(one_byte.refusal, m, std::regex("([0-9]+) for OneBatchPAM's batch table")));
    const std::size_t table = std::stoull(m[1].str());
    REQUIRE(table % (c.n * sizeof(double)) == 0);
    dtwc::Problem prob{ "batch" };
    prob.set_data(data_of(walks(c.n, 2)));
    dtwc::algorithms::OneBatchPAMStats stats;
    (void)dtwc::algorithms::one_batch_pam(prob, { .n_clusters = c.k, .batch_size = c.batch }, &stats);
    CHECK(table / (c.n * sizeof(double)) - 8 == stats.batch_size);
  }
}

TEST_CASE("The plan holds TADPole's envelopes exactly where tadpole prunes", "[thread_scratch][run][plan]")
{
  // run.cpp copies tadpole.cpp's bounds_valid (Standard DTW under L1 on univariate float64 series, no missing-data
  // strategy) to price the envelopes: its refusal names them exactly where TADPoleStats::pruning_enabled says
  // algorithms::tadpole pruned, on the same settings and series, for L1 (it prunes), squared Euclidean, float32
  // series, two features a time step and a missing-data strategy (each not).
  struct Case
  {
    const char *what;
    dtwc::core::MetricType metric;
    bool f32;
    std::size_t ndim;
    dtwc::core::MissingStrategy missing;
  };
  using dtwc::core::MetricType, dtwc::core::MissingStrategy;
  for (const Case c : { Case{ "L1", MetricType::L1, false, 1, MissingStrategy::Error },
                        Case{ "squared Euclidean", MetricType::SquaredL2, false, 1, MissingStrategy::Error },
                        Case{ "float32", MetricType::L1, true, 1, MissingStrategy::Error },
                        Case{ "ndim 2", MetricType::L1, false, 2, MissingStrategy::Error },
                        Case{ "zero_cost", MetricType::L1, false, 1, MissingStrategy::ZeroCost } }) {
    const auto series = [&] {
      auto values = walks(64, 20);
      std::vector<std::string> names(values.size());
      for (std::size_t i = 0; i < names.size(); ++i) names[i] = "s" + std::to_string(i);
      if (!c.f32) return dtwc::Data(std::move(values), std::move(names), c.ndim);
      std::vector<std::vector<float>> single(values.size());
      for (std::size_t i = 0; i < values.size(); ++i)
        for (const double value : values[i]) single[i].push_back(static_cast<float>(value));
      return dtwc::Data(std::move(single), std::move(names), c.ndim);
    };
    dtwc::Config config;
    config.method = dtwc::Method::TADPole;
    config.k = 3;
    config.metric = c.metric;
    config.missing = c.missing;
    config.output.clear();
    const Attempt one_byte = attempt(config, 1, [&](const dtwc::Config &cfg) { return dtwc::run(cfg, series()); });
    dtwc::Problem prob{ "prunes" };
    dtwc::apply(config, prob);
    prob.set_data(series());
    dtwc::algorithms::TADPoleStats stats;
    (void)dtwc::algorithms::tadpole(prob, 3, 1.0, true, &stats);
    INFO(c.what << ": pruning_enabled " << stats.pruning_enabled << "; " << one_byte.refusal);
    REQUIRE_FALSE(one_byte.ran);
    CHECK(stats.pruning_enabled == (std::string(c.what) == "L1"));
    const bool priced = one_byte.refusal.find("the series' envelopes") != std::string::npos;
    CHECK(priced == stats.pruning_enabled);
  }
}
