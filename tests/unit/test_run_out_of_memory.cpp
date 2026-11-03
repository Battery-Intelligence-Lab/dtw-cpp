/**
 * @file test_run_out_of_memory.cpp
 * @brief dtwc::run where the system refuses an allocation: the run's std::bad_alloc becomes an InvalidInput naming the
 *        stage, the memory plan and the process's peak resident memory, and the process goes on.
 *
 * @details The process caps its own memory at what it holds plus 256 MiB, once a first run has started the OpenMP
 *          threads (their stacks and allocator arenas, which the cap then counts): a Job object's process memory limit
 *          on Windows (committed bytes), RLIMIT_AS on Linux (address space). The cap is on this process only and is
 *          lifted when the case ends. 12,000 one-step series as FastPAM under --ram-limit 1T, nothing written, plan
 *          their 576 MB distance matrix in RAM, which the cap refuses. macOS does not enforce RLIMIT_AS, and a
 *          sanitizer's allocator reserves its shadow memory and reports a refused allocation rather than throwing:
 *          there the cases skip (registered MAY_SKIP there). Then peak_memory() is checked against the process's own
 *          measure of its resident memory while it holds 256 MiB it has just written.
 *
 *          The second case refuses a reader's allocation (a CSV of 3 rows of 6,000,000 ones), a --dist-matrix read's
 *          (4,000 x 4,000 cells, then the matrix's 64 MB) and, read from text, the 12,000 series' matrix, each under
 *          what the process holds plus 32 MiB: each refused allocation is its run's first large one, so a smaller
 *          margin than the first case's (which leaves room for a run's work beside the matrix it refuses) keeps the
 *          files small. Each is one block of 48 MB or more, which the allocator maps afresh: what the cap counts
 *          includes the heap's freed blocks the allocator keeps, which could hold smaller ones (a CSV of 48 KB rows
 *          then read under it on Linux, 2026-11-03). In a build with Parquet a third case checks that Arrow's
 *          out-of-memory status becomes std::bad_alloc where the reader meets it.
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/distance_matrix.hpp> // peak_memory
#ifdef DTWC_HAS_PARQUET
#include <io/parquet_chunk_reader.hpp>
#endif

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <regex>
#include <string>
#include <vector>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define DTWC_TEST_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
#define DTWC_TEST_SANITIZED 1
#endif
#endif

#if (defined(_WIN32) || defined(__linux__)) && !defined(DTWC_TEST_SANITIZED)
#define DTWC_TEST_CAPS 1 // this process can cap its own memory, and its allocator throws when refused
#endif

#ifdef DTWC_TEST_CAPS
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h> // K32GetProcessMemoryInfo
#else
#include <sys/resource.h> // setrlimit
#include <unistd.h>       // sysconf
#endif

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

namespace {

/// `n` series of `steps` time steps, each its own smooth walk.
dtwc::Data walks(std::size_t n, std::size_t steps)
{
  std::vector<std::vector<double>> series(n);
  std::vector<std::string> names(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t v = 0; v < steps; ++v) series[i].push_back(std::sin(0.37 * double(i + 1) + 0.11 * double(v)));
    names[i] = "s" + std::to_string(i);
  }
  return dtwc::Data(std::move(series), std::move(names), 1);
}

/// What the cap counts, now: the bytes the process has committed on Windows (a Job's process memory limit counts
/// them), its address space on Linux (RLIMIT_AS counts it).
std::size_t capped_now()
{
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX counters{};
  REQUIRE(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                                  sizeof counters));
  return counters.PrivateUsage;
#else
  std::ifstream statm("/proc/self/statm");
  std::size_t pages = 0;
  statm >> pages;
  REQUIRE(pages > 0);
  return pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
}

/// The process's own measure of its resident memory now, which peak_memory() is checked against: the working set on
/// Windows, /proc/self/statm's resident pages on Linux.
std::size_t resident_now()
{
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  REQUIRE(K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters));
  return counters.WorkingSetSize;
#else
  std::ifstream statm("/proc/self/statm");
  std::size_t pages = 0, resident = 0;
  statm >> pages >> resident;
  REQUIRE(resident > 0);
  return resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
}

/// The process's memory capped at `bytes` (what capped_now() counts) while it lives; the cap lifted when it goes.
class MemoryCap
{
public:
  explicit MemoryCap(std::size_t bytes)
  {
#ifdef _WIN32
    job_ = CreateJobObjectW(nullptr, nullptr);
    REQUIRE(job_ != nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limit.ProcessMemoryLimit = bytes;
    REQUIRE(SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &limit, sizeof limit));
    // Nested in a job the process is already in (ctest's, Windows 8 on).
    const bool assigned = AssignProcessToJobObject(job_, GetCurrentProcess()) != 0;
    const DWORD error = assigned ? 0 : GetLastError();
    INFO("AssignProcessToJobObject: error " << error);
    REQUIRE(assigned);
#else
    REQUIRE(getrlimit(RLIMIT_AS, &previous_) == 0);
    rlimit capped = previous_;
    capped.rlim_cur = bytes; // the hard limit stays, so the soft one can be raised back
    REQUIRE(setrlimit(RLIMIT_AS, &capped) == 0);
#endif
  }
  ~MemoryCap()
  {
#ifdef _WIN32
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION none{}; // no limit: the process stays in the job, which no longer caps it
    SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &none, sizeof none);
    CloseHandle(job_);
#else
    setrlimit(RLIMIT_AS, &previous_);
#endif
  }
  MemoryCap(const MemoryCap &) = delete;
  MemoryCap &operator=(const MemoryCap &) = delete;

private:
#ifdef _WIN32
  HANDLE job_ = nullptr;
#else
  rlimit previous_{};
#endif
};

/// The InvalidInput `run()` throws under a cap of what the process holds plus `margin` (a run that runs, or throws
/// anything else, fails the case); then a small run under the same cap, which runs: nothing of the refused one is held.
template <typename Run>
std::string refused_under_cap(std::size_t margin, Run &&run)
{
  const std::size_t cap = capped_now() + margin;
  std::string message;
  {
    const MemoryCap capped(cap);
    try {
      (void)run();
      FAIL("a run the cap of " << cap << " bytes refuses ran");
    } catch (const dtwc::InvalidInput &e) {
      message = e.what();
    }
    dtwc::Config small;
    small.method = dtwc::Method::PAM;
    small.k = 3;
    small.output.clear();
    CHECK(dtwc::run(small, walks(256, 20)).labels().size() == 256);
  }
  std::cout << "capped at " << cap << " bytes: dtwc::run threw InvalidInput: " << message << "\n";
  return message;
}

/// A CSV of `rows` lines of `columns` ones; a square one's diagonal 0 (`zero_diagonal`), a distance matrix.
void write_ones(const std::filesystem::path &path, std::size_t rows, std::size_t columns, bool zero_diagonal)
{
  std::string row;
  for (std::size_t c = 0; c < columns; ++c) row += c ? ",1" : "1"; // column c's digit at 2c
  row += '\n';
  std::ofstream out(path, std::ios::binary);
  for (std::size_t r = 0; r < rows; ++r) {
    if (zero_diagonal) row[2 * r] = '0';
    out << row;
    if (zero_diagonal) row[2 * r] = '1';
  }
}

} // namespace
#endif

TEST_CASE("A run the system refuses memory reports its stage, its memory plan and the process's peak, and the process "
          "goes on",
          "[run][ram_limit][out_of_memory]")
{
#if defined(DTWC_TEST_SANITIZED)
  SKIP("a sanitizer's allocator reserves its shadow memory and reports a refused allocation rather than throwing");
#elif !defined(DTWC_TEST_CAPS)
  SKIP("this process cannot cap its own memory here: RLIMIT_AS is not enforced (macOS)");
#else
  dtwc::Config config;
  config.method = dtwc::Method::PAM;
  config.k = 3;
  config.output.clear();                     // nothing written: the matrix stays in RAM
  config.ram_limit = std::size_t{ 1 } << 40; // 1T: the plan holds it all in RAM
  // A first run starts the OpenMP threads, their stacks and allocator arenas, before the cap counts them.
  REQUIRE(dtwc::run(config, walks(1000, 20)).labels().size() == 1000);

  dtwc::Data many = walks(12000, 1); // made before the cap: 12,000 x 12,001 / 2 doubles of matrix, 576,048,000 bytes
  const std::size_t held = capped_now(), cap = held + (std::size_t{ 256 } << 20);
  std::string message;
  {
    const MemoryCap capped(cap);
    auto verbose = config;
    verbose.verbose = true;
    try {
      (void)dtwc::run(verbose, std::move(many));
      FAIL("a run whose 576 MB matrix exceeds the cap of " << cap << " bytes ran");
    } catch (const dtwc::InvalidInput &e) {
      message = e.what();
    }
    // The process goes on, the cap still in place: nothing of the refused run is held.
    CHECK(dtwc::run(config, walks(256, 20)).labels().size() == 256);
  }
  std::cout << "capped at " << cap << " bytes (" << held << " held + 256 MiB); dtwc::run threw InvalidInput: "
            << message << "\n";
  CHECK_THAT(message, StartsWith("cluster stage: out of memory; plan: budget 1099511627776 bytes, reserve "));
  const auto budget = message.find("budget"), data_cap = message.find("data cap"), peak = message.find("peak");
  CHECK(budget < data_cap);
  CHECK(data_cap < peak);
  CHECK(peak != std::string::npos);
  std::smatch m;
  REQUIRE(std::regex_search(message, m, std::regex("the process peaked at ([0-9]+) bytes resident")));
  CHECK(std::stoull(m[1].str()) > 0);
  // What the plan holds in RAM, the series' 96,000 bytes and the matrix's 576,048,000; the matrix the load, which the
  // first way out halves.
  CHECK_THAT(message, ContainsSubstring("; the plan holds 576144000 bytes in RAM; "));
  CHECK_THAT(message, ContainsSubstring("ways out: --matrix-dtype float32, "));

  // peak_memory() against the process's own measure while it holds 256 MiB it has just written, a page at a time.
  const std::size_t bytes = std::size_t{ 256 } << 20;
  const auto block = std::make_unique_for_overwrite<char[]>(bytes);
  volatile char *const page = block.get();
  for (std::size_t i = 0; i < bytes; i += 4096) page[i] = 1;
  const std::size_t resident = resident_now(), peak_bytes = dtwc::core::peak_memory();
  std::cout << "peak_memory() " << peak_bytes << " bytes; resident now " << resident << " bytes (" << bytes
            << " just written)\n";
  CHECK(resident >= bytes);
  CHECK(peak_bytes >= resident - resident / 5);
  CHECK(peak_bytes <= resident + resident / 5);
#endif
}

TEST_CASE("A run the system refuses memory at a text file's read, at a --dist-matrix read or at its matrix reports the "
          "stage, and ways out true of the run",
          "[run][ram_limit][out_of_memory]")
{
#if defined(DTWC_TEST_SANITIZED)
  SKIP("a sanitizer's allocator reserves its shadow memory and reports a refused allocation rather than throwing");
#elif !defined(DTWC_TEST_CAPS)
  SKIP("this process cannot cap its own memory here: RLIMIT_AS is not enforced (macOS)");
#else
  const dtwc::test_support::ScratchDirectory scratch{ "run_out_of_memory" };
  dtwc::Config config;
  config.method = dtwc::Method::PAM;
  config.k = 3;
  config.output.clear();
  config.ram_limit = std::size_t{ 1 } << 40;
  REQUIRE(dtwc::run(config, walks(1000, 20)).labels().size() == 1000); // the OpenMP threads, before any cap
  constexpr std::size_t margin = std::size_t{ 32 } << 20;

  // The reader's refusal, before the plan, which a text file's series get once read: a row of 6,000,000 values, 48 MB
  // (band 1: should the cap not bind, the run ends at once, and the case fails on its running).
  const auto ones = scratch.path / "ones.csv";
  write_ones(ones, 3, 6000000, false);
  auto read = config;
  read.input = dtwc::path_to_utf8(ones);
  read.band = 1;
  const std::string at_read = refused_under_cap(margin, [&] { return dtwc::run(read); });
  CHECK_THAT(at_read, StartsWith("read stage: out of memory; plan: none yet; the process peaked at "));
  CHECK_THAT(at_read, ContainsSubstring("ways out: --dtype float32, or the file converted to Arrow IPC"));

  // A --dist-matrix read: 16,000,000 cells parsed beside the matrix the plan holds, 64,016,000 bytes, and the 4,000
  // series' 32,000.
  const auto matrix = scratch.path / "matrix.csv";
  write_ones(matrix, 4000, 4000, true);
  auto supplied = config;
  supplied.dist_matrix = dtwc::path_to_utf8(matrix);
  const std::string at_storage = refused_under_cap(margin, [&] { return dtwc::run(supplied, walks(4000, 1)); });
  CHECK_THAT(at_storage, StartsWith("storage stage: out of memory; plan: budget 1099511627776 bytes, reserve "));
  CHECK_THAT(at_storage, ContainsSubstring("; the plan holds 64048000 bytes in RAM; "));
  CHECK_THAT(at_storage, ContainsSubstring("ways out: the distances computed without --dist-matrix"));

  // The 12,000 one-step series read from text: their matrix is the load, so the ways out halve it, not the series.
  const auto steps = scratch.path / "steps.csv";
  {
    std::ofstream out(steps, std::ios::binary);
    for (std::size_t i = 0; i < 12000; ++i) out << i % 97 << '\n';
  }
  auto text = config;
  text.input = dtwc::path_to_utf8(steps);
  const std::string matrix_bound = refused_under_cap(margin, [&] { return dtwc::run(text); });
  CHECK_THAT(matrix_bound, StartsWith("cluster stage: out of memory; plan: budget 1099511627776 bytes, reserve "));
  CHECK_THAT(matrix_bound, ContainsSubstring("ways out: --matrix-dtype float32, "));
  CHECK_THAT(matrix_bound, ContainsSubstring("--method clara"));
  CHECK_THAT(matrix_bound, !ContainsSubstring("--dtype float32"));
  CHECK_THAT(matrix_bound, !ContainsSubstring("OMP_NUM_THREADS"));
#endif
}

#ifdef DTWC_HAS_PARQUET
TEST_CASE("Arrow's out-of-memory status is std::bad_alloc where the Parquet reader meets it", "[run][out_of_memory]")
{
  // The system's refusal reaches dtwc::run as one, to report with its stage, not as a broken file.
  CHECK_THROWS_AS(dtwc::io::detail::check_arrow_chunk(arrow::Status::OutOfMemory("realloc of size 32768 failed"),
                                                      "read_rows"),
                  std::bad_alloc);
  CHECK_THROWS_AS(dtwc::io::detail::check_arrow_chunk(arrow::Status::IOError("short read"), "read_rows"),
                  dtwc::IOError);
}
#endif
