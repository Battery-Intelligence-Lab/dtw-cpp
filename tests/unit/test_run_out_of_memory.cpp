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
 *          there the case skips (registered MAY_SKIP there). Then peak_memory() is checked against the process's own
 *          measure of its resident memory while it holds 256 MiB it has just written.
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/distance_matrix.hpp> // peak_memory

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
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
#include <fstream>
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
  CHECK_THAT(message, ContainsSubstring("ways out: "));

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
