/**
 * @file report.hpp
 * @brief The run report: dtwc::run's plain-text account of a run, Result::report()'s text and `<name>_report.txt`
 *        (Volkan 2026-11-02: how long it took, on which computer, the utilisation, each stage's time, the stopping
 *        criterion).
 *
 * @details run.cpp sets a RunReport's plain values as the run goes: each stage's wall and CPU time and the process's
 * peak resident memory at its end, the memory plan's figures, the input's and the result's; render() prints them in
 * eight numbered sections of plain ASCII, at most 100 columns. Nothing time-dependent reaches another output.
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1b)
 * @date 03 Nov 2026
 */

#pragma once

#include "../config.hpp"
#include "../core/machine.hpp"

#include <chrono>
#include <cstddef>
#include <ctime>
#include <optional>
#include <string>
#include <vector>

namespace dtwc::detail {

/// A run's report, filled by dtwc::run (cli/run.cpp) and printed by render().
struct RunReport
{
  /// One row of the stage table: a stage of run.cpp's one list (check, plan, read, storage, fill, cluster, write).
  struct Stage
  {
    const char *name = "";
    double begin = 0.0, end = 0.0; ///< seconds since the run started
    double cpu = 0.0;              ///< the process's CPU seconds from begin to end
    int threads = 0;               ///< the threads the memory plan gave the stage; 0: no plan yet
    std::size_t peak = 0;          ///< the process's peak resident bytes at its end (core::peak_memory())
    std::string note;
  };

  /// The memory plan, as run.cpp's MemoryPlan holds it.
  struct Plan
  {
    std::size_t budget = 0, reserve = 0, data_cap = 0, held = 0, per_thread = 0, length = 0;
    int threads = 1, requested = 1;
    const char *route = ""; ///< "in RAM", "in RAM, the distance matrix mapped", "streamed" or "refused"
  };

  /// The series, as the run read them, or as a stream's plan knew them.
  struct Input
  {
    const char *format = ""; ///< "text", "Parquet", "Arrow IPC" or "in memory"
    std::size_t series = 0, values = 0, ndim = 1, bytes = 0;
    std::size_t shortest = 0, longest = 0; ///< in time steps; longest 0: not known (a stream's)
    bool streamed = false;
  };

  /// What the clustering returned.
  struct Clustering
  {
    Method method{};
    const char *label = ""; ///< the method's name in -v's progress lines
    double cost = 0.0;
    int iterations = 0;
    bool converged = false;
    std::vector<std::size_t> sizes; ///< each cluster's points
    std::vector<index_t> medoids;
  };

  /// The silhouettes the write computed (detail::WriteTimes): their wall and CPU seconds, part of the write row's,
  /// which render() prints as a row of their own after it, and their mean, or why there is none.
  struct Silhouettes
  {
    double wall = 0.0, cpu = 0.0;
    std::optional<double> mean;
    std::string why_not;
  };

  /// The run's start: its clocks, and the machine's facts (the RAM available now).
  explicit RunReport(const Config &config);

  /// Close the open row, run on `threads` threads (0: none planned yet), and open `name`'s.
  void begin(const char *name, int threads);
  /// Close the open row, run on `threads` threads: the run's end, or its throw.
  void close(int threads);

  /// The report's text: completed, or failed in the last row's stage where `error_kind` is set.
  std::string render() const;

  const Config &config;
  Method method = Method::Auto; ///< the method asked for, then the one that runs once the run resolves it
  std::time_t started = 0, finished = 0;
  std::chrono::steady_clock::time_point start;
  double cpu_start = 0.0, cpu_end = 0.0;
  double wall = 0.0;    ///< seconds from the start to the last close
  std::size_t peak = 0; ///< the process's peak resident bytes at the last close
  bool open = false;    ///< the last row is still running
  core::MachineFacts machine;
  std::optional<Plan> plan;
  std::optional<Input> input;
  std::vector<Stage> stages;
  std::optional<Clustering> clustering;
  std::optional<Silhouettes> silhouettes;
  std::vector<std::string> notes;
  std::string error_kind, error_text; ///< a failed run's typed error and its text
};

/// The command line that repeats `config` with its method as `method`: dtwc_cl and every key whose value is not its
/// default (to_config_text, the one key table), a flag as `--key`.
std::string command_of(const Config &config, Method method);

} // namespace dtwc::detail
