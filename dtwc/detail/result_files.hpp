/**
 * @file result_files.hpp
 * @brief write_result_files, measuring its silhouettes for dtwc::run's report. A header of its own: Problem.hpp, which
 *        the kernels' and algorithms' translation units include, keeps the five-argument form the bindings call.
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1b)
 * @date 03 Nov 2026
 */

#pragma once

#include "../Problem.hpp"

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace dtwc::detail {

/// What write_result_files measures of the silhouettes for a run's report.
struct WriteTimes
{
  /// The process's CPU seconds so far (dtwc::run passes core::cpu_seconds): read around the silhouettes. A pointer,
  /// so that dtwc_core's write does not link core/machine.cpp, which the bindings do not.
  double (*cpu_clock)() = nullptr;
  double silhouette_wall = 0.0;          ///< the seconds scores::silhouette took
  double silhouette_cpu = 0.0;           ///< the process's CPU seconds in that time
  std::optional<double> mean_silhouette; ///< the mean of the silhouettes written, as -v prints it
  std::string skipped;                   ///< why no silhouettes were written, where none were
};

/// write_result_files(prob, directory, complete, progress, streamed_names) (Problem.hpp), which also sets `times`.
void write_result_files(Problem &prob, const std::filesystem::path &directory, bool complete, std::ostream *progress,
                        const std::vector<std::string> &streamed_names, WriteTimes *times);

} // namespace dtwc::detail
