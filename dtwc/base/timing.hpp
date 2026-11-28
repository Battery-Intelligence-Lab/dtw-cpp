/**
 * @file timing.hpp
 * @brief High-resolution wall-clock timer.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 01 Mar 2022
 */

#pragma once

#include <chrono>
#include <cmath>
#include <ostream>

namespace dtwc {

/// Wall-clock timer based on std::chrono::steady_clock.
struct Clock
{
  std::chrono::time_point<std::chrono::steady_clock> tstart{ std::chrono::steady_clock::now() };
  Clock() = default;
  /// Return the current steady-clock time point.
  auto now() const { return std::chrono::steady_clock::now(); }
  /// Return the start time point captured at construction.
  auto start() const { return tstart; }

  /// Elapsed time in seconds since construction.
  double duration() const
  {
    return std::chrono::duration<double>(now() - start()).count();
  }

  /// Format @p duration in min:sec to an output stream.
  static void write_duration(std::ostream &ofs, double duration)
  {
    const double mins = std::floor(duration / 60);
    ofs << mins << ":" << duration - mins * 60 << " min:sec";
  }

  /// Format @p duration in min:sec followed by a newline to an output stream.
  static void print_duration(std::ostream &ofs, double duration)
  {
    write_duration(ofs, duration);
    ofs << '\n';
  }
};

/// Stream the elapsed duration of @p clk formatted as min:sec.
inline std::ostream &operator<<(std::ostream &ofs, const Clock &clk)
{
  Clock::write_duration(ofs, clk.duration());
  return ofs;
}
} // namespace dtwc
