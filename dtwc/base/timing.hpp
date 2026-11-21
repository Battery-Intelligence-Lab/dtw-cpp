/**
 * @file timing.hpp
 * @brief High-resolution wall-clock timer.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 01 Mar 2022
 */

#pragma once

#include <iostream>
#include <cmath>
#include <chrono>

namespace dtwc {

struct Clock
{
  std::chrono::time_point<std::chrono::high_resolution_clock> tstart{ std::chrono::high_resolution_clock::now() };
  Clock() = default;
  auto now() const { return std::chrono::high_resolution_clock::now(); }
  auto start() const { return tstart; }

  /// Elapsed time in seconds since construction.
  double duration() const
  {
    return std::chrono::duration<double>(now() - start()).count();
  }

  static void print_duration(std::ostream &ofs, double duration)
  {
    const double mins = std::floor(duration / 60);
    ofs << mins << ":" << duration - mins * 60 << " min:sec\n";
  }
};

inline std::ostream &operator<<(std::ostream &ofs, const Clock &clk)
{
  const double d = clk.duration(), mins = std::floor(d / 60);
  return ofs << mins << ":" << d - mins * 60 << " min:sec";
}
} // namespace dtwc
