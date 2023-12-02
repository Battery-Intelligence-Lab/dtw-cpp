/*
 * timing.hpp
 *
 * Timing functions
 *
 * Created on: 01 Mar 2022
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include <ctime>
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
  double duration() const
  {
    std::chrono::duration<double> elapsed_seconds = now() - start();
    return elapsed_seconds.count();
  }

  static void print_duration(std::ostream &ofs, double duration)
  {
    ofs << std::floor(duration / 60) << ":"
        << duration - std::floor(duration / 60) * 60
        << " min:sec\n";
  }
};

inline std::ostream &operator<<(std::ostream &ofs, const Clock &clk)
{
  const auto duration = clk.duration();
  ofs << std::floor(duration / 60) << ":"
      << duration - std::floor(duration / 60) * 60
      << " min:sec";

  return ofs;
}
} // namespace dtwc
