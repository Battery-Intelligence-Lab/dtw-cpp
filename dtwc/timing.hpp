// Vk 2022.03.01
// Timing functions

#pragma once

#include <ctime>
#include <iostream>
#include <cmath>
#include <chrono>

namespace dtwc {
struct Clock
{
  std::chrono::time_point<std::chrono::steady_clock> tstart{ std::chrono::steady_clock::now() };
  Clock() = default;
  auto now() const { return std::chrono::steady_clock::now(); }
  auto start() const { return tstart; }
  double duration() const
  {
    std::chrono::duration<double> elapsed_seconds = now() - start();
    return elapsed_seconds.count();
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
