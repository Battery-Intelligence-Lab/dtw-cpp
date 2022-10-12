// Vk 2022.03.01
// Timing functions

#pragma once

#include <ctime>
#include <iostream>
#include <cmath>

namespace dtwc {
struct Clock
{
  std::clock_t tstart{ std::clock() };
  Clock() = default;
  auto now() const { return std::clock(); }
  auto start() const { return tstart; }
  double duration() const { return (now() - start()) / static_cast<double>(CLOCKS_PER_SEC); }
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
