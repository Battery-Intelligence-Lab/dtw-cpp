/**
 * @file timing.hpp
 * @brief Timing functions
 *
 * Contains the definition and implementation of a Clock structure for timing purposes.
 *
 * @author Volkan Kumtepeli, Becky Perriment
 * @date 01 Mar 2022
 */

#pragma once

#include <ctime>
#include <iostream>
#include <cmath>
#include <chrono>

namespace dtwc {

/**
 * @brief Structure representing a high-resolution clock.
 *
 * Clock structure to measure time intervals with high resolution.
 */
struct Clock
{
  std::chrono::time_point<std::chrono::high_resolution_clock> tstart{ std::chrono::high_resolution_clock::now() };
  Clock() = default;
  auto now() const { return std::chrono::high_resolution_clock::now(); }
  auto start() const { return tstart; }

  /**
   * @brief Calculates the duration since the clock started.
   * @return Duration in seconds.
   */
  double duration() const
  {
    std::chrono::duration<double> elapsed_seconds = now() - start();
    return elapsed_seconds.count();
  }

  /**
   * @brief Prints the duration in minutes and seconds format.
   * @param ofs Output stream to print the duration.
   * @param duration Duration in seconds.
   */
  static void print_duration(std::ostream &ofs, double duration)
  {
    ofs << std::floor(duration / 60) << ":"
        << duration - std::floor(duration / 60) * 60
        << " min:sec\n";
  }
};

/**
 * @brief Overloads the << operator for the Clock structure.
 * @param ofs Output stream.
 * @param clk Clock instance.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &ofs, const Clock &clk)
{
  const auto duration = clk.duration();
  ofs << std::floor(duration / 60) << ":"
      << duration - std::floor(duration / 60) * 60
      << " min:sec";

  return ofs;
}
} // namespace dtwc
