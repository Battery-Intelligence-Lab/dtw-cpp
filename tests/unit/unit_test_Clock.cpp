/*
 * unit_test_Clock.cpp
 *
 * Unit test file for time Clock class
 *  Created on: 16 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sstream>
#include <thread>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("Clock class functionality", "[Clock]")
{
  Clock clk;

  SECTION("Clock starts at zero or close to zero")
  {
    REQUIRE(clk.duration() >= 0.0);
  }

  SECTION("Duration increases over time")
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(clk.duration() > 0.0);
  }

  SECTION("Start time is consistent")
  {
    auto start_time = clk.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(clk.start() == start_time);
  }

  SECTION("Now time is later than start time")
  {
    auto now_time = clk.now();
    REQUIRE(now_time >= clk.start());
  }

  SECTION("Print duration outputs correct format")
  {
    std::ostringstream os;
    Clock::print_duration(os, 150.0); // 2 min 30 sec
    REQUIRE(os.str() == "2:30 min:sec\n");
  }

  SECTION("<< operator outputs correct format")
  {
    std::ostringstream os;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    os << clk;
    auto str = os.str();
    REQUIRE(str.find("min:sec") != std::string::npos);
  }
}