/*
 * dtwc_cl.hpp
 *
 * Command line interface functions for DTWC++
 *
 * Created on: 11 Dec 2023
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "dtwc.hpp"

#include <iostream>
#include <string>

namespace dtwc::cli_util {

dtwc::Range str_to_range(std::string str)
{
  dtwc::Range range{};
  try {
    size_t pos = str.find("..");
    if (pos != std::string::npos) {
      const int start = std::stoi(str.substr(0, pos));
      const int end = std::stoi(str.substr(pos + 2)) + 1;
      range = dtwc::Range(start, end);
    } else {
      const int number = std::stoi(str);
      range = dtwc::Range(number, number + 1);
    }

  } catch (const std::exception &e) {
    std::cerr << "Error processing input: " << e.what() << std::endl;
  }

  return range;
}

}