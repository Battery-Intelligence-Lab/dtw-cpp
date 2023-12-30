/*
 * test_util.hpp
 *
 * Auxillary functions for testing
 *  Created on: 29 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "../dtwc/settings.hpp"

#include <vector>
#include <string>
#include <set>
#include <random>
#include <filesystem>
#include <iostream>

namespace dtwc::test_util {

template <typename data_t>
std::vector<std::vector<data_t>> get_random_data(int N_data, int L_data)
{
  std::vector<std::vector<double>> random_data;
  std::uniform_int_distribution<> dis(0, L_data);


  for (int i = 0; i < N_data; ++i) {
    int innerSize = dis(randGenerator); // Random size for the inner vector
    std::vector<double> innerVector;

    for (int j = 0; j < innerSize; ++j)
      innerVector.push_back(dis(randGenerator)); // Generate random number

    random_data.push_back(std::move(innerVector));
  }

  return random_data;
}

inline std::vector<std::string> get_random_names(int N_names, int N_length = 10)
{
  std::uniform_int_distribution<> distString(97, 122);

  std::vector<std::string> uniqueNames;
  std::set<std::string> tempNames;

  while (tempNames.size() < static_cast<size_t>(N_names)) {
    std::string str;
    for (int j = 0; j < N_length; ++j) {
      char randomChar = static_cast<char>(distString(randGenerator));
      str.push_back(randomChar);
    }
    tempNames.insert(str);
  }

  std::move(tempNames.begin(), tempNames.end(), std::back_inserter(uniqueNames));

  return uniqueNames;
}

template <typename data_t>
inline void write_data_to_folder(std::string folder_name, const std::vector<std::vector<data_t>> &random_data, const std::vector<std::string> &random_names)
{
  // write the files
  fs::create_directory(folder_name);
  char delimiter{ ',' };
  std::string extension{ ".csv" };

  if (folder_name == "TSV") {
    char delimiter{ '\t' };
    std::string extension{ ".tsv" };
  }

  for (int i = 0; i < random_data.size(); ++i) {
    std::ofstream out(folder_name + "/" + random_names[i] + extension, std::ios_base::out);

    for (size_t j{}; j < random_data[i].size(); ++j)
      out << j << delimiter << random_data[i][j] << '\n';

    out.close();
  }
}

} // namespace dtwc::test_util