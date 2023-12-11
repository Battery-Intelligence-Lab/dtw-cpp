#include "dtwc.hpp"
#include "examples.hpp"
#include <iostream>
#include <string>
#include <CLI/CLI.hpp>


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

int main(int argc, char **argv)
{
  auto app_description = "A C++ library for fast Dynamic Time Wrapping Clustering";
  auto default_name = "DTWC++_results";


  // Input parameters:
  std::string Nc_str;
  std::string inputPath{ "." };
  std::string outPath{ "." };


  CLI::App app{ app_description };

  //  app.add_option("-Nc", Nc_str, "Number of clusters");
  app.add_option("--Nc,--clusters,--number_of_clusters", Nc_str, "Number range in the format i..j or single number i");
  CLI11_PARSE(app, argc, argv);

  auto Nc = str_to_range(Nc_str);

  for (auto nc : Nc)
    std::cout << nc << ", ";

  std::cout << std::endl;
}
