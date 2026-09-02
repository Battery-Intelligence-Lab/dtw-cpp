/**
 * @file f14_result_save_writer.cc
 * @brief Live native Result::save producer for the F14 public CSV gate.
 */

#include <dtwc.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
  try {
    if (argc != 3) {
      throw std::invalid_argument(
        "usage: f14_result_save_writer <conformance.csv> <output-directory>");
    }

    const fs::path input{argv[1]};
    const fs::path output{argv[2]};
    if (!fs::is_regular_file(input))
      throw std::runtime_error("conformance input is not a regular file");

    dtwc::device("cpu");
    const auto dataset = dtwc::load(input, 0, 0, ',', "conformance");
    const auto result = dtwc::cluster(dataset, 3, "pam", 3, "cpu", 100);
    if (result.labels().size() != 27 || result.medoids().size() != 3
        || result.device() != "cpu") {
      throw std::runtime_error(
        "native conformance clustering did not reach the registered result");
    }

    result.save(output);
    const auto matrix = output / "conformance_distance_matrix.csv";
    if (!fs::is_regular_file(matrix))
      throw std::runtime_error("Result::save did not create its matrix file");

    std::cout
      << "F14_RESULT_SAVE subject=native_result labels=27 medoids=3 device=cpu\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "F14_RESULT_SAVE_ERROR: " << error.what() << '\n';
    return 1;
  }
}
