/**
 * @file f17_checkpoint_writer.cc
 * @brief Production-serializer fixtures for the F17 real-CLI resume gate.
 */

#include <dtwc.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

dtwc::core::ClusteringResult make_result(const std::string &mode)
{
  dtwc::core::ClusteringResult result;
  result.medoid_indices = {0, 9, 18};
  result.labels.resize(27);
  for (int i = 0; i < 27; ++i)
    result.labels[static_cast<std::size_t>(i)] = i / 9;
  result.total_cost = 1650.0;
  result.iterations = 41;
  result.converged = false;

  if (mode == "valid") {
    return result;
  } else if (mode == "wrong-n") {
    result.labels.pop_back();
  } else if (mode == "wrong-k") {
    result.medoid_indices.pop_back();
  } else if (mode == "bad-label") {
    result.labels[0] = 3;
  } else if (mode == "bad-medoid") {
    result.medoid_indices[0] = 27;
  } else if (mode == "duplicate-medoid") {
    result.medoid_indices[1] = result.medoid_indices[0];
  } else if (mode == "negative-iterations") {
    result.iterations = -1;
  } else if (mode == "nonfinite-cost") {
    result.total_cost = std::numeric_limits<double>::infinity();
  } else {
    throw std::invalid_argument("unknown fixture mode: " + mode);
  }
  return result;
}

bool same_result(const dtwc::core::ClusteringResult &lhs,
                 const dtwc::core::ClusteringResult &rhs)
{
  const bool same_cost =
    (std::isinf(lhs.total_cost) && std::isinf(rhs.total_cost)
     && std::signbit(lhs.total_cost) == std::signbit(rhs.total_cost))
    || lhs.total_cost == rhs.total_cost;
  return lhs.medoid_indices == rhs.medoid_indices
    && lhs.labels == rhs.labels
    && same_cost
    && lhs.iterations == rhs.iterations
    && lhs.converged == rhs.converged;
}

} // namespace

int main(int argc, char **argv)
{
  try {
    if (argc != 3)
      throw std::invalid_argument(
        "usage: f17_checkpoint_writer <output.bin> <mode>");

    const fs::path output{argv[1]};
    const std::string mode{argv[2]};
    if (output.has_parent_path())
      fs::create_directories(output.parent_path());

    if (mode == "malformed") {
      std::ofstream bytes(output, std::ios::binary | std::ios::trunc);
      bytes << "not-a-checkpoint";
      if (!bytes)
        throw std::runtime_error("failed to write malformed fixture");
      bytes.close();
    } else {
      const auto expected = make_result(mode);
      dtwc::save_binary_checkpoint(expected, output);

      dtwc::core::ClusteringResult loaded;
      if (!dtwc::load_binary_checkpoint(loaded, output))
        throw std::runtime_error("production loader rejected written fixture");
      if (!same_result(expected, loaded))
        throw std::runtime_error("production binary round-trip changed a field");
    }

    // A replay must leave the source artifact untouched. Pin an old timestamp
    // so an accidental rewrite is observable even when it reproduces the same
    // bytes and SHA-256.
    fs::last_write_time(
      output, fs::file_time_type::clock::now() - std::chrono::hours(24));

    std::cout << "F17_CHECKPOINT_FIXTURE writer=production_serializer mode="
              << mode << " bytes=" << fs::file_size(output) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "F17_CHECKPOINT_FIXTURE_ERROR: " << error.what() << '\n';
    return 1;
  }
}
