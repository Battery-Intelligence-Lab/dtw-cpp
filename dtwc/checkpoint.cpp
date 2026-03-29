/**
 * @file checkpoint.cpp
 * @brief Implementation of checkpoint save/load for distance matrix computation.
 *
 * @date 29 Mar 2026
 */

#include "checkpoint.hpp"
#include "Problem.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dtwc {

namespace fs = std::filesystem;

namespace {

/// Get current UTC timestamp as ISO 8601 string.
std::string current_timestamp()
{
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  gmtime_s(&tm_buf, &time_t_now);
#else
  gmtime_r(&time_t_now, &tm_buf);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}

/// Convert DTWVariant enum to string for metadata.
std::string variant_to_string(core::DTWVariant v)
{
  switch (v) {
  case core::DTWVariant::Standard: return "Standard";
  case core::DTWVariant::DDTW: return "DDTW";
  case core::DTWVariant::WDTW: return "WDTW";
  case core::DTWVariant::ADTW: return "ADTW";
  case core::DTWVariant::SoftDTW: return "SoftDTW";
  default: return "Unknown";
  }
}

/// Parse DTWVariant from string. Returns Standard if unrecognized.
core::DTWVariant string_to_variant(const std::string &s)
{
  if (s == "DDTW") return core::DTWVariant::DDTW;
  if (s == "WDTW") return core::DTWVariant::WDTW;
  if (s == "ADTW") return core::DTWVariant::ADTW;
  if (s == "SoftDTW") return core::DTWVariant::SoftDTW;
  return core::DTWVariant::Standard;
}

/// Write metadata to a key=value text file.
void write_metadata(const fs::path &path, size_t n, int band,
                    core::DTWVariant variant, size_t pairs_computed)
{
  std::ofstream file(path);
  if (!file.good())
    throw std::runtime_error("Cannot open metadata file for writing: " + path.string());

  file << "n=" << n << '\n';
  file << "band=" << band << '\n';
  file << "variant=" << variant_to_string(variant) << '\n';
  file << "pairs_computed=" << pairs_computed << '\n';
  file << "timestamp=" << current_timestamp() << '\n';

  if (!file.good())
    throw std::runtime_error("Write error on metadata file: " + path.string());
}

/// Read metadata from a key=value text file.
/// Returns false if the file cannot be read.
struct CheckpointMetadata {
  size_t n = 0;
  int band = -1;
  std::string variant_str;
  size_t pairs_computed = 0;
  std::string timestamp;
};

bool read_metadata(const fs::path &path, CheckpointMetadata &meta)
{
  std::ifstream file(path);
  if (!file.good())
    return false;

  std::string line;
  while (std::getline(file, line)) {
    // Strip trailing \r for Windows line endings
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;

    auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) continue;

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);

    if (key == "n") meta.n = std::stoull(value);
    else if (key == "band") meta.band = std::stoi(value);
    else if (key == "variant") meta.variant_str = value;
    else if (key == "pairs_computed") meta.pairs_computed = std::stoull(value);
    else if (key == "timestamp") meta.timestamp = value;
  }

  return meta.n > 0;
}

} // anonymous namespace


void save_checkpoint(const Problem &prob, const std::string &path)
{
  fs::path dir(path);

  // Create directory if it does not exist
  if (!fs::exists(dir))
    fs::create_directories(dir);

  const auto &dm = prob.distance_matrix();
  const size_t n = dm.size();

  // Write distance matrix CSV
  dm.write_csv(dir / "distances.csv");

  // Write metadata
  write_metadata(dir / "metadata.txt",
                 n,
                 prob.band,
                 prob.variant_params.variant,
                 dm.count_computed());

  std::cout << "Checkpoint saved to " << dir.string()
            << " (" << dm.count_computed() << "/" << (n * n) << " entries computed)" << std::endl;
}


bool load_checkpoint(Problem &prob, const std::string &path)
{
  fs::path dir(path);

  // Check that both files exist
  fs::path csv_path = dir / "distances.csv";
  fs::path meta_path = dir / "metadata.txt";

  if (!fs::exists(csv_path) || !fs::exists(meta_path)) {
    std::cout << "No checkpoint found at " << dir.string() << std::endl;
    return false;
  }

  // Read and validate metadata
  CheckpointMetadata meta;
  if (!read_metadata(meta_path, meta)) {
    std::cout << "Checkpoint metadata is invalid at " << dir.string() << std::endl;
    return false;
  }

  // Validate dimension matches the Problem's data
  if (meta.n != prob.size()) {
    std::cout << "Checkpoint dimension mismatch: checkpoint has n=" << meta.n
              << " but Problem has n=" << prob.size() << std::endl;
    return false;
  }

  // Load the distance matrix CSV
  try {
    prob.distance_matrix().read_csv(csv_path);
  } catch (const std::exception &e) {
    std::cout << "Failed to read checkpoint distances: " << e.what() << std::endl;
    return false;
  }

  // Check if all pairs are computed
  const auto &dm = prob.distance_matrix();
  if (dm.all_computed()) {
    prob.set_distance_matrix_filled(true);
    std::cout << "Checkpoint fully loaded from " << dir.string()
              << " (all " << (meta.n * meta.n) << " entries computed)" << std::endl;
  } else {
    prob.set_distance_matrix_filled(false);
    std::cout << "Checkpoint partially loaded from " << dir.string()
              << " (" << dm.count_computed() << "/" << (meta.n * meta.n) << " entries computed)" << std::endl;
  }

  return true;
}

} // namespace dtwc
