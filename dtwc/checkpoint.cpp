/**
 * @file checkpoint.cpp
 * @brief Implementation of checkpoint save/load for distance matrix computation.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include "checkpoint.hpp"
#include "Problem.hpp"
#include "core/matrix_io.hpp"

#include <chrono>
#include <cstring>
#include <cstdint>
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

  // Checkpoint CSV save only works with DenseDistanceMatrix.
  // MmapDistanceMatrix is file-backed and doesn't need CSV checkpointing.
  const auto &dm = prob.dense_distance_matrix();
  const size_t n = dm.size();

  // Write distance matrix CSV
  io::write_csv(dm, dir / "distances.csv");

  // Write metadata
  write_metadata(dir / "metadata.txt",
                 n,
                 prob.band,
                 prob.variant_params.variant,
                 dm.count_computed());

  std::cout << "Checkpoint saved to " << dir.string()
            << " (" << dm.count_computed() << "/" << dm.packed_count() << " entries computed)" << '\n';
}


bool load_checkpoint(Problem &prob, const std::string &path)
{
  fs::path dir(path);

  // Check that both files exist
  fs::path csv_path = dir / "distances.csv";
  fs::path meta_path = dir / "metadata.txt";

  if (!fs::exists(csv_path) || !fs::exists(meta_path)) {
    std::cout << "No checkpoint found at " << dir.string() << '\n';
    return false;
  }

  // Read and validate metadata
  CheckpointMetadata meta;
  if (!read_metadata(meta_path, meta)) {
    std::cout << "Checkpoint metadata is invalid at " << dir.string() << '\n';
    return false;
  }

  // Validate dimension matches the Problem's data
  if (meta.n != prob.size()) {
    std::cout << "Checkpoint dimension mismatch: checkpoint has n=" << meta.n
              << " but Problem has n=" << prob.size() << '\n';
    return false;
  }

  // Load the distance matrix CSV (Dense only — mmap doesn't use CSV checkpointing).
  auto &dm_load = prob.dense_distance_matrix();
  try {
    io::read_csv(dm_load, csv_path);
  } catch (const std::exception &e) {
    std::cout << "Failed to read checkpoint distances: " << e.what() << '\n';
    return false;
  }

  // Check if all pairs are computed
  if (dm_load.all_computed()) {
    std::cout << "Checkpoint fully loaded from " << dir.string()
              << " (all " << dm_load.packed_count() << " entries computed)" << '\n';
  } else {
    std::cout << "Checkpoint partially loaded from " << dir.string()
              << " (" << dm_load.count_computed() << "/" << dm_load.packed_count() << " entries computed)" << '\n';
  }

  return true;
}


// ---- Binary checkpoint for ClusteringResult --------------------------------

namespace {

constexpr char BINARY_MAGIC[4] = { 'D', 'C', 'K', 'P' };
constexpr uint16_t BINARY_VERSION = 1;

} // anonymous namespace


void save_binary_checkpoint(const core::ClusteringResult &result,
                            const fs::path &path)
{
  // Ensure parent directory exists
  if (path.has_parent_path())
    fs::create_directories(path.parent_path());

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open())
    throw std::runtime_error("Cannot open binary checkpoint for writing: " + path.string());

  // Header
  out.write(BINARY_MAGIC, 4);

  const uint16_t version = BINARY_VERSION;
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));

  const uint16_t reserved = 0;
  out.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));

  const int32_t k = static_cast<int32_t>(result.medoid_indices.size());
  const int32_t N = static_cast<int32_t>(result.labels.size());
  const int32_t iterations = static_cast<int32_t>(result.iterations);
  out.write(reinterpret_cast<const char *>(&k), sizeof(k));
  out.write(reinterpret_cast<const char *>(&N), sizeof(N));
  out.write(reinterpret_cast<const char *>(&iterations), sizeof(iterations));

  const uint8_t converged = result.converged ? 1 : 0;
  out.write(reinterpret_cast<const char *>(&converged), sizeof(converged));

  const char padding[3] = { 0, 0, 0 };
  out.write(padding, 3);

  out.write(reinterpret_cast<const char *>(&result.total_cost), sizeof(result.total_cost));

  // Medoid indices
  for (int32_t i = 0; i < k; ++i) {
    const int32_t val = static_cast<int32_t>(result.medoid_indices[i]);
    out.write(reinterpret_cast<const char *>(&val), sizeof(val));
  }

  // Labels
  for (int32_t i = 0; i < N; ++i) {
    const int32_t val = static_cast<int32_t>(result.labels[i]);
    out.write(reinterpret_cast<const char *>(&val), sizeof(val));
  }

  if (!out.good())
    throw std::runtime_error("Write error on binary checkpoint: " + path.string());
}


bool load_binary_checkpoint(core::ClusteringResult &result,
                            const fs::path &path)
{
  if (!fs::exists(path))
    return false;

  std::ifstream in(path, std::ios::binary);
  if (!in.is_open())
    return false;

  // Read and validate magic
  char magic[4];
  in.read(magic, 4);
  if (!in.good() || std::memcmp(magic, BINARY_MAGIC, 4) != 0)
    return false;

  // Read and validate version
  uint16_t version = 0;
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (!in.good() || version != BINARY_VERSION)
    return false;

  // Skip reserved
  uint16_t reserved = 0;
  in.read(reinterpret_cast<char *>(&reserved), sizeof(reserved));

  // Read header fields
  int32_t k = 0, N = 0, iterations = 0;
  in.read(reinterpret_cast<char *>(&k), sizeof(k));
  in.read(reinterpret_cast<char *>(&N), sizeof(N));
  in.read(reinterpret_cast<char *>(&iterations), sizeof(iterations));

  uint8_t converged = 0;
  in.read(reinterpret_cast<char *>(&converged), sizeof(converged));

  // Skip padding
  char padding[3];
  in.read(padding, 3);

  double total_cost = 0.0;
  in.read(reinterpret_cast<char *>(&total_cost), sizeof(total_cost));

  if (!in.good())
    return false;

  // Read medoid indices
  std::vector<int> medoid_indices(k);
  for (int32_t i = 0; i < k; ++i) {
    int32_t val = 0;
    in.read(reinterpret_cast<char *>(&val), sizeof(val));
    medoid_indices[i] = val;
  }

  // Read labels
  std::vector<int> labels(N);
  for (int32_t i = 0; i < N; ++i) {
    int32_t val = 0;
    in.read(reinterpret_cast<char *>(&val), sizeof(val));
    labels[i] = val;
  }

  if (!in.good())
    return false;

  // Populate result
  result.medoid_indices = std::move(medoid_indices);
  result.labels = std::move(labels);
  result.total_cost = total_cost;
  result.iterations = iterations;
  result.converged = (converged != 0);

  return true;
}

} // namespace dtwc
