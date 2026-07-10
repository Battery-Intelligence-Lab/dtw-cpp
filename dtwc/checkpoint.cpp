/**
 * @file checkpoint.cpp
 * @brief Implementation of checkpoint save/load for distance matrix computation.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include "checkpoint.hpp"
#include "Problem.hpp"
#include "core/sha256.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#endif

namespace dtwc {

namespace fs = std::filesystem;

namespace {

// Dense checkpoint v2 helpers are intentionally kept in this translation unit:
// their byte encodings are a persistence contract, not a public API.
constexpr std::string_view DENSE_FORMAT = "dtwc-dense-checkpoint";
constexpr std::string_view PAYLOAD_HASH_DOMAIN =
  "dtwc-dense-checkpoint-payload-v2";
constexpr std::size_t DIGEST_HEX_SIZE = 64;
constexpr std::size_t CURRENT_FILE_SIZE = DIGEST_HEX_SIZE + 1;
constexpr std::size_t MAX_MANIFEST_SIZE = 2048;
constexpr std::size_t MAX_NUMERIC_TOKEN_SIZE = 32;

std::string digest_hex(const core::detail::Sha256::Digest &digest)
{
  static constexpr char HEX[] = "0123456789abcdef";
  std::string result;
  result.reserve(DIGEST_HEX_SIZE);
  for (const std::uint8_t byte : digest) {
    result.push_back(HEX[byte >> 4]);
    result.push_back(HEX[byte & 0x0fu]);
  }
  return result;
}

bool is_lower_hex_digest(std::string_view value)
{
  if (value.size() != DIGEST_HEX_SIZE) return false;
  for (const char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  }
  return true;
}

std::string current_timestamp()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  if (gmtime_s(&tm_buf, &time_t_now) != 0)
    throw std::runtime_error("Cannot convert checkpoint timestamp to UTC.");
#else
  if (gmtime_r(&time_t_now, &tm_buf) == nullptr)
    throw std::runtime_error("Cannot convert checkpoint timestamp to UTC.");
#endif
  std::array<char, 21> text{};
  const int written = std::snprintf(
    text.data(), text.size(), "%04d-%02d-%02dT%02d:%02d:%02dZ",
    tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
    tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
  if (written != 20)
    throw std::runtime_error("Cannot format checkpoint timestamp.");
  return std::string(text.data(), 20);
}

bool is_leap_year(int year)
{
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int decimal_at(std::string_view text, std::size_t offset, std::size_t count)
{
  int result = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const char c = text[offset + i];
    if (c < '0' || c > '9') return -1;
    result = result * 10 + (c - '0');
  }
  return result;
}

bool valid_timestamp(std::string_view text)
{
  if (text.size() != 20 || text[4] != '-' || text[7] != '-'
      || text[10] != 'T' || text[13] != ':' || text[16] != ':'
      || text[19] != 'Z')
    return false;
  const int year = decimal_at(text, 0, 4);
  const int month = decimal_at(text, 5, 2);
  const int day = decimal_at(text, 8, 2);
  const int hour = decimal_at(text, 11, 2);
  const int minute = decimal_at(text, 14, 2);
  const int second = decimal_at(text, 17, 2);
  if (year < 1 || month < 1 || month > 12 || day < 1
      || hour < 0 || hour > 23 || minute < 0 || minute > 59
      || second < 0 || second > 59)
    return false;
  static constexpr std::array<int, 12> DAYS{
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };
  const int month_days = month == 2 && is_leap_year(year)
    ? 29 : DAYS[static_cast<std::size_t>(month - 1)];
  return day <= month_days;
}

bool parse_unsigned(std::string_view text, std::uint64_t &value)
{
  if (text.empty() || (text.size() > 1 && text.front() == '0')) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool checked_multiply(std::size_t a, std::size_t b, std::size_t &result)
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  result = a * b;
  return true;
}

bool checked_packed_count(std::size_t n, std::size_t &result)
{
  if (n == std::numeric_limits<std::size_t>::max()) return false;
  return n % 2 == 0
    ? checked_multiply(n / 2, n + 1, result)
    : checked_multiply(n, (n + 1) / 2, result);
}

bool is_regular_file_without_symlink(const fs::path &path)
{
  std::error_code error;
  const fs::file_status status = fs::symlink_status(path, error);
  return !error && status.type() == fs::file_type::regular;
}

bool is_directory_without_symlink(const fs::path &path)
{
  std::error_code error;
  const fs::file_status status = fs::symlink_status(path, error);
  return !error && status.type() == fs::file_type::directory;
}

bool read_small_file(const fs::path &path, std::size_t maximum,
                     std::string &result)
{
  if (!is_regular_file_without_symlink(path)) return false;
  std::error_code error;
  const std::uintmax_t disk_size = fs::file_size(path, error);
  if (error || disk_size > maximum
      || disk_size > std::numeric_limits<std::size_t>::max())
    return false;
  result.resize(static_cast<std::size_t>(disk_size));
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return false;
  if (!result.empty()) {
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (input.gcount() != static_cast<std::streamsize>(result.size())) return false;
  }
  return input.peek() == std::char_traits<char>::eof() && !input.bad();
}

struct CheckpointMetadata {
  std::size_t n = 0;
  std::size_t pairs_computed = 0;
  std::string timestamp;
  std::string identity_sha256;
  std::string payload_sha256;
};

int metadata_key_index(std::string_view key)
{
  static constexpr std::array<std::string_view, 7> KEYS{
    "format", "version", "n", "pairs_computed", "timestamp",
    "identity_sha256", "payload_sha256"
  };
  for (std::size_t i = 0; i < KEYS.size(); ++i)
    if (key == KEYS[i]) return static_cast<int>(i);
  return -1;
}

bool parse_manifest(const std::string &text, CheckpointMetadata &metadata)
{
  if (text.empty() || text.size() > MAX_MANIFEST_SIZE || text.back() != '\n')
    return false;

  std::array<std::string_view, 7> values{};
  std::array<bool, 7> present{};
  std::size_t line_start = 0;
  std::size_t line_count = 0;
  while (line_start < text.size()) {
    const std::size_t line_end = text.find('\n', line_start);
    if (line_end == std::string::npos || line_end == line_start) return false;
    const std::string_view line(text.data() + line_start, line_end - line_start);
    if (line.find('\r') != std::string_view::npos) return false;
    const std::size_t separator = line.find('=');
    if (separator == std::string_view::npos || separator == 0
        || separator + 1 == line.size()
        || line.find('=', separator + 1) != std::string_view::npos)
      return false;
    const int index = metadata_key_index(line.substr(0, separator));
    if (index < 0 || present[static_cast<std::size_t>(index)]) return false;
    present[static_cast<std::size_t>(index)] = true;
    values[static_cast<std::size_t>(index)] = line.substr(separator + 1);
    ++line_count;
    line_start = line_end + 1;
  }
  if (line_count != values.size()) return false;
  for (const bool found : present)
    if (!found) return false;

  if (values[0] != DENSE_FORMAT || values[1] != "2") return false;
  std::uint64_t n = 0;
  std::uint64_t pairs = 0;
  if (!parse_unsigned(values[2], n) || n == 0
      || !parse_unsigned(values[3], pairs)
      || n > std::numeric_limits<std::size_t>::max()
      || pairs > std::numeric_limits<std::size_t>::max()
      || !valid_timestamp(values[4])
      || !is_lower_hex_digest(values[5])
      || !is_lower_hex_digest(values[6]))
    return false;
  metadata.n = static_cast<std::size_t>(n);
  metadata.pairs_computed = static_cast<std::size_t>(pairs);
  metadata.timestamp.assign(values[4]);
  metadata.identity_sha256.assign(values[5]);
  metadata.payload_sha256.assign(values[6]);
  return true;
}

bool resolve_active_generation(const fs::path &root, fs::path &generation)
{
  std::string current;
  if (!read_small_file(root / "CURRENT", CURRENT_FILE_SIZE, current)
      || current.size() != CURRENT_FILE_SIZE || current.back() != '\n')
    return false;
  current.pop_back();
  if (!is_lower_hex_digest(current)) return false;

  const fs::path generations = root / "generations";
  if (!is_directory_without_symlink(generations)) return false;
  generation = generations / current;
  return is_directory_without_symlink(generation);
}

bool read_bounded_line(std::istream &input, std::size_t maximum,
                       std::string &line)
{
  line.clear();
  char c = 0;
  while (input.get(c)) {
    if (c == '\n') return true;
    if (c == '\r' || line.size() == maximum) return false;
    line.push_back(c);
  }
  return false;
}

bool parse_finite_double(std::string_view token, double &value)
{
  if (token.empty() || token.size() > MAX_NUMERIC_TOKEN_SIZE) return false;
  const auto parsed = std::from_chars(
    token.data(), token.data() + token.size(), value,
    std::chars_format::general);
  return parsed.ec == std::errc{}
      && parsed.ptr == token.data() + token.size() && std::isfinite(value);
}

bool parse_csv(const fs::path &path, const CheckpointMetadata &metadata,
               core::DenseDistanceMatrix &candidate)
{
  if (!is_regular_file_without_symlink(path)) return false;
  std::size_t square = 0;
  std::size_t maximum_file_size = 0;
  std::size_t maximum_line_size = 0;
  std::size_t packed = 0;
  if (!checked_multiply(metadata.n, metadata.n, square)
      || !checked_multiply(square, MAX_NUMERIC_TOKEN_SIZE + 1,
                           maximum_file_size)
      || !checked_multiply(metadata.n, MAX_NUMERIC_TOKEN_SIZE + 1,
                           maximum_line_size)
      || maximum_line_size == 0 || !checked_packed_count(metadata.n, packed)
      || metadata.pairs_computed > packed)
    return false;
  --maximum_line_size;

  std::error_code error;
  const std::uintmax_t disk_size = fs::file_size(path, error);
  if (error || disk_size < square || disk_size > maximum_file_size) return false;

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return false;
  candidate.resize(metadata.n);
  core::detail::Sha256 payload_hash;
  payload_hash.update(PAYLOAD_HASH_DOMAIN.data(), PAYLOAD_HASH_DOMAIN.size());

  std::string line;
  line.reserve(maximum_line_size);
  static constexpr char NEWLINE = '\n';
  for (std::size_t row = 0; row < metadata.n; ++row) {
    if (!read_bounded_line(input, maximum_line_size, line)) return false;
    payload_hash.update(line.data(), line.size());
    payload_hash.update(&NEWLINE, 1);

    std::size_t token_start = 0;
    for (std::size_t column = 0; column < metadata.n; ++column) {
      const std::size_t comma = line.find(',', token_start);
      const bool final_column = column + 1 == metadata.n;
      if ((!final_column && comma == std::string::npos)
          || (final_column && comma != std::string::npos))
        return false;
      const std::size_t token_end = final_column ? line.size() : comma;
      const std::string_view token(
        line.data() + token_start, token_end - token_start);

      const bool computed = !token.empty();
      double value = 0.0;
      if (computed && !parse_finite_double(token, value)) return false;

      if (row <= column) {
        if (computed) candidate.set(row, column, value);
      } else {
        const bool mirror_computed = candidate.is_computed(row, column);
        if (computed != mirror_computed) return false;
        if (computed
            && std::bit_cast<std::uint64_t>(value)
                 != std::bit_cast<std::uint64_t>(candidate.get(row, column)))
          return false;
      }
      token_start = token_end + (final_column ? 0 : 1);
    }
  }
  if (input.peek() != std::char_traits<char>::eof() || input.bad()) return false;
  return digest_hex(payload_hash.digest()) == metadata.payload_sha256
      && candidate.count_computed() == metadata.pairs_computed;
}

std::string generation_id(
  const core::MmapDistanceMatrix::fingerprint_type &identity)
{
  static std::atomic<std::uint64_t> sequence{0};
  core::detail::Sha256 hash;
  static constexpr char GENERATION_DOMAIN[] =
    "dtwc-dense-checkpoint-generation-v2";
  hash.update(GENERATION_DOMAIN, sizeof(GENERATION_DOMAIN) - 1);
  hash.update(identity);
  const auto now = std::chrono::high_resolution_clock::now()
                     .time_since_epoch().count();
  hash.update(&now, sizeof(now));
  const std::uint64_t serial = sequence.fetch_add(1, std::memory_order_relaxed);
  hash.update(&serial, sizeof(serial));
  std::random_device random;
  for (int i = 0; i < 8; ++i) {
    const auto value = random();
    hash.update(&value, sizeof(value));
  }
  return digest_hex(hash.digest());
}

struct GenerationCleanup
{
  fs::path root;
  fs::path generations;
  fs::path generation;
  fs::path current_temporary;
  bool root_created{false};
  bool generations_created{false};
  bool published{false};

  ~GenerationCleanup()
  {
    if (published) return;
    std::error_code ignored;
    if (!current_temporary.empty()) fs::remove(current_temporary, ignored);
    if (!generation.empty()) fs::remove_all(generation, ignored);
    if (generations_created) fs::remove(generations, ignored);
    if (root_created) fs::remove(root, ignored);
  }
};

void write_file(const fs::path &path, std::string_view bytes)
{
  if (bytes.size()
      > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::runtime_error("Checkpoint file exceeds stream limits.");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open())
    throw std::runtime_error("Cannot open checkpoint file for writing: "
                             + path.string());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output)
    throw std::runtime_error("Write error on checkpoint file: " + path.string());
}

void replace_current(const fs::path &temporary, const fs::path &current)
{
#ifdef _WIN32
  if (!MoveFileExW(temporary.native().c_str(), current.native().c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    throw std::system_error(
      static_cast<int>(GetLastError()), std::system_category(),
      "Cannot atomically publish checkpoint CURRENT");
  }
#else
  if (::rename(temporary.c_str(), current.c_str()) != 0)
    throw std::system_error(
      errno, std::generic_category(),
      "Cannot atomically publish checkpoint CURRENT");
#endif
}

} // anonymous namespace


void save_checkpoint(const Problem &prob, const std::string &path)
{
  // Complete every semantic, shape, identity, and finite-value check before
  // touching the filesystem. A zero-sized Dense matrix is the valid deferred
  // representation of an all-uncomputed logical N-by-N matrix.
  const core::DenseDistanceMatrix &matrix = prob.dense_distance_matrix();
  const std::size_t n = prob.size();
  if (n == 0)
    throw std::runtime_error("Dense checkpoint requires at least one series.");
  if (matrix.size() != 0 && matrix.size() != n)
    throw std::runtime_error(
      "Dense checkpoint matrix dimension does not match resident data.");
  std::size_t packed = 0;
  if (!checked_packed_count(n, packed))
    throw std::runtime_error("Dense checkpoint dimension overflows packed size.");

  const std::size_t pairs_computed = matrix.size() == 0
    ? 0 : matrix.count_computed();
  if (pairs_computed > packed)
    throw std::runtime_error("Dense checkpoint computed-pair count is invalid.");
  if (matrix.size() != 0) {
    for (std::size_t i = 0; i < matrix.packed_count(); ++i) {
      const double value = matrix.raw()[i];
      if (!std::isnan(value) && !std::isfinite(value))
        throw std::runtime_error(
          "Dense checkpoint contains a computed non-finite distance.");
    }
  }
  const auto identity = prob.distance_checkpoint_identity();
  const std::string identity_hex = digest_hex(identity);
  const std::string timestamp = current_timestamp();
  std::size_t maximum_row_size = 0;
  if (!checked_multiply(n, MAX_NUMERIC_TOKEN_SIZE + 1, maximum_row_size)
      || maximum_row_size == 0
      || maximum_row_size
           > static_cast<std::size_t>(
               std::numeric_limits<std::streamsize>::max()))
    throw std::runtime_error("Dense checkpoint row size overflows.");

  const fs::path root(path);
  const fs::path generations = root / "generations";
  GenerationCleanup cleanup{root, generations};
  cleanup.root_created = fs::create_directories(root);
  if (!is_directory_without_symlink(root))
    throw std::runtime_error("Checkpoint root is not a directory: " + root.string());
  cleanup.generations_created = fs::create_directories(generations);
  if (!is_directory_without_symlink(generations))
    throw std::runtime_error(
      "Checkpoint generations path is not a directory: " + generations.string());

  std::string id;
  for (int attempt = 0; attempt < 128; ++attempt) {
    id = generation_id(identity);
    const fs::path candidate = generations / id;
    std::error_code error;
    if (fs::create_directory(candidate, error)) {
      cleanup.generation = candidate;
      break;
    }
    if (error)
      throw std::system_error(error, "Cannot create checkpoint generation");
  }
  if (cleanup.generation.empty())
    throw std::runtime_error("Cannot allocate a unique checkpoint generation.");

  const fs::path csv_path = cleanup.generation / "distances.csv";
  std::ofstream csv(csv_path, std::ios::binary | std::ios::trunc);
  if (!csv.is_open())
    throw std::runtime_error(
      "Cannot open checkpoint distances for writing: " + csv_path.string());
  core::detail::Sha256 payload_hash;
  payload_hash.update(PAYLOAD_HASH_DOMAIN.data(), PAYLOAD_HASH_DOMAIN.size());

  std::string row;
  row.reserve(maximum_row_size);
  std::array<char, 64> number{};
  for (std::size_t i = 0; i < n; ++i) {
    row.clear();
    for (std::size_t j = 0; j < n; ++j) {
      if (j != 0) row.push_back(',');
      if (matrix.size() != 0 && matrix.is_computed(i, j)) {
        const auto formatted = std::to_chars(
          number.data(), number.data() + number.size(), matrix.get(i, j),
          std::chars_format::general, std::numeric_limits<double>::max_digits10);
        if (formatted.ec != std::errc{})
          throw std::runtime_error("Cannot format checkpoint distance.");
        row.append(number.data(), formatted.ptr);
      }
    }
    row.push_back('\n');
    payload_hash.update(row.data(), row.size());
    csv.write(row.data(), static_cast<std::streamsize>(row.size()));
    if (!csv)
      throw std::runtime_error(
        "Write error on checkpoint distances: " + csv_path.string());
  }
  csv.close();
  if (!csv)
    throw std::runtime_error(
      "Write error on checkpoint distances: " + csv_path.string());

  std::string manifest;
  manifest.reserve(320);
  manifest += "format=";
  manifest += DENSE_FORMAT;
  manifest += "\nversion=2\nn=";
  manifest += std::to_string(n);
  manifest += "\npairs_computed=";
  manifest += std::to_string(pairs_computed);
  manifest += "\ntimestamp=";
  manifest += timestamp;
  manifest += "\nidentity_sha256=";
  manifest += identity_hex;
  manifest += "\npayload_sha256=";
  manifest += digest_hex(payload_hash.digest());
  manifest.push_back('\n');
  write_file(cleanup.generation / "metadata.txt", manifest);

  cleanup.current_temporary = root / (".CURRENT." + id + ".tmp");
  if (fs::exists(cleanup.current_temporary))
    throw std::runtime_error("Checkpoint CURRENT staging path already exists.");
  write_file(cleanup.current_temporary, id + "\n");
  replace_current(cleanup.current_temporary, root / "CURRENT");
  cleanup.published = true;
}


bool load_checkpoint(Problem &prob, const std::string &path)
{
  try {
    // Capture the proven Dense destination without invoking a mutable accessor.
    // No Problem state changes before the final nothrow assignment below.
    auto *destination = std::get_if<core::DenseDistanceMatrix>(&prob.distMat);
    if (destination == nullptr) return false;
    prob.validate_dense_cache_configuration();
    const auto expected_identity = prob.distance_checkpoint_identity();
    const std::string expected_identity_hex = digest_hex(expected_identity);
    const std::size_t expected_n = prob.size();
    if (expected_n == 0) return false;

    const fs::path root(path);
    fs::path generation;
    if (!resolve_active_generation(root, generation)) return false;

    std::string manifest_text;
    if (!read_small_file(generation / "metadata.txt", MAX_MANIFEST_SIZE,
                         manifest_text))
      return false;
    CheckpointMetadata metadata;
    if (!parse_manifest(manifest_text, metadata)
        || metadata.n != expected_n
        || metadata.identity_sha256 != expected_identity_hex)
      return false;

    core::DenseDistanceMatrix candidate;
    if (!parse_csv(generation / "distances.csv", metadata, candidate))
      return false;

    static_assert(
      std::is_nothrow_move_assignable_v<core::DenseDistanceMatrix>,
      "Dense checkpoint publication must preserve the strong guarantee");
    *destination = std::move(candidate);
    return true;
  } catch (...) {
    return false;
  }
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
