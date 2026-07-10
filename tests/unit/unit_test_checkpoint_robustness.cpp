/**
 * @file unit_test_checkpoint_robustness.cpp
 * @brief Dense CSV checkpoint identity, integrity, and transaction contract (M49).
 *
 * Binary ClusteringResult checkpoints have a separate format and fuzz target;
 * this file deliberately exercises only save_checkpoint/load_checkpoint.
 */

#include <dtwc.hpp>
#include <core/sha256.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using namespace dtwc;
namespace fs = std::filesystem;

namespace {

constexpr std::size_t series_count = 3;

Data make_f64_data(double shift = 0.0)
{
  return Data(
    std::vector<std::vector<double>>{
      {shift + 0.0, shift + 1.0, shift + 2.0},
      {shift + 1.0, shift + 3.0, shift + 4.0},
      {shift + 2.0, shift + 5.0, shift + 8.0}},
    std::vector<std::string>{"a", "b", "c"});
}

Data make_f32_data()
{
  return Data(
    std::vector<std::vector<float>>{
      {0.0f, 1.0f, 2.0f},
      {1.0f, 3.0f, 4.0f},
      {2.0f, 5.0f, 8.0f}},
    std::vector<std::string>{"a", "b", "c"});
}

struct ScratchDirectory
{
  fs::path root;

  explicit ScratchDirectory(std::string_view stem)
    : root(fs::temp_directory_path()
           / (std::string(stem) + "_"
              + std::to_string(reinterpret_cast<std::uintptr_t>(this))))
  {
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);
  }

  ~ScratchDirectory()
  {
    std::error_code error;
    fs::remove_all(root, error);
  }
};

class SilenceCout
{
  std::ostringstream sink_;
  std::streambuf *original_;

public:
  SilenceCout() : original_(std::cout.rdbuf(sink_.rdbuf())) {}
  ~SilenceCout() { std::cout.rdbuf(original_); }
};

std::string read_text(const fs::path &path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.is_open());
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

void write_text(const fs::path &path, std::string_view text)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  REQUIRE(output.is_open());
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  REQUIRE(output.good());
}

std::string trim_line(std::string value)
{
  while (!value.empty()
         && (value.back() == '\n' || value.back() == '\r'))
    value.pop_back();
  return value;
}

// V2 may publish immutable generations through CURRENT. The fallback keeps
// these preregistration tests able to mutate the legacy direct-file format.
fs::path active_payload_directory(const fs::path &checkpoint)
{
  const fs::path current = checkpoint / "CURRENT";
  if (!fs::exists(current)) return checkpoint;
  const std::string generation = trim_line(read_text(current));
  return checkpoint / "generations" / generation;
}

fs::path metadata_path(const fs::path &checkpoint)
{
  return active_payload_directory(checkpoint) / "metadata.txt";
}

fs::path distances_path(const fs::path &checkpoint)
{
  return active_payload_directory(checkpoint) / "distances.csv";
}

void copy_tree(const fs::path &source, const fs::path &destination)
{
  fs::create_directories(destination);
  for (const auto &entry : fs::recursive_directory_iterator(source)) {
    const fs::path target = destination / fs::relative(entry.path(), source);
    if (entry.is_directory()) {
      fs::create_directories(target);
    } else if (entry.is_regular_file()) {
      fs::create_directories(target.parent_path());
      fs::copy_file(entry.path(), target, fs::copy_options::overwrite_existing);
    }
  }
}

bool params_equal(const core::DTWVariantParams &a,
                  const core::DTWVariantParams &b)
{
  return a.variant == b.variant
      && std::bit_cast<std::uint64_t>(a.wdtw_g)
           == std::bit_cast<std::uint64_t>(b.wdtw_g)
      && std::bit_cast<std::uint64_t>(a.adtw_penalty)
           == std::bit_cast<std::uint64_t>(b.adtw_penalty)
      && std::bit_cast<std::uint64_t>(a.sdtw_gamma)
           == std::bit_cast<std::uint64_t>(b.sdtw_gamma)
      && std::bit_cast<std::uint64_t>(a.msm_c)
           == std::bit_cast<std::uint64_t>(b.msm_c)
      && std::bit_cast<std::uint64_t>(a.twe_nu)
           == std::bit_cast<std::uint64_t>(b.twe_nu)
      && std::bit_cast<std::uint64_t>(a.twe_lambda)
           == std::bit_cast<std::uint64_t>(b.twe_lambda)
      && a.mv_mode == b.mv_mode;
}

std::vector<std::uint64_t> data_bits(const Problem &problem)
{
  std::vector<std::uint64_t> result;
  for (std::size_t i = 0; i < problem.size(); ++i) {
    result.push_back(static_cast<std::uint64_t>(problem.data.series_flat_size(i)));
    if (problem.data.is_f32()) {
      for (float value : problem.data.series_f32(i))
        result.push_back(std::bit_cast<std::uint32_t>(value));
    } else {
      for (double value : problem.series(i))
        result.push_back(std::bit_cast<std::uint64_t>(value));
    }
  }
  return result;
}

struct ProblemSnapshot
{
  const double *matrix_address{};
  std::size_t matrix_size{};
  std::vector<std::uint64_t> matrix_bits;
  int band{};
  core::DTWVariantParams params;
  core::MissingStrategy missing{};
  DistanceMatrixStrategy distance_strategy{};
  int cuda_device{};
  int cuda_precision{};
  core::Precision precision{};
  std::size_t ndim{};
  std::vector<std::uint64_t> series_bits;
  std::vector<int> labels;
  std::vector<int> medoids;
};

ProblemSnapshot snapshot(const Problem &problem)
{
  const auto &matrix = problem.dense_distance_matrix();
  ProblemSnapshot result;
  result.matrix_address = matrix.raw();
  result.matrix_size = matrix.size();
  result.matrix_bits.reserve(matrix.packed_count());
  for (std::size_t i = 0; i < matrix.packed_count(); ++i)
    result.matrix_bits.push_back(std::bit_cast<std::uint64_t>(matrix.raw()[i]));
  result.band = problem.band;
  result.params = problem.variant_params;
  result.missing = problem.missing_strategy;
  result.distance_strategy = problem.distance_strategy;
  result.cuda_device = problem.cuda_settings.device_id;
  result.cuda_precision = problem.cuda_settings.precision;
  result.precision = problem.data.precision;
  result.ndim = problem.data.ndim;
  result.series_bits = data_bits(problem);
  result.labels = problem.labels();
  result.medoids = problem.medoids();
  return result;
}

bool state_matches(const Problem &problem, const ProblemSnapshot &before)
{
  const auto &matrix = problem.dense_distance_matrix();
  if (matrix.raw() != before.matrix_address
      || matrix.size() != before.matrix_size
      || matrix.packed_count() != before.matrix_bits.size())
    return false;
  for (std::size_t i = 0; i < matrix.packed_count(); ++i) {
    if (std::bit_cast<std::uint64_t>(matrix.raw()[i]) != before.matrix_bits[i])
      return false;
  }
  return problem.band == before.band
      && params_equal(problem.variant_params, before.params)
      && problem.missing_strategy == before.missing
      && problem.distance_strategy == before.distance_strategy
      && problem.cuda_settings.device_id == before.cuda_device
      && problem.cuda_settings.precision == before.cuda_precision
      && problem.data.precision == before.precision
      && problem.data.ndim == before.ndim
      && data_bits(problem) == before.series_bits
      && problem.labels() == before.labels
      && problem.medoids() == before.medoids;
}

void install_target_cache(Problem &problem, double base = 900.0)
{
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(problem.size());
  for (std::size_t i = 0; i < problem.size(); ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      matrix.set(i, j, i == j ? 0.0 : base + static_cast<double>(10 * i + j));
    }
  }
  problem.clusters_ind = {2, 1, 0};
  problem.centroids_ind = {2, 0};
}

void install_full_source_cache(Problem &problem, double offset = 0.0)
{
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(problem.size());
  matrix.set(0, 0, 0.0);
  matrix.set(1, 1, 0.0);
  matrix.set(2, 2, 0.0);
  matrix.set(0, 1, std::nextafter(1.0 + offset, 2.0 + offset));
  matrix.set(0, 2, std::nextafter(2.0 + offset, 3.0 + offset));
  matrix.set(1, 2, std::nextafter(3.0 + offset, 4.0 + offset));
}

struct LoadOutcome
{
  bool loaded{false};
  bool threw{false};
  std::string exception;
};

LoadOutcome try_load(Problem &problem, const fs::path &checkpoint)
{
  SilenceCout silence;
  LoadOutcome outcome;
  try {
    outcome.loaded = load_checkpoint(problem, checkpoint.string());
  } catch (const std::exception &error) {
    outcome.threw = true;
    outcome.exception = error.what();
  } catch (...) {
    outcome.threw = true;
    outcome.exception = "non-standard exception";
  }
  return outcome;
}

void require_rejected_unchanged(Problem &problem,
                                const fs::path &checkpoint,
                                std::string_view discriminator)
{
  CAPTURE(discriminator);
  const ProblemSnapshot before = snapshot(problem);
  const LoadOutcome outcome = try_load(problem, checkpoint);
  INFO("load exception: " << outcome.exception);
  CHECK_FALSE(outcome.threw);
  CHECK_FALSE(outcome.loaded);
  CHECK(state_matches(problem, before));
}

struct SaveOutcome
{
  bool threw{false};
  std::string exception;
};

SaveOutcome try_save(const Problem &problem, const fs::path &checkpoint)
{
  SilenceCout silence;
  SaveOutcome outcome;
  try {
    save_checkpoint(problem, checkpoint.string());
  } catch (const std::exception &error) {
    outcome.threw = true;
    outcome.exception = error.what();
  } catch (...) {
    outcome.threw = true;
    outcome.exception = "non-standard exception";
  }
  return outcome;
}

void save_without_noise(const Problem &problem, const fs::path &checkpoint)
{
  const SaveOutcome outcome = try_save(problem, checkpoint);
  INFO("save exception: " << outcome.exception);
  REQUIRE_FALSE(outcome.threw);
}

std::optional<std::string> metadata_value(std::string_view metadata,
                                          std::string_view key)
{
  std::istringstream lines{std::string(metadata)};
  std::string line;
  const std::string prefix = std::string(key) + "=";
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.starts_with(prefix)) return line.substr(prefix.size());
  }
  return std::nullopt;
}

bool is_lower_hex_64(std::string_view value)
{
  return value.size() == 64
      && std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

bool is_canonical_timestamp(std::string_view value)
{
  static const std::regex expression{
    R"(^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$)"};
  return std::regex_match(value.begin(), value.end(), expression);
}

bool has_v2_manifest(const fs::path &checkpoint)
{
  if (!fs::exists(metadata_path(checkpoint))) return false;
  const std::string metadata = read_text(metadata_path(checkpoint));
  const auto format = metadata_value(metadata, "format");
  const auto version = metadata_value(metadata, "version");
  const auto identity = metadata_value(metadata, "identity_sha256");
  const auto payload = metadata_value(metadata, "payload_sha256");
  const auto timestamp = metadata_value(metadata, "timestamp");
  return format == std::optional<std::string>{"dtwc-dense-checkpoint"}
      && version == std::optional<std::string>{"2"}
      && identity && is_lower_hex_64(*identity)
      && payload && is_lower_hex_64(*payload)
      && timestamp && is_canonical_timestamp(*timestamp);
}

std::vector<std::string> split_lines(std::string_view text)
{
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find('\n', start);
    const std::size_t length = end == std::string_view::npos
      ? text.size() - start
      : end - start;
    std::string line{text.substr(start, length)};
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(std::move(line));
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  return lines;
}

std::vector<std::string> split_cells(std::string_view line)
{
  std::vector<std::string> cells;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = line.find(',', start);
    if (end == std::string_view::npos) {
      cells.emplace_back(line.substr(start));
      break;
    }
    cells.emplace_back(line.substr(start, end - start));
    start = end + 1;
  }
  return cells;
}

std::string join_cells(const std::vector<std::string> &cells)
{
  std::ostringstream output;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (i != 0) output << ',';
    output << cells[i];
  }
  return output.str();
}

std::string join_lines(const std::vector<std::string> &lines)
{
  std::ostringstream output;
  for (const std::string &line : lines) output << line << '\n';
  return output.str();
}

void set_csv_cell(std::string &csv, std::size_t row, std::size_t column,
                  std::string value)
{
  auto lines = split_lines(csv);
  REQUIRE(row < lines.size());
  auto cells = split_cells(lines[row]);
  REQUIRE(column < cells.size());
  cells[column] = std::move(value);
  lines[row] = join_cells(cells);
  csv = join_lines(lines);
}

void replace_metadata_value(std::string &metadata, std::string_view key,
                            std::string_view value)
{
  auto lines = split_lines(metadata);
  const std::string prefix = std::string(key) + "=";
  bool replaced = false;
  for (std::string &line : lines) {
    if (!replaced && line.starts_with(prefix)) {
      line = prefix + std::string(value);
      replaced = true;
    }
  }
  if (!replaced) lines.push_back(prefix + std::string(value));
  metadata = join_lines(lines);
}

void remove_metadata_key(std::string &metadata, std::string_view key)
{
  auto lines = split_lines(metadata);
  const std::string prefix = std::string(key) + "=";
  std::erase_if(lines, [&](const std::string &line) {
    return line.starts_with(prefix);
  });
  metadata = join_lines(lines);
}

std::string payload_sha256(std::string_view csv)
{
  core::detail::Sha256 hash;
  static constexpr char domain[] = "dtwc-dense-checkpoint-payload-v2";
  hash.update(domain, sizeof(domain) - 1);
  hash.update(csv.data(), csv.size());
  const auto digest = hash.digest();
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 * digest.size());
  for (std::uint8_t byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0fu]);
  }
  return result;
}

void refresh_payload_hash(const fs::path &checkpoint)
{
  const std::string csv = read_text(distances_path(checkpoint));
  std::string metadata = read_text(metadata_path(checkpoint));
  replace_metadata_value(metadata, "payload_sha256", payload_sha256(csv));
  write_text(metadata_path(checkpoint), metadata);
}

std::map<fs::path, std::string> tree_snapshot(const fs::path &root)
{
  std::map<fs::path, std::string> result;
  if (!fs::exists(root)) return result;
  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file())
      result.emplace(fs::relative(entry.path(), root), read_text(entry.path()));
  }
  return result;
}

bool matrix_bits_equal(const core::DenseDistanceMatrix &a,
                       const core::DenseDistanceMatrix &b)
{
  if (a.size() != b.size() || a.packed_count() != b.packed_count())
    return false;
  for (std::size_t i = 0; i < a.packed_count(); ++i) {
    if (std::bit_cast<std::uint64_t>(a.raw()[i])
        != std::bit_cast<std::uint64_t>(b.raw()[i]))
      return false;
  }
  return true;
}

} // namespace

TEST_CASE("dense checkpoint v2 full and partial round trips are bit exact",
          "[checkpoint][identity][integrity][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_roundtrip"};

  SECTION("full")
  {
    Problem source{"m49_full_source"};
    source.set_data(make_f64_data());
    install_full_source_cache(source);
    const fs::path checkpoint = scratch.root / "full";
    save_without_noise(source, checkpoint);

    CHECK(has_v2_manifest(checkpoint));

    Problem target{"m49_full_target"};
    target.set_data(make_f64_data());
    install_target_cache(target);
    const LoadOutcome outcome = try_load(target, checkpoint);
    INFO("load exception: " << outcome.exception);
    CHECK_FALSE(outcome.threw);
    CHECK(outcome.loaded);
    CHECK(matrix_bits_equal(source.dense_distance_matrix(),
                            target.dense_distance_matrix()));
    CHECK(target.is_distance_matrix_filled());
  }

  SECTION("one computed pair")
  {
    Problem source{"m49_partial_source"};
    source.set_data(make_f64_data());
    auto &matrix = source.dense_distance_matrix();
    matrix.resize(series_count);
    matrix.set(0, 1, std::nextafter(1.0, 2.0));
    const fs::path checkpoint = scratch.root / "partial";
    save_without_noise(source, checkpoint);

    CHECK(has_v2_manifest(checkpoint));

    Problem target{"m49_partial_target"};
    target.set_data(make_f64_data());
    install_target_cache(target);
    const LoadOutcome outcome = try_load(target, checkpoint);
    CHECK_FALSE(outcome.threw);
    CHECK(outcome.loaded);
    CHECK(matrix_bits_equal(source.dense_distance_matrix(),
                            target.dense_distance_matrix()));
    CHECK(target.dense_distance_matrix().count_computed() == 1);
    CHECK_FALSE(target.dense_distance_matrix().is_computed(0, 0));
    CHECK_FALSE(target.dense_distance_matrix().is_computed(0, 2));
  }

  SECTION("zero computed pairs")
  {
    Problem source{"m49_empty_source"};
    source.set_data(make_f64_data());
    const fs::path checkpoint = scratch.root / "empty";
    save_without_noise(source, checkpoint);

    CHECK(has_v2_manifest(checkpoint));

    Problem target{"m49_empty_target"};
    target.set_data(make_f64_data());
    install_target_cache(target);
    const LoadOutcome outcome = try_load(target, checkpoint);
    CHECK_FALSE(outcome.threw);
    CHECK(outcome.loaded);
    CHECK(target.dense_distance_matrix().size() == series_count);
    CHECK(target.dense_distance_matrix().count_computed() == 0);
  }
}

TEST_CASE("dense checkpoint identity covers every distance semantic axis",
          "[checkpoint][identity][transaction][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_identity"};
  const fs::path checkpoint = scratch.root / "source";
  Problem source{"m49_identity_source"};
  source.set_data(make_f64_data());
  install_full_source_cache(source, 10.0);
  save_without_noise(source, checkpoint);

  using Mutation = std::pair<std::string_view, std::function<void(Problem &)>>;
  const std::vector<Mutation> mutations{
    {"same-N data value bits", [](Problem &p) { p.set_data(make_f64_data(10.0)); }},
    {"band", [](Problem &p) { p.set_band(1); }},
    {"variant selector", [](Problem &p) { p.set_variant(core::DTWVariant::DDTW); }},
    {"WDTW g", [](Problem &p) {
       auto v = p.variant_params; v.wdtw_g = 0.75; p.set_variant(v);
     }},
    {"ADTW penalty", [](Problem &p) {
       auto v = p.variant_params; v.adtw_penalty = 0.75; p.set_variant(v);
     }},
    {"Soft-DTW gamma", [](Problem &p) {
       auto v = p.variant_params; v.sdtw_gamma = 0.75; p.set_variant(v);
     }},
    {"MSM c", [](Problem &p) {
       auto v = p.variant_params; v.msm_c = 0.75; p.set_variant(v);
     }},
    {"TWE nu", [](Problem &p) {
       auto v = p.variant_params; v.twe_nu = 0.75; p.set_variant(v);
     }},
    {"TWE lambda", [](Problem &p) {
       auto v = p.variant_params; v.twe_lambda = 0.75; p.set_variant(v);
     }},
    {"multivariate mode", [](Problem &p) {
       auto v = p.variant_params; v.mv_mode = core::MVMode::Independent; p.set_variant(v);
     }},
    {"missing strategy", [](Problem &p) {
       p.set_missing_strategy(core::MissingStrategy::ZeroCost);
     }},
    {"distance strategy", [](Problem &p) {
       p.set_distance_strategy(DistanceMatrixStrategy::BruteForce);
     }},
    {"data precision", [](Problem &p) { p.set_data(make_f32_data()); }},
    {"CUDA device", [](Problem &p) {
       CUDASettings settings = p.cuda_settings; settings.device_id = 7;
       p.set_cuda_settings(settings);
     }},
    {"CUDA precision", [](Problem &p) {
       CUDASettings settings = p.cuda_settings; settings.precision = 2;
       p.set_cuda_settings(settings);
     }},
  };

  for (const auto &[name, mutate] : mutations) {
    Problem target{"m49_identity_target"};
    target.set_data(make_f64_data());
    mutate(target);
    install_target_cache(target);
    require_rejected_unchanged(target, checkpoint, name);
  }
}

TEST_CASE("dense checkpoint metadata parsing is strict and transactional",
          "[checkpoint][metadata][transaction][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_metadata"};
  const fs::path baseline = scratch.root / "baseline";
  Problem source{"m49_metadata_source"};
  source.set_data(make_f64_data());
  install_full_source_cache(source);
  save_without_noise(source, baseline);

  using Mutation = std::pair<std::string_view,
                             std::function<void(std::string &)>>;
  const std::string zeros(64, '0');
  const std::vector<Mutation> mutations{
    {"unknown key", [](std::string &m) { m += "surprise=1\n"; }},
    {"duplicate key", [](std::string &m) { m += "n=3\n"; }},
    {"missing format", [](std::string &m) { remove_metadata_key(m, "format"); }},
    {"missing version", [](std::string &m) { remove_metadata_key(m, "version"); }},
    {"missing n", [](std::string &m) { remove_metadata_key(m, "n"); }},
    {"missing pairs", [](std::string &m) { remove_metadata_key(m, "pairs_computed"); }},
    {"missing timestamp", [](std::string &m) { remove_metadata_key(m, "timestamp"); }},
    {"missing identity", [](std::string &m) { remove_metadata_key(m, "identity_sha256"); }},
    {"missing payload hash", [](std::string &m) { remove_metadata_key(m, "payload_sha256"); }},
    {"wrong format marker", [](std::string &m) {
       replace_metadata_value(m, "format", "not-dtwc");
     }},
    {"malformed line", [](std::string &m) { m += "not-a-key-value\n"; }},
    {"partial n token", [](std::string &m) { replace_metadata_value(m, "n", "3x"); }},
    {"partial pair token", [](std::string &m) {
       replace_metadata_value(m, "pairs_computed", "6x");
     }},
    {"n overflow", [](std::string &m) {
       replace_metadata_value(m, "n", "18446744073709551616");
     }},
    {"pair overflow", [](std::string &m) {
       replace_metadata_value(m, "pairs_computed", "18446744073709551616");
     }},
    {"n zero", [](std::string &m) { replace_metadata_value(m, "n", "0"); }},
    {"pair count beyond packed size", [](std::string &m) {
       replace_metadata_value(m, "pairs_computed", "7");
     }},
    {"in-range pair count disagrees with payload", [](std::string &m) {
       replace_metadata_value(m, "pairs_computed", "5");
     }},
    {"unsupported version", [](std::string &m) {
       replace_metadata_value(m, "version", "999");
     }},
    {"partial version", [](std::string &m) {
       replace_metadata_value(m, "version", "2x");
     }},
    {"wrong identity", [zeros](std::string &m) {
       replace_metadata_value(m, "identity_sha256", zeros);
     }},
    {"malformed identity", [](std::string &m) {
       replace_metadata_value(m, "identity_sha256", "xyz");
     }},
    {"malformed payload hash", [](std::string &m) {
       replace_metadata_value(m, "payload_sha256", "xyz");
     }},
    {"shape-valid impossible timestamp", [](std::string &m) {
       replace_metadata_value(m, "timestamp", "2026-02-30T12:00:00Z");
     }},
    {"truncated manifest", [](std::string &m) { m.resize(m.size() / 2); }},
  };

  for (std::size_t i = 0; i < mutations.size(); ++i) {
    const auto &[name, mutate] = mutations[i];
    const fs::path candidate = scratch.root / ("case_" + std::to_string(i));
    copy_tree(baseline, candidate);
    std::string metadata = read_text(metadata_path(candidate));
    mutate(metadata);
    write_text(metadata_path(candidate), metadata);

    Problem target{"m49_metadata_target"};
    target.set_data(make_f64_data());
    install_target_cache(target);
    require_rejected_unchanged(target, candidate, name);
  }
}

TEST_CASE("dense checkpoint CSV parsing is exact and transactional",
          "[checkpoint][csv][transaction][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_csv"};
  const fs::path baseline = scratch.root / "baseline";
  Problem source{"m49_csv_source"};
  source.set_data(make_f64_data());
  install_full_source_cache(source);
  save_without_noise(source, baseline);

  struct Mutation {
    std::string_view name;
    std::function<void(std::string &)> mutate;
    bool refresh_hash;
  };
  std::vector<Mutation> mutations{
    {"empty CSV", [](std::string &csv) { csv.clear(); }, true},
    {"fewer rows", [](std::string &csv) {
       auto lines = split_lines(csv); lines.pop_back(); csv = join_lines(lines);
     }, true},
    {"extra finite row", [](std::string &csv) { csv += "0,0,0\n"; }, true},
    {"empty interior row", [](std::string &csv) {
       auto lines = split_lines(csv); lines[1].clear(); csv = join_lines(lines);
     }, true},
    {"short row", [](std::string &csv) {
       auto lines = split_lines(csv); auto cells = split_cells(lines[1]);
       cells.pop_back(); lines[1] = join_cells(cells); csv = join_lines(lines);
     }, true},
    {"long row", [](std::string &csv) {
       auto lines = split_lines(csv); lines[1] += ",0"; csv = join_lines(lines);
     }, true},
    {"malformed numeric token", [](std::string &csv) {
       set_csv_cell(csv, 0, 1, "not-a-number");
     }, true},
    {"partial numeric token", [](std::string &csv) {
       set_csv_cell(csv, 0, 1, "1junk");
     }, true},
    {"numeric overflow", [](std::string &csv) {
       set_csv_cell(csv, 0, 1, "1e9999");
     }, true},
    {"positive infinity", [](std::string &csv) {
       set_csv_cell(csv, 0, 1, "inf");
     }, true},
    {"asymmetric finite values", [](std::string &csv) {
       set_csv_cell(csv, 0, 1, "123.25");
     }, true},
    {"numeric versus uncomputed asymmetry", [](std::string &csv) {
       set_csv_cell(csv, 0, 1, "");
     }, true},
    {"finite digit corruption", [](std::string &csv) {
       set_csv_cell(csv, 1, 2, "3.125");
       set_csv_cell(csv, 2, 1, "3.125");
     }, false},
  };
#ifdef NDEBUG
  // The unfixed Debug implementation asserts while publishing NaN. Release is
  // the authoritative public-input boundary; the repaired parser rejects it
  // before DenseDistanceMatrix::set in every build mode.
  mutations.push_back({"textual NaN is not an uncomputed sentinel",
                       [](std::string &csv) {
                         set_csv_cell(csv, 0, 1, "nan");
                       }, true});
#endif

  for (std::size_t i = 0; i < mutations.size(); ++i) {
    const auto &[name, mutate, recompute_hash] = mutations[i];
    const fs::path candidate = scratch.root / ("case_" + std::to_string(i));
    copy_tree(baseline, candidate);
    std::string csv = read_text(distances_path(candidate));
    mutate(csv);
    write_text(distances_path(candidate), csv);
    if (recompute_hash) refresh_payload_hash(candidate);

    Problem target{"m49_csv_target"};
    target.set_data(make_f64_data());
    install_target_cache(target);
    require_rejected_unchanged(target, candidate, name);
  }
}

TEST_CASE("dense checkpoint detects cross-file tears before publication",
          "[checkpoint][integrity][transaction][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_torn"};
  const fs::path first = scratch.root / "first";
  const fs::path second = scratch.root / "second";

  Problem source_a{"m49_torn_a"};
  source_a.set_data(make_f64_data());
  install_full_source_cache(source_a, 0.0);
  save_without_noise(source_a, first);

  Problem source_b{"m49_torn_b"};
  source_b.set_data(make_f64_data());
  install_full_source_cache(source_b, 20.0);
  save_without_noise(source_b, second);

  // Same identity and shape, but metadata and payload came from distinct save
  // generations. An identity-only format cannot detect this torn pair.
  write_text(distances_path(first), read_text(distances_path(second)));

  Problem target{"m49_torn_target"};
  target.set_data(make_f64_data());
  install_target_cache(target);
  require_rejected_unchanged(target, first, "cross-generation payload");
}

TEST_CASE("legacy dense checkpoint directories never bypass v2 identity",
          "[checkpoint][legacy][identity][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_legacy"};
  const fs::path legacy = scratch.root / "legacy";
  fs::create_directories(legacy);
  write_text(legacy / "metadata.txt",
             "n=3\nband=-1\nvariant=Standard\npairs_computed=6\n"
             "timestamp=2026-07-10T00:00:00\n");
  write_text(legacy / "distances.csv",
             "0,1,2\n1,0,3\n2,3,0\n");

  Problem target{"m49_legacy_target"};
  target.set_data(make_f64_data());
  install_target_cache(target);
  require_rejected_unchanged(target, legacy, "unversioned v1 directory");

  // A deliberate save to the same path is the supported upgrade: it must
  // publish a complete v2 generation, not make the unsafe legacy files live.
  Problem source{"m49_legacy_upgrade"};
  source.set_data(make_f64_data());
  install_full_source_cache(source);
  save_without_noise(source, legacy);
  CHECK(has_v2_manifest(legacy));

  Problem upgraded_target{"m49_legacy_upgraded_target"};
  upgraded_target.set_data(make_f64_data());
  install_target_cache(upgraded_target);
  const LoadOutcome outcome = try_load(upgraded_target, legacy);
  CHECK_FALSE(outcome.threw);
  CHECK(outcome.loaded);
  CHECK(matrix_bits_equal(source.dense_distance_matrix(),
                          upgraded_target.dense_distance_matrix()));
}

TEST_CASE("invalid checkpoint saves have no filesystem effects",
          "[checkpoint][save][transaction][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_save_preflight"};

  SECTION("matrix dimension disagrees with resident data")
  {
    Problem problem{"m49_bad_shape"};
    problem.set_data(make_f64_data());
    problem.dense_distance_matrix().resize(2);
    const fs::path checkpoint = scratch.root / "bad_shape";
    const SaveOutcome outcome = try_save(problem, checkpoint);
    INFO(outcome.exception);
    CHECK(outcome.threw);
    CHECK_FALSE(fs::exists(checkpoint));
  }

  SECTION("computed value is non-finite")
  {
    Problem problem{"m49_bad_value"};
    problem.set_data(make_f64_data());
    auto &matrix = problem.dense_distance_matrix();
    matrix.resize(series_count);
    matrix.set(0, 1, std::numeric_limits<double>::infinity());
    const fs::path checkpoint = scratch.root / "bad_value";
    const SaveOutcome outcome = try_save(problem, checkpoint);
    INFO(outcome.exception);
    CHECK(outcome.threw);
    CHECK_FALSE(fs::exists(checkpoint));
  }

  SECTION("raw semantic drift is rejected before directory creation")
  {
    Problem problem{"m49_stale"};
    problem.set_data(make_f64_data());
    install_target_cache(problem);
    problem.band = 1;
    const fs::path checkpoint = scratch.root / "stale";
    const SaveOutcome outcome = try_save(problem, checkpoint);
    INFO(outcome.exception);
    CHECK(outcome.threw);
    CHECK_FALSE(fs::exists(checkpoint));
  }

#ifdef DTWC_HAS_MMAP
  SECTION("mmap-backed Problems remain outside the legacy dense format")
  {
    Problem problem{"m49_mmap"};
    problem.set_data(make_f64_data());
    problem.use_mmap_distance_matrix(scratch.root / "matrix.dtwcache");
    const fs::path checkpoint = scratch.root / "mmap_checkpoint";
    const SaveOutcome outcome = try_save(problem, checkpoint);
    INFO(outcome.exception);
    CHECK(outcome.threw);
    CHECK_FALSE(fs::exists(checkpoint));
  }
#endif
}

TEST_CASE("failed overwrite preserves the previous checkpoint generation",
          "[checkpoint][save][transaction][integrity][m49]")
{
  ScratchDirectory scratch{"dtwc_m49_overwrite"};
  const fs::path checkpoint = scratch.root / "checkpoint";

  Problem original{"m49_original"};
  original.set_data(make_f64_data());
  install_full_source_cache(original);
  save_without_noise(original, checkpoint);
  const auto before_files = tree_snapshot(checkpoint);

  Problem invalid{"m49_invalid_overwrite"};
  invalid.set_data(make_f64_data());
  invalid.dense_distance_matrix().resize(2);
  const SaveOutcome outcome = try_save(invalid, checkpoint);
  INFO(outcome.exception);
  CHECK(outcome.threw);
  CHECK(tree_snapshot(checkpoint) == before_files);

  Problem target{"m49_overwrite_target"};
  target.set_data(make_f64_data());
  install_target_cache(target);
  const LoadOutcome load = try_load(target, checkpoint);
  CHECK_FALSE(load.threw);
  CHECK(load.loaded);
  CHECK(matrix_bits_equal(original.dense_distance_matrix(),
                          target.dense_distance_matrix()));
}
