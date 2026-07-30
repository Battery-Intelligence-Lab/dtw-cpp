/**
 * @file unit_test_checkpoint_binary.cpp
 * @brief F51 canonical binary-v1 ClusteringResult checkpoint contract.
 *
 * The valid fixture freezes every byte of the public little-endian wire
 * format.  The malformed corpus freezes structural rejection, exception
 * safety, and destination transactionality before the reader is exposed
 * through another language binding.
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef DTWC_F51_TEST_ROOT
#error "DTWC_F51_TEST_ROOT must name the configured build-local test root"
#endif

namespace allocation_probe {

constexpr std::size_t target_bytes = 257U * sizeof(int);
static_assert(target_bytes == 1028U);

std::atomic<bool> enabled{ false };
std::atomic<std::size_t> exact_allocations{ 0 };

void *allocate(std::size_t size)
{
  if (enabled.load(std::memory_order_relaxed) && size == target_bytes)
    exact_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size == 0 ? 1 : size))
    return memory;
  throw std::bad_alloc{};
}

} // namespace allocation_probe

void *operator new(std::size_t size)
{
  return allocation_probe::allocate(size);
}

void *operator new[](std::size_t size)
{
  return allocation_probe::allocate(size);
}

void operator delete(void *memory) noexcept
{
  std::free(memory);
}

void operator delete[](void *memory) noexcept
{
  std::free(memory);
}

void operator delete(void *memory, std::size_t) noexcept
{
  std::free(memory);
}

void operator delete[](void *memory, std::size_t) noexcept
{
  std::free(memory);
}

namespace fs = std::filesystem;

namespace {

using Result = dtwc::core::ClusteringResult;

constexpr std::array<std::uint8_t, 72> valid_wire{
  0x44, 0x43, 0x4b, 0x50, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x04, 0x03, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x2a, 0xc0, 0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

fs::path configured_root()
{
  const fs::path root = fs::path{ DTWC_F51_TEST_ROOT }.lexically_normal();
  if (!root.is_absolute() || root.filename() != "f51-binary-checkpoint")
    throw std::runtime_error("unsafe F51 configured test root: " + root.string());
  return root;
}

struct ScratchDirectory
{
  fs::path root;

  explicit ScratchDirectory(std::string_view leaf)
    : root(configured_root() / std::string(leaf))
  {
    if (root.parent_path() != configured_root()
        || root.filename() != fs::path{ leaf }) {
      throw std::runtime_error("F51 scratch path escaped its build root");
    }
    std::error_code error;
    fs::remove_all(root, error);
    if (error)
      throw std::runtime_error("cannot reset F51 scratch root: " + error.message());
    fs::create_directories(root);
  }

  ~ScratchDirectory()
  {
    std::error_code error;
    fs::remove_all(root, error);
  }

  fs::path file(std::string_view leaf) const
  {
    const fs::path path = root / std::string(leaf);
    if (path.parent_path() != root || path.filename() != fs::path{ leaf })
      throw std::runtime_error("F51 artifact path escaped its scratch root");
    return path;
  }
};

template <class ByteRange>
void write_bytes(const fs::path &path, const ByteRange &bytes)
{
  std::ofstream output(
    path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output.is_open())
    throw std::runtime_error("cannot create F51 fixture: " + path.string());
  if (!bytes.empty()) {
    output.write(
      reinterpret_cast<const char *>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  }
  output.close();
  if (!output)
    throw std::runtime_error("cannot write F51 fixture: " + path.string());
}

std::vector<std::uint8_t> read_bytes(const fs::path &path)
{
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("cannot read F51 artifact: " + path.string());
  std::vector<std::uint8_t> bytes;
  char byte = 0;
  while (input.get(byte))
    bytes.push_back(static_cast<std::uint8_t>(
      static_cast<unsigned char>(byte)));
  if (!input.eof())
    throw std::runtime_error("cannot finish reading F51 artifact: " + path.string());
  return bytes;
}

std::size_t matching_bytes(
  const std::vector<std::uint8_t> &actual,
  const std::array<std::uint8_t, 72> &expected)
{
  const auto common = std::min(actual.size(), expected.size());
  std::size_t matches = 0;
  for (std::size_t index = 0; index < common; ++index)
    matches += static_cast<std::size_t>(actual[index] == expected[index]);
  return matches;
}

Result valid_result()
{
  Result result;
  result.labels = { 2, 1, 0, 2, 2, 0, 0 };
  result.medoid_indices = { 6, 1, 4 };
  result.total_cost = -13.25;
  result.iterations = 0x01020304;
  result.converged = true;
  return result;
}

Result sentinel_result()
{
  Result result;
  result.labels = { -7, -8, -9, -10 };
  result.medoid_indices = { 42, 41 };
  result.total_cost = 911.25;
  result.iterations = -333;
  result.converged = false;
  return result;
}

bool same_result(const Result &lhs, const Result &rhs)
{
  return lhs.labels == rhs.labels
         && lhs.medoid_indices == rhs.medoid_indices
         && std::bit_cast<std::uint64_t>(lhs.total_cost)
              == std::bit_cast<std::uint64_t>(rhs.total_cost)
         && lhs.iterations == rhs.iterations
         && lhs.converged == rhs.converged;
}

std::vector<std::pair<std::string, Result>> semantic_compatibility_results()
{
  std::vector<std::pair<std::string, Result>> fixtures;

  auto result = valid_result();
  result.labels.pop_back();
  result.converged = false;
  fixtures.emplace_back("wrong-n", std::move(result));

  result = valid_result();
  result.medoid_indices.pop_back();
  fixtures.emplace_back("wrong-k", std::move(result));

  result = valid_result();
  result.labels.front() = -1;
  fixtures.emplace_back("bad-label", std::move(result));

  result = valid_result();
  result.medoid_indices.front() = 7;
  fixtures.emplace_back("bad-medoid", std::move(result));

  result = valid_result();
  result.medoid_indices[1] = result.medoid_indices[0];
  fixtures.emplace_back("duplicate-medoid", std::move(result));

  result = valid_result();
  result.iterations = -1;
  fixtures.emplace_back("negative-iterations", std::move(result));

  result = valid_result();
  result.total_cost = std::numeric_limits<double>::infinity();
  fixtures.emplace_back("nonfinite-cost", std::move(result));

  return fixtures;
}

enum class CorruptionClass : std::size_t {
  truncation,
  negative_count,
  reserved,
  padding,
  convergence,
  trailing_payload,
  bad_magic,
  bad_version,
  wrong_endian_count,
  count
};

struct Corruption
{
  std::string name;
  CorruptionClass kind;
  std::vector<std::uint8_t> bytes;
};

std::vector<std::uint8_t> oracle_vector()
{
  return { valid_wire.begin(), valid_wire.end() };
}

void store_le_i32(
  std::vector<std::uint8_t> &bytes, std::size_t offset, std::int32_t value)
{
  const auto bits = static_cast<std::uint32_t>(value);
  for (std::size_t byte = 0; byte < 4; ++byte)
    bytes[offset + byte] =
      static_cast<std::uint8_t>((bits >> (8U * byte)) & UINT32_C(0xff));
}

std::vector<Corruption> corruption_corpus()
{
  std::vector<Corruption> corpus;
  corpus.reserve(85);

  for (std::size_t length = 0; length < valid_wire.size(); ++length) {
    corpus.push_back({ "truncate-" + std::to_string(length),
                       CorruptionClass::truncation,
                       { valid_wire.begin(),
                         valid_wire.begin() + static_cast<std::ptrdiff_t>(length) } });
  }

  for (const auto &[name, offset] :
       std::array<std::pair<std::string_view, std::size_t>, 2>{
         std::pair{ "negative-k", std::size_t{ 8 } },
         std::pair{ "negative-N", std::size_t{ 12 } } }) {
    auto bytes = oracle_vector();
    store_le_i32(bytes, offset, -1);
    corpus.push_back({ std::string{ name }, CorruptionClass::negative_count, std::move(bytes) });
  }

  for (const std::size_t offset : { 6U, 7U }) {
    auto bytes = oracle_vector();
    bytes[offset] = 1;
    corpus.push_back({ "reserved-" + std::to_string(offset),
                       CorruptionClass::reserved,
                       std::move(bytes) });
  }

  for (const std::size_t offset : { 21U, 22U, 23U }) {
    auto bytes = oracle_vector();
    bytes[offset] = 1;
    corpus.push_back({ "padding-" + std::to_string(offset),
                       CorruptionClass::padding,
                       std::move(bytes) });
  }

  for (const std::uint8_t value : { std::uint8_t{ 2 }, std::uint8_t{ 255 } }) {
    auto bytes = oracle_vector();
    bytes[20] = value;
    corpus.push_back({ "convergence-" + std::to_string(value),
                       CorruptionClass::convergence,
                       std::move(bytes) });
  }

  auto trailing = oracle_vector();
  trailing.push_back(0);
  corpus.push_back({ "trailing-zero", CorruptionClass::trailing_payload, std::move(trailing) });

  auto bad_magic = oracle_vector();
  bad_magic[0] = 'X';
  corpus.push_back({ "bad-magic", CorruptionClass::bad_magic, std::move(bad_magic) });

  auto bad_version = oracle_vector();
  bad_version[4] = 2;
  bad_version[5] = 0;
  corpus.push_back({ "bad-version", CorruptionClass::bad_version, std::move(bad_version) });

  auto wrong_endian_count = oracle_vector();
  wrong_endian_count[8] = 0x00;
  wrong_endian_count[9] = 0x00;
  wrong_endian_count[10] = 0x00;
  wrong_endian_count[11] = 0x80;
  corpus.push_back({ "wrong-endian-k-128",
                     CorruptionClass::wrong_endian_count,
                     std::move(wrong_endian_count) });

  return corpus;
}

class ExactAllocationProbe
{
public:
  ExactAllocationProbe()
  {
    allocation_probe::exact_allocations.store(0, std::memory_order_relaxed);
    allocation_probe::enabled.store(true, std::memory_order_relaxed);
  }

  ~ExactAllocationProbe()
  {
    allocation_probe::enabled.store(false, std::memory_order_relaxed);
  }

  ExactAllocationProbe(const ExactAllocationProbe &) = delete;
  ExactAllocationProbe &operator=(const ExactAllocationProbe &) = delete;
};

} // namespace

TEST_CASE("binary-v1 checkpoint valid bytes and compatibility are exact",
          "[checkpoint][binary][f51]")
{
  ScratchDirectory scratch{ "valid" };
  const Result expected = valid_result();
  const fs::path checkpoint = scratch.file("valid.bin");
  dtwc::save_binary_checkpoint(expected, checkpoint);

  const auto bytes = read_bytes(checkpoint);
  const auto valid_byte_matches = matching_bytes(bytes, valid_wire);
  CHECK(bytes.size() == valid_wire.size());
  CHECK(valid_byte_matches == valid_wire.size());

  Result loaded;
  const bool loaded_ok = dtwc::load_binary_checkpoint(loaded, checkpoint);
  CHECK(loaded_ok);
  CHECK(loaded.labels == expected.labels);
  CHECK(loaded.medoid_indices == expected.medoid_indices);
  CHECK(std::bit_cast<std::uint64_t>(loaded.total_cost)
        == std::bit_cast<std::uint64_t>(expected.total_cost));
  CHECK(loaded.iterations == expected.iterations);
  CHECK(loaded.converged == expected.converged);
  const std::size_t field_matches =
    static_cast<std::size_t>(loaded.labels == expected.labels)
    + static_cast<std::size_t>(loaded.medoid_indices == expected.medoid_indices)
    + static_cast<std::size_t>(
      std::bit_cast<std::uint64_t>(loaded.total_cost)
      == std::bit_cast<std::uint64_t>(expected.total_cost))
    + static_cast<std::size_t>(loaded.iterations == expected.iterations)
    + static_cast<std::size_t>(loaded.converged == expected.converged);

  const fs::path resaved = scratch.file("resaved.bin");
  dtwc::save_binary_checkpoint(loaded, resaved);
  const auto resaved_bytes = read_bytes(resaved);
  const auto resaved_byte_matches = matching_bytes(resaved_bytes, valid_wire);
  CHECK(resaved_bytes.size() == valid_wire.size());
  CHECK(resaved_byte_matches == valid_wire.size());

  const Result sentinel = sentinel_result();
  Result missing_destination = sentinel;
  CHECK_FALSE(dtwc::load_binary_checkpoint(
    missing_destination, scratch.file("missing.bin")));
  CHECK(same_result(missing_destination, sentinel));

  const fs::path blocked_parent = scratch.file("blocked-parent");
  write_bytes(blocked_parent, std::array<std::uint8_t, 1>{ 0 });
  bool writer_threw_io_error = false;
  bool writer_threw_other = false;
  try {
    dtwc::save_binary_checkpoint(expected, blocked_parent / "state.bin");
  } catch (const dtwc::IOError &) {
    writer_threw_io_error = true;
  } catch (...) {
    writer_threw_other = true;
  }
  CHECK(writer_threw_io_error);
  CHECK_FALSE(writer_threw_other);
  CHECK_FALSE(fs::exists(blocked_parent / "state.bin"));

  std::size_t semantic_compatibility = 0;
  auto semantic_fixtures = semantic_compatibility_results();
  REQUIRE(semantic_fixtures.size() == 7);
  for (std::size_t index = 0; index < semantic_fixtures.size(); ++index) {
    const auto &[name, source] = semantic_fixtures[index];
    const fs::path path = scratch.file(
      "semantic-" + std::to_string(index) + ".bin");
    dtwc::save_binary_checkpoint(source, path);
    Result roundtrip;
    const bool read_ok = dtwc::load_binary_checkpoint(roundtrip, path);
    const bool fields_ok = same_result(roundtrip, source);
    CAPTURE(name);
    CHECK(read_ok);
    CHECK(fields_ok);
    if (read_ok && fields_ok)
      ++semantic_compatibility;
  }
  CHECK(semantic_compatibility == 7);

  std::size_t coherent_medoids = 0;
  for (std::size_t slot = 0; slot < expected.medoid_indices.size(); ++slot) {
    const auto medoid = static_cast<std::size_t>(expected.medoid_indices[slot]);
    const bool coherent =
      medoid < expected.labels.size()
      && expected.labels[medoid] == static_cast<int>(slot);
    CHECK(coherent);
    coherent_medoids += static_cast<std::size_t>(coherent);
  }

  CHECK(field_matches == 5);
  CHECK(coherent_medoids == 3);
}

TEST_CASE("binary-v1 checkpoint rejects the registered malformed wire corpus",
          "[checkpoint][binary][transaction][f51]")
{
  ScratchDirectory scratch{ "corruptions" };
  const Result sentinel = sentinel_result();
  const auto corpus = corruption_corpus();
  REQUIRE(corpus.size() == 85);

  std::array<std::size_t,
             static_cast<std::size_t>(CorruptionClass::count)>
    class_counts{};
  for (const auto &corruption : corpus)
    ++class_counts[static_cast<std::size_t>(corruption.kind)];
  const std::array<std::size_t, 9> expected_class_counts{
    72, 2, 2, 3, 2, 1, 1, 1, 1
  };
  REQUIRE(class_counts == expected_class_counts);

  std::size_t returned_false = 0;
  std::size_t accepted = 0;
  std::size_t threw = 0;
  std::size_t unchanged = 0;

  for (std::size_t index = 0; index < corpus.size(); ++index) {
    const auto &corruption = corpus[index];
    const fs::path path =
      scratch.file("case-" + std::to_string(index) + ".bin");
    write_bytes(path, corruption.bytes);

    Result destination = sentinel;
    bool returned = false;
    bool accepted_case = false;
    bool threw_case = false;
    std::string exception;
    try {
      accepted_case = dtwc::load_binary_checkpoint(destination, path);
      returned = true;
    } catch (const std::exception &error) {
      threw_case = true;
      exception = error.what();
    } catch (...) {
      threw_case = true;
      exception = "non-std exception";
    }

    returned_false += static_cast<std::size_t>(returned && !accepted_case);
    accepted += static_cast<std::size_t>(returned && accepted_case);
    threw += static_cast<std::size_t>(threw_case);
    const bool unchanged_case = same_result(destination, sentinel);
    unchanged += static_cast<std::size_t>(unchanged_case);

    CAPTURE(index, corruption.name, exception);
    CHECK_FALSE(accepted_case);
    CHECK_FALSE(threw_case);
    CHECK(unchanged_case);
  }

  auto preflight_bytes = oracle_vector();
  store_le_i32(preflight_bytes, 8, 257);
  const fs::path preflight_path = scratch.file("size-preflight-k257.bin");
  write_bytes(preflight_path, preflight_bytes);

  Result preflight_destination = sentinel;
  bool preflight_returned = false;
  bool preflight_accepted = false;
  bool preflight_threw = false;
  {
    ExactAllocationProbe probe;
    try {
      preflight_accepted =
        dtwc::load_binary_checkpoint(preflight_destination, preflight_path);
      preflight_returned = true;
    } catch (...) {
      preflight_threw = true;
    }
  }
  const auto exact_allocations =
    allocation_probe::exact_allocations.load(std::memory_order_relaxed);
  const bool preflight_unchanged =
    same_result(preflight_destination, sentinel);

  CHECK(preflight_returned);
  CHECK_FALSE(preflight_accepted);
  CHECK_FALSE(preflight_threw);
  CHECK(preflight_unchanged);
  CHECK(exact_allocations == 0);

  std::cout
    << "F51_RED_OBSERVATION corpus=" << corpus.size()
    << " false=" << returned_false
    << " accepted=" << accepted
    << " threw=" << threw
    << " unchanged=" << unchanged << '/' << corpus.size() << '\n';
  std::cout
    << "F51_SIZE_PREFLIGHT_OBSERVATION false="
    << static_cast<std::size_t>(preflight_returned && !preflight_accepted)
    << " threw=" << static_cast<std::size_t>(preflight_threw)
    << " unchanged="
    << static_cast<std::size_t>(preflight_unchanged) << "/1"
    << " allocations_1028=" << exact_allocations << '\n';

  const bool corpus_green =
    returned_false == 85
    && accepted == 0
    && threw == 0
    && unchanged == 85;
  const bool preflight_green =
    preflight_returned
    && !preflight_accepted
    && !preflight_threw
    && preflight_unchanged
    && exact_allocations == 0;
  // The sibling valid-wire case independently asserts every count named by
  // this marker. Catch2 randomizes case order, so marker publication cannot
  // depend on mutable cross-case state.
  if (corpus_green && preflight_green) {
    std::cout
      << "F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 "
         "unchanged=85/85 size_preflight=1/1 valid_bytes=72/72 fields=5/5 "
         "resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS\n";
  }
}
