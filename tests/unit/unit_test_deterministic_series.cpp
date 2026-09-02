/**
 * @file unit_test_deterministic_series.cpp
 * @brief F15 byte and reachability contract for shared deterministic support.
 */

#include <dtwc.hpp>
#include <core/sha256.hpp>

#include "gpu_fixed_band_oracle.hpp"
#include "../support/deterministic_series.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef DTWC_F15_SOURCE_ROOT
#error "DTWC_F15_SOURCE_ROOT must name the repository source root"
#endif

namespace fs = std::filesystem;

namespace {

// One registered value per generator schedule. The helper draws through
// std::mt19937 (bit-exact by [rand.eng.mers]) and converts with exact
// power-of-two arithmetic, so these bytes hold on every conforming C++20
// implementation regardless of standard library or floating-point flags.
constexpr std::string_view kScalarHash =
  "61AB689BE59BF61D2BA25B45F78AE17E78B4F2419468C2D6610A45648179DDAA";
constexpr std::string_view kRowsHash =
  "274D8720FE3233369FE21B8972F41AF3AC04D0D7424B55CC14E250A2719482F0";
constexpr std::string_view kAcceleratorHash =
  "7BE394A68EC73B6097B38F6965E9BE03DEA4030191D8BAE6692A3B42A473214A";

constexpr std::array<std::uint64_t, 5> kScalarBits{
  UINT64_C(0xBFD00F11C3415C28), UINT64_C(0x3FECD880D177ACA8),
  UINT64_C(0x3FDDB1FA3C799D44), UINT64_C(0x3FC941AEB3196580),
  UINT64_C(0xBFE603CA646EEF3E),
};

constexpr std::array<std::uint64_t, 12> kRowsBits{
  UINT64_C(0x3FB6392C2AF436C0), UINT64_C(0xBFDC5E645968420C),
  UINT64_C(0xBFC352D0AF27DFF8), UINT64_C(0x3FE610CFE93057BC),
  UINT64_C(0x3FA0CACD46B9A300), UINT64_C(0x3FC217455E7C63E8),
  UINT64_C(0xBFEE2D7A701FD9EE), UINT64_C(0xBFE505CA07FD467E),
  UINT64_C(0x3FC901A90CB8CA58), UINT64_C(0x3FD686BD2766D08C),
  UINT64_C(0xBFD9B122CB0964CC), UINT64_C(0x3FDD88E5A84510FC),
};

constexpr std::array<std::uint64_t, 12> kAcceleratorBits{
  UINT64_C(0xC00412D63411B332), UINT64_C(0x4022075082EACBE9),
  UINT64_C(0x40128F3C65CC024A), UINT64_C(0x3FFF921A5FDFBEE0),
  UINT64_C(0xC01B84BCFD8AAB0E), UINT64_C(0xC01B853B73001070),
  UINT64_C(0xC021AD394BB42953), UINT64_C(0x401D4B4997564B44),
  UINT64_C(0x40002DABBEDB7A61), UINT64_C(0x4010A5538E824645),
  UINT64_C(0xC0232D36FBB7CB0F), UINT64_C(0x4022CBE07B9C288E),
};

void update_little_endian_double(
    dtwc::core::detail::Sha256 &sha, double value)
{
  const auto bits = std::bit_cast<std::uint64_t>(value);
  std::array<std::uint8_t, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::uint8_t>(bits >> (8 * i));
  sha.update(bytes);
}

std::string finish_hex(const dtwc::core::detail::Sha256 &sha)
{
  constexpr std::string_view digits = "0123456789ABCDEF";
  const auto digest = sha.digest();
  std::string hex;
  hex.reserve(2 * digest.size());
  for (const auto byte : digest) {
    hex.push_back(digits[byte >> 4]);
    hex.push_back(digits[byte & 0x0f]);
  }
  return hex;
}

std::string ieee_sha256(const std::vector<double> &values)
{
  dtwc::core::detail::Sha256 sha;
  for (const double value : values)
    update_little_endian_double(sha, value);
  return finish_hex(sha);
}

std::string ieee_sha256(
    const std::vector<std::vector<double>> &series)
{
  dtwc::core::detail::Sha256 sha;
  for (const auto &row : series)
    for (const double value : row)
      update_little_endian_double(sha, value);
  return finish_hex(sha);
}

std::string read_source(std::string_view relative)
{
  const fs::path root{DTWC_F15_SOURCE_ROOT};
  if (!root.is_absolute())
    throw std::runtime_error("F15 source root is not absolute");
  const fs::path path = root / fs::path(relative);
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("cannot read F15 consumer: " + path.string());
  return {
    std::istreambuf_iterator<char>(input),
    std::istreambuf_iterator<char>()};
}

bool contains(std::string_view text, std::string_view token)
{
  return text.find(token) != std::string_view::npos;
}

std::size_t count_occurrences(
    std::string_view text, std::string_view token)
{
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = text.find(token, offset)) != std::string_view::npos) {
    ++count;
    offset += token.size();
  }
  return count;
}

std::vector<double> independent_matrix(
    const std::vector<std::vector<double>> &series, int band)
{
  namespace oracle = dtwc::test::gpu_fixed_band;
  const std::size_t count = series.size();
  std::vector<double> matrix(count * count, 0.0);
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < count; ++j) {
      if (i == j) continue;
      matrix[i * count + j] =
        oracle::full_matrix_oracle(series[i], series[j], band, false);
    }
  }
  return matrix;
}

} // namespace

TEST_CASE("F15 benchmark scalar bytes retain the registered schedule",
          "[f15][test_support][scalar]")
{
  const auto series = dtwc::test_support::benchmark_series(5, 42);

  REQUIRE(series.size() == kScalarBits.size());
  CHECK(ieee_sha256(series) == kScalarHash);
  CHECK(series == dtwc::test_support::benchmark_series(5, 42));
  CHECK(series != dtwc::test_support::benchmark_series(5, 43));
  for (std::size_t i = 0; i < series.size(); ++i) {
    CAPTURE(i);
    CHECK(std::bit_cast<std::uint64_t>(series[i]) == kScalarBits[i]);
    CHECK(series[i] >= -1.0);
    CHECK(series[i] < 1.0);
  }
}

TEST_CASE("F15 benchmark rows use a fresh base-plus-row engine",
          "[f15][test_support][row_seeded]")
{
  const auto rows = dtwc::test_support::benchmark_series_set(3, 4, 100);

  REQUIRE(rows.size() == 3);
  CHECK(ieee_sha256(rows) == kRowsHash);
  std::size_t offset = 0;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    CAPTURE(row);
    CHECK(rows[row].size() == 4);
    CHECK(rows[row] == dtwc::test_support::benchmark_series(
      4, 100 + static_cast<unsigned>(row)));
    for (const double value : rows[row]) {
      CAPTURE(offset);
      CHECK(std::bit_cast<std::uint64_t>(value) == kRowsBits[offset]);
      ++offset;
    }
  }
  CHECK(offset == kRowsBits.size());
  CHECK(rows[0] != rows[1]);
  CHECK(rows[1] != rows[2]);
  CHECK(rows != dtwc::test_support::benchmark_series_set(3, 4, 101));
}

TEST_CASE("F15 accelerator bytes are toolchain independent",
          "[f15][test_support][continuous]")
{
  const auto series =
    dtwc::test_support::accelerator_series_set(3, 4, 42);
  REQUIRE(series.size() == 3);
  for (const auto &row : series)
    CHECK(row.size() == 4);

  CHECK(ieee_sha256(series) == kAcceleratorHash);
  std::size_t offset = 0;
  for (const auto &row : series) {
    for (const double value : row) {
      CAPTURE(offset);
      CHECK(std::bit_cast<std::uint64_t>(value) == kAcceleratorBits[offset]);
      CHECK(value >= -10.0);
      CHECK(value < 10.0);
      ++offset;
    }
  }
  CHECK(offset == kAcceleratorBits.size());
}

TEST_CASE("F15 dense assembly calls only the ordered upper triangle",
          "[f15][test_support][dense]")
{
  const std::vector<std::vector<int>> series{{0}, {1}, {2}, {3}};
  std::vector<std::pair<int, int>> calls;
  const auto matrix = dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [&](const auto &left, const auto &right) {
      calls.emplace_back(left[0], right[0]);
      return 10 * left[0] + right[0];
    });
  const std::vector<std::pair<int, int>> expected_calls{
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
  };

  REQUIRE(matrix.size() == 16);
  CHECK(calls == expected_calls);
  CHECK(calls.size() == 6);
  for (std::size_t i = 0; i < series.size(); ++i)
    CHECK(matrix[i * series.size() + i] == 0);
  for (const auto [left, right] : expected_calls) {
    const int expected = 10 * left + right;
    CHECK(matrix[static_cast<std::size_t>(left) * 4
                 + static_cast<std::size_t>(right)] == expected);
    CHECK(matrix[static_cast<std::size_t>(right) * 4
                 + static_cast<std::size_t>(left)] == expected);
  }
}

TEST_CASE("F15 production dense references equal an independent full-matrix DP",
          "[f15][test_support][oracle]")
{
  // DTW outputs depend on the build floating-point flags, so the contract here
  // is agreement with an independent DP inside the same build, never a
  // registered cross-toolchain hash.
  const auto series =
    dtwc::test_support::accelerator_series_set(3, 4, 42);
  REQUIRE(ieee_sha256(series) == kAcceleratorHash);

  const auto full = dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [](const auto &left, const auto &right) {
      return dtwc::dtwFull_L<double>(left, right);
    });
  const auto band0 = dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [](const auto &left, const auto &right) {
      return dtwc::dtwBanded<double>(left, right, 0);
    });
  const auto independent_full = independent_matrix(series, -1);
  const auto independent_band0 = independent_matrix(series, 0);

  REQUIRE(full.size() == 9);
  REQUIRE(band0.size() == 9);
  CHECK(full == independent_full);
  CHECK(band0 == independent_band0);
  CHECK(full != band0);
  for (std::size_t i = 0; i < full.size(); ++i) {
    CAPTURE(i);
    CHECK(std::bit_cast<std::uint64_t>(full[i])
          == std::bit_cast<std::uint64_t>(independent_full[i]));
    CHECK(std::bit_cast<std::uint64_t>(band0[i])
          == std::bit_cast<std::uint64_t>(independent_band0[i]));
  }
}

TEST_CASE("F15 all registered consumers reach shared support",
          "[f15][test_support][source_audit]")
{
  struct Consumer
  {
    std::string_view path;
    std::size_t shared_calls;
  };
  constexpr std::array benchmark_consumers{
    Consumer{"benchmarks/bench_cuda_dtw.cpp", 2},
    Consumer{"benchmarks/bench_dtw_baseline.cpp", 1},
    Consumer{"benchmarks/bench_metal_dtw.cpp", 1},
    Consumer{"benchmarks/bench_mmap_access.cpp", 1},
    Consumer{"benchmarks/bench_mpi_dtw.cpp", 1},
  };
  for (const auto &consumer : benchmark_consumers) {
    INFO(consumer.path);
    const std::string source = read_source(consumer.path);
    CHECK(contains(source, "tests/support/deterministic_series.hpp"));
    CHECK(count_occurrences(
      source, "dtwc::test_support::benchmark_series_set")
      == consumer.shared_calls);
    CHECK_FALSE(contains(
      source, "static std::vector<double> random_series"));
  }

  constexpr std::array accelerator_consumers{
    Consumer{"tests/unit/test_cuda_correctness.cpp", 46},
    Consumer{"tests/unit/test_cuda_lb_keogh.cpp", 7},
    Consumer{"tests/unit/test_metal_correctness.cpp", 18},
  };
  for (const auto &consumer : accelerator_consumers) {
    INFO(consumer.path);
    const std::string source = read_source(consumer.path);
    CHECK(contains(source, "support/deterministic_series.hpp"));
    CHECK(contains(source, "dtwc::test_support::accelerator_series_set"));
    CHECK(count_occurrences(source, "generate_random_series(")
          == consumer.shared_calls);
    CHECK_FALSE(contains(
      source,
      "std::vector<std::vector<double>> generate_random_series("));
  }

  constexpr std::array dense_consumers{
    Consumer{"tests/unit/test_cuda_correctness.cpp", 2},
    Consumer{"tests/unit/test_metal_correctness.cpp", 2},
    Consumer{"tests/unit/test_metal_lb_keogh.cpp", 1},
    Consumer{"tests/unit/test_cuda_kernel_override.cpp", 1},
    Consumer{"tests/unit/unit_test_mpi.cpp", 2},
  };
  for (const auto &consumer : dense_consumers) {
    INFO(consumer.path);
    const std::string source = read_source(consumer.path);
    CHECK(contains(source, "support/deterministic_series.hpp"));
    CHECK(count_occurrences(
      source, "dtwc::test_support::symmetric_zero_diagonal_matrix")
      == consumer.shared_calls);
  }

  const std::string mpi = read_source("tests/unit/unit_test_mpi.cpp");
  CHECK(contains(mpi, "dtwc::test_support::benchmark_series"));
  CHECK_FALSE(contains(
    mpi, "static std::vector<double> make_series"));

  const std::string metal_lb =
    read_source("tests/unit/test_metal_lb_keogh.cpp");
  CHECK(contains(
    metal_lb,
    "std::uniform_real_distribution<double> dist(-5.0, 5.0)"));
  CHECK(contains(
    metal_lb,
    "std::vector<std::vector<double>> random_series("));

  const std::string fixed_band =
    read_source("tests/unit/gpu_fixed_band_oracle.hpp");
  CHECK(contains(fixed_band, "full_matrix_oracle"));
  CHECK(contains(fixed_band, "enumerate_paths"));
  CHECK_FALSE(contains(fixed_band, "deterministic_series.hpp"));

  // The shared generator must not reintroduce an implementation-defined
  // real-value mapping; that is what made the registered bytes unportable.
  const std::string support =
    read_source("tests/support/deterministic_series.hpp");
  CHECK(contains(support, "std::mt19937"));
  // Both spellings: CTAD (`uniform_real_distribution dist(...)`) drops the
  // angle bracket, and generate_canonical is the same unportable mapping.
  CHECK_FALSE(contains(support, "uniform_real_distribution"));
  CHECK_FALSE(contains(support, "generate_canonical"));

  REQUIRE(ieee_sha256(dtwc::test_support::benchmark_series(5, 42))
          == kScalarHash);
  REQUIRE(ieee_sha256(dtwc::test_support::benchmark_series_set(3, 4, 100))
          == kRowsHash);
  REQUIRE(ieee_sha256(dtwc::test_support::accelerator_series_set(3, 4, 42))
          == kAcceleratorHash);
  std::cout
    << "F15_TEST_SUPPORT generator=portable scalar=ran row_seeded=ran"
       " continuous=ran dense=ran source_audit=ran skips=0\n";
}
