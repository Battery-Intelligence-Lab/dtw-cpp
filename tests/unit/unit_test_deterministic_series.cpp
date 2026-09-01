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

constexpr std::string_view kWindowsScalarHash =
  "70CC88A06F050E253AF62FF5A73D2AE2D659A287A286D88EEB1871C7C45E13EB";
constexpr std::string_view kWindowsRowsHash =
  "B930ACBCF449361A1370E185F5610D9B4BDEBD71FAC07772C3E3449AB29C1733";
constexpr std::string_view kLibstdcxxScalarHash =
  "194FB0E76C52FCD84F09960547EEDC6A43788FDCC89F739DF44E3C49AF7B16E0";
constexpr std::string_view kLibstdcxxRowsHash =
  "1F9E6847BA0FFC7943EBCA024827CD6B5A890B8911BFF4F6C59211F3C28892AB";

constexpr std::array<std::uint64_t, 5> kWindowsScalarBits{
  UINT64_C(0x3FE2FA8F6CD7F876), UINT64_C(0xBFE4429ABC432780),
  UINT64_C(0x3FE1E67511EED8FC), UINT64_C(0x3FC8CB2C149941A8),
  UINT64_C(0xBFBBBBCF0DB01E60),
};

constexpr std::array<std::uint64_t, 5> kLibstdcxxScalarBits{
  UINT64_C(0x3FE2FA8F6CD7F878), UINT64_C(0xBFE4429ABC43277F),
  UINT64_C(0x3FE1E67511EED8FE), UINT64_C(0x3FC8CB2C149941B0),
  UINT64_C(0xBFBBBBCF0DB01E50),
};

struct Profile
{
  std::string_view name;
  std::string_view accelerator_hash;
  std::string_view full_hash;
  std::string_view band0_hash;
  std::array<std::uint64_t, 12> accelerator_bits;
};

constexpr Profile kRelaxed{
  "relaxed",
  "BFACE25683F745B965459B36DDA75C9C4EDA8035EF4AFBED4722BC2E1F34758B",
  "700163C5AC813BABD00845295CEEC6D3FD2C8DD13CEE7A918B442EE2440549B3",
  "494C9E299092EED43414A2ABEA110CC4A05D579F3B8162257D4D637F45C933EB",
  {
    UINT64_C(0x4017B933480DF694), UINT64_C(0xC01953416B53F160),
    UINT64_C(0x40166012566A8F3B), UINT64_C(0x3FFEFDF719BF9212),
    UINT64_C(0xBFF15561688E12FC), UINT64_C(0xC0200041BE89C29E),
    UINT64_C(0xBFEA14A986DAD398), UINT64_C(0xC00A9B4B96F5696D),
    UINT64_C(0xC01C92166F9FE92C), UINT64_C(0x4008246451B14AA3),
    UINT64_C(0xC021BE586FDF2D38), UINT64_C(0x4011C288EB7D97C0),
  }
};

constexpr Profile kPrecise{
  "precise",
  "C53236B99DC783C3AC0129B58652C25C95ED72F7922A39B369A9E89AF94824B0",
  "E79B3ACEAF951A614278879049CA66B887AEC388754DA9005308897ACB12526C",
  "2405A565FEEBAF9998D3A1A778105911EC8DEA996A6EE5EFC58A9997D8432ABD",
  {
    UINT64_C(0x4017B933480DF694), UINT64_C(0xC01953416B53F160),
    UINT64_C(0x40166012566A8F3C), UINT64_C(0x3FFEFDF719BF9210),
    UINT64_C(0xBFF15561688E1300), UINT64_C(0xC0200041BE89C29E),
    UINT64_C(0xBFEA14A986DAD3A0), UINT64_C(0xC00A9B4B96F5696C),
    UINT64_C(0xC01C92166F9FE92C), UINT64_C(0x4008246451B14AA4),
    UINT64_C(0xC021BE586FDF2D38), UINT64_C(0x4011C288EB7D97C0),
  }
};

constexpr Profile kLibstdcxx{
  "libstdcxx",
  "5D3594B036ED60CAA686D8472C630488B86290BAD1805806FBB38894B96F2C53",
  "7DE312EABCFFB71D857BF97B9CFCE9C08A6F855E7CE25B863342BE46CBC38B73",
  "E81034CE0654315D254D07FA518DDCE7472A542A4EECBD740935E9DFF891022E",
  {
    UINT64_C(0x4017B933480DF696), UINT64_C(0xC01953416B53F15F),
    UINT64_C(0x40166012566A8F3E), UINT64_C(0x3FFEFDF719BF9220),
    UINT64_C(0xBFF15561688E12F0), UINT64_C(0xC0200041BE89C29E),
    UINT64_C(0xBFEA14A986DAD3A0), UINT64_C(0xC00A9B4B96F5696C),
    UINT64_C(0xC01C92166F9FE92A), UINT64_C(0x4008246451B14AA8),
    UINT64_C(0xC021BE586FDF2D37), UINT64_C(0x4011C288EB7D97C0),
  }
};

constexpr Profile kLibcxx{
  "libcxx",
  "1D063EF12CB8680807CEEEF9F8C2F35331D0AA91186F824A16D9F3B2954AEF77",
  "7DE312EABCFFB71D857BF97B9CFCE9C08A6F855E7CE25B863342BE46CBC38B73",
  "E81034CE0654315D254D07FA518DDCE7472A542A4EECBD740935E9DFF891022E",
  {
    UINT64_C(0x4017B933480DF696), UINT64_C(0xC01953416B53F15F),
    UINT64_C(0x40166012566A8F3E), UINT64_C(0x3FFEFDF719BF921C),
    UINT64_C(0xBFF15561688E12F2), UINT64_C(0xC0200041BE89C29E),
    UINT64_C(0xBFEA14A986DAD398), UINT64_C(0xC00A9B4B96F5696D),
    UINT64_C(0xC01C92166F9FE92A), UINT64_C(0x4008246451B14AA8),
    UINT64_C(0xC021BE586FDF2D37), UINT64_C(0x4011C288EB7D97C0),
  }
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

const Profile *profile_for(std::string_view accelerator_hash)
{
  if (accelerator_hash == kRelaxed.accelerator_hash) return &kRelaxed;
  if (accelerator_hash == kPrecise.accelerator_hash) return &kPrecise;
  if (accelerator_hash == kLibstdcxx.accelerator_hash) return &kLibstdcxx;
  if (accelerator_hash == kLibcxx.accelerator_hash) return &kLibcxx;
  return nullptr;
}

const Profile *coherent_profile(
    std::string_view accelerator_hash,
    std::string_view full_hash,
    std::string_view band0_hash)
{
  for (const Profile *profile :
       {&kRelaxed, &kPrecise, &kLibstdcxx, &kLibcxx}) {
    if (accelerator_hash == profile->accelerator_hash
        && full_hash == profile->full_hash
        && band0_hash == profile->band0_hash) {
      return profile;
    }
  }
  return nullptr;
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
  const std::string hash = ieee_sha256(series);
  const auto *expected =
    (hash == kWindowsScalarHash) ? &kWindowsScalarBits
    : (hash == kLibstdcxxScalarHash) ? &kLibstdcxxScalarBits
    : nullptr;

  REQUIRE(expected != nullptr);
  REQUIRE(series.size() == expected->size());
  CHECK(series == dtwc::test_support::benchmark_series(5, 42));
  CHECK(series != dtwc::test_support::benchmark_series(5, 43));
  for (std::size_t i = 0; i < series.size(); ++i) {
    CAPTURE(i);
    CHECK(std::bit_cast<std::uint64_t>(series[i]) == (*expected)[i]);
    CHECK(series[i] >= -1.0);
    CHECK(series[i] <= 1.0);
  }
}

TEST_CASE("F15 benchmark rows use a fresh base-plus-row engine",
          "[f15][test_support][row_seeded]")
{
  const auto rows = dtwc::test_support::benchmark_series_set(3, 4, 100);
  const std::string scalar_hash =
    ieee_sha256(dtwc::test_support::benchmark_series(5, 42));
  const std::string rows_hash = ieee_sha256(rows);
  REQUIRE(rows.size() == 3);
  CHECK((
    (scalar_hash == kWindowsScalarHash
     && rows_hash == kWindowsRowsHash)
    || (scalar_hash == kLibstdcxxScalarHash
        && rows_hash == kLibstdcxxRowsHash)));
  for (std::size_t row = 0; row < rows.size(); ++row) {
    CAPTURE(row);
    CHECK(rows[row].size() == 4);
    CHECK(rows[row] == dtwc::test_support::benchmark_series(
      4, 100 + static_cast<unsigned>(row)));
  }
  CHECK(rows[0] != rows[1]);
  CHECK(rows[1] != rows[2]);
  CHECK(rows != dtwc::test_support::benchmark_series_set(3, 4, 101));
}

TEST_CASE("F15 accelerator bytes select one coherent compiler profile",
          "[f15][test_support][continuous]")
{
  const auto series =
    dtwc::test_support::accelerator_series_set(3, 4, 42);
  REQUIRE(series.size() == 3);
  for (const auto &row : series)
    CHECK(row.size() == 4);

  const std::string hash = ieee_sha256(series);
  const Profile *profile = profile_for(hash);
  REQUIRE(profile != nullptr);
  CHECK(hash == profile->accelerator_hash);

  std::size_t offset = 0;
  for (const auto &row : series) {
    for (const double value : row) {
      CAPTURE(offset, profile->name);
      CHECK(std::bit_cast<std::uint64_t>(value)
            == profile->accelerator_bits[offset]);
      CHECK(value >= -10.0);
      CHECK(value <= 10.0);
      ++offset;
    }
  }
  CHECK(offset == profile->accelerator_bits.size());
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
  const auto series =
    dtwc::test_support::accelerator_series_set(3, 4, 42);
  const Profile *profile = profile_for(ieee_sha256(series));
  REQUIRE(profile != nullptr);

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
  const std::string accelerator_hash = ieee_sha256(series);
  const std::string full_hash = ieee_sha256(full);
  const std::string band0_hash = ieee_sha256(band0);
  const Profile *coherent =
    coherent_profile(accelerator_hash, full_hash, band0_hash);

  REQUIRE(full.size() == 9);
  REQUIRE(band0.size() == 9);
  REQUIRE(coherent != nullptr);
  CHECK(coherent == profile);
  CHECK(full == independent_full);
  CHECK(band0 == independent_band0);
  CHECK(full != band0);
  CHECK(full_hash == profile->full_hash);
  CHECK(band0_hash == profile->band0_hash);
  for (std::size_t i = 0; i < full.size(); ++i) {
    CAPTURE(i, profile->name);
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

  const auto series =
    dtwc::test_support::accelerator_series_set(3, 4, 42);
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
  const Profile *profile = coherent_profile(
    ieee_sha256(series), ieee_sha256(full), ieee_sha256(band0));
  REQUIRE(profile != nullptr);
  std::cout
    << "F15_TEST_SUPPORT profile=" << profile->name
    << " scalar=ran row_seeded=ran continuous=ran dense=ran"
       " source_audit=ran skips=0\n";
}
