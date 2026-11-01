/**
 * @file test_ipc_stream.cpp
 * @brief FastCLARA's stream of an Arrow IPC file a record batch at a time (algorithms::detail::fast_clara_chunked over
 *        io::IpcFile, and run's planner, in every build): the in-RAM run's labels, medoids and cost, and the CSV
 *        twin's; the Parquet stream's verdicts, the cases' texts being those its serial check gave on their Parquet
 *        twins (2026-11-01, at dd7ba144); a record batch larger than the limit, and a limit below the union beside its
 *        decode, refused before any batch is decoded. Fixtures: tests/data/reader/make_ipc_fixtures.py.
 * @author Volkan Kumtepeli
 */

#include <Problem.hpp>
#include <algorithms/detail/fast_clara_plan.hpp>
#include <algorithms/detail/fast_clara_stream.hpp>
#include <algorithms/fast_clara.hpp>
#include <base/error.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/portable_random.hpp>
#include <fileOperations.hpp>
#include <io/arrow_ipc.hpp>
#include <io/read_data.hpp>

#include "../../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Message;
using Catch::Matchers::MessageMatches;
using dtwc::core::Precision;

namespace {

std::filesystem::path fixture(const char *name)
{
  return std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader" / name;
}

/// The directory of every file this binary writes: unique to the process, removed at exit.
std::filesystem::path scratch()
{
  static const dtwc::test_support::ScratchDirectory dir{ "ipc_stream_test" };
  return dir.path;
}

/// std::cout's text while `f` runs.
template <typename F>
std::string printed(F &&f)
{
  std::ostringstream out;
  auto *const previous = std::cout.rdbuf(out.rdbuf());
  try {
    f();
  } catch (...) {
    std::cout.rdbuf(previous);
    throw;
  }
  std::cout.rdbuf(previous);
  return out.str();
}

/// FastCLARA streamed from `path` a record batch at a time, its series as `dtype`, within `limit` bytes.
dtwc::core::ClusteringResult streamed(dtwc::Problem &prob, dtwc::algorithms::CLARAOptions options,
                                      const std::filesystem::path &path, Precision dtype, std::size_t limit)
{
  const dtwc::io::IpcFile file(path);
  options.ram_limit_bytes = limit;
  const auto plan = dtwc::algorithms::detail::resolve_clara_plan(file.num_rows(), options, "fast_clara");
  return dtype == Precision::Float32 ? dtwc::algorithms::detail::fast_clara_chunked<true>(prob, options, file, plan)
                                     : dtwc::algorithms::detail::fast_clara_chunked<false>(prob, options, file, plan);
}

/// The in-RAM run: the file read by dtwc::read_data as `dtype`, clustered by fast_clara.
dtwc::core::ClusteringResult in_ram(const std::filesystem::path &path, const dtwc::algorithms::CLARAOptions &options,
                                    Precision dtype)
{
  dtwc::Problem prob{ "in_ram" };
  prob.set_data(dtwc::read_data(path, 0, 0, '\0', {}, dtype));
  return dtwc::algorithms::fast_clara(prob, options);
}

std::vector<std::byte> bytes_of(const std::filesystem::path &path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> chars{ std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
  std::vector<std::byte> bytes(chars.size());
  std::memcpy(bytes.data(), chars.data(), chars.size());
  return bytes;
}

void write(const std::filesystem::path &path, const std::vector<std::byte> &bytes)
{
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(out.good());
}

/// A copy of `name` in which record batch 0's first LZ4 frame (its magic 04 22 4D 18, after the buffer's 8-byte decoded
/// length) is broken: the walk takes it, a decode of that batch is an IOError.
std::filesystem::path undecodable(const char *name)
{
  auto bytes = bytes_of(fixture(name));
  const std::byte magic[] = { std::byte{ 0x04 }, std::byte{ 0x22 }, std::byte{ 0x4D }, std::byte{ 0x18 } };
  const auto at = std::search(bytes.begin(), bytes.end(), std::begin(magic), std::end(magic));
  REQUIRE(at != bytes.end());
  *at = std::byte{ 0x05 };
  const auto path = scratch() / (std::string("undecodable_") + name);
  write(path, bytes);
  return path;
}

dtwc::Config clara_config(const std::filesystem::path &path)
{
  dtwc::Config config;
  config.input = dtwc::path_to_utf8(path);
  config.method = dtwc::Method::CLARA;
  config.k = 3;
  config.sample_size = 8;
  config.n_samples = 2;
  config.seed = 42;
  config.output.clear();
  config.verbose = true;
  return config;
}

} // namespace

TEST_CASE("Arrow IPC: FastCLARA streamed a record batch at a time gives the in-RAM run's clustering",
          "[io][arrow][streaming]")
{
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 3;
  options.sample_size = 8;
  options.n_samples = 2;
  options.random_seed = 42;
  const auto csv = in_ram(fixture("clara_256.csv"), options, Precision::Float64);
  // 256 rows in eight record batches of 32 (uncompressed, LZ4, ZSTD, Float32 LZ4): within 24,000 bytes a batch of
  // the stream holds three record batches, read by a team; each precision streamed, the file's and the other.
  for (const char *file : { "clara_256.arrow", "clara_256_lz4.arrow", "clara_256_zstd.arrow", "clara_256_f32.arrow" })
    for (const Precision dtype : { Precision::Float64, Precision::Float32 }) {
      INFO(file << (dtype == Precision::Float32 ? " as float32" : " as float64"));
      const auto expected = in_ram(fixture(file), options, dtype);
      dtwc::Problem settings_only{ "streamed" };
      const auto result = streamed(settings_only, options, fixture(file), dtype, 24000);
      CHECK(result.labels == expected.labels);
      CHECK(result.medoid_indices == expected.medoid_indices);
      CHECK(result.total_cost == expected.total_cost);
      CHECK(result.labels == csv.labels);
      CHECK(result.medoid_indices == csv.medoid_indices);
      CHECK(result.total_cost == csv.total_cost);
    }
  // S1's and S2's fixtures: one record batch of three rows (uncompressed, LZ4, ZSTD, Float32) and three batches of
  // six rows (uncompressed, LZ4, ZSTD), each a batch of the stream.
  options.n_clusters = 2;
  for (const char *file : { "ipc_no_name.arrow", "ipc_lz4.arrow", "ipc_zstd.arrow", "ipc_float32.arrow",
                            "ipc_batches.arrow", "ipc_lz4_batches.arrow", "ipc_zstd_batches.arrow" }) {
    INFO(file);
    options.sample_size = dtwc::io::IpcFile(fixture(file)).num_rows() - 1;
    const auto expected = in_ram(fixture(file), options, Precision::Float64);
    dtwc::Problem settings_only{ "streamed" };
    const auto result = streamed(settings_only, options, fixture(file), Precision::Float64, std::size_t{ 1 } << 20);
    CHECK(result.labels == expected.labels);
    CHECK(result.medoid_indices == expected.medoid_indices);
    CHECK(result.total_cost == expected.total_cost);
  }
}

TEST_CASE("Arrow IPC: run streams a file over --ram-limit through FastCLARA, as dtwc_cl does",
          "[io][arrow][streaming]")
{
  dtwc::Config csv_config = clara_config(fixture("clara_256.csv"));
  csv_config.verbose = false;
  const auto csv = dtwc::run(csv_config);
  for (const char *file : { "clara_256.arrow", "clara_256_lz4.arrow", "clara_256_zstd.arrow", "clara_256_f32.arrow" })
    for (const Precision dtype : { Precision::Float64, Precision::Float32 }) {
      INFO(file << (dtype == Precision::Float32 ? " as float32" : " as float64"));
      dtwc::Config config = clara_config(fixture(file));
      config.dtype = dtype;
      std::vector<dtwc::index_t> labels, medoids;
      double cost = 0;
      const std::string resident = printed([&] {
        const auto result = dtwc::run(config);
        labels = result.labels();
        medoids = result.medoids();
        cost = result.cost();
      });
      CHECK_THAT(resident, ContainsSubstring("Data loaded from Arrow IPC: 256 series"));
      config.ram_limit = 24000; // below what the series hold in RAM, above what the stream holds
      const std::string stream = printed([&] {
        const auto result = dtwc::run(config);
        CHECK(result.labels() == labels);
        CHECK(result.medoids() == medoids);
        CHECK(result.cost() == cost);
      });
      // A mapped file holds its values in the page cache: in RAM it holds what the stream would.
      const bool mapped = std::string(file) == "clara_256.arrow" && dtype == Precision::Float64;
#if defined(DTWC_HAS_MMAP)
      CHECK((stream.find("Arrow IPC walk selected streaming: 256 series") == std::string::npos) == mapped);
#else
      CHECK_THAT(stream, ContainsSubstring("Arrow IPC walk selected streaming: 256 series"));
      (void)mapped;
#endif
      CHECK(labels == csv.labels());
      CHECK(medoids == csv.medoids());
      CHECK(cost == csv.cost());
    }
  // The same values as two time steps of two features (ndim 2): streamed, the distance takes the file's features, as
  // in RAM, and the clustering is not the univariate walks'.
  dtwc::Config config = clara_config(fixture("clara_256_ndim2.arrow"));
  config.verbose = false;
  const auto resident = dtwc::run(config);
  config.ram_limit = 24000;
  config.verbose = true;
  const std::string stream = printed([&] {
    const auto result = dtwc::run(config);
    CHECK(result.labels() == resident.labels());
    CHECK(result.medoids() == resident.medoids());
    CHECK(result.cost() == resident.cost());
  });
  CHECK_THAT(stream, ContainsSubstring("Arrow IPC walk selected streaming: 256 series"));
  CHECK(resident.cost() != csv.cost());
}

// The cases of the Parquet stream's verdicts (test_io_readers.cpp's), in record batches read by a team: each batch's
// rows checked by the thread that read it, its RowCheck's extremes merged in batch order, the lowest failing batch's
// refusal rethrown. The texts are the Parquet stream's on each case's Parquet twin (a row group a record batch).
TEST_CASE("Arrow IPC: the stream's team gives the serial check's verdicts", "[io][arrow][streaming][mm]")
{
  using dtwc::core::MissingStrategy;
  std::mt19937_64 rng(3);
  REQUIRE(dtwc::core::portable_sample_indices<dtwc::index_t>(64, 8, rng)
          == std::vector<dtwc::index_t>{ 10, 15, 22, 23, 30, 37, 52, 55 }); // the rows the fixtures poison around
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 2;
  options.sample_size = 8;
  options.n_samples = 1;
  options.random_seed = 3;
  struct Case
  {
    const char *file;
    MissingStrategy missing;
    int band;
    const char *says;
  };
  const Case cases[] = {
    // Rows 18 and 33, unsampled, in record batches 1 and 2, read together: the lower.
    { "clara_verdict_rows.arrow", MissingStrategy::Error, -1,
      "fast_clara: series 'series_18' (index 18)[1] is NaN. NaN is a missing value only to the missing-data distances "
      "(missing, arow, or a ZeroCost, AROW or Interpolate missing strategy): use one of those, or remove it." },
    // The sampled row 52 before the lower unsampled 20: the union is checked first.
    { "clara_verdict_union.arrow", MissingStrategy::Error, -1,
      "fast_clara: series 'series_52' (index 52)[1] is -inf. Distances are defined on finite values only: replace or "
      "remove it (NaN marks a missing value for the missing-data distances)." },
    // Row 25 entirely NaN before row 45's +inf, in another record batch: refused at its row.
    { "clara_verdict_nan.arrow", MissingStrategy::Interpolate, -1,
      "fast_clara: series 'series_25' (index 25) is entirely NaN, so MissingStrategy::Interpolate has nothing to "
      "interpolate from. Use ZeroCost or AROW, or drop the series." },
    // After the pass: the first shortest row (0, of record batch 0) and the first longest (40, of batch 2; 60 of
    // batch 3 ties it).
    { "clara_verdict_band.arrow", MissingStrategy::Error, 2,
      "fast_clara: band = 2 is narrower than the length difference between series 'series_0' (index 0, length 8) and "
      "series 'series_40' (index 40, length 30), so no warping path fits that pair. The smallest feasible band is 22; "
      "pass band >= 22, or band = -1 for full DTW." },
  };
  for (const Case &c : cases) {
    INFO(c.file);
    REQUIRE(dtwc::io::IpcFile(fixture(c.file)).num_row_groups() == 4);
    dtwc::Problem prob{ "streamed" };
    prob.set_missing_strategy(c.missing);
    prob.set_band(c.band);
    // A MiB: the four record batches of the pass are one batch of the stream, read by the team together.
    CHECK_THROWS_MATCHES(streamed(prob, options, fixture(c.file), Precision::Float64, std::size_t{ 1 } << 20),
                         dtwc::InvalidInput, Message(c.says));
  }
  // Two features a time step (ndim 2): a row whose values are not whole time steps is refused with the in-RAM route's
  // text (Data::validate_ndim's, which the reader raises), Parquet holding no such file. The stream's Problem holds an
  // empty Data of the file's ndim, as run gives it; without it the distance would be univariate: a logic error.
  const auto cut = fixture("clara_verdict_ndim.arrow");
  const std::string in_ram = [&] {
    try {
      dtwc::Problem prob{ "in_ram" };
      prob.set_data(dtwc::read_data(cut));
    } catch (const dtwc::InvalidInput &refused) {
      return std::string(refused.what());
    }
    FAIL("the in-RAM route took row 41");
    return std::string();
  }();
  CHECK(in_ram == "Series 41 has flat size 7 which is not divisible by ndim=2");
  dtwc::Problem prob{ "streamed" };
  CHECK_THROWS_AS(streamed(prob, options, fixture("clara_256_ndim2.arrow"), Precision::Float64, std::size_t{ 1 } << 20),
                  std::logic_error);
  prob.set_data(dtwc::Data(std::vector<std::vector<double>>{}, std::vector<std::string>{}, 2));
  CHECK_THROWS_MATCHES(streamed(prob, options, cut, Precision::Float64, std::size_t{ 1 } << 20), dtwc::InvalidInput,
                       Message(in_ram));
}

TEST_CASE("Arrow IPC: the stream refuses, before any record batch is decoded, a limit it cannot read within",
          "[io][arrow][streaming][ram]")
{
  // ipc_lz4.arrow is one record batch: streamed under a limit that holds its names and labels but not the batch, the
  // refusal names the batch's bytes, before any batch is decoded (record batch 0 of this copy cannot be).
  const auto one = undecodable("ipc_lz4.arrow");
  CHECK_THROWS_AS(dtwc::read_data(one), dtwc::IOError); // its decode fails
  const std::size_t batch = dtwc::io::IpcFile(one).largest_row_group_bytes(false);
  dtwc::Config config = clara_config(one);
  config.k = 2;
  config.sample_size = 2;
  config.verbose = false;
  config.ram_limit = 200;
  REQUIRE(batch > config.ram_limit);
  CHECK_THROWS_MATCHES(
    dtwc::run(config), dtwc::InvalidInput,
    Message("io::IpcFile::read_rows: record batch 0 of the Arrow IPC file holds " + std::to_string(batch)
            + " bytes once read (its 3 series), more than --ram-limit=200, and FastCLARA's stream reads each record "
              "batch whole; raise the limit, or rewrite the file in smaller record batches"));

  // The union's 16 rows beside one record batch, read whole: a byte short, refused before any batch is decoded.
  const auto copy = undecodable("clara_256_lz4.arrow");
  const dtwc::io::IpcFile file(copy);
  const std::size_t largest = file.largest_row_group_bytes(false);
  const std::size_t objects = dtwc::io::IpcFile::series_bytes(0, 16, false);
  std::vector<dtwc::index_t> rows;
  for (dtwc::index_t i = 0; i < 16; ++i) rows.push_back(16 * i);
  CHECK_THROWS_AS(file.read_rows<double>(rows, std::size_t{ 1 } << 20), dtwc::IOError); // record batch 0 is read
  CHECK_THROWS_MATCHES(
    file.read_rows<double>(rows, objects + largest - 1), dtwc::InvalidInput,
    Message("io::IpcFile::read_rows: the 16 selected series hold " + std::to_string(objects)
            + " bytes so far and reading a record batch of their rows needs " + std::to_string(largest) + " more, "
            + std::to_string(objects + largest) + " in all, exceeding --ram-limit="
            + std::to_string(objects + largest - 1) + "; raise the limit"));
}

TEST_CASE("Arrow IPC: the planner prices a mapped view at its views, and the file read without llfio whole",
          "[io][arrow][streaming][ram]")
{
  // 256 rows in RAM: a view's 46 bytes a row (its values the page cache's), the compressed file's decoded batches, and
  // without llfio the whole file beside them.
  const dtwc::io::IpcFile plain(fixture("clara_256.arrow")), lz4(fixture("clara_256_lz4.arrow"));
  const auto size = static_cast<std::size_t>(std::filesystem::file_size(fixture("clara_256.arrow")));
#if defined(DTWC_HAS_MMAP)
  CHECK(plain.resident_bytes(Precision::Float64, false) == 256 * dtwc::detail::view_series_bytes);
  CHECK_FALSE(plain.streams(Precision::Float64));
#else
  CHECK(plain.resident_bytes(Precision::Float64, false) == 256 * dtwc::detail::view_series_bytes + size);
  CHECK(plain.streams(Precision::Float64));
#endif
  CHECK(plain.streams(Precision::Float32)); // its copy in Float32
  CHECK(lz4.streams(Precision::Float64));
  CHECK(lz4.resident_bytes(Precision::Float64, false) > 256 * dtwc::detail::view_series_bytes);
  // A GPU uploads a view's copy.
  CHECK(plain.resident_bytes(Precision::Float64, true)
        == plain.resident_bytes(Precision::Float64, false) + dtwc::io::IpcFile::series_bytes(1024, 256, false)
             + plain.series_names_bytes() - 256 * sizeof(std::string));
  // The automatic limit (none without llfio) streams no file this small and refuses none: in RAM.
  dtwc::Config config = clara_config(fixture("clara_256_lz4.arrow"));
  CHECK_THAT(printed([&] { (void)dtwc::run(config); }), ContainsSubstring("Data loaded from Arrow IPC: 256 series"));
  // Beyond an explicit limit the other methods are refused, naming the bytes, the limit and CLARA.
  config.method = dtwc::Method::PAM;
  config.ram_limit = 16000;
  REQUIRE(lz4.resident_bytes(Precision::Float64, false) > config.ram_limit);
  CHECK_THROWS_MATCHES(dtwc::run(config), dtwc::InvalidInput,
                       Message("Arrow IPC input needs approximately "
                               + std::to_string(lz4.resident_bytes(Precision::Float64, false))
                               + " bytes of resident series storage, exceeding --ram-limit=16000; method 'pam' cannot "
                                 "stream it. Use --method clara, which reads the file a record batch at a time, or "
                                 "raise --ram-limit."));
}

#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
TEST_CASE("Arrow IPC: a file on a GPU, filled or streamed, is a typed refusal", "[io][arrow][streaming][gpu]")
{
  // In RAM a view, which the GPU's fill cannot upload; streamed, views of its record batches.
  dtwc::Config config = clara_config(fixture("clara_256_lz4.arrow"));
  config.verbose = false;
  config.device = dtwc::Device::GPU;
  config.method = dtwc::Method::PAM;
  const std::string filled = [&] {
    try {
      (void)dtwc::run(config);
    } catch (const dtwc::DeviceError &refused) {
      return std::string(refused.what());
    }
    FAIL("a view on a GPU ran");
    return std::string();
  }();
  config.method = dtwc::Method::CLARA;
  config.ram_limit = 24000;
  const std::string stream = [&] {
    try {
      (void)dtwc::run(config);
    } catch (const dtwc::DeviceError &refused) {
      return std::string(refused.what());
    }
    FAIL("a stream on a GPU ran");
    return std::string();
  }();
  if (dtwc::gpu_available()) {
    CHECK_THAT(filled, ContainsSubstring("a non-owning view"));
    CHECK_THAT(stream, ContainsSubstring("fast_clara: a GPU assigns owned series in RAM, but a file streamed a record "
                                         "batch at a time (--ram-limit) is read as views of its batches"));
  }
}
#endif
