/**
 * @file test_ipc_stream.cpp
 * @brief FastCLARA's stream of an Arrow IPC file a record batch at a time (algorithms::detail::fast_clara_chunked over
 *        io::IpcFile, and run's planner, in every build): the in-RAM run's labels, medoids and cost, and the CSV
 *        twin's; the Parquet stream's verdicts, the cases' texts being those its serial check gives on their Parquet
 *        twins; a record batch larger than the limit, and a limit below the union beside its
 *        decode, refused before any batch is decoded; a streamed run's Result: its score and distance matrix refused,
 *        its save dtwc_cl's files. Fixtures: tests/data/reader/make_ipc_fixtures.py.
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::EndsWith;
using Catch::Matchers::Message;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::StartsWith;
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

/// One heap block of `bytes`, as io::IpcFile prices it (measured on Windows' heap, 2026-11-01): up to 16,384 bytes an
/// eighth and 18 bytes more, to 262,128 a 64th and 64, above them its page tables (1/512) and 4,700 bytes; none for
/// none.
std::size_t heap(std::size_t bytes)
{
  if (bytes == 0) return 0;
  return bytes + (bytes <= 16384 ? bytes / 8 + 18 : bytes <= 262128 ? bytes / 64 + 64 : bytes / 512 + 4700);
}

/// The reserve for its threads' working set that `config`'s run sets aside within --ram-limit `limit`, as -v's memory
/// plan names it, refused or run: a stream's data cap is the limit less it, so the reserve beside a stream's own limit
/// gives the stream that limit.
std::size_t reserve_of(dtwc::Config config, std::size_t limit)
{
  config.ram_limit = limit;
  config.verbose = true;
  std::ostringstream out;
  auto *const previous = std::cout.rdbuf(out.rdbuf());
  try {
    (void)dtwc::run(config);
  } catch (const dtwc::InvalidInput &) {
  } catch (...) {
    std::cout.rdbuf(previous);
    throw;
  }
  std::cout.rdbuf(previous);
  const std::string text = out.str();
  std::smatch reserve;
  INFO(text);
  REQUIRE(std::regex_search(text, reserve, std::regex("Memory plan: budget [0-9]+ bytes, reserve ([0-9]+) bytes")));
  return std::stoull(reserve[1].str());
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
  // 256 rows in eight record batches of 32 (uncompressed, LZ4, ZSTD, Float32 LZ4): within a limit that holds what
  // reading holds, three record batches each with its reader's scratch, and 16,000 bytes for the medoids and the
  // labels, a batch of the stream holds about three record batches, read by a team; each precision streamed, the
  // file's and the other.
  for (const char *file : { "clara_256.arrow", "clara_256_lz4.arrow", "clara_256_zstd.arrow", "clara_256_f32.arrow" })
    for (const Precision dtype : { Precision::Float64, Precision::Float32 }) {
      INFO(file << (dtype == Precision::Float32 ? " as float32" : " as float64"));
      const auto expected = in_ram(fixture(file), options, dtype);
      dtwc::Problem settings_only{ "streamed" };
      const dtwc::io::IpcFile walked(fixture(file));
      const std::size_t limit =
        walked.decode_bytes()
        + 3 * (walked.largest_row_group_bytes(dtype == Precision::Float32) + walked.reader_bytes()) + 16000;
      const auto result = streamed(settings_only, options, fixture(file), dtype, limit);
      CHECK(result.labels == expected.labels);
      CHECK(result.medoid_indices == expected.medoid_indices);
      CHECK(result.total_cost == expected.total_cost);
      CHECK(result.labels == csv.labels);
      CHECK(result.medoid_indices == csv.medoid_indices);
      CHECK(result.total_cost == csv.total_cost);
    }
  // The small fixtures: one record batch of three rows (uncompressed, LZ4, ZSTD, Float32) and three batches of
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
      // A data cap below what the series hold in RAM (the walk's price, CLARA's terms not counted), above what the
      // stream holds, beside the threads' reserve. A mapped file holds its values in the page cache, and is not
      // streamed: it is read into RAM within a MiB.
      const dtwc::io::IpcFile walked(fixture(file));
      const bool mapped = !walked.streams(dtype);
#if defined(DTWC_HAS_MMAP)
      CHECK(mapped == (std::string(file) == "clara_256.arrow" && dtype == Precision::Float64));
#else
      CHECK_FALSE(mapped); // the file read without llfio is anonymous
#endif
      const std::size_t cap = walked.resident_bytes(dtype, false);
      config.ram_limit = mapped ? std::size_t{ 1 } << 20 : reserve_of(config, cap) + cap;
      const std::string stream = printed([&] {
        const auto result = dtwc::run(config);
        CHECK(result.labels() == labels);
        CHECK(result.medoids() == medoids);
        CHECK(result.cost() == cost);
      });
      CHECK((stream.find("Arrow IPC walk selected streaming: 256 series") == std::string::npos) == mapped);
      CHECK(labels == csv.labels());
      CHECK(medoids == csv.medoids());
      CHECK(cost == csv.cost());
    }
  // The same values as time steps of two features (ndim 2), of two time steps or three: streamed, the distance takes
  // the file's features, as in RAM, the band of 1 counts time steps (in values the rows differ by 2), and the
  // clustering is not the univariate walks'.
  dtwc::Config config = clara_config(fixture("clara_256_ndim2.arrow"));
  config.verbose = false;
  config.band = 1;
  const auto resident = dtwc::run(config);
  const std::size_t cap =
    dtwc::io::IpcFile(fixture("clara_256_ndim2.arrow")).resident_bytes(Precision::Float64, false);
  config.ram_limit = reserve_of(config, cap) + cap;
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

TEST_CASE("Arrow IPC: a streamed run's Result refuses score and the distance matrix, naming the stream and "
          "--ram-limit, and saves dtwc_cl's labels and medoids, then the report",
          "[io][arrow][streaming]")
{
  // clara_256_lz4.arrow's record batches are decoded, never mapped: a limit below its series streams it in every
  // build. dtwc_cl runs dtwc::run(config) with its -o, so the files that run writes are dtwc_cl's (test_cli_ipc_stream
  // drives the binary on the same fixtures).
  const auto file = fixture("clara_256_lz4.arrow");
  dtwc::Config config = clara_config(file);
  const std::size_t cap = dtwc::io::IpcFile(file).resident_bytes(Precision::Float64, false);
  config.ram_limit = reserve_of(config, cap) + cap;
  config.name = "streamed";
  const auto cli = scratch() / "streamed_cli";
  config.output = dtwc::path_to_utf8(cli);
  std::optional<dtwc::Result> result;
  REQUIRE_THAT(printed([&] { result.emplace(dtwc::run(config)); }),
               ContainsSubstring("Arrow IPC walk selected streaming: 256 series"));
  // A composed matcher keeps its parts' addresses: the parts outlive each check.
  const auto stream = ContainsSubstring("the run streamed its 256 series");
  const auto limit = ContainsSubstring("--ram-limit");
  CHECK_THROWS_MATCHES(result->score("silhouette"), dtwc::InvalidInput, MessageMatches(stream && limit));
  CHECK_THROWS_MATCHES(result->distance_matrix(), dtwc::InvalidInput, MessageMatches(stream && limit));
  CHECK_THROWS_MATCHES(dtwc::detail::filled_matrix(*result), dtwc::InvalidInput, MessageMatches(stream && limit));

  const auto saved = scratch() / "streamed_saved";
  REQUIRE_NOTHROW(result->save(saved));
  for (const char *written : { "streamed_labels.csv", "streamed_medoids.csv", "streamed_report.txt" }) {
    INFO(written);
    CHECK(bytes_of(saved / written) == bytes_of(cli / written));
  }
  CHECK_FALSE(result->report().empty());
  CHECK_FALSE(std::filesystem::exists(saved / "streamed_distance_matrix.csv"));
  CHECK_FALSE(std::filesystem::exists(saved / "streamed_silhouettes.csv"));
}

// The cases of the Parquet stream's verdicts (test_io_readers.cpp's), in record batches read by a team: each batch's
// rows checked by the thread that read it, its RowCheck's extremes merged in batch order, the lowest failing batch's
// refusal rethrown. The texts are the Parquet stream's on each case's Parquet twin (a row group a record batch).
TEST_CASE("Arrow IPC: the stream's team gives the serial check's verdicts", "[io][arrow][streaming]")
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
  // refusal names the batch's bytes, before any batch is decoded (record batch 0 of this copy cannot be). That limit
  // is the stream's data cap, beside the threads' reserve.
  const auto one = undecodable("ipc_lz4.arrow");
  CHECK_THROWS_AS(dtwc::read_data(one), dtwc::IOError); // its decode fails
  const std::size_t batch = dtwc::io::IpcFile(one).largest_row_group_bytes(false);
  dtwc::Config config = clara_config(one);
  config.k = 2;
  config.sample_size = 2;
  config.verbose = false;
  config.ram_limit = reserve_of(config, batch - 1) + batch - 1;
  // run says what that --ram-limit is: the stream's data cap, the limit less the reserve.
  CHECK_THROWS_MATCHES(
    dtwc::run(config), dtwc::InvalidInput,
    MessageMatches(StartsWith("io::IpcFile::read_rows: record batch 0 of the Arrow IPC file holds "
                              + std::to_string(batch) + " bytes once read (its 3 series), more than --ram-limit="
                              + std::to_string(batch - 1)
                              + ", and FastCLARA's stream reads each record batch whole; raise the limit, or rewrite "
                                "the file in smaller record batches. The --ram-limit it names is the stream's data "
                                "cap: --ram-limit = "
                              + std::to_string(config.ram_limit) + " less the reserve of "
                              + std::to_string(config.ram_limit - (batch - 1)) + " bytes for ")));

  // The union's rows 0, 16, 32 and 48, one in each record batch of 16 rows of 8 values (no name column): the first
  // batch read whole beside what reading holds (decode_bytes), with the copies of its rows, at most its 128 values: a
  // byte short, refused before any batch is decoded.
  const auto copy = undecodable("clara_verdict_rows.arrow");
  const dtwc::io::IpcFile file(copy);
  const std::size_t objects = dtwc::io::IpcFile::series_bytes(0, 4, false), reading = file.decode_bytes();
  const std::size_t member = file.largest_row_group_bytes(false) + dtwc::io::IpcFile::series_bytes(128, 0, false);
  const std::size_t need = objects + reading + member;
  const std::vector<dtwc::index_t> rows{ 0, 16, 32, 48 };
  CHECK_THROWS_AS(file.read_rows<double>(rows, need), dtwc::IOError); // record batch 0 is read
  CHECK_THROWS_MATCHES(
    file.read_rows<double>(rows, need - 1), dtwc::InvalidInput,
    Message("io::IpcFile::read_rows: the 4 selected series hold " + std::to_string(objects)
            + " bytes so far and reading the file " + std::to_string(reading) + " (its walk and a reader's scratch); a "
            + "record batch of their rows read whole and its rows copied need " + std::to_string(member) + " more, "
            + std::to_string(need) + " in all, exceeding --ram-limit=" + std::to_string(need - 1)
            + "; raise the limit"));

  // A sample's PAM beside the sampled series and what the open file holds (its walk): one sample of 200 of
  // clara_256.arrow's 256 rows, read within the limit, a byte short of its copy and matrix beside them.
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 3;
  options.sample_size = 200;
  options.n_samples = 1;
  options.random_seed = 42;
  const dtwc::io::IpcFile plain(fixture("clara_256.arrow"));
  const auto plan = dtwc::algorithms::detail::resolve_clara_plan(plain.num_rows(), options, "fast_clara");
  const auto sampled = dtwc::algorithms::detail::sample_union(
    dtwc::algorithms::detail::draw_clara_samples(plan.n_points, plan.sample_size, options));
  REQUIRE(sampled.size() == 200);
  const dtwc::Data all = dtwc::read_data(fixture("clara_256.arrow"));
  std::size_t values = 0;
  for (const auto row : sampled) values += all.series_flat_size(static_cast<std::size_t>(row));
  const std::size_t union_bytes = dtwc::io::IpcFile::series_bytes(values, 200, false);
  const std::size_t pam = union_bytes + dtwc::core::packed_size(200) * sizeof(double);
  const std::size_t needs = plain.open_bytes() + union_bytes + pam;
  dtwc::Problem settings_only{ "pam" };
  CHECK_THROWS_MATCHES(streamed(settings_only, options, fixture("clara_256.arrow"), Precision::Float64, needs - 1),
                       dtwc::InvalidInput,
                       Message("fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(needs - 1)
                               + " cannot hold a sample's PAM beside the samples' series, which needs "
                               + std::to_string(needs) + " bytes: " + std::to_string(plain.open_bytes())
                               + " the open file holds, " + std::to_string(union_bytes) + " for the 200 sampled "
                               + "series, " + std::to_string(pam) + " for a sample's copy and its distance matrix "
                               + "(sample_size = 200); raise the limit, or lower sample_size."));
}

TEST_CASE("Arrow IPC: the planner prices a mapped view at its views, and the file read without llfio whole",
          "[io][arrow][streaming][ram]")
{
  using dtwc::io::IpcFile;
  // 256 rows in eight record batches of four fields (the root, 'data' and its values, 'name'): in RAM a view's 46 bytes
  // a row (its values the page cache's), each batch's objects, the walk's schema and one reader's decoders; the
  // compressed file's decoded batches beside them; without llfio the whole file, one heap block.
  const IpcFile plain(fixture("clara_256.arrow")), lz4(fixture("clara_256_lz4.arrow")),
    zstd(fixture("clara_256_zstd.arrow"));
  const auto size = static_cast<std::size_t>(std::filesystem::file_size(fixture("clara_256.arrow")));
  const std::size_t decoders = 4 * IpcFile::field_bytes;
  const std::size_t resident = 256 * dtwc::detail::view_series_bytes + 8 * IpcFile::resident_batch_bytes
                               + 4 * IpcFile::schema_field_bytes + decoders;
  CHECK(plain.reader_bytes() == decoders);
#if defined(DTWC_HAS_MMAP)
  CHECK(plain.resident_bytes(Precision::Float64, false) == resident);
  CHECK_FALSE(plain.streams(Precision::Float64));
  // A mapped record batch of 3 rows holds their spans and name views (16 bytes each), its made names (series_<i>, 8
  // bytes each) and the rows it marks unnamed, each a heap block; its share of the block's spans and views; its
  // objects; the two names its check copies, made, within a string's own buffer.
  CHECK(IpcFile(fixture("ipc_no_name.arrow")).largest_row_group_bytes(false)
        == 2 * heap(3 * 16) + heap(3 * 8) + heap(8) + 2 * 3 * 16 + IpcFile::batch_object_bytes);
  // A batch of 32 named rows (r<i>, every seventh null): its two copied names at most its longest, which the walk of
  // the mapping read from its offsets, or series_<i>, within a string's own buffer as well.
  CHECK(plain.largest_row_group_bytes(false)
        == 2 * heap(32 * 16) + heap(32 * 10) + heap(8) + 2 * (32 * 16 + 1) + IpcFile::batch_object_bytes);
  // A batch whose 'name' column holds no validity buffer (x, y, z): no name is null, so none is made, none priced.
  CHECK(IpcFile(fixture("ipc_float32.arrow")).largest_row_group_bytes(true)
        == 2 * heap(3 * 16) + heap(8) + 2 * 3 * 16 + IpcFile::batch_object_bytes);
#else
  CHECK(plain.resident_bytes(Precision::Float64, false) == resident + heap(size));
  CHECK(plain.streams(Precision::Float64));
#endif
  // A reading thread's codec scratch, from each frame's descriptor, each a heap block: pyarrow's LZ4 frames of one
  // block are independent blocks of 64 KB (LZ4F's context, 288 bytes; its 64 KB in, with its checksum, and out);
  // zstd's context.
  CHECK(lz4.reader_bytes() == decoders + heap(288) + heap(65536 + 4) + heap(65536));
  CHECK(zstd.reader_bytes() == decoders + heap(95968));
  // The same frames declaring linked blocks of 4 MB (FLG 0x40, BD 0x70; walked, never decoded): 128 KB more out.
  auto big = bytes_of(fixture("clara_256_lz4.arrow"));
  const std::byte magic[] = { std::byte{ 0x04 }, std::byte{ 0x22 }, std::byte{ 0x4D }, std::byte{ 0x18 } };
  std::size_t frames = 0;
  for (auto at = big.begin(); (at = std::search(at, big.end(), std::begin(magic), std::end(magic))) != big.end();
       at += 4, ++frames) {
    at[4] = std::byte{ 0x40 };
    at[5] = std::byte{ 0x70 };
  }
  REQUIRE(frames == 40); // eight batches of five buffers
  write(scratch() / "blocks_4mb.arrow", big);
  CHECK(IpcFile(scratch() / "blocks_4mb.arrow").reader_bytes()
        == decoders + heap(288) + heap(4194304 + 4) + heap(4194304 + 131072));
  // read_arrow_ipc decodes the eight LZ4 batches with a thread each, up to the team, each its reader's scratch, and
  // each batch's decoded columns' objects and its view of the schema's four fields last the read.
  const auto readers = static_cast<std::size_t>(std::min(8, dtwc::get_max_threads()));
  CHECK(lz4.resident_bytes(Precision::Float64, false)
        > resident + readers * lz4.reader_bytes() - decoders
            + 8 * (IpcFile::resident_data_bytes + IpcFile::resident_name_bytes + 4 * IpcFile::view_field_bytes));
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
  // Beyond an explicit limit the other methods are refused, naming the bytes, the limit and CLARA, which streams it.
  config.method = dtwc::Method::PAM;
  config.ram_limit = 16000;
  REQUIRE(lz4.resident_bytes(Precision::Float64, false) > config.ram_limit);
  CHECK_THROWS_MATCHES(dtwc::run(config), dtwc::InvalidInput,
                       Message("Arrow IPC input needs approximately "
                               + std::to_string(lz4.resident_bytes(Precision::Float64, false))
                               + " bytes of resident series storage, exceeding --ram-limit=16000; method 'pam' cannot "
                                 "stream it. Use --method clara, which reads the file a record batch at a time, or "
                                 "raise --ram-limit."));
#if defined(DTWC_HAS_MMAP)
  // A mapped file's values are not streamed: beyond an explicit limit no method names CLARA as the file is read; the
  // other precision, whose copy FastCLARA streams, is a way out (the run test streams clara_256.arrow as float32).
  const std::string mapped = "its values are mapped where they lie in the file, and a mapped file's values are not "
                             "streamed: the views of its 256 series are held at once. Raise --ram-limit, or read its "
                             "values as float32 (--dtype float32)";
  dtwc::Config view = clara_config(fixture("clara_256.arrow"));
  view.verbose = false;
  view.method = dtwc::Method::PAM;
  view.ram_limit = 16000;
  REQUIRE(plain.resident_bytes(Precision::Float64, false) > view.ram_limit);
  CHECK_THROWS_MATCHES(dtwc::run(view), dtwc::InvalidInput,
                       Message("Arrow IPC input needs approximately "
                               + std::to_string(plain.resident_bytes(Precision::Float64, false))
                               + " bytes of resident series storage, exceeding --ram-limit=16000; " + mapped
                               + " with --method clara, whose copy FastCLARA reads a record batch at a time."));
  view.method = dtwc::Method::CLARA;
  CHECK_THROWS_MATCHES(dtwc::run(view), dtwc::InvalidInput,
                       MessageMatches(StartsWith("Arrow IPC input needs approximately ")
                                      && EndsWith("; " + mapped + ", whose copy FastCLARA reads a record batch at a "
                                                  "time.")));
  // A sample of every series streams no copy either: the other precision is then no way out, by CLARA or for PAM.
  view.sample_size = 256;
  const std::string held = "its values are mapped where they lie in the file, and a mapped file's values are not "
                           "streamed: the views of its 256 series are held at once. Raise --ram-limit.";
  for (const auto method : { dtwc::Method::CLARA, dtwc::Method::PAM }) {
    view.method = method;
    CHECK_THROWS_MATCHES(dtwc::run(view), dtwc::InvalidInput,
                         MessageMatches(StartsWith("Arrow IPC input needs approximately ") && EndsWith("; " + held)));
  }
#endif
  // Nor is CLARA for a file that streams, where its sample would hold every series.
  config.sample_size = 256;
  CHECK_THROWS_MATCHES(dtwc::run(config), dtwc::InvalidInput,
                       Message("Arrow IPC input needs approximately "
                               + std::to_string(lz4.resident_bytes(Precision::Float64, false))
                               + " bytes of resident series storage, exceeding --ram-limit=16000; method 'pam' cannot "
                                 "stream it. Raise --ram-limit."));
  config.sample_size = 8;
  // CLARA with more clusters than series is refused as for every input, before its plan.
  config.method = dtwc::Method::CLARA;
  config.ram_limit = 0;
  config.k = 300;
  CHECK_THROWS_MATCHES(dtwc::run(config), dtwc::InvalidInput,
                       Message("cluster: k must not exceed the number of series."));
}

TEST_CASE("Arrow IPC: a file's names read alone are the in-RAM read's", "[io][arrow][streaming]")
{
  // Each record batch's 'name' column alone: in the mapping, or without llfio its header and the bytes it spans read;
  // viewed (uncompressed) or decoded (LZ4, ZSTD), a null named series_<row>; by one reader within the names' price,
  // and by a team; a byte below that price refused, naming the bytes.
  for (const char *file : { "clara_256.arrow", "clara_256_lz4.arrow", "ipc_batches.arrow", "ipc_lz4_batches.arrow",
                            "ipc_zstd_batches.arrow", "ipc_large_list.arrow" }) {
    INFO(file);
    const dtwc::Data data = dtwc::read_data(fixture(file));
    std::vector<std::string> names;
    for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i) names.emplace_back(data.name(i));
    const dtwc::io::IpcFile walked(fixture(file));
    const std::size_t one = walked.series_names_bytes() + walked.names_read_bytes();
    for (const std::size_t budget : { one, std::size_t{ 1 } << 30 }) CHECK(walked.series_names(budget) == names);
    CHECK_THROWS_MATCHES(walked.series_names(one - 1), dtwc::InvalidInput,
                         MessageMatches(StartsWith("io::IpcFile::series_names: the " + std::to_string(names.size())
                                                   + " names and the file's walk hold ")
                                        && EndsWith(" " + std::to_string(one) + " in all, exceeding the "
                                                    + std::to_string(one - 1) + " the limit leaves them; raise the "
                                                    "limit")));
  }
}

#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
// In RAM the series are views (of the mapping, of the file read without llfio, of decoded record batches), which
// a GPU uploads as an owned copy; streamed, views of its record batches, which CUDA's assignment uploads as an owned
// copy each (Metal assigns on the CPU). The oracle: the same run on the file's CSV twin, whose series are owned and hold
// the same values. Beyond an explicit limit the planner prices the copy, and names CLARA, which streams on a GPU too.
TEST_CASE("Arrow IPC: a file on a GPU, filled or streamed, gives its CSV twin's clustering",
          "[io][arrow][streaming][gpu]")
{
  const auto on_gpu = [](const char *file, dtwc::Method method) {
    dtwc::Config config = clara_config(fixture(file));
    config.verbose = false;
    config.method = method;
    config.device = dtwc::Device::GPU;
#  if defined(DTWC_HAS_CUDA)
    config.gpu_precision = dtwc::GpuPrecision::FP64; // CUDA's Auto computes in FP32 on a GPU slow at FP64
#  endif
    return config;
  };
  const dtwc::io::IpcFile lz4(fixture("clara_256_lz4.arrow"));
  // Refused at the plan, before any device is used.
  dtwc::Config pam = on_gpu("clara_256_lz4.arrow", dtwc::Method::PAM);
  pam.ram_limit = 16000;
  CHECK_THROWS_MATCHES(dtwc::run(pam), dtwc::InvalidInput,
                       Message("Arrow IPC input needs approximately "
                               + std::to_string(lz4.resident_bytes(Precision::Float64, true))
                               + " bytes of resident series storage, exceeding --ram-limit=16000; method 'pam' cannot "
                                 "stream it. Use --method clara, which reads the file a record batch at a time, or "
                                 "raise --ram-limit."));
  // Series passed in memory as a view (read_data's of the file): the plan holds their values, the owned copy the GPU
  // uploads of them, and the GPU's staging of them, the series padded to the longest (4 values each), beside the matrix.
  dtwc::Config memory = on_gpu("clara_256.arrow", dtwc::Method::PAM);
  memory.input.clear();
  memory.ram_limit = 16000;
  const dtwc::Data view = dtwc::read_data(fixture("clara_256.arrow"));
  REQUIRE(view.is_view());
  const std::size_t series = 1024 * sizeof(double) + dtwc::detail::owned_series_bytes(1024, 256, false);
#  if defined(DTWC_HAS_CUDA)
  const std::size_t staging = 256 * 4 * sizeof(double); // FP64
#  else
  const std::size_t staging = 256 * 4 * sizeof(float) + dtwc::core::packed_size(256) * sizeof(float); // and the matrix
#  endif
  CHECK_THROWS_MATCHES(dtwc::run(memory, view), dtwc::InvalidInput,
                       MessageMatches(ContainsSubstring(
                         std::to_string(series) + " for the series and the owned copy the GPU uploads of them, "
                         + std::to_string(staging) + " for the GPU's staging of them, "
                         + std::to_string(dtwc::core::packed_size(256) * sizeof(double)) + " for the distance matrix")));
#  if defined(DTWC_HAS_MMAP)
  // A mapped file's values are not streamed: below its views and their copy, the CPU, which holds no copy, is a way out.
  dtwc::Config mapped = on_gpu("clara_256.arrow", dtwc::Method::PAM);
  const std::size_t views = dtwc::io::IpcFile(fixture("clara_256.arrow")).resident_bytes(Precision::Float64, true);
  mapped.ram_limit = views - 1;
  CHECK_THROWS_MATCHES(dtwc::run(mapped), dtwc::InvalidInput,
                       Message("Arrow IPC input needs approximately " + std::to_string(views)
                               + " bytes of resident series storage, exceeding --ram-limit=" + std::to_string(views - 1)
                               + "; its values are mapped where they lie in the file, and a mapped file's values are "
                                 "not streamed: the views of its 256 series and the owned copy the GPU uploads of them "
                                 "are held at once. Raise --ram-limit, or run on the CPU (--device cpu), which holds no "
                                 "copy."));
#  endif
  if (!dtwc::gpu_available()) { // the backend's refusal, never a view's or the stream's
    CHECK_THROWS_MATCHES(dtwc::run(on_gpu("clara_256_lz4.arrow", dtwc::Method::PAM)), dtwc::DeviceError,
                         MessageMatches(!ContainsSubstring("view")));
    return;
  }
  for (const auto method : { dtwc::Method::PAM, dtwc::Method::CLARA }) {
    const auto twin = dtwc::run(on_gpu("clara_256.csv", method));
    for (const char *file : { "clara_256.arrow", "clara_256_lz4.arrow" }) {
      INFO(file << (method == dtwc::Method::PAM ? ", pam" : ", clara"));
      const auto result = dtwc::run(on_gpu(file, method));
      CHECK(result.labels() == twin.labels());
      CHECK(result.medoids() == twin.medoids());
      CHECK(result.cost() == twin.cost());
    }
  }
  // Streamed: a data cap below what the series hold in RAM for a GPU (their copy among it).
  const auto twin = dtwc::run(on_gpu("clara_256.csv", dtwc::Method::CLARA));
  dtwc::Config config = on_gpu("clara_256_lz4.arrow", dtwc::Method::CLARA);
  const std::size_t cap = lz4.resident_bytes(Precision::Float64, true);
  config.ram_limit = reserve_of(config, cap) + cap;
  config.verbose = true;
  const std::string stream = printed([&] {
    const auto result = dtwc::run(config);
    CHECK(result.labels() == twin.labels());
    CHECK(result.medoids() == twin.medoids());
    CHECK(result.cost() == twin.cost());
  });
  CHECK_THAT(stream, ContainsSubstring("Arrow IPC walk selected streaming: 256 series"));
#  if defined(DTWC_HAS_CUDA)
  // CUDA's pass holds the medoids beside its staging of them (its launch workspace), and each record batch beside its
  // owned copy and its staging: a byte below that, refused naming them; at it, the stream runs. clara_256's rows are 4
  // values each, so the two samples' 3 medoids hold 24.
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 3;
  options.sample_size = 8;
  options.n_samples = 2;
  options.random_seed = 42;
  const auto plan = dtwc::algorithms::detail::resolve_clara_plan(lz4.num_rows(), options, "fast_clara");
  const std::size_t medoids = dtwc::io::IpcFile::series_bytes(24, 6, false);
  const std::size_t fixed = 2 * medoids + dtwc::algorithms::detail::clara_pass_bytes(plan, 2) + lz4.decode_bytes();
  const std::size_t group = 3 * lz4.largest_row_group_bytes(false);
  const std::size_t finish = lz4.open_bytes() + medoids + 3 * 256 * sizeof(dtwc::index_t) + 256 / 8;
  const std::size_t needs = std::max(fixed + group, finish);
  dtwc::Problem gpu{ "streamed_gpu" };
  gpu.set_device(dtwc::Device::GPU);
  gpu.set_gpu_precision(dtwc::GpuPrecision::FP64);
  CHECK_THROWS_MATCHES(streamed(gpu, options, fixture("clara_256_lz4.arrow"), Precision::Float64, needs - 1),
                       dtwc::InvalidInput,
                       Message("fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(needs - 1)
                               + " cannot hold the pass over the file, which needs " + std::to_string(needs)
                               + " bytes: " + std::to_string(fixed) + " for the medoids' series and the GPU's staging "
                               + "of them, every sample's labels and nearest distances (n_samples = 2, k = 3, 256 "
                               + "rows) and decoding, " + std::to_string(group) + " for one record batch's series, the "
                               + "owned copy the GPU uploads of them and its staging of them, " + std::to_string(finish)
                               + " at its end with the result's labels and the open file; raise the limit, or rewrite "
                               + "the file with smaller record batches."));
  const auto at = streamed(gpu, options, fixture("clara_256_lz4.arrow"), Precision::Float64, needs);
  CHECK(at.labels == twin.labels());
  CHECK(at.medoid_indices == twin.medoids());
  CHECK(at.total_cost == twin.cost());
#  endif
}
#endif
