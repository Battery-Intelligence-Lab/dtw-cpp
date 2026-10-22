/**
 * @file test_arrow_ipc.cpp
 * @brief The Arrow IPC reader of every build (dtwc::read_data, dtwc::io::read_arrow_ipc): each fixture pyarrow wrote
 *        (tests/data/reader/make_ipc_fixtures.py) to its series and names, or to its error, a compressed one to its
 *        uncompressed twin's; arrow-testing's big-endian file to its little-endian twin's; corrupt, truncated and
 *        unaligned copies made here; the series are views of the bytes, or of the decoded batches, which the Data's
 *        owner keeps alive.
 *
 * Python (tests/python/test_io.py) and MATLAB (tests/matlab/test_tier1_route_parity.m) read the same files to the same
 * series, names and errors.
 *
 * @author Volkan Kumtepeli
 */

#include <base/error.hpp>
#include <io/arrow_ipc.hpp>
#include <io/read_data.hpp>
#if defined(DTWC_HAS_ARROW)
#include <io/read_arrow.hpp>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::EndsWith;
using Catch::Matchers::Equals;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::StartsWith;
using dtwc::core::Precision;
using Series = std::vector<std::vector<double>>;
using Names = std::vector<std::string>;

namespace {

std::filesystem::path fixture(const char *name)
{
  return std::filesystem::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader" / name;
}

std::vector<std::byte> bytes_of(const char *name)
{
  std::ifstream in(fixture(name), std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> chars{ std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
  std::vector<std::byte> bytes(chars.size());
  std::memcpy(bytes.data(), chars.data(), chars.size());
  return bytes;
}

Series series_of(const dtwc::Data &data)
{
  Series out;
  for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i) {
    if (data.is_f32()) {
      const auto s = data.series_f32(i);
      out.emplace_back(s.begin(), s.end());
    } else {
      const auto s = data.series(i);
      out.emplace_back(s.begin(), s.end());
    }
  }
  return out;
}

Names names_of(const dtwc::Data &data)
{
  Names out;
  for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i) out.emplace_back(data.name(i));
  return out;
}

/// `bytes` with each of the `expect` occurrences of the little-endian values `from` replaced by `to`, as long.
template <typename Int>
std::vector<std::byte> patched(std::vector<std::byte> bytes, const std::vector<Int> &from, const std::vector<Int> &to,
                               std::size_t expect = 1)
{
  REQUIRE(from.size() == to.size());
  std::vector<std::byte> pattern(from.size() * sizeof(Int)), replacement(pattern.size());
  std::memcpy(pattern.data(), from.data(), pattern.size()); // the tests run on little-endian machines, as the files
  std::memcpy(replacement.data(), to.data(), replacement.size());
  std::size_t found = 0;
  for (auto at = bytes.begin(); (at = std::search(at, bytes.end(), pattern.begin(), pattern.end())) != bytes.end();
       ++found)
    at = std::copy(replacement.begin(), replacement.end(), at);
  REQUIRE(found == expect);
  return bytes;
}

dtwc::Data read_bytes(const std::vector<std::byte> &bytes, Precision dtype = Precision::Float64)
{
  return dtwc::io::detail::read_ipc(bytes, nullptr, dtype);
}

/// Whether `p` points into `bytes`.
bool inside(std::span<const std::byte> bytes, const void *p)
{
  return std::less_equal<>{}(static_cast<const void *>(bytes.data()), p)
         && std::less<>{}(p, static_cast<const void *>(bytes.data() + bytes.size()));
}

/// The rows [0, 3), [3, 5), [5, 6) of the six values of type `T` at byte `at` of `bytes`, as the series of the
/// fixtures' list offsets read them.
template <typename T>
Series three_at(const std::vector<std::byte> &bytes, std::size_t at)
{
  std::vector<double> values;
  for (std::size_t j = 0; j < 6; ++j) {
    T value;
    std::memcpy(&value, bytes.data() + at + j * sizeof value, sizeof value);
    values.push_back(value);
  }
  return { { values[0], values[1], values[2] }, { values[3], values[4] }, { values[5] } };
}

const Series three{ { 0, 0.5, 0.25 }, { 1.5, 2 }, { 3 } };

} // namespace

TEST_CASE("Arrow IPC: each fixture's series and names, as Python and MATLAB read them", "[io][arrow]")
{
  struct Case
  {
    const char *file;
    Series series;
    Names names;
    std::size_t ndim;
  };
  const std::vector<Case> cases{
    { "ipc_list.arrow", three, { "a", "series_1", "c" }, 1 },
    { "ipc_large_list.arrow", three, { "a", "series_1", "c" }, 1 },
    { "ipc_float32.arrow", three, { "x", "y", "z" }, 1 },
    { "ipc_no_name.arrow", three, { "series_0", "series_1", "series_2" }, 1 },
    { "ipc_ndim2.arrow", { { 0, 1, 2, 3 }, { 4, 5 } }, { "series_0", "series_1" }, 2 },
    { "ipc_batches.arrow", { { 0, 1 }, { 2 }, { 3 }, { 4, 5 }, { 6 }, { 7, 8, 9 } },
      { "a", "b", "series_2", "d", "series_4", "f" }, 1 },
    { "ipc_stream.arrow", three, { "a", "series_1", "c" }, 1 },
  };
  for (const auto &c : cases) {
    INFO(c.file);
    const auto data = dtwc::read_data(fixture(c.file));
    CHECK(data.is_view() == (std::string(c.file) != "ipc_float32.arrow")); // a Float32 file read as Float64 is copied
    CHECK_FALSE(data.is_f32());
    CHECK(data.ndim == c.ndim);
    CHECK(series_of(data) == c.series);
    CHECK(names_of(data) == c.names);
#if defined(DTWC_HAS_ARROW)
    const auto through_dtwc_io = dtwc::io::read_arrow(fixture(c.file));
    CHECK(series_of(through_dtwc_io) == c.series);
    CHECK(names_of(through_dtwc_io) == c.names);
#endif
  }
  CHECK(dtwc::read_data(fixture("ipc_ndim2.arrow")).series_length(0) == 2);
}

TEST_CASE("Arrow IPC: a file read in the other precision is an owned copy, each value cast once", "[io][arrow]")
{
  const auto narrowed = dtwc::read_data(fixture("ipc_list.arrow"), 0, 0, '\0', {}, Precision::Float32);
  CHECK(narrowed.is_f32());
  CHECK_FALSE(narrowed.is_view());
  CHECK(series_of(narrowed) == three);
  CHECK(names_of(narrowed) == Names{ "a", "series_1", "c" });
  const auto view = dtwc::read_data(fixture("ipc_float32.arrow"), 0, 0, '\0', {}, Precision::Float32);
  CHECK(view.is_f32());
  CHECK(view.is_view());
  CHECK(series_of(view) == three);
  CHECK(names_of(view) == Names{ "x", "y", "z" });
}

TEST_CASE("Arrow IPC: a compressed file is decoded to its uncompressed twin's series and names", "[io][arrow]")
{
  // LZ4_FRAME and ZSTD, one batch without names and three with nulls among them; decoded, so no series lies in the
  // file's bytes.
  for (const auto &[file, twin] :
       { std::pair{ "ipc_lz4.arrow", "ipc_no_name.arrow" }, std::pair{ "ipc_zstd.arrow", "ipc_no_name.arrow" },
         std::pair{ "ipc_lz4_batches.arrow", "ipc_batches.arrow" },
         std::pair{ "ipc_zstd_batches.arrow", "ipc_batches.arrow" } }) {
    INFO(file);
    const auto data = dtwc::read_data(fixture(file));
    const auto plain = dtwc::read_data(fixture(twin));
    CHECK(data.ndim == plain.ndim);
    CHECK(series_of(data) == series_of(plain));
    CHECK(names_of(data) == names_of(plain));
    const auto bytes = bytes_of(file);
    const auto decoded = read_bytes(bytes);
    for (std::size_t i = 0; i < static_cast<std::size_t>(decoded.size()); ++i)
      CHECK_FALSE(inside(bytes, decoded.series(i).data()));
  }
}

TEST_CASE("Arrow IPC: a big-endian file's batches decode to its little-endian twin's", "[io][arrow]")
{
  // arrow-testing's 1.0.0 generated_primitive files, two batches each, hold no 'data' column: read_ipc gets past the
  // big-endian schema to refuse that, and the batches are compared through its decoding, column by column, for every
  // element type a used column has (int32 and int64 offsets, float32 and float64 values, utf8 names).
  const auto big = bytes_of("arrow-testing/1.0.0-bigendian/generated_primitive.arrow_file");
  const auto little = bytes_of("arrow-testing/1.0.0-littleendian/generated_primitive.arrow_file");
  CHECK_THROWS_MATCHES(read_bytes(big), dtwc::IOError, MessageMatches(StartsWith("no 'data' column")));
  for (const char *type : { "int32", "int64", "float32", "float64", "utf8" })
    for (const char *nulls : { "_nullable", "_nonnullable" }) {
      const std::string column = std::string(type) + nulls;
      INFO(column);
      const auto decoded = dtwc::io::detail::decoded_buffers(big, column);
      REQUIRE(decoded.size() == 2);
      CHECK_FALSE(decoded[0][1].empty()); // the values or offsets
      CHECK(decoded == dtwc::io::detail::decoded_buffers(little, column));
    }
}

TEST_CASE("Arrow IPC: each bad fixture's error, as Python and MATLAB raise it", "[io][arrow]")
{
  const std::vector<std::pair<const char *, const char *>> io_errors{
    { "feather_v1.feather", "a Feather version 1 file (magic FEA1), which is not Arrow IPC" },
    { "ipc_scalar_data.arrow", "the 'data' column must be a List or LargeList of Float32/Float64, got double." },
    { "ipc_int64_names.arrow", "the 'name' column must be Utf8 or LargeUtf8, got int64." },
    { "ipc_ndim_2x.arrow", "schema metadata 'ndim' must be a positive integer, got '2x'." },
    { "ipc_dictionary.arrow", "Schema message field with DictionaryEncoding not supported" },
  };
  for (const auto &[file, says] : io_errors) {
    INFO(file);
    const std::string named = "load: failed to read '" + fixture(file).string() + "': ";
    CHECK_THROWS_MATCHES(dtwc::read_data(fixture(file)), dtwc::IOError,
                         MessageMatches(StartsWith(named) && ContainsSubstring(says)));
  }
  CHECK_THROWS_MATCHES(dtwc::read_data(fixture("ipc_null_series.arrow")), dtwc::InvalidInput,
                       MessageMatches(ContainsSubstring("data_from_arrow: null series at row 1 (drop or fill nulls")));
  CHECK_THROWS_MATCHES(dtwc::read_data(fixture("ipc_null_value.arrow")), dtwc::InvalidInput,
                       MessageMatches(ContainsSubstring("null value inside series at row 1 (drop or fill nulls")));
  // A reader option Arrow IPC cannot honour is refused before the file is opened.
  CHECK_THROWS_MATCHES(dtwc::read_data(fixture("ipc_list.arrow"), 0, 1), dtwc::InvalidInput,
                       MessageMatches(ContainsSubstring("cannot be honoured for this input")));
}

TEST_CASE("Arrow IPC: corrupt and truncated copies of a fixture are refused, unaligned ones read",
          "[io][arrow][security]")
{
  const auto no_name = bytes_of("ipc_no_name.arrow");
  REQUIRE(series_of(read_bytes(no_name)) == three);
  // The list offsets [0, 3, 5, 6]: the middle ones point anywhere, which validation does not read.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int32_t>(no_name, { 0, 3, 5, 6 }, { 0, 7, 5, 6 })), dtwc::IOError,
                       MessageMatches(ContainsSubstring(
                         "list offset [0, 7) of the series at row 0 is outside the values [0, 6).")));
  // The values' field node claims a value more than their buffer holds.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(no_name, { 3, 0, 6, 0 }, { 3, 0, 7, 0 })), dtwc::IOError,
                       MessageMatches(StartsWith("Arrow IPC record batch 0, column 'data': ")
                                      && ContainsSubstring("to have size >= 56 bytes but found buffer with 48 bytes")));
  // ndim neither whole nor positive, in the schema message and in the footer's copy of it.
  for (const auto *text : { "-1", "00" }) {
    INFO(text);
    const auto bytes = patched<char>(bytes_of("ipc_ndim_2x.arrow"), { '2', 'x' }, { text[0], text[1] }, 2);
    CHECK_THROWS_MATCHES(read_bytes(bytes), dtwc::IOError,
                         MessageMatches(ContainsSubstring("schema metadata 'ndim' must be a positive integer, got '"
                                                          + std::string(text) + "'.")));
  }
  // A name's offsets outside the names.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int32_t>(bytes_of("ipc_list.arrow"), { 0, 1, 1, 2 }, { 0, 5, 1, 2 })),
                       dtwc::IOError,
                       MessageMatches(ContainsSubstring(
                         "name offset [0, 5) of the series at row 0 is outside the names [0, 2).")));
  // Truncated: the file format's end, and the stream format's last record batch.
  const std::vector<std::byte> cut(no_name.begin(), no_name.end() - 20);
  CHECK_THROWS_MATCHES(read_bytes(cut), dtwc::IOError, MessageMatches(ContainsSubstring("it is truncated")));
  const auto stream = bytes_of("ipc_stream.arrow");
  CHECK_THROWS_MATCHES(read_bytes({ stream.begin(), stream.end() - 40 }), dtwc::IOError,
                       MessageMatches(Equals("Arrow IPC message at byte 544: its body of 72 bytes runs past the 40 "
                                             "left: the file is truncated or corrupt")));
  CHECK_THROWS_MATCHES(read_bytes(std::vector<std::byte>(8)), dtwc::IOError,
                       MessageMatches(StartsWith("not an Arrow IPC file")));
  // Fed at a 4-byte offset (flatcc verifies a header's tables at their alignment in memory), the bytes are read from
  // an 8-aligned copy the owner holds: the same series and names, none of them in the bytes fed.
  for (const char *file : { "ipc_list.arrow", "ipc_zstd_batches.arrow" }) {
    INFO(file);
    const auto bytes = bytes_of(file);
    std::vector<std::byte> buffer(bytes.size() + 4);
    std::copy(bytes.begin(), bytes.end(), buffer.begin() + 4);
    const auto fed = std::span<const std::byte>(buffer).subspan(4);
    const auto data = dtwc::io::detail::read_ipc(fed, nullptr, Precision::Float64);
    const auto aligned = read_bytes(bytes);
    CHECK(series_of(data) == series_of(aligned));
    CHECK(names_of(data) == names_of(aligned));
    for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i)
      CHECK_FALSE(inside(fed, data.series(i).data()));
  }
  // A writer that put the values 4 bytes into their 8 (Float64: the buffer {offset 16, length 48} at offset 20), or
  // 2 into their 4 (Float32: {16, 24} at 18): the batch is decoded into aligned buffers, the values those the file
  // puts there (each body at byte 480).
  const auto f64 = patched<std::int64_t>(bytes_of("ipc_list.arrow"), { 16, 48 }, { 20, 48 });
  CHECK(series_of(read_bytes(f64)) == three_at<double>(f64, 480 + 20));
  CHECK(names_of(read_bytes(f64)) == Names{ "a", "series_1", "c" });
  const auto f32 = patched<std::int64_t>(bytes_of("ipc_float32.arrow"), { 16, 24 }, { 18, 24 });
  CHECK(series_of(read_bytes(f32)) == three_at<float>(f32, 480 + 18));
}

TEST_CASE("Arrow IPC: the offsets and lengths a file gives nanoarrow are checked before a byte is read through them",
          "[io][arrow][security]")
{
  const auto no_name = bytes_of("ipc_no_name.arrow");
  const std::string batch = "Arrow IPC record batch 0, column 'data': ";
  // The values' buffer {16, 48} moved 2^63 - 8 bytes on, past the body of 64 bytes, or given a negative length (-1
  // is what nanoarrow takes for a size it is to compute).
  CHECK_THROWS_MATCHES(
    read_bytes(patched<std::int64_t>(no_name, { 16, 48 }, { std::numeric_limits<std::int64_t>::max() - 7, 48 })),
    dtwc::IOError,
    MessageMatches(Equals(batch + "a buffer of 48 bytes is not inside the batch's body of 64 bytes: the file is "
                                  "corrupt")));
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(no_name, { 16, 48 }, { 16, -1 })), dtwc::IOError,
                       MessageMatches(Equals(batch + "a buffer of -1 bytes is not inside the batch's body of 64 "
                                                     "bytes: the file is corrupt")));
  // The values' field node claims 2^58 values, which nanoarrow's minimal buffer size does not hold, and the last
  // list offset 40 of them: row 2 would be 35 values over a buffer of 6.
  CHECK_THROWS_MATCHES(
    read_bytes(patched<std::int64_t>(patched<std::int32_t>(no_name, { 0, 3, 5, 6 }, { 0, 3, 5, 40 }), { 3, 0, 6, 0 },
                                     { 3, 0, std::int64_t{ 1 } << 58, 0 })),
    dtwc::IOError,
    MessageMatches(Equals(batch + "a length of 288230376151711744 does not fit in the batch's body of 64 bytes: the "
                                  "file is corrupt")));
  // A stream whose schema's metadata length is 2^31 - 8 bytes, which with its 8-byte prefix is no header in the file.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int32_t>(bytes_of("ipc_stream.arrow"), { -1, 208 },
                                                        { -1, 0x7FFFFFF8 })),
                       dtwc::IOError,
                       MessageMatches(StartsWith("Arrow IPC message at byte 0: its header of ")
                                      && EndsWith(" bytes does not fit in the 880 left: the file is truncated or "
                                                  "corrupt")));
  // An end-of-stream marker before the file format's record batch, which would end the walk before it.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int32_t>(no_name, { -1, 184 }, { -1, 0 })), dtwc::IOError,
                       MessageMatches(Equals("Arrow IPC message at byte 184 is an end-of-stream marker, but the "
                                             "file's messages run on to its footer at byte 448: the file is "
                                             "corrupt")));
  // The data column's field node a row short of the batch's three.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(no_name, { 3, 0, 6, 0 }, { 2, 0, 6, 0 })), dtwc::IOError,
                       MessageMatches(Equals(batch + "its length of 2 is not the batch's 3: the file is corrupt")));
  // Offsets a writer put 2 bytes off their alignment, the list's ({0, 16} at 2) and the names' ({72, 16} at 74): never
  // validated where they lie, but decoded into aligned buffers and checked as any batch, where these offsets point
  // outside their values and their names.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(no_name, { 0, 0, 0, 16 }, { 0, 0, 2, 16 })), dtwc::IOError,
                       MessageMatches(Equals("data_from_arrow: list offset [196608, 327680) of the series at row 0 "
                                             "is outside the values [0, 6).")));
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(bytes_of("ipc_list.arrow"), { 72, 16 }, { 74, 16 })),
                       dtwc::IOError,
                       MessageMatches(Equals("Arrow IPC record batch 0, column 'name': Expected string array buffer "
                                             "2 to have size >= 1667301376 bytes but found buffer with 2 bytes")));
  // The footer's length enlarged by the record batch and the end-of-stream marker before it (208 + 256 + 8 bytes),
  // which ends the messages at the batch; and a footer whose second block names the third batch.
  auto long_footer = no_name;
  const std::int32_t footer = 208 + 256 + 8;
  std::memcpy(long_footer.data() + long_footer.size() - 10, &footer, sizeof footer);
  CHECK_THROWS_MATCHES(read_bytes(long_footer), dtwc::IOError,
                       MessageMatches(StartsWith("Arrow IPC footer: Footer flatbuffer verification failed")));
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(bytes_of("ipc_batches.arrow"), { 544, 256, 32 },
                                                        { 832, 256, 32 })),
                       dtwc::IOError,
                       MessageMatches(Equals("the Arrow IPC footer puts record batch 1 at byte 832, and the file's "
                                             "messages put it at byte 544: the file is corrupt")));
}

TEST_CASE("Arrow IPC: a compressed batch's offsets and lengths are checked before nanoarrow decodes through them",
          "[io][arrow][security]")
{
  const auto lz4 = bytes_of("ipc_lz4.arrow");
  const std::string batch = "Arrow IPC record batch 0, column 'data': ";
  // The values' buffer {40, 51} moved 2^63 - 8 bytes on, past the body of 96 bytes: nanoarrow would read its declared
  // length there and decompress from there.
  CHECK_THROWS_MATCHES(
    read_bytes(patched<std::int64_t>(lz4, { 40, 51 }, { std::numeric_limits<std::int64_t>::max() - 7, 51 })),
    dtwc::IOError,
    MessageMatches(Equals(batch + "a buffer of 51 bytes is not inside the batch's body of 96 bytes: the file is "
                                  "corrupt")));
  // The values' field node claims 2^58 values, which nanoarrow's validation would size their buffer for in a product
  // that wraps.
  CHECK_THROWS_MATCHES(
    read_bytes(patched<std::int64_t>(lz4, { 3, 0, 6, 0 }, { 3, 0, std::int64_t{ 1 } << 58, 0 })), dtwc::IOError,
    MessageMatches(Equals(batch + "a length of 288230376151711744 does not fit in the column's 64 decoded bytes: the "
                                  "file is corrupt")));
  // The values' declared length (their buffer's first 8 bytes) 512 GiB: refused whether it cannot be allocated or the
  // frame does not fill it.
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(lz4, { 48 }, { std::int64_t{ 1 } << 39 })), dtwc::IOError,
                       MessageMatches(StartsWith(batch)));
}

TEST_CASE("Arrow IPC: the series are views of the bytes, or of the decoded batches, which the Data's owner keeps "
          "alive",
          "[io][arrow]")
{
  auto buffer = std::make_shared<std::vector<std::byte>>(bytes_of("ipc_list.arrow"));
  const std::weak_ptr<std::vector<std::byte>> alive = buffer;
  const std::span<const std::byte> bytes(*buffer);
  auto data = dtwc::io::detail::read_ipc(bytes, std::move(buffer), Precision::Float64);
  for (std::size_t i = 0; i < 3; ++i) CHECK(inside(bytes, data.series(i).data())); // zero-copy: every span in the bytes
  CHECK(inside(bytes, data.name(0).data()));
  CHECK_FALSE(inside(bytes, data.name(1).data())); // series_1, made: the name is null in the file
  CHECK(inside(bytes, data.name(2).data()));
  auto copy = data;
  data = dtwc::Data{};
  CHECK_FALSE(alive.expired());
  CHECK(series_of(copy) == three);
  CHECK(names_of(copy) == Names{ "a", "series_1", "c" });
  copy = dtwc::Data{};
  CHECK(alive.expired());
  CHECK(dtwc::io::read_arrow_ipc(fixture("ipc_list.arrow")).owner() != nullptr); // the mapping, or the read buffer
  // Every batch decoded: the owner holds the decoded columns, and not the bytes.
  auto compressed = std::make_shared<std::vector<std::byte>>(bytes_of("ipc_zstd_batches.arrow"));
  const std::weak_ptr<std::vector<std::byte>> released = compressed;
  const std::span<const std::byte> zstd(*compressed);
  const auto decoded = dtwc::io::detail::read_ipc(zstd, std::move(compressed), Precision::Float64);
  CHECK(released.expired());
  CHECK(series_of(decoded) == series_of(dtwc::read_data(fixture("ipc_batches.arrow"))));
  CHECK(names_of(decoded) == Names{ "a", "b", "series_2", "d", "series_4", "f" });
}
