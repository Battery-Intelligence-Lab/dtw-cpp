/**
 * @file test_arrow_ipc.cpp
 * @brief The Arrow IPC reader of every build (dtwc::read_data, dtwc::io::read_arrow_ipc): each fixture pyarrow wrote
 *        (tests/data/reader/make_ipc_fixtures.py) to its series and names, or to its error; corrupt, truncated and
 *        unaligned copies made here; the series are views of the bytes, which the Data's owner keeps alive.
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

TEST_CASE("Arrow IPC: each bad fixture's error, as Python and MATLAB raise it", "[io][arrow]")
{
  const std::vector<std::pair<const char *, const char *>> io_errors{
    { "feather_v1.feather", "a Feather version 1 file (magic FEA1), which is not Arrow IPC" },
    { "ipc_scalar_data.arrow", "the 'data' column must be a List or LargeList of Float32/Float64, got double." },
    { "ipc_int64_names.arrow", "the 'name' column must be Utf8 or LargeUtf8, got int64." },
    { "ipc_ndim_2x.arrow", "schema metadata 'ndim' must be a positive integer, got '2x'." },
    { "ipc_dictionary.arrow", "Schema message field with DictionaryEncoding not supported" },
    { "ipc_lz4.arrow", "Arrow IPC record batch 0 is compressed (LZ4_FRAME)" },
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

TEST_CASE("Arrow IPC: corrupt, truncated and unaligned copies of a fixture are refused", "[io][arrow][security]")
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
  // Fed at a 4-byte offset: flatcc's verifier refuses each record batch's header, whose int64 fields lie unaligned.
  const auto shifted = [](const std::vector<std::byte> &bytes) {
    std::vector<std::byte> buffer(bytes.size() + 4);
    std::copy(bytes.begin(), bytes.end(), buffer.begin() + 4);
    return buffer;
  };
  for (const auto &[file, batch] :
       { std::pair{ "ipc_no_name.arrow", "184" }, std::pair{ "ipc_float32.arrow", "224" } }) {
    INFO(file);
    const auto buffer = shifted(bytes_of(file));
    CHECK_THROWS_MATCHES(dtwc::io::detail::read_ipc(std::span(buffer).subspan(4), nullptr, Precision::Float32),
                         dtwc::IOError,
                         MessageMatches(Equals("Arrow IPC message at byte " + std::string(batch)
                                               + ": Message flatbuffer verification failed (12) table field not "
                                                 "aligned")));
  }
  // A writer that put the values 4 bytes into their 8 (Float64: the buffer {offset 16, length 48} at offset 20), or
  // 2 into their 4 (Float32: {16, 24} at 18).
  const std::string unaligned = "Arrow IPC record batch 0: a buffer is not aligned to its elements, so it cannot be "
                                "viewed in place (Arrow IPC writers align every buffer to 8 bytes: rewrite the file "
                                "with one)";
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(bytes_of("ipc_list.arrow"), { 16, 48 }, { 20, 48 })),
                       dtwc::IOError, MessageMatches(Equals(unaligned)));
  CHECK_THROWS_MATCHES(read_bytes(patched<std::int64_t>(bytes_of("ipc_float32.arrow"), { 16, 24 }, { 18, 24 })),
                       dtwc::IOError, MessageMatches(Equals(unaligned)));
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
}

TEST_CASE("Arrow IPC: the series are views of the bytes, which the Data's owner keeps alive", "[io][arrow]")
{
  auto buffer = std::make_shared<std::vector<std::byte>>(bytes_of("ipc_list.arrow"));
  const std::weak_ptr<std::vector<std::byte>> alive = buffer;
  const std::span<const std::byte> bytes(*buffer);
  auto data = dtwc::io::detail::read_ipc(bytes, std::move(buffer), Precision::Float64);
  const auto inside = [&](const void *p) {
    return std::less_equal<>{}(static_cast<const void *>(bytes.data()), p)
           && std::less<>{}(p, static_cast<const void *>(bytes.data() + bytes.size()));
  };
  for (std::size_t i = 0; i < 3; ++i) CHECK(inside(data.series(i).data())); // zero-copy: every span in the bytes
  CHECK(inside(data.name(0).data()));
  CHECK_FALSE(inside(data.name(1).data())); // series_1, made: the name is null in the file
  CHECK(inside(data.name(2).data()));
  auto copy = data;
  data = dtwc::Data{};
  CHECK_FALSE(alive.expired());
  CHECK(series_of(copy) == three);
  CHECK(names_of(copy) == Names{ "a", "series_1", "c" });
  copy = dtwc::Data{};
  CHECK(alive.expired());
  CHECK(dtwc::io::read_arrow_ipc(fixture("ipc_list.arrow")).owner() != nullptr); // the mapping, or the read buffer
}
