/**
 * @file unit_test_checkpoint.cpp
 * @brief save_checkpoint / load_checkpoint: one `.dtwm` file and its typed outcomes.
 *
 * @details The table is the contract: no file is `false`; a file for other
 * series or other distance settings is InvalidInput; a file that is not a whole
 * `.dtwm` file is IOError; a round trip is bit-exact. A checkpoint has the layout
 * of a mapped matrix, so with llfio every row also runs through
 * use_mmap_distance_matrix, which must answer the same.
 *
 * @author Volkan Kumtepeli
 * @date 29 Sep 2026
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

using namespace dtwc;
namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

namespace {

/// n short integer-valued series, all of one length.
Problem make_problem(const std::string &name, std::size_t n = 5)
{
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n; ++i) {
    series.push_back({ double(i), double(i * i % 7), double(3 * i % 5), 1.0, double(i % 2) });
    names.push_back("s" + std::to_string(i));
  }
  Problem prob(name);
  prob.set_data(Data(std::move(series), std::move(names)));
  return prob;
}

/// 300 series of lengths 3 to 11 (neighbours differ), so a fill of a few rows
/// per slab runs many slabs and blocks of mixed lengths.
Problem ragged_problem(const std::string &name, double shift = 0.0)
{
  constexpr std::size_t n = 300;
  std::vector<std::vector<double>> series(n);
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < 3 + i * 7 % 9; ++t) series[i].push_back(10 * std::sin(0.37 * double(i) + 0.61 * double(t)) + shift);
    names.push_back("s" + std::to_string(i));
  }
  Problem prob(name);
  prob.set_data(Data(std::move(series), std::move(names)));
  return prob;
}

/// The packed values' bits, widened to double.
std::vector<std::uint64_t> packed_bits(const Problem &prob)
{
  const auto &m = prob.distance_matrix();
  std::vector<std::uint64_t> bits;
  m.visit([&](const auto *values) {
    for (std::size_t k = 0; k < m.packed_count(); ++k)
      bits.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(values[k])));
  });
  return bits;
}

std::vector<char> read_bytes(const fs::path &path)
{
  std::ifstream in(path, std::ios::binary);
  return { std::istreambuf_iterator<char>{ in }, std::istreambuf_iterator<char>{} };
}

void write_bytes(const fs::path &path, const std::vector<char> &bytes)
{
  std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

enum class Outcome { Loaded, Absent, InvalidInput, IOError };

Outcome load_outcome(Problem &prob, const std::string &dir)
{
  try {
    return load_checkpoint(prob, dir) ? Outcome::Loaded : Outcome::Absent;
  } catch (const dtwc::InvalidInput &) {
    return Outcome::InvalidInput;
  } catch (const dtwc::IOError &) {
    return Outcome::IOError;
  }
}

#ifdef DTWC_HAS_MMAP
Outcome map_outcome(Problem &prob, const fs::path &file)
{
  try {
    prob.use_mmap_distance_matrix(file);
    return Outcome::Loaded;
  } catch (const dtwc::InvalidInput &) {
    return Outcome::InvalidInput;
  } catch (const dtwc::IOError &) {
    return Outcome::IOError;
  }
}
#endif

} // namespace

TEST_CASE("A checkpoint round trip is bit-exact, full and partial", "[checkpoint]")
{
  const ScratchDirectory scratch("round_trip");
  auto full = make_problem("full");
  full.fill_distance_matrix();
  save_checkpoint(full, scratch.path.string());
  REQUIRE(fs::file_size(scratch.path / "full.dtwm") == 64 + 15 * sizeof(double));

  auto partial = make_problem("partial");
  partial.writable_distance_matrix().resize(5);
  partial.writable_distance_matrix().set(3, 1, -0.0);
  partial.writable_distance_matrix().set(4, 4, 5e-324);
  save_checkpoint(partial, scratch.path.string());

  for (const Problem *source : { &full, &partial }) {
    auto target = make_problem(source->name());
    REQUIRE(load_checkpoint(target, scratch.path.string()));
    REQUIRE_FALSE(target.distance_matrix().is_mapped());
    REQUIRE(packed_bits(target) == packed_bits(*source)); // NaN = not computed, bit for bit
#ifdef DTWC_HAS_MMAP
    auto mapped = make_problem(source->name()); // the same file, mapped
    mapped.use_mmap_distance_matrix(checkpoint_path(mapped, scratch.path.string()));
    REQUIRE(packed_bits(mapped) == packed_bits(*source));
#endif
  }
  REQUIRE(partial.distance_matrix().count_computed() == 2);
}

TEST_CASE("Loading a checkpoint: the table of typed outcomes", "[checkpoint][error]")
{
  struct Row
  {
    const char *what;
    Outcome expected;
    std::function<void(std::vector<char> &)> damage; // applied to the saved file
    std::size_t reader_n = 5;
    int reader_band = -1;
  };
  const std::vector<Row> rows{
    { "other N", Outcome::InvalidInput, [](std::vector<char> &) {}, 4 },
    { "other identity (band)", Outcome::InvalidInput, [](std::vector<char> &) {}, 5, 2 },
    { "short", Outcome::IOError, [](std::vector<char> &b) { b.resize(20); } },
    { "empty", Outcome::IOError, [](std::vector<char> &b) { b.clear(); } },
    { "bad magic", Outcome::IOError, [](std::vector<char> &b) { b[0] = 'X'; } },
    { "bad version", Outcome::IOError, [](std::vector<char> &b) { b[4] = 3; } },
    { "wrong length (a double more)", Outcome::IOError, [](std::vector<char> &b) { b.resize(b.size() + 8); } },
    { "wrong length (a byte less)", Outcome::IOError, [](std::vector<char> &b) { b.pop_back(); } },
  };
  const ScratchDirectory scratch("table");
  auto writer = make_problem("table");
  writer.fill_distance_matrix();
  save_checkpoint(writer, scratch.path.string());
  const fs::path file = scratch.path / "table.dtwm";
  const auto saved = read_bytes(file);

  {
    auto reader = make_problem("absent");
    REQUIRE(load_outcome(reader, scratch.path.string()) == Outcome::Absent);
    REQUIRE(reader.distance_matrix().size() == 0);
  }
  for (const auto &row : rows) {
    INFO(row.what);
    auto bytes = saved;
    row.damage(bytes);
    write_bytes(file, bytes);
    auto reader = make_problem("table", row.reader_n);
    reader.set_band(row.reader_band);
    CHECK(load_outcome(reader, scratch.path.string()) == row.expected);
    CHECK(reader.distance_matrix().size() == 0); // nothing was installed
#ifdef DTWC_HAS_MMAP
    CHECK(map_outcome(reader, file) == row.expected);
    CHECK_FALSE(reader.distance_matrix().is_mapped());
#endif
    CHECK(read_bytes(file) == bytes); // a rejected file is left as it is
  }
}

TEST_CASE("save_checkpoint makes its directory, replaces its file and fails typed", "[checkpoint]")
{
  const ScratchDirectory scratch("save");
  const fs::path dir = scratch.path / "a" / "b";
  auto prob = make_problem("replace");
  save_checkpoint(prob, dir.string()); // nothing computed yet: all NaN
  prob.fill_distance_matrix();
  save_checkpoint(prob, dir.string());
  REQUIRE_FALSE(fs::exists(dir / "replace.dtwm.tmp"));
  auto reader = make_problem("replace");
  REQUIRE(load_checkpoint(reader, dir.string()));
  REQUIRE(reader.is_distance_matrix_filled());

  std::ofstream(scratch.path / "plain") << "a file where the directory belongs";
  REQUIRE_THROWS_AS(save_checkpoint(prob, (scratch.path / "plain").string()), dtwc::IOError);
  REQUIRE_THROWS_AS(save_checkpoint(prob, (scratch.path / "plain" / "sub").string()), dtwc::IOError);
  REQUIRE_THROWS_AS(save_checkpoint(Problem("empty"), scratch.path.string()), dtwc::InvalidInput);
}

TEST_CASE("Automatic checkpointing saves after every row block", "[checkpoint][fill]")
{
  const ScratchDirectory scratch("auto");
  auto reference = make_problem("auto", 6);
  reference.fill_distance_matrix();

  auto prob = make_problem("auto", 6);
  prob.checkpoint = { scratch.path.string(), 1, true };
  prob.fill_distance_matrix();
  auto restored = make_problem("auto", 6);
  REQUIRE(load_checkpoint(restored, scratch.path.string()));
  REQUIRE(packed_bits(restored) == packed_bits(reference));

  auto disabled = make_problem("off", 6);
  disabled.checkpoint.directory = (scratch.path / "never").string();
  disabled.fill_distance_matrix();
  REQUIRE_FALSE(fs::exists(scratch.path / "never"));

  for (const CheckpointOptions bad : { CheckpointOptions{ scratch.path.string(), 0, true }, CheckpointOptions{ "", 1, true } }) {
    auto rejected = make_problem("bad", 6);
    rejected.checkpoint = bad;
    REQUIRE_THROWS_AS(rejected.fill_distance_matrix(), dtwc::InvalidInput);
    REQUIRE(rejected.distance_matrix().count_computed() == 0);
  }
}

TEST_CASE("A resumed fill computes only the missing cells", "[checkpoint][fill]")
{
  constexpr std::size_t N = 6;
  const ScratchDirectory scratch("resume");
  auto reference = make_problem("resume", N);
  reference.fill_distance_matrix();

  // A crash after two rows: those cells hold values no DTW kernel produces.
  auto crashed = make_problem("resume", N);
  crashed.writable_distance_matrix().resize(N);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = i + 1; j < N; ++j) crashed.writable_distance_matrix().set(i, j, 900.0 + 10.0 * i + j);
  save_checkpoint(crashed, scratch.path.string());

  auto resumed = make_problem("resume", N);
  REQUIRE(load_checkpoint(resumed, scratch.path.string()));
  REQUIRE(resumed.distance_matrix().count_computed() == 9);
  resumed.checkpoint = { scratch.path.string(), 2, true };
  resumed.fill_distance_matrix();
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j)
      REQUIRE(resumed.dist_by_ind(int(i), int(j)) == (i < 2 ? 900.0 + 10.0 * i + j : reference.dist_by_ind(int(i), int(j))));

  auto reread = make_problem("resume", N); // the last block's save holds the whole matrix
  REQUIRE(load_checkpoint(reread, scratch.path.string()));
  REQUIRE(packed_bits(reread) == packed_bits(resumed));
}

TEST_CASE("A failing automatic save keeps the computed distances", "[checkpoint][fill]")
{
  constexpr std::size_t N = 40; // a save every lane width of rows: several slabs
  const ScratchDirectory scratch("save_fails");
  fs::create_directories(scratch.path);
  std::ofstream(scratch.path / "not_a_directory") << "occupied";
  auto reference = make_problem("fails", N);
  reference.fill_distance_matrix();

  auto prob = make_problem("fails", N);
  prob.checkpoint = { (scratch.path / "not_a_directory").string(), 2, true };
  REQUIRE_THROWS_AS(prob.fill_distance_matrix(), dtwc::IOError);
  // The first slab ran before its save failed: its distances stay, the reference's.
  const auto bits = packed_bits(prob), want = packed_bits(reference);
  std::size_t kept = 0;
  for (std::size_t k = 0; k < bits.size(); ++k)
    if (!std::isnan(std::bit_cast<double>(bits[k]))) {
      ++kept;
      REQUIRE(bits[k] == want[k]);
    }
  REQUIRE(kept > 0);
  REQUIRE(kept < bits.size());
  prob.checkpoint.enabled = false;
  prob.fill_distance_matrix();
  REQUIRE(packed_bits(prob) == want);
}

TEST_CASE("A checkpointed fill equals the fill in RAM, and its file holds it", "[checkpoint][fill]")
{
  const ScratchDirectory scratch("checkpointed");
  auto reference = ragged_problem("ckpt");
  reference.fill_distance_matrix();
  // Another matrix's file where the checkpoint goes, as long (same N, other
  // data): the first save replaces it whole, and only then do saves write rows in place.
  auto other = ragged_problem("ckpt", 1.0);
  other.fill_distance_matrix();
  save_checkpoint(other, scratch.path.string());

  // A save every 20 rows, rounded up to the lane width: many slabs, each saved.
  auto prob = ragged_problem("ckpt");
  prob.checkpoint = { scratch.path.string(), 20, true };
  prob.fill_distance_matrix();
  REQUIRE(packed_bits(prob) == packed_bits(reference));
  auto restored = ragged_problem("ckpt");
  REQUIRE(load_checkpoint(restored, scratch.path.string()));
  REQUIRE(packed_bits(restored) == packed_bits(reference));
}

TEST_CASE("A resumed checkpointed fill equals an uninterrupted one", "[checkpoint][fill]")
{
  const ScratchDirectory scratch("resumed");
  auto reference = ragged_problem("resumed");
  reference.fill_distance_matrix();
  // A fill stopped part way: a fifth of the entries, in no order, still NaN.
  auto stopped = ragged_problem("resumed");
  auto &partial = stopped.writable_distance_matrix();
  partial.resize(reference.size());
  for (std::size_t i = 0; i < reference.size(); ++i)
    for (std::size_t j = 0; j <= i; ++j)
      if ((i * 7 + j * 3) % 5 != 0) partial.set(i, j, reference.distance_matrix().get(i, j));
  save_checkpoint(stopped, scratch.path.string());

  auto resumed = ragged_problem("resumed");
  REQUIRE(load_checkpoint(resumed, scratch.path.string()));
  resumed.checkpoint = { scratch.path.string(), 20, true };
  resumed.fill_distance_matrix();
  REQUIRE(packed_bits(resumed) == packed_bits(reference));
  auto reread = ragged_problem("resumed");
  REQUIRE(load_checkpoint(reread, scratch.path.string()));
  REQUIRE(packed_bits(reread) == packed_bits(reference));
}

#ifdef DTWC_HAS_MMAP
TEST_CASE("A matrix mapped to its checkpoint file is saved in place", "[checkpoint][mmap]")
{
  const ScratchDirectory scratch("mapped");
  fs::create_directories(scratch.path);
  auto reference = make_problem("mapped", 6);
  reference.fill_distance_matrix();

  {
    auto prob = make_problem("mapped", 6);
    prob.use_mmap_distance_matrix(checkpoint_path(prob, scratch.path.string()));
    prob.checkpoint = { scratch.path.string(), 2, true };
    prob.fill_distance_matrix(); // each block's save flushes the mapping
    REQUIRE(prob.distance_matrix().is_mapped());
    REQUIRE_FALSE(fs::exists(scratch.path / "mapped.dtwm.tmp"));
    save_checkpoint(prob, scratch.path.string());
  }
  auto reader = make_problem("mapped", 6);
  REQUIRE(load_checkpoint(reader, scratch.path.string()));
  REQUIRE(packed_bits(reader) == packed_bits(reference));
}

TEST_CASE("A mapped fill of several slabs equals the fill in RAM, bit for bit", "[checkpoint][mmap][fill]")
{
  // A 361 KB matrix; a 1 MiB ram_limit leaves slabs of an eighth of what the
  // series leave of it, about 125 KB: several slabs, each prefetched and flushed.
  for (const int band : { -1, 8 }) { // ragged lanes; blocks of mixed lengths to the per-pair kernel
    CAPTURE(band);
    const ScratchDirectory scratch("mapped_slabs");
    fs::create_directories(scratch.path);
    auto reference = ragged_problem("slabs");
    reference.set_band(band);
    reference.fill_distance_matrix();
    {
      auto mapped = ragged_problem("slabs");
      mapped.set_band(band);
      mapped.set_ram_limit(std::size_t{ 1 } << 20);
      mapped.use_mmap_distance_matrix(scratch.path / "slabs.dtwm");
      mapped.fill_distance_matrix();
      REQUIRE(packed_bits(mapped) == packed_bits(reference));
    }
    auto reopened = ragged_problem("slabs");
    reopened.set_band(band);
    reopened.use_mmap_distance_matrix(scratch.path / "slabs.dtwm");
    REQUIRE(reopened.is_distance_matrix_filled()); // the fill recorded every row written
    REQUIRE(packed_bits(reopened) == packed_bits(reference));
  }
}
#endif

TEST_CASE("A float32 fill holds the float64 fill's distances rounded to floats, in RAM and mapped",
          "[checkpoint][fill][float32]")
{
  for (const int band : { -1, 8 }) {
    CAPTURE(band);
    auto reference = ragged_problem("narrow");
    reference.set_band(band);
    reference.fill_distance_matrix();
    std::vector<std::uint64_t> rounded;
    reference.distance_matrix().visit([&](const auto *values) {
      for (std::size_t k = 0; k < reference.distance_matrix().packed_count(); ++k)
        rounded.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(static_cast<float>(values[k]))));
    });
    auto in_ram = ragged_problem("narrow");
    in_ram.set_band(band);
    in_ram.set_matrix_dtype(core::Precision::Float32);
    in_ram.fill_distance_matrix();
    REQUIRE(in_ram.distance_matrix().width() == core::Precision::Float32);
    REQUIRE(packed_bits(in_ram) == rounded);
#ifdef DTWC_HAS_MMAP
    const ScratchDirectory scratch("narrow_mapped");
    fs::create_directories(scratch.path);
    auto mapped = ragged_problem("narrow");
    mapped.set_band(band);
    mapped.set_matrix_dtype(core::Precision::Float32);
    mapped.set_ram_limit(std::size_t{ 1 } << 20); // several slabs
    mapped.use_mmap_distance_matrix(scratch.path / "narrow.dtwm");
    mapped.fill_distance_matrix();
    REQUIRE(packed_bits(mapped) == rounded);
#endif
  }
}

TEST_CASE("A checkpoint of the other matrix_dtype is InvalidInput, loaded or mapped, naming both",
          "[checkpoint][error][float32]")
{
  const ScratchDirectory scratch("width");
  auto writer = make_problem("width");
  writer.fill_distance_matrix();
  save_checkpoint(writer, scratch.path.string());
  auto reader = make_problem("width");
  reader.set_matrix_dtype(core::Precision::Float32);
  try {
    (void)load_checkpoint(reader, scratch.path.string());
    FAIL("a float64 checkpoint loaded into a float32 Problem");
  } catch (const InvalidInput &e) {
    const std::string what = e.what();
    CHECK(what.find("holds float64 distances, but this Problem stores float32") != std::string::npos);
    CHECK(what.find("set matrix_dtype float64") != std::string::npos);
  }
  CHECK(reader.distance_matrix().size() == 0); // nothing was installed
#ifdef DTWC_HAS_MMAP
  CHECK_THROWS_AS(reader.use_mmap_distance_matrix(checkpoint_path(reader, scratch.path.string())), InvalidInput);
  CHECK_FALSE(reader.distance_matrix().is_mapped());
#endif
  // At its own width the round trip is bit-exact.
  reader.fill_distance_matrix();
  const fs::path narrow = scratch.path / "narrow";
  save_checkpoint(reader, narrow.string());
  REQUIRE(fs::file_size(checkpoint_path(reader, narrow.string())) == 64 + 15 * sizeof(float));
  auto restored = make_problem("width");
  restored.set_matrix_dtype(core::Precision::Float32);
  REQUIRE(load_checkpoint(restored, narrow.string()));
  REQUIRE(restored.is_distance_matrix_filled());
  REQUIRE(packed_bits(restored) == packed_bits(reader));
}
